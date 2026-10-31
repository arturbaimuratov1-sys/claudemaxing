"""Memory balance of one training step of the real Qwen2.5-1.5B on this GPU (investigation, not
part of the solution). Loads the model the way the base run does (4-bit NF4, fp16 compute,
prepare_model_for_kbit_training, LoRA r=16 on q/v), applies solution/, runs forward + fused head
+ backward on one batch of B x T tokens under fp16 autocast and prints what is alive when.
usage (from a folder holding solution/): python memreal.py [B] [T] [apply|base]"""

import collections
import sys
import traceback

import torch

sys.path.insert(0, ".")
B = int(sys.argv[1]) if len(sys.argv) > 1 else 4
T = int(sys.argv[2]) if len(sys.argv) > 2 else 340
mode = sys.argv[3] if len(sys.argv) > 3 else "apply"
MB = 2 ** 20
if mode == "apply":
    import solution

    solution.apply()

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import AutoModelForCausalLM, BitsAndBytesConfig  # noqa: E402


def mem(tag):
    torch.cuda.synchronize()
    print(f"{tag:28s} alloc {torch.cuda.memory_allocated() / MB:8.1f}  "
          f"peak {torch.cuda.max_memory_allocated() / MB:8.1f}  "
          f"reserved {torch.cuda.memory_reserved() / MB:8.1f}")


quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", quantization_config=quant,
                                             dtype=torch.float16, device_map={"": 0},
                                             attn_implementation="sdpa")
mem("loaded")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
mem("prepared")
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                         target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.train()
mem("peft")
lm = model.get_base_model()
emb = lm.get_input_embeddings().weight
print("embedding", emb.dtype, "max|emb|", emb.abs().max().item(),
      "exact in fp16:", torch.equal(emb.half().float(), emb))

saved = collections.Counter()
seen = set()


def pack(t):
    key = t.untyped_storage().data_ptr()
    if key not in seen:
        seen.add(key)
        where = "?"
        for fr in reversed(traceback.extract_stack()[:-1]):
            f = fr.filename.replace("\\", "/")
            if "torch/" in f and "autograd" in f:
                continue
            if any(s in f for s in ("transformers/", "peft/", "solution/", "bitsandbytes/",
                                    "torch/nn/")):
                where = f.split("/")[-1] + ":" + str(fr.lineno) + " " + fr.name
                break
        saved[(where, str(t.dtype))] += t.untyped_storage().nbytes()
    return t


torch.manual_seed(0)
ids = torch.randint(100, 150000, (B, T), device="cuda")
am = torch.ones(B, T, dtype=torch.long, device="cuda")
for i in range(1, B):  # right padding like the collator
    am[i, T - 37 * i:] = 0
labels = ids.masked_fill(am == 0, -100)
torch.cuda.reset_peak_memory_stats()
mem("before step")
m0 = torch.cuda.memory_allocated()
with torch.autocast("cuda", dtype=torch.float16), \
        torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
    if mode == "apply":
        from solution.embedding import shrink_tied_embedding
        from solution.lm_head_loss import fused_head_loss

        shrink_tied_embedding(lm)
        mem("after shrink")
        hidden = lm.model(input_ids=ids, attention_mask=am).last_hidden_state
        mem("after decoder fwd")
        loss, _ = fused_head_loss(hidden, lm.lm_head.weight, labels)
    else:
        loss = lm(input_ids=ids, attention_mask=am, labels=labels).loss
mem("after fwd+head")
tok = B * T
L = lm.config.num_hidden_layers
for (w, d), b in sorted(saved.items(), key=lambda kv: -kv[1])[:14]:
    print(f"  {b / MB:8.1f} MiB {b / tok / L:7.0f} B/tok/layer {d:14s} {w[:60]}")
print(f"saved total {sum(saved.values()) / MB:.1f} MiB for {tok} tokens")
torch.cuda.reset_peak_memory_stats()
(loss * 1024).backward()
mem("after backward (peak = bwd)")
