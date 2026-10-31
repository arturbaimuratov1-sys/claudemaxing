"""End to end: a few training steps of a tiny Qwen2 through the same stack as the base run
(4-bit NF4 weights, prepare_model_for_kbit_training, LoRA r=16 on q_proj/v_proj with dropout,
fp16 mixed precision, TRL's SFTTrainer wrapped in Soup's rewind recorder), once as is and once
with the fused head patches, compared by the rules of harness/check_rules.py:

- num_tokens per step, rewind rows and token counts, logged metric names: identical;
- entropy, token accuracy, rewind row losses, loss curve: mean relative gap <= 3%;
- first-step loss within 1%.

Slow (two short trainings). Needs peft, trl, bitsandbytes and soup-cli; skipped without them.
Run with -s to see the numbers.
"""

import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

VOCAB, WORDS, STEPS, SEED = 151936, 3000, 8, 1


def mean_relative(a, b):
    return sum(abs(x - y) / max(abs(y), 1e-12) for x, y in zip(a, b)) / len(a)


def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{f"w{i}": i + 3 for i in range(WORDS)}}
    raw = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>",
                                   unk_token="<unk>")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A tiny random Qwen2 with a sharp output distribution, and training rows whose answers
    the model itself mostly predicts (greedy continuations with 30% of the tokens replaced),
    so that token accuracy, entropy and the loss are not trivial numbers."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    path = tmp_path_factory.mktemp("tiny_qwen2")
    config = Qwen2Config(vocab_size=VOCAB, hidden_size=128, intermediate_size=352,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         tie_word_embeddings=True, hidden_act="silu", initializer_range=0.5)
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config).to(torch.float16)
    model.save_pretrained(path)
    model.cuda().eval()
    rng = random.Random(0)
    rows = []
    with torch.no_grad():
        for _ in range(STEPS * 4):
            prompt = [rng.randrange(3, WORDS + 3) for _ in range(rng.randrange(8, 40))]
            length = rng.randrange(8, 80)
            out = model.generate(torch.tensor([prompt], device="cuda"), max_new_tokens=length,
                                 min_new_tokens=length, do_sample=False, pad_token_id=0)
            answer = [t if rng.random() > 0.3 else rng.randrange(3, WORDS + 3)
                      for t in out[0, len(prompt):].tolist()] + [1]
            rows.append({"input_ids": prompt + answer,
                         "completion_mask": [0] * len(prompt) + [1] * len(answer)})
    del model
    torch.cuda.empty_cache()
    return path, rows


def dataset(rows):
    from datasets import Dataset

    return Dataset.from_list(rows)


def train(fixture, tmp, fused):
    checkpoint, rows = fixture
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from soup_cli.trainer.rewind_hf import attach_rewind_state, make_rewind_trainer_class
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(SEED)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=quant,
                                                 dtype=torch.float16, device_map={"": 0},
                                                 attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    set_seed(SEED)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    probe = {"head": model.get_output_embeddings().weight.dtype,
             "norm": model.get_base_model().model.norm.weight.dtype}
    args = SFTConfig(output_dir=str(tmp), per_device_train_batch_size=4, max_steps=STEPS,
                     learning_rate=2e-4, fp16=True, bf16=False, logging_steps=1, seed=SEED,
                     report_to=[], save_strategy="no", max_length=512, optim="adamw_torch",
                     lr_scheduler_type="cosine", dataloader_num_workers=0)
    trainer = make_rewind_trainer_class(SFTTrainer)(model=model, args=args, train_dataset=dataset(rows),
                                                    processing_class=tokenizer())
    # TRL casts QLoRA adapters to bf16; Soup casts them back to fp32 for fp16 training, as here.
    from soup_cli.utils.mixed_precision import align_trainable_dtype_for_fp16

    align_trainable_dtype_for_fp16(trainer.model, fp16=True, bf16=False)
    records = []

    class Sink:
        def record_batch(self, *, step, micro, rows, row_loss, row_tokens):
            records.append((list(rows), list(row_tokens), list(row_loss)))

    state = attach_rewind_state(trainer, Sink())
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    assert not state.failed, "the rewind recorder was disabled"
    logs = [e for e in trainer.state.log_history if "loss" in e]
    lora = {n: p.detach().float().cpu() for n, p in model.named_parameters() if "lora_" in n}
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    del trainer, model
    torch.cuda.empty_cache()
    return logs, records, lora, probe, peak


@pytest.mark.parametrize("variant", ["head", "apply"])
def test_training_matches_the_base_run(checkpoint, tmp_path, variant):
    """`head`: only the fused head patches; `apply`: the whole solution.apply()."""
    for name in ("peft", "trl", "bitsandbytes", "soup_cli", "datasets"):
        pytest.importorskip(name)
    if torch.cuda.device_count() > 1:
        pytest.skip("run with CUDA_VISIBLE_DEVICES=0, as the harness does: with several GPUs the "
                    "Trainer uses DataParallel and the fused head falls back to TRL's path")
    from transformers.models.qwen2 import modeling_qwen2
    from trl.trainer import sft_trainer as trl_sft

    import solution
    from solution import lm_head_loss as mod

    base = train(checkpoint, tmp_path / "base", fused=False)

    saved = [(modeling_qwen2.Qwen2ForCausalLM, "forward"), (trl_sft.SFTTrainer, "compute_loss"),
             (modeling_qwen2.Qwen2MLP, "forward"), (modeling_qwen2.Qwen2RMSNorm, "forward")]
    saved = [(cls, name, getattr(cls, name)) for cls, name in saved]
    calls = []
    real = mod.fused_head_loss
    mod.fused_head_loss = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        solution.apply() if variant == "apply" else mod.patch_fused_head_loss()
        sol = train(checkpoint, tmp_path / "sol", fused=True)
    finally:
        mod.fused_head_loss = real
        for cls, name, value in saved:
            setattr(cls, name, value)

    (b_logs, b_rw, b_lora, probe, b_peak), (s_logs, s_rw, s_lora, _, s_peak) = base, sol
    print(f"\nprobe after prepare_model_for_kbit_training: {probe}")
    print(f"peak allocated: base {b_peak:.0f} MiB, fused {s_peak:.0f} MiB")
    assert len(calls) == STEPS, "the fused head did not run on every step"

    assert len(b_logs) == len(s_logs) == STEPS
    assert sorted(b_logs[0]) == sorted(s_logs[0]), "logged metric names differ"
    assert [e["num_tokens"] for e in b_logs] == [e["num_tokens"] for e in s_logs]
    gaps = {}
    for key in ("loss", "entropy", "mean_token_accuracy"):
        gaps[key] = mean_relative([e[key] for e in s_logs], [e[key] for e in b_logs])
    first = abs(s_logs[0]["loss"] - b_logs[0]["loss"]) / abs(b_logs[0]["loss"])

    assert len(b_rw) == len(s_rw) == STEPS
    assert [(r, t) for r, t, _ in b_rw] == [(r, t) for r, t, _ in s_rw]
    gaps["rewind row loss"] = mean_relative([x for *_, l in s_rw for x in l],
                                            [x for *_, l in b_rw for x in l])
    lora_gap = max((s_lora[n] - b_lora[n]).abs().max().item() for n in b_lora)
    print("first-step loss gap {:.4%}; mean gaps: {}; max |dLoRA| {:.2e}".format(
        first, {k: f"{v:.4%}" for k, v in gaps.items()}, lora_gap))
    for e_b, e_s in zip(b_logs, s_logs):
        print(f"  loss {e_b['loss']:.5f} / {e_s['loss']:.5f}   entropy {e_b['entropy']:.4f} / "
              f"{e_s['entropy']:.4f}   acc {e_b['mean_token_accuracy']:.4f} / "
              f"{e_s['mean_token_accuracy']:.4f}   grad_norm {e_b.get('grad_norm')} / "
              f"{e_s.get('grad_norm')}")
    assert first <= 0.01
    for key, gap in gaps.items():
        assert gap <= 0.03, (key, gap)
