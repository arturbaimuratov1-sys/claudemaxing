# Sources and ideas

Ideas taken from other work. The code in `solution/` is written by us; nothing below was
copied.

## solution/rmsnorm.py

- **Reference behaviour**: `Qwen2RMSNorm.forward` in transformers 5.16.1,
  `transformers/models/qwen2/modeling_qwen2.py` (class at line 237, `forward` at line 247).
  Read to reproduce the output dtype and the order of roundings (normalise in fp32, cast to
  the input dtype, then multiply by the weight). Repository:
  https://github.com/huggingface/transformers
- **Pattern: per-program partial sums for the weight gradient, added afterwards**. A
  well-known way to reduce over rows without atomics. It is described in the Triton
  "Layer Normalization" tutorial: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
  (recalled from memory when designing the kernel; the page was not re-read and no code was
  taken from it. Our version differs: a persistent row loop, a final `torch` sum instead of a
  second kernel, no layer-norm bias or mean).
- **Math of the RMSNorm backward** (`dx = rstd * (g - x_hat * mean(g * x_hat))`): derived by
  hand from the formula of the reference.

## solution/swiglu_recompute.py

- **Reference behaviour**: `Qwen2MLP.forward` in transformers 5.16.1,
  `transformers/models/qwen2/modeling_qwen2.py`, lines 46-48
  (`down_proj(act_fn(gate_proj(x)) * up_proj(x))`). Read to see that silu is rounded to the
  dtype of `gate` before the product, and that the output dtype is the type promotion of the
  two inputs. Repository: https://github.com/huggingface/transformers
- **Idea: recompute a cheap activation in backward instead of storing it**. The general
  technique of trading recomputation for memory: Chen, Xu, Zhang, Guestrin, "Training Deep
  Nets with Sublinear Memory Cost", https://arxiv.org/abs/1604.06174 (known from the paper's
  abstract; no code involved). Applied here to one elementwise op only.
- **Idea: a fixed number of programs looping over tiles ("persistent" kernels)**. Used in the
  Triton persistent matmul tutorial: https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
  (recalled from memory, the page was not re-read, no code taken; ours is an elementwise loop).
- **Writing gradients over the input buffers** (`inplace`): our own design for this MLP; no
  external source. The tests pin that the buffers are reused and that nothing else is saved.
- `sigmoid(x) = 1 / (1 + 2^(-x * log2 e))`: textbook identity.

## solution/lm_head_loss.py

- **Soup's own chunked head** (`soup_cli/utils/stream_loss_chunk.py` in soup-cli 0.77.0, the
  `training.stream_loss_chunk` option, Apache-2.0): read to learn exactly what TRL's metrics
  and the rewind recorder need from the logits, and the trainer attribute
  (`_soup_chunked_step` with `token_loss`) through which Soup's rewind recorder accepts
  per-token losses for a step without logits. We do not switch that option on. What we took
  from it, stated plainly: the field names of the per-step record (`token_loss`,
  `token_entropy`, `correct`, `labelled`), the label-shift helper and its docstring wording,
  and the name of the metric logger. Its chunk loop is torch code; ours is a Triton kernel and
  our own autograd function and patches.
- **TRL 0.29.1 `SFTTrainer.compute_loss`** (`trl/trainer/sft_trainer.py` lines 1238-1320,
  Apache-2.0): `_log_trl_metrics` repeats its metric formulas (entropy weighted by the
  attention mask, num_tokens, token accuracy, gathers) line by line on purpose, because the
  logged values must be the same as the base run's. This is glue code, not a kernel.
- **Not materialising the logits of a large vocabulary**: Wijmans et al., "Cut Your Losses in
  Large-Vocabulary Language Models", https://arxiv.org/abs/2411.09009 (idea only, recalled
  from memory; that work has its own library, which we neither use nor copy).
- **Gradient of a scalar loss computed in forward and written over the logits buffer, then one
  matmul with the frozen head**: the approach of fused linear + cross-entropy kernels, e.g.
  the one in https://github.com/linkedin/Liger-Kernel (a banned library: idea only, recalled
  from memory, nothing installed, imported or copied).
- **Online log-sum-exp in one pass over tiles** (running max with rescaling): Milakov and
  Gimelshein, "Online normalizer calculation for softmax", https://arxiv.org/abs/1805.02867
  (recalled from memory). The running `sum(exp(z) * z)` for the entropy is our addition.
- **Base-path numerics**: read in the installed libraries — accelerate 1.14.0
  `accelerator.py` lines 1818-1829 (forward wrapped in fp16 autocast, outputs converted to
  fp32), transformers 5.16.1 `loss/loss_utils.py` (`ForCausalLMLoss`, `fixed_cross_entropy`),
  TRL 0.29.1 `trainer/sft_trainer.py` lines 1200-1328 (entropy, num_tokens, token accuracy),
  soup-cli 0.77.0 `trainer/rewind_hf.py` (per-row losses).

## solution/rope.py

- **Reference behaviour**: `apply_rotary_pos_emb` and `rotate_half` in transformers 5.16.1,
  `transformers/models/qwen2/modeling_qwen2.py` lines 105-135, called by name from
  `Qwen2Attention.forward`. Read to reproduce the formula, the layout (q/k transposed views,
  cos/sin of batch 1) and the dtype.
- **Rotary position embedding** itself: Su et al., "RoFormer: Enhanced Transformer with Rotary
  Position Embedding", https://arxiv.org/abs/2104.09864 (formula only). The backward as the
  transposed rotation is derived by hand. Fusing RoPE into one elementwise kernel is a common
  idea; no implementation was looked at or copied.

## solution/attention.py

- **Reference behaviour**: `sdpa_attention_forward` and `repeat_kv` in transformers 5.16.1,
  `transformers/integrations/sdpa_attention.py`, and `Qwen2Attention.forward` in
  `transformers/models/qwen2/modeling_qwen2.py` (lines 195-235). Read to learn the call
  signature, the layouts (q/k/v as transposed views, output [B, T, H, D]), the mask
  transformers passes for padded batches and that the base expands k/v when a mask is given.
- **Tiled attention with an online softmax and recomputation in backward** (FlashAttention):
  Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness",
  https://arxiv.org/abs/2205.14135, and Dao, "FlashAttention-2",
  https://arxiv.org/abs/2307.08691 (the algorithm from the papers, recalled from memory; the
  flash-attn library is banned and was neither installed nor read). The Triton "Fused
  Attention" tutorial (https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
  is a known implementation of the same idea; it was not opened in this work and no code was
  taken from it. Our version: grouped-query heads read in place (no k/v expansion), a key
  padding vector instead of a full mask, backward split into a dk/dv kernel that loops over the
  query heads of a group and a separate dq kernel (no atomics), fixed block sizes for the T4.
- **Online softmax**: Milakov and Gimelshein, https://arxiv.org/abs/1805.02867 (as above).

## solution/embedding.py

- Our own observation from reading `peft.prepare_model_for_kbit_training` (peft 0.20.0,
  `peft/utils/other.py`), which upcasts the non-quantized parameters to fp32: the embedding is
  loaded in fp16, so its fp32 copy holds fp16 numbers only. No external code involved.

## solution/nf4.py

- **Reference behaviour**: bitsandbytes 0.50.2 `functional.dequantize_4bit` and
  `backends/cuda/ops.py` (`_dequant_linear_fallback`, `_dequantize_4bit_impl`,
  `_dequantize_blockwise_impl`): read for the packed layout (high nibble first), the NF4 table
  (`get_4bit_type("nf4")`), the double-quantized absmax (`code2[q] * absmax2 + offset`, fp32) and
  the order of roundings, so that our output is bit-identical. bitsandbytes also has a Triton
  backend (`backends/triton/kernels_4bit.py`) with a 4-bit dequant kernel; it was not used as a
  template and no code was taken from it. Our kernel fuses the nested absmax into the same pass.

## solution/lora.py

- **Reference behaviour**: peft 0.20.0 `tuners/lora/bnb.py` (`Linear4bit.forward`) and
  `tuners/lora/layer.py` (`Linear.forward`): `base(x) + lora_B(lora_A(dropout(x))) * scaling`.
  Packing a dropout mask into bits and rebuilding the dropped input in backward is our own
  design; `torch.native_dropout` gives the same mask as `nn.Dropout` (tested).
