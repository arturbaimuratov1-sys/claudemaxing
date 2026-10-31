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
