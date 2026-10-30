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
