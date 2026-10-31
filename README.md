# Soup Kernel Challenge: our solution

Own Triton kernels plugged into `soup train` (Soup 0.77.0, Qwen2.5-1.5B-Instruct, NF4, LoRA r=16
on q/v, batch 4, up to 512 tokens, 64 steps) on a free Kaggle T4 through one function,
`solution.apply()`. The training task, Soup's settings and Soup's outputs stay the same; the run
needs **5x less GPU memory** than the base and is not slower.

## Results (Kaggle T4, the submission notebook below, commit 873043d)

| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 85.6 | 69.4 | speedup 1.233x |
| Peak GPU memory (nvidia-smi), MiB | 14345 | 2755 | base/ours 5.207x |
| Eval loss (public rows) | | | -0.008% |
| Own kernels, share of training time | — | 15.3% | rule: >= 5% |

An earlier measurement of the same code (3e1da26 in [EXPERIMENTS.md](EXPERIMENTS.md)): 77.1 s
against 78.3 s, 2839 MiB, eval loss +0.000%, own kernels 15.0% (11.5% without the SwiGLU pair).

All 10 checks of `check_rules.py` passed (first-step loss gap 0.02%, mean 0.01%; 79/79 Soup
tests with our patches). Steady-state step 0.721 s against 0.857 s for the base (-16%); the wall
time gains less because of ~24 s fixed start-up and ~6 s of Triton JIT in the first two steps
(the harness gives every run a fresh Triton cache, and compile time is part of the measurement).
Full history of every measured version: [EXPERIMENTS.md](EXPERIMENTS.md).

## What we did and why

The base step on the T4 is dominated by memory (14 GB) and by many small elementwise ops and fp32
copies. Where the memory went and what replaced it:

| Item in the base | What we do | Kernel(s), file |
|---|---|---|
| Logits [B, T, 152K] in fp16 + fp32 copies, log-softmax for backward, separate entropy / accuracy / rewind passes (GBs) | Fused head + cross-entropy + entropy + accuracy + per-row loss, walking the vocabulary in chunks with an online log-sum-exp; the hidden-state gradient is built in the same pass; full logits never exist | `lm_head_vocab_chunk_kernel`, `lm_head_xent_rows_kernel` — `solution/lm_head_loss.py` |
| SwiGLU keeps gate, up and their product for backward | silu(gate)*up fused; backward recomputes silu from gate/up and writes the gradients over the input buffers | `silu_mul_fwd_persistent`, `silu_mul_bwd_recompute` — `solution/swiglu_recompute.py` |
| RMSNorm in fp32 keeps its fp32 input | Fused RMSNorm forward/backward; keeps one fp16 copy of its output (shared with the LoRA and MLP inputs) and fp32 rstd; tiny-weight channels keep their normalised value in a side buffer | `qwen_rmsnorm_fwd_kernel`, `qwen_rmsnorm_bwd_kernel` — `solution/rmsnorm.py` |
| RoPE as several torch ops | One elementwise kernel, forward and backward | `qwen_rope_kernel` — `solution/rope.py` |
| bitsandbytes NF4 dequantization | Own dequant kernel, nested absmax fused in the same pass, bit-identical to bitsandbytes | `nf4_dequant_kernel` — `solution/nf4.py` |
| LoRA dropout keeps the dropped fp32 input | The dropout mask from `torch.native_dropout` (same RNG, same distribution) is packed into bits; backward rebuilds the dropped input | `lora_dropout_pack_kernel`, `lora_dropout_unpack_kernel` — `solution/lora.py` |
| Tied embedding upcast to fp32 (890 MiB) | Stored in fp16 (every value is an exact fp16 number, checked), embedding output cast to fp32 — same values | `solution/embedding.py` |
| Pad tokens: every token-wise op (projections, MLP, norms, LoRA, head) runs on [B, T] including padding | The real tokens of the batch run as one row [1, N] with their own position ids and a mask that keeps attention inside each row exactly as the base mask; per-token outputs go back to [B, T]. Same rows, same batches (removing pad tokens inside a batch is allowed) | `solution/lm_head_loss.py` (`_pack_batch`) |
| SDPA with k/v expanded by `repeat_kv` | torch SDPA per kv head with stride-0 k/v views (same products, no copies) | `solution/attention.py` |

Every kernel was tested against the torch reference in fp16 (values and gradients) in
`tests_local/`, and reviewed for dtype, fp32 accumulation and gradients. Sources of the ideas
(papers, library code read for reference behaviour; no code copied) are in
[SOURCES.md](SOURCES.md).

## How to run

Kaggle notebook (template `templates/kaggle_notebook.ipynb`, `COMMIT` =
`873043d747dbcdadb0b7b0d933c44456d57176c5`):
https://www.kaggle.com/code/togamyssovabilkaiyr/notebook9b2547a27b

1. Open the public Kaggle notebook above, settings
   **GPU T4 x2**, **Internet on**, and **Run All**. It clones this repository at `COMMIT`,
   installs the pinned packages, trains base Soup and our solution, scores both adapters and runs
   the rule checks.
2. Locally: `pip install "soup-cli[train]==0.77.0"` with the versions of the notebook, then
   `python task/prepare_data.py`, `python harness/run_task.py`, `python harness/check_rules.py`.
   Our own tests: `python -m pytest tests_local -q`.

## Limits and what did not work

- **Own flash-attention**: correct, but Triton 3.6 compiles `tl.dot` to FMA (no tensor cores) on
  the T4's sm75; it was 15x slower than torch SDPA, so it is kept in `solution/attention.py` but
  not applied.
- **Memory floor**: NF4 weights (~645 MiB), fp16 embedding (445 MiB), CUDA context (~140 MiB) and
  the activations of the longest padded batch. The next steps would be computing without pad
  tokens (the batch keeps the same rows; only token-wise ops run on the real tokens), an int8
  embedding, and fewer Triton specialisations to cut the ~6 s JIT.
- **Wall time**: the steady-state step is 16% faster, but fixed start-up and JIT dominate the
  measured total at 64 steps.
