# Experiments

Kaggle T4 numbers come from `harness/run_task.py` and `harness/check_rules.py` of the kit;
local numbers (RTX 4060, Windows) are only for tests and rough memory estimates.

| date | commit | version | where | soup train, s | peak, MiB | eval loss | own kernels share | checks |
|---|---|---|---|---|---|---|---|---|
| 2026-10-30 | — | base Soup 0.77.0 | T4 | 79-80 (83.3 in one run) | 14345 | 2.04968 | — | — |
| 2026-10-30 | c2ec325 (safe-v1) | RMSNorm + SwiGLU | T4 | ~80 | 12383 | 2.04965 | ≥5% | 10/10 |
| 2026-10-31 | wip | + fused head/loss/metrics (lm_head_loss.py) | local 4060, tiny Qwen2 | — | 882 → 257 (tiny model) | gaps ≤0.01% vs base | — | tests_local 105/105, tests/ 79/79 |
| 2026-10-31 | f386eca (safe-v2, local tag) | RMSNorm + SwiGLU + fused head | T4, run 1 of 5, `--repeats 1 --profile` | 70.9 (base 86.3 in the same session) | 8483 (base 14345) | 2.04963 (base 2.04967, -0.002%) | 9.7% | 10/10 |

Run 1 details (Kaggle T4, same session, base and solution once each):
- base: wall 86.28 s, train loop 53.78 s, median step 0.856 s, nvidia-smi peak 14345 MiB, torch max allocated 13373 MiB / reserved 14202 MiB.
- solution: wall 70.9 s, train loop 49.11 s, nvidia-smi peak 8483 MiB, torch max allocated 8135 MiB / reserved 8340 MiB, `apply()` 7.51 s; speedup 1.217x, memory ratio 1.691x.
- check 6: first step 0.01%, mean of all steps 0.01%; check 8: time x0.822, memory x0.591.
- own kernel time over timed steps (ms): silu_mul_bwd_recompute 1927.7, silu_mul_fwd_persistent 1139.0, qwen_rmsnorm_bwd_kernel 488.2, lm_head_xent_rows_kernel 361.1, qwen_rmsnorm_fwd_kernel 287.6 (timed loop 43.29 s).
- profiler top (16 steps, 10.56 s GPU): fp16 GEMMs 20.1% + 15.0% + 7.7%; fp32<->fp16 copies 8.5% + 8.1%; NF4 dequant 5.3%; SDPA backward 5.2%.
- tests_local on Kaggle: 103 passed, 2 failed — both `test_lm_head_trainer` variants: Kaggle exposes two T4s, so the test's Trainer wraps the model in DataParallel and the fused head's backward failed (`size of tensor a (864) must match ... b (2)`). The harness runs with `CUDA_VISIBLE_DEVICES=0`, so the measured runs were not affected; fixed in the next commit by falling back to TRL's path under DataParallel.

| 2026-10-31 | 67d0625 | + MLP gate/up recomputed in backward, DataParallel fallback | T4, run 2 of 5, `--repeats 1 --profile` | 78.8 (base 75.7 in the same session) | 4803 (base 14345) | 2.04964 (base 2.04965, -0.000%) | 8.3% | 10/10 |

Run 2 details:
- base: wall 75.66 s, train loop 53.72 s; solution: wall 78.77 s, train loop 56.96 s, timed loop 51.00 s; speedup 0.961x, memory ratio 2.987x.
- check 6: first step 0.01%, mean 0.01%; check 8: time x1.041, memory x0.335.
- own kernel time (ms): silu_mul_bwd_recompute 1928.4, silu_mul_fwd_persistent 1138.6, qwen_rmsnorm_bwd 491.2, lm_head_xent_rows 367.9, qwen_rmsnorm_fwd 287.3.
- profiler (16 steps): total GPU 12.55 s (run 1: 10.56 s); fp32<->fp16 copies 10.1% + 7.2% (1263 + 906 ms), NF4 dequant 728 ms (561 ms in run 1): the recompute costs GEMMs, dequantisation and dtype casts.
- tests_local on Kaggle with CUDA_VISIBLE_DEVICES=0: 107/107.
- The base wall time moved from 86.3 s (run 1) to 75.7 s (run 2) between sessions: wall-time ratios are noisy at this level.

| 2026-10-31 | fb7a219 | + NF4 fast path for the MLP recompute, fewer Triton specialisations | T4, run 3 of 5, `--repeats 1 --profile` | 75.4 (base 76.0 in the same session) | 4803 (base 14345) | 2.04968 (base 2.04961, +0.003%) | 6.2% | 10/10 |

Run 3 details:
- base: wall 76.05 s, train loop 53.86 s; solution: wall 75.36 s, train loop 53.24 s, timed loop 47.54 s; speedup 1.009x, memory ratio 2.987x.
- check 6: first step 0.01%, mean 0.01%; check 8: time x0.991, memory x0.335.
- own kernel time (ms): silu_mul_bwd_recompute 1135.5, silu_mul_fwd_persistent 785.3, qwen_rmsnorm_bwd 405.0, lm_head_xent_rows 345.0, qwen_rmsnorm_fwd 288.8 -> 6.2% (the SwiGLU kernels now read fp16 gate/up, so they take less time; the share dropped with them).
- profiler (16 steps): total GPU 12.29 s.
- tests_local on Kaggle with CUDA_VISIBLE_DEVICES=0: 109/109.
