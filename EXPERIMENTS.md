# Experiments

| date | commit | version | time | peak mem | eval loss |
|---|---|---|---|---|---|
| 2026-10-30 | no-git | base Soup, T4, repeats 1 | 83.3 s | 14345 MiB | 2.04965 |
| 2026-10-30 | no-git | base Soup, repeat 2 (T4) | 80.1 s | 14345 MiB | 2.04967 |
| 2026-10-31 | no-git | RMSNorm v1 (Builder handoff), T4, check_rules 9/10 (rule 4 FAIL: own kernels 1.5% of time) | 80.2 s (base 79.3 s) | 12383 MiB (base 14345) | 2.04965 (-0.001%) |
| 2026-10-31 | c2ec325 (tag safe-v1) | RMSNorm + SwiGLU-recompute (handoff_1), T4, check_rules 10/10, own kernels 7.6% of time (8.1% in solution-only run) | 81.2 s (base 80.2 s) | 12383 MiB (base 14345) | 2.04965 (base 2.04968) |

Own-kernel profile (full run): silu_mul_bwd_recompute 489.3 ms (448 launches), silu_mul_fwd_persistent 288.4 ms (448), qwen_rmsnorm_bwd_kernel 98.8 ms (896), qwen_rmsnorm_fwd_kernel 68.0 ms (912).
