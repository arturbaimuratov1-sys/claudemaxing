# Experiments

Kaggle T4 numbers come from `harness/run_task.py` and `harness/check_rules.py` of the kit;
local numbers (RTX 4060, Windows) are only for tests and rough memory estimates.

| date | commit | version | where | soup train, s | peak, MiB | eval loss | own kernels share | checks |
|---|---|---|---|---|---|---|---|---|
| 2026-10-30 | — | base Soup 0.77.0 | T4 | 79-80 (83.3 in one run) | 14345 | 2.04968 | — | — |
| 2026-10-30 | c2ec325 (safe-v1) | RMSNorm + SwiGLU | T4 | ~80 | 12383 | 2.04965 | ≥5% | 10/10 |
| 2026-10-31 | wip | + fused head/loss/metrics (lm_head_loss.py) | local 4060, tiny Qwen2 | — | 882 → 257 (tiny model) | gaps ≤0.01% vs base | — | tests_local 105/105, tests/ 79/79 |
