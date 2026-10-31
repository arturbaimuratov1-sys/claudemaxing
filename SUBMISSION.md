# Submission: claudemaxing

## Links
- Repository, final commit: https://github.com/arturbaimuratov1-sys/claudemaxing/commit/873043d747dbcdadb0b7b0d933c44456d57176c5
  (later commits only add these links to README.md and SUBMISSION.md; `solution/` is the same)
- Public Kaggle notebook, the saved version (`COMMIT` in it = the sha above; the same link is in
  the README): https://www.kaggle.com/code/togamyssovabilkaiyr/notebook9b2547a27b
- Demo video (up to 3 minutes): <link>
- Presentation: <link>

## Team (up to 4 people)
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| | | | | |

## Our kernels
| Kernel (`@triton.jit` name) | File | What it replaces | Share of the training time (check 4) |
|---|---|---|---|
| `nf4_dequant_kernel` | `solution/nf4.py` | bitsandbytes NF4 dequantization (bit-identical) | 6.0% |
| `silu_mul_bwd_recompute`, `silu_mul_fwd_persistent` | `solution/swiglu_recompute.py` | SwiGLU forward, backward with recomputation | 3.5% |
| `lm_head_vocab_chunk_kernel` | `solution/lm_head_loss.py` | logits + cross-entropy + entropy + accuracy + rewind losses over 152K vocab | 1.3% |
| `qwen_rmsnorm_fwd_kernel`, `qwen_rmsnorm_bwd_kernel` | `solution/rmsnorm.py` | fp32 RMSNorm | 1.8% |
| `qwen_rope_kernel` | `solution/rope.py` | rotary embedding | 0.9% |
| `lora_dropout_pack_kernel`, `lora_dropout_unpack_kernel` | `solution/lora.py` | LoRA dropout keeping the fp32 dropped input | 1.4% |
| total | | | 15.0% (11.5% without SwiGLU) |

Ideas taken from other work, with links (we wrote the code ourselves): see [SOURCES.md](SOURCES.md)
(FlashAttention papers, online softmax, Cut Your Losses, Chen et al. recomputation, Triton
tutorials read for patterns only; reference behaviour read in transformers, peft, TRL, Soup and
bitsandbytes).

## Results on Kaggle T4 (the notebook above, version 1, commit 873043d)
| | Base Soup | Ours | Ratio |
|---|---|---|---|
| Time of `soup train`, s | 85.6 | 69.4 | speedup 1.233x |
| Peak GPU memory, MiB | 14345 | 2755 | base/ours 5.207x |
| Eval loss (public rows) | | | -0.008% |

- [x] All 10 checks of `harness/check_rules.py` passed (own kernels 15.3% of the training time;
  loss curve within the limits; 79/79 Soup tests)
- [x] README.md rewritten for our solution: what we did, how to run it, results, limits

## Limitations
See the README: own flash-attention is too slow on the T4 with Triton 3.6 (no tensor cores for
`tl.dot` on sm75), so torch SDPA stays; the wall time is dominated by fixed start-up and Triton
JIT; an int8 embedding and per-row attention for the packed tokens are the next steps.
