"""Micro-benchmark of solution/attention.py block configs on the current GPU (run from a folder
that holds solution/). Not part of the solution; prints ms per call and whether tl.dot became
tensor-core mma instructions."""

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
import solution.attention as A  # noqa: E402

B, H, KVH, T, D = 4, 12, 2, 336, 128
pad = torch.arange(T, device="cuda")[None] < torch.tensor([336, 300, 200, 150], device="cuda")[:, None]
mask = (torch.ones(T, T, dtype=torch.bool, device="cuda").tril()[None] & pad[:, None, :])[:, None]
q = torch.randn(B, T, H, D, device="cuda").transpose(1, 2).requires_grad_()
k = torch.randn(B, T, KVH, D, device="cuda").transpose(1, 2).requires_grad_()
v = torch.randn(B, T, KVH, D, device="cuda").transpose(1, 2).requires_grad_()
g = torch.randn(B, T, H, D, device="cuda").half()
kok = A._key_ok(mask, T)


def timed(f, n=10):
    f()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def ours_fwd():
    with torch.no_grad():
        A.qwen_attention(q, k, v, kok, None, torch.float16)


def ours_all():
    A.qwen_attention(q, k, v, kok, None, torch.float16).backward(g)


def sdpa(dtype):
    def f():
        qq, kk, vv = (x.to(dtype) for x in (q, k, v))
        kk, vv = kk.repeat_interleave(6, 1), vv.repeat_interleave(6, 1)
        F.scaled_dot_product_attention(qq, kk, vv, attn_mask=mask).transpose(1, 2).float() \
            .backward(g.float())
    return f


print(torch.cuda.get_device_name(0), "triton", __import__("triton").__version__)
print(f"torch sdpa fp32 fwd+bwd {timed(sdpa(torch.float32)):.2f} ms, "
      f"fp16 {timed(sdpa(torch.float16)):.2f} ms")
configs = [(64, 64, 4), (32, 32, 4), (64, 32, 4), (16, 64, 4), (64, 64, 8), (32, 64, 4)]
for cfg in configs:
    A._BLOCKS["fwd"] = list(cfg)
    try:
        print(f"fwd {cfg}: {timed(ours_fwd):.2f} ms")
    except Exception as e:  # noqa: BLE001
        print(f"fwd {cfg}: {type(e).__name__} {str(e)[:80]}")
A._BLOCKS["fwd"] = [64, 64, 4]
try:
    print(f"ours fwd+bwd default: {timed(ours_all):.2f} ms")
except Exception as e:  # noqa: BLE001
    print("ours fwd+bwd failed", e)
for kern in (A.qwen_attn_fwd_kernel, A.qwen_attn_bwd_dkdv_kernel, A.qwen_attn_bwd_dq_kernel):
    try:
        for dev_cache in kern.device_caches.values():
            for c in list(dev_cache[0].values())[:1]:
                ptx = c.asm.get("ptx", "")
                print(kern.fn.__name__, "mma:", ptx.count("mma.sync"), "fma:", ptx.count("fma.rn"),
                      "local:", ptx.count("ld.local") + ptx.count("st.local"),
                      "regs:", getattr(c, "n_regs", "?"), "spills:", getattr(c, "n_spills", "?"),
                      "shared:", getattr(c, "metadata", None) and c.metadata.shared)
    except Exception as e:  # noqa: BLE001
        print(kern.fn.__name__, "no asm:", type(e).__name__, str(e)[:80])
