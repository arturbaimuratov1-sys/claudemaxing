"""torch SDPA variants for Qwen2's GQA on this GPU (investigation, not part of the solution):
time and memory kept for backward, base path (repeat_kv + boolean mask) against
is_causal + enable_gqa (no k/v expansion, no mask: same rows for right padding)."""

import time

import torch
import torch.nn.functional as F

B, H, KVH, T, D = 4, 12, 2, 340, 128
pad = torch.arange(T, device="cuda")[None] < torch.tensor([340, 300, 200, 150], device="cuda")[:, None]
mask = (torch.ones(T, T, dtype=torch.bool, device="cuda").tril()[None] & pad[:, None, :])[:, None]
q = torch.randn(B, H, T, D, device="cuda", requires_grad=True)
k = torch.randn(B, KVH, T, D, device="cuda", requires_grad=True)
v = torch.randn(B, KVH, T, D, device="cuda", requires_grad=True)
g = torch.randn(B, H, T, D, device="cuda")


def base():
    kk, vv = k.repeat_interleave(H // KVH, 1), v.repeat_interleave(H // KVH, 1)
    return F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)


def grouped():
    """one SDPA call per kv head, k/v expanded as stride-0 views (no copy in our code)"""
    g = H // KVH
    q16, k16, v16 = q.half(), k.half(), v.half()
    return torch.cat([F.scaled_dot_product_attention(
        q16[:, j * g:(j + 1) * g], k16[:, j:j + 1].expand(B, g, T, D),
        v16[:, j:j + 1].expand(B, g, T, D), attn_mask=mask) for j in range(KVH)], 1)


def gqa():
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)


for name, f in (("base repeat_kv+mask", base), ("grouped stride-0 k/v", grouped),
                ("is_causal+enable_gqa", gqa)):
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            out = f()
        out.float().backward(g)
        torch.cuda.synchronize()
        m0 = torch.cuda.memory_allocated()
        with torch.autocast("cuda", dtype=torch.float16):
            out = f()
        kept = (torch.cuda.memory_allocated() - m0) / 2 ** 20
        out.float().backward(g)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            with torch.autocast("cuda", dtype=torch.float16):
                out = f()
            out.float().backward(g)
        torch.cuda.synchronize()
        rows = pad[:, None, :, None].expand_as(out)
        with torch.autocast("cuda", dtype=torch.float16):
            ref = base()
        same = torch.equal(out[rows], ref[rows])
        diff = (out.float() - ref.float())[rows].abs().max().item()
        print(f"{name}: {(time.perf_counter() - t) * 100:.2f} ms fwd+bwd, kept {kept:.1f} MiB "
              f"({kept * 2 ** 20 / (B * T):.0f} B/token), valid rows equal to base: {same} "
              f"(max diff {diff:.2e}), out dtype {out.dtype}")
    except Exception as e:  # noqa: BLE001
        print(name, "failed:", type(e).__name__, str(e)[:120])
