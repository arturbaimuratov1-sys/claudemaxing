"""Our causal GQA attention against torch SDPA: values and gradients of q, k, v.

The reference is torch in fp32 (what the base run computes on a T4); the bar is torch's own
SDPA in fp16 on the same inputs: our error against the fp32 reference must stay within a small
multiple of the fp16 SDPA error.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

from solution.attention import _key_ok, qwen_attention  # noqa: E402


def reference(q, k, v, mask, dtype):
    """transformers' sdpa path: repeat_kv, boolean mask (or is_causal), output [B, T, H, D]."""
    rep = q.shape[1] // k.shape[1]
    q, k, v = (x.to(dtype) for x in (q, k, v))
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    if mask is None:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out.transpose(1, 2)


def masks(lengths, T):
    pad = torch.arange(T, device="cuda")[None, :] < torch.tensor(lengths, device="cuda")[:, None]
    causal = torch.ones(T, T, dtype=torch.bool, device="cuda").tril()
    return (causal[None] & pad[:, None, :])[:, None], pad


def rel(a, b, rows):
    a, b = a.float()[rows], b.float()[rows]
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("B,H,KVH,T,D,lengths", [
    (2, 12, 2, 128, 128, None),
    (4, 12, 2, 200, 128, [200, 150, 77, 1]),
    (1, 4, 4, 70, 64, [70]),
    (3, 12, 2, 512, 128, [512, 300, 129]),
])
def test_against_torch(B, H, KVH, T, D, lengths):
    torch.manual_seed(0)
    # fp32 inputs laid out like Qwen2's (projection [B, T, H*D] viewed and transposed)
    q = torch.randn(B, T, H, D, device="cuda").transpose(1, 2).requires_grad_(True)
    k = torch.randn(B, T, KVH, D, device="cuda").transpose(1, 2).requires_grad_(True)
    v = torch.randn(B, T, KVH, D, device="cuda").transpose(1, 2).requires_grad_(True)
    mask, pad = masks(lengths, T) if lengths else (None, torch.ones(B, T, dtype=torch.bool,
                                                                     device="cuda"))
    key_ok = None if mask is None else _key_ok(mask, T)
    assert key_ok is not False
    g = torch.randn(B, T, H, D, device="cuda")
    g = g * pad[:, :, None, None]  # padded rows carry no loss in training

    results = {}
    for name in ("ours", "fp32", "fp16"):
        for x in (q, k, v):
            x.grad = None
        if name == "ours":
            out = qwen_attention(q, k, v, key_ok, None, torch.float16)
            assert out.dtype == torch.float16 and out.shape == (B, T, H, D)
        else:
            out = reference(q, k, v, mask, torch.float32 if name == "fp32" else torch.float16)
        out.float().backward(g)
        results[name] = (out.detach(), q.grad.clone(), k.grad.clone(), v.grad.clone())
    for x, d in zip((q, k, v), results["ours"][1:]):
        assert d.dtype == x.dtype and d.shape == x.shape
    rows = pad  # padded query rows are not compared (their outputs are never used)
    for i, name in enumerate(("out", "dq", "dk", "dv")):
        r = rows if name in ("out", "dq") else slice(None)
        a, b, c = (results[n][i] for n in ("ours", "fp32", "fp16"))
        if name in ("out",):
            a, b, c = a.transpose(1, 2), b.transpose(1, 2), c.transpose(1, 2)
            a, b, c = (x.permute(0, 2, 1, 3) for x in (a, b, c))  # [B, T, H, D]
        elif name == "dq":
            a, b, c = (x.transpose(1, 2) for x in (a, b, c))  # [B, T, H, D]
        err, bar = rel(a, b, r), rel(c, b, r)
        assert torch.isfinite(a).all(), name
        print(name, f"{err:.2e} vs fp16 sdpa {bar:.2e}")
        assert err <= 3 * bar + 1e-4, (name, err, bar)


def test_mask_detection():
    mask, _ = masks([5, 3], 6)
    assert _key_ok(mask, 6).tolist() == [[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]]
    bad = mask.clone()
    bad[0, 0, 2, 4] = True  # not causal
    assert _key_ok(bad, 6) is False
    assert _key_ok(mask[:1], 6, 2) is False  # a broadcast batch dim is refused
    left = masks([6, 6], 6)[0].clone()
    left[1, 0, :, 0] = False  # left padding: still "causal & key ok"
    assert _key_ok(left, 6) is not False


def test_patch_routes_qwen2():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from solution.attention import patch_qwen2_attention

    patch_qwen2_attention()
    cfg = Qwen2Config(vocab_size=100, hidden_size=256, intermediate_size=128,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                      attn_implementation="sdpa")
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg).cuda()
    ids = torch.randint(0, 100, (2, 20), device="cuda")
    am = torch.ones(2, 20, dtype=torch.long, device="cuda")
    am[1, 13:] = 0
    calls = []
    import solution.attention as A

    original = A.QwenAttention.forward

    def spy(ctx, *a):
        calls.append(1)
        return original(ctx, *a)

    from transformers.modeling_utils import AttentionInterface

    A.QwenAttention.forward = staticmethod(spy)
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            ours = model(input_ids=ids, attention_mask=am).logits
    finally:
        A.QwenAttention.forward = staticmethod(original)
        AttentionInterface._global_mapping["sdpa"] = A.ORIGINAL  # do not leak the patch
        A.patch_qwen2_attention.__globals__["ORIGINAL"] = None
    assert calls, "the kernel did not run"
    model.config._attn_implementation = "eager"
    with torch.autocast("cuda", dtype=torch.float16):
        ref = model(input_ids=ids, attention_mask=am).logits
    keep = am.bool()
    assert ((ours - ref)[keep].norm() / ref[keep].norm()).item() < 2e-3


@pytest.mark.parametrize("blocks", [{"fwd": [32, 32, 4], "dkdv": [16, 32, 4], "dq": [32, 16, 4]},
                                    {"fwd": [16, 64, 4], "dkdv": [64, 16, 4], "dq": [16, 64, 4]}])
def test_smaller_blocks_same_result(blocks):
    """The fallback block sizes (after an out-of-resources launch) give the same numbers."""
    import solution.attention as A

    torch.manual_seed(1)
    B, H, KVH, T, D = 2, 12, 2, 150, 128
    q = torch.randn(B, T, H, D, device="cuda").transpose(1, 2).requires_grad_(True)
    k = torch.randn(B, T, KVH, D, device="cuda").transpose(1, 2).requires_grad_(True)
    v = torch.randn(B, T, KVH, D, device="cuda").transpose(1, 2).requires_grad_(True)
    mask, _ = masks([150, 90], T)
    key_ok = _key_ok(mask, T)
    g = torch.randn(B, T, H, D, device="cuda")

    def run():
        for x in (q, k, v):
            x.grad = None
        out = qwen_attention(q, k, v, key_ok, None, torch.float16)
        out.float().backward(g)
        return out.float(), q.grad.clone(), k.grad.clone(), v.grad.clone()

    ref = run()
    saved = {n: list(b) for n, b in A._BLOCKS.items()}
    try:
        A._BLOCKS.update({n: list(b) for n, b in blocks.items()})
        got = run()
    finally:
        A._BLOCKS.update(saved)
    for a, b in zip(got, ref):
        assert ((a - b).norm() / b.norm()).item() < 2e-3
