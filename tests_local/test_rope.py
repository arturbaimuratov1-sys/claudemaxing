"""Rotary position embedding kernel against transformers' apply_rotary_pos_emb (Qwen2).

Run on a GPU from the repository root: `python -m pytest tests_local -q`.
q and k arrive as views (projection output .view(B, T, H, D).transpose(1, 2)), so the tests
use the same non-contiguous layout. In the 4-bit run q, k, cos and sin are fp32.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

F16, F32 = torch.float16, torch.float32


def original():
    modeling = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    from solution import rope

    return getattr(rope, "ORIGINAL", None) or modeling.apply_rotary_pos_emb


def make(b, t, h, kv, d, dtype, seed=0, cos_batch=None):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, t, h * d, device="cuda", generator=gen).to(dtype)
    k = torch.randn(b, t, kv * d, device="cuda", generator=gen).to(dtype)
    q = q.view(b, t, h, d).transpose(1, 2)
    k = k.view(b, t, kv, d).transpose(1, 2)
    pos = torch.arange(t, device="cuda", dtype=F32)[None].expand(cos_batch or b, t)
    inv = 1.0 / (1e6 ** (torch.arange(0, d, 2, device="cuda", dtype=F32) / d))
    freqs = pos[..., None] * inv
    emb = torch.cat((freqs, freqs), -1)
    return q, k, emb.cos().to(dtype), emb.sin().to(dtype)


def assert_close(a, b, tol):
    scale = b.float().abs().max().clamp(min=1.0).item()
    torch.testing.assert_close(a.float(), b.float(), rtol=tol, atol=tol * scale)


@pytest.mark.parametrize("cos_batch", [None, 1])  # 1: positions shared by the batch (training)
@pytest.mark.parametrize("dtype", [F32, F16])
@pytest.mark.parametrize("shape", [(4, 512, 12, 2, 128), (2, 37, 4, 4, 64), (1, 1, 2, 1, 128)])
def test_forward_and_backward_match(dtype, shape, cos_batch):
    from solution.rope import rope

    ref_fn = original()
    q, k, cos, sin = make(*shape, dtype, cos_batch=cos_batch)
    tol = 2e-3 if dtype == F16 else 1e-5
    gq = torch.randn(q.shape, device="cuda").to(dtype)
    gk = torch.randn(k.shape, device="cuda").to(dtype)

    q1, k1 = q.detach().clone().requires_grad_(), k.detach().clone().requires_grad_()
    rq, rk = ref_fn(q1, k1, cos, sin)
    torch.autograd.backward((rq, rk), (gq, gk))

    q2, k2 = q.detach().clone().requires_grad_(), k.detach().clone().requires_grad_()
    oq, ok = rope(q2, k2, cos, sin)
    torch.autograd.backward((oq, ok), (gq, gk))

    assert oq.dtype == rq.dtype and ok.dtype == rk.dtype
    assert oq.shape == rq.shape and ok.shape == rk.shape
    assert_close(oq, rq, tol)
    assert_close(ok, rk, tol)
    assert_close(q2.grad, q1.grad, tol)
    assert_close(k2.grad, k1.grad, tol)


def test_non_contiguous_gradient():
    from solution.rope import rope

    q, k, cos, sin = make(2, 64, 4, 2, 128, F32)
    q.requires_grad_()
    k.requires_grad_()
    oq, ok = rope(q, k, cos, sin)
    (oq.transpose(1, 2).sum() + ok.mean()).backward()  # gradients with odd strides
    q_ref, k_ref = q.detach().clone().requires_grad_(), k.detach().clone().requires_grad_()
    rq, rk = original()(q_ref, k_ref, cos, sin)
    (rq.transpose(1, 2).sum() + rk.mean()).backward()
    assert_close(q.grad, q_ref.grad, 1e-5)
    assert_close(k.grad, k_ref.grad, 1e-5)


def test_patch_and_fallbacks():
    modeling = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    from solution import rope as mod

    saved = modeling.apply_rotary_pos_emb
    try:
        mod.patch_qwen2_rope()
        assert modeling.apply_rotary_pos_emb is not saved
        mod.patch_qwen2_rope()  # idempotent
        patched = modeling.apply_rotary_pos_emb
        q, k, cos, sin = make(2, 16, 4, 2, 64, F32)
        calls = []
        real = mod.rope
        mod.rope = lambda *a: (calls.append(1), real(*a))[1]
        try:
            patched(q, k, cos, sin)
            assert calls
            # unsqueeze_dim=2 layout and CPU tensors go to the original
            qc, kc = q.transpose(1, 2), k.transpose(1, 2)
            for a, b in zip(patched(qc, kc, cos, sin, unsqueeze_dim=2), saved(qc, kc, cos, sin, 2)):
                assert torch.equal(a, b)
            for a, b in zip(patched(q.cpu(), k.cpu(), cos.cpu(), sin.cpu()),
                            saved(q.cpu(), k.cpu(), cos.cpu(), sin.cpu())):
                assert torch.equal(a, b)
        finally:
            mod.rope = real
    finally:
        modeling.apply_rotary_pos_emb = saved
