"""RMSNorm kernel against plain PyTorch, forward and backward, in fp16 first.

Run on a GPU from the repository root: `python -m pytest tests_local -q`.
The reference is the body of transformers' Qwen2RMSNorm.forward (modeling_qwen2.py), copied
here as plain torch so the numbers do not depend on any patch; one test checks that the copy
and the installed class agree bit for bit.

The first thing these tests pin down is the dtype of the reference output. The reference
casts the normalised value back to the input dtype and only then multiplies by the weight,
so with an fp16 input and an fp32 weight (what prepare_model_for_kbit_training leaves in a
4-bit model) the result follows torch type promotion. `test_reference_output_dtype` prints
what it finds (run with -s); the kernel must return exactly that dtype.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

EPS = 1e-6
F32, F16 = torch.float32, torch.float16
# (input dtype, weight dtype): the T4 case first, then the plain ones.
DTYPES = [(F16, F32), (F16, F16), (F32, F32)]
SHAPES = [(4, 512, 1536), (3, 77, 1001), (2, 1, 4096)]


def reference(x, weight, eps=EPS):
    """Qwen2RMSNorm.forward, line for line."""
    input_dtype = x.dtype
    hidden_states = x.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)


def tolerance(*dtypes):
    return 2e-3 if F16 in dtypes else 1e-4


def assert_close(actual, expected, tol):
    """Relative to the largest value of the tensor, so a single rounding flip of the fp16
    output (one unit in the last place) does not fail a big tensor."""
    scale = expected.float().abs().max().clamp(min=1.0).item()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=tol, atol=tol * scale)


def make(shape, x_dtype, w_dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(shape, device="cuda", generator=g) * 3).to(x_dtype)
    weight = (1 + 0.5 * torch.randn(shape[-1], device="cuda", generator=g)).to(w_dtype)
    return x, weight


@pytest.mark.parametrize("x_dtype,w_dtype", DTYPES)
def test_reference_output_dtype(x_dtype, w_dtype):
    x, weight = make((2, 8, 64), x_dtype, w_dtype)
    out = reference(x, weight)
    print(f"\nreference: x {x_dtype}, weight {w_dtype} -> output {out.dtype}")
    assert out.dtype == torch.promote_types(x_dtype, w_dtype)


def test_reference_copy_matches_transformers_class():
    modeling = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    x, weight = make((3, 40, 1536), F16, F32)
    norm = modeling.Qwen2RMSNorm(1536, eps=EPS).cuda()
    norm.weight.data = weight.clone()
    assert torch.equal(norm(x), reference(x, weight))


@pytest.mark.parametrize("x_dtype,w_dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_forward_matches_pytorch(x_dtype, w_dtype, shape):
    from solution.rmsnorm import rmsnorm

    x, weight = make(shape, x_dtype, w_dtype)
    ref = reference(x, weight)
    out = rmsnorm(x, weight, EPS)
    assert out.dtype == ref.dtype
    assert out.shape == ref.shape
    assert_close(out, ref, tolerance(x_dtype, w_dtype))


@pytest.mark.parametrize("x_dtype,w_dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_backward_matches_pytorch(x_dtype, w_dtype, shape):
    from solution.rmsnorm import rmsnorm

    x, weight = make(shape, x_dtype, w_dtype)
    grad_dtype = torch.promote_types(x_dtype, w_dtype)
    grad = torch.randn(shape, device="cuda", dtype=torch.float32).to(grad_dtype)

    x_ref, w_ref = x.clone().requires_grad_(), weight.clone().requires_grad_()
    reference(x_ref, w_ref).backward(grad)

    x_new, w_new = x.clone().requires_grad_(), weight.clone().requires_grad_()
    rmsnorm(x_new, w_new, EPS).backward(grad)

    assert x_new.grad.dtype == x_ref.grad.dtype == x_dtype
    assert w_new.grad.dtype == w_ref.grad.dtype == w_dtype
    tol = tolerance(x_dtype, w_dtype)
    assert_close(x_new.grad, x_ref.grad, tol)
    assert_close(w_new.grad, w_ref.grad, tol)


def test_backward_with_frozen_weight():
    """In training the norm weights are frozen and only the input needs a gradient."""
    from solution.rmsnorm import rmsnorm

    x, weight = make((4, 128, 1536), F16, F32)
    grad = torch.randn(x.shape, device="cuda", dtype=F32)

    x_ref = x.clone().requires_grad_()
    reference(x_ref, weight).backward(grad)
    x_new = x.clone().requires_grad_()
    rmsnorm(x_new, weight, EPS).backward(grad)
    assert weight.grad is None
    assert_close(x_new.grad, x_ref.grad, tolerance(F16, F32))


def test_non_contiguous_input():
    from solution.rmsnorm import rmsnorm

    base = torch.randn(4, 64, 2 * 1536, device="cuda").to(F16)
    x = base[..., :1536]
    assert not x.is_contiguous()
    weight = torch.ones(1536, device="cuda", dtype=F32)
    assert_close(rmsnorm(x, weight, EPS), reference(x, weight), tolerance(F16, F32))


def test_patch_routes_qwen2_rmsnorm_through_the_kernel():
    modeling = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    from solution import rmsnorm as mod

    cls = modeling.Qwen2RMSNorm
    saved = cls.forward
    try:
        norm = cls(1536, eps=EPS).cuda()
        norm.weight.data = (1 + 0.5 * torch.randn(1536, device="cuda")).to(F32)
        x = (torch.randn(4, 256, 1536, device="cuda") * 3).to(F16)
        ref = saved(norm, x)

        mod.patch_qwen2_rmsnorm()
        assert cls.forward is not saved
        calls = []
        real = mod.rmsnorm
        mod.rmsnorm = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
        try:
            out = norm(x)
        finally:
            mod.rmsnorm = real
        assert calls, "the patched forward did not call the Triton path"
        assert out.dtype == ref.dtype
        assert_close(out, ref, tolerance(F16, F32))
    finally:
        cls.forward = saved


def test_patch_falls_back_for_cpu_and_wide_rows():
    modeling = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    from solution import rmsnorm as mod

    cls = modeling.Qwen2RMSNorm
    saved = cls.forward
    try:
        mod.patch_qwen2_rmsnorm()
        cpu = cls(64, eps=EPS)
        x = torch.randn(2, 5, 64)
        assert torch.equal(cpu(x), saved(cpu, x))

        wide = cls(mod.MAX_COLS + 1, eps=EPS).cuda()
        xw = torch.randn(2, mod.MAX_COLS + 1, device="cuda")
        assert torch.equal(wide(xw), saved(wide, xw))
    finally:
        cls.forward = saved


def test_patch_is_idempotent():
    modeling = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    from solution import rmsnorm as mod

    cls = modeling.Qwen2RMSNorm
    saved = cls.forward
    try:
        mod.patch_qwen2_rmsnorm()
        first = cls.forward
        mod.patch_qwen2_rmsnorm()
        assert cls.forward is first
    finally:
        cls.forward = saved
