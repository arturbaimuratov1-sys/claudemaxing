"""SwiGLU with recomputation in backward, against plain PyTorch, fp16 first.

Run on a GPU from the repository root: `python -m pytest tests_local -q`.

The reference is the activation of Qwen2MLP written with ordinary torch ops on the real
dtypes: `F.silu(gate) * up`. Note what that does: silu is computed in fp32 inside but rounded
to the dtype of `gate` before it meets `up`, so the product is taken of an already rounded
value, and the output dtype follows torch type promotion of the two inputs. The kernel has to
reproduce both. `test_reference_output_dtype` prints the dtype it finds (run with -s).

Besides the numbers these tests pin the point of the kernel: backward must not keep anything
but the two inputs (silu(gate) is recomputed), and the in-place variant must write its
gradients into the buffers of the inputs instead of allocating new ones.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

F16, F32 = torch.float16, torch.float32
# (dtype of gate, dtype of up): the Qwen2 case first, then the mixed and plain ones.
DTYPES = [(F16, F16), (F16, F32), (F32, F16), (F32, F32)]
SHAPES = [(4, 512, 8960), (3, 77, 1001), (1, 1, 8960), (5, 3)]


def reference(gate, up):
    return F.silu(gate) * up


def tolerance(*dtypes):
    return 2e-3 if F16 in dtypes else 1e-4


def assert_close(actual, expected, tol):
    """Relative to the largest value of the tensor, so one rounding flip of an fp16 element
    (a unit in the last place) does not fail a big tensor."""
    scale = expected.float().abs().max().clamp(min=1.0).item()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=tol, atol=tol * scale)


def make(shape, g_dtype, u_dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    gate = (torch.randn(shape, device="cuda", generator=gen) * 3).to(g_dtype)
    up = (torch.randn(shape, device="cuda", generator=gen) * 2).to(u_dtype)
    return gate, up


@pytest.mark.parametrize("g_dtype,u_dtype", DTYPES)
def test_reference_output_dtype(g_dtype, u_dtype):
    gate, up = make((2, 8, 64), g_dtype, u_dtype)
    out = reference(gate, up)
    print(f"\nreference: gate {g_dtype}, up {u_dtype} -> output {out.dtype}")
    assert out.dtype == torch.promote_types(g_dtype, u_dtype)


@pytest.mark.parametrize("g_dtype,u_dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_forward_matches_pytorch(g_dtype, u_dtype, shape):
    from solution.swiglu_recompute import swiglu_recompute

    gate, up = make(shape, g_dtype, u_dtype)
    ref = reference(gate, up)
    out = swiglu_recompute(gate, up)
    assert out.dtype == ref.dtype
    assert out.shape == ref.shape
    assert_close(out, ref, tolerance(g_dtype, u_dtype))


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("g_dtype,u_dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_backward_matches_pytorch(g_dtype, u_dtype, shape, inplace):
    from solution.swiglu_recompute import swiglu_recompute

    gate, up = make(shape, g_dtype, u_dtype)
    out_dtype = torch.promote_types(g_dtype, u_dtype)
    grad = torch.randn(shape, device="cuda", dtype=F32).to(out_dtype)

    g_ref, u_ref = gate.clone().requires_grad_(), up.clone().requires_grad_()
    reference(g_ref, u_ref).backward(grad)

    g_new, u_new = gate.clone().requires_grad_(), up.clone().requires_grad_()
    swiglu_recompute(g_new, u_new, inplace).backward(grad)

    assert g_new.grad.dtype == g_ref.grad.dtype == g_dtype
    assert u_new.grad.dtype == u_ref.grad.dtype == u_dtype
    tol = tolerance(g_dtype, u_dtype)
    assert_close(g_new.grad, g_ref.grad, tol)
    assert_close(u_new.grad, u_ref.grad, tol)


def test_backward_keeps_only_the_inputs():
    """silu(gate) is recomputed: the only tensors autograd stores are gate and up."""
    from solution.swiglu_recompute import swiglu_recompute

    gate, up = make((4, 256, 8960), F16, F16)
    gate.requires_grad_()
    up.requires_grad_()
    saved = []

    def pack(tensor):
        saved.append(tensor.data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        out = swiglu_recompute(gate, up)
    assert sorted(saved) == sorted([gate.data_ptr(), up.data_ptr()])

    saved_ref = []
    with torch.autograd.graph.saved_tensors_hooks(
            lambda t: (saved_ref.append(t.data_ptr()), t)[1], lambda t: t):
        reference(gate, up)
    assert len(set(saved_ref)) > 2, "the torch reference stores more than the two inputs"
    del out


def test_inplace_backward_reuses_the_buffers():
    from solution.swiglu_recompute import swiglu_recompute

    # gate and up are temporaries (outputs of other ops), as they are in the MLP. A leaf would
    # be copied by autograd when its gradient is accumulated, which hides the reuse.
    base_g, base_u = make((2, 128, 1001), F16, F16)
    base_g.requires_grad_()
    base_u.requires_grad_()
    gate, up = base_g * 1.0, base_u * 1.0
    grad = torch.randn(gate.shape, device="cuda", dtype=F16)
    seen = {}
    gate.register_hook(lambda g: seen.__setitem__("gate", g.data_ptr()))
    up.register_hook(lambda g: seen.__setitem__("up", g.data_ptr()))
    pointers = {"gate": gate.data_ptr(), "up": up.data_ptr()}
    swiglu_recompute(gate, up, True).backward(grad)
    assert seen == pointers

    g_ref, u_ref = base_g.detach().clone().requires_grad_(), base_u.detach().clone().requires_grad_()
    reference(g_ref, u_ref).backward(grad)
    assert_close(base_g.grad, g_ref.grad, 2e-3)
    assert_close(base_u.grad, u_ref.grad, 2e-3)


def test_not_inplace_leaves_inputs_untouched():
    from solution.swiglu_recompute import swiglu_recompute

    gate, up = make((2, 128, 1001), F16, F16)
    before = gate.clone(), up.clone()
    gate.requires_grad_()
    up.requires_grad_()
    swiglu_recompute(gate, up, False).backward(torch.randn(gate.shape, device="cuda", dtype=F16))
    assert torch.equal(gate, before[0]) and torch.equal(up, before[1])


@pytest.mark.parametrize("inplace", [False, True])
def test_only_one_input_needs_grad(inplace):
    from solution.swiglu_recompute import swiglu_recompute

    gate, up = make((3, 64, 1001), F16, F16)
    grad = torch.randn(gate.shape, device="cuda", dtype=F16)

    g_ref = gate.clone().requires_grad_()
    reference(g_ref, up).backward(grad)
    g_new = gate.clone().requires_grad_()
    swiglu_recompute(g_new, up.clone(), inplace).backward(grad)
    assert_close(g_new.grad, g_ref.grad, 2e-3)

    u_ref = up.clone().requires_grad_()
    reference(gate, u_ref).backward(grad)
    u_new = up.clone().requires_grad_()
    swiglu_recompute(gate.clone(), u_new, inplace).backward(grad)
    assert_close(u_new.grad, u_ref.grad, 2e-3)


def test_non_contiguous_input():
    from solution.swiglu_recompute import swiglu_recompute

    base = (torch.randn(4, 64, 2 * 1001, device="cuda") * 3).to(F16)
    gate = base[..., :1001]
    up = (torch.randn(4, 64, 1001, device="cuda") * 2).to(F16)
    assert not gate.is_contiguous()
    assert_close(swiglu_recompute(gate, up), reference(gate, up), 2e-3)


def _small_mlp(modeling):
    config = modeling.Qwen2Config(hidden_size=256, intermediate_size=704, hidden_act="silu")
    torch.manual_seed(0)
    return modeling.Qwen2MLP(config).cuda().half()


def test_patched_qwen2_mlp_matches_the_original():
    modeling = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    from solution import swiglu_recompute as mod

    cls = modeling.Qwen2MLP
    saved = cls.forward
    try:
        mlp = _small_mlp(modeling)
        x = (torch.randn(4, 96, 256, device="cuda") * 2).half()
        grad = torch.randn(4, 96, 256, device="cuda").half()

        x_ref = x.clone().requires_grad_()
        saved(mlp, x_ref).backward(grad)
        ref_out = saved(mlp, x)
        ref_grads = {n: p.grad.clone() for n, p in mlp.named_parameters()}
        mlp.zero_grad()

        mod.patch_qwen2_mlp_recompute()
        assert cls.forward is not saved
        x_new = x.clone().requires_grad_()
        out = mlp(x_new)
        out.backward(grad)
        assert out.dtype == ref_out.dtype
        assert_close(out, ref_out, 2e-3)
        assert_close(x_new.grad, x_ref.grad, 2e-3)
        for name, p in mlp.named_parameters():
            assert_close(p.grad, ref_grads[name], 2e-3)
    finally:
        cls.forward = saved


def test_patch_falls_back_for_cpu_and_other_activations():
    modeling = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    from solution import swiglu_recompute as mod

    cls = modeling.Qwen2MLP
    saved = cls.forward
    try:
        mod.patch_qwen2_mlp_recompute()
        cpu = modeling.Qwen2MLP(modeling.Qwen2Config(hidden_size=32, intermediate_size=64))
        x = torch.randn(2, 5, 32)
        assert torch.equal(cpu(x), saved(cpu, x))

        gelu = modeling.Qwen2MLP(
            modeling.Qwen2Config(hidden_size=32, intermediate_size=64, hidden_act="gelu")).cuda()
        xg = torch.randn(2, 5, 32, device="cuda")
        assert torch.equal(gelu(xg), saved(gelu, xg))
    finally:
        cls.forward = saved


def test_patch_is_idempotent():
    modeling = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    from solution import swiglu_recompute as mod

    cls = modeling.Qwen2MLP
    saved = cls.forward
    try:
        mod.patch_qwen2_mlp_recompute()
        first = cls.forward
        mod.patch_qwen2_mlp_recompute()
        assert cls.forward is first
    finally:
        cls.forward = saved


def test_apply_uses_the_new_mlp_patch_not_the_example():
    modeling = pytest.importorskip("transformers.models.qwen2.modeling_qwen2")
    import solution

    mlp_saved = modeling.Qwen2MLP.forward
    norm_saved = modeling.Qwen2RMSNorm.forward
    try:
        solution.apply()
        assert getattr(modeling.Qwen2MLP.forward, "_swiglu_recompute", False)
    finally:
        modeling.Qwen2MLP.forward = mlp_saved
        modeling.Qwen2RMSNorm.forward = norm_saved
