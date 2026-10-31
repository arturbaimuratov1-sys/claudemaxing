"""SwiGLU of the Qwen2 MLP, `silu(gate) * up`, with recomputation in backward.

What is different from a plain elementwise port of the three torch ops:

* Backward stores nothing but the two inputs. silu(gate) is recomputed from `gate` (one exp2),
  where torch keeps gate, silu(gate) and up. Nothing is saved on the output side either.
* Backward can write its gradients over the input buffers: dgate lands in the memory of
  `gate`, dup in the memory of `up`, because each element is read once before it is written
  and nothing reads the inputs again. That removes the two fresh (batch, seq, intermediate)
  allocations that backward would make in every layer. It is opt-in (`inplace`) and used by
  the MLP patch, where gate and up are temporaries nobody else holds.
* The kernels are persistent: a fixed number of programs walks the tiles in a loop, instead of
  one program per block, and a program writes only the gradients that were asked for.
* The rounding is the reference's. torch computes silu in fp32 but rounds it to the dtype of
  `gate` before it meets `up`; the gradient of that product is rounded to the same dtype
  before it goes back through silu. The kernels do the same, so values and output dtypes
  (torch.promote_types of the inputs) agree with F.silu(gate) * up. All arithmetic is fp32.
"""

import functools

import torch
import triton
import triton.language as tl

BLOCK = 1024
WARPS = 4
_SUPPORTED = (torch.float16, torch.bfloat16, torch.float32)


# ROUND_SILU: round silu (and its gradient) to the dtype of the gate buffer, as torch does when
# gate is that dtype. False when the buffers hold fp16 copies of what the reference keeps in
# fp32 (the 4-bit MLP fast path below), where the reference does not round.
@triton.jit
def silu_mul_fwd_persistent(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr,
                            ROUND_SILU: tl.constexpr):
    n_tiles = tl.cdiv(n_elements, BLOCK)
    for tile in range(tl.program_id(0), n_tiles, tl.num_programs(0)):
        offs = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        silu = g / (1.0 + tl.exp2(g * -1.4426950408889634))
        if ROUND_SILU:
            silu = silu.to(gate_ptr.dtype.element_ty).to(tl.float32)
        tl.store(out_ptr + offs, (silu * u).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def silu_mul_bwd_recompute(dy_ptr, gate_ptr, up_ptr, dgate_ptr, dup_ptr, n_elements,
                           BLOCK: tl.constexpr, WANT_DGATE: tl.constexpr, WANT_DUP: tl.constexpr,
                           ROUND_SILU: tl.constexpr):
    n_tiles = tl.cdiv(n_elements, BLOCK)
    for tile in range(tl.program_id(0), n_tiles, tl.num_programs(0)):
        offs = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        # Recompute the activation instead of reading a saved copy of it.
        denom = 1.0 + tl.exp2(g * -1.4426950408889634)
        sig = 1.0 / denom
        if WANT_DUP:
            silu = g / denom
            if ROUND_SILU:
                silu = silu.to(gate_ptr.dtype.element_ty).to(tl.float32)
            d_up = (dy * silu).to(dup_ptr.dtype.element_ty)
        if WANT_DGATE:
            u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            d_silu = dy * u
            if ROUND_SILU:
                d_silu = d_silu.to(gate_ptr.dtype.element_ty).to(tl.float32)
            d_gate = (d_silu * sig * (1.0 + g * (1.0 - sig))).to(dgate_ptr.dtype.element_ty)
        # Both outputs are computed before either store, so writing over the inputs is safe.
        if WANT_DGATE:
            tl.store(dgate_ptr + offs, d_gate, mask=mask)
        if WANT_DUP:
            tl.store(dup_ptr + offs, d_up, mask=mask)


@functools.lru_cache(maxsize=None)
def _sm_count(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _programs(n_elements, device):
    return max(1, min(triton.cdiv(n_elements, BLOCK), 8 * _sm_count(device.index or 0)))


class SwiGLURecompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up, inplace):
        gate, up = gate.contiguous(), up.contiguous()
        out = torch.empty(gate.shape, dtype=torch.promote_types(gate.dtype, up.dtype),
                          device=gate.device)
        n = gate.numel()
        if n:
            silu_mul_fwd_persistent[(_programs(n, gate.device),)](gate, up, out, n, BLOCK=BLOCK,
                                                                  ROUND_SILU=True,
                                                                  num_warps=WARPS)
        ctx.save_for_backward(gate, up)
        ctx.inplace = inplace
        return out

    @staticmethod
    def backward(ctx, grad):
        gate, up = ctx.saved_tensors
        want_gate, want_up = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if not (want_gate or want_up):
            return None, None, None
        n = gate.numel()
        if ctx.inplace:
            d_gate, d_up = gate, up
        else:
            d_gate = torch.empty_like(gate) if want_gate else gate
            d_up = torch.empty_like(up) if want_up else up
        if n:
            silu_mul_bwd_recompute[(_programs(n, gate.device),)](
                grad.contiguous(), gate, up, d_gate, d_up, n, BLOCK=BLOCK,
                WANT_DGATE=want_gate, WANT_DUP=want_up, ROUND_SILU=True, num_warps=WARPS)
        return (d_gate if want_gate else None), (d_up if want_up else None), None


def swiglu_recompute(gate, up, inplace=False):
    """silu(gate) * up. With `inplace=True` backward overwrites the saved `gate` and `up`, so
    use it only when they are temporaries and backward runs once (no retain_graph)."""
    return SwiGLURecompute.apply(gate, up, inplace)


def _autocast_state():
    try:
        return torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda")
    except (TypeError, AttributeError):  # torch < 2.4
        return torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype()


class MLPInputRecompute(torch.autograd.Function):
    """silu(gate_proj(x)) * up_proj(x) for frozen projections, keeping only `x` for backward.

    In the 4-bit model the projections return the dtype of their input, and the input of the
    MLP is fp32 (the final norm weights are fp32), so gate and up are two fp32 [tokens, 8960]
    tensors per layer -- the largest thing the activation would otherwise keep. Backward runs
    the two frozen projections again on the saved input (under the autocast state of the
    forward, so the values are the same), takes the activation gradient with the recomputing
    kernel written over the fresh gate/up buffers, and lets autograd carry it back to `x`.

    4-bit fast path: bitsandbytes' Linear4bit computes `matmul_4bit(x.to(compute_dtype))` and
    casts the result back to the dtype of `x`, so the fp32 gate/up are exact fp32 copies of
    fp16 results. With two such layers the input is cast once, gate/up and their gradients
    stay in fp16 (the kernel then skips the fp16 rounding of silu that the fp32 reference does
    not do; the gradients are rounded to fp16 exactly where the reference casts them), and each
    branch's input gradient is cast to the dtype of `x` and added, as the reference does."""

    @staticmethod
    def forward(ctx, x, gate_proj, up_proj):
        nf4 = _nf4_pair(gate_proj, up_proj, x)
        if nf4 is not None:
            import bitsandbytes as bnb

            (wg, qg), (wu, qu), compute = nf4
            x_low = x.to(compute)
            gate = bnb.matmul_4bit(x_low, wg, quant_state=qg)
            up = bnb.matmul_4bit(x_low, wu, quant_state=qu)
            out_dtype, round_silu = x.dtype, x.dtype == compute
        else:
            gate, up = gate_proj(x), up_proj(x)
            out_dtype, round_silu = torch.promote_types(gate.dtype, up.dtype), True
        out = torch.empty(gate.shape, dtype=out_dtype, device=gate.device)
        n = gate.numel()
        if n:
            silu_mul_fwd_persistent[(_programs(n, gate.device),)](
                gate.contiguous(), up.contiguous(), out, n, BLOCK=BLOCK, ROUND_SILU=round_silu,
                num_warps=WARPS)
        ctx.save_for_backward(x)
        ctx.projs, ctx.nf4, ctx.round_silu = (gate_proj, up_proj), nf4, round_silu
        ctx.autocast = _autocast_state()
        return out

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        enabled, dtype = ctx.autocast
        with torch.enable_grad(), torch.autocast("cuda", dtype=dtype, enabled=enabled):
            if ctx.nf4 is not None:
                import bitsandbytes as bnb

                (wg, qg), (wu, qu), compute = ctx.nf4
                x_again = x.detach().to(compute).requires_grad_(True)
                gate = bnb.matmul_4bit(x_again, wg, quant_state=qg)
                up = bnb.matmul_4bit(x_again, wu, quant_state=qu)
            else:
                gate_proj, up_proj = ctx.projs
                x_again = x.detach().requires_grad_(True)
                gate, up = gate_proj(x_again), up_proj(x_again)
        g, u = gate.detach(), up.detach()
        n = g.numel()
        if n:  # gradients written over the recomputed buffers, nothing else allocated
            silu_mul_bwd_recompute[(_programs(n, g.device),)](
                grad.contiguous(), g, u, g, u, n, BLOCK=BLOCK, WANT_DGATE=True, WANT_DUP=True,
                ROUND_SILU=ctx.round_silu, num_warps=WARPS)
        if ctx.nf4 is not None:
            (dx_gate,) = torch.autograd.grad(gate, x_again, g)
            (dx_up,) = torch.autograd.grad(up, x_again, u)
            return dx_gate.to(x.dtype) + dx_up.to(x.dtype), None, None
        torch.autograd.backward((gate, up), (g, u), inputs=[x_again])
        return x_again.grad, None, None


def _nf4_layer(linear):
    """(weight, quant_state, compute dtype) of a bias-free bitsandbytes Linear4bit whose compute
    dtype is fixed, else None (then the module itself is called)."""
    if type(linear).__name__ != "Linear4bit" or getattr(linear, "bias", None) is not None:
        return None
    # Calling bitsandbytes directly skips Module.__call__: keep the module path when hooks
    # (accelerate dispatch/offload, user hooks) would run on it.
    if linear._forward_hooks or linear._forward_pre_hooks or hasattr(linear, "_hf_hook"):
        return None
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn.modules import fix_4bit_weight_quant_state_from_module
    except Exception:  # noqa: BLE001 - no bitsandbytes: not a 4-bit layer
        return None
    if (not isinstance(linear, bnb.nn.Linear4bit) or not linear.compute_type_is_set
            or linear.compute_dtype is None):
        return None
    fix_4bit_weight_quant_state_from_module(linear)
    return (linear.weight, linear.weight.quant_state), linear.compute_dtype


def _nf4_pair(gate_proj, up_proj, x):
    gate, up = _nf4_layer(gate_proj), _nf4_layer(up_proj)
    if gate is None or up is None or gate[1] != up[1] or x.dtype not in (torch.float32, gate[1]):
        return None
    return gate[0], up[0], gate[1]


def _frozen(module):
    return not any(p.requires_grad for p in module.parameters())


def patch_qwen2_mlp_recompute():
    """Send the activation of Qwen2MLP.forward through the recomputing kernel. Class-level,
    like the other patches: works before or after Soup builds the model. Anything the kernel
    does not cover (CPU, another activation, unusual dtypes) keeps the original behaviour."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = getattr(modeling_qwen2, "Qwen2MLP", None)
    if cls is None or getattr(cls.forward, "_swiglu_recompute", False):
        return
    original = cls.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or not isinstance(x, torch.Tensor) or not x.is_cuda:
            return original(self, x)
        if (x.dtype in _SUPPORTED and _frozen(self.gate_proj) and _frozen(self.up_proj)
                and torch.is_grad_enabled() and x.requires_grad):
            return self.down_proj(MLPInputRecompute.apply(x, self.gate_proj, self.up_proj))
        gate, up = self.gate_proj(x), self.up_proj(x)
        if (gate.shape != up.shape or gate.dtype not in _SUPPORTED
                or up.dtype not in _SUPPORTED):
            return self.down_proj(self.act_fn(gate) * up)
        return self.down_proj(swiglu_recompute(gate, up, True))

    forward._swiglu_recompute = True
    cls.forward = forward
