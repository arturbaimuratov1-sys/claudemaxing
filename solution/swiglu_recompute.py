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


@triton.jit
def silu_mul_fwd_persistent(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    n_tiles = tl.cdiv(n_elements, BLOCK)
    for tile in range(tl.program_id(0), n_tiles, tl.num_programs(0)):
        offs = tile.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        silu = (g / (1.0 + tl.exp2(g * -1.4426950408889634))).to(gate_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, (silu.to(tl.float32) * u).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def silu_mul_bwd_recompute(dy_ptr, gate_ptr, up_ptr, dgate_ptr, dup_ptr, n_elements,
                           BLOCK: tl.constexpr, WANT_DGATE: tl.constexpr, WANT_DUP: tl.constexpr):
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
            silu = (g / denom).to(gate_ptr.dtype.element_ty).to(tl.float32)
            d_up = (dy * silu).to(dup_ptr.dtype.element_ty)
        if WANT_DGATE:
            u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            d_silu = (dy * u).to(gate_ptr.dtype.element_ty).to(tl.float32)
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
                WANT_DGATE=want_gate, WANT_DUP=want_up, num_warps=WARPS)
        return (d_gate if want_gate else None), (d_up if want_up else None), None


def swiglu_recompute(gate, up, inplace=False):
    """silu(gate) * up. With `inplace=True` backward overwrites the saved `gate` and `up`, so
    use it only when they are temporaries and backward runs once (no retain_graph)."""
    return SwiGLURecompute.apply(gate, up, inplace)


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
        gate, up = self.gate_proj(x), self.up_proj(x)
        if (gate.shape != up.shape or gate.dtype not in _SUPPORTED
                or up.dtype not in _SUPPORTED):
            return self.down_proj(self.act_fn(gate) * up)
        return self.down_proj(swiglu_recompute(gate, up, True))

    forward._swiglu_recompute = True
    cls.forward = forward
