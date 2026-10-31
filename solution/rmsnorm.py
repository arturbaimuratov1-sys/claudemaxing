"""Fused RMSNorm for Qwen2: forward and backward in Triton, one launch each way.

transformers' Qwen2RMSNorm.forward is a chain of small torch ops (cast to fp32, pow, mean,
rsqrt, multiply, cast back, multiply by the weight) that writes several fp32 temporaries and
keeps them for backward. Here one program handles one row: forward reads x once and writes
the output and one fp32 `rstd` per row; backward recomputes the normalised value from x and
`rstd`, so nothing else is stored.

The output reproduces the reference exactly in dtype and in rounding: the normalised value
is rounded to the input dtype first, and only then multiplied by the weight, so the result
dtype is torch.promote_types(x.dtype, weight.dtype) (fp32 for an fp16 input with the fp32
weight that prepare_model_for_kbit_training leaves in a 4-bit model). All sums are fp32.
"""

import functools
import weakref

import torch
import triton
import triton.language as tl

# Rows wider than this go to the original forward (one program holds a whole row).
MAX_COLS = 16384
_SUPPORTED = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def qwen_rmsnorm_fwd_kernel(x_ptr, w_ptr, out_ptr, rstd_ptr, low_ptr, n_cols, eps,
                            BLOCK: tl.constexpr, SAVE_LOW: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offs = row * n_cols + cols
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Round to the input dtype before the weight, as the reference does.
    normed = (x * rstd).to(x_ptr.dtype.element_ty).to(tl.float32)
    tl.store(rstd_ptr + row, rstd)
    tl.store(out_ptr + offs, (normed * w).to(out_ptr.dtype.element_ty), mask=mask)
    if SAVE_LOW:  # fp16 copy of the output, kept for backward (see RMSNorm.forward)
        tl.store(low_ptr + offs, (normed * w).to(low_ptr.dtype.element_ty), mask=mask)


# n_rows changes with every batch; not specialising on it keeps one compiled variant.
@triton.jit(do_not_specialize=["n_rows"])
def qwen_rmsnorm_bwd_kernel(dy_ptr, x_ptr, w_ptr, rstd_ptr, dx_ptr, dw_part_ptr, n_rows, n_cols,
                            BLOCK: tl.constexpr, COMPUTE_DW: tl.constexpr,
                            SAVED_OUTPUT: tl.constexpr):
    # SAVED_OUTPUT: x_ptr holds the fp16 copy of the output (x * rstd) * w of an fp32 x (frozen
    # weight without zeros, so COMPUTE_DW is False); the normalised value is that divided by w,
    # and the fp32 reference does no rounding. Otherwise x_ptr holds x.
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    dw_acc = tl.zeros([BLOCK], dtype=tl.float32)
    # Persistent loop: each program walks rows pid, pid + n_programs, ... and keeps its own
    # partial sum of dw; torch adds the partial rows afterwards.
    for row in range(pid, n_rows, tl.num_programs(0)):
        offs = row * n_cols + cols
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row)
        if SAVED_OUTPUT:
            normed = tl.where(mask, x / w, 0.0)  # masked lanes: w = 0 there
        else:
            normed = x * rstd
        if COMPUTE_DW:
            dw_acc += dy * normed.to(x_ptr.dtype.element_ty).to(tl.float32)
        # The reference multiplies the rounded value by w, so its gradient is rounded to the
        # input dtype too before it flows back through the cast.
        grad_normed = dy * w
        if not SAVED_OUTPUT:  # the reference rounds to the dtype of x (a no-op for fp32 x)
            grad_normed = grad_normed.to(x_ptr.dtype.element_ty).to(tl.float32)
        mean_dot = tl.sum(grad_normed * normed, axis=0) / n_cols
        dx = (grad_normed - normed * mean_dot) * rstd
        tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)
    if COMPUTE_DW:
        tl.store(dw_part_ptr + pid * n_cols + cols, dw_acc, mask=mask)


@functools.lru_cache(maxsize=None)
def _sm_count(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _launch_shape(n_cols):
    block = triton.next_power_of_2(n_cols)
    return block, min(max(block // 256, 1), 16)


_LAST_LOW = [None, -1, None]  # (weakref to the last output, its version, weakref to its fp16 copy)
_NO_ZEROS = {}


def _no_zeros(weight):
    """True if the frozen weight has no zero (checked once per weight version)."""
    key = id(weight)
    entry = _NO_ZEROS.get(key)
    if entry is None or entry[0]() is not weight or entry[1] != weight._version:
        entry = (weakref.ref(weight), weight._version, not bool((weight == 0).any()))
        _NO_ZEROS[key] = entry
    return entry[2]


def fp16_copy_of(t):
    """The fp16 copy of `t` if `t` is the last output of the fused RMSNorm (bit-identical to
    t.to(torch.float16)), else None."""
    ref, version, low_ref = _LAST_LOW
    if ref is None or ref() is not t or t._version != version:
        return None
    low = low_ref()
    return None if low is None else low.view(t.shape)


class RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        shape = x.shape
        n_cols = shape[-1]
        x2 = x.contiguous().view(-1, n_cols)
        weight = weight.contiguous()
        n_rows = x2.shape[0]
        out = torch.empty((n_rows, n_cols), dtype=torch.promote_types(x.dtype, weight.dtype),
                          device=x.device)
        rstd = torch.empty(n_rows, dtype=torch.float32, device=x.device)
        # Activation compression: with an fp32 x and a frozen weight, backward needs x only
        # for the normalised value x * rstd inside dx. The kernel writes the fp16 copy of the
        # output, (x * rstd) * w, and backward divides it by w (a weight without zeros):
        # 2^-11 relative per element, dx changes by ~1e-5 relative. That fp16 copy is exactly
        # what the next layers cast the output to (q/k/v and the MLP compute in fp16), so they
        # take it from here (fp16_copy_of) and keep the same tensor: one copy for all of them.
        # A trainable weight, or one with zeros, keeps the exact fp32 x.
        low = (torch.empty((n_rows, n_cols), dtype=torch.float16, device=x.device)
               if x.dtype == torch.float32 and weight.dtype == torch.float32
               and not weight.requires_grad and ctx.needs_input_grad[0]
               and _no_zeros(weight) else None)
        if n_rows:
            block, warps = _launch_shape(n_cols)
            qwen_rmsnorm_fwd_kernel[(n_rows,)](x2, weight, out, rstd,
                                               low if low is not None else rstd, n_cols, eps,
                                               BLOCK=block, SAVE_LOW=low is not None,
                                               num_warps=warps)
        ctx.save_for_backward(low if low is not None else x2, weight, rstd)
        ctx.x_dtype = x.dtype
        ctx.shape = shape
        result = out.view(shape)
        if low is not None:
            _LAST_LOW[0], _LAST_LOW[1], _LAST_LOW[2] = (weakref.ref(result), result._version,
                                                         weakref.ref(low))
        return result

    @staticmethod
    def backward(ctx, grad):
        x2, weight, rstd = ctx.saved_tensors
        n_rows, n_cols = x2.shape
        grad2 = grad.contiguous().view(n_rows, n_cols)
        dx = torch.empty(x2.shape, dtype=ctx.x_dtype, device=x2.device)
        need_dw = ctx.needs_input_grad[1]
        dw = None
        if n_rows:
            block, warps = _launch_shape(n_cols)
            programs = min(n_rows, 4 * _sm_count(x2.device.index or 0))
            partial = (torch.empty((programs, n_cols), dtype=torch.float32, device=x2.device)
                       if need_dw else rstd)
            qwen_rmsnorm_bwd_kernel[(programs,)](grad2, x2, weight, rstd, dx, partial, n_rows,
                                                 n_cols, BLOCK=block, COMPUTE_DW=need_dw,
                                                 SAVED_OUTPUT=x2.dtype != ctx.x_dtype,
                                                 num_warps=warps)
            if need_dw:
                dw = partial.sum(0).to(weight.dtype)
        elif need_dw:
            dw = torch.zeros_like(weight)
        return dx.view(ctx.shape), dw, None


def rmsnorm(x, weight, eps):
    return RMSNorm.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    """Send Qwen2RMSNorm.forward through the fused kernel. Class-level, like the MLP patch:
    works before or after Soup builds the model. Anything the kernel does not cover (CPU,
    unusual dtypes, very wide rows, a weight that does not match) goes to the original."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = getattr(modeling_qwen2, "Qwen2RMSNorm", None)
    if cls is None or getattr(cls.forward, "_fused_rmsnorm", False):
        return
    original = cls.forward

    def forward(self, hidden_states):
        weight = self.weight
        if (not isinstance(hidden_states, torch.Tensor) or not hidden_states.is_cuda
                or hidden_states.dtype not in _SUPPORTED or weight.dtype not in _SUPPORTED
                or weight.dim() != 1 or weight.device != hidden_states.device
                or hidden_states.shape[-1] != weight.shape[0]
                or hidden_states.shape[-1] > MAX_COLS):
            return original(self, hidden_states)
        return rmsnorm(hidden_states, weight, self.variance_epsilon)

    forward._fused_rmsnorm = True
    cls.forward = forward
