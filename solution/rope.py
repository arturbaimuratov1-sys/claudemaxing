"""Rotary position embedding for Qwen2 attention: q * cos + rotate_half(q) * sin, in one pass.

transformers' apply_rotary_pos_emb builds the result from five torch ops per tensor (two
multiplies, a negate, a concatenation, an add), each a full pass over q or k, and backward
runs the same number again. Here one program handles one (batch, head, position) row of
head_dim values: it reads the row and the matching cos/sin rows once and writes the rotated
row; backward is the same kernel with the transposed rotation. q and k arrive as transposed
views of the projection outputs, so the kernel takes their strides instead of copying.
Arithmetic is fp32; the output dtype is torch's promotion of q and cos, as in the reference.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["seq_len"])
def qwen_rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, n_heads, seq_len, stride_b, stride_h,
                     stride_t, cos_stride_b, cos_stride_t, HALF: tl.constexpr,
                     BACKWARD: tl.constexpr):
    row = tl.program_id(0)
    t = row % seq_len
    bh = row // seq_len
    h = bh % n_heads
    b = bh // n_heads
    x_row = x_ptr + b.to(tl.int64) * stride_b + h.to(tl.int64) * stride_h + t.to(tl.int64) * stride_t
    c_row = b.to(tl.int64) * cos_stride_b + t.to(tl.int64) * cos_stride_t
    i = tl.arange(0, HALF)
    x1 = tl.load(x_row + i).to(tl.float32)
    x2 = tl.load(x_row + HALF + i).to(tl.float32)
    c1 = tl.load(cos_ptr + c_row + i).to(tl.float32)
    c2 = tl.load(cos_ptr + c_row + HALF + i).to(tl.float32)
    s1 = tl.load(sin_ptr + c_row + i).to(tl.float32)
    s2 = tl.load(sin_ptr + c_row + HALF + i).to(tl.float32)
    if BACKWARD:  # x is the incoming gradient: g * cos + rotate_half^T(g * sin)
        o1 = x1 * c1 + x2 * s2
        o2 = x2 * c2 - x1 * s1
    else:  # x * cos + [-x2, x1] * sin
        o1 = x1 * c1 - x2 * s1
        o2 = x2 * c2 + x1 * s2
    out_row = out_ptr + row.to(tl.int64) * (2 * HALF)
    tl.store(out_row + i, o1.to(out_ptr.dtype.element_ty))
    tl.store(out_row + HALF + i, o2.to(out_ptr.dtype.element_ty))


def _rotate(x, cos, sin, out_dtype, backward):
    b, h, t, d = x.shape
    if x.stride(-1) != 1:
        x = x.contiguous()
    out = torch.empty((b, h, t, d), dtype=out_dtype, device=x.device)
    if out.numel():
        # cos/sin usually have batch 1 (shared positions), broadcast over the batch.
        cos_stride_b = cos.stride(0) if cos.shape[0] > 1 else 0
        qwen_rope_kernel[(b * h * t,)](
            x, cos, sin, out, h, t, x.stride(0), x.stride(1), x.stride(2), cos_stride_b,
            cos.stride(1), HALF=d // 2, BACKWARD=backward, num_warps=1 if d <= 128 else 2)
    return out


class RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        cos, sin = cos.contiguous(), sin.contiguous()
        q_dtype, k_dtype = torch.promote_types(q.dtype, cos.dtype), torch.promote_types(k.dtype, cos.dtype)
        ctx.save_for_backward(cos, sin)
        ctx.dtypes = (q.dtype, k.dtype)
        return _rotate(q, cos, sin, q_dtype, False), _rotate(k, cos, sin, k_dtype, False)

    @staticmethod
    def backward(ctx, grad_q, grad_k):
        cos, sin = ctx.saved_tensors
        q_dtype, k_dtype = ctx.dtypes
        want_q, want_k = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        dq = _rotate(grad_q, cos, sin, q_dtype, True) if want_q and grad_q is not None else None
        dk = _rotate(grad_k, cos, sin, k_dtype, True) if want_k and grad_k is not None else None
        return dq, dk, None, None


def rope(q, k, cos, sin):
    """apply_rotary_pos_emb(q, k, cos, sin) for q/k of shape [batch, heads, seq, head_dim] and
    cos/sin of shape [batch, seq, head_dim]."""
    return RoPE.apply(q, k, cos, sin)


ORIGINAL = None
_SUPPORTED = (torch.float16, torch.bfloat16, torch.float32)


def patch_qwen2_rope():
    """Replace transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb, which
    Qwen2Attention.forward looks up by name on every call. Anything the kernel does not cover
    (another layout, CPU, odd or large head_dim, gradients for cos/sin) goes to the original."""
    global ORIGINAL
    from transformers.models.qwen2 import modeling_qwen2

    current = getattr(modeling_qwen2, "apply_rotary_pos_emb", None)
    if current is None or getattr(current, "_fused_rope", False):
        return
    ORIGINAL = original = current

    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        d = q.shape[-1] if isinstance(q, torch.Tensor) else 0
        if (unsqueeze_dim != 1 or not all(isinstance(x, torch.Tensor) and x.is_cuda
                                          for x in (q, k, cos, sin))
                or q.dim() != 4 or k.dim() != 4 or cos.dim() != 3 or sin.shape != cos.shape
                or k.shape[-1] != d or cos.shape[-1] != d or d % 2 or d > 256
                or (d // 2) & (d // 2 - 1)  # the half must be a power of two
                or cos.shape[0] not in (1, q.shape[0]) or q.shape[2] != cos.shape[1]
                or k.shape[0] != q.shape[0] or k.shape[2] != q.shape[2]
                or cos.requires_grad or sin.requires_grad or sin.dtype != cos.dtype
                or len({x.device for x in (q, k, cos, sin)}) != 1
                or any(x.dtype not in _SUPPORTED for x in (q, k, cos, sin))):
            return original(q, k, cos, sin, unsqueeze_dim)
        return rope(q, k, cos, sin)

    apply_rotary_pos_emb._fused_rope = True
    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb
