"""LoRA branch of peft's 4-bit Linear (q_proj / v_proj) with a compact backward state.

peft computes `lora_B(lora_A(dropout(x))) * scaling` per adapter layer. Under fp16 autocast
each layer keeps, per token, the fp16 copy of its dropped-out input (lora_A's input, 3072 bytes
for hidden size 1536) and dropout's boolean mask (1536 bytes); q_proj and v_proj do this
separately for the same x. Here:

* the mask comes from the same `torch.native_dropout` call on the same fp32 x (same random
  numbers, same distribution, same scaling 1 / (1 - p));
* the mask is packed to bits by our kernel (192 bytes per token);
* one fp16 copy of x is shared by the q and v layers of a block (they get the same tensor);
* lora_A's input is rebuilt in backward from that copy and the bits.

Per token and layer that is 3.5 KB instead of 9.3 KB. The forward is peft's bit for bit (lora_A's
input fp16(x * scale) is computed from the fp32 x). Only lora_A's weight gradient differs: it
uses the input rebuilt from the fp16 copy, fp16(fp16(x) * scale), at most 2^-11 relative away.
Everything else (fp16 matmuls of the adapter, the scaling, the fp32 sum with the base output,
the gradient casts of autocast) follows peft's computation.
"""

import weakref

import torch
import triton
import triton.language as tl

PACK_BYTES = 512  # mask bytes (8 elements each) per program


@triton.jit
def lora_dropout_pack_kernel(x_ptr, mask_ptr, xd_ptr, bits_ptr, n_bytes, scale,
                             BLOCK: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, 8)
    offs = rows[:, None] * 8 + lane[None, :]
    ok = rows[:, None] < n_bytes
    keep = tl.load(mask_ptr + offs, mask=ok, other=0).to(tl.int32)
    x = tl.load(x_ptr + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(xd_ptr + offs, tl.where(keep != 0, x * scale, 0.0).to(xd_ptr.dtype.element_ty),
             mask=ok)
    tl.store(bits_ptr + rows, tl.sum(keep << lane[None, :], axis=1).to(tl.uint8),
             mask=rows < n_bytes)


@triton.jit
def lora_dropout_unpack_kernel(x_ptr, bits_ptr, g_ptr, xd_ptr, dx_ptr, n_bytes, scale,
                               BLOCK: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, 8)
    offs = rows[:, None] * 8 + lane[None, :]
    ok = rows[:, None] < n_bytes
    bits = tl.load(bits_ptr + rows, mask=rows < n_bytes, other=0).to(tl.int32)
    keep = ((bits[:, None] >> lane[None, :]) & 1) != 0
    x = tl.load(x_ptr + offs, mask=ok, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=ok, other=0.0).to(tl.float32)
    tl.store(xd_ptr + offs, tl.where(keep, x * scale, 0.0).to(xd_ptr.dtype.element_ty), mask=ok)
    tl.store(dx_ptr + offs, tl.where(keep, g * scale, 0.0).to(dx_ptr.dtype.element_ty), mask=ok)


def _pack(x, mask, scale):
    n = x.numel()
    xd = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    bits = torch.empty(n // 8, dtype=torch.uint8, device=x.device)
    lora_dropout_pack_kernel[(triton.cdiv(n // 8, PACK_BYTES),)](
        x, mask, xd, bits, n // 8, scale, BLOCK=PACK_BYTES, num_warps=4)
    return xd, bits


def _unpack(x16, bits, g16, scale, dx_dtype):
    n = x16.numel()
    xd = torch.empty_like(x16)
    dx = torch.empty(x16.shape, dtype=dx_dtype, device=x16.device)
    lora_dropout_unpack_kernel[(triton.cdiv(n // 8, PACK_BYTES),)](
        x16, bits, g16, xd, dx, n // 8, scale, BLOCK=PACK_BYTES, num_warps=4)
    return xd, dx


class LoRABranch(torch.autograd.Function):
    """fp16 `lora_B(lora_A(dropout(x))) * scaling` for fp32 x under fp16 autocast."""

    @staticmethod
    def forward(ctx, x, x16, weight_a, weight_b, p, scaling):
        dropped, mask = torch.native_dropout(x, p, True)
        del dropped  # only the mask is used; free the fp32 copy now
        scale = 1.0 / (1.0 - p)
        xd, bits = _pack(x, mask, scale)  # from fp32 x: lora_A's input exactly as peft's
        del mask
        a16, b16 = weight_a.to(torch.float16), weight_b.to(torch.float16)
        a = torch.nn.functional.linear(xd, a16)
        out = torch.nn.functional.linear(a, b16) * scaling
        ctx.save_for_backward(x16, bits, a, weight_a, weight_b)
        ctx.scale, ctx.scaling, ctx.x_dtype = scale, scaling, x.dtype
        return out

    @staticmethod
    def backward(ctx, grad):
        x16, bits, a, weight_a, weight_b = ctx.saved_tensors
        a16, b16 = weight_a.to(torch.float16), weight_b.to(torch.float16)
        g = grad.to(torch.float16) * ctx.scaling  # MulBackward of `* scaling`, in fp16
        g2 = g.reshape(-1, g.shape[-1])
        grad_b = g2.t().mm(a.reshape(-1, a.shape[-1]))  # lora_B's weight gradient, fp16
        grad_a_in = g.matmul(b16)  # gradient at lora_A's output, fp16
        ga2 = grad_a_in.reshape(-1, grad_a_in.shape[-1])
        grad_xd = grad_a_in.matmul(a16)  # gradient at lora_A's input, fp16
        xd, dx = _unpack(x16, bits, grad_xd.contiguous(), ctx.scale, ctx.x_dtype)
        grad_a = ga2.t().mm(xd.reshape(-1, xd.shape[-1]))
        return (dx, None, grad_a.to(weight_a.dtype), grad_b.to(weight_b.dtype), None, None)


_X16 = [None, None, None]  # (weakref to x, (version, data_ptr, shape), weakref to fp16 copy)


def _shared_fp16(x):
    ref, key, x16_ref = _X16
    now = (x._version, x.data_ptr(), tuple(x.shape))
    x16 = x16_ref() if x16_ref is not None else None
    if x16 is not None and ref is not None and ref() is x and key == now:
        return x16
    from .rmsnorm import fp16_copy_of

    x16 = fp16_copy_of(x)  # the norm before q/v already wrote this exact copy
    if x16 is None:
        x16 = x.detach().to(torch.float16).contiguous()
    # weak references only: the saved tensors of LoRABranch keep x16 alive as long as needed
    _X16[0], _X16[1], _X16[2] = weakref.ref(x), now, weakref.ref(x16)
    return x16


def _autocast_fp16():
    try:
        return torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16
    except (TypeError, AttributeError):  # torch < 2.4
        return torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16


def _make_forward(original):
    def forward(self, x, *args, **kwargs):
        active = self.active_adapters
        name = active[0] if len(active) == 1 else None
        drop = self.lora_dropout[name] if name in getattr(self, "lora_dropout", {}) else None
        if (args or kwargs or self.disable_adapters or self.merged or name is None
                or name not in self.lora_A or name in getattr(self, "lora_variant", {})
                or not self.training or not torch.is_grad_enabled() or not _autocast_fp16()
                or not isinstance(x, torch.Tensor) or not x.is_cuda
                or x.dtype != torch.float32 or x.shape[-1] % 8 or not x.is_contiguous()
                or type(drop) is not torch.nn.Dropout or not 0.0 < drop.p < 1.0
                or type(self.lora_A[name]) is not torch.nn.Linear
                or type(self.lora_B[name]) is not torch.nn.Linear
                or self.lora_A[name].bias is not None or self.lora_B[name].bias is not None
                or self.lora_A[name].weight.dtype != torch.float32
                or self.lora_B[name].weight.dtype != torch.float32
                or getattr(self, "use_dora", {}).get(name, False)
                or getattr(self, "fan_in_fan_out", False)):
            return original(self, x, *args, **kwargs)
        self._check_forward_args(x)
        result = self.base_layer(x)
        lora = LoRABranch.apply(x, _shared_fp16(x), self.lora_A[name].weight,
                                self.lora_B[name].weight, float(drop.p),
                                float(self.scaling[name]))
        return result + lora

    forward._lora_branch = True
    forward._original = original
    return forward


def patch_peft_lora_4bit():
    """peft's LoRA Linear layers (the bitsandbytes 4-bit one the run uses, and the plain one):
    vanilla LoRA with dropout on an fp32 input under fp16 autocast in training goes through
    LoRABranch; everything else keeps peft's forward."""
    classes = []
    try:
        from peft.tuners.lora import bnb as lora_bnb

        classes.append(getattr(lora_bnb, "Linear4bit", None))
    except Exception:  # noqa: BLE001 - no bitsandbytes
        pass
    try:
        from peft.tuners.lora import layer as lora_layer

        classes.append(getattr(lora_layer, "Linear", None))
    except Exception:  # noqa: BLE001 - no peft
        pass
    for cls in classes:
        if cls is not None and not getattr(cls.forward, "_lora_branch", False):
            cls.forward = _make_forward(cls.forward)
