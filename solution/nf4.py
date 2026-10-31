"""NF4 weight dequantization (bitsandbytes 4-bit, double-quantized absmax) in one Triton kernel.

Every 4-bit matmul of the run (forward, the input gradient in backward, the MLP recompute)
first expands the frozen NF4 weight to fp16. bitsandbytes does that in two CUDA launches: the
8-bit absmax is dequantized (code2[q] * absmax2 + offset, fp32), then each 4-bit code is looked
up and multiplied by its block's absmax. This kernel does both in one pass over the packed
weight, with the same fp32 operations in the same order (fp contraction off, so no fused
multiply-add changes a rounding), and writes the fp16 weight: bit-identical to bitsandbytes.

Layout (bitsandbytes 0.50): byte i holds elements 2i (high nibble) and 2i+1 (low nibble);
element j belongs to absmax block j // blocksize; with double quantization that block's absmax
is code2[absmax8[b]] * absmax2[b // blocksize2] + offset.
"""

import torch
import triton
import triton.language as tl

BLOCK = 1024  # packed bytes per program
WARPS = 4


@triton.jit
def nf4_dequant_kernel(packed_ptr, absmax_ptr, code_ptr, code2_ptr, absmax2_ptr, offset_ptr,
                       out_ptr, n_bytes, BLOCKSIZE: tl.constexpr, BLOCKSIZE2: tl.constexpr,
                       NESTED: tl.constexpr, BLOCK: tl.constexpr):
    # One program: BLOCK packed bytes -> 2 * BLOCK consecutive outputs.
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_bytes
    byte = tl.load(packed_ptr + offs, mask=mask, other=0).to(tl.int32)
    blk = (offs * 2) // BLOCKSIZE  # both elements of a byte share a block (even blocksize)
    if NESTED:
        q8 = tl.load(absmax_ptr + blk, mask=mask, other=0).to(tl.int32)
        scale = tl.load(code2_ptr + q8, mask=mask, other=0.0) *             tl.load(absmax2_ptr + blk // BLOCKSIZE2, mask=mask, other=0.0)
        scale = scale + tl.load(offset_ptr)
    else:
        scale = tl.load(absmax_ptr + blk, mask=mask, other=0.0)
    hi = tl.load(code_ptr + (byte >> 4), mask=mask, other=0.0) * scale
    lo = tl.load(code_ptr + (byte & 0xF), mask=mask, other=0.0) * scale
    values = tl.reshape(tl.join(hi, lo), (2 * BLOCK,))  # element 2i = high nibble of byte i
    out_offs = tl.program_id(0).to(tl.int64) * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    tl.store(out_ptr + out_offs, values.to(out_ptr.dtype.element_ty),
             mask=out_offs < 2 * n_bytes)


def _supported(packed, state, dtype):
    return (packed.is_cuda and packed.dtype == torch.uint8 and state is not None
            and state.quant_type == "nf4" and dtype in (torch.float16, torch.float32)
            and state.code is not None and state.code.numel() == 16
            and state.code.dtype == torch.float32 and state.code.is_cuda
            and state.absmax.is_cuda and len(state.shape) == 2
            and (state.shape[0] * state.shape[1]) % 2 == 0 and state.blocksize % 2 == 0
            and (not state.nested or (state.state2.blocksize > 0
                                      and state.state2.code.dtype == torch.float32
                                      and state.state2.absmax.dtype == torch.float32
                                      and state.absmax.dtype == torch.uint8))
            and (state.nested or state.absmax.dtype == torch.float32))


def dequantize_nf4(packed, state, dtype=None, out=None):
    """The [N, K] weight of an NF4 QuantState in `dtype` (default: state.dtype)."""
    dtype = dtype or state.dtype
    n = state.shape[0] * state.shape[1]
    if out is None:
        out = torch.empty(tuple(state.shape), dtype=dtype, device=packed.device)
    if n:
        nested = bool(state.nested)
        s2 = state.state2 if nested else None
        absmax2 = s2.absmax if nested else state.absmax
        code2 = s2.code if nested else state.code
        # offset stays on the GPU (a float() here would synchronise every call)
        offset = _offset_tensor(state.offset, packed.device) if nested else absmax2
        n_bytes = n // 2  # n is even: the [N, K] weights here have even K
        nf4_dequant_kernel[(triton.cdiv(n_bytes, BLOCK),)](
            packed, state.absmax, state.code, code2, absmax2, offset, out, n_bytes,
            BLOCKSIZE=state.blocksize, BLOCKSIZE2=s2.blocksize if nested else 1, NESTED=nested,
            BLOCK=BLOCK, num_warps=WARPS, enable_fp_fusion=False)
    return out


ORIGINALS = {}


def patch_bnb_nf4_dequant():
    """Route bitsandbytes' NF4 weight dequantization through our kernel: the dequant + F.linear
    path of gemm_4bit (forward) and functional.dequantize_4bit (the input gradient in
    MatMul4Bit.backward). Anything else (fp4, bf16, CPU, an `out=` buffer) keeps the original."""
    if ORIGINALS:
        return
    import bitsandbytes.functional as bnbF
    from bitsandbytes.backends.cuda import ops as cuda_ops
    from bitsandbytes.functional import QuantState

    original_fallback = cuda_ops._dequant_linear_fallback
    original_dequant = bnbF.dequantize_4bit
    ORIGINALS.update(fallback=original_fallback, dequant=original_dequant)

    def _dequant_linear_fallback(A, B, shapeB, absmax, blocksize, quant_type, bias=None,
                                 absmax_8bit=None, absmax_code=None, absmax_offset=None):
        if (quant_type != "nf4" or A.dtype not in (torch.float16, torch.float32) or not A.is_cuda
                or (absmax_8bit is not None and (absmax_code is None or absmax_offset is None))):
            return original_fallback(A, B, shapeB, absmax, blocksize, quant_type, bias,
                                     absmax_8bit=absmax_8bit, absmax_code=absmax_code,
                                     absmax_offset=absmax_offset)
        code = _nf4_code(A.device)
        if absmax_8bit is not None:  # absmax here is state2's absmax (see MatMul4Bit.forward)
            state2 = QuantState(absmax=absmax, code=absmax_code, blocksize=256, dtype=torch.float32)
            state = QuantState(absmax=absmax_8bit, shape=torch.Size(shapeB), code=code,
                               blocksize=blocksize, quant_type="nf4", dtype=A.dtype,
                               offset=absmax_offset, state2=state2)
        else:
            state = QuantState(absmax=absmax, shape=torch.Size(shapeB), code=code,
                               blocksize=blocksize, quant_type="nf4", dtype=A.dtype)
        if not _supported(B, state, A.dtype):
            return original_fallback(A, B, shapeB, absmax, blocksize, quant_type, bias,
                                     absmax_8bit=absmax_8bit, absmax_code=absmax_code,
                                     absmax_offset=absmax_offset)
        return torch.nn.functional.linear(A, dequantize_nf4(B, state, A.dtype), bias)

    def dequantize_4bit(A, quant_state=None, absmax=None, out=None, blocksize=None,
                        quant_type="fp4"):
        if out is None and quant_state is not None and _supported(A, quant_state,
                                                                    quant_state.dtype):
            return dequantize_nf4(A, quant_state)
        return original_dequant(A, quant_state, absmax, out, blocksize, quant_type)

    cuda_ops._dequant_linear_fallback = _dequant_linear_fallback
    bnbF.dequantize_4bit = dequantize_4bit


_CODE = {}


def _offset_tensor(offset, device):
    if isinstance(offset, torch.Tensor):
        return offset.to(device=device, dtype=torch.float32).reshape(1)
    return torch.tensor([float(offset)], dtype=torch.float32, device=device)


def _nf4_code(device):
    """bitsandbytes' NF4 code table (fp32, 16 values) on `device`."""
    if device not in _CODE:
        import bitsandbytes.functional as bnbF

        _CODE[device] = bnbF.get_4bit_type("nf4", device=device).float()
    return _CODE[device]
