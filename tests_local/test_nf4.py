"""Our NF4 dequantization against bitsandbytes: bit-identical weights, and bit-identical 4-bit
matmuls (forward and input gradient) with the patch on and off."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
bnb = pytest.importorskip("bitsandbytes")

from solution import nf4  # noqa: E402


def quantized(n, k, nested, seed=0):
    torch.manual_seed(seed)
    w = (torch.randn(n, k, device="cuda") * 0.05).half()
    w[0, :5] = torch.tensor([0.0, -0.0, 1e-7, 3.0, -2.5])  # zeros, a tiny value, outliers
    packed, state = bnb.functional.quantize_4bit(w, blocksize=64, quant_type="nf4",
                                                 compress_statistics=nested)
    return packed, state


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("shape", [(1536, 1536), (256, 1536), (8960, 1536), (1536, 8960),
                                   (3, 64)])
def test_weight_bit_identical(nested, shape):
    packed, state = quantized(*shape, nested)
    ref = nf4.ORIGINALS.get("dequant", bnb.functional.dequantize_4bit)(packed, state)
    ours = nf4.dequantize_nf4(packed, state)
    assert ours.dtype == ref.dtype == torch.float16 and ours.shape == ref.shape
    assert torch.equal(ours.view(torch.int16), ref.view(torch.int16))
    ours32 = nf4.dequantize_nf4(packed, state, torch.float32)
    assert torch.equal(ours32.half(), ref)


def test_matmul_4bit_bit_identical_with_patch():
    """The real path: Linear4bit-style matmul_4bit forward (dequant + F.linear fallback for many
    rows) and the input gradient (functional.dequantize_4bit in backward)."""
    packed, state = quantized(8960, 1536, True)
    x = torch.randn(4, 340, 1536, device="cuda").half()
    g = torch.randn(4, 340, 8960, device="cuda").half()

    def run():
        xx = x.clone().requires_grad_(True)
        out = bnb.matmul_4bit(xx, packed.t(), quant_state=state)
        out.backward(g)
        return out.detach(), xx.grad

    import bitsandbytes.functional as bnbF
    from bitsandbytes.backends.cuda import ops as cuda_ops

    if nf4.ORIGINALS:  # compare against the unpatched functions
        saved = (cuda_ops._dequant_linear_fallback, bnbF.dequantize_4bit)
        cuda_ops._dequant_linear_fallback = nf4.ORIGINALS["fallback"]
        bnbF.dequantize_4bit = nf4.ORIGINALS["dequant"]
        ref = run()
        cuda_ops._dequant_linear_fallback, bnbF.dequantize_4bit = saved
    else:
        ref = run()
        nf4.patch_bnb_nf4_dequant()
    calls = []
    real = nf4.dequantize_nf4

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    nf4.dequantize_nf4 = spy
    try:
        ours = run()
    finally:
        nf4.dequantize_nf4 = real
    assert len(calls) >= 2, "forward and backward should both use our kernel"
    assert torch.equal(ours[0], ref[0])
    assert torch.equal(ours[1], ref[1])


def test_unsupported_falls_back():
    torch.manual_seed(0)
    w = torch.randn(128, 64, device="cuda").half()
    packed, state = bnb.functional.quantize_4bit(w, blocksize=64, quant_type="fp4")
    nf4.patch_bnb_nf4_dequant()
    out = bnb.functional.dequantize_4bit(packed, state)  # fp4: the original path
    ref = nf4.ORIGINALS["dequant"](packed, state)
    assert torch.equal(out, ref)


def test_no_fused_multiply_add_in_the_kernel():
    """Bit-exactness needs the absmax step as a separate mul and add (no FMA contraction)."""
    packed, state = quantized(256, 128, True)
    nf4.dequantize_nf4(packed, state)
    found = 0
    for dev_cache in nf4.nf4_dequant_kernel.device_caches.values():
        for compiled in dev_cache[0].values():
            ptx = compiled.asm["ptx"]
            assert "fma.rn.f32" not in ptx
            found += 1
    assert found


def test_fp32_output_matches_bitsandbytes_fp32():
    import bitsandbytes.functional as bnbF

    torch.manual_seed(2)
    w = torch.randn(512, 256, device="cuda") * 0.05
    packed, state = bnbF.quantize_4bit(w, blocksize=64, quant_type="nf4", compress_statistics=True)
    assert state.dtype == torch.float32
    ref = nf4.ORIGINALS.get("dequant", bnbF.dequantize_4bit)(packed, state)
    assert torch.equal(nf4.dequantize_nf4(packed, state), ref)
