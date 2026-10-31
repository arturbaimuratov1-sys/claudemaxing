"""Our LoRA branch for peft's 4-bit Linear against peft itself (fp16 autocast, dropout 0.05):
same dropout mask (same random numbers), values and gradients within fp16 rounding, smaller
backward state, and the pack/unpack kernels exact."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
pytest.importorskip("bitsandbytes")
pytest.importorskip("peft")

from solution import lora as L  # noqa: E402


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_pack_unpack_exact():
    torch.manual_seed(0)
    x16 = torch.randn(3, 37, 64, device="cuda").half()
    mask = torch.rand(3, 37, 64, device="cuda") > 0.3
    xd, bits = L._pack(x16, mask, 1.25)
    assert torch.equal(xd, torch.where(mask, (x16.float() * 1.25).half(), 0))
    g = torch.randn_like(x16)
    xd2, dx = L._unpack(x16, bits, g, 1.25, torch.float32)
    assert torch.equal(xd2, xd)
    assert torch.equal(dx, torch.where(mask, g.float() * 1.25, 0))


def model_pair():
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = bnb.nn.Linear4bit(256, 256, bias=False, compute_dtype=torch.float16,
                                            quant_type="nf4")
            self.v_proj = bnb.nn.Linear4bit(256, 64, bias=False, compute_dtype=torch.float16,
                                            quant_type="nf4")

        def forward(self, x):
            return self.q_proj(x).sum(-1) + self.v_proj(x).sum(-1)

    torch.manual_seed(0)
    block = Block()
    for m in (block.q_proj, block.v_proj):
        m.weight = bnb.nn.Params4bit((torch.randn(m.out_features, m.in_features) * 0.05).half(),
                                     requires_grad=False, quant_type="nf4")
    block = block.cuda()
    model = get_peft_model(block, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"]))
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = (torch.randn_like(p) * 0.1).float()  # non-zero B so every path matters
    model.train()
    return model


def run(model, x, seed):
    for p in model.parameters():
        p.grad = None
    xx = x.clone().requires_grad_(True)
    torch.manual_seed(seed)
    saved = []

    def pack(t):
        saved.append(t)
        return t

    with torch.autocast("cuda", dtype=torch.float16), \
            torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = model(xx)
    (out.float() * torch.linspace(-1, 1, out.shape[-1], device="cuda")).sum().backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    nbytes = sum(t.untyped_storage().nbytes() for t in
                 {t.untyped_storage().data_ptr(): t for t in saved}.values())
    return out.detach(), xx.grad, grads, nbytes


def test_matches_peft():
    from peft.tuners.lora import bnb as lora_bnb
    from peft.tuners.lora import layer as lora_layer

    model = model_pair()
    x = torch.randn(4, 50, 256, device="cuda")
    current = (lora_bnb.Linear4bit.forward, lora_layer.Linear.forward)
    originals = tuple(getattr(f, "_original", f) for f in current)
    lora_bnb.Linear4bit.forward, lora_layer.Linear.forward = originals  # peft's own forward
    ref = run(model, x, seed=7)
    L.patch_peft_lora_4bit()
    try:
        assert getattr(lora_bnb.Linear4bit.forward, "_lora_branch", False)
        calls = []
        real = L.LoRABranch.forward
        L.LoRABranch.forward = staticmethod(lambda ctx, *a: (calls.append(1), real(ctx, *a))[1])
        try:
            ours = run(model, x, seed=7)
        finally:
            L.LoRABranch.forward = staticmethod(real)
        assert len(calls) == 2
    finally:
        lora_bnb.Linear4bit.forward, lora_layer.Linear.forward = current
        if not getattr(current[1], "_lora_branch", False):
            lora_bnb.Linear4bit.forward, lora_layer.Linear.forward = originals
    assert ours[0].dtype == ref[0].dtype
    assert rel(ours[0], ref[0]) < 1e-3
    assert rel(ours[1], ref[1]) < 2e-3
    assert ref[2].keys() == ours[2].keys() and len(ref[2]) == 4
    for n in ref[2]:
        assert ours[2][n].dtype == ref[2][n].dtype
        assert rel(ours[2][n], ref[2][n]) < 5e-3, n
    assert ours[3] < ref[3], (ours[3], ref[3])


def test_same_mask_as_peft_dropout():
    """native_dropout draws the same mask as nn.Dropout for the same seed."""
    x = torch.randn(4, 50, 256, device="cuda")
    torch.manual_seed(3)
    a = torch.nn.functional.dropout(x, 0.05, True)
    torch.manual_seed(3)
    b, mask = torch.native_dropout(x, 0.05, True)
    assert torch.equal(a, b) and torch.equal(a != 0, mask | (x == 0) & mask)
