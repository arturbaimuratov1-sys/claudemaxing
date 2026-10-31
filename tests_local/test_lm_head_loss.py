"""The fused output head + loss against what the base run computes from full logits.

Run on a GPU from the repository root: `python -m pytest tests_local -q`.

The base path, as it runs inside the trainer (accelerate wraps the model's forward in fp16
autocast and converts its outputs to fp32):
- logits = lm_head(hidden) under autocast, so in fp16 even with an fp32 hidden state and an
  fp32 (tied) head weight;
- loss = ForCausalLMLoss: logits upcast to fp32, labels shifted, cross-entropy summed and
  divided by num_items_in_batch (or averaged when there is none);
- entropy (TRL entropy_from_logits), token accuracy (argmax of logits[:, :-1] against
  labels[:, 1:]) and Soup's per-row losses (rewind row_losses) read the fp32 copy;
- the gradient reaches the hidden state through the fp16 head.
The reference below rebuilds that from the installed libraries where they are available.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

F16, F32 = torch.float16, torch.float32


def trl_entropy(logits):
    try:
        from trl.trainer.utils import entropy_from_logits
    except ImportError:
        logps = F.log_softmax(logits, dim=-1)
        return -(torch.exp(logps) * logps).sum(-1)
    return entropy_from_logits(logits)


def soup_row_losses(logits, labels):
    try:
        from soup_cli.trainer.rewind_hf import row_losses
    except ImportError:
        row_losses = None
    if row_losses is not None:
        return row_losses(logits, labels)
    shift = labels[:, 1:]
    ce = F.cross_entropy(logits[:, :-1].transpose(1, 2), shift, reduction="none",
                         ignore_index=-100).float()
    mask = (shift != -100).float()
    tokens = mask.sum(1)
    return (ce * mask).sum(1).div(tokens.clamp(min=1)).tolist(), tokens.int().tolist()


def make(batch, seq, hidden, vocab, h_dtype, w_dtype, seed=0, drop=0.3):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn(batch, seq, hidden, device="cuda", generator=gen).to(h_dtype)
    w = (torch.randn(vocab, hidden, device="cuda", generator=gen) * hidden ** -0.5).to(w_dtype)
    labels = torch.randint(0, vocab, (batch, seq), device="cuda", generator=gen)
    # Prompt tokens and padding are -100, as the collator leaves them.
    labels[torch.rand(batch, seq, device="cuda", generator=gen) < drop] = -100
    labels[:, :3] = -100
    return h, w, labels


def reference(h, w, labels, num_items, autocast, grad_scale):
    """Loss, per-token entropy [B,T], (correct, labelled), per-row losses, dL/dh."""
    h = h.detach().clone().requires_grad_()
    ctx = torch.autocast("cuda", dtype=F16) if autocast else torch.autocast("cuda", enabled=False)
    with ctx:
        logits = F.linear(h, w)
        shift = F.pad(labels, (0, 1), value=-100)[..., 1:]
        flat = logits.float().view(-1, logits.shape[-1])
        reduction = "sum" if num_items is not None else "mean"
        loss = F.cross_entropy(flat, shift.reshape(-1), ignore_index=-100, reduction=reduction)
        if num_items is not None:
            loss = loss / num_items
    (loss * grad_scale).backward()
    with torch.no_grad():
        full = logits.detach().float()  # accelerate's fp32 copy of the output
        entropy = trl_entropy(full)
        pred = full[:, :-1].argmax(-1)
        mask = labels[:, 1:] != -100
        correct = ((pred == labels[:, 1:]) & mask).sum().item()
        rows, tokens = soup_row_losses(full, labels)
    return loss.detach(), entropy, (correct, mask.sum().item()), (rows, tokens), h.grad


def ours(h, w, labels, num_items, autocast, grad_scale, chunk_cols=None):
    from solution.lm_head_loss import fused_head_loss

    h = h.detach().clone().requires_grad_()
    ctx = torch.autocast("cuda", dtype=F16) if autocast else torch.autocast("cuda", enabled=False)
    with ctx:
        loss, step = fused_head_loss(h, w, labels, num_items_in_batch=num_items,
                                     chunk_cols=chunk_cols)
    (loss * grad_scale).backward()
    from soup_cli.trainer.rewind_hf import row_losses_from_token_losses

    rows, tokens = row_losses_from_token_losses(step.token_loss, labels)
    return (loss.detach(), step.token_entropy, (step.correct.item(), step.labelled.item()),
            (rows, tokens), h.grad)


def close(actual, expected, tol, what):
    scale = expected.float().abs().max().clamp(min=1e-6).item()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=tol, atol=tol * scale,
                               msg=lambda m: f"{what}: {m}")


CASES = [
    # (batch, seq, hidden, vocab, hidden dtype, weight dtype, autocast) -- the T4 case first:
    # an fp32 hidden state (the final norm has an fp32 weight) and an fp32 tied head, autocast.
    (2, 96, 256, 151936, F32, F32, True),
    (4, 64, 128, 1000, F32, F32, True),
    (3, 50, 64, 37, F16, F16, False),
    (2, 40, 64, 5000, F32, F32, False),
]


@pytest.mark.parametrize("num_items", [None, "count"])
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"V{c[3]}-{c[4]}-{c[5]}-ac{c[6]}")
def test_matches_the_base_path(case, num_items):
    pytest.importorskip("soup_cli")
    batch, seq, hidden, vocab, h_dtype, w_dtype, autocast = case
    h, w, labels = make(batch, seq, hidden, vocab, h_dtype, w_dtype)
    shift = F.pad(labels, (0, 1), value=-100)[..., 1:]
    n_items = None if num_items is None else (shift != -100).sum()
    scale = 1024.0 if autocast else 1.0  # a loss scale, as GradScaler applies it

    r_loss, r_ent, r_acc, r_rows, r_grad = reference(h, w, labels, n_items, autocast, scale)
    o_loss, o_ent, o_acc, o_rows, o_grad = ours(h, w, labels, n_items, autocast, scale)

    low = F16 in (h_dtype, w_dtype) or autocast
    close(o_loss, r_loss, 1e-4, "loss")
    close(o_ent, r_ent, 1e-3 if low else 1e-4, "entropy")
    assert o_acc[1] == r_acc[1]
    assert abs(o_acc[0] - r_acc[0]) <= max(1, 0.01 * r_acc[1]), (o_acc, r_acc)
    assert o_rows[1] == r_rows[1]
    close(torch.tensor(o_rows[0]), torch.tensor(r_rows[0]), 1e-3, "row losses")
    assert o_grad.dtype == r_grad.dtype == h_dtype
    close(o_grad, r_grad, 1e-2 if low else 1e-4, "grad of hidden")


@pytest.mark.parametrize("scale", [65536.0, 2.0 ** 24])
def test_grad_scaler_scales_match_the_base(scale):
    """fp16 rounding of the head gradient at GradScaler's scales (65536 is its initial scale;
    2**24 is far above what training reaches): the hidden-state gradient agrees with the base
    path and is finite exactly when the base one is, so the scaler skips the same steps."""
    pytest.importorskip("soup_cli")
    h, w, labels = make(4, 256, 1536, 151936, F32, F32, seed=3)
    shift = F.pad(labels, (0, 1), value=-100)[..., 1:]
    n_items = (shift != -100).sum()
    r = reference(h, w, labels, n_items, True, scale)
    o = ours(h, w, labels, n_items, True, scale)
    assert torch.isfinite(r[4]).all().item() == torch.isfinite(o[4]).all().item()
    if torch.isfinite(r[4]).all():
        close(o[4], r[4], 1e-2, "grad of hidden at loss scale %g" % scale)


@pytest.mark.parametrize("sharpness", [6.0, 12.0, 24.0])
def test_confident_labels_match_the_base(sharpness):
    """Late in training most labels carry p close to 1, where the gradient (p - 1) is small
    and fp16 rounding matters most. Labels here are the argmax of a sharp distribution."""
    pytest.importorskip("soup_cli")
    h, w, labels = make(2, 128, 256, 151936, F32, F32, seed=5, drop=0.1)
    w = w * sharpness
    with torch.no_grad(), torch.autocast("cuda", dtype=F16):
        pred = F.linear(h, w).float().argmax(-1)
    labels = torch.where(labels == -100, labels, F.pad(pred, (1, 0))[..., :-1])
    with torch.no_grad(), torch.autocast("cuda", dtype=F16):
        p = F.linear(h, w).float().softmax(-1).max(-1).values
    print(f"\nsharpness {sharpness}: median p of the argmax {p.median().item():.4f}")
    shift = F.pad(labels, (0, 1), value=-100)[..., 1:]
    n_items = (shift != -100).sum()
    r = reference(h, w, labels, n_items, True, 65536.0)
    o = ours(h, w, labels, n_items, True, 65536.0)
    close(o[0], r[0], 1e-3, "loss")
    close(o[4], r[4], 1e-2, "grad of hidden, confident labels")


def test_chunking_does_not_change_the_result():
    pytest.importorskip("soup_cli")
    h, w, labels = make(3, 70, 128, 3001, F32, F32)
    a = ours(h, w, labels, None, True, 1.0, chunk_cols=None)
    b = ours(h, w, labels, None, True, 1.0, chunk_cols=7)
    close(b[0], a[0], 1e-5, "loss")
    close(b[1], a[1], 1e-5, "entropy")
    close(b[4], a[4], 1e-3, "grad")


def test_rows_without_labels_give_zero_loss_and_zero_grad():
    from solution.lm_head_loss import fused_head_loss

    h, w, labels = make(2, 16, 64, 500, F32, F32)
    labels[0] = -100
    h = h.requires_grad_()
    with torch.autocast("cuda", dtype=F16):
        loss, step = fused_head_loss(h, w, labels)
    loss.backward()
    assert torch.all(step.token_loss[0] == 0)
    assert torch.all(h.grad[0] == 0)
    assert torch.all(step.token_entropy[0] > 0)  # entropy is over every position


def test_no_grad_forward_keeps_no_gradient_buffer():
    from solution.lm_head_loss import fused_head_loss

    h, w, labels = make(2, 16, 64, 500, F32, F32)
    with torch.no_grad(), torch.autocast("cuda", dtype=F16):
        loss, step = fused_head_loss(h, w, labels)
    assert not loss.requires_grad
    assert torch.isfinite(loss)


def test_logits_dtype_matches_base_rule():
    """The head runs in the autocast dtype when autocast is on, else in the weight's dtype."""
    from solution.lm_head_loss import head_compute_dtype

    w = torch.zeros(4, 4, device="cuda")
    with torch.autocast("cuda", dtype=F16):
        assert head_compute_dtype(torch.zeros(1, device="cuda"), w) == F16
    assert head_compute_dtype(torch.zeros(1, device="cuda"), w) == F32
    assert head_compute_dtype(torch.zeros(1, device="cuda", dtype=F16), w) is None
