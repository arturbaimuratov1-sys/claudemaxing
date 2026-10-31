"""The output head and everything the trainer reads from its logits, without the full logits.

In the base run the head builds a [batch, seq, vocab] logits tensor (vocab ~152K for Qwen2.5)
and five consumers walk it: the loss (an fp32 upcast copy and a log-softmax kept for
backward), accelerate's fp32 copy of the model output, TRL's entropy, TRL's token accuracy (a
contiguous copy of logits[:, :-1]) and Soup's per-row losses for rewind.jsonl. Here the head
runs `chunk_rows` positions at a time; for each chunk one Triton program per row makes two
passes over the vocabulary:

  pass 1: running max and sum of exp (log-sum-exp), the running sum of exp(z) * z (entropy),
          the argmax (accuracy) and the label's logit (loss);
  pass 2: the gradient of the loss with respect to that row's logits, written over the logits
          in place, which one matmul with the frozen head weight turns into the gradient of the
          hidden state.

The chunk is then dropped, so the head never holds more than one chunk of logits. The loss is
a scalar, so the hidden-state gradient is computed in forward and only scaled in backward.

The numbers follow the base path, measured in the installed libraries: under accelerate's fp16
autocast the head runs in fp16 even though the hidden state (after a final norm with an fp32
weight) and the tied head weight are fp32; the loss is the fp32 cross-entropy of those fp16
logits summed over the shifted labels and divided by num_items_in_batch (or averaged); the
metrics read the same fp16 values in fp32.

Two patches put it in place: Qwen2ForCausalLM.forward returns the loss and no logits when the
trainer asks for it, and TRL's SFTTrainer.compute_loss asks for it, logs entropy, token accuracy
and num_tokens by TRL's formulas, and hands Soup's rewind recorder the per-token losses through
the attribute Soup reads for steps without logits.
"""

import functools
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

IGNORE_INDEX = -100
# Pass 2 writes the per-row gradient scaled by this power of two (exact in any float format),
# so the small probabilities of a 152K vocabulary do not flush to zero in fp16; backward
# divides it out again together with the loss scale and the denominator.
GRAD_SCALE = 1024.0
CHUNK_BYTES = 256 * 2 ** 20  # one chunk of logits at most this big
_REQUEST_ATTR = "_fused_head_request"
_STEP_ATTR = "_fused_head_step"
try:  # the trainer attribute Soup's rewind recorder reads when a step has no logits
    from soup_cli.utils.stream_loss_chunk import STEP_ATTR as SOUP_STEP_ATTR
except Exception:  # noqa: BLE001 - Soup missing or reorganised: same name as in 0.77.0
    SOUP_STEP_ATTR = "_soup_chunked_step"


@triton.jit
def lm_head_xent_rows_kernel(logits_ptr, target_ptr, loss_ptr, entropy_ptr, correct_ptr,
                             n_cols, row_stride, grad_scale, BLOCK: tl.constexpr,
                             WRITE_GRAD: tl.constexpr):
    row = tl.program_id(0)
    row_ptr = logits_ptr + row.to(tl.int64) * row_stride
    target = tl.load(target_ptr + row).to(tl.int32)
    labelled = target != -100
    cols = tl.arange(0, BLOCK)

    run_max = float("-inf")
    run_sum = 0.0
    run_dot = 0.0
    best = 0
    for start in range(0, n_cols, BLOCK):
        idx = start + cols
        mask = idx < n_cols
        z = tl.load(row_ptr + idx, mask=mask, other=float("-inf")).to(tl.float32)
        tile_max = tl.max(z, axis=0)
        # First index of the maximum, as torch.argmax: a later tile wins only if strictly larger.
        best = tl.where(tile_max > run_max, tl.argmax(z, axis=0) + start, best)
        new_max = tl.maximum(run_max, tile_max)
        rescale = tl.exp(run_max - new_max)
        e = tl.exp(z - new_max)
        run_sum = run_sum * rescale + tl.sum(e, axis=0)
        run_dot = run_dot * rescale + tl.sum(tl.where(mask, e * z, 0.0), axis=0)
        run_max = new_max

    lse = run_max + tl.log(run_sum)
    target_logit = tl.load(row_ptr + tl.where(labelled, target, 0)).to(tl.float32)
    tl.store(loss_ptr + row, tl.where(labelled, lse - target_logit, 0.0))
    tl.store(entropy_ptr + row, lse - run_dot / run_sum)
    tl.store(correct_ptr + row, (labelled & (best == target)).to(tl.int32))

    if WRITE_GRAD:
        for start in range(0, n_cols, BLOCK):
            idx = start + cols
            mask = idx < n_cols
            z = tl.load(row_ptr + idx, mask=mask, other=0.0).to(tl.float32)
            prob = tl.exp(z - lse)
            grad = tl.where(idx == target, prob - 1.0, prob) * grad_scale
            grad = tl.where(labelled, grad, 0.0)
            tl.store(row_ptr + idx, grad.to(logits_ptr.dtype.element_ty), mask=mask)


def _launch_shape(n_cols):
    block = min(4096, triton.next_power_of_2(n_cols))
    return block, 8 if block >= 2048 else 4


def _rows(logits, target, token_loss, entropy, correct, write_grad):
    block, warps = _launch_shape(logits.shape[1])
    lm_head_xent_rows_kernel[(logits.shape[0],)](
        logits, target, token_loss, entropy, correct, logits.shape[1], logits.stride(0),
        GRAD_SCALE, BLOCK=block, WRITE_GRAD=write_grad, num_warps=warps)


def head_compute_dtype(hidden, weight):
    """The dtype `F.linear(hidden, weight)` computes in: the autocast dtype when autocast is
    on, else the common dtype; None when the two differ and nothing would cast them."""
    try:
        enabled, dtype = torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda")
    except (TypeError, AttributeError):  # torch < 2.4: no device argument
        enabled, dtype = torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype()
    if enabled:
        return dtype
    return weight.dtype if hidden.dtype == weight.dtype else None


def shift_labels_left(labels):
    """Position t is scored against token t + 1, the last position against nothing."""
    return torch.nn.functional.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]


class _FusedHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, target, denominator, compute_dtype, chunk_rows, want_grad):
        shape = hidden.shape
        h2 = hidden.reshape(-1, shape[-1])
        tgt = target.reshape(-1).contiguous()
        n, vocab = h2.shape[0], weight.shape[0]
        w = weight.detach().to(compute_dtype)
        token_loss = torch.empty(n, dtype=torch.float32, device=hidden.device)
        entropy = torch.empty(n, dtype=torch.float32, device=hidden.device)
        correct = torch.empty(n, dtype=torch.int32, device=hidden.device)
        grad_raw = torch.empty(n, shape[-1], dtype=torch.float32, device=hidden.device) \
            if want_grad else None
        if chunk_rows is None:
            chunk_rows = max(1, CHUNK_BYTES // (vocab * w.element_size()))
        for start in range(0, n, chunk_rows):
            stop = min(n, start + chunk_rows)
            logits = torch.mm(h2[start:stop].to(compute_dtype), w.t())
            _rows(logits, tgt[start:stop], token_loss[start:stop], entropy[start:stop],
                  correct[start:stop], want_grad)
            if want_grad:  # the logits now hold GRAD_SCALE * d(sum of CE)/d(logits)
                grad_raw[start:stop] = torch.mm(logits, w)
            del logits
        del w
        loss = token_loss.sum() / denominator
        ctx.shape, ctx.hidden_dtype, ctx.denominator = shape, hidden.dtype, denominator
        if want_grad:
            ctx.save_for_backward(grad_raw)
        ctx.want_grad = want_grad
        token_loss, entropy, correct = token_loss.view(shape[:-1]), entropy.view(shape[:-1]), \
            correct.view(shape[:-1])
        ctx.mark_non_differentiable(token_loss, entropy, correct)
        return loss, token_loss, entropy, correct

    @staticmethod
    def backward(ctx, grad_loss, _token_loss, _entropy, _correct):
        if not ctx.want_grad:
            return (None,) * 7
        (grad_raw,) = ctx.saved_tensors
        scale = grad_loss.float() / (ctx.denominator * GRAD_SCALE)
        grad = (grad_raw * scale).to(ctx.hidden_dtype).view(ctx.shape)
        return grad, None, None, None, None, None, None


def fused_head_loss(hidden, weight, labels, num_items_in_batch=None, chunk_rows=None):
    """ForCausalLMLoss(linear(hidden, weight), labels) and the per-token terms the trainer's
    metrics need, without materialising the logits. `weight` must be frozen: the gradient
    goes to `hidden` only. Returns (loss, step) with step.token_loss and step.token_entropy of
    shape [batch, seq] (fp32), step.correct and step.labelled (counts)."""
    target = shift_labels_left(labels).to(hidden.device)
    labelled = (target != IGNORE_INDEX).sum()
    if num_items_in_batch is None:
        denominator = labelled
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(hidden.device)
    else:
        denominator = num_items_in_batch
    compute_dtype = head_compute_dtype(hidden, weight)
    if compute_dtype is None:
        raise TypeError(f"hidden {hidden.dtype} and head {weight.dtype} differ outside autocast")
    want_grad = torch.is_grad_enabled() and hidden.requires_grad
    loss, token_loss, entropy, correct = _FusedHeadLoss.apply(
        hidden, weight, target, denominator, compute_dtype, chunk_rows, want_grad)
    step = SimpleNamespace(token_loss=token_loss, token_entropy=entropy,
                           correct=correct.sum(), labelled=labelled)
    return loss, step


@torch.no_grad()
def head_metrics_from_logits(logits, labels):
    """The same per-token terms from logits that were built anyway (fallback path)."""
    target = shift_labels_left(labels).to(logits.device).reshape(-1).contiguous()
    flat = logits.reshape(-1, logits.shape[-1]).contiguous()
    n = flat.shape[0]
    token_loss = torch.empty(n, dtype=torch.float32, device=logits.device)
    entropy = torch.empty_like(token_loss)
    correct = torch.empty(n, dtype=torch.int32, device=logits.device)
    _rows(flat, target, token_loss, entropy, correct, False)
    shape = logits.shape[:-1]
    return SimpleNamespace(token_loss=token_loss.view(shape), token_entropy=entropy.view(shape),
                           correct=correct.sum(), labelled=(target != IGNORE_INDEX).sum())


def patch_qwen2_causal_lm_head():
    """Qwen2ForCausalLM.forward: when the trainer has asked for it (and only then), build the
    loss from the hidden states with the fused head and return no logits."""
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    cls = getattr(modeling_qwen2, "Qwen2ForCausalLM", None)
    if cls is None or getattr(cls.forward, "_fused_head", False):
        return
    original = cls.forward

    # wraps: the trainer reads the forward's signature (label names, loss kwargs).
    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        asked = self.__dict__.pop(_REQUEST_ATTR, False)
        labels = kwargs.get("labels")
        keep = kwargs.get("logits_to_keep", 0)
        if (not asked or args or labels is None or not isinstance(keep, int) or keep != 0
                or "shift_labels" in kwargs or kwargs.get("return_dict") is False):
            return original(self, *args, **kwargs)
        kwargs.pop("labels")
        kwargs.pop("logits_to_keep", None)
        outputs = self.model(**kwargs)
        hidden = outputs.last_hidden_state
        head = self.lm_head
        if (type(head) is not torch.nn.Linear or head.bias is not None
                or head.weight.requires_grad
                or head_compute_dtype(hidden, head.weight) is None):
            logits = head(hidden)  # what the original forward does from here on
            loss = self.loss_function(logits=logits, labels=labels,
                                      vocab_size=self.config.vocab_size, **kwargs)
            return CausalLMOutputWithPast(loss=loss, logits=logits,
                                          past_key_values=outputs.past_key_values,
                                          hidden_states=outputs.hidden_states,
                                          attentions=outputs.attentions)
        loss, step = fused_head_loss(hidden, head.weight, labels,
                                     num_items_in_batch=kwargs.get("num_items_in_batch"))
        self.__dict__[_STEP_ATTR] = step
        return CausalLMOutputWithPast(loss=loss, logits=None,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    forward._fused_head = True
    cls.forward = forward


def _causal_lm(model):
    while isinstance(getattr(model, "module", None), torch.nn.Module):
        model = model.module
    get_base = getattr(model, "get_base_model", None)
    return get_base() if callable(get_base) else model


def _fused_target(trainer, model, inputs):
    """The causal LM to run with the fused head, or None to keep TRL's own path."""
    if (not isinstance(model, torch.nn.Module) or not getattr(trainer.model, "training", False)
            or not isinstance(inputs, dict) or "labels" not in inputs
            or "shift_labels" in inputs
            or getattr(trainer.args, "use_liger_kernel", False)
            or getattr(trainer, "num_virtual_tokens", 0)
            or getattr(trainer, "aux_loss_enabled", False)
            or getattr(trainer, "compute_loss_func", None) is not None
            or getattr(trainer, "label_smoother", None) is not None
            or not hasattr(trainer, "_metrics")):
        return None
    lm = _causal_lm(model)
    if not getattr(type(lm).forward, "_fused_head", False):
        return None
    head = lm.get_output_embeddings()
    if (type(head) is not torch.nn.Linear or head.bias is not None or head.weight.requires_grad
            or not head.weight.is_cuda
            or getattr(lm.config, "final_logit_softcapping", None) is not None):
        return None
    return lm


def _log_trl_metrics(trainer, mode, inputs, step):
    """entropy, num_tokens and mean_token_accuracy exactly as SFTTrainer.compute_loss logs
    them from the logits (same masks, same gathers, same order of reductions)."""
    gather = trainer.accelerator.gather_for_metrics
    if "attention_mask" in inputs:
        mask = inputs["attention_mask"]
        entropy = torch.sum(step.token_entropy * mask) / mask.sum()
    elif "position_ids" in inputs:
        entropy = torch.mean(step.token_entropy)
    else:
        raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
    trainer._metrics[mode]["entropy"].append(gather(entropy).mean().item())
    if mode == "train":
        if "attention_mask" in inputs:
            num_tokens = gather(inputs["attention_mask"].sum()).sum().item()
        else:
            local = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
            num_tokens = gather(local).sum().item()
        trainer._total_train_tokens += num_tokens
    trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    correct, total = gather(step.correct), gather(step.labelled)
    total_sum = total.sum()
    accuracy = (correct.sum() / total_sum).item() if total_sum > 0 else 0.0
    trainer._metrics[mode]["mean_token_accuracy"].append(accuracy)


def patch_sft_compute_loss():
    """TRL's SFTTrainer.compute_loss, for a training step of a patched causal LM: ask the model
    for the fused head, log TRL's metrics from the per-token terms and leave the per-token
    losses where Soup's rewind recorder looks for them. Anything else keeps TRL's path."""
    try:
        from trl.trainer import sft_trainer as trl_sft
    except Exception:  # noqa: BLE001 - no TRL, nothing to patch
        return
    cls = trl_sft.SFTTrainer
    if getattr(cls.compute_loss, "_fused_head", False):
        return
    original = cls.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        lm = _fused_target(self, model, inputs)
        if lm is None:
            setattr(self, SOUP_STEP_ATTR, None)  # never leave an earlier step's losses behind
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        inputs.pop("_prediction_loss_only", None)
        labels = inputs["labels"]
        inputs["use_cache"] = False
        setattr(self, SOUP_STEP_ATTR, None)
        lm.__dict__[_REQUEST_ATTR] = True
        try:
            loss, outputs = super(cls, self).compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        finally:
            lm.__dict__.pop(_REQUEST_ATTR, None)
        step = lm.__dict__.pop(_STEP_ATTR, None)
        if step is None:  # the forward fell back and built the logits after all
            step = head_metrics_from_logits(outputs.logits, labels)
        _log_trl_metrics(self, "train", inputs, step)
        setattr(self, SOUP_STEP_ATTR, SimpleNamespace(token_loss=step.token_loss))
        return (loss, outputs) if return_outputs else loss

    compute_loss._fused_head = True
    cls.compute_loss = compute_loss


def patch_fused_head_loss():
    patch_qwen2_causal_lm_head()
    patch_sft_compute_loss()
