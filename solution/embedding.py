"""Keep the frozen, tied embedding / head weight of the 4-bit model in fp16 instead of fp32.

The model is loaded in fp16 and `prepare_model_for_kbit_training` upcasts the embedding (tied to
the head) to fp32: 151936 x 1536 x 4 bytes = 890 MiB, every value an exact fp16 number. Storing
it in fp16 halves that and changes no value:

* the embedding output is `weight[ids]` cast to fp32, the same fp32 numbers as before, so the
  residual stream, RoPE's dtype and everything downstream stay as they were;
* the fused head (lm_head_loss.py) already casts each vocabulary chunk of the weight to fp16
  before its matmul (fp16 autocast), so it reads the same fp16 numbers.

It is done only when every value is exactly representable in fp16 (checked chunk by chunk), the
weight is frozen and tied, and the module is a plain nn.Embedding; otherwise nothing changes.
The conversion goes through pinned host memory, so the GPU never holds both copies.
"""

import types

import torch
import torch.nn.functional as F

_DONE = "_fp16_tied_embedding"


def _fp16_embedding_forward(self, input):
    """nn.Embedding.forward on the fp16 weight, returning the fp32 values the base returns."""
    return F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type,
                       self.scale_grad_by_freq, self.sparse).to(torch.float32)


def _to_fp16_via_host(weight, rows=8192):
    """fp16 copy of the fp32 `weight` that never has both copies on the GPU: each chunk is
    checked (exactly an fp16 number?) and staged in pinned host memory, the fp32 storage is
    released, then the fp16 tensor is uploaded. None (weight untouched) if a value is not exact."""
    host = torch.empty(weight.shape, dtype=torch.float16, pin_memory=True)
    for r0 in range(0, weight.shape[0], rows):
        chunk = weight.data[r0:r0 + rows]
        low = chunk.half()
        if not torch.equal(low.float(), chunk):
            return None
        host[r0:r0 + rows].copy_(low)
    del chunk, low  # views of the fp32 storage: drop them, or it is not freed below
    device = weight.device
    weight.data = torch.empty(0, dtype=torch.float16, device=device)  # frees the fp32 storage
    weight.data = host.to(device)
    return True


def _autocast_fp16():
    try:
        return torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16
    except (TypeError, AttributeError):  # torch < 2.4
        return torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16


def shrink_tied_embedding(lm):
    """Store the tied fp32 embedding/head weight of `lm` (a *ForCausalLM) in fp16 if lossless.
    Called from the fused training step, which runs under fp16 autocast (the head is then only
    used through fp16 matmuls). Returns True when the weight is (now) stored in fp16 by us."""
    if getattr(lm, _DONE, None) is not None:
        return getattr(lm, _DONE)
    if not _autocast_fp16():
        return False  # decide on a later call, under the training autocast
    ok = False
    emb, head = lm.get_input_embeddings(), lm.get_output_embeddings()
    w = getattr(emb, "weight", None)
    if (type(emb) is torch.nn.Embedding and type(head) is torch.nn.Linear and w is not None
            and head.weight is w and w.dtype == torch.float32 and w.is_cuda
            and not w.requires_grad and emb.max_norm is None and not emb.sparse
            and not emb._forward_pre_hooks and "forward" not in emb.__dict__
            and not hasattr(emb, "_hf_hook")):
        with torch.no_grad():
            ok = _to_fp16_via_host(w) is not None  # one Parameter, shared with the head
        if ok:
            # bound method on the instance: module hooks (peft's) still run around it, and it
            # follows the module through deepcopy
            emb.forward = types.MethodType(_fp16_embedding_forward, emb)
    setattr(lm, _DONE, ok)
    return ok
