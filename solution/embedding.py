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
"""

import torch
import torch.nn.functional as F

_DONE = "_fp16_tied_embedding"


def _exact_in_fp16(weight, rows=8192):
    for r0 in range(0, weight.shape[0], rows):
        chunk = weight[r0:r0 + rows]
        if not torch.equal(chunk.half().float(), chunk):
            return False
    return True


def shrink_tied_embedding(lm):
    """Store the tied fp32 embedding/head weight of `lm` (a *ForCausalLM) in fp16 if lossless.
    Returns True when the weight is (now) stored in fp16 by this function."""
    if getattr(lm, _DONE, None) is not None:
        return getattr(lm, _DONE)
    ok = False
    emb, head = lm.get_input_embeddings(), lm.get_output_embeddings()
    w = getattr(emb, "weight", None)
    if (type(emb) is torch.nn.Embedding and type(head) is torch.nn.Linear and w is not None
            and head.weight is w and w.dtype == torch.float32 and w.is_cuda
            and not w.requires_grad and emb.max_norm is None and not emb.sparse
            and not emb._forward_pre_hooks and _exact_in_fp16(w.detach())):
        with torch.no_grad():
            w.data = w.data.half()  # one Parameter, shared by the embedding and the head

        def forward(input, _emb=emb):
            return F.embedding(input, _emb.weight, _emb.padding_idx, _emb.max_norm,
                               _emb.norm_type, _emb.scale_grad_by_freq,
                               _emb.sparse).to(torch.float32)

        emb.forward = forward  # instance attribute: module hooks (peft's) still run around it
        ok = True
    setattr(lm, _DONE, ok)
    return ok
