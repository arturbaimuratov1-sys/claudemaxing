"""Causal attention of Qwen2 (grouped-query, right padding) in Triton, fp16 tiles, fp32 sums.

Why: on a T4 the base run hands SDPA fp32 q/k/v (the 4-bit projections return the dtype of
their fp32 input), expands k/v to all query heads (repeat_kv, because a padding mask is
passed), and the fp32 kernel keeps all of that for backward: about 43 KB per token per layer,
the largest item of the activation memory. Here:

* k/v are never expanded: a query head h reads kv head h // group.
* q/k/v are rounded to fp16 once (the o_proj after attention rounds the output to fp16 anyway);
  products are summed in fp32 (tl.dot), softmax runs in fp32 with exp2.
* Backward keeps fp16 q, k, v, the fp16 output and one fp32 log-sum-exp per row: about 7 KB per
  token per layer. The probabilities are recomputed from q, k and the log-sum-exp.
* The output is fp16 under fp16 autocast. The base output is fp32, but its only consumer is
  o_proj, which casts it to fp16 first, so o_proj sees the same values; its gradient arrives
  in fp16, as the base's does before the cast.
* The mask must be "causal and key is not padding" (what transformers builds for right padded
  batches); anything else goes to the original SDPA path.
"""

import weakref

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634


@triton.jit(do_not_specialize=["T"])  # T changes with every batch
def qwen_attn_fwd_kernel(Q, K, V, KOK, O, LSE, sqb, sqh, sqt, skb, skh, skt, sob, soh, sot,
                         T, H, GROUP, qk_scale, HAS_MASK: tl.constexpr, BM: tl.constexpr,
                         BN: tl.constexpr, D: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    kh = h // GROUP
    rows = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    q = tl.load(Q + b * sqb + h * sqh + rows[:, None] * sqt + offs_d[None, :],
                mask=rows[:, None] < T, other=0.0).to(tl.float16)
    m_i = tl.full([BM], -1.0e30, tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    k_base = K + b * skb + kh * skh
    v_base = V + b * skb + kh * skh
    hi = tl.minimum((pid_m + 1) * BM, T)
    for start in range(0, hi, BN):
        cols = start + offs_n
        kt = tl.load(k_base + cols[None, :] * skt + offs_d[:, None], mask=cols[None, :] < T,
                     other=0.0).to(tl.float16)
        s = tl.dot(q, kt) * qk_scale
        ok = (cols[None, :] <= rows[:, None]) & (cols[None, :] < T)
        if HAS_MASK:
            kok = tl.load(KOK + b * T + cols, mask=cols < T, other=0)
            ok = ok & (kok[None, :] != 0)
        s = tl.where(ok, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_base + cols[:, None] * skt + offs_d[None, :], mask=cols[:, None] < T,
                    other=0.0).to(tl.float16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
        m_i = m_new
    has = l_i > 0.0
    l_safe = tl.where(has, l_i, 1.0)
    out = acc / l_safe[:, None]
    tl.store(O + b * sob + h * soh + rows[:, None] * sot + offs_d[None, :],
             out.to(O.dtype.element_ty), mask=rows[:, None] < T)
    # log2-sum-exp2 of the scaled scores; +inf for a row with no visible key (p = 0 in backward)
    tl.store(LSE + bh * T + rows, tl.where(has, m_i + tl.log2(l_safe), float("inf")),
             mask=rows < T)


@triton.jit(do_not_specialize=["T"])  # T changes with every batch
def qwen_attn_bwd_dkdv_kernel(Q, K, V, DO, KOK, LSE, DELTA, DK, DV, sqb, sqh, sqt, skb, skh,
                              skt, sob, soh, sot, sdb, sdh, sdt, T, H, KVH, GROUP, SPLIT,
                              qk_scale, sm_scale, HAS_MASK: tl.constexpr, BM: tl.constexpr,
                              BN: tl.constexpr, D: tl.constexpr):
    # One program: BN keys of kv head kh and GROUP // SPLIT of its query heads. With SPLIT > 1
    # DK/DV are fp32 partial sums [B, KVH * SPLIT, T, D], added up over the split afterwards.
    pid_n = tl.program_id(0)
    bks = tl.program_id(1)
    b = bks // (KVH * SPLIT)
    slot = bks % (KVH * SPLIT)
    kh = slot // SPLIT
    g_per = GROUP // SPLIT
    g_lo = (slot % SPLIT) * g_per
    cols = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    kv_off = b * skb + kh * skh + cols[:, None] * skt + offs_d[None, :]
    k = tl.load(K + kv_off, mask=cols[:, None] < T, other=0.0).to(tl.float16)
    v = tl.load(V + kv_off, mask=cols[:, None] < T, other=0.0).to(tl.float16)
    col_ok = cols < T
    if HAS_MASK:
        kok = tl.load(KOK + b * T + cols, mask=col_ok, other=0)
        col_ok = col_ok & (kok != 0)
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    lo = (pid_n * BN // BM) * BM
    for g in range(g_lo, g_lo + g_per):
        h = kh * GROUP + g
        q_base = Q + b * sqb + h * sqh
        do_base = DO + b * sob + h * soh
        for start in range(lo, T, BM):
            rows = start + offs_m
            q = tl.load(q_base + rows[:, None] * sqt + offs_d[None, :], mask=rows[:, None] < T,
                        other=0.0).to(tl.float16)
            do = tl.load(do_base + rows[:, None] * sot + offs_d[None, :],
                         mask=rows[:, None] < T, other=0.0).to(tl.float16)
            lse = tl.load(LSE + (b * H + h) * T + rows, mask=rows < T, other=float("inf"))
            delta = tl.load(DELTA + (b * H + h) * T + rows, mask=rows < T, other=0.0)
            st = tl.dot(k, tl.trans(q)) * qk_scale  # [BN, BM]
            ok = (cols[:, None] <= rows[None, :]) & col_ok[:, None] & (rows[None, :] < T)
            pt = tl.where(ok, tl.exp2(st - lse[None, :]), 0.0)
            dv += tl.dot(pt.to(tl.float16), do)
            dpt = tl.dot(v, tl.trans(do))
            dst = pt * (dpt - delta[None, :])
            dk += tl.dot(dst.to(tl.float16), q)
    out_off = b * sdb + slot * sdh + cols[:, None] * sdt + offs_d[None, :]
    tl.store(DK + out_off, (dk * sm_scale).to(DK.dtype.element_ty), mask=cols[:, None] < T)
    tl.store(DV + out_off, dv.to(DV.dtype.element_ty), mask=cols[:, None] < T)


@triton.jit(do_not_specialize=["T"])  # T changes with every batch
def qwen_attn_bwd_dq_kernel(Q, K, V, DO, KOK, LSE, DELTA, DQ, sqb, sqh, sqt, skb, skh, skt,
                            sob, soh, sot, T, H, GROUP, qk_scale, sm_scale,
                            HAS_MASK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                            D: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    kh = h // GROUP
    rows = pid_m * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    q_off = b * sqb + h * sqh + rows[:, None] * sqt + offs_d[None, :]
    q = tl.load(Q + q_off, mask=rows[:, None] < T, other=0.0).to(tl.float16)
    do = tl.load(DO + b * sob + h * soh + rows[:, None] * sot + offs_d[None, :],
                 mask=rows[:, None] < T, other=0.0).to(tl.float16)
    lse = tl.load(LSE + bh * T + rows, mask=rows < T, other=float("inf"))
    delta = tl.load(DELTA + bh * T + rows, mask=rows < T, other=0.0)
    dq = tl.zeros([BM, D], tl.float32)
    k_base = K + b * skb + kh * skh
    v_base = V + b * skb + kh * skh
    hi = tl.minimum((pid_m + 1) * BM, T)
    for start in range(0, hi, BN):
        cols = start + offs_n
        kv_off = cols[:, None] * skt + offs_d[None, :]
        k = tl.load(k_base + kv_off, mask=cols[:, None] < T, other=0.0).to(tl.float16)
        v = tl.load(v_base + kv_off, mask=cols[:, None] < T, other=0.0).to(tl.float16)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        ok = (cols[None, :] <= rows[:, None]) & (cols[None, :] < T)
        if HAS_MASK:
            kok = tl.load(KOK + b * T + cols, mask=cols < T, other=0)
            ok = ok & (kok[None, :] != 0)
        p = tl.where(ok, tl.exp2(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(tl.float16), k)
    tl.store(DQ + q_off, (dq * sm_scale).to(DQ.dtype.element_ty), mask=rows[:, None] < T)


# (BM, BN, warps) per kernel, sized for the T4 (64 KB of shared memory per block, fp16 tiles of
# head_dim 128, one pipeline stage). If a GPU refuses a size, it is halved once and remembered.
_BLOCKS = {"fwd": [64, 64, 4], "dkdv": [32, 64, 8], "dq": [64, 32, 8]}
DKDV_SPLIT = 3  # dk/dv programs per kv head (query heads of a group split between them)


def _launch(name, kernel, grid, args, **meta):
    from triton.runtime.errors import OutOfResources

    while True:
        bm, bn, warps = _BLOCKS[name]
        try:
            kernel[grid(bm, bn)](*args, BM=bm, BN=bn, num_warps=warps, num_stages=1, **meta)
            return
        except OutOfResources:
            if min(bm, bn) <= 16:
                raise
            _BLOCKS[name] = [max(16, bm // 2), max(16, bn // 2), warps]


def _strides(x):
    return x.stride(0), x.stride(1), x.stride(2)


class QwenAttention(torch.autograd.Function):
    """q [B, H, T, D], k/v [B, KVH, T, D] (any strides), key_ok None or
    int8 [B, T]. Returns [B, T, H, D] in `out_dtype`."""

    @staticmethod
    def forward(ctx, q, k, v, key_ok, scale, out_dtype):
        B, H, T, D = q.shape
        KVH = k.shape[1]
        q16, k16, v16 = (torch.empty(x.shape, dtype=torch.float16, device=x.device).copy_(x)
                         for x in (q, k, v))  # contiguous fp16 copies, kept for backward
        out = torch.empty((B, T, H, D), dtype=torch.float16, device=q.device)
        lse = torch.empty((B, H, T), dtype=torch.float32, device=q.device)
        has_mask = key_ok is not None
        kok = key_ok if has_mask else lse  # unused pointer when there is no mask
        o_view = out.transpose(1, 2)
        qk_scale = scale * LOG2E
        _launch("fwd", qwen_attn_fwd_kernel, lambda bm, bn: (triton.cdiv(T, bm), B * H),
                (q16, k16, v16, kok, o_view, lse, *_strides(q16), *_strides(k16),
                 *_strides(o_view), T, H, H // KVH, qk_scale), HAS_MASK=has_mask, D=D)
        ctx.save_for_backward(q16, k16, v16, out, lse, key_ok)
        ctx.scale, ctx.dtypes = scale, (q.dtype, k.dtype, v.dtype)
        return out if out_dtype == torch.float16 else out.to(out_dtype)

    @staticmethod
    def backward(ctx, grad):
        q16, k16, v16, out, lse, key_ok = ctx.saved_tensors
        B, H, T, D = q16.shape
        KVH = k16.shape[1]
        do = grad.to(torch.float16).contiguous()  # [B, T, H, D]
        delta = (do.float() * out.float()).sum(-1).transpose(1, 2).contiguous()  # [B, H, T]
        do_view = do.transpose(1, 2)
        has_mask = key_ok is not None
        kok = key_ok if has_mask else lse
        q_dt, k_dt, v_dt = ctx.dtypes
        dq = torch.empty(q16.shape, dtype=q_dt, device=q16.device)
        split = DKDV_SPLIT if (H // KVH) % DKDV_SPLIT == 0 else 1
        if split == 1:
            dk = torch.empty(k16.shape, dtype=k_dt, device=q16.device)
            dv = torch.empty(k16.shape, dtype=v_dt, device=q16.device)
        else:  # fp32 partial sums over parts of each group, added below
            dk = torch.empty((B, KVH * split, T, D), dtype=torch.float32, device=q16.device)
            dv = torch.empty_like(dk)
        qk_scale = ctx.scale * LOG2E
        _launch("dkdv", qwen_attn_bwd_dkdv_kernel,
                lambda bm, bn: (triton.cdiv(T, bn), B * KVH * split),
                (q16, k16, v16, do_view, kok, lse, delta, dk, dv, *_strides(q16),
                 *_strides(k16), *_strides(do_view), *_strides(dk), T, H, KVH, H // KVH, split,
                 qk_scale, ctx.scale),
                HAS_MASK=has_mask, D=D)
        if split > 1:
            dk = dk.view(B, KVH, split, T, D).sum(2).to(k_dt)
            dv = dv.view(B, KVH, split, T, D).sum(2).to(v_dt)
        _launch("dq", qwen_attn_bwd_dq_kernel, lambda bm, bn: (triton.cdiv(T, bm), B * H),
                (q16, k16, v16, do_view, kok, lse, delta, dq, *_strides(q16), *_strides(k16),
                 *_strides(do_view), T, H, H // KVH, qk_scale, ctx.scale),
                HAS_MASK=has_mask, D=D)
        return dq, dk, dv, None, None, None


def qwen_attention(q, k, v, key_ok=None, scale=None, out_dtype=None):
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    return QwenAttention.apply(q, k, v, key_ok, float(scale), out_dtype or q.dtype)


_MASK_CACHE = [None, None]  # (weakref to the last mask, its key_ok or False)


def _key_ok(mask, T, B=None):
    """int8 [B, T] key validity if `mask` [B, 1, T, T] (bool) is exactly causal & key_ok, else
    False. Cached for the mask object, which transformers passes to every layer."""
    B = mask.shape[0] if B is None else B
    if mask.dtype != torch.bool or tuple(mask.shape) != (B, 1, T, T):
        return False
    ref = _MASK_CACHE[0]
    if ref is not None and ref() is mask:
        return _MASK_CACHE[1]
    result = False
    if True:
        m = mask[:, 0]
        kok = m[:, -1, :]
        causal = torch.ones((T, T), dtype=torch.bool, device=mask.device).tril_()
        if torch.equal(m, causal[None] & kok[:, None, :]):
            result = kok.to(torch.int8).contiguous()
    _MASK_CACHE[0], _MASK_CACHE[1] = weakref.ref(mask), result
    return result


def _autocast_fp16():
    try:
        return torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") == torch.float16
    except (TypeError, AttributeError):  # torch < 2.4
        return torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16


ORIGINAL = None


def patch_qwen2_attention():
    """Register our attention as transformers' "sdpa" implementation. Calls the kernel does not
    cover (dropout, sliding window, a cache, packed sequences, another mask, CPU, bf16, other
    head sizes) go to the original function."""
    global ORIGINAL
    from transformers.modeling_utils import AttentionInterface

    current = AttentionInterface._global_mapping.get("sdpa")
    if current is None or getattr(current, "_qwen_triton", False):
        return
    ORIGINAL = original = current

    def sdpa_attention_forward(module, query, key, value, attention_mask, dropout=0.0,
                               scaling=None, is_causal=None, **kwargs):
        T = query.shape[2] if isinstance(query, torch.Tensor) and query.dim() == 4 else -1
        is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
        # fp16 autocast (the training run) or fp16 inputs only: elsewhere the reference keeps
        # fp32/bf16, which our fp16 tiles would not reproduce.
        usable = (T > 1 and dropout == 0.0 and is_causal and query.is_cuda
                  and (query.dtype == torch.float16 or _autocast_fp16())
                  and kwargs.get("sliding_window") is None
                  and kwargs.get("position_bias") is None
                  and not kwargs.get("output_attentions", False)
                  and not any(k.startswith("cu_seq_lens") for k in kwargs)
                  and key.dim() == 4 and value.shape == key.shape
                  and key.shape[0] == query.shape[0] and key.shape[2] == T
                  and query.shape[-1] == key.shape[-1] in (32, 64, 128)
                  and query.shape[1] % key.shape[1] == 0
                  and all(x.dtype in (torch.float16, torch.float32) and x.stride(-1) == 1
                          for x in (query, key, value)))
        key_ok = None
        if usable and attention_mask is not None:
            key_ok = (_key_ok(attention_mask, T, query.shape[0])
                      if isinstance(attention_mask, torch.Tensor) else False)
            usable = key_ok is not False
        if not usable:
            return original(module, query, key, value, attention_mask, dropout=dropout,
                            scaling=scaling, is_causal=is_causal, **kwargs)
        out_dtype = torch.float16 if (query.dtype == torch.float16 or _autocast_fp16()) \
            else query.dtype
        return qwen_attention(query, key, value, key_ok, scaling, out_dtype), None

    sdpa_attention_forward._qwen_triton = True
    AttentionInterface._global_mapping["sdpa"] = sdpa_attention_forward


def _grouped_sdpa(query, key, value, attention_mask, scaling):
    """torch SDPA once per kv head, its query heads against k/v expanded as stride-0 views:
    the same products as transformers' repeat_kv path, without materialising the copies."""
    groups = key.shape[1]
    per = query.shape[1] // groups
    B, _, T, D = query.shape
    outs = [torch.nn.functional.scaled_dot_product_attention(
        query[:, j * per:(j + 1) * per],
        key[:, j:j + 1].expand(B, per, key.shape[2], D),
        value[:, j:j + 1].expand(B, per, value.shape[2], D),
        attn_mask=attention_mask, dropout_p=0.0, scale=scaling) for j in range(groups)]
    return torch.cat(outs, 1).transpose(1, 2).contiguous()


def patch_qwen2_grouped_sdpa():
    """transformers' "sdpa" attention for grouped-query models with a boolean mask: torch SDPA
    per kv head (see _grouped_sdpa) instead of repeat_kv + one call. Same kernel, same values;
    anything else (no groups, dropout, sliding window, no mask, other masks) is unchanged."""
    from transformers.modeling_utils import AttentionInterface

    current = AttentionInterface._global_mapping.get("sdpa")
    if current is None or getattr(current, "_grouped_sdpa", False):
        return
    original = current

    def sdpa_attention_forward(module, query, key, value, attention_mask, dropout=0.0,
                               scaling=None, is_causal=None, **kwargs):
        groups = getattr(module, "num_key_value_groups", 1)
        if (groups <= 1 or dropout != 0.0 or not isinstance(attention_mask, torch.Tensor)
                or attention_mask.dtype != torch.bool or attention_mask.dim() != 4
                or kwargs.get("sliding_window") is not None
                or kwargs.get("position_bias") is not None
                or kwargs.get("output_attentions", False)
                or not all(isinstance(x, torch.Tensor) and x.dim() == 4 and x.is_cuda
                           for x in (query, key, value))
                or query.shape[1] != key.shape[1] * groups or key.shape != value.shape):
            return original(module, query, key, value, attention_mask, dropout=dropout,
                            scaling=scaling, is_causal=is_causal, **kwargs)
        return _grouped_sdpa(query, key, value, attention_mask, scaling), None

    sdpa_attention_forward._grouped_sdpa = True
    AttentionInterface._global_mapping["sdpa"] = sdpa_attention_forward
