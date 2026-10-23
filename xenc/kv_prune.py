"""D8 (phase 2): training-free COMPRESSION of the reader's native document memory (full-attention K/V columns).

D7 (xenc/native_state.py) showed that the reader's own full-attention K/V of a one-time document prefill, used as a
prefix cache, carries the document (kv_only − wrong_kv ≈ +0.13–0.15). Here the K/V is compressed by
query-independent COLUMN PRUNING, chosen once per document (no question):

  score "acc"  : attention RECEIVED by each document position during the document prefill itself, summed over all
                 document queries and over the query heads of each KV head (H2O-style heavy hitters);
  score "mean" : the same divided by the number of queries that can see the position (T − i), removing the
                 causal early-position bias;
  score "rand" : random positions (control).
Per layer and per KV head the top-b positions are kept (b = frac·T, or per-layer budgets).

Positions stay consistent: kept keys are post-RoPE at their ORIGINAL positions; the reader then continues at
cache length b (not T), so every kept key is re-rotated by −(T − b) (RoPE composes: R(p)R(−Δ) = R(p − Δ)), which
preserves every query–key distance exactly. Hence a pruned cache == the full cache with the dropped columns masked
(tests/test_kv_prune.py checks this). Per-layer budgets keep b_max columns everywhere (columns stored in
descending-score order) and mask columns ≥ b_ℓ at layer ℓ via an additive attention-mask hook (`ColumnBudget`).

    imp = harvest_with_importance(model, tok, doc_text)        # -> (st, {li: [n_kv, T] acc}, ...)
    st_p, info = prune_states(model, st, imp, frac=0.10, score="acc")
    cache = build_prefix_cache(model, st_p, keep_kv=True, keep_state=False, keep_conv=False)   # D7 builder
"""

from __future__ import annotations

import torch

import xenc  # noqa: F401
from deltaskill.backbone_utils import decoder_layers
from xenc.native_state import harvest_states


def _rotary(model):
    for name, m in model.named_modules():
        if name.endswith("rotary_emb") and "visual" not in name:
            return m
    raise ValueError("no rotary_emb module found")


def _apply_rope(x, cos, sin):
    """x [B,H,T,hd]; cos/sin [B,T,rot] (partial rotary on the first rot dims, rotate_half convention)."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    x1, x2 = xr[..., : r // 2], xr[..., r // 2:]
    rh = torch.cat((-x2, x1), dim=-1)
    return torch.cat([xr * cos + rh * sin, xp], dim=-1)


@torch.no_grad()
def shift_keys(model, K: torch.Tensor, delta: int) -> torch.Tensor:
    """Post-RoPE keys [B,n_kv,b,hd] moved by `delta` positions (negative = earlier)."""
    if delta == 0:
        return K
    rot = _rotary(model)
    pos = torch.full((1, 1), float(delta), device=K.device)
    cos, sin = rot(K.float(), pos)                               # [1,1,rot]
    return _apply_rope(K.float(), cos.float(), sin.float()).to(K.dtype)


@torch.no_grad()
def received_attention(q: torch.Tensor, K: torch.Tensor, scaling: float, block: int = 512) -> torch.Tensor:
    """Attention received per key over all causal queries: q [1,n_q,T,hd] (post-RoPE), K [1,n_kv,T,hd].
    -> [n_kv, T] float32 (summed over each KV head's query group and over queries)."""
    _, n_q, T, hd = q.shape
    n_kv = K.shape[1]
    g = n_q // n_kv
    Kf = K[0].float()                                            # [n_kv,T,hd]
    acc = torch.zeros(n_kv, T, dtype=torch.float32, device=K.device)
    col = torch.arange(T, device=K.device)
    for s in range(0, T, block):
        e = min(T, s + block)
        qb = q[0, :, s:e].float().reshape(n_kv, g, e - s, hd)     # [n_kv,g,B,hd]
        lg = torch.einsum("kgbd,ktd->kgbt", qb, Kf[:, :e]) * scaling
        row = torch.arange(s, e, device=K.device).view(-1, 1)
        lg = lg.masked_fill((col[:e].view(1, -1) > row).view(1, 1, e - s, e), float("-inf"))
        acc[:, :e] += torch.softmax(lg, -1).sum((1, 2))
    return acc


@torch.no_grad()
def harvest_with_importance(model, tok, doc_text: str | None = None, ids: torch.Tensor | None = None,
                            max_tokens: int = 0, block: int = 512) -> tuple[dict, dict]:
    """D7's harvest_states + per-layer received-attention importance {li: [n_kv, T]} of the same prefill."""
    from xenc.native_attn import attn_geometry, reader_queries
    blocks = decoder_layers(model)
    types = list((model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config).layer_types)
    full = [i for i, t in enumerate(types) if t != "linear_attention"]
    cap, hs = {}, []
    for li in full:
        def pre(mod, args, kwargs, li=li):
            x = kwargs.get("hidden_states", args[0] if args else None)
            cap[li] = (x.detach(), tuple(t.detach() for t in kwargs["position_embeddings"]))
        hs.append(blocks[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        st = harvest_states(model, tok, doc_text, ids=ids, max_tokens=max_tokens)
    finally:
        for h in hs:
            h.remove()
    imp = {}
    for li in full:
        x, (cos, sin) = cap.pop(li)
        att = blocks[li].self_attn
        geo = attn_geometry(att)
        q = reader_queries(att, x, geo)[0]                       # [1,n_q,T,hd] pre-RoPE
        q = _apply_rope(q.float(), cos.float(), sin.float())
        imp[li] = received_attention(q, st["kv"][li][0], geo["scaling"], block)
        del x, q
    return st, imp


def position_scores(imp: torch.Tensor, score: str, seed: int = 0) -> torch.Tensor:
    """imp [n_kv,T] (acc) -> scores [n_kv,T] for the chosen rule."""
    n_kv, T = imp.shape
    if score == "acc":
        return imp
    if score == "mean":
        return imp / torch.arange(T, 0, -1, device=imp.device, dtype=imp.dtype)
    if score == "rand":
        g = torch.Generator().manual_seed(seed)
        return torch.rand(n_kv, T, generator=g).to(imp.device)
    raise ValueError(f"unknown score {score}")


@torch.no_grad()
def prune_states(model, st: dict, imp: dict, frac: float = 0.1, score: str = "acc", budgets: dict | None = None,
                 seed: int = 0, min_keep: int = 1) -> tuple[dict, dict]:
    """-> (pruned state dict (same layout as harvest_states; DeltaNet states shared, not copied), info).
    budgets: optional {li: fraction} per full-attention layer (overrides frac); columns are stored in descending
    score order, every layer keeps b_max = max_ℓ b_ℓ columns, info["col_budget"] = {li: b_ℓ} for `ColumnBudget`."""
    T = int(st["n_doc"])
    fr = {li: float(budgets.get(li, frac)) if budgets else float(frac) for li in st["kv"]}
    b = {li: max(min_keep, min(T, int(round(f * T)))) for li, f in fr.items()}
    bmax = max(b.values())
    kv = {}
    for li, (K, V) in st["kv"].items():
        sc = position_scores(imp[li], score, seed=seed * 1000 + li)          # [n_kv,T]
        idx = sc.topk(bmax, -1).indices                                       # descending score
        gi = idx.to(K.device).view(1, idx.shape[0], bmax, 1).expand(-1, -1, -1, K.shape[-1])
        Kp = shift_keys(model, K.gather(2, gi), -(T - bmax))
        kv[li] = (Kp, V.gather(2, gi))
    out = {**st, "kv": kv, "n_doc": bmax}
    info = {"n_doc_full": T, "b_max": bmax, "col_budget": b, "kept_frac": sum(b.values()) / (len(b) * T)}
    return out, info


class ColumnBudget:
    """Context manager: at full-attention layer ℓ, mask cached prefix columns [b_ℓ, b_max) for every new token
    (additive -inf mask merged with causality), so layer ℓ reads only its own b_ℓ best columns."""

    def __init__(self, model, col_budget: dict, b_max: int):
        self.model, self.b = model, {int(k): int(v) for k, v in col_budget.items() if int(v) < int(b_max)}
        self.b_max = int(b_max)
        self._h = []

    def __enter__(self):
        blocks = decoder_layers(self.model)
        for li, bl in self.b.items():
            def pre(mod, args, kwargs, bl=bl, li=li):
                x = kwargs.get("hidden_states", args[0] if args else None)
                cache = kwargs.get("past_key_values")
                B, T = x.shape[:2]
                past = cache.layers[li].get_seq_length() if cache is not None else 0
                kvn = past + T
                col = torch.arange(kvn, device=x.device).view(1, -1)
                row = torch.arange(T, device=x.device).view(-1, 1) + past
                ok = (col <= row) & ((col < bl) | (col >= self.b_max))
                m = torch.zeros(B, 1, T, kvn, dtype=x.dtype, device=x.device)
                m.masked_fill_(~ok.view(1, 1, T, kvn), torch.finfo(x.dtype).min)
                kwargs["attention_mask"] = m
                return args, kwargs
            self._h.append(blocks[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []
        return False


def kv_nbytes(st: dict) -> int:
    return sum(x.numel() * x.element_size() for kv in st["kv"].values() for x in kv)


__all__ = ["harvest_with_importance", "received_attention", "prune_states", "shift_keys", "position_scores",
           "ColumnBudget", "kv_nbytes"]
