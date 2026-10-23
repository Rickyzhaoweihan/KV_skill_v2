"""D18 (2/2): a LIBRARY of D documents' operators read by the reader's own query in ONE pass — the paper-form read of
xenc/d13_read.py generalised from one operator to many, plus a BM25 retriever for the RAG baselines.

Per paired reader layer ℓ and reader row t (own earlier rows P, positions relative):
    x̃_t = A_ℓ(RMSNorm(h_t)); q, gate, k, v from the encoder's frozen projections (exactly D13Interface.read)
    doc d (operator K_d, V_d over its N_d prefill positions; the reader rows sit right after it, at N_d + P + t):
        s_d   = RoPE_{N_d+P+t}(q) · K_dᵀ / √hd                          slot scores
        o_d   = softmax([s_d ; s_own]) · [V_d ; V_own]                    the D13 read of document d alone
        r_d   = logsumexp(s_d[skip:])                                     route logit = content mass of d
    w = softmax_d(β · r)  (per head and row; optional top-k / forced document)   routing softmax over documents
    out = B_ℓ W_o (σ(gate) ⊙ Σ_d w_d o_d)                                 replaces the reader's attention output
`skip` = number of leading operator positions shared by every document (the chat header up to the document text;
identical keys in every operator because the prefill is causal), so routing compares document content, not sinks.
With D = 1 the read equals D13Interface.read exactly (RoPE is relative; tests/test_d18_library.py).
Routing accuracy (mass on the gold document) is recorded at the last row of every forward: R.route[li] [Hkv,G,D].
"""

from __future__ import annotations

import math
import re
import types
from collections import Counter

import torch

import xenc  # noqa: F401


def _rope(itf, xe, pos: torch.Tensor):
    cos, sin = itf.rot(xe.unsqueeze(0), pos.view(1, 1, -1).expand(3, 1, -1))
    return cos.float(), sin.float()


def lib_read(itf, li: int, x: torch.Tensor, ops: list, past=None, skip: int = 0, beta: float = 1.0,
             top_k: int = 0, force: int | None = None, wfix: torch.Tensor | None = None) -> dict:
    """x [T,d_r] fp32 reader rows; ops = [(K_d, V_d) [Hkv,N_d,hd] post-RoPE at 0..N_d-1] for this layer;
    past = (K_own, V_own) own earlier rows rotated at relative positions 0..P-1.
    -> {"y" [T,d_r], "o" [T,d_e], "w" [Hkv,G,T,D] routing weights, "past"}."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    att = itf.att[str(li)]
    hd = att.head_dim
    xe = itf.map_x(li, x)
    T = xe.shape[0]
    qh, gate = torch.chunk(att.q_proj(xe).view(T, -1, hd * 2), 2, dim=-1)
    gate = gate.reshape(T, -1)
    q = att.q_norm(qh).transpose(0, 1)                                   # [Hq,T,hd] (unrotated)
    k = att.k_norm(att.k_proj(xe).view(T, -1, hd)).transpose(0, 1)       # [Hkv,T,hd]
    v = att.v_proj(xe).view(T, -1, hd).transpose(0, 1)
    Hq, Hkv = q.shape[0], k.shape[0]
    G = Hq // Hkv
    P = 0 if past is None else past[0].shape[1]
    rel = torch.arange(P, P + T, device=xe.device)
    cos, sin = _rope(itf, xe, rel)
    qo, ko = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
    qo, ko = qo[0], ko[0]
    Kown = ko if past is None else torch.cat([past[0], ko], 1)
    Vown = v if past is None else torch.cat([past[1], v], 1)
    s_own = torch.einsum("kgtd,kcd->kgtc", qo.view(Hkv, G, T, hd), Kown) * att.scaling
    col = torch.arange(P + T, device=xe.device).view(1, -1)
    s_own = s_own.masked_fill((col > P + torch.arange(T, device=xe.device).view(-1, 1)).view(1, 1, T, -1),
                              float("-inf"))
    outs, routes, rmax, nlen = [], [], [], []
    for K, V in ops:
        N = K.shape[1]
        cos, sin = _rope(itf, xe, rel + N)
        qd, _ = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        s = torch.einsum("kgtd,kcd->kgtc", qd[0].view(Hkv, G, T, hd), K.float()) * att.scaling   # [Hkv,G,T,N]
        m = torch.maximum(s.amax(-1, keepdim=True), s_own.amax(-1, keepdim=True))
        e_d, e_o = (s - m).exp(), (s_own - m).exp()
        Z = e_d.sum(-1, keepdim=True) + e_o.sum(-1, keepdim=True)
        o = (torch.einsum("kgtc,kcd->kgtd", e_d, V.float()) + torch.einsum("kgtc,kcd->kgtd", e_o, Vown)) / Z
        outs.append(o)
        sc = s[..., skip:] if N > skip else s
        routes.append(torch.logsumexp(sc, -1))                          # [Hkv,G,T]
        rmax.append(sc.amax(-1))
        nlen.append(float(sc.shape[-1]))
    r = torch.stack(routes, -1)                                         # [Hkv,G,T,D]
    D = r.shape[-1]
    if force is not None:
        w = torch.zeros_like(r)
        w[..., int(force)] = 1.0
    elif wfix is not None:                                              # fixed document weights (D18 router, soft)
        w = wfix.to(r).view(1, 1, 1, D).expand_as(r)
    else:
        logit = beta * r
        if top_k and top_k < D:
            _, i = logit.topk(top_k, -1)
            logit = logit.masked_fill(~torch.zeros_like(logit, dtype=torch.bool).scatter(-1, i, True), float("-inf"))
        w = torch.softmax(logit, -1)
    O = torch.stack(outs, -2)                                           # [Hkv,G,T,D,hd]
    o = (w.unsqueeze(-1) * O).sum(-2).reshape(Hq, T, hd).transpose(0, 1).reshape(T, Hq * hd)
    o = att.o_proj(o * torch.sigmoid(gate))
    return {"y": itf.Bmap(li, o), "o": o, "w": w, "past": (Kown.detach(), Vown.detach()),
            "r": r.detach(), "rmax": torch.stack(rmax, -1).detach(), "n": torch.tensor(nlen, device=r.device)}


class D18LibReader:
    """Patches the reader's attention at itf.layers. mode "off" (stock) | "lib" (the library read REPLACES the
    attention output, gated own + g·(read − own), as D13Reader "student"). R.ops = {enc_layer: [(K,V), ...]}."""

    def __init__(self, model, itf):
        from deltaskill.backbone_utils import decoder_layers
        self.model, self.itf = model, itf
        L = decoder_layers(model)
        self.mods = {li: L[li].self_attn for li in itf.layers}
        self.mode = "off"
        self.ops: dict = {}
        self.skip, self.beta, self.top_k, self.force = 0, 1.0, 0, None
        self.wfix = None                       # [D] fixed routing weights (overrides beta / top_k when set)
        self.past: dict = {}
        self.route: dict = {}                 # li -> [Hkv,G,D] routing weights at the last row of the last forward
        self.route_rows: dict = {}            # li -> [Hkv,G,T,D] of the last multi-row forward (prefill)
        self.logit_rows: dict = {}            # li -> (r_lse, r_max [Hkv,G,T,D], n [D]) of the last multi-row forward
        self._orig = {}
        for li, m in self.mods.items():
            self._orig[li] = m.forward
            m.forward = types.MethodType(_lib_forward(self, li), m)

    def load(self, ops: dict, skip: int = 0):
        self.ops, self.skip, self.past, self.route, self.route_rows = ops, int(skip), {}, {}, {}
        self.logit_rows = {}

    def reset(self):
        self.past, self.route, self.route_rows, self.logit_rows = {}, {}, {}, {}

    def detach(self):
        for li, f in self._orig.items():
            self.mods[li].forward = f
        self._orig.clear()


def _lib_forward(R: D18LibReader, li: int):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        out, w = type(self).forward(self, hidden_states, position_embeddings, attention_mask,
                                    past_key_values=past_key_values, **kwargs)
        if R.mode == "off":
            return out, w
        itf = R.itf
        ops = R.ops[itf.layer_map[li]]
        res = lib_read(itf, li, hidden_states[0].float(), ops, past=R.past.get(li), skip=R.skip, beta=R.beta,
                       top_k=R.top_k, force=R.force, wfix=R.wfix)
        R.past[li] = res["past"]
        R.route[li] = res["w"][:, :, -1].detach()
        if res["w"].shape[2] > 1:
            R.route_rows[li] = res["w"].detach()
            R.logit_rows[li] = (res["r"], res["rmax"], res["n"])
        g = itf.gate(li)
        y = out[0].float() + g * (res["y"] - out[0].float())
        return y.to(out.dtype).unsqueeze(0), w

    return forward


def route_stats(route: dict, gold: int) -> dict:
    """{li: [Hkv,G,D]} -> mass on gold (mean over layers/heads), argmax-of-mean-route == gold, per-layer mass."""
    if not route:
        return {}
    W = torch.stack([w.detach().float().reshape(-1, w.shape[-1]).mean(0) for w in route.values()])   # [L,D]
    mean = W.mean(0)
    return {"gold_mass": float(mean[gold]), "top1": float(int(mean.argmax()) == gold),
            "layer_mass": {str(li): float(w.float().reshape(-1, w.shape[-1]).mean(0)[gold]) for li, w in route.items()},
            "chance": 1.0 / W.shape[-1]}


ROUTE_AGGS = ("lse_norm", "lse", "max")


def question_route(logit_rows: dict, rows: slice) -> dict:
    """Question-level routing from one prefill: per aggregator, the route score of every document averaged over the
    given reader rows, all heads and all paired layers -> {agg: [D] scores}.
      lse_norm  logsumexp(s_d) − log N_d  (log of the MEAN exp-score: length-unbiased content match; PRIMARY,
                pre-declared)
      lse       logsumexp(s_d)            (the per-row soft-routing logit of lib_read; length-biased)
      max       max_c s_d,c               (best single slot)"""
    out = {k: [] for k in ROUTE_AGGS}
    for r, rm, n in logit_rows.values():
        rr, mm = r[:, :, rows].float(), rm[:, :, rows].float()
        out["lse"].append(rr.mean((0, 1, 2)))
        out["lse_norm"].append((rr - n.log()).mean((0, 1, 2)))
        out["max"].append(mm.mean((0, 1, 2)))
    return {k: torch.stack(v).mean(0) for k, v in out.items() if v}


# ----------------------------------------------------------------------------- BM25 (pure python)
_TOK = re.compile(r"[a-z0-9]+")
_SW = set("a an the of and or to in on at for by with from is are was were be been being this that these those it its "
          "as which what who whom whose when where why how do does did not no yes his her their he she they them".split())


def bm25_tokens(text: str) -> list[str]:
    return [w for w in _TOK.findall(text.lower()) if w not in _SW]


class BM25:
    def __init__(self, docs: dict, k1: float = 1.5, b: float = 0.75):
        self.ids = list(docs)
        self.tf = [Counter(bm25_tokens(docs[d])) for d in self.ids]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.ids)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str, subset=None) -> dict:
        q = bm25_tokens(query)
        out = {}
        for i, d in enumerate(self.ids):
            if subset is not None and d not in subset:
                continue
            c, L = self.tf[i], self.len[i]
            s = 0.0
            for w in q:
                f = c.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out[d] = s
        return out

    def top1(self, query: str, subset=None) -> str:
        sc = self.scores(query, subset)
        return max(sorted(sc), key=lambda d: sc[d])


__all__ = ["lib_read", "D18LibReader", "route_stats", "question_route", "ROUTE_AGGS", "BM25", "bm25_tokens"]
