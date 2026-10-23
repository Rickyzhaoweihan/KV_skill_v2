"""D18 direction B: a trained ROUTER over a library of document operators, read by the reader's own question pass.

Pass 1 (routing): the stock reader prefills its rows (doc-free chat header + conversation); at every paired reader
layer ℓ the self-attention input x_t of the question rows is mapped exactly as D13's read maps it,
    x̃_t = A_ℓ(x_t) (+MLP)            (D13 interface, FROZEN)
    x̃ʳ_t = x̃_t + U_ℓ V_ℓ x̃_t         (router residual, rank r, U zero-init → starts at D13's own query)
    q_t  = q_norm(W_q^enc x̃ʳ_t)       (frozen encoder projections), RoPE at N_d + t (as xenc/d18_library.lib_read)
and every document d scores
    S_{ℓ,h,d} = mean_t [ logsumexp_c (q_{t,h} · K_{d,c} / √hd)  −  log N_d ]      (c over the content slots ≥ skip)
    score_d   = β · Σ_{ℓ,h} softmax(a)_{ℓ,h} · S_{ℓ,h,d}
At init (U = 0, a = 0, β = 1) score_d is exactly the pre-declared "lse_norm" aggregator of question_route.
Pass 2 (reading): the D13 read of the routed operator (hard top-1) or the library read with fixed mixture weights
softmax(score) (soft). Training: InfoNCE over gold + negatives (scripts/train_d18router.py); the read stays frozen.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

import xenc  # noqa: F401

FORMAT = "d18_router.v1"


class D18Router(nn.Module):
    def __init__(self, itf, rank: int = 64, seed: int = 0):
        super().__init__()
        self.itf = [itf]                                    # not a submodule: frozen, saved separately (D13 file)
        self.layers = list(itf.layers)
        self.rank = int(rank)
        d_e = int(itf.d_e)
        att = itf.att[str(self.layers[0])]
        self.hd = int(att.head_dim)
        self.Hq = int(att.q_proj.out_features // (2 * self.hd))
        g = torch.Generator().manual_seed(seed)
        self.V = nn.ParameterDict({str(li): nn.Parameter(torch.randn(self.rank, d_e, generator=g) / d_e ** 0.5)
                                   for li in self.layers})
        self.U = nn.ParameterDict({str(li): nn.Parameter(torch.zeros(d_e, self.rank)) for li in self.layers})
        self.agg = nn.Parameter(torch.zeros(len(self.layers), self.Hq))
        self.logbeta = nn.Parameter(torch.zeros(()))

    # ------------------------------------------------------------------ pieces
    def queries(self, li: int, x: torch.Tensor) -> torch.Tensor:
        """x [T,d_r] fp32 reader self-attention inputs -> unrotated q [Hq,T,hd]."""
        itf = self.itf[0]
        att = itf.att[str(li)]
        with torch.no_grad():
            xe = itf.map_x(li, x)
        k = str(li)
        xr = xe + (xe @ self.V[k].t()) @ self.U[k].t()
        T = xr.shape[0]
        q, _ = torch.chunk(att.q_proj(xr).view(T, -1, self.hd * 2), 2, dim=-1)
        return att.q_norm(q).transpose(0, 1)

    def doc_scores(self, li: int, q: torch.Tensor, pos: torch.Tensor, K: torch.Tensor, skip: int) -> torch.Tensor:
        """q [Hq,T,hd] unrotated; pos [T] reader-row positions; K [Hkv,N,hd] post-RoPE at 0..N-1 -> [Hq]."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        from xenc.d18_library import _rope
        itf = self.itf[0]
        att = itf.att[str(li)]
        N = K.shape[1]
        cos, sin = _rope(itf, q.new_zeros(1, 1), pos + N)
        qd, _ = apply_rotary_pos_emb(q.unsqueeze(0), q.unsqueeze(0), cos, sin)
        Hkv = K.shape[0]
        G = self.Hq // Hkv
        Kc = K[:, skip:] if N > skip else K
        s = torch.einsum("kgtd,kcd->kgtc", qd[0].view(Hkv, G, -1, self.hd), Kc.float()) * att.scaling
        return (torch.logsumexp(s, -1) - math.log(Kc.shape[1])).reshape(self.Hq, -1).mean(-1)

    def table(self, xs: dict, pos: torch.Tensor, Kdocs: dict, skip: int, ckpt: bool = False) -> torch.Tensor:
        """xs {reader_layer: [T,d_r]}; Kdocs {enc_layer: [K_d ...]} -> S [L,Hq,D]."""
        from torch.utils.checkpoint import checkpoint
        itf = self.itf[0]
        rows = []
        for li in self.layers:
            q = self.queries(li, xs[li])
            Ks = Kdocs[itf.layer_map[li]]
            if ckpt and torch.is_grad_enabled():
                cols = [checkpoint(self.doc_scores, li, q, pos, K, skip, use_reentrant=False) for K in Ks]
            else:
                cols = [self.doc_scores(li, q, pos, K, skip) for K in Ks]
            rows.append(torch.stack(cols, -1))
        return torch.stack(rows)

    def combine(self, S: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.agg.flatten(), 0).view_as(self.agg)
        return self.logbeta.exp() * (w.unsqueeze(-1) * S).sum((0, 1))

    def forward(self, xs, pos, Kdocs, skip, ckpt=False):
        return self.combine(self.table(xs, pos, Kdocs, skip, ckpt))

    # ------------------------------------------------------------------ io
    def save(self, path: str, extra: dict | None = None):
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"format": FORMAT, "config": {"rank": self.rank, "layers": self.layers},
                    "state_dict": {k: v.detach().cpu() for k, v in self.state_dict().items()}, "extra": extra or {}},
                   path + ".tmp")
        os.replace(path + ".tmp", path)

    @classmethod
    def load(cls, path: str, itf):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        assert blob.get("format") == FORMAT, blob.get("format")
        r = cls(itf, rank=blob["config"]["rank"])
        assert r.layers == blob["config"]["layers"]
        r.load_state_dict(blob["state_dict"])
        return r, blob.get("extra") or {}


class HiddenCap:
    """Forward pre-hooks on the reader's self-attention modules: records their `hidden_states` input (the x that the
    D13 read maps). cap.on = True to record; cap.h {li: [1,T,d_r]}."""

    def __init__(self, model, layers):
        from deltaskill.backbone_utils import decoder_layers
        L = decoder_layers(model)
        self.h: dict = {}
        self.on = False
        self.handles = [L[li].self_attn.register_forward_pre_hook(self._hook(li), with_kwargs=True) for li in layers]

    def _hook(self, li):
        def f(mod, args, kwargs):
            if self.on:
                x = kwargs.get("hidden_states", args[0] if args else None)
                self.h[li] = x.detach()
        return f

    def remove(self):
        for h in self.handles:
            h.remove()


__all__ = ["D18Router", "HiddenCap", "FORMAT"]
