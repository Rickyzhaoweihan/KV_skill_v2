"""D12: context write-back B for the encoder-side read (non-token-wise).

For reader layer ℓ the injected attention output is an affine map of a FEATURE vector built from the encoder's
attention outputs at SEVERAL encoder full-attention layers and/or a short causal window of encoder tokens:

    x_ℓ[t] = concat_{(e, lag) in src[ℓ]}  o_e[t − lag]          (o_e: encoder attention output, level post/pre)
    y_ℓ[t] = W_ℓ (x_ℓ[t] − μx_ℓ) + μy_ℓ

W, μx, μy come from `scripts/fit_encside_B2.py` (closed-form ridge, label-free: target = the reader's own
in-context attention output at the same token). Blob format "xenc.encside_B2.v1":
    {"format", "level", "layers": {str(ℓ): {"src": [[e, lag], ...], "match": e, "W": [d_out, D], "mx": [D], "my": [d_out]}}}

`build(o2, reset)` turns {enc_layer: [1, T, d]} (one encoder forward) into {ℓ: [1, T, D]}, keeping the last
max-lag rows of every encoder layer so that single-token decode steps see their causal window.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xenc.encside_read import EncSideInterface


class CtxEncSideInterface(EncSideInterface):
    def __init__(self, blob: dict, layers, mode: str = "replace", gate_init: float = 1.0, mix_init: float = 0.0,
                 w_dtype=torch.bfloat16):
        spec = {int(k): v for k, v in blob["layers"].items()}
        layers = [int(x) for x in layers]
        missing = [li for li in layers if li not in spec]
        if missing:
            raise ValueError(f"context B has no map for reader layers {missing}")
        lmap = {li: int(spec[li]["match"]) for li in layers}
        W0 = spec[layers[0]]["W"]
        super().__init__(lmap, int(W0.shape[1]), int(W0.shape[0]), mode, blob.get("level", "post"), identity=True,
                         gate_init=gate_init, mix_init=mix_init)
        self.identity = False
        self.src = {li: [(int(e), int(g)) for e, g in spec[li]["src"]] for li in layers}
        for li in layers:
            self.register_buffer(f"W{li}", spec[li]["W"].to(w_dtype), persistent=False)
            self.register_buffer(f"mx{li}", spec[li]["mx"].float(), persistent=False)
            self.register_buffer(f"my{li}", spec[li]["my"].float(), persistent=False)
        self.enc_layers = sorted({e for li in layers for e, _ in self.src[li]})
        self.max_lag = max(g for li in layers for _, g in self.src[li])
        self._hist: dict = {}

    def map(self, li, x):
        W = getattr(self, f"W{li}")
        y = (x.float() - getattr(self, f"mx{li}")).to(W.dtype) @ W.t()
        return y.float() + getattr(self, f"my{li}")

    @torch.no_grad()
    def build(self, o2: dict, reset: bool = False) -> dict:
        if reset:
            self._hist = {}
        full = {}
        for e in self.enc_layers:
            h = self._hist.get(e)
            full[e] = o2[e] if h is None else torch.cat([h.to(o2[e].device), o2[e]], dim=1)
        T = next(iter(o2.values())).shape[1]
        out = {}
        for li in self.layers:
            parts = []
            for e, g in self.src[li]:
                s = full[e]
                L = s.shape[1]
                if g == 0:
                    parts.append(s[:, L - T:])
                else:
                    sh = torch.cat([s[:, :1].expand(-1, g, -1), s[:, :-g]], dim=1)   # pad by repeating row 0
                    parts.append(sh[:, L - T:])
            out[li] = torch.cat(parts, dim=-1)
        if self.max_lag:
            self._hist = {e: full[e][:, -self.max_lag:] for e in self.enc_layers}
        return out


def neighbour_src(enc_full: list, e: int, n_layers: int = 3, lags=(0,), match_lags=None) -> list:
    """Feature sources for a reader layer matched to encoder layer e: the n_layers encoder full-attention layers
    nearest e in depth order (e first), each at lag 0, plus e at the extra `match_lags`."""
    k = enc_full.index(e)
    order = sorted(range(len(enc_full)), key=lambda j: (abs(j - k), j))[: max(1, n_layers)]
    src = [(enc_full[j], 0) for j in sorted(order, key=lambda j: (j != k, j))]
    src = [(x, g) for x, _ in src for g in lags] if tuple(lags) != (0,) else src
    for g in (match_lags or []):
        if (e, g) not in src:
            src.append((e, g))
    return src


class TrainCtxEncSide(CtxEncSideInterface):
    """Trainable context B (D12 stage 3), initialised from a closed-form blob W0:
        y_ℓ = exp(lg_ℓ) · ( W_ℓ (x − μx) + μy_ℓ + δ_ℓ ),   W_ℓ = W0_ℓ + U_ℓ V_ℓᵀ  (rank > 0)  or a free W_ℓ (rank 0).
    `reg()` = mean_ℓ ‖W_ℓ − W0_ℓ‖²_F / ‖W0_ℓ‖²_F (keeps B near the fitted ridge map). `export()` folds the gain into
    a plain "xenc.encside_B2.v1" blob loadable by CtxEncSideInterface / eval_encside.py --maps."""

    def __init__(self, blob: dict, layers, rank: int = 128, mode: str = "replace", seed: int = 0):
        super().__init__(blob, layers, mode, w_dtype=torch.float32)
        self.rank = int(rank)
        self.blob_meta = {k: v for k, v in blob.items() if k != "layers"}
        g = torch.Generator().manual_seed(seed)
        self.U, self.V, self.Wf = nn.ParameterDict(), nn.ParameterDict(), nn.ParameterDict()
        self.dmy = nn.ParameterDict()
        for li in self.layers:
            W0 = getattr(self, f"W{li}")
            do, D = W0.shape
            if self.rank:
                self.U[str(li)] = nn.Parameter(torch.zeros(do, self.rank))
                self.V[str(li)] = nn.Parameter(torch.randn(D, self.rank, generator=g) / D ** 0.5)
            else:
                self.Wf[str(li)] = nn.Parameter(W0.clone())
            self.dmy[str(li)] = nn.Parameter(torch.zeros(do))
        self.lg = nn.Parameter(torch.zeros(len(self.layers)))
        self._w0n = {li: float(getattr(self, f"W{li}").pow(2).sum()) for li in self.layers}

    def weight(self, li):
        if self.rank:
            return getattr(self, f"W{li}") + self.U[str(li)] @ self.V[str(li)].t()
        return self.Wf[str(li)]

    def map(self, li, x):
        xc = x.float() - getattr(self, f"mx{li}")
        if self.rank:
            y = xc @ getattr(self, f"W{li}").t() + (xc @ self.V[str(li)]) @ self.U[str(li)].t()
        else:
            y = xc @ self.Wf[str(li)].t()
        return (y + getattr(self, f"my{li}") + self.dmy[str(li)]) * self.lg[self._i[li]].exp()

    def reg(self):
        out = []
        for li in self.layers:
            if self.rank:
                U, V = self.U[str(li)], self.V[str(li)]
                d2 = ((U.t() @ U) * (V.t() @ V)).sum()          # ‖U Vᵀ‖²_F
            else:
                d2 = (self.Wf[str(li)] - getattr(self, f"W{li}")).pow(2).sum()
            out.append(d2 / self._w0n[li])
        return torch.stack(out).mean()

    def param_groups(self, lr: float, lr_gain: float):
        main = list(self.U.values()) + list(self.V.values()) + list(self.Wf.values())
        return [{"params": main, "lr": lr}, {"params": list(self.dmy.values()) + [self.lg], "lr": lr_gain,
                                            "weight_decay": 0.0}]

    @torch.no_grad()
    def export(self) -> dict:
        layers = {}
        for li in self.layers:
            gn = float(self.lg[self._i[li]].exp())
            layers[str(li)] = {"src": [[e, g] for e, g in self.src[li]], "match": int(self.layer_map[li]),
                               "W": (self.weight(li) * gn).to(torch.bfloat16).cpu(),
                               "mx": getattr(self, f"mx{li}").float().cpu(),
                               "my": ((getattr(self, f"my{li}") + self.dmy[str(li)]) * gn).float().cpu(),
                               "gain": gn}
        return {**self.blob_meta, "format": "xenc.encside_B2.v1", "layers": layers}


def from_tokenwise(blob: dict) -> dict:
    """D10 token-wise blob (enc_B, no intercept) -> an equivalent "xenc.encside_B2.v1" blob."""
    layers = {}
    for li, W in blob["enc_B"].items():
        e = int(blob["layer_pairs"][int(li)] if int(li) in blob["layer_pairs"] else blob["layer_pairs"][str(li)])
        layers[str(li)] = {"src": [[e, 0]], "match": e, "W": W.float(), "mx": torch.zeros(W.shape[1]),
                           "my": torch.zeros(W.shape[0])}
    return {"format": "xenc.encside_B2.v1", "level": blob.get("level", "post"), "layers": layers}


__all__ = ["CtxEncSideInterface", "TrainCtxEncSide", "from_tokenwise", "neighbour_src"]
