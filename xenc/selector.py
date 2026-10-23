"""LayerSelector — sparse top-k gate over a candidate set of reader layers (CODE_PLAN §2(f)).

Selection must be known at the FIRST candidate layer (later layers run after it), so the selector
reads the RMS-normed residual there once per forward and stores per-token weights w [B,T,n_cand]
for the remaining candidate layers. Each layer's update is first rescaled to a MATCHED norm
(||delta||/||h|| == alpha per token), so the selection decides *where* to write, not *how much*:
after `apply`, ||Δ_l||/||h|| == w_l · alpha exactly.

Lifecycle per forward (the injector should call `reset()` before every model forward):
    compute(hn0)            optional, explicit, at the first candidate layer
    apply(delta, h, li)     at each candidate layer; at the first candidate layer, if `compute`
                            was not called explicitly since the last forward, weights are recomputed
                            from rms_norm(h.float()) — so T=1 decode steps never reuse stale weights
                            even if the caller forgets `reset()`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import xenc  # noqa: F401  (sys.path bootstrap for deltaskill)
from deltaskill.derive import rms_norm

_EPS = 1e-6


def _on(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Sharded readers: bring `t` to like.device (identity when already there; grads route back)."""
    return t if t.device == like.device else t.to(like.device)


class LayerSelector(nn.Module):
    def __init__(self, d_m: int, cand_layers: list[int], k: int = 2, alpha_init: float = 0.02,
                 straight_through: bool = False):
        super().__init__()
        cand_layers = [int(li) for li in cand_layers]
        if len(set(cand_layers)) != len(cand_layers):
            raise ValueError(f"duplicate candidate layers: {cand_layers}")
        if not 1 <= k <= len(cand_layers):
            raise ValueError(f"k={k} must be in [1, {len(cand_layers)}]")
        if alpha_init <= 0:
            raise ValueError("alpha_init must be > 0")
        self.d_m, self.k = int(d_m), int(k)
        self.cand_layers = sorted(cand_layers)
        self.n_cand = len(self.cand_layers)
        self.pos = {li: i for i, li in enumerate(self.cand_layers)}
        self.first_layer = self.cand_layers[0]
        self.straight_through = bool(straight_through)
        self.u = nn.Linear(self.d_m, self.n_cand, bias=True)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init)))
        self._w: torch.Tensor | None = None
        self._explicit = False  # compute() was called for the current forward

    # ------------------------------------------------------------------ weights
    def _weights_from_logits(self, z: torch.Tensor) -> torch.Tensor:
        v, idx = z.topk(self.k, dim=-1)
        w_sparse = torch.zeros_like(z).scatter(-1, idx, torch.softmax(v, dim=-1))
        if not self.straight_through:
            return w_sparse
        # hard forward (uniform 1/k on the kept layers), soft backward (dense softmax over all n).
        w_hard = torch.zeros_like(z).scatter(-1, idx, torch.full_like(v, 1.0 / self.k))
        w_soft = torch.softmax(z, dim=-1)
        return w_hard + (w_soft - w_soft.detach())

    def compute(self, hn0: torch.Tensor) -> None:
        """hn0: [B,T,d_m] RMS-normed residual at the first candidate layer -> stores w [B,T,n_cand]."""
        x = hn0.to(self.u.weight.dtype)
        z = F.linear(x, _on(self.u.weight, x), _on(self.u.bias, x))
        self._w = self._weights_from_logits(z)
        self._explicit = True

    def reset(self) -> None:
        self._w = None
        self._explicit = False

    def weights(self) -> torch.Tensor | None:
        return self._w

    @torch.no_grad()
    def entropy(self) -> torch.Tensor | None:
        """Mean per-token entropy (nats) of the current layer weights; None before compute."""
        if self._w is None:
            return None
        w = self._w.detach().float().clamp_min(0)
        return -(w * torch.log(w.clamp_min(1e-12))).sum(-1).mean()

    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------ apply
    def apply(self, delta: torch.Tensor, h: torch.Tensor, li: int) -> torch.Tensor:
        """Matched-norm rescale of `delta` then weight by w[..., pos[li]]. Returns delta.dtype."""
        p = self.pos[li]  # KeyError for a non-candidate layer is intended
        if li == self.first_layer:
            if not self._explicit or self._w is None:
                self.compute(rms_norm(h.float()))
            self._explicit = False  # consumed: the next first-layer call is a new forward
        elif self._w is None or self._w.shape[:-1] != h.shape[:-1]:
            # defensive fallback (hooks entered past the first candidate layer)
            self.compute(rms_norm(h.float()))
            self._explicit = False
        d = delta.float()
        rel = d.norm(dim=-1, keepdim=True) / (h.float().norm(dim=-1, keepdim=True) + _EPS)
        d = d / (rel + _EPS) * _on(self.log_alpha, d).exp()
        out = _on(self._w[..., p], d).unsqueeze(-1).to(d.dtype) * d
        return out.to(delta.dtype)

    def extra_repr(self) -> str:
        return (f"d_m={self.d_m}, cand_layers={self.cand_layers}, k={self.k}, "
                f"straight_through={self.straight_through}")

    def config(self) -> dict:
        return {"d_m": self.d_m, "cand_layers": list(self.cand_layers), "k": self.k,
                "straight_through": self.straight_through}
