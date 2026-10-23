"""Library: a stack of C chunk operators of one document (or several), keyed by READER layer.

    K[li]    : [C, N_max, d_s]  zero-padded chunk slots (V == K, tied)
    mask[li] : [C, N_max] bool  valid slots
    S[li]    : [C, d_s]         summary key = L2(mean of the chunk's valid rows), used for routing
    V[li]    : [C, N_max, d_s]  values, ONLY when the chunks are untied (harvest="attn", D3); else
               `V is None` and the read uses K as values. Centering applies to K only.

Built from ρ-keyed chunks (cache dicts or KVSkill); the ρ -> reader-layer map is the one passed in
(`reader_layers[i]` serves `sorted(depths)[i]`), normally `rho_layer_map(reader_model, depths)`.
"""

from __future__ import annotations

import torch

import xenc  # noqa: F401
from xenc.depth_skill import _key, rho_layer_map


def _l2(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


class Library:
    def __init__(self, chunks: list, reader_layers: list[int], depths, device="cpu",
                 dtype=torch.float32, meta: list[dict] | None = None,
                 center: dict | None = None):
        """``center``: optional {rho: mu[d_s]} per-depth mean slot (from scripts/compute_center.py).
        When given, every valid slot becomes L2(slot - mu[rho]) — removes the common (anisotropic)
        direction so keys/values discriminate documents instead of all pointing at the corpus mean."""
        if not chunks:
            raise ValueError("Library needs at least one chunk")
        depths = sorted(float(r) for r in depths)
        if len(reader_layers) != len(depths):
            raise ValueError(f"{len(reader_layers)} reader layers for {len(depths)} depths")
        self.depths = depths
        self.reader_layers = list(reader_layers)
        self.rho_of = dict(zip(self.reader_layers, depths))
        Ks = [c.K if hasattr(c, "K") else c["K"] for c in chunks]
        Vs = [c.V if hasattr(c, "V") else c.get("V") for c in chunks]
        untied = any(v is not None and v is not k for v, k in zip(Vs, Ks))
        self.chunk_meta = meta if meta is not None else [
            dict(getattr(c, "meta", None) or {k: v for k, v in c.items() if k not in ("K", "V")}) for c in chunks]
        self.K, self.mask, self.S = {}, {}, {}
        self.V = {} if untied else None
        for li, rho in self.rho_of.items():
            rows = [k[_key(k, rho)] for k in Ks]
            n = [r.shape[0] for r in rows]
            d = rows[0].shape[-1]
            Kt = torch.zeros(len(rows), max(n), d, dtype=dtype, device=device)
            M = torch.zeros(len(rows), max(n), dtype=torch.bool, device=device)
            for c, r in enumerate(rows):
                Kt[c, : n[c]] = r.to(device=device, dtype=dtype)
                M[c, : n[c]] = True
            if center is not None and _key(center, rho) in center:
                mu = center[_key(center, rho)].to(device=device, dtype=dtype)
                Kt = torch.where(M.unsqueeze(-1), _l2(Kt - mu), torch.zeros_like(Kt))
            self.K[li], self.mask[li] = Kt, M
            if untied:
                Vt = torch.zeros_like(Kt)
                for c, v in enumerate(Vs):
                    Vt[c, : n[c]] = (v if v is not None else Ks[c])[_key(v if v is not None else Ks[c], rho)].to(
                        device=device, dtype=dtype)
                self.V[li] = Vt
        self.center = center
        self._summaries()

    def _summaries(self) -> None:
        self.S = {}
        for li, Kt in self.K.items():
            M = self.mask[li]
            mean = (Kt * M.unsqueeze(-1).to(Kt.dtype)).sum(1) / M.sum(1, keepdim=True).clamp_min(1).to(Kt.dtype)
            self.S[li] = _l2(mean)

    # ------------------------------------------------------------------ properties
    @property
    def n_chunks(self) -> int:
        return next(iter(self.K.values())).shape[0]

    @property
    def d_s(self) -> int:
        return next(iter(self.K.values())).shape[-1]

    @property
    def layers(self) -> list[int]:
        return sorted(self.K)

    def n_valid(self, li: int) -> torch.Tensor:
        return self.mask[li].sum(-1)                  # [C]

    # ------------------------------------------------------------------ ops
    def to(self, device=None, dtype=None) -> "Library":
        """In-place move/cast (mask stays bool); returns self."""
        for li in list(self.K):
            self.K[li] = self.K[li].to(device=device, dtype=dtype)
            self.S[li] = self.S[li].to(device=device, dtype=dtype)
            self.mask[li] = self.mask[li].to(device=device)
            if self.V is not None:
                self.V[li] = self.V[li].to(device=device, dtype=dtype)
        return self

    def subset(self, idx) -> "Library":
        """Library restricted to chunks `idx` (e.g. oracle_chunk); re-trimmed so it is identical to a
        Library built from those chunks alone."""
        idx = [int(i) for i in (idx if hasattr(idx, "__iter__") else [idx])]
        if not idx:
            raise ValueError("empty subset")
        new = object.__new__(Library)
        new.depths, new.reader_layers, new.rho_of = self.depths, self.reader_layers, dict(self.rho_of)
        new.chunk_meta = [self.chunk_meta[i] for i in idx]
        new.K, new.mask = {}, {}
        new.V = {} if getattr(self, "V", None) is not None else None
        it = torch.tensor(idx, device=next(iter(self.K.values())).device)
        for li in self.K:
            M = self.mask[li].index_select(0, it)
            n = int(M.sum(-1).max())
            new.mask[li] = M[:, :n].clone()
            new.K[li] = self.K[li].index_select(0, it)[:, :n].clone()
            if new.V is not None:
                new.V[li] = self.V[li].index_select(0, it)[:, :n].clone()
        new._summaries()
        return new

    @staticmethod
    def from_doc_cache(cache: dict, reader_model, depths=None, device=None,
                       dtype=torch.float32, center: dict | None = None) -> "Library":
        depths = sorted(float(r) for r in (depths or cache["depths"]))
        rmap = rho_layer_map(reader_model, depths)
        if device is None:
            device = reader_model.device if hasattr(reader_model, "device") else "cpu"
        meta = [{"doc_id": cache.get("doc_id", ""), "start": int(c.get("start", 0)),
                 "n_tokens": int(c.get("n_tokens", 0)), "encoder": cache.get("encoder", "")}
                for c in cache["chunks"]]
        return Library(cache["chunks"], [rmap[r] for r in depths], depths, device, dtype, meta=meta,
                       center=center)


__all__ = ["Library"]
