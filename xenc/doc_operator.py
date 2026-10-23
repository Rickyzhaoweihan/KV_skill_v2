"""Per-document COMPACT operators for reward learning (direction D4, "RL from document").

A compact operator is the paper's factorized external operator M = W Uᵀ with U (keys) and W (values)
INDEPENDENT, m slots × d_s, one pair per reader injection layer. It is read through the unchanged
LibraryInjector as a one-chunk, untied Library (K = U, V = W, all m slots valid), so the read form
(q = L2(A·RMSNorm h), softmax/linear read over slots, h' = h + γ g B·Read) is exactly the registered one.

    svd_init(lib, m)                 text-informed init: top-m right singular vectors of the stacked
                                     (centered) harvested slots of each layer -> {li: [m, d_s]}
    operator_library(U, W, ...)      a Library whose K/V ARE the given tensors (grads flow into them)
    operator_cache(U, W, ...)        Library-compatible doc-cache dict (one untied chunk, ρ-keyed)
    save_operator / load_operator    <dir>/<doc_id>.pt (fp32), loadable by Library.from_doc_cache
"""

from __future__ import annotations

import os

import torch

import xenc  # noqa: F401
from xenc.library import Library


def svd_init(lib: Library, m: int = 64, center: bool = False, device=None) -> dict:
    """{li: [m, d_s]} unit rows = top-m right singular vectors of the valid slots of layer li.

    The slots are taken as stored in `lib` (already L2(slot - mu) when the Library was built with a
    corpus center). center=True additionally subtracts the document's own mean slot first.
    Sign convention (SVD signs are arbitrary): each direction is flipped so that the projections of
    the slots onto it have non-negative third moment (points toward the heavy tail of the document).
    If the document has fewer than m valid slots the remaining rows are random unit vectors."""
    out = {}
    g = torch.Generator().manual_seed(0)
    for li in lib.layers:
        X = lib.K[li][lib.mask[li]].detach().float().to(device or "cpu")   # [n, d]
        if center:
            X = X - X.mean(0, keepdim=True)
        k = min(m, X.shape[0], X.shape[1])
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)
        D = Vh[:k].clone()                                             # [k, d] orthonormal rows
        proj = X @ D.T                                                 # [n, k]
        sgn = torch.sign((proj ** 3).sum(0))
        sgn[sgn == 0] = 1.0
        D = D * sgn.unsqueeze(1)
        if k < m:
            R = torch.randn(m - k, X.shape[1], generator=g).to(D.device)
            D = torch.cat([D, R / R.norm(dim=-1, keepdim=True)], 0)
        out[li] = D.cpu()
    return out


def operator_library(U: dict, W: dict, reader_layers: list[int], depths, meta: dict | None = None) -> Library:
    """Library view over trainable {li: [m, d]} U (keys) and W (values): one chunk of m valid slots.
    K[li] = U[li].unsqueeze(0) is a view, so gradients reach U/W through the injector read."""
    depths = sorted(float(r) for r in depths)
    lib = object.__new__(Library)
    lib.depths, lib.reader_layers = depths, list(reader_layers)
    lib.rho_of = dict(zip(lib.reader_layers, depths))
    lib.chunk_meta = [dict(meta or {}, start=0, n_tokens=int(next(iter(U.values())).shape[0]))]
    lib.K, lib.V, lib.mask = {}, {}, {}
    for li in lib.reader_layers:
        lib.K[li] = U[li].unsqueeze(0)
        lib.V[li] = W[li].unsqueeze(0)
        lib.mask[li] = torch.ones(1, U[li].shape[0], dtype=torch.bool, device=U[li].device)
    lib.center = None
    lib._summaries()
    return lib


def operator_cache(U: dict, W: dict, rho_of: dict, doc_id: str, encoder: str, extra: dict | None = None) -> dict:
    """Doc-cache dict (xenc.depth_skill format) holding ONE untied chunk: K = U, V = W, ρ-keyed.
    `rho_of`: reader layer -> ρ. Library.from_doc_cache(cache, reader) rebuilds operator_library."""
    depths = sorted(float(r) for r in rho_of.values())
    K = {float(rho_of[li]): U[li].detach().float().cpu().contiguous() for li in rho_of}
    V = {float(rho_of[li]): W[li].detach().float().cpu().contiguous() for li in rho_of}
    m = int(next(iter(K.values())).shape[0])
    return {"chunks": [{"K": K, "V": V, "n_tokens": m, "start": 0}],
            "d_s": int(next(iter(K.values())).shape[-1]), "doc_id": doc_id, "encoder": encoder,
            "chunk_tokens": m, "depths": depths, "harvest": "operator", "tied": False,
            "operator": dict(extra or {})}


def operator_path(op_dir: str, doc_id: str) -> str:
    return os.path.join(op_dir, f"{doc_id}.pt")


def save_operator(cache: dict, op_dir: str) -> str:
    os.makedirs(op_dir, exist_ok=True)
    path = operator_path(op_dir, cache["doc_id"])
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(cache, tmp)
    os.replace(tmp, path)
    return path


def load_operator(op_dir: str, doc_id: str) -> dict | None:
    p = operator_path(op_dir, doc_id)
    return torch.load(p, map_location="cpu", weights_only=True) if os.path.exists(p) else None


__all__ = ["svd_init", "operator_library", "operator_cache", "operator_path", "save_operator", "load_operator"]
