"""D21: post-hoc, training-free compression of a shared D13 base interface (xenc/d13_read.py state dict).

The interface's stored tensors (the encoder's frozen attention copies are NOT part of it):
    A.{li}.weight [d_e,d_r] + bias   M.{li}.0.weight [h,d_r] + bias, M.{li}.2.weight [d_e,h] + bias
    B.{li} [d_r,d_e]   Bb.{li} [d_r]   g [n_layers]
Matrices (A, M.0, M.2, B) are compressed; biases and gates stay fp32 (they are < 0.1% of the size).

A variant is a '+'-joined spec, applied left to right:
    full                    unchanged
    bf16 | int8 | int4g<G>  storage dtype of the matrices (or of their SVD factors if svd came first)
                            int8: per-output-row symmetric absmax, fp16 scale per row
                            int4g<G>: asymmetric min/max per group of G input columns, fp16 scale + zero per group
    svd<r>[_<AMB>]          truncated SVD to rank r of the A / M / B matrices (default all; e.g. svd128_A only A)
    asvd<r>[_<AMB>]         activation-aware SVD (whitened by the matrix's input covariance, `covs`), same storage
    noM                     remove the residual MLP M (its output is zero)
    drop<li>[-<li>...]      remove the read at those reader layers (gate 0: the reader's own attention output)
compress(sd, spec) -> (new tensors {key: fp32 dense reconstruction}, report {params, bytes, MiB, dropped layers}).
"""

from __future__ import annotations

import re

import torch

FP32 = 4


def _mat_keys(sd: dict) -> list[str]:
    return [k for k in sd if re.fullmatch(r"A\.\d+\.weight|M\.\d+\.[02]\.weight|B\.\d+", k)]


def _group(k: str) -> str:
    return k[0]


def _layer(k: str) -> int:
    return int(k.split(".")[1])


def quant_int8(W: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Per-row symmetric int8 -> (dequantized fp32, storage bytes)."""
    W = W.float()
    s = (W.abs().amax(1, keepdim=True) / 127.0).clamp_min(1e-12).half().float()
    q = torch.clamp(torch.round(W / s), -127, 127)
    return q * s, W.numel() + 2 * W.shape[0]


def quant_int4(W: torch.Tensor, group: int = 128) -> tuple[torch.Tensor, int]:
    """Asymmetric int4 per group of `group` input columns (the last group may be shorter)."""
    W = W.float()
    out = torch.empty_like(W)
    n_groups = 0
    for c0 in range(0, W.shape[1], group):
        w = W[:, c0:c0 + group]
        lo, hi = w.amin(1, keepdim=True), w.amax(1, keepdim=True)
        s = ((hi - lo) / 15.0).clamp_min(1e-12).half().float()
        z = torch.round(-lo / s).clamp(0, 15)
        q = torch.clamp(torch.round(w / s) + z, 0, 15)
        out[:, c0:c0 + group] = (q - z) * s
        n_groups += W.shape[0]
    return out, (W.numel() + 1) // 2 + 4 * n_groups          # 4 bits/weight + fp16 scale + fp16 zero per group


def quant(W: torch.Tensor, mode: str) -> tuple[torch.Tensor, int]:
    if mode == "fp32":
        return W.float(), W.numel() * FP32
    if mode == "bf16":
        return W.to(torch.bfloat16).float(), W.numel() * 2
    if mode == "int8":
        return quant_int8(W)
    m = re.fullmatch(r"int4g(\d+)", mode)
    if m:
        return quant_int4(W, int(m.group(1)))
    raise ValueError(f"unknown dtype {mode}")


def _svd(W: torch.Tensor, cache: dict | None, key: str):
    if cache is not None and key in cache:
        return cache[key]
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    r = (U * S, Vh)
    if cache is not None:
        cache[key] = r
    return r


def _asvd(W: torch.Tensor, C: torch.Tensor, cache: dict | None, key: str, damp: float = 1e-2):
    """Activation-aware SVD: W L = U S Vh with C = L Lᵀ the input covariance (damped); rank-r W ≈ (U S)_r (Vh_r L⁻¹),
    which minimises E‖(W − W_r) x‖² over the input distribution. -> (U S, Vh L⁻¹)."""
    ck = "asvd:" + key
    if cache is not None and ck in cache:
        return cache[ck]
    C = C.double().to(W.device)
    C = C + damp * C.diagonal().mean() * torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
    L = torch.linalg.cholesky(C)
    U, S, Vh = torch.linalg.svd(W.double() @ L, full_matrices=False)
    Vw = torch.linalg.solve_triangular(L, Vh, upper=False, left=False)       # Vh L⁻¹
    r = ((U * S).float(), Vw.float())
    if cache is not None:
        cache[ck] = r
    return r


def parse(spec: str) -> dict:
    """-> {"svd": {group: rank}, "asvd": bool, "dtype": str, "noM": bool, "drop": [li]}."""
    out = {"svd": {}, "asvd": False, "dtype": "fp32", "noM": False, "drop": []}
    for t in [x for x in spec.split("+") if x and not x.startswith("full")]:   # full_rerun = noise floor
        m = re.fullmatch(r"(a?)svd(\d+)(?:_([AMB]+))?", t)
        if m:
            out["asvd"] = out["asvd"] or bool(m.group(1))
            for g in (m.group(3) or "AMB"):
                out["svd"][g] = int(m.group(2))
        elif t in ("bf16", "int8") or re.fullmatch(r"int4g\d+", t):
            out["dtype"] = t
        elif t == "noM":
            out["noM"] = True
        elif t.startswith("drop"):
            out["drop"] += [int(x) for x in t[4:].split("-")]
        else:
            raise ValueError(f"unknown token {t!r} in {spec!r}")
    return out


@torch.no_grad()
def compress(sd: dict, spec: str, svd_cache: dict | None = None, layers: list[int] | None = None,
             covs: dict | None = None) -> tuple[dict, dict]:
    """sd: the interface tensors (A., M., B., Bb., g). Returns (changed tensors as fp32 dense, size report).
    covs: {matrix key: input covariance} for asvd<r> (activation-aware SVD).
    Sizes count only what a compressed checkpoint must store: kept layers' matrices (as stored), their biases (fp32)
    and the gates (fp32)."""
    p = parse(spec)
    layers = sorted(layers or {_layer(k) for k in _mat_keys(sd)})
    keep = [li for li in layers if li not in p["drop"]]
    new, n_params, n_bytes, dense_params = {}, 0, 0, 0
    for k in _mat_keys(sd):
        li, g, W = _layer(k), _group(k), sd[k]
        if li not in keep:
            continue
        if g == "M" and p["noM"]:
            new[k] = torch.zeros_like(W, dtype=torch.float32)
            continue
        dense_params += W.numel()
        r = p["svd"].get(g)
        if r is not None and r * (W.shape[0] + W.shape[1]) < W.numel():
            if p["asvd"]:
                assert covs is not None and k in covs, f"asvd needs the input covariance of {k}"
                US, Vh = _asvd(W, covs[k], svd_cache, k)
            else:
                US, Vh = _svd(W, svd_cache, k)
            Uq, b1 = quant(US[:, :r].contiguous(), p["dtype"])
            Vq, b2 = quant(Vh[:r].contiguous(), p["dtype"])
            new[k] = Uq @ Vq
            n_params += r * (W.shape[0] + W.shape[1])
            n_bytes += b1 + b2
        else:
            Wq, b = quant(W, p["dtype"])
            if p["dtype"] != "fp32":
                new[k] = Wq
            n_params += W.numel()
            n_bytes += b
    for k, v in sd.items():                                    # biases (fp32), only of kept parts
        if k.endswith(".bias") or k.startswith("Bb."):
            li = _layer(k)
            if li in keep and not (k.startswith("M.") and p["noM"]):
                n_params += v.numel()
                n_bytes += v.numel() * FP32
            elif k.startswith("M.") and p["noM"]:
                new[k] = torch.zeros_like(v, dtype=torch.float32)
    if "g" in sd:
        g = sd["g"].detach().float().clone()
        for li in p["drop"]:
            g[layers.index(li)] = 0.0
        if p["drop"]:
            new["g"] = g
        n_params += len(keep)
        n_bytes += len(keep) * FP32
    return new, {"spec": spec, "params": n_params, "bytes": n_bytes, "MiB": n_bytes / 2 ** 20,
                 "layers": keep, "dropped": sorted(p["drop"])}


def interface_tensors(itf) -> dict:
    """The stored (trainable) tensors of a D13Interface, as in save_d13."""
    return {k: v for k, v in itf.state_dict().items() if k.startswith(("A.", "M.", "B.", "Bb.")) or k == "g"}


__all__ = ["compress", "parse", "quant", "quant_int8", "quant_int4", "interface_tensors"]
