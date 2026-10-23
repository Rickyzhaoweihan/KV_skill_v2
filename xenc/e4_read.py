"""E4: read variants for the D13 read (xenc/d13_read.py), installed by an in-process patch (default: inactive, i.e.
the read is bit-identical to D13's). Used by scripts/e4_hopdiag.py (diagnostic) and scripts/e4_register.py (judge).

Why a temperature: the D13 query is q = q_norm(W_q A RMSNorm(h)) — the encoder's RMSNorm fixes |q| (up to its frozen
weight), so training A can rotate the query but can never sharpen or flatten the read. `temp` multiplies the read's
logits (all columns: operator + own rows) at the selected reader layers.

    Variant.temp     float, logits × temp                       (1.0 = off)
    Variant.layers   set of reader layers the variant applies to (None = every read layer)
    Variant.topk     keep the K highest operator columns per head and row (0 = off); own rows always visible
    Variant.colmask  bool [N] operator columns visible (None = all)
    Variant.raw_logits  return the UN-modified logits (causal mask only) in out["logits"], so the stage-1
                     distillation term compares the query direction with the encoder teacher's attention at the
                     teacher's own temperature (registration); False returns the modified logits (diagnostic).
"""

from __future__ import annotations

import torch

import xenc.d13_read as D13


class Variant:
    temp: float = 1.0
    topk: int = 0
    colmask: torch.Tensor | None = None
    layers: set | None = None
    raw_logits: bool = False
    cur_layer: int | None = None

    @classmethod
    def reset(cls):
        cls.temp, cls.topk, cls.colmask, cls.layers, cls.raw_logits = 1.0, 0, None, None, False

    @classmethod
    def active(cls) -> bool:
        if cls.layers is not None and cls.cur_layer not in cls.layers:
            return False
        return bool(cls.topk) or cls.temp != 1.0 or cls.colmask is not None

    @classmethod
    def describe(cls) -> str:
        return (f"temp {cls.temp} topk {cls.topk} colmask {None if cls.colmask is None else int(cls.colmask.sum())} "
                f"layers {sorted(cls.layers) if cls.layers is not None else 'all'} raw_logits {cls.raw_logits}")


_ORIG_ATTEND = D13._attend
_ORIG_READ = D13.D13Interface.read


def attend_variant(q, Kop, Vop, Kown, Vown, scale, need_logits=False):
    """xenc.d13_read._attend with the Variant applied (identical call when the variant is inactive)."""
    if not Variant.active():
        return _ORIG_ATTEND(q, Kop, Vop, Kown, Vown, scale, need_logits)
    Hq, T, hd = q.shape
    Hkv, N = Kop.shape[0], Kop.shape[1]
    G = Hq // Hkv
    K = torch.cat([Kop, Kown], 1)
    V = torch.cat([Vop, Vown], 1)
    C = K.shape[1]
    s0 = torch.einsum("kgtd,kcd->kgtc", q.view(Hkv, G, T, hd), K) * scale
    first_own = C - T
    col = torch.arange(C, device=q.device).view(1, C)
    row = torch.arange(T, device=q.device).view(T, 1)
    s0 = s0.masked_fill((col > first_own + row).view(1, 1, T, C), float("-inf"))
    s = s0 * Variant.temp if Variant.temp != 1.0 else s0.clone()
    if Variant.colmask is not None:
        m = Variant.colmask.to(s.device).view(1, 1, 1, N)
        s = torch.cat([s[..., :N].masked_fill(~m, float("-inf")), s[..., N:]], -1)
    if Variant.topk and Variant.topk < N:
        so = s[..., :N]
        thr = so.topk(Variant.topk, dim=-1).values[..., -1:]
        s = torch.cat([so.masked_fill(so < thr, float("-inf")), s[..., N:]], -1)
    p = torch.softmax(s, -1)
    o = torch.einsum("kgtc,kcd->kgtd", p, V).reshape(Hq, T, hd).transpose(0, 1).reshape(T, Hq * hd)
    out = {"o": o}
    if need_logits:
        out["logits"] = s0 if Variant.raw_logits else s
    return out


def read_with_layer(self, li, *args, **kwargs):
    Variant.cur_layer = int(li)
    try:
        return _ORIG_READ(self, li, *args, **kwargs)
    finally:
        Variant.cur_layer = None


def install():
    """Route every D13 read through the variant (inactive until Variant fields are set)."""
    D13._attend = attend_variant
    D13.D13Interface.read = read_with_layer


def parse(name: str) -> dict:
    """'base' | 'temp1.5' | 'top64' | 'temp1.5@19' | 'temp1.5@3+7+11' -> {temp, topk, layers}."""
    spec, _, lay = name.partition("@")
    out = {"temp": 1.0, "topk": 0, "layers": {int(x) for x in lay.split("+")} if lay else None}
    if spec == "base":
        pass
    elif spec.startswith("temp"):
        out["temp"] = float(spec[4:])
    elif spec.startswith("top"):
        out["topk"] = int(spec[3:])
    else:
        raise ValueError(name)
    return out


def set_from(name: str, raw_logits: bool = False):
    Variant.reset()
    p = parse(name)
    Variant.temp, Variant.topk, Variant.layers, Variant.raw_logits = p["temp"], p["topk"], p["layers"], raw_logits


__all__ = ["Variant", "attend_variant", "install", "parse", "set_from"]
