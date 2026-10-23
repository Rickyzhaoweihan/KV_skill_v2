"""D14: compress the SHIPPED encoder cache of the cross-model encoder-side read (xenc/encside_read.py).

In the encoder-side read the small encoder prefills the document once; its memory (full-attention K/V + Gated-
DeltaNet recurrent and conv states) is what is shipped, and the encoder later processes [doc tail; question] over
it. This module turns a harvested memory (`xenc.kv_prune.harvest_with_importance`) into a compressed one:

  * column pruning of the full-attention K/V (D8's acc / rand rules, xenc/kv_prune.py), on all or a subset of the
    attention layers (per-layer budgets -> `ColumnBudget` mask);
  * fake quantization (quantize -> dequantize) of K/V and of the DeltaNet states, symmetric, per channel:
      K/V [1,n_kv,T,hd]  : one scale per (kv head, channel) over tokens (group > 0: per group of `group` tokens);
      recurrent [1,H,dk,dv]: one scale per (head, dv column) over dk;  conv [1,C,k]: one scale per C over k;
  * state edits: drop the recurrent state (conv only), drop both, scale the recurrent state.

Byte accounting (`shipped_bytes`) counts what has to be sent: kept K/V columns (b_ℓ per layer, not b_max) at the
stored bit width + fp16 scales, and the states kept. `needed_upto` restricts the count to layers ≤ the deepest
encoder layer the read uses (deeper layers cannot influence the captured attention outputs: lossless truncation).

    spec = OpSpec(score="acc", frac=0.25, kv_bits=8, state_bits=8)
    st_c, info = compress_memory(enc, st, imp, spec)
    cache = build_prefix_cache(enc, st_c, keep_kv=True, keep_state=spec.keep_state, keep_conv=spec.keep_conv,
                               state_scale=spec.state_scale)
    with ColumnBudget(enc, info["col_budget"], info["b_max"]): ...      # when info["masked"]
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

import xenc  # noqa: F401
from xenc.kv_prune import prune_states


@dataclass
class OpSpec:
    score: str | None = None          # None (no pruning) | "acc" | "mean" | "rand"
    frac: float = 1.0                 # keep fraction on the pruned layers
    prune_layers: tuple | None = None # None = all full-attention layers; else only these (others keep 100%)
    kv_bits: int = 16                 # 16 (bf16, as harvested) | 8 | 4
    kv_group: int = 0                 # 0 = per channel over all tokens; >0 = per channel per token group
    state_bits: int = 16              # quantization of recurrent + conv states
    keep_state: bool = True           # recurrent state shipped
    keep_conv: bool = True            # conv state shipped
    state_scale: float = 1.0
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@torch.no_grad()
def fake_quant(x: torch.Tensor, bits: int, dim: int, group: int = 0) -> torch.Tensor:
    """Symmetric quantize -> dequantize; the absmax scale is taken over `dim` (one scale per remaining index).
    group > 0: `dim` is split into groups of `group` (last group may be shorter), one scale per group."""
    if bits >= 16:
        return x
    qmax = 2 ** (bits - 1) - 1
    xf = x.float()
    if group and group < xf.shape[dim]:
        parts = [fake_quant(p, bits, dim, 0) for p in torch.split(x, group, dim=dim)]
        return torch.cat(parts, dim=dim)
    s = xf.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8) / qmax
    s = s.half().float()                                   # scales are shipped as fp16
    return (torch.round(xf / s).clamp(-qmax, qmax) * s).to(x.dtype)


def _n_scales(shape, dim: int, group: int = 0) -> int:
    n = 1
    for i, s in enumerate(shape):
        if i != dim % len(shape):
            n *= s
    if group:
        n *= -(-shape[dim] // group)
    return n


@torch.no_grad()
def compress_memory(model, st: dict, imp: dict | None, spec: OpSpec, seed: int = 0) -> tuple[dict, dict]:
    """-> (compressed memory in harvest_states layout, info{col_budget, b_max, masked, n_doc_full})."""
    T = int(st["n_doc"])
    info = {"n_doc_full": T, "b_max": T, "col_budget": {li: T for li in st["kv"]}, "masked": False}
    if spec.score is not None and spec.frac < 1.0:
        budgets = None
        if spec.prune_layers is not None:
            budgets = {li: (spec.frac if li in set(spec.prune_layers) else 1.0) for li in st["kv"]}
        st, pi = prune_states(model, st, imp, frac=spec.frac, score=spec.score, budgets=budgets, seed=seed)
        info.update(b_max=pi["b_max"], col_budget=dict(pi["col_budget"]))
        info["masked"] = any(b < pi["b_max"] for b in pi["col_budget"].values())
    kv = st["kv"]
    if spec.kv_bits < 16:
        kv = {}
        for li, (K, V) in st["kv"].items():
            b = info["col_budget"][li]
            # only the readable columns [0, b_ℓ) are shipped; quantize them (the masked rest is never read)
            Kq, Vq = K.clone(), V.clone()
            Kq[:, :, :b] = fake_quant(K[:, :, :b], spec.kv_bits, dim=2, group=spec.kv_group)
            Vq[:, :, :b] = fake_quant(V[:, :, :b], spec.kv_bits, dim=2, group=spec.kv_group)
            kv[li] = (Kq, Vq)
    rec, conv = st["recurrent"], st["conv"]
    if spec.state_bits < 16:
        rec = {li: fake_quant(S, spec.state_bits, dim=2) for li, S in st["recurrent"].items()}
        conv = {li: fake_quant(c, spec.state_bits, dim=-1) for li, c in st["conv"].items()}
    return {**st, "kv": kv, "recurrent": rec, "conv": conv}, info


def shipped_bytes(st: dict, info: dict, spec: OpSpec, needed_upto: int | None = None) -> dict:
    """Bytes of the shipped operator (per document). st = the HARVESTED (uncompressed) memory (shapes only)."""
    def ok(li):
        return needed_upto is None or li <= needed_upto
    kv_b = 0
    for li, (K, V) in st["kv"].items():
        if not ok(li):
            continue
        _, n_kv, _, hd = K.shape
        b = int(info["col_budget"][li])
        for X in (K, V):
            n = n_kv * b * hd
            if spec.kv_bits >= 16:
                kv_b += n * X.element_size()
            else:
                kv_b += n * spec.kv_bits / 8 + 2 * _n_scales((1, n_kv, b, hd), 2, spec.kv_group)
    rec_b = conv_b = 0
    for li, S in st["recurrent"].items():
        if not ok(li):
            continue
        if spec.keep_state:
            rec_b += S.numel() * (S.element_size() if spec.state_bits >= 16 else spec.state_bits / 8)
            if spec.state_bits < 16:
                rec_b += 2 * _n_scales(tuple(S.shape), 2)
        c = st["conv"][li]
        if spec.keep_conv:
            conv_b += c.numel() * (c.element_size() if spec.state_bits >= 16 else spec.state_bits / 8)
            if spec.state_bits < 16:
                conv_b += 2 * _n_scales(tuple(c.shape), -1)
    return {"kv": kv_b, "recurrent": rec_b, "conv": conv_b, "total": kv_b + rec_b + conv_b}


def parse_spec(name: str) -> OpSpec:
    """Condition-name grammar (tokens joined by '_'):
       full | acc25 | rand10 | mean5 ...      pruning on all attention layers (keep %)
       late:<l1>+<l2>...                       restrict pruning to these attention layers (e.g. late:11+15+19+23)
       q8 | q4 | q4g64                         K/V bits (+ token group)      s8 | s4   state bits
       conv                                    drop the recurrent state (conv kept)   nostate   drop both
       sc0.5                                   recurrent-state scale
    e.g. "acc25_q8_s8", "late:11+15+19+23_acc10", "conv_acc25"."""
    sp = OpSpec()
    for t in name.split("_"):
        if t == "full":
            continue
        if t.startswith("late:"):
            sp.prune_layers = tuple(int(x) for x in t[5:].split("+"))
        elif any(t.startswith(r) and t[len(r):].replace(".", "").isdigit() for r in ("acc", "rand", "mean")):
            r = next(r for r in ("acc", "rand", "mean") if t.startswith(r))
            sp.score, sp.frac = r, float(t[len(r):]) / 100.0
        elif t.startswith("q") and t[1:2].isdigit():
            b, _, g = t[1:].partition("g")
            sp.kv_bits, sp.kv_group = int(b), int(g) if g else 0
        elif t.startswith("s") and t[1:].isdigit():
            sp.state_bits = int(t[1:])
        elif t.startswith("sc"):
            sp.state_scale = float(t[2:])
        elif t == "conv":
            sp.keep_state = False
        elif t == "nostate":
            sp.keep_state, sp.keep_conv = False, False
        else:
            raise ValueError(f"bad condition token {t!r} in {name!r}")
    return sp


__all__ = ["OpSpec", "fake_quant", "compress_memory", "shipped_bytes", "parse_spec"]
