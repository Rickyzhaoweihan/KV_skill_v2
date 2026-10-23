"""D12b H-slots: a hypernetwork that writes per-document MEMORY SLOTS for the paper-form read (27B reader).

Test time: the encoder prefills the document once -> operator {enc_layer: (K, V) [Hkv,N,hd]} (stored, never modified)
-> ONE forward of SlotH(operator) -> per read layer l, S extra (K_s, V_s) [Hkv,S,hd] slots -> the reader's read attends
to the operator AND to the slots. No reader pass over the document.

SlotH (Perceiver-style, in the operator's own key space): per encoder read layer e, S learned latents; n_blocks of
[cross-attention of the latents over the operator (per kv head: q = LN(lat)·Wq, softmax(q·K_op/√hd)·V_op) + MLP];
per reader layer l (paired with e): K_s = √hd·normalize(LN(lat_e)·Wk_l)·kscale_l, V_s = LN(lat_e)·Wv_l.
Wv is ZERO-INITIALISED, so the slots contribute exactly 0 at step 0 (read == base read).

Slot read (inside the reader's read, `attach_slots` patches the interface INSTANCE's `read`): the slots form a separate
softmax branch over S columns, queried by the reader's own query BEFORE RoPE (slots are content addresses, no
position); its output is added to the operator read's per-head output before the output gate / o_proj:
    o = o_proj((o_op + softmax(q·K_sᵀ/√hd)·V_s) ⊙ σ(gate)),   y = B·o + b
Diagnostics (no grad): slot share = ‖o_proj(o_slot·σ(gate))‖ / ‖o_op-part‖ and slot attention entropy / log S, per
read, accumulated in itf._slot_stats while it is a list.
"""
from __future__ import annotations

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

FORMAT = "xenc.d12b_slots.v1"


class _Block(nn.Module):
    def __init__(self, D: int, hkv: int, hd: int, mlp_mult: int = 2):
        super().__init__()
        self.hkv, self.hd = hkv, hd
        self.ln_q = nn.LayerNorm(D)
        self.Wq = nn.Linear(D, D, bias=False)
        self.Wo = nn.Linear(D, D, bias=False)
        self.ln_m = nn.LayerNorm(D)
        self.mlp = nn.Sequential(nn.Linear(D, mlp_mult * D), nn.GELU(), nn.Linear(mlp_mult * D, D))
        nn.init.normal_(self.Wq.weight, 0, D ** -0.5)
        nn.init.normal_(self.Wo.weight, 0, 0.5 * D ** -0.5)

    def forward(self, lat, K, V):
        S = lat.shape[0]
        q = self.Wq(self.ln_q(lat)).view(S, self.hkv, self.hd).transpose(0, 1)          # [Hkv,S,hd]
        p = torch.softmax(torch.einsum("hsd,hnd->hsn", q, K) / math.sqrt(self.hd), -1)
        z = torch.einsum("hsn,hnd->hsd", p, V).transpose(0, 1).reshape(S, -1)          # [S,D]
        lat = lat + self.Wo(z)
        return lat + self.mlp(self.ln_m(lat))


class SlotH(nn.Module):
    kind = "slots"

    def __init__(self, itf, n_slots: int = 64, n_blocks: int = 2, seed: int = 0, mlp_mult: int = 2):
        super().__init__()
        self.layers = list(itf.layers)
        self.layer_map = {int(k): int(v) for k, v in itf.layer_map.items()}
        self.enc_layers = sorted(set(self.layer_map.values()))
        att = itf.att[str(self.layers[0])]
        hd = int(att.head_dim)
        hkv = int(att.k_proj.out_features) // hd
        self.hkv, self.hd, self.S = hkv, hd, int(n_slots)
        D = hkv * hd
        self.cfg = {"n_slots": int(n_slots), "n_blocks": int(n_blocks), "seed": int(seed), "mlp_mult": int(mlp_mult)}
        g = torch.Generator().manual_seed(seed + 31)
        self.lat = nn.ParameterDict({str(e): nn.Parameter(torch.randn(self.S, D, generator=g)) for e in self.enc_layers})
        self.blocks = nn.ModuleDict({str(e): nn.ModuleList(_Block(D, hkv, hd, mlp_mult) for _ in range(n_blocks))
                                     for e in self.enc_layers})
        self.ln_out = nn.ModuleDict({str(e): nn.LayerNorm(D) for e in self.enc_layers})
        self.Wk = nn.ModuleDict({str(li): nn.Linear(D, D, bias=False) for li in self.layers})
        self.Wv = nn.ModuleDict({str(li): nn.Linear(D, D, bias=False) for li in self.layers})
        self.kscale = nn.Parameter(torch.ones(len(self.layers)))
        for li in self.layers:
            nn.init.normal_(self.Wk[str(li)].weight, 0, D ** -0.5)
            nn.init.zeros_(self.Wv[str(li)].weight)            # slots contribute exactly 0 at init

    def forward(self, op: dict) -> dict:
        lat = {}
        for e in self.enc_layers:
            K, V = op[e]
            K, V = K.float(), V.float()
            z = self.lat[str(e)].to(K.device)
            for blk in self.blocks[str(e)]:
                z = blk(z, K, V)
            lat[e] = self.ln_out[str(e)](z)
        Ks, Vs = {}, {}
        for i, li in enumerate(self.layers):
            z = lat[self.layer_map[li]]
            k = self.Wk[str(li)](z).view(self.S, self.hkv, self.hd).transpose(0, 1)
            Ks[li] = F.normalize(k, dim=-1) * math.sqrt(self.hd) * self.kscale[i]
            Vs[li] = self.Wv[str(li)](z).view(self.S, self.hkv, self.hd).transpose(0, 1)
        out = {"K": Ks, "V": Vs}
        out["_norm"] = torch.stack([v.pow(2).mean() for v in Vs.values()]).mean()
        return out

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def adapter_numel(self) -> int:
        return len(self.layers) * 2 * self.S * self.hkv * self.hd


def attach_slots(itf):
    """Patch the interface INSTANCE's read with the slot branch (itf._slots: None = base read, or SlotH output)."""
    assert not getattr(itf, "_slots_attached", False), "already attached"
    itf._slots, itf._slots_attached, itf._slot_stats = None, True, None
    cls = type(itf)

    def read(self, li, x, Kop, Vop, pos0: int, past=None, need_logits: bool = False):
        res = cls.read(self, li, x, Kop, Vop, pos0, past=past, need_logits=need_logits)
        Sl = self._slots
        if Sl is None:
            return res
        att = self.att[str(li)]
        hd = att.head_dim
        xe = self.map_x(li, x)
        T = xe.shape[0]
        q, gate = torch.chunk(att.q_proj(xe).view(T, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(T, -1)
        q = att.q_norm(q).transpose(0, 1)                                   # [Hq,T,hd], no RoPE (content address)
        Ks, Vs = Sl["K"][li].to(q.device).float(), Sl["V"][li].to(q.device).float()
        Hq, Hkv = q.shape[0], Ks.shape[0]
        G = Hq // Hkv
        s = torch.einsum("kgtd,ksd->kgts", q.view(Hkv, G, T, hd), Ks) * att.scaling
        p = torch.softmax(s, -1)
        os_ = torch.einsum("kgts,ksd->kgtd", p, Vs).reshape(Hq, T, hd).transpose(0, 1).reshape(T, Hq * hd)
        o_add = att.o_proj(os_ * torch.sigmoid(gate))
        o = res["o"] + o_add
        if self._slot_stats is not None:
            with torch.no_grad():
                ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean() / math.log(max(Ks.shape[1], 2))
                self._slot_stats.append((li, float(o_add.norm() / res["o"].norm().clamp_min(1e-9)), float(ent)))
        res = dict(res)
        res["o"], res["y"] = o, self.Bmap(li, o)
        return res

    itf.read = types.MethodType(read, itf)
    return itf


def slot_stats_summary(itf, reset: bool = True) -> dict:
    st = itf._slot_stats or []
    if reset:
        itf._slot_stats = [] if itf._slot_stats is not None else None
    if not st:
        return {}
    by = {}
    for li, r, e in st:
        by.setdefault(li, []).append((r, e))
    return {"share": sum(r for _, r, _ in st) / len(st), "entropy": sum(e for _, _, e in st) / len(st),
            "share_by_layer": {li: round(sum(r for r, _ in v) / len(v), 4) for li, v in sorted(by.items())}}


def save_slots(H: SlotH, path: str, extra: dict | None = None):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sd = {k: v.detach().cpu().clone() for k, v in H.state_dict().items()}
    torch.save({"format": FORMAT, "config": H.cfg, "state_dict": sd, "extra": extra or {}}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_slots(path: str, itf):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert blob.get("format") == FORMAT, blob.get("format")
    H = SlotH(itf, **blob["config"])
    H.load_state_dict(blob["state_dict"])
    return H, blob.get("extra") or {}


__all__ = ["SlotH", "attach_slots", "slot_stats_summary", "save_slots", "load_slots", "FORMAT"]
