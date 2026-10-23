"""D10: the combined cross-model document read = DeltaNet-state branch (xenc/state_interface.py) and/or
full-attention K/V branch (xenc/xattn_read.py) behind one load/unload/save API.

    R = build_d10(model, branches={"state","attn"}, enc_state_dims=..., enc_attn_dims=..., cfg=...)
    R.load(harvest_dict)   # needs "recurrent" for state, "kv"+"n_doc" for attn
    R.unload(); R.enabled = False; R.save(path, extra); R = D10Read.from_file(path, model)
"""

from __future__ import annotations

import os

import torch

import xenc  # noqa: F401
from xenc.state_interface import StateInterface, StateReader, make_state_interface
from xenc.xattn_read import AttnInterface, AttnReader, make_attn_interface

FORMAT = "xenc.d10_read.v1"


class D10Read:
    def __init__(self, model, state_itf: StateInterface | None = None, attn_itf: AttnInterface | None = None,
                 record_mag: bool = True):
        self.model = model
        self.state_itf, self.attn_itf = state_itf, attn_itf
        self.sr = StateReader(model, state_itf, record_mag) if state_itf is not None else None
        self.ar = AttnReader(model, attn_itf, record_mag) if attn_itf is not None else None
        self._enabled = True

    @property
    def readers(self):
        return [r for r in (self.sr, self.ar) if r is not None]

    @property
    def branches(self) -> list[str]:
        return [n for n, r in (("state", self.sr), ("attn", self.ar)) if r is not None]

    @property
    def needs_kv(self) -> bool:
        return self.ar is not None

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, v):
        self._enabled = bool(v)
        for r in self.readers:
            r.enabled = bool(v)

    def load(self, st):
        for r in self.readers:
            r.load(st)

    def unload(self):
        for r in self.readers:
            r.unload()

    def detach(self):
        for r in self.readers:
            r.detach()

    @property
    def mag_terms(self) -> list:
        return [t for r in self.readers for t in r.mag_terms]

    def clear_mag(self):
        for r in self.readers:
            r.mag_terms.clear()

    def set_gain(self, g: float):
        """State branch: scales y. Attention branch: multiplies the document's attention odds (log g added to
        the document log-mass), i.e. a soft gate that cannot exceed the native mixture."""
        import math
        if self.sr is not None:
            self.sr.gain_scale = g
        if self.ar is not None:
            self.ar.bias_shift = math.log(max(g, 1e-6))

    def param_groups(self, lr: float, lr_g: float, train_maps: bool = True) -> list[dict]:
        gates, maps = [], []
        if self.state_itf is not None:
            gates.append(self.state_itf.g)
            maps += list(self.state_itf.A.parameters()) + list(self.state_itf.B.parameters())
        if self.attn_itf is not None:
            gates.append(self.attn_itf.b)
            maps += list(self.attn_itf.A.parameters()) + list(self.attn_itf.B.parameters())
        for p in maps:
            p.requires_grad_(bool(train_maps))
        groups = [{"params": gates, "lr": lr_g}]
        if train_maps and maps:
            groups.append({"params": maps, "lr": lr})
        return groups

    def gates(self) -> dict:
        out = {}
        if self.state_itf is not None:
            out["state_g"] = [round(float(x), 4) for x in self.state_itf.g.detach().cpu()]
        if self.attn_itf is not None:
            out["attn_b"] = [round(float(x), 4) for x in self.attn_itf.b.detach().cpu()]
        return out

    def save(self, path: str, extra: dict | None = None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        blob = {"format": FORMAT, "extra": extra or {}}
        for n, itf in (("state", self.state_itf), ("attn", self.attn_itf)):
            if itf is not None:
                blob[n] = {"config": itf.config(), "state_dict": {k: v.detach().cpu() for k, v in itf.state_dict().items()}}
        torch.save(blob, path + ".tmp")
        os.replace(path + ".tmp", path)

    @staticmethod
    def interfaces_from_file(path: str):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        si = ai = None
        if blob.get("state"):
            c = blob["state"]["config"]
            si = StateInterface(c["layer_map"], c["r_dims"], c["e_dims"], c["a_mode"], c["b_mode"], 0.0, c["decay"])
            si.load_state_dict(blob["state"]["state_dict"])
        if blob.get("attn"):
            c = blob["attn"]["config"]
            ai = AttnInterface(c["layer_map"], c["r"], c["e"], c["a_mode"], c["b_mode"])
            ai.load_state_dict(blob["attn"]["state_dict"])
        return si, ai, blob.get("extra") or {}


def build_d10(model, branches, enc_state_dims=None, enc_attn_dims=None, state_kw=None, attn_kw=None,
              maps: str | None = None, device=None, record_mag: bool = True) -> D10Read:
    """Fresh interfaces (identity / structural init), optionally overwritten by ridge maps (fit_state_maps.py)."""
    from xenc.state_skill import depth_map, state_dims
    si = ai = None
    blob = torch.load(maps, map_location="cpu", weights_only=False) if maps else {}
    if "state" in branches:
        kw = dict(state_kw or {})
        layers = kw.pop("layers", None)
        rd = state_dims(model)
        lm = depth_map(rd["linear"], rd["n_layers"], enc_state_dims["linear"], enc_state_dims["n_layers"])
        if layers:
            lm = {k: v for k, v in lm.items() if k in set(layers)}
        si = StateInterface(lm, rd, enc_state_dims, **kw)
        with torch.no_grad():
            for li in si.layers:
                if si.a_mode == "full" and str(li) in blob.get("A", {}):
                    si.A[str(li)].copy_(blob["A"][str(li)].float())
                if si.b_mode == "full" and str(li) in blob.get("B", {}):
                    si.B[str(li)].copy_(blob["B"][str(li)].float())
    if "attn" in branches:
        kw = dict(attn_kw or {})
        ai = make_attn_interface(model, enc_attn_dims, **kw)
        with torch.no_grad():
            for li in ai.layers:
                if ai.a_mode in ("head", "mix") and str(li) in blob.get("attn_A", {}) and \
                        tuple(blob["attn_A"][str(li)].shape) == tuple(ai.A[str(li)].shape):
                    ai.A[str(li)].copy_(blob["attn_A"][str(li)].float())
                if ai.b_mode == "head" and str(li) in blob.get("attn_B", {}):
                    ai.B[str(li)].copy_(blob["attn_B"][str(li)].float())
    if device is not None:
        si = si.to(device) if si is not None else None
        ai = ai.to(device) if ai is not None else None
    return D10Read(model, si, ai, record_mag)


def load_d10(path: str, model, device=None, record_mag: bool = False) -> tuple[D10Read, dict]:
    si, ai, extra = D10Read.interfaces_from_file(path)
    if device is not None:
        si = si.to(device) if si is not None else None
        ai = ai.to(device) if ai is not None else None
    return D10Read(model, si, ai, record_mag), extra


__all__ = ["D10Read", "build_d10", "load_d10"]
