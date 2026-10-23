"""Interface helpers for the encoder->reader setting: build / save / load the per-reader
`deltaskill.interface.ModelInterface` (d_m = reader hidden, d_s = encoder hidden), an optional
per-encoder `EncoderLens`, optimizer param groups, and parameter counts.

The branch is a no-op at init (B zero-init inside ModelInterface) — nothing here changes that.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

import xenc  # noqa: F401  (sys.path bootstrap for deltaskill)
from deltaskill.interface import ModelInterface
from xenc.selector import LayerSelector

FORMAT = "xenc.interface.v1"


def make_interface(d_reader: int, d_encoder: int, reader_layers: list[int], rank: int = 0,
                   gate_bias_init: float = -2.0, gain_init: float = 0.0) -> ModelInterface:
    layers = [int(li) for li in reader_layers]
    if len(set(layers)) != len(layers):
        raise ValueError(f"duplicate reader layers: {layers}")
    return ModelInterface(d_m=int(d_reader), d_s=int(d_encoder), inject_layers=layers,
                          gate_bias_init=gate_bias_init, gain_init=gain_init, rank=int(rank or 0))


class EncoderLens(nn.Module):
    """Per-encoder linear projection d_enc -> d_core (orthogonal init, no bias), on the last dim."""

    def __init__(self, d_enc: int, d_core: int):
        super().__init__()
        self.d_enc, self.d_core = int(d_enc), int(d_core)
        self.proj = nn.Linear(self.d_enc, self.d_core, bias=False)
        nn.init.orthogonal_(self.proj.weight)

    def forward(self, K: torch.Tensor) -> torch.Tensor:
        if K.shape[-1] != self.d_enc:
            raise ValueError(f"EncoderLens expects last dim {self.d_enc}, got {K.shape[-1]}")
        return self.proj(K.to(self.proj.weight.dtype))


def interface_param_groups(itf: nn.Module, lr: float, selector: LayerSelector | None = None,
                           selector_lr: float | None = None, lr_A: float | None = None) -> list[dict]:
    """AdamW param groups: [{interface params, lr}] (+ [{selector params, selector_lr or lr}]).
    lr_A (D5): if set, the query map `itf.A` gets its own group at lr_A; the rest stays at lr."""
    if lr_A is not None and hasattr(itf, "A"):
        a_ids = {id(p) for p in itf.A.parameters()}
        groups = [{"params": [p for p in itf.parameters() if p.requires_grad and id(p) not in a_ids],
                   "lr": lr, "name": "interface"},
                  {"params": [p for p in itf.A.parameters() if p.requires_grad], "lr": float(lr_A),
                   "name": "interface_A"}]
        groups = [g for g in groups if g["params"]]
    else:
        groups = [{"params": [p for p in itf.parameters() if p.requires_grad], "lr": lr,
                   "name": "interface"}]
    if selector is not None:
        seen = {id(p) for g in groups for p in g["params"]}
        sp = [p for p in selector.parameters() if p.requires_grad and id(p) not in seen]
        if sp:
            groups.append({"params": sp, "lr": lr if selector_lr is None else selector_lr,
                           "name": "selector"})
    return groups


def _itf_config(itf: ModelInterface) -> dict:
    return {"d_m": itf.d_m, "d_s": itf.d_s, "inject_layers": list(itf.inject_layers),
            "rank": int(itf.rank or 0)}


def save_interface(itf: ModelInterface, path: str, selector: LayerSelector | None = None,
                   extra: dict | None = None) -> None:
    blob = {"format": FORMAT, "config": _itf_config(itf),
            "state_dict": {k: v.detach().cpu() for k, v in itf.state_dict().items()},
            "selector": None, "extra": extra}
    if selector is not None:
        blob["selector"] = {"config": selector.config(),
                            "state_dict": {k: v.detach().cpu() for k, v in selector.state_dict().items()}}
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def load_interface(path: str, d_reader: int, d_encoder: int, reader_layers: list[int],
                   map_location="cpu"):
    """-> (itf, selector_or_None, extra). Validates dims/layers against the saved config."""
    blob = torch.load(path, map_location=map_location, weights_only=False)
    cfg = blob["config"]
    want = (int(d_reader), int(d_encoder), [int(li) for li in reader_layers])
    got = (cfg["d_m"], cfg["d_s"], list(cfg["inject_layers"]))
    if want != got:
        raise ValueError(f"interface at {path} has (d_m, d_s, layers)={got}, expected {want}")
    itf = make_interface(d_reader, d_encoder, reader_layers, rank=cfg.get("rank", 0))
    itf.load_state_dict(blob["state_dict"], strict=True)
    selector = None
    if blob.get("selector") is not None:
        sc = blob["selector"]["config"]
        selector = LayerSelector(sc["d_m"], sc["cand_layers"], k=sc["k"],
                                 straight_through=sc.get("straight_through", False))
        selector.load_state_dict(blob["selector"]["state_dict"], strict=True)
    return itf, selector, blob.get("extra")


def count_params(itf: nn.Module, selector: LayerSelector | None = None) -> dict:
    n_itf = sum(p.numel() for p in itf.parameters())
    out = {"interface": n_itf}
    for name in ("A", "B"):
        m = getattr(itf, name, None)
        if isinstance(m, nn.Module):
            out[name] = sum(p.numel() for p in m.parameters())
    out["gate_gain"] = n_itf - out.get("A", 0) - out.get("B", 0)
    out["selector"] = sum(p.numel() for p in selector.parameters()) if selector is not None else 0
    out["total"] = n_itf + out["selector"]
    return out
