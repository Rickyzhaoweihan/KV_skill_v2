"""D10 (fallback): ENCODER-SIDE addressing. The encoder (2B) reads the document AND the question natively, so its
own queries address its own keys (no query map A). At each paired layer (reader full-attention ℓ <- encoder
full-attention ℓ') the encoder's attention output at the question tokens, o2[t], is carried into the reader by a
value-space map B_ℓ and combined with the reader's own attention output at the same (token-aligned) position:

    level "pre" : o2 = per-head attention output before the output gate / o_proj, concatenated  [Hq_e·hd]
                  reader: o = combine(o_own_pre [Hq_r·hd], B o2)   then the reader's own gate and o_proj
    level "post": o2 = the attention module's output (post o_proj)                               [d_e]
                  reader: out = combine(out_own [d_r], B o2)
    combine: replace (o = B o2) | add (o = o_own + g·B o2) | mix (o = (1−σ(c))·o_own + σ(c)·B o2)

B = None means identity (same model). With the reader as its own encoder, B = I and "replace", the patched layers
output exactly what the reader computes with the document in context (the L/L anchor).
The caller sets `R.cur = {li: tensor [B,T,d_in]}` aligned with the tokens of the next reader forward.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn

import xenc  # noqa: F401


class EncSideInterface(nn.Module):
    def __init__(self, layer_map: dict, d_in: int, d_out: int, mode: str = "replace", level: str = "pre",
                 identity: bool = False, gate_init: float = 1.0, mix_init: float = 0.0, mlp_hidden: int = 0):
        super().__init__()
        self.layer_map = {int(k): int(v) for k, v in layer_map.items()}
        self.layers = sorted(self.layer_map)
        self.mode, self.level, self.identity = mode, level, bool(identity)
        self.d_in, self.d_out = d_in, d_out
        self.B = nn.ParameterDict()
        if not identity:
            for li in self.layers:
                self.B[str(li)] = nn.Parameter(torch.eye(d_out, d_in))
        self.mlp_hidden = int(mlp_hidden)
        self.M = nn.ModuleDict()           # optional nonlinear correction: y = B x + W2 gelu(W1 x)
        if self.mlp_hidden and not identity:
            for li in self.layers:
                m = nn.Sequential(nn.Linear(d_in, self.mlp_hidden), nn.GELU(), nn.Linear(self.mlp_hidden, d_out))
                nn.init.zeros_(m[2].weight)
                nn.init.zeros_(m[2].bias)
                self.M[str(li)] = m
        self.g = nn.Parameter(torch.full((len(self.layers),), float(gate_init)))
        self.c = nn.Parameter(torch.full((len(self.layers),), float(mix_init)))
        self._i = {li: i for i, li in enumerate(self.layers)}

    def map(self, li, o2):
        if self.identity:
            return o2
        y = o2 @ self.B[str(li)].t().to(o2.dtype)
        if self.mlp_hidden:
            y = y + self.M[str(li)](o2.float()).to(y.dtype)
        return y

    def combine(self, li, own, o2):
        y = self.map(li, o2.to(own.device).float())
        own_f = own.float()
        mode = getattr(self, "layer_mode", {}).get(li, self.mode)
        if mode == "replace":
            out = y * getattr(self, "rgain", {}).get(li, 1.0)      # D12: optional per-layer scalar gain
        elif mode == "add":
            out = own_f + self.g[self._i[li]] * y
        else:
            w = torch.sigmoid(self.c[self._i[li]])
            out = (1 - w) * own_f + w * y
        return out.to(own.dtype)


class EncSideReader:
    def __init__(self, model, itf: EncSideInterface):
        from deltaskill.backbone_utils import decoder_layers
        self.model, self.itf = model, itf
        L = decoder_layers(model)
        self.mods = {li: L[li].self_attn for li in itf.layers}
        self.cur: dict | None = None
        self.enabled = True
        self.own_rows: dict = {}          # optional: record the reader's own (pre-combine) output rows
        self.record = False
        self.capture: dict | None = None  # {li: own pre-gate attention output [B,T,Hq·hd]} (L/L encoder / fidelity)
        self._orig = {}
        for li, m in self.mods.items():
            self._orig[li] = m.forward
            m.forward = types.MethodType(_make_forward(self, li), m)

    @property
    def active(self):
        return self.enabled and self.cur is not None

    def detach(self):
        for li, f in self._orig.items():
            self.mods[li].forward = f
        self._orig.clear()


def _make_forward(R: EncSideReader, li: int):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        if not R.active and R.capture is None:
            return type(self).forward(self, hidden_states, position_embeddings, attention_mask,
                                      past_key_values=past_key_values, **kwargs)
        itf = R.itf
        o2 = R.cur[li] if R.active else None
        if itf.level == "post" and R.active:
            out, w = type(self).forward(self, hidden_states, position_embeddings, attention_mask,
                                        past_key_values=past_key_values, **kwargs)
            if R.record:
                R.own_rows[li] = out.detach()
            return itf.combine(li, out, o2), w
        input_shape = hidden_states.shape[:-1]
        T = input_shape[1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        q = self.q_norm(q.view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        Kr, Vr = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
        o = torch.nn.functional.scaled_dot_product_attention(q, Kr, Vr, is_causal=k.shape[2] == T and T > 1,
                                                             scale=self.scaling)
        own = o.transpose(1, 2).reshape(*input_shape, -1)
        if R.capture is not None:
            R.capture[li] = own.detach()
        o = itf.combine(li, own, o2) if R.active else own
        return self.o_proj(o * torch.sigmoid(gate)), None

    return forward


class EncCapture:
    """Record an encoder's attention outputs at given layers: level pre (per-head, pre-gate, concat) or post."""

    def __init__(self, enc, layers, level: str = "pre"):
        from xenc.xattn_read import AttnReader, attn_dims, make_attn_interface
        self.level, self.layers = level, sorted(set(int(x) for x in layers))
        self.out: dict = {}
        self._h = []
        if level == "pre":
            self.ar = AttnReader(enc, make_attn_interface(enc, attn_dims(enc), layers=self.layers), record_mag=False)
        else:
            from deltaskill.backbone_utils import decoder_layers
            L = decoder_layers(enc)
            self.ar = None
            for li in self.layers:
                self._h.append(L[li].self_attn.register_forward_hook(self._hook(li)))

    def _hook(self, li):
        def h(mod, inp, out):
            self.out[li] = (out[0] if isinstance(out, tuple) else out).detach()
        return h

    def start(self):
        self.out = {}
        if self.ar is not None:
            self.ar.capture = {}

    def get(self) -> dict:
        """{enc_layer: [B,T,d]}"""
        if self.ar is not None:
            res = {}
            for li, (_, o) in self.ar.capture.items():
                B_, H, T, d = o.shape
                res[li] = o.transpose(1, 2).reshape(B_, T, H * d)
            self.ar.capture = None
            return res
        return dict(self.out)

    def detach(self):
        for h in self._h:
            h.remove()
        if self.ar is not None:
            self.ar.detach()


__all__ = ["EncSideInterface", "EncSideReader", "EncCapture"]
