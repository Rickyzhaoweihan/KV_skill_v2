"""D10: cross-model read of an encoder's Gated-DeltaNet states inside the reader's own DeltaNet layers.

The document KV-Skill is the encoder's recurrent states S_e,ℓ' ∈ R^{Hv_e × dk × dv} (xenc/state_skill.py),
harvested once and fixed. For each reader linear-attention layer ℓ (assigned an encoder linear layer ℓ' by
normalized depth) a separate read branch is added inside the reader's Qwen3_5GatedDeltaNet:

    q_ℓ   = l2norm(conv(in_proj_qkv(h)))_q        the reader's own per-k-head query (as its kernel uses it)
    q'    = l2norm_head(A_ℓ q_ℓ) / √dk            A: reader k-head space -> encoder k-head space
    o     = S_e,ℓ'ᵀ q'                            the exact delta-rule read of the encoder state
                                                 (encoder v-head j reads with its k-head j // (Hv_e/Hk_e))
    y     = g_ℓ · B_ℓ(o) [· exp(Σ_{s≤t} g^dec_s)] B: encoder v-head space -> reader v-head space; optional
                                                 decay by the reader's own cumulative gate (what an in-context
                                                 state would have undergone over the question tokens)
    core_attn_out += y                            before the reader's gated RMSNorm / out_proj

Only A, B, g train (registration); the backbone is frozen and the states are fixed. For encoder == reader
(L/L) A = B = I, g = 1 is the "native" initialisation (a valid read of the reader's own document state).

Map parameterisations ("full" | "head" | "none"):
    A full: [Hk_e·dk, Hk_r·dk]          A head: per encoder k-head a dk×dk block reading reader k-head e·Hk_r//Hk_e
    B full: [Hv_r·dv, Hv_e·dv]          B head: per reader v-head a dv×dv block reading encoder v-head h·Hv_e//Hv_r
    none  : the structural identity (head assignment, no parameters)
Identity init = the structural identity above (exact identity when the dims match).
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import xenc  # noqa: F401
from xenc.state_skill import depth_map, gdn_modules, state_dims

FORMAT = "xenc.state_interface.v1"


def _l2(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _assign(n_out: int, n_in: int) -> list[int]:
    return [h * n_in // n_out for h in range(n_out)]


def _struct_eye(n_out: int, n_in: int, d: int) -> torch.Tensor:
    """[n_out·d, n_in·d] block matrix with I_d at (h, assign(h))."""
    W = torch.zeros(n_out * d, n_in * d)
    for h, j in enumerate(_assign(n_out, n_in)):
        W[h * d:(h + 1) * d, j * d:(j + 1) * d] = torch.eye(d)
    return W


class StateInterface(nn.Module):
    """Per-reader-linear-layer (A_ℓ, B_ℓ, g_ℓ). `layer_map`: reader layer -> encoder layer."""

    def __init__(self, layer_map: dict, r_dims: dict, e_dims: dict, a_mode: str = "full", b_mode: str = "full",
                 gate_init: float = 0.1, decay: bool = True):
        super().__init__()
        self.layer_map = {int(k): int(v) for k, v in layer_map.items()}
        self.layers = sorted(self.layer_map)
        self.r, self.e = dict(r_dims), dict(e_dims)
        self.a_mode, self.b_mode, self.decay = a_mode, b_mode, bool(decay)
        r, e = self.r, self.e
        if r["dk"] != e["dk"] and (a_mode != "full"):
            raise ValueError("A head/none needs equal key head dims")
        if r["dv"] != e["dv"] and (b_mode != "full"):
            raise ValueError("B head/none needs equal value head dims")
        self.a_assign = _assign(e["n_k"], r["n_k"])          # encoder k-head <- reader k-head
        self.b_assign = _assign(r["n_v"], e["n_v"])          # reader v-head <- encoder v-head
        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        for li in self.layers:
            k = str(li)
            if a_mode == "full":
                W = (_struct_eye(e["n_k"], r["n_k"], r["dk"]) if r["dk"] == e["dk"]
                     else torch.eye(e["n_k"] * e["dk"], r["n_k"] * r["dk"]))
                self.A[k] = nn.Parameter(W)
            elif a_mode == "head":
                self.A[k] = nn.Parameter(torch.eye(r["dk"]).repeat(e["n_k"], 1, 1))
            if b_mode == "full":
                W = (_struct_eye(r["n_v"], e["n_v"], e["dv"]) if r["dv"] == e["dv"]
                     else torch.eye(r["n_v"] * r["dv"], e["n_v"] * e["dv"]))
                self.B[k] = nn.Parameter(W)
            elif b_mode == "head":
                self.B[k] = nn.Parameter(torch.eye(e["dv"]).repeat(r["n_v"], 1, 1))
        self.g = nn.Parameter(torch.full((len(self.layers),), float(gate_init)))
        self._gi = {li: i for i, li in enumerate(self.layers)}

    # ---- config / io
    def config(self) -> dict:
        return {"layer_map": self.layer_map, "r_dims": {k: v for k, v in self.r.items() if k != "linear"},
                "e_dims": {k: v for k, v in self.e.items() if k != "linear"}, "a_mode": self.a_mode,
                "b_mode": self.b_mode, "decay": self.decay}

    def gate(self, li: int) -> torch.Tensor:
        return self.g[self._gi[li]]

    # ---- maps
    def map_query(self, li: int, qn: torch.Tensor) -> torch.Tensor:
        """qn [B,T,Hk_r,dk] (l2-normed) -> [B,T,Hk_e,dk] (re-l2-normed per head, NOT yet scaled)."""
        B_, T = qn.shape[:2]
        e = self.e
        if self.a_mode == "full":
            W = self.A[str(li)]
            q2 = (qn.reshape(B_, T, -1) @ W.t()).reshape(B_, T, e["n_k"], e["dk"])
        elif self.a_mode == "head":
            q2 = torch.einsum("bthk,hjk->bthj", qn[:, :, self.a_assign], self.A[str(li)])
        else:
            q2 = qn[:, :, self.a_assign]
        return _l2(q2)

    def map_out(self, li: int, o: torch.Tensor) -> torch.Tensor:
        """o [B,T,Hv_e,dv] -> [B,T,Hv_r,dv_r]."""
        B_, T = o.shape[:2]
        r = self.r
        if self.b_mode == "full":
            W = self.B[str(li)]
            return (o.reshape(B_, T, -1) @ W.t()).reshape(B_, T, r["n_v"], r["dv"])
        if self.b_mode == "head":
            return torch.einsum("bthv,hwv->bthw", o[:, :, self.b_assign], self.B[str(li)])
        return o[:, :, self.b_assign]

    def read(self, li: int, qn: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Branch output before gate/decay: B(Sᵀ A q). S [Hv_e,dk,dv] fp32."""
        q2 = self.map_query(li, qn.float()) * (self.e["dk"] ** -0.5)
        rep = self.e["n_v"] // self.e["n_k"]
        if rep > 1:
            q2 = q2.repeat_interleave(rep, dim=2)
        o = torch.einsum("bthk,hkv->bthv", q2, S)
        return self.map_out(li, o)


def make_state_interface(reader, encoder, a_mode="full", b_mode="full", gate_init=0.1, decay=True,
                         layers: list[int] | None = None) -> StateInterface:
    rd, ed = state_dims(reader), state_dims(encoder)
    lm = depth_map(rd["linear"], rd["n_layers"], ed["linear"], ed["n_layers"])
    if layers:
        lm = {k: v for k, v in lm.items() if k in set(int(x) for x in layers)}
    return StateInterface(lm, rd, ed, a_mode, b_mode, gate_init, decay)


def save_state_interface(itf: StateInterface, path: str, extra: dict | None = None) -> None:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"format": FORMAT, "config": itf.config(), "state_dict": {k: v.detach().cpu() for k, v in itf.state_dict().items()},
                "extra": extra or {}}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_state_interface(path: str):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    c = blob["config"]
    itf = StateInterface(c["layer_map"], c["r_dims"], c["e_dims"], c["a_mode"], c["b_mode"], 0.0, c["decay"])
    itf.load_state_dict(blob["state_dict"])
    return itf, blob.get("extra") or {}


# ============================================================================ the reader patch
class StateReader:
    """Patches each mapped reader Qwen3_5GatedDeltaNet.forward to add the encoder-state read branch.

    load(states) with a harvest dict (recurrent keyed by ENCODER layer); unload() turns the branch off
    (forward == stock HF forward); detach() restores the original forward methods."""

    def __init__(self, model, itf: StateInterface, record_mag: bool = True):
        self.model, self.itf = model, itf
        self.mods = gdn_modules(model)
        self.S: dict | None = None
        self.enabled = True
        self.record_mag = record_mag
        self.mag_terms: list = []
        self.gain_scale = 1.0
        self.capture: dict | None = None     # fit_state_maps: {li: (q_own [B,T,Hk,dk], core [B,T,Hv,dv])}
        self._cum: dict = {}
        self._orig = {}
        for li in itf.layers:
            m = self.mods[li]
            self._orig[li] = m.forward
            m.forward = types.MethodType(_make_forward(self, li), m)

    # ---- operator
    def load(self, states: dict | None, device=None):
        if states is None:
            self.S = None
            return
        dev = device or next(self.itf.parameters()).device
        need = set(self.itf.layer_map.values())
        rec = states["recurrent"] if "recurrent" in states else states
        self.S = {int(k): v.to(dev, torch.float32) for k, v in rec.items() if int(k) in need}
        missing = need - set(self.S)
        if missing:
            raise KeyError(f"states lack encoder layers {sorted(missing)}")

    def unload(self):
        self.S = None
        self._cum.clear()

    def detach(self):
        for li, f in self._orig.items():
            self.mods[li].forward = f
        self._orig.clear()

    @property
    def active(self) -> bool:
        return self.enabled and self.S is not None

    # ---- the branch (called from the patched forward)
    def branch(self, li: int, qn, g_log, core, fresh: bool, keep_cum: bool):
        itf = self.itf
        S = self.S[itf.layer_map[li]]
        y = itf.read(li, qn.to(S.device), S) * itf.gate(li) * self.gain_scale        # [B,T,Hv_r,dv]
        if itf.decay:
            c = torch.cumsum(g_log.float().to(y.device), dim=1)                         # [B,T,Hv_r]
            if not fresh and li in self._cum:
                c = c + self._cum[li].unsqueeze(1)
            if keep_cum:
                self._cum[li] = c[:, -1].detach()
            y = y * c.exp().unsqueeze(-1)
        if self.record_mag and torch.is_grad_enabled():
            rel = y.float().norm(dim=-1) / (core.float().norm(dim=-1).to(y.device) + 1e-6)
            self.mag_terms.append(rel.mean())
        return y.to(core.device, core.dtype)


def _make_forward(R: StateReader, li: int):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        """Copy of Qwen3_5GatedDeltaNet.forward (transformers 5.13) + the encoder-state read branch."""
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        if use_precomputed_states and seq_len == 1:
            mixed_qkv = self.causal_conv1d_update(mixed_qkv, conv_state, self.conv1d.weight.squeeze(1),
                                                  self.conv1d.bias, self.activation)
        else:
            if use_precomputed_states:
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            if cache_params is not None:
                new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.update_conv_state(new_conv_state, self.layer_idx)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(x=mixed_qkv, weight=self.conv1d.weight.squeeze(1),
                                                  bias=self.conv1d.bias, activation=self.activation,
                                                  seq_idx=kwargs.get("seq_idx"))
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
            if use_precomputed_states:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        q_own = query                                              # [B,T,Hk_r,dk] (pre-norm, pre-repeat)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=recurrent_state,
                output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True)
        else:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.get("cu_seq_lens_q"))
        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)
        if R.capture is not None:
            R.capture[li] = (q_own.detach(), core_attn_out.detach())
        if R.active:                                               # ---- D10 branch
            qn = _l2(q_own.float())
            y = R.branch(li, qn, g, core_attn_out, fresh=not use_precomputed_states,
                         keep_cum=cache_params is not None)
            core_attn_out = core_attn_out + y
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)

    return forward


__all__ = ["StateInterface", "StateReader", "make_state_interface", "save_state_interface", "load_state_interface"]
