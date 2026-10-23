"""D7: native-attention read of a fixed, one-prefill document operator.

The operator is the paper's factorized form M = W Uᵀ, fixed, position-free and OUTSIDE the attention cache:
    U^(ℓ) = the encoder's attention keys at the full-attention layer nearest ρ (pre-RoPE, post k_norm),
    W^(ℓ) = the matching value states (v_proj output), per KV head (harvest="attn_raw": no row L2).
The read is a separate residual branch at the reader's full-attention layer ℓ:

    x     = the input of the layer's self-attention (post input_layernorm), captured by a pre-hook
    q_h   = q_norm(q_proj(x))_h                         reader's OWN query map (A), per query head
    s_h,i = (q_h · U_{kv(h),i}) · scaling · exp(τ_ℓ)    no RoPE on either side (content attention),
                                                       GQA: query head h reads KV head h // groups
    a_h   = softmax([s_h, b_ℓ])  (one learned null logit b_ℓ: the read may decline; null value = 0)
    o     = concat_h(a_h · W_{kv(h)}) ⊙ σ(gate)        (Qwen3.5's gated-attention output gate)
    h'    = h + γ_ℓ g_ℓ · o_proj(o)                    reader's OWN output map (B), added to the layer output

For encoder == reader (L/L) U, W live in the reader's head space, so A and B need no training: the branch
is the reader attending over the document's K/V at that layer. Trainable: per-layer null logit b, gate g,
log-gain log γ, log-temperature τ (4 scalars per layer). For encoder != reader (S/L) per-layer linear
maps U' = M_ℓ U + c, W' = N_ℓ W + d (encoder KV space -> reader KV space; ridge-initialised by
scripts/fit_kv_map.py) form the cross-model interface and may be fine-tuned.

API mirrors xenc.injector.LibraryInjector (load_library / unload / enabled / mag_terms / route_terms /
attn_terms + training_attn + attn_pos / record_attn + last_attn) so register_doc / eval_doc / probe_read
drive it unchanged.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

import xenc  # noqa: F401
from deltaskill.backbone_utils import decoder_layers
from xenc.depth_skill import _key, attn_layer_map

FORMAT = "xenc.native_attn.v1"


def _on(t, like):
    return t if t.device == like.device else t.to(like.device)


# ============================================================================ library
class NativeAttnLibrary:
    """Raw per-document attention K/V keyed by READER full-attention layer.

        K[li], V[li] : [C, Nc, d_src]  zero-padded chunk slots (d_src = encoder n_kv·head_dim)
        mask[li]     : [C, Nc] bool
    Shape-compatible with xenc.library.Library where gold_slots / eval_doc touch it (K, mask, V,
    chunk_meta, subset, _summaries)."""

    def __init__(self, K: dict, V: dict, mask: dict, chunk_meta: list, rho_of: dict):
        self.K, self.V, self.mask = K, V, mask
        self.chunk_meta = chunk_meta
        self.rho_of = rho_of
        self.S = {}
        self.center = None

    # ---- construction
    @staticmethod
    def from_doc_cache(cache: dict, reader_model, depths=None, read_layers=None, device="cpu",
                       dtype=torch.float32) -> "NativeAttnLibrary":
        depths = sorted(float(r) for r in (depths or cache["depths"]))
        lmap = attn_layer_map(reader_model, depths)          # rho -> reader full-attention layer
        if len(set(lmap.values())) != len(lmap):
            raise ValueError(f"two depths map to the same reader attention layer: {lmap}")
        if read_layers:
            lmap = {r: li for r, li in lmap.items() if li in set(int(x) for x in read_layers)}
        chunks = cache["chunks"]
        if chunks[0].get("V") is None:
            raise ValueError("native_attn needs an untied attention harvest (encode with --harvest attn_raw)")
        K, V, M = {}, {}, {}
        for rho, li in lmap.items():
            rows = [c["K"][_key(c["K"], rho)] for c in chunks]
            vals = [c["V"][_key(c["V"], rho)] for c in chunks]
            n = [r.shape[0] for r in rows]
            d = rows[0].shape[-1]
            Kt = torch.zeros(len(rows), max(n), d, dtype=dtype, device=device)
            Vt = torch.zeros_like(Kt)
            Mt = torch.zeros(len(rows), max(n), dtype=torch.bool, device=device)
            for c in range(len(rows)):
                Kt[c, : n[c]] = rows[c].to(device=device, dtype=dtype)
                Vt[c, : n[c]] = vals[c].to(device=device, dtype=dtype)
                Mt[c, : n[c]] = True
            K[li], V[li], M[li] = Kt, Vt, Mt
        meta = [{"doc_id": cache.get("doc_id", ""), "start": int(c.get("start", 0)),
                 "n_tokens": int(c.get("n_tokens", 0)), "encoder": cache.get("encoder", "")} for c in chunks]
        return NativeAttnLibrary(K, V, M, meta, {li: r for r, li in lmap.items()})

    # ---- Library-compatible helpers
    def _summaries(self) -> None:
        self.S = {}

    @property
    def layers(self):
        return sorted(self.K)

    @property
    def d_s(self) -> int:
        return next(iter(self.K.values())).shape[-1]

    @property
    def n_chunks(self) -> int:
        return next(iter(self.K.values())).shape[0]

    def to(self, device=None, dtype=None) -> "NativeAttnLibrary":
        for li in list(self.K):
            self.K[li] = self.K[li].to(device=device, dtype=dtype)
            self.V[li] = self.V[li].to(device=device, dtype=dtype)
            self.mask[li] = self.mask[li].to(device=device)
        return self

    def subset(self, idx) -> "NativeAttnLibrary":
        idx = [int(i) for i in (idx if hasattr(idx, "__iter__") else [idx])]
        if not idx:
            raise ValueError("empty subset")
        K, V, M = {}, {}, {}
        for li in self.K:
            it = torch.tensor(idx, device=self.K[li].device)
            m = self.mask[li].index_select(0, it)
            n = int(m.sum(-1).max())
            M[li] = m[:, :n].clone()
            K[li] = self.K[li].index_select(0, it)[:, :n].clone()
            V[li] = self.V[li].index_select(0, it)[:, :n].clone()
        return NativeAttnLibrary(K, V, M, [self.chunk_meta[i] for i in idx], dict(self.rho_of))

    def null_like(self, mode: str = "random", seed: int = 0) -> "NativeAttnLibrary":
        """Content-free operator with the same slots: Gaussian rows rescaled to each real row's norm
        (keys and values separately), or all-zero (zero ⇒ uniform read of zero values ⇒ Δ = 0)."""
        g = torch.Generator().manual_seed(seed)
        K, V = {}, {}
        for li in self.K:
            m = self.mask[li].unsqueeze(-1).to(self.K[li].dtype)
            out = []
            for X in (self.K[li], self.V[li]):
                if mode == "zero":
                    out.append(torch.zeros_like(X))
                    continue
                R = torch.randn(X.shape, generator=g).to(X.device, X.dtype)
                R = R / (R.norm(dim=-1, keepdim=True) + 1e-6) * X.norm(dim=-1, keepdim=True)
                out.append(R * m)
            K[li], V[li] = out
        return NativeAttnLibrary(K, V, {li: m.clone() for li, m in self.mask.items()}, list(self.chunk_meta),
                                 dict(self.rho_of))


class NativeSpec:
    """Attached to a DocRuntime as `rt.native`: turns a doc cache into a NativeAttnLibrary."""

    def __init__(self, reader_model, depths, read_layers=None):
        self.reader = reader_model
        self.depths = sorted(float(r) for r in depths)
        full = attn_layer_map(reader_model, self.depths)
        self.layers = sorted(li for li in full.values() if not read_layers or li in set(read_layers))
        self.rho_of = {li: r for r, li in full.items() if li in self.layers}

    def library(self, cache: dict) -> NativeAttnLibrary:
        return NativeAttnLibrary.from_doc_cache(cache, self.reader, self.depths, self.layers)


# ============================================================================ interface
class NativeAttnInterface(nn.Module):
    """Per-layer scalars (null logit b, gate g, log-gain, log-temperature) + optional KV maps
    (encoder KV space d_src -> reader KV space d_dst) for cross-model reads."""

    def __init__(self, layers, d_src: int, d_dst: int, maps: bool = False, null_init: float = -30.0,
                 gate_init: float = 1.0, gain_init: float = 0.0, temp_init: float = 0.0,
                 out_gate: bool = True):
        super().__init__()
        self.inject_layers = [int(li) for li in layers]
        self._pos = {li: i for i, li in enumerate(self.inject_layers)}
        L = len(self.inject_layers)
        self.d_src, self.d_dst, self.d_s, self.d_m = int(d_src), int(d_dst), int(d_src), None
        self.rank = 0
        self.out_gate = bool(out_gate)
        self.null_logit = nn.Parameter(torch.full((L,), float(null_init)))
        self.gate = nn.Parameter(torch.full((L,), float(gate_init)))
        self.log_gain = nn.Parameter(torch.full((L,), float(gain_init)))
        self.log_temp = nn.Parameter(torch.full((L,), float(temp_init)))
        self.has_maps = bool(maps)
        if maps:
            self.kmap = nn.ModuleDict({str(li): nn.Linear(self.d_src, self.d_dst) for li in self.inject_layers})
            self.vmap = nn.ModuleDict({str(li): nn.Linear(self.d_src, self.d_dst) for li in self.inject_layers})
        elif self.d_src != self.d_dst:
            raise ValueError(f"d_src {d_src} != reader KV dim {d_dst}: cross-model reads need maps=True")
        self._init = dict(null_init=null_init, gate_init=gate_init, gain_init=gain_init, temp_init=temp_init)

    def config_dict(self) -> dict:
        return {"native_attn": True, "inject_layers": self.inject_layers, "d_src": self.d_src,
                "d_dst": self.d_dst, "maps": self.has_maps, "out_gate": self.out_gate,
                "d_s": self.d_src, "d_m": None, "rank": 0, **self._init}

    @torch.no_grad()
    def load_maps(self, path: str, rho_of: dict) -> dict:
        """Initialise kmap/vmap from scripts/fit_kv_map.py output ({rho: {Wk,bk,Wv,bv,...}})."""
        blob = torch.load(path, map_location="cpu", weights_only=False)
        stats = {}
        for li in self.inject_layers:
            e = blob["maps"][_key(blob["maps"], rho_of[li])]
            for name, mod in (("k", self.kmap[str(li)]), ("v", self.vmap[str(li)])):
                mod.weight.copy_(e[f"W{name}"].to(mod.weight.dtype))
                mod.bias.copy_(e[f"b{name}"].to(mod.bias.dtype))
            stats[li] = {k: v for k, v in e.items() if not isinstance(v, torch.Tensor)}
        return stats


def save_native_interface(itf: NativeAttnInterface, path: str, selector=None, extra: dict | None = None) -> None:
    blob = {"format": FORMAT, "config": itf.config_dict(),
            "state_dict": {k: v.detach().cpu() for k, v in itf.state_dict().items()},
            "selector": None, "extra": extra}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(blob, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_native_interface(path: str) -> tuple[NativeAttnInterface, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("format") != FORMAT:
        raise ValueError(f"{path} is not a native_attn interface (format={blob.get('format')})")
    c = blob["config"]
    itf = NativeAttnInterface(c["inject_layers"], c["d_src"], c["d_dst"], maps=c["maps"],
                              null_init=c.get("null_init", -30.0), gate_init=c.get("gate_init", 1.0),
                              gain_init=c.get("gain_init", 0.0), temp_init=c.get("temp_init", 0.0),
                              out_gate=c.get("out_gate", True))
    itf.load_state_dict(blob["state_dict"], strict=True)
    return itf, (blob.get("extra") or {})


# ============================================================================ injector
def attn_geometry(att) -> dict:
    hd = int(getattr(att, "head_dim"))
    n_kv = att.k_proj.out_features // hd
    groups = int(getattr(att, "num_key_value_groups", 1))
    n_q = n_kv * groups
    gated = att.q_proj.out_features == 2 * n_q * hd
    if not gated and att.q_proj.out_features != n_q * hd:
        raise ValueError(f"q_proj out {att.q_proj.out_features} inconsistent with {n_q} heads x {hd}")
    return {"hd": hd, "n_kv": n_kv, "groups": groups, "n_q": n_q, "gated": gated,
            "scaling": float(getattr(att, "scaling", hd ** -0.5))}


def reader_queries(att, x: torch.Tensor, geo: dict):
    """The module's own pre-RoPE queries [B, n_q, T, hd] (post q_norm) and output gate [B,T,n_q·hd] or None."""
    B, T, _ = x.shape
    hd = geo["hd"]
    qg = att.q_proj(x)
    if geo["gated"]:
        q, gate = torch.chunk(qg.view(B, T, -1, 2 * hd), 2, dim=-1)
        gate = gate.reshape(B, T, -1)
    else:
        q, gate = qg.view(B, T, -1, hd), None
    if hasattr(att, "q_norm"):
        q = att.q_norm(q)
    return q.transpose(1, 2), gate


class NativeAttnInjector:
    def __init__(self, model, interface: NativeAttnInterface):
        self.model, self.itf = model, interface
        blocks = decoder_layers(model)
        self.blocks = blocks
        self.attn, self.geo = {}, {}
        for li in interface.inject_layers:
            att = getattr(blocks[li], "self_attn", None)
            if att is None or not hasattr(att, "q_proj"):
                raise ValueError(f"reader layer {li} has no softmax self-attention (q_proj)")
            self.attn[li], self.geo[li] = att, attn_geometry(att)
            if self.geo[li]["n_kv"] * self.geo[li]["hd"] != interface.d_dst:
                raise ValueError(f"layer {li}: reader KV dim {self.geo[li]['n_kv'] * self.geo[li]['hd']} "
                                 f"!= interface d_dst {interface.d_dst}")
        self.lib: NativeAttnLibrary | None = None
        self.enabled = True
        self.gain_scale = 1.0
        self.read_fp32 = True
        self.mag_terms: list = []
        self.route_terms: list = []
        self.last_route = None
        self.training_attn = False
        self.attn_pos = -1
        self.attn_terms: list = []
        self.record_attn = False
        self.last_attn: dict = {}
        self.last_null: dict = {}
        self.last_lse: dict = {}
        self.last_attn_heads: dict = {}      # li -> [n_q, S] per-head attention (valid slots only)
        self.last_idx: dict = {}             # li -> [S] flat index of valid slots into [C*N]
        self.skill = None
        # RoPE-consistent read (coordinator follow-up): rope_mode "doc" rotates operator keys by their in-document
        # token index i and the query by N_doc + its own context position, i.e. the geometry of "question after
        # the document"; the operator still occupies no cache positions. "none" = position-free content read.
        self.rope_mode = "none"
        self.head_mask: dict | None = None    # optional {li: bool [n_q]}: only these query heads write
        self._rotary = None
        self._qpos: dict = {}
        self._xin: dict = {}
        self._kv: dict = {}
        self._handles = []
        self._attach()

    # ---- load / unload
    def _active(self) -> bool:
        return self.enabled and self.lib is not None

    def load_library(self, lib: NativeAttnLibrary) -> None:
        missing = set(self.itf.inject_layers) - set(lib.K)
        if missing:
            raise KeyError(f"library has no operator for reader layers {sorted(missing)}")
        if lib.d_s != self.itf.d_src:
            raise ValueError(f"library d={lib.d_s} != interface d_src={self.itf.d_src}")
        self.lib = lib.to(dtype=torch.float32)
        self._kv = {}

    def unload(self) -> None:
        self.lib = None
        self._kv = {}
        self._xin = {}
        self.mag_terms.clear()
        self.route_terms.clear()
        self.attn_terms.clear()

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    # ---- hooks
    def _attach(self) -> None:
        for li in self.itf.inject_layers:
            def pre(mod, args, kwargs, li=li):
                if self._active():
                    self._xin[li] = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                    self._qpos[li] = kwargs.get("position_ids")

            def hook(mod, args, output, li=li):
                if not self._active() or li not in self._xin:
                    return output
                is_tuple = isinstance(output, tuple)
                h = output[0] if is_tuple else output
                delta = self._branch(self._xin.pop(li), li).to(h.dtype)
                rel = delta.float().norm(dim=-1) / (h.float().norm(dim=-1) + 1e-6)
                self.mag_terms.append(rel.mean())
                h = h + delta
                return (h,) + output[1:] if is_tuple else h

            self._handles.append(self.attn[li].register_forward_pre_hook(pre, with_kwargs=True))
            self._handles.append(self.blocks[li].register_forward_hook(hook))

    # ---- operator K/V in the reader's head space
    def _operator(self, li: int, like: torch.Tensor):
        """-> (K [S, n_kv, hd], V [S, n_kv, hd], flat valid index [S]) on like.device, fp32."""
        cached = self._kv.get(li)
        maps = self.itf.has_maps
        if cached is not None and not (maps and torch.is_grad_enabled()):
            return cached
        lib = self.lib
        if lib.K[li].device != like.device:
            lib.K[li], lib.V[li], lib.mask[li] = (lib.K[li].to(like.device), lib.V[li].to(like.device),
                                                  lib.mask[li].to(like.device))
        M = lib.mask[li].flatten()
        idx = M.nonzero().squeeze(-1)
        K = lib.K[li].reshape(-1, lib.K[li].shape[-1]).index_select(0, idx)
        V = lib.V[li].reshape(-1, lib.V[li].shape[-1]).index_select(0, idx)
        if maps:
            km, vm = self.itf.kmap[str(li)], self.itf.vmap[str(li)]
            K = torch.nn.functional.linear(K, _on(km.weight, K).float(), _on(km.bias, K).float())
            V = torch.nn.functional.linear(V, _on(vm.weight, V).float(), _on(vm.bias, V).float())
        g = self.geo[li]
        K = K.reshape(-1, g["n_kv"], g["hd"])
        if self.rope_mode == "doc":
            pos = self._slot_positions(li, like)                                  # [S] in-document token index
            cos, sin = self._rope(pos.view(1, -1), like)
            K = self._rotate(li, K.permute(1, 0, 2).unsqueeze(0), cos, sin)[0].permute(1, 0, 2)
        out = (K, V.reshape(-1, g["n_kv"], g["hd"]), idx)
        if not (maps and torch.is_grad_enabled()):
            self._kv[li] = out
        return out

    # ---- rotary helpers (RoPE-consistent read)
    def _rotary_module(self):
        if self._rotary is None:
            cands = [m for m in self.model.modules()
                     if type(m).__name__.endswith("RotaryEmbedding") and "Vision" not in type(m).__name__]
            if not cands:
                raise ValueError("rope_mode='doc': no rotary embedding module found on the reader")
            self._rotary = cands[0]
        return self._rotary

    def _rope(self, pos: torch.Tensor, like: torch.Tensor):
        rot = self._rotary_module()
        dummy = torch.zeros(1, device=like.device, dtype=torch.float32)
        cos, sin = rot(dummy, pos.to(like.device))
        return cos.float(), sin.float()

    def _rotate(self, li, x, cos, sin):
        """x [B,H,T,hd] rotated with the module's own apply_rotary_pos_emb (partial rotary aware)."""
        import sys as _sys
        fn = getattr(_sys.modules[type(self.attn[li]).__module__], "apply_rotary_pos_emb")
        return fn(x, x, cos, sin)[0]

    def _slot_positions(self, li, like):
        M = self.lib.mask[li]
        starts = torch.tensor([int(m.get("start", c * M.shape[1])) for c, m in enumerate(self.lib.chunk_meta)],
                              device=like.device)
        pos = starts.view(-1, 1) + torch.arange(M.shape[1], device=like.device).view(1, -1)
        return pos.flatten()[M.to(like.device).flatten()]

    def _n_doc(self, li) -> int:
        return int(self.lib.mask[li].sum())

    def _to_cn(self, li: int, a: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        C, N = self.lib.mask[li].shape
        full = torch.zeros(C * N, dtype=a.dtype, device=a.device).index_put((idx,), a)
        return full.view(C, N)

    def _branch(self, x: torch.Tensor, li: int) -> torch.Tensor:
        att, g = self.attn[li], self.geo[li]
        p_i = self.itf._pos[li]
        B, T, _ = x.shape
        q, gate = reader_queries(att, x, g)                                   # [B,n_q,T,hd]
        K, V, idx = self._operator(li, x)
        if self.rope_mode == "doc":
            qp = self._qpos.pop(li, None)
            if qp is None:
                qp = torch.arange(T, device=x.device).view(1, T).expand(B, T)
            if qp.dim() == 3:
                qp = qp[0]
            cos, sin = self._rope(qp + self._n_doc(li), x)
            q = self._rotate(li, q.float(), cos, sin)
        qh = q.float().reshape(B, g["n_kv"], g["groups"], T, g["hd"])
        temp = g["scaling"] * _on(self.itf.log_temp, x)[p_i].exp()
        s = torch.einsum("bkgtd,skd->bkgts", qh, K) * temp                   # [B,k,g,T,S]
        lse = torch.logsumexp(s, -1, keepdim=True)
        p = torch.exp(s - lse)
        keep = torch.sigmoid(lse - _on(self.itf.null_logit, x)[p_i])        # 1 − null mass
        o = torch.einsum("bkgts,skd->bkgtd", p, V) * keep                     # [B,k,g,T,hd]
        o = o.reshape(B, g["n_q"], T, g["hd"])
        if self.head_mask is not None and li in self.head_mask:
            o = o * _on(self.head_mask[li], o).to(o.dtype).view(1, -1, 1, 1)
        o = o.transpose(1, 2).reshape(B, T, -1)
        if gate is not None and self.itf.out_gate:
            o = o * torch.sigmoid(gate.float())
        out = att.o_proj(o.to(att.o_proj.weight.dtype)).float()
        if self.record_attn:
            aw = (p[0, :, :, -1] * keep[0, :, :, -1]).mean((0, 1))           # [S], mean over heads
            self.last_attn[li] = self._to_cn(li, aw.detach(), idx).cpu()
            self.last_null[li] = float(1 - keep[0, :, :, -1].detach().mean())
            self.last_lse[li] = lse[0, :, :, -1, 0].detach().float().cpu()
            self.last_attn_heads[li] = (p[0, :, :, -1] * keep[0, :, :, -1]).detach().reshape(g["n_q"], -1).float().cpu()
            self.last_idx[li] = idx.detach().cpu()
        if self.training_attn and torch.is_grad_enabled():
            aw = (p[0, :, :, self.attn_pos] * keep[0, :, :, self.attn_pos]).mean((0, 1))
            self.attn_terms.append(self._to_cn(li, aw, idx))
        scale = _on(self.itf.log_gain, x)[p_i].exp() * _on(self.itf.gate, x)[p_i]
        return self.gain_scale * scale * out


# ============================================================================ teacher ceiling
@torch.no_grad()
def teacher_attention(model, ids: torch.Tensor, layers) -> dict:
    """Real in-context attention of the LAST token at each full-attention layer: one prefill with pre-hooks
    capturing the attention input and its position embeddings; recompute q (last pos) and all keys with the
    module's own projections, norms and RoPE (exactly what the module attends with). -> {li: [n_q, T] probs}."""
    import sys as _sys
    blocks = decoder_layers(model)
    cap, hs = {}, []
    for li in layers:
        def pre(mod, args, kwargs, li=li):
            cap[li] = (kwargs.get("hidden_states", args[0] if args else None), kwargs["position_embeddings"])
        hs.append(blocks[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
    finally:
        for h in hs:
            h.remove()
    out = {}
    for li in layers:
        att = blocks[li].self_attn
        g = attn_geometry(att)
        x, (cos, sin) = cap[li]
        T = x.shape[1]
        q, _ = reader_queries(att, x[:, -1:], g)                              # [1,n_q,1,hd]
        k = att.k_norm(att.k_proj(x).view(1, T, g["n_kv"], g["hd"])).transpose(1, 2)   # [1,n_kv,T,hd]
        fn = getattr(_sys.modules[type(att).__module__], "apply_rotary_pos_emb")
        q = fn(q, q, cos[:, -1:], sin[:, -1:])[0]
        k = fn(k, k, cos, sin)[0]
        kq = k.float().repeat_interleave(g["groups"], 1)                       # [1,n_q,T,hd]
        s = (q.float() @ kq.transpose(-1, -2))[0, :, 0] * g["scaling"]         # [n_q,T]
        out[li] = torch.softmax(s, -1).cpu()
        del cap[li]
    return out


# ============================================================================ script glue
def native_layers_arg(s) -> list[int] | None:
    if not s:
        return None
    return [int(x) for x in str(s).split(",") if str(x).strip()]


def build_native(model, rt, args, d_enc: int, device):
    """register_doc: attach rt.native, build interface (+ maps) and injector.
    -> (itf, injector, save_fn)."""
    spec = NativeSpec(model, rt.depths, native_layers_arg(getattr(args, "native_layers", None)))
    rt.native = spec
    att = getattr(decoder_layers(model)[spec.layers[0]], "self_attn")
    geo = attn_geometry(att)
    d_dst = geo["n_kv"] * geo["hd"]
    maps = bool(getattr(args, "kv_map", None)) or d_enc != d_dst
    if getattr(args, "init", None):
        itf, _ = load_native_interface(args.init)
    else:
        itf = NativeAttnInterface(spec.layers, d_enc, d_dst, maps=maps, null_init=args.null_init,
                                  gain_init=args.gain_init, temp_init=0.0, out_gate=bool(args.out_gate))
        if getattr(args, "kv_map", None):
            itf.load_maps(args.kv_map, spec.rho_of)
    if itf.has_maps and not getattr(args, "train_maps", 0):
        for m in (itf.kmap, itf.vmap):
            m.requires_grad_(False)
    itf = itf.to(device)
    inj = NativeAttnInjector(model, itf)
    return itf, inj, save_native_interface


def native_param_groups(itf: NativeAttnInterface, lr: float, map_lr: float | None) -> list[dict]:
    scal = [itf.null_logit, itf.gate, itf.log_gain, itf.log_temp]
    groups = [{"params": scal, "lr": lr, "name": "native_scalars"}]
    if itf.has_maps:
        mp = [p for m in (itf.kmap, itf.vmap) for p in m.parameters() if p.requires_grad]
        if mp:
            groups.append({"params": mp, "lr": map_lr if map_lr is not None else lr, "name": "kv_maps"})
    return groups


__all__ = ["NativeAttnLibrary", "NativeSpec", "NativeAttnInterface", "NativeAttnInjector",
           "save_native_interface", "load_native_interface", "build_native", "native_param_groups",
           "attn_geometry", "reader_queries"]
