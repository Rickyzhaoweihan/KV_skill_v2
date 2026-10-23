"""D10: cross-model read of an encoder's full-attention K/V inside the reader's own full-attention layers.

Document KV-Skill (attention part) = the encoder's post-RoPE keys U = K_enc and values W = V_enc at each
encoder full-attention layer, harvested by one frozen prefill (xenc/state_skill.harvest_states keep_kv=True),
fixed. Reader full-attention layer ℓ is paired with encoder full-attention layer ℓ' by normalized depth.
For reader query head h (kv group j = h // G_r), encoder kv-head e(h) = j·Hkv_e // Hkv_r:

    q_h   = q_norm(q_proj(x))_h                      reader's own pre-RoPE query
    q'_h  = RoPE(A_ℓ,h q_h, pos = n_doc + p)         A: per-head 256×256 (pre-RoPE), document positions follow
                                                     the question as if the document preceded it
    o_doc = softmax(q'_h K_e(h)ᵀ · d^-½) V_e(h),   lse_doc = logsumexp of those scores
    o_own, lse_own = the reader's own causal attention over the question tokens
    o     = (e^{lse_own} o_own + e^{lse_doc + b_ℓ} B_ℓ,h o_doc) / (e^{lse_own} + e^{lse_doc + b_ℓ})
then the reader's own output gate σ(gate) and o_proj. A = B = I, b = 0 with the reader's own K/V (L/L) is
EXACTLY the reader attending over the document prefix (kv_only prefix cache), so it is the correctness anchor.
Only A, B, b train. The branch never writes into the KV cache: the document stays an external operator.
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn as nn

import xenc  # noqa: F401
from xenc.state_skill import full_layers, text_config


def attn_dims(model) -> dict:
    c = text_config(model)
    hd = int(getattr(c, "head_dim", c.hidden_size // c.num_attention_heads))
    return {"n_q": int(c.num_attention_heads), "n_kv": int(c.num_key_value_heads), "hd": hd,
            "n_layers": int(c.num_hidden_layers), "full": full_layers(model)}


def full_depth_map(r: dict, e: dict) -> dict:
    out = {}
    for li in r["full"]:
        tgt = li / max(r["n_layers"] - 1, 1) * max(e["n_layers"] - 1, 1)
        out[int(li)] = int(min(e["full"], key=lambda x: (abs(x - tgt), x)))
    return out


class AttnInterface(nn.Module):
    def __init__(self, layer_map: dict, r: dict, e: dict, a_mode: str = "head", b_mode: str = "head",
                 bias_init: float = 0.0):
        super().__init__()
        if r["hd"] != e["hd"]:
            raise ValueError("reader / encoder head_dim differ")
        self.layer_map = {int(k): int(v) for k, v in layer_map.items()}
        self.layers = sorted(self.layer_map)
        self.r, self.e = dict(r), dict(e)
        self.a_mode, self.b_mode = a_mode, b_mode
        g_r = r["n_q"] // r["n_kv"]
        self.e_head = [(h // g_r) * e["n_kv"] // r["n_kv"] for h in range(r["n_q"])]   # reader q-head -> enc kv-head
        hd = r["hd"]
        self.A, self.B = nn.ParameterDict(), nn.ParameterDict()
        for li in self.layers:
            if a_mode == "head":
                self.A[str(li)] = nn.Parameter(torch.eye(hd).repeat(r["n_q"], 1, 1))
            elif a_mode == "mix":     # each reader head's encoder-space query mixes ALL reader heads' queries
                W = torch.zeros(r["n_q"], hd, r["n_q"] * hd)
                for h in range(r["n_q"]):
                    W[h, :, h * hd:(h + 1) * hd] = torch.eye(hd)
                self.A[str(li)] = nn.Parameter(W)
            if b_mode == "head":
                self.B[str(li)] = nn.Parameter(torch.eye(hd).repeat(r["n_q"], 1, 1))
        self.b = nn.Parameter(torch.full((len(self.layers),), float(bias_init)))
        self._bi = {li: i for i, li in enumerate(self.layers)}

    def config(self) -> dict:
        return {"layer_map": self.layer_map, "r": {k: v for k, v in self.r.items() if k != "full"},
                "e": {k: v for k, v in self.e.items() if k != "full"}, "a_mode": self.a_mode, "b_mode": self.b_mode}

    def bias(self, li):
        return self.b[self._bi[li]]

    def map_q(self, li, q):          # q [B,H,T,hd]
        if self.a_mode == "head":
            return torch.einsum("bhtd,hed->bhte", q, self.A[str(li)])
        if self.a_mode == "mix":
            B_, H, T, d = q.shape
            qa = q.transpose(1, 2).reshape(B_, T, H * d)
            return torch.einsum("btf,hef->bhte", qa, self.A[str(li)])
        return q

    def map_o(self, li, o):          # o [B,H,T,hd]
        if self.b_mode == "head":
            return torch.einsum("bhtd,hed->bhte", o, self.B[str(li)])
        return o


def make_attn_interface(reader, encoder_dims: dict, a_mode="head", b_mode="head", bias_init=0.0,
                        layers=None) -> AttnInterface:
    r = attn_dims(reader)
    lm = full_depth_map(r, encoder_dims)
    if layers:
        lm = {k: v for k, v in lm.items() if k in set(layers)}
    return AttnInterface(lm, r, encoder_dims, a_mode, b_mode, bias_init)


def _rotary(model):
    for _, m in model.named_modules():
        if type(m).__name__.endswith("TextRotaryEmbedding"):
            return m
    raise RuntimeError("no rotary embedding module found")


class AttnReader:
    """Patches the mapped reader Qwen3_5Attention modules. load(states) takes a harvest dict with "kv"
    ({enc_layer: (K [Hkv_e,N,hd], V)}) and "n_doc"; unload() restores the stock computation."""

    def __init__(self, model, itf: AttnInterface, record_mag: bool = True):
        from deltaskill.backbone_utils import decoder_layers
        self.model, self.itf = model, itf
        L = decoder_layers(model)
        self.mods = {li: L[li].self_attn for li in itf.layers}
        self.rot = _rotary(model)
        self.KV = None
        self.n_doc = 0
        self.enabled = True
        self.record_mag = record_mag
        self.mag_terms: list = []
        self.gain_scale = 1.0          # unused (API parity); the bias b is the gate
        self.capture: dict | None = None  # {li: (q_pre [B,H,T,hd], o_own [B,H,T,hd])}
        self.capture_kv: dict | None = None  # {li: (K [B,Hkv,T,hd] post-RoPE, V)}
        self.bias_shift = 0.0             # eval-time gain: log(gain) added to the document log-mass
        self.tail = 0                     # last `tail` document tokens are in the reader's own prompt (native tail)
        self.tcap = 0                     # teacher pass: keep the last tcap rows' attention over n_doc_t columns
        self.n_doc_t = 0
        self.teacher_probs: dict = {}
        self.keep_rows = 0                # student pass: keep log-softmax doc rows + doc mass (with grad)
        self.doc_rows: list = []
        self._orig = {}
        for li, m in self.mods.items():
            self._orig[li] = m.forward
            m.forward = types.MethodType(_make_attn_forward(self, li), m)

    def load(self, states: dict | None, device=None):
        if states is None or not states.get("kv"):
            self.KV = None          # e.g. the null condition's mean DeltaNet state: attention branch off
            return
        dev = device or next(self.itf.parameters()).device
        need = set(self.itf.layer_map.values())
        kv = states["kv"]
        self.KV = {int(k): (K.to(dev), V.to(dev)) for k, (K, V) in kv.items() if int(k) in need}
        if need - set(self.KV):
            raise KeyError(f"states lack encoder attention layers {sorted(need - set(self.KV))}")
        self.n_doc = int(states["n_doc"])

    def unload(self):
        self.KV = None

    def detach(self):
        for li, f in self._orig.items():
            self.mods[li].forward = f
        self._orig.clear()

    @property
    def active(self):
        return self.enabled and self.KV is not None


def _make_attn_forward(R: AttnReader, li: int):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb, repeat_kv

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        if not R.active and R.capture is None and R.capture_kv is None and not R.tcap:
            return type(self).forward(self, hidden_states, position_embeddings, attention_mask,
                                      past_key_values=past_key_values, **kwargs)
        itf = R.itf
        input_shape = hidden_states.shape[:-1]
        B_, T = input_shape
        hidden_shape = (*input_shape, -1, self.head_dim)
        q_pre, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        q_pre = self.q_norm(q_pre.view(hidden_shape)).transpose(1, 2)                  # [B,H,T,hd]
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q_pre, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        if not R.active:                     # capture only (long prompts): fused causal attention, no lse
            Kr, Vr = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
            o_own = torch.nn.functional.scaled_dot_product_attention(q, Kr, Vr, is_causal=k.shape[2] == T,
                                                                     scale=self.scaling)
            if R.capture is not None:
                R.capture[li] = (q_pre.detach(), o_own.detach())
            if R.capture_kv is not None:     # post-RoPE keys / values (pre GQA repeat)
                R.capture_kv[li] = (k.detach(), v.detach())
            if R.tcap:                       # teacher: attention of the last rows over the document columns
                K_ = int(R.tcap)
                with torch.no_grad():
                    st = torch.matmul(q[:, :, -K_:].float(), Kr.float().transpose(2, 3)) * self.scaling
                    Kn = st.shape[-1]
                    qi = torch.arange(K_, device=st.device).view(K_, 1) + (Kn - K_)
                    st = st.masked_fill(torch.arange(Kn, device=st.device).view(1, Kn) > qi, float("-inf"))
                    pt = torch.softmax(st, -1)[..., : R.n_doc_t]
                    R.teacher_probs[li] = pt.detach()
            attn_output = o_own.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            return self.o_proj(attn_output * torch.sigmoid(gate)), None
        # ---- own causal attention (eager, with log-sum-exp)
        Kf = repeat_kv(k, self.num_key_value_groups).float()
        Vf = repeat_kv(v, self.num_key_value_groups).float()
        s_own = torch.matmul(q.float(), Kf.transpose(2, 3)) * self.scaling             # [B,H,T,K]
        Kn = s_own.shape[-1]
        qi = torch.arange(T, device=s_own.device).view(T, 1) + (Kn - T)
        mask = torch.arange(Kn, device=s_own.device).view(1, Kn) > qi
        s_own = s_own.masked_fill(mask, float("-inf"))
        lse_own = torch.logsumexp(s_own, -1, keepdim=True)
        o_own = torch.matmul(torch.softmax(s_own, -1), Vf)                               # [B,H,T,hd]
        # ---- document branch
        pos = kwargs.get("position_ids")
        if pos is None:
            pos = torch.arange(Kn - T, Kn, device=hidden_states.device).view(1, T).expand(B_, T)
        pos = pos.to(hidden_states.device) + (R.n_doc - R.tail)
        cos_d, sin_d = R.rot(hidden_states, pos.view(1, B_, T).expand(3, B_, T))
        qa = itf.map_q(li, q_pre.float())
        qd, _ = apply_rotary_pos_emb(qa, qa[:, :1], cos_d.float(), sin_d.float())
        Ke, Ve = R.KV[itf.layer_map[li]]
        idx = torch.tensor(itf.e_head, device=Ke.device)
        Ncol = Ke.shape[1] - R.tail if R.tail else Ke.shape[1]     # tail tokens are re-read natively by the reader
        Kh, Vh = Ke[:, :Ncol].float()[idx], Ve[:, :Ncol].float()[idx]                    # [H,N,hd]
        s_doc = torch.einsum("bhtd,hnd->bhtn", qd, Kh.to(qd.device)) * self.scaling
        lse_doc = torch.logsumexp(s_doc, -1, keepdim=True) + itf.bias(li) + R.bias_shift
        o_doc = itf.map_o(li, torch.einsum("bhtn,hnd->bhtd", torch.softmax(s_doc, -1), Vh.to(qd.device)))
        w = torch.sigmoid(lse_doc - lse_own)                                              # doc share of the mass
        o = (1 - w) * o_own + w * o_doc
        if R.record_mag and torch.is_grad_enabled():
            R.mag_terms.append(w.mean())
        if R.keep_rows:                               # student rows (distillation / addressing metric)
            K_ = int(R.keep_rows)
            R.doc_rows.append((li, torch.log_softmax(s_doc[:, :, -K_:], -1), w[:, :, -K_:, 0]))
        attn_output = o.to(hidden_states.dtype).transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output), None

    return forward


def save_attn_interface(itf: AttnInterface, path: str, extra=None):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"format": "xenc.xattn_read.v1", "config": itf.config(),
                "state_dict": {k: v.detach().cpu() for k, v in itf.state_dict().items()}, "extra": extra or {}},
               path + ".tmp")
    os.replace(path + ".tmp", path)


def load_attn_interface(path_or_blob):
    blob = torch.load(path_or_blob, map_location="cpu", weights_only=False) if isinstance(path_or_blob, str) else path_or_blob
    c = blob["config"]
    itf = AttnInterface(c["layer_map"], c["r"], c["e"], c["a_mode"], c["b_mode"])
    itf.load_state_dict(blob["state_dict"])
    return itf, blob.get("extra") or {}


def attn_distill_loss(R: "AttnReader", mass_w: float = 1.0):
    """Σ over distilled layers of KL(p_T(doc | row) ‖ p_S(doc | row)) (mean over heads/rows), plus mass_w·MSE
    between the teacher's total attention mass on the document and the student's doc share w. None if empty."""
    terms = []
    for li, lps, w in R.doc_rows:
        pt = R.teacher_probs.get(li)
        if pt is None:
            continue
        n = min(pt.shape[-1], lps.shape[-1])
        mt = pt.sum(-1)                                                  # [B,H,K] teacher doc mass
        ptn = pt[..., :n] / pt[..., :n].sum(-1, keepdim=True).clamp_min(1e-9)
        ls = lps[..., :n].to(ptn.device)
        ls = ls - torch.logsumexp(ls, -1, keepdim=True)
        kl = (ptn * (ptn.clamp_min(1e-12).log() - ls)).sum(-1).mean()
        terms.append(kl + mass_w * ((w.to(mt.device) - mt) ** 2).mean())
    return torch.stack(terms).mean() if terms else None


@torch.no_grad()
def attn_js(R: "AttnReader") -> dict:
    """Addressing agreement (no labels): per distilled layer, mean over heads/rows of the Jensen-Shannon
    divergence (nats, max ln2) between the teacher's in-context attention over document positions
    (renormalised) and the branch's softmax over the encoder's document columns; plus both doc masses and
    the JS of a uniform read (reference). Positions align token-for-token (same tokenizer, same prefix)."""
    out = {}
    for li, lps, w in R.doc_rows:
        pt = R.teacher_probs.get(li)
        if pt is None:
            continue
        n = min(pt.shape[-1], lps.shape[-1])
        mt = pt.sum(-1)
        P = pt[..., :n] / pt[..., :n].sum(-1, keepdim=True).clamp_min(1e-12)
        Q = torch.softmax(lps[..., :n].float().to(P.device), -1)
        U = torch.full_like(P, 1.0 / n)

        def js(a, b):
            m = 0.5 * (a + b)
            kl = lambda x, y: (x * (x.clamp_min(1e-12).log() - y.clamp_min(1e-12).log())).sum(-1)   # noqa: E731
            return float((0.5 * kl(a, m) + 0.5 * kl(b, m)).mean())

        k = min(5, n)
        top_t = P.topk(k, -1).indices
        top_s = Q.topk(k, -1).indices
        overlap = float((top_t.unsqueeze(-1) == top_s.unsqueeze(-2)).any(-1).float().mean())
        out[int(li)] = {"js": js(P, Q), "js_uniform": js(P, U), "mass_teacher": float(mt.mean()),
                        "mass_branch": float(w.float().mean()), "top5_overlap": overlap}
    return out


def rotate_keys(rot, K, sign: float = 1.0):
    """K [H,N,hd]: apply (sign=+1) or undo (sign=−1) RoPE at positions 0..N−1 with rotary module `rot`."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import rotate_half
    N = K.shape[1]
    pos = torch.arange(N, device=K.device).view(1, 1, N).expand(3, 1, N)
    cos, sin = rot(K.unsqueeze(0).float(), pos)
    cos, sin = cos[0].float(), sin[0].float()                       # [N, rd]
    rd = cos.shape[-1]
    x = K.float()
    xr, xp = x[..., :rd], x[..., rd:]
    xr = xr * cos + rotate_half(xr) * (sign * sin)
    return torch.cat([xr, xp], -1)


def map_kv_states(st: dict, maps: dict, rot, layer_pairs: dict, device=None) -> dict:
    """Transform the OPERATOR once: encoder K (un-RoPE'd) and V at encoder layer layer_pairs[li] -> reader K/V space
    by affine maps (maps["kv_K"][li], maps["kv_V"][li]: [n_kv_r·hd, n_kv_e·hd + 1]), then the reader's RoPE at the
    document positions. Returns a states dict whose "kv" is keyed by READER layer (read with the reader's own queries)."""
    out = {"n_doc": st["n_doc"], "kv": {}}
    for li, e in layer_pairs.items():
        K, V = st["kv"][e] if e in st["kv"] else st["kv"][str(e)]
        dev = device or K.device
        K, V = K.to(dev), V.to(dev)
        He, N, hd = K.shape
        Kp = rotate_keys(rot, K, -1.0)                                 # pre-RoPE encoder keys
        WK, WV = maps["kv_K"][str(li)].to(dev).float(), maps["kv_V"][str(li)].to(dev).float()
        one = torch.ones(N, 1, device=dev)
        Kx = torch.cat([Kp.transpose(0, 1).reshape(N, He * hd), one], 1) @ WK.t()
        Vx = torch.cat([V.float().transpose(0, 1).reshape(N, He * hd), one], 1) @ WV.t()
        Hr = WK.shape[0] // hd
        Kr = Kx.view(N, Hr, hd).transpose(0, 1)
        out["kv"][li] = (rotate_keys(rot, Kr, 1.0).half(), Vx.view(N, Hr, hd).transpose(0, 1).half())
    return out


__all__ = ["rotate_keys", "map_kv_states", "attn_js", "attn_distill_loss", "AttnInterface", "AttnReader", "make_attn_interface", "attn_dims", "save_attn_interface",
           "load_attn_interface", "full_depth_map"]
