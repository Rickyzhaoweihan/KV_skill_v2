"""D12p (copy of D13 for the 27B reader). D13: the paper's read AS WRITTEN — the READER's own question state queries the small encoder's fixed operator —
with A fitted to the encoder's OWN question-pass attention (the passing encoder-side read is the teacher). Label-free.

Operator (fixed harvest, never trained): the encoder's post-RoPE keys / values U = K_e, W = V_e at encoder
full-attention layer e over the document prefix (one frozen prefill; the last `tail` document tokens are left out
because the reader prefills them itself). Read at reader full-attention layer ℓ (paired with e by normalized depth),
for every reader row t (tail + question + generated):

    x̃_t       = A_ℓ(RMSNorm_ℓ(h_t))                    A: d_r -> d_e (affine; optional residual MLP, before the norm)
    q_t, g_t   = q_norm(W_q x̃_t), W_g x̃_t                the encoder's frozen projections: q = norm(A'·RMSNorm(h)), A' = W_q A
    k_t, v_t   = k_norm(W_k x̃_t), W_v x̃_t                the reader's own rows mapped into the operator's space
    p_t        = softmax(RoPE(q_t)·[U ; k_≤t]ᵀ / √d)     over the operator's keys (+ own earlier rows, causal)
    out_t      = B_ℓ W_o (σ(g_t) ⊙ p_t·[W ; v_≤t])       B: D10's closed-form encoder-side post map (fixed)

out_t REPLACES the reader's attention output at ℓ (backbone frozen; no prompt positions or reader-cache entries are
added). The teacher (training only) is the encoder's own pass over document + the same rows: its per-head attention
over the same columns and its post-o_proj output. Only A trains:  KL(p_teacher ‖ p_student) + λ·relMSE(B o_s, B o_t).
With reader = encoder, A = I, B = I the read equals the encoder's own attention output at those rows (tests/test_d13.py).
"""

from __future__ import annotations

import copy
import types

import torch
import torch.nn as nn

import xenc  # noqa: F401

FORMAT = "xenc.d13_read.v1"   # D12p copy of xenc/d13_read.py (same file format)


def _rotary(model):
    from xenc.xattn_read import _rotary as r
    return r(model)


class D13Interface(nn.Module):
    """A (trainable) + the encoder's frozen attention projections at the paired layers + fixed B."""

    def __init__(self, enc, layer_map: dict, d_r: int, d_e: int, B: dict | None = None, mlp_hidden: int = 0):
        super().__init__()
        from deltaskill.backbone_utils import decoder_layers
        self.layer_map = {int(k): int(v) for k, v in layer_map.items()}
        self.layers = sorted(self.layer_map)
        self.d_r, self.d_e, self.mlp_hidden = int(d_r), int(d_e), int(mlp_hidden)
        L = decoder_layers(enc)
        self.att = nn.ModuleDict({str(li): copy.deepcopy(L[e].self_attn).float().requires_grad_(False)
                                  for li, e in self.layer_map.items()})
        self.rot = copy.deepcopy(_rotary(enc)).float()
        self.A = nn.ModuleDict()
        self.M = nn.ModuleDict()
        for li in self.layers:
            lin = nn.Linear(self.d_r, self.d_e)
            with torch.no_grad():
                lin.bias.zero_()
                lin.weight.copy_(torch.eye(self.d_e, self.d_r))
            self.A[str(li)] = lin
            if self.mlp_hidden:
                m = nn.Sequential(nn.Linear(self.d_r, self.mlp_hidden), nn.GELU(), nn.Linear(self.mlp_hidden, self.d_e))
                nn.init.zeros_(m[2].weight)
                nn.init.zeros_(m[2].bias)
                self.M[str(li)] = m
        self.identity_B = B is None
        self.B = nn.ParameterDict()
        if B is not None:
            for li in self.layers:
                self.B[str(li)] = nn.Parameter(B[li].float().clone(), requires_grad=False)
        self.g = nn.Parameter(torch.ones(len(self.layers)))      # per-layer gate: own + g·(read − own); 1 = replace
        self._gi = {li: i for i, li in enumerate(self.layers)}
        self.Bb = nn.ParameterDict({str(li): nn.Parameter(torch.zeros(self.d_r), requires_grad=False)
                                    for li in self.layers} if B is not None else {})   # write-back bias
        self.delta_rank = 0
        self.D = nn.ParameterDict()                              # optional low-rank deltas on A and B (stage 2)

    def add_delta(self, rank: int, seed: int = 0):
        """Low-rank deltas A += Ua Va, B += Ub Vb (U zero-init: starts exactly at the init). The init weights freeze."""
        g = torch.Generator().manual_seed(seed)
        dev = self.A[str(self.layers[0])].weight.device
        self.delta_rank = int(rank)
        for li in self.layers:
            k = str(li)
            self.D[f"Ua{k}"] = nn.Parameter(torch.zeros(self.d_e, rank, device=dev))
            self.D[f"Va{k}"] = nn.Parameter((torch.randn(rank, self.d_r, generator=g) / self.d_r ** 0.5).to(dev))
            if not self.identity_B:
                self.D[f"Ub{k}"] = nn.Parameter(torch.zeros(self.d_r, rank, device=dev))
                self.D[f"Vb{k}"] = nn.Parameter((torch.randn(rank, self.d_e, generator=g) / self.d_e ** 0.5).to(dev))
        self._A0n = {li: float(self.A[str(li)].weight.detach().pow(2).sum()) for li in self.layers}
        self._B0n = {li: float(self.B[str(li)].detach().pow(2).sum()) for li in self.layers} if not self.identity_B else {}

    def delta_reg(self):
        """mean_ℓ ‖ΔA_ℓ‖²/‖A0_ℓ‖² + ‖ΔB_ℓ‖²/‖B0_ℓ‖² (relative drift from the init)."""
        terms = []
        for li in self.layers:
            k = str(li)
            t = (self.D[f"Ua{k}"] @ self.D[f"Va{k}"]).pow(2).sum() / self._A0n[li]
            if not self.identity_B:
                t = t + (self.D[f"Ub{k}"] @ self.D[f"Vb{k}"]).pow(2).sum() / self._B0n[li]
            terms.append(t)
        return torch.stack(terms).mean()

    @torch.no_grad()
    def merge_delta(self):
        """Fold the low-rank deltas into A / B (afterwards the deltas are dropped)."""
        if not self.delta_rank:
            return
        for li in self.layers:
            k = str(li)
            self.A[k].weight.add_(self.D[f"Ua{k}"] @ self.D[f"Va{k}"])
            if not self.identity_B:
                self.B[k].add_(self.D[f"Ub{k}"] @ self.D[f"Vb{k}"])
        self.D = nn.ParameterDict()
        self.delta_rank = 0

    def trainable(self, train_B: bool = False, train_g: bool = False):
        ps = list(self.A.parameters()) + list(self.M.parameters())
        for p in list(self.B.parameters()) + list(self.Bb.parameters()):
            p.requires_grad_(bool(train_B))
        self.g.requires_grad_(bool(train_g))
        return ps + (list(self.B.parameters()) + list(self.Bb.parameters()) if train_B else []) + \
            ([self.g] if train_g else [])

    def gate(self, li):
        return self.g[self._gi[li]]

    def config(self) -> dict:
        return {"layer_map": self.layer_map, "d_r": self.d_r, "d_e": self.d_e, "mlp_hidden": self.mlp_hidden}

    def map_x(self, li, x):
        y = self.A[str(li)](x)
        if self.delta_rank:
            k = str(li)
            y = y + (x @ self.D[f"Va{k}"].t()) @ self.D[f"Ua{k}"].t()
        if self.mlp_hidden:
            y = y + self.M[str(li)](x)
        return y

    def Bmap(self, li, o):
        if self.identity_B:
            return o
        k = str(li)
        y = o @ self.B[k].t() + self.Bb[k]
        if self.delta_rank:
            y = y + (o @ self.D[f"Vb{k}"].t()) @ self.D[f"Ub{k}"].t()
        return y

    def read(self, li, x, Kop, Vop, pos0: int, past=None, need_logits: bool = False):
        """x [T,d_r] (fp32) reader rows; Kop/Vop [Hkv,N,hd] operator (post-RoPE at positions 0..N-1);
        pos0 = N (position of the first reader row, i.e. after the operator); past = (Kp, Vp) own earlier rows.
        -> dict(y [T,d_r] B-mapped output, o [T,d_e] encoder-space output, logits [Hkv,G,T,C] (if asked), past)."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        att = self.att[str(li)]
        hd = att.head_dim
        xe = self.map_x(li, x)
        T = xe.shape[0]
        q, gate = torch.chunk(att.q_proj(xe).view(T, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(T, -1)
        q = att.q_norm(q).transpose(0, 1)                                   # [Hq,T,hd]
        k = att.k_norm(att.k_proj(xe).view(T, -1, hd)).transpose(0, 1)      # [Hkv,T,hd]
        v = att.v_proj(xe).view(T, -1, hd).transpose(0, 1)
        P = 0 if past is None else past[0].shape[1]
        pos = torch.arange(pos0 + P, pos0 + P + T, device=xe.device).view(1, 1, T).expand(3, 1, T)
        cos, sin = self.rot(xe.unsqueeze(0), pos)
        qr, kr = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos.float(), sin.float())
        qr, kr = qr[0], kr[0]
        Kown = kr if past is None else torch.cat([past[0], kr], 1)
        Vown = v if past is None else torch.cat([past[1], v], 1)
        out = _attend(qr, Kop.float(), Vop.float(), Kown, Vown, att.scaling, need_logits)
        o = att.o_proj(out["o"] * torch.sigmoid(gate))
        return {"y": self.Bmap(li, o), "o": o, "logits": out.get("logits"),
                "past": (Kown.detach(), Vown.detach())}


def _attend(q, Kop, Vop, Kown, Vown, scale, need_logits=False):
    """q [Hq,T,hd]; Kop [Hkv,N,hd]; Kown [Hkv,P+T,hd] (last T = these rows). GQA without repeat.
    -> {"o": [T,Hq·hd], "logits": [Hkv,G,T,N+P+T] masked}."""
    Hq, T, hd = q.shape
    Hkv = Kop.shape[0]
    G = Hq // Hkv
    K = torch.cat([Kop, Kown], 1)
    V = torch.cat([Vop, Vown], 1)
    C = K.shape[1]
    qg = q.view(Hkv, G, T, hd)
    s = torch.einsum("kgtd,kcd->kgtc", qg, K) * scale
    first_own = C - T                                     # column of this forward's first row
    col = torch.arange(C, device=q.device).view(1, C)
    row = torch.arange(T, device=q.device).view(T, 1)
    s = s.masked_fill((col > first_own + row).view(1, 1, T, C), float("-inf"))
    p = torch.softmax(s, -1)
    o = torch.einsum("kgtc,kcd->kgtd", p, V).reshape(Hq, T, hd).transpose(0, 1).reshape(T, Hq * hd)
    out = {"o": o}
    if need_logits:
        out["logits"] = s
    return out


def teacher_logits(q_t, Kop, k_t, scale):
    """Teacher scores over the same columns: q_t [Hq,T,hd] (post-RoPE), Kop [Hkv,N,hd], k_t [Hkv,T,hd]."""
    Hq, T, hd = q_t.shape
    Hkv = Kop.shape[0]
    K = torch.cat([Kop.float(), k_t.float()], 1)
    C = K.shape[1]
    s = torch.einsum("kgtd,kcd->kgtc", q_t.float().view(Hkv, Hq // Hkv, T, hd), K) * scale
    col = torch.arange(C, device=s.device).view(1, C)
    row = torch.arange(T, device=s.device).view(T, 1)
    return s.masked_fill((col > (C - T) + row).view(1, 1, T, C), float("-inf"))


def d13_terms(ls_logits, lt_logits, y_s, y_t, N: int, metrics: bool = False):
    """KL(p_t ‖ p_s) over all columns (mean heads/rows) and relMSE(y_s, y_t) (mean rows); optional diagnostics:
    JS (full and over the operator columns renormalised), operator mass teacher/student, top-5 overlap, cos(y)."""
    ls = torch.log_softmax(ls_logits, -1)
    lt = torch.log_softmax(lt_logits, -1)
    pt = lt.exp()
    kl = (pt * (torch.nan_to_num(lt, neginf=0.0) - torch.nan_to_num(ls, neginf=0.0))).sum(-1).mean()
    rel = (((y_s - y_t) ** 2).sum(-1) / (y_t ** 2).sum(-1).clamp_min(1e-6)).mean()
    out = {"kl": kl, "rel": rel}
    if metrics:
        with torch.no_grad():
            ps = ls.exp()
            out["js"] = float(_js(pt, ps).mean())
            mt, ms = pt[..., :N].sum(-1), ps[..., :N].sum(-1)
            a = pt[..., :N] / mt.unsqueeze(-1).clamp_min(1e-9)
            b = ps[..., :N] / ms.unsqueeze(-1).clamp_min(1e-9)
            out["js_doc"] = float(_js(a, b).mean())
            out["js_doc_uniform"] = float(_js(a, torch.full_like(a, 1.0 / max(N, 1))).mean())
            out["mass_t"], out["mass_s"] = float(mt.mean()), float(ms.mean())
            k = min(5, N)
            ti, si = a.topk(k, -1).indices, b.topk(k, -1).indices
            out["top5"] = float((ti.unsqueeze(-1) == si.unsqueeze(-2)).any(-1).float().mean())
            out["cos"] = float(torch.nn.functional.cosine_similarity(y_s, y_t, dim=-1).mean())
            out["kl"], out["rel"] = float(kl), float(rel)
    return out


def _js(a, b):
    m = 0.5 * (a + b)
    kl = lambda x, y: (x * (x.clamp_min(1e-12).log() - y.clamp_min(1e-12).log())).sum(-1)   # noqa: E731
    return 0.5 * kl(a, m) + 0.5 * kl(b, m)


# ----------------------------------------------------------------------------- encoder teacher / operator capture
class EncTeacher:
    """Patches the encoder's attention modules at `layers`. With rec = {} set, every forward records per layer
    {"k","v"} (post-RoPE, [Hkv,T,hd]) and, unless kv_only, {"x" [T,d] input, "q" [Hq,T,hd] post-RoPE, "out" [T,d]}."""

    def __init__(self, enc, layers):
        from deltaskill.backbone_utils import decoder_layers
        L = decoder_layers(enc)
        self.layers = sorted(set(int(x) for x in layers))
        self.mods = {e: L[e].self_attn for e in self.layers}
        self.rec: dict | None = None
        self.kv_only = False
        self._orig = {}
        for e, m in self.mods.items():
            self._orig[e] = m.forward
            m.forward = types.MethodType(_teacher_forward(self, e), m)

    def detach(self):
        for e, f in self._orig.items():
            self.mods[e].forward = f
        self._orig.clear()


def _teacher_forward(Tc: EncTeacher, e: int):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        if Tc.rec is not None:
            shp = hidden_states.shape[:-1]
            hs = (*shp, -1, self.head_dim)
            k = self.k_norm(self.k_proj(hidden_states).view(hs)).transpose(1, 2)
            v = self.v_proj(hidden_states).view(hs).transpose(1, 2)
            q = None
            if not Tc.kv_only:
                q, _ = torch.chunk(self.q_proj(hidden_states).view(*shp, -1, self.head_dim * 2), 2, dim=-1)
                q = self.q_norm(q.reshape(hs)).transpose(1, 2)
            cos, sin = position_embeddings
            qq, kk = apply_rotary_pos_emb(q if q is not None else k, k, cos, sin)
            r = {"k": kk[0].detach(), "v": v[0].detach()}
            if not Tc.kv_only:
                r["q"], r["x"] = qq[0].detach(), hidden_states[0].detach()
            Tc.rec[e] = r
        out = type(self).forward(self, hidden_states, position_embeddings, attention_mask,
                                 past_key_values=past_key_values, **kwargs)
        if Tc.rec is not None and not Tc.kv_only:
            Tc.rec[e]["out"] = out[0][0].detach()
        return out

    return forward


# ----------------------------------------------------------------------------- reader patch
class D13Reader:
    """Patches the reader's attention at itf.layers. mode:
        "off"     stock reader
        "student" REPLACE the attention output by the D13 read of R.op (eval / on-policy inputs); own-row K/V
                  accumulate in R.past across forwards (generation)
        "train"   compute the read with grad and store it in R.rec[li]; pass downstream R.inject[li] (teacher
                  forcing) or, if R.onpolicy, the student's own (detached) read
        "teacher" pass R.inject[li] downstream and record the layer input x in R.xcap[li] (ridge statistics)
        "e2e"     end-to-end registration: the gated read (with grad) replaces the output and flows downstream;
                  the read (with logits if R.need_logits) is stored in R.rec[li]
    R.op = {enc_layer: (K, V)} operator; R.n_op = its length."""

    def __init__(self, model, itf: D13Interface):
        from deltaskill.backbone_utils import decoder_layers
        self.model, self.itf = model, itf
        L = decoder_layers(model)
        self.mods = {li: L[li].self_attn for li in itf.layers}
        self.mode = "off"
        self.op: dict = {}
        self.n_op = 0
        self.past: dict = {}
        self.rec: dict = {}
        self.inject: dict = {}
        self.xcap: dict = {}
        self.onpolicy = False
        self.need_logits = False
        self.grad = True
        self._orig = {}
        for li, m in self.mods.items():
            self._orig[li] = m.forward
            m.forward = types.MethodType(_reader_forward(self, li), m)

    def load(self, op: dict, n_op: int):
        self.op, self.n_op, self.past = op, int(n_op), {}

    def detach(self):
        for li, f in self._orig.items():
            self.mods[li].forward = f
        self._orig.clear()


def _reader_forward(R: D13Reader, li: int):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        out, w = type(self).forward(self, hidden_states, position_embeddings, attention_mask,
                                    past_key_values=past_key_values, **kwargs)
        if R.mode == "off":
            return out, w
        itf = R.itf
        e = itf.layer_map[li]
        K, V = R.op[e]
        idev = itf.g.device                                    # D12p: reader may be sharded over GPUs
        x = hidden_states[0].float().to(idev)
        if R.mode == "student":
            res = itf.read(li, x, K, V, R.n_op, past=R.past.get(li))
            R.past[li] = res["past"]
            g = itf.gate(li)
            y = out[0].float() + g.to(out.device) * (res["y"].to(out.device) - out[0].float())
        elif R.mode == "teacher":
            R.xcap[li] = x.detach()
            y = R.inject[li]
            if getattr(R, "gate_tf", False):                   # D12p deep: teacher forcing honours the gates
                y = out[0].float() + itf.gate(li).detach().to(out.device) * (y.to(out.device) - out[0].float())
        elif R.mode == "e2e":
            if torch.is_grad_enabled() and getattr(R, "ckpt", True):
                # D12p: recompute the read's [Hkv,G,T,C] intermediates in backward (27B on one GPU, long documents)
                from torch.utils.checkpoint import checkpoint
                nl = bool(R.need_logits)

                def _rd(x_):
                    r_ = itf.read(li, x_, K, V, R.n_op, need_logits=nl)
                    return (r_["y"], r_["o"], r_["logits"]) if nl else (r_["y"], r_["o"])
                t_ = checkpoint(_rd, x, use_reentrant=False)
                res = {"y": t_[0], "o": t_[1], "logits": t_[2] if nl else None}
            else:
                res = itf.read(li, x, K, V, R.n_op, need_logits=R.need_logits)
            R.rec[li] = res
            g = itf.gate(li)
            y = out[0].float() + g.to(out.device) * (res["y"].to(out.device) - out[0].float())
        else:
            with torch.set_grad_enabled(R.grad):
                res = itf.read(li, x, K, V, R.n_op, need_logits=True)
            R.rec[li] = res
            y = res["y"].detach() if R.onpolicy else R.inject[li]
            if getattr(R, "gate_tf", False):                   # D12p deep: a gate-0 layer leaves the stream unchanged
                y = out[0].float() + itf.gate(li).detach().to(out.device) * (y.to(out.device) - out[0].float())
        return y.to(device=out.device, dtype=out.dtype).unsqueeze(0), w

    return forward


# ----------------------------------------------------------------------------- save / load
def save_d13(itf: D13Interface, path: str, extra: dict | None = None):
    """Saves A, M, B, Bb, g; low-rank deltas (if any) are folded into A / B in the saved copy."""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sd = {k: v.detach().cpu().clone() for k, v in itf.state_dict().items()
          if k.startswith(("A.", "M.", "B.", "Bb.")) or k == "g"}
    if itf.delta_rank:
        with torch.no_grad():
            for li in itf.layers:
                k = str(li)
                sd[f"A.{k}.weight"] += (itf.D[f"Ua{k}"] @ itf.D[f"Va{k}"]).detach().cpu()
                if not itf.identity_B:
                    sd[f"B.{k}"] += (itf.D[f"Ub{k}"] @ itf.D[f"Vb{k}"]).detach().cpu()
    torch.save({"format": FORMAT, "config": itf.config(), "state_dict": sd, "extra": extra or {}}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_d13(path: str, enc, B: dict | None):
    """B: the fixed encoder-side maps (load_B); a B stored in the file (stage-2 registration) overrides them."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    c = blob["config"]
    itf = D13Interface(enc, c["layer_map"], c["d_r"], c["d_e"], B=B, mlp_hidden=c["mlp_hidden"])
    sd = blob["state_dict"]
    if B is None:
        sd = {k: v for k, v in sd.items() if not k.startswith(("B.", "Bb."))}
    missing, unexpected = itf.load_state_dict(sd, strict=False)
    assert not unexpected and not [k for k in missing if k.startswith(("A.", "M."))], (missing, unexpected)
    return itf, blob.get("extra") or {}


def load_B(maps_path: str, layers) -> dict:
    blob = torch.load(maps_path, map_location="cpu", weights_only=False)
    assert blob.get("level", "post") == "post", "D13 writes back at the post-o_proj level"
    assert not blob.get("mlp"), "D13 uses the linear encoder-side B"
    return {int(li): blob["enc_B"][str(li)].float() for li in layers}


__all__ = ["D13Interface", "D13Reader", "EncTeacher", "teacher_logits", "d13_terms", "save_d13", "load_d13", "load_B"]
