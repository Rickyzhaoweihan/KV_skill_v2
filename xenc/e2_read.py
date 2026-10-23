"""E2: make the operator carry more of the encoder's information (extensions of the D13 read, xenc/d13_read.py).

Both extensions keep the D13 read's ADDRESSING (the reader's mapped query against the operator keys U = K_e of the
paired encoder layer e) and change only what the attention reads out. Both start exactly at the D13 read (zero-init
mixing weights), so step 0 == the parent base.

  dv  (deep values, axis 3)    at reader layer ℓ (paired e), extra value tensor harvested once from the document
                               prefill:  V'_e = v_proj_e(input_layernorm_e(h_{e'}))  with h_{e'} the encoder's residual
                               stream at the output of a DEEPER layer e' (question-independent at doc positions).
                               o = p·[V_e + α(V'_e − V_e) ; V_own]   α per kv head (init 0)
  ms  (multi-source, axis 2)   the same attention pattern p also reads a deeper attention layer e2's own operator values
                               V_{e2} through e2's o_proj, mapped by its own B2 (ridge init, scripts/e2_probe.py):
                               y2 = B2 o_proj_{e2}(σ(gate) ⊙ p·[V_{e2} ; v_proj_{e2}(x̃)]) + b2
                               y  = y_D13 + β(y2 − y_D13)          β per layer (init 0): a learned mix of source layers

  rs  (retrieval source)       reader layer ℓ ALSO reads a deeper encoder layer e3 (the 4B's evidence-retrieval layers
                               19-27, scripts/e2_probe.py) with its OWN addressing: A3 (init = the base A_ℓ / MLP_ℓ),
                               e3's frozen q/k/v/gate/o projections over op[e3], B3 (ridge init); y3 = B3 o3 + b3,
                               y = y + γ(y3 − y), γ per layer (zero init). Its own distillation term (KL to the encoder's
                               layer-e3 question-pass attention + relMSE to B3_0 o_enc(e3)) trains the new addressing
                               (`extra_dist`, added by register_d13.py).

  tr  (operator translator)     a per-layer, document-independent map of the 4B operator before the reader reads it:
                               K' = RoPE(k + k·W_k + b_k) with k = RoPE⁻¹(K) (pre-RoPE keys, so the map is position-
                               consistent), V' = V + V·W_v + b_v, per kv head, W / b zero init (step 0 == base). It can
                               be folded into the operator at harvest time (no reader pass). The distillation relMSE
                               uses the UNtranslated values (the attention KL still sees the translated keys).

The stage-1 distillation terms (attention KL + relMSE to the encoder's own layer-e output) are computed on the
UNMIXED D13 read-out ("y_base"), so the regularizer keeps the addressing on the document without pulling the new
read-out back to layer e. The write-back uses the mixed read-out.
"""

from __future__ import annotations

import copy
import types

import torch
import torch.nn as nn

from xenc.d13_read import D13Interface, D13Reader


def parse_pairs(spec: str) -> dict:
    """'3:15,7:19' -> {3: 15, 7: 19}."""
    return {int(k): int(v) for k, v in (s.split(":") for s in (spec or "").split(",") if s)}


class E2Interface(D13Interface):
    def __init__(self, enc, layer_map: dict, d_r: int, d_e: int, B: dict | None = None, mlp_hidden: int = 0,
                 e2: dict | None = None, B2: dict | None = None, B3: dict | None = None):
        super().__init__(enc, layer_map, d_r, d_e, B=B, mlp_hidden=mlp_hidden)
        from deltaskill.backbone_utils import decoder_layers
        e2 = e2 or {}
        self.dv = {int(k): int(v) for k, v in (e2.get("dv") or {}).items()}
        self.ms = {int(k): int(v) for k, v in (e2.get("ms") or {}).items()}
        self.rs = {int(k): int(v) for k, v in (e2.get("rs") or {}).items()}
        self.tr = bool(e2.get("tr", 0))
        assert set(self.dv) | set(self.ms) | set(self.rs) <= set(self.layers), (self.dv, self.ms, self.rs, self.layers)
        for li, e1 in self.dv.items():
            assert e1 > self.layer_map[li], f"dv source must be deeper than the paired layer ({li}: {e1})"
        L = decoder_layers(enc)
        hkv = L[self.layer_map[self.layers[0]]].self_attn.k_proj.out_features // L[self.layer_map[self.layers[0]]].self_attn.head_dim
        self.att2 = nn.ModuleDict({str(li): copy.deepcopy(L[e2_].self_attn).float().requires_grad_(False)
                                   for li, e2_ in self.ms.items()})
        self.E2 = nn.ModuleDict({
            "alpha": nn.ParameterDict({str(li): nn.Parameter(torch.zeros(hkv)) for li in self.dv}),
            "beta": nn.ParameterDict({str(li): nn.Parameter(torch.zeros(1)) for li in self.ms}),
            "B2": nn.ParameterDict({str(li): nn.Parameter(
                (B2[li]["W"].float().clone() if B2 is not None else torch.zeros(self.d_r, self.d_e))) for li in self.ms}),
            "b2": nn.ParameterDict({str(li): nn.Parameter(
                (B2[li]["b"].float().clone() if B2 is not None else torch.zeros(self.d_r))) for li in self.ms}),
            "gamma": nn.ParameterDict({str(li): nn.Parameter(torch.zeros(1)) for li in self.rs}),
            "B3": nn.ParameterDict({str(li): nn.Parameter(
                (B3[li]["W"].float().clone() if B3 is not None else torch.zeros(self.d_r, self.d_e))) for li in self.rs}),
            "b3": nn.ParameterDict({str(li): nn.Parameter(
                (B3[li]["b"].float().clone() if B3 is not None else torch.zeros(self.d_r))) for li in self.rs}),
            "A3": nn.ModuleDict({str(li): nn.Linear(self.d_r, self.d_e) for li in self.rs}),
            "M3": nn.ModuleDict({str(li): nn.Sequential(nn.Linear(self.d_r, self.mlp_hidden), nn.GELU(),
                                                        nn.Linear(self.mlp_hidden, self.d_e))
                                 for li in self.rs} if self.mlp_hidden else {}),
        })
        a0 = L[self.layer_map[self.layers[0]]].self_attn
        hd0 = a0.head_dim
        self.E2["tr"] = nn.ParameterDict({f"{n}{li}": nn.Parameter(torch.zeros(hkv, hd0, hd0) if n[0] == "W"
                                                                    else torch.zeros(hkv, hd0))
                                          for li in (self.layers if self.tr else []) for n in ("Wk", "bk", "Wv", "bv")})
        self.att3 = nn.ModuleDict({str(li): copy.deepcopy(L[e3].self_attn).float().requires_grad_(False)
                                   for li, e3 in self.rs.items()})
        # frozen copy of B3 at the START of training (captured at the first extra_dist call): the relMSE target side
        # of the rs distillation term, like B0 for the base read
        self.B3_0 = None

    # -------------------------------------------------------------- bookkeeping
    def config(self) -> dict:
        c = super().config()
        c["e2"] = {"dv": {str(k): v for k, v in self.dv.items()}, "ms": {str(k): v for k, v in self.ms.items()},
                   "rs": {str(k): v for k, v in self.rs.items()}, "tr": int(self.tr)}
        return c

    def enc_layers(self) -> list:
        """Encoder attention layers whose K/V are harvested (EncTeacher): the paired layers + the ms sources."""
        return sorted(set(self.layer_map.values()) | set(self.ms.values()) | set(self.rs.values()))

    def e2_params(self) -> dict:
        return {"mix": list(self.E2["alpha"].parameters()) + list(self.E2["beta"].parameters())
                + list(self.E2["gamma"].parameters()),
                "B2": list(self.E2["B2"].parameters()) + list(self.E2["b2"].parameters())
                + list(self.E2["B3"].parameters()) + list(self.E2["b3"].parameters()),
                "A3": list(self.E2["A3"].parameters()) + list(self.E2["M3"].parameters()),
                "tr": list(self.E2["tr"].parameters())}

    def e2_state(self) -> dict:
        return {"alpha": {li: [round(float(x), 4) for x in p] for li, p in self.E2["alpha"].items()},
                "beta": {li: round(float(p), 4) for li, p in self.E2["beta"].items()},
                "gamma": {li: round(float(p), 4) for li, p in self.E2["gamma"].items()},
                "tr_norm": {k: round(float(p.detach().norm()), 4) for k, p in self.E2["tr"].items() if k[0] == "W"}}

    def make_reader(self, model):
        return E2Reader(model, self)

    # -------------------------------------------------------------- harvest (document prefill, once per document)
    def harvest_hooks(self, enc):
        return _DVHarvest(enc, self) if self.dv else None

    # -------------------------------------------------------------- the read
    def read(self, li, x, Kop, Vop, pos0: int, past=None, need_logits: bool = False, ext: dict | None = None):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        att = self.att[str(li)]
        hd = att.head_dim
        xe = self.map_x(li, x)
        T = xe.shape[0]
        q, gate = torch.chunk(att.q_proj(xe).view(T, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(T, -1)
        q = att.q_norm(q).transpose(0, 1)
        k = att.k_norm(att.k_proj(xe).view(T, -1, hd)).transpose(0, 1)
        v = att.v_proj(xe).view(T, -1, hd).transpose(0, 1)
        P = 0 if past is None else past[0].shape[1]
        pos = torch.arange(pos0 + P, pos0 + P + T, device=xe.device).view(1, 1, T).expand(3, 1, T)
        cos, sin = self.rot(xe.unsqueeze(0), pos)
        qr, kr = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos.float(), sin.float())
        qr, kr = qr[0], kr[0]
        Kown = kr if past is None else torch.cat([past[0], kr], 1)
        Vown = v if past is None else torch.cat([past[1], v], 1)
        Hq = qr.shape[0]
        Hkv = Kop.shape[0]
        G = Hq // Hkv
        Vop0 = Vop
        if self.tr:
            Kop, Vop = self.translate(li, Kop, Vop)
        K = torch.cat([Kop.float(), Kown], 1)
        C = K.shape[1]
        s = torch.einsum("kgtd,kcd->kgtc", qr.view(Hkv, G, T, hd), K) * att.scaling
        col = torch.arange(C, device=xe.device).view(1, C)
        row = torch.arange(T, device=xe.device).view(T, 1)
        s = s.masked_fill((col > (C - T) + row).view(1, 1, T, C), float("-inf"))
        p = torch.softmax(s, -1)

        def contract(Vo, Vw):
            return torch.einsum("kgtc,kcd->kgtd", p, torch.cat([Vo.float(), Vw], 1))

        def heads_out(oh, a_):
            return a_.o_proj(oh.reshape(Hq, T, hd).transpose(0, 1).reshape(T, Hq * hd) * torch.sigmoid(gate))

        ob = contract(Vop, Vown)                                           # [Hkv,G,T,hd]
        o_base = heads_out(ob, att)
        y = self.Bmap(li, o_base)
        # distillation side: the UNtranslated operator values (tr), otherwise identical to y
        y_base = self.Bmap(li, heads_out(contract(Vop0, Vown), att)) if self.tr else y
        if li in self.dv:
            od = contract(ext["dv"], Vown)
            al = self.E2["alpha"][str(li)].view(Hkv, 1, 1, 1)
            y = self.Bmap(li, heads_out(ob + al * (od - ob), att))
        px = {} if past is None or len(past) < 3 else past[2]
        nx = {}
        if li in self.ms:
            a2 = self.att2[str(li)]
            v2 = a2.v_proj(xe).view(T, -1, hd).transpose(0, 1)
            V2own = v2 if "ms" not in px else torch.cat([px["ms"], v2], 1)
            o2 = heads_out(contract(ext["ms"], V2own), a2)
            y2 = o2 @ self.E2["B2"][str(li)].t() + self.E2["b2"][str(li)]
            y = y + self.E2["beta"][str(li)] * (y2 - y)
            nx["ms"] = V2own.detach()
        out = {"y_base": y_base, "o": o_base, "logits": s if need_logits else None}
        if li in self.rs:
            r3 = self.read_rs(li, x, ext["rs"][0], ext["rs"][1], pos0, px.get("rs"), need_logits)
            y = y + self.E2["gamma"][str(li)] * (r3["y"] - y)
            nx["rs"] = r3["past"]
            out["y3"], out["logits3"] = r3["y"], r3["logits"]
        out["y"] = y
        out["past"] = (Kown.detach(), Vown.detach()) + ((nx,) if nx else ())
        return out

    def translate(self, li, Kop, Vop):
        """Operator translator (tr): K [Hkv,N,hd] post-RoPE at positions 0..N-1 -> RoPE(k + k W_k + b_k) with
        k = RoPE⁻¹(K); V -> V + V W_v + b_v. Document-independent; zero init == identity (up to fp error)."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        k_ = str(li)
        P = self.E2["tr"]
        K, V = Kop.float(), Vop.float()
        N = K.shape[1]
        pos = torch.arange(N, device=K.device).view(1, 1, N).expand(3, 1, N)
        cos, sin = self.rot(K[:1].transpose(0, 1).reshape(1, N, -1), pos)
        cos, sin = cos.float(), sin.float()
        kp = apply_rotary_pos_emb(K[None], K[None], cos, -sin)[1][0]                 # undo RoPE
        dk = torch.einsum("hnd,hde->hne", kp, P[f"Wk{k_}"]) + P[f"bk{k_}"][:, None]
        # RoPE is linear: RoPE(k + dk) = K + RoPE(dk). Adding the rotated DELTA keeps zero init bit-exact (the first E2
        # tr pilots used RoPE(k + dk), whose fp32 round trip flipped 2/296 val items at step 0)
        Kt = K + apply_rotary_pos_emb(dk[None], dk[None], cos, sin)[1][0]
        Vt = V + torch.einsum("hnd,hde->hne", V, P[f"Wv{k_}"]) + P[f"bv{k_}"][:, None]
        return Kt, Vt

    def read_rs(self, li, x, Kop, Vop, pos0: int, past=None, need_logits: bool = False):
        """Independent D13-style read of encoder layer rs[li] with A3/M3 and that layer's frozen projections."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        from xenc.d13_read import _attend
        att = self.att3[str(li)]
        hd = att.head_dim
        k_ = str(li)
        xe = self.E2["A3"][k_](x)
        if self.mlp_hidden:
            xe = xe + self.E2["M3"][k_](x)
        T = xe.shape[0]
        q, gate = torch.chunk(att.q_proj(xe).view(T, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(T, -1)
        q = att.q_norm(q).transpose(0, 1)
        k = att.k_norm(att.k_proj(xe).view(T, -1, hd)).transpose(0, 1)
        v = att.v_proj(xe).view(T, -1, hd).transpose(0, 1)
        P = 0 if past is None else past[0].shape[1]
        pos = torch.arange(pos0 + P, pos0 + P + T, device=xe.device).view(1, 1, T).expand(3, 1, T)
        cos, sin = self.rot(xe.unsqueeze(0), pos)
        qr, kr = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos.float(), sin.float())
        qr, kr = qr[0], kr[0]
        Kown = kr if past is None else torch.cat([past[0], kr], 1)
        Vown = v if past is None else torch.cat([past[1], v], 1)
        r = _attend(qr, Kop.float(), Vop.float(), Kown, Vown, att.scaling, need_logits)
        o = att.o_proj(r["o"] * torch.sigmoid(gate))
        y = o @ self.E2["B3"][k_].t() + self.E2["b3"][k_]
        return {"y": y, "o": o, "logits": r.get("logits"), "past": (Kown.detach(), Vown.detach())}

    def extra_dist(self, Rrec: dict, rec: dict, op: dict, n_op: int, rel_w: float = 1.0):
        """Mean over rs layers of KL(enc layer-e3 attention ‖ path-3 attention) + rel_w·relMSE(y3, B3_0 o_enc(e3))."""
        from xenc.d13_read import d13_terms, teacher_logits
        terms = []
        if self.B3_0 is None:
            self.B3_0 = {li: (self.E2["B3"][str(li)].detach().clone(), self.E2["b3"][str(li)].detach().clone())
                         for li in self.rs}
        for li, e3 in self.rs.items():
            r = Rrec.get(li)
            if r is None or r.get("logits3") is None:
                continue
            lt = teacher_logits(rec[e3]["q"], op[e3][0], rec[e3]["k"], self.att3[str(li)].scaling)
            W0, b0 = self.B3_0[li]
            yt = (rec[e3]["out"].float() @ W0.t() + b0).detach()
            d = d13_terms(r["logits3"], lt, r["y3"], yt, n_op, False)
            terms.append(d["kl"] + rel_w * d["rel"])
        return torch.stack(terms).mean() if terms else torch.zeros(())


class _DVHarvest:
    """Forward hooks on the encoder's decoder layers e' during the operator prefill -> V'_e per dv reader layer."""

    def __init__(self, enc, itf: E2Interface):
        from deltaskill.backbone_utils import decoder_layers
        self.L, self.itf, self.h = decoder_layers(enc), itf, {}
        self._hooks = []
        for e1 in sorted(set(itf.dv.values())):
            def hk(mod, inp, out, e1=e1):
                self.h[e1] = (out[0] if isinstance(out, tuple) else out)[0].detach()
            self._hooks.append(self.L[e1].register_forward_hook(hk))

    @torch.no_grad()
    def finish(self, dev) -> dict:
        for h in self._hooks:
            h.remove()
        out = {}
        for li, e1 in self.itf.dv.items():
            e = self.itf.layer_map[li]
            lay = self.L[e]
            att = lay.self_attn
            hs = self.h[e1]
            V = att.v_proj(lay.input_layernorm(hs)).view(hs.shape[0], -1, att.head_dim).transpose(0, 1)
            out[f"dv{li}"] = V.to(dev)
        self.h = {}
        return out


# ----------------------------------------------------------------------------- reader patch
class E2Reader(D13Reader):
    """D13Reader whose patched attention passes the E2 extras (deep values, extra-source values) to itf.read and
    stores the UNMIXED read-out as rec["y"] (the distillation regularizer's target side)."""

    def __init__(self, model, itf: E2Interface):
        super().__init__(model, itf)
        for li, m in self.mods.items():
            m.forward = types.MethodType(_e2_forward(self, li), m)

    def ext(self, li) -> dict:
        itf = self.itf
        return {"dv": self.op.get(f"dv{li}") if li in itf.dv else None,
                "ms": self.op[itf.ms[li]][1] if li in itf.ms else None,
                "rs": self.op[itf.rs[li]] if li in itf.rs else None}


def _e2_forward(R: E2Reader, li: int):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        out, w = R._orig[li](hidden_states, position_embeddings, attention_mask,
                             past_key_values=past_key_values, **kwargs)
        if R.mode == "off":
            return out, w
        itf = R.itf
        e = itf.layer_map[li]
        K, V = R.op[e]
        ext = R.ext(li)
        x = hidden_states[0].float()
        own = out[0].float()
        if R.mode == "student":
            res = itf.read(li, x, K, V, R.n_op, past=R.past.get(li), ext=ext)
            R.past[li] = res["past"]
            y = own + itf.gate(li) * (res["y"] - own)
        elif R.mode == "e2e":
            nl = bool(R.need_logits)
            if torch.is_grad_enabled() and getattr(R, "ckpt", False):
                from torch.utils.checkpoint import checkpoint

                rs_ = li in itf.rs

                def _rd(x_):
                    r_ = itf.read(li, x_, K, V, R.n_op, need_logits=nl, ext=ext)
                    return ((r_["y"], r_["y_base"], r_["o"]) + ((r_["logits"],) if nl else ())
                            + ((r_["y3"], r_["logits3"]) if rs_ and nl else ()))
                t_ = checkpoint(_rd, x, use_reentrant=False)
                res = {"y": t_[0], "y_base": t_[1], "o": t_[2], "logits": t_[3] if nl else None}
                if rs_ and nl:
                    res["y3"], res["logits3"] = t_[4], t_[5]
            else:
                res = itf.read(li, x, K, V, R.n_op, need_logits=nl, ext=ext)
            R.rec[li] = {"y": res["y_base"], "o": res["o"], "logits": res["logits"], "y_mix": res["y"],
                         "y3": res.get("y3"), "logits3": res.get("logits3")}
            y = own + itf.gate(li) * (res["y"] - own)
        elif R.mode == "train":
            with torch.set_grad_enabled(R.grad):
                res = itf.read(li, x, K, V, R.n_op, need_logits=True, ext=ext)
            R.rec[li] = {"y": res["y_base"], "o": res["o"], "logits": res["logits"], "y_mix": res["y"]}
            y = res["y"].detach() if R.onpolicy else R.inject[li]
        else:                                                           # "teacher" (stage-1 ridge statistics)
            R.xcap[li] = x.detach()
            y = R.inject[li]
        return y.to(out.dtype).unsqueeze(0), w

    return forward


def upgrade(itf: D13Interface, enc, dv: dict, ms: dict, B2: dict | None, rs: dict | None = None,
            B3: dict | None = None, tr: bool = False) -> E2Interface:
    """A loaded D13 base -> E2Interface with identical A / MLP / B / gates and zero-init mixing (step 0 == base).
    rs paths: A3 / M3 start as copies of the base A_ℓ / MLP_ℓ."""
    B = None if itf.identity_B else {li: itf.B[str(li)].detach().cpu() for li in itf.layers}
    new = E2Interface(enc, itf.layer_map, itf.d_r, itf.d_e, B=B, mlp_hidden=itf.mlp_hidden,
                      e2={"dv": dv, "ms": ms, "rs": rs or {}, "tr": int(bool(tr))}, B2=B2, B3=B3)
    sd = {k: v for k, v in itf.state_dict().items() if not k.startswith("D.")}
    missing, unexpected = new.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith(("E2.", "att2.", "att3.")) for k in missing), missing
    with torch.no_grad():
        for li in new.rs:
            new.E2["A3"][str(li)].load_state_dict(itf.A[str(li)].state_dict())
            if itf.mlp_hidden:
                new.E2["M3"][str(li)].load_state_dict(itf.M[str(li)].state_dict())
    return new


def load_B2(path: str, ms: dict) -> dict | None:
    """(also used for the rs B3 init: the probe's ms(ℓ, e) ridge maps live in the same layer-e o_proj space)"""
    return _load_maps(path, ms)


def _load_maps(path: str, ms: dict) -> dict | None:
    """Ridge init for the ms maps (scripts/e2_probe.py --save_B2): {li: {"W": [d_r,d_e], "b": [d_r]}}."""
    if not ms:
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for li, e2 in ms.items():
        k = f"ms{li}_{e2}"
        assert k in blob, f"{k} not in {path} ({sorted(blob)[:8]})"
        out[li] = {"W": blob[k]["W"].float(), "b": blob[k]["b"].float()}
    return out


__all__ = ["E2Interface", "E2Reader", "parse_pairs", "upgrade", "load_B2"]
