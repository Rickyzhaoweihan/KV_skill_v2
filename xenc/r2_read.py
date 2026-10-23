"""R2: query-side read forms for the D13 paper-form read (xenc/d13_read.py). Default off. The operator (U, W), the
read layers, A / MLP and the write-back B stay as they are. Only the path from the reader's state h at a read layer to
the vector written back changes. Every form starts EXACTLY at the parent base (zero-init mixing / scale parameters),
so step 0 reproduces the base bit for bit.

Notation (per read layer l, query head h, reader row t): x_t = RMSNorm_l(h_t) (the attention input), xe = A(x) (+MLP),
q = RoPE(q_norm(W_q xe)), s_tj = q·k_j · scale (raw D13 logits over operator + own causal rows), heads = softmax(s)·V,
o = o_proj(heads ⊙ σ(gate)), y = B o, write-back y_attn = own + g (y − own).

  temp   learned per-head sharpness: s' = s · exp(θ_h), θ init 0. The encoder's q_norm fixes |q|, so A can rotate the
         query but never sharpen the read (E4: a fixed temp 1.25 helped). Length-independent (L1 owns log-N forms).
  hop    two-hop re-query inside the layer: read once (hop 1, as the base), fold the result into the residual stream,
         re-normalise and query again with the same A / projections:
             x2 = RMSNorm_l(h + own + g (y1 − own)),  q2 from A(x2),  heads2 = softmax(s2)·V (same columns)
             heads = heads1 + β_h (heads2 − heads1)      β per query head, init 0
         The own-row keys / values are hop 1's (unchanged), so incremental decode == the full pass.
  tgate  token-conditional write-back gate: g_t = g + wᵀ x_t / sqrt(d_r), w init 0 (the reader can damp reads that are
         irrelevant at row t).
  ctx    query from more context: the query input adds a low-rank map of the causal mean of the rows so far,
             xq_t = A(x_t) + U V c_t,  c_t = mean_{t' <= t} x_t'   (U init 0; V random from a private generator)
         (the reader prompt is doc tail + question + generated tokens, so c_t pools the question states).
         Keys / values of the own rows stay A(x); the running (sum, count) is carried in `past` for decode.
  mq     multi-query: J extra queries per head, xe_j = A(x) + U_j V_j x (U_j, V_j small random, private generator),
             heads = heads0 + Σ_j β_jh (heads_j − heads0)   β init 0
         (the extra queries differ from the base at init, so β gets gradient from step 1).

Distillation (register_d13 --dist_w): the returned logits are the RAW hop-1 / base-query logits (no temperature), and
rec["y"] is the unmixed base-query read-out y_base, so the stage-1 term keeps constraining the query DIRECTION against
the encoder teacher at the teacher's own temperature; the new parts are trained by the task loss (KL to text + CE).
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn as nn

from xenc.d13_read import D13Interface, D13Reader

FORMS = ("temp", "hop", "tgate", "ctx", "mq")


def parse_r2(temp: int = 0, hop: int = 0, tgate: int = 0, ctx: int = 0, mq: int = 0, mq_rank: int = 16,
             mq_init: float = 0.1, seed: int = 0) -> dict:
    """CLI -> r2 config ({} = every form off). ctx = rank of the context map (0 = off); mq = number of extra queries."""
    if not (int(temp) or int(hop) or int(tgate) or int(ctx) or int(mq)):
        return {}
    assert int(ctx) >= 0 and int(mq) >= 0 and int(mq_rank) > 0
    return {"temp": int(bool(int(temp))), "hop": int(bool(int(hop))), "tgate": int(bool(int(tgate))),
            "ctx": int(ctx), "mq": int(mq), "mq_rank": int(mq_rank), "mq_init": float(mq_init), "seed": int(seed)}


def _flat(heads):
    """[Hkv,G,T,hd] -> [T, Hq·hd] (exactly the reshape of xenc.d13_read._attend)."""
    Hkv, G, T, hd = heads.shape
    return heads.reshape(Hkv * G, T, hd).transpose(0, 1).reshape(T, Hkv * G * hd)


class R2Interface(D13Interface):
    """D13Interface + the R2 query-side read forms (trainable parameters in self.R2)."""

    def __init__(self, enc, layer_map: dict, d_r: int, d_e: int, B: dict | None = None, mlp_hidden: int = 0,
                 r2: dict | None = None):
        super().__init__(enc, layer_map, d_r, d_e, B=B, mlp_hidden=mlp_hidden)
        self.r2 = dict(r2 or {})
        assert self.r2, "R2Interface needs an r2 config (parse_r2)"
        a0 = self.att[str(self.layers[0])]
        self.r2_hd = a0.head_dim
        self.r2_hq = a0.q_proj.out_features // (2 * a0.head_dim)
        self.r2_hkv = a0.k_proj.out_features // a0.head_dim
        g = torch.Generator().manual_seed(1000 + int(self.r2.get("seed", 0)))   # private: never touches the global RNG
        P = {}
        for li in self.layers:
            if self.r2["temp"]:
                P[f"t{li}"] = nn.Parameter(torch.zeros(self.r2_hq))
            if self.r2["hop"]:
                P[f"b{li}"] = nn.Parameter(torch.zeros(self.r2_hq))
            if self.r2["tgate"]:
                P[f"w{li}"] = nn.Parameter(torch.zeros(self.d_r))
            if self.r2["ctx"]:
                r = self.r2["ctx"]
                P[f"cU{li}"] = nn.Parameter(torch.zeros(self.d_e, r))
                P[f"cV{li}"] = nn.Parameter(torch.randn(r, self.d_r, generator=g) / self.d_r ** 0.5)
            if self.r2["mq"]:
                r = self.r2["mq_rank"]
                P[f"mb{li}"] = nn.Parameter(torch.zeros(self.r2["mq"], self.r2_hq))
                for j in range(self.r2["mq"]):
                    # placeholder scale; upgrade_r2 rescales U_j relative to the loaded A (see there)
                    P[f"mU{li}_{j}"] = nn.Parameter(torch.randn(self.d_e, r, generator=g) / (self.d_e * r) ** 0.5)
                    P[f"mV{li}_{j}"] = nn.Parameter(torch.randn(r, self.d_r, generator=g) / self.d_r ** 0.5)
        self.R2 = nn.ParameterDict(P)
        self.r2_on = True                    # False: the same checkpoint read with the plain D13 read (ablation)

    # -------------------------------------------------------------- bookkeeping
    def config(self) -> dict:
        c = super().config()
        c["r2"] = dict(self.r2)
        return c

    def make_reader(self, model):
        return R2Reader(model, self)

    def r2_params(self) -> dict:
        """{"scal": per-head scalars (θ, β), "mat": vectors / matrices (w, U, V)}."""
        sc, mt = [], []
        for k, p in self.R2.items():
            (sc if k[0] in ("t", "b") or k.startswith("mb") else mt).append(p)
        return {"scal": sc, "mat": mt}

    def r2_named_grads(self) -> dict:
        return {k: (round(float(p.grad.norm()), 6) if p.grad is not None else None) for k, p in self.R2.items()}

    @torch.no_grad()
    def r2_state(self) -> dict:
        out = {}
        for li in self.layers:
            d = {}
            if self.r2["temp"]:
                m = self.R2[f"t{li}"].exp()
                d["temp"] = [round(float(m.min()), 4), round(float(m.mean()), 4), round(float(m.max()), 4)]
            if self.r2["hop"]:
                b = self.R2[f"b{li}"]
                d["hop_b"] = [round(float(b.min()), 4), round(float(b.mean()), 4), round(float(b.max()), 4)]
            if self.r2["tgate"]:
                d["tg_w"] = round(float(self.R2[f"w{li}"].norm()), 5)
            if self.r2["ctx"]:
                d["ctx_U"] = round(float(self.R2[f"cU{li}"].norm()), 5)
            if self.r2["mq"]:
                b = self.R2[f"mb{li}"]
                d["mq_b"] = [round(float(b.min()), 4), round(float(b.abs().mean()), 4), round(float(b.max()), 4)]
            out[str(li)] = d
        return out

    def gate_t(self, li, x):
        """Write-back gate: the per-layer scalar g (bit-identical to D13) or, with tgate, g + wᵀx/sqrt(d_r) [T,1]."""
        g = self.gate(li)
        if not (self.r2_on and self.r2["tgate"]):
            return g
        return g + (x @ self.R2[f"w{li}"]).unsqueeze(-1) / math.sqrt(self.d_r)

    # -------------------------------------------------------------- the read
    def _scores(self, li, q, K, T, scale):
        """q [Hq,T,hd] post-RoPE, K [Hkv,C,hd] -> (raw masked logits s0, logits with temp) [Hkv,G,T,C]."""
        Hq, _, hd = q.shape
        Hkv = K.shape[0]
        G = Hq // Hkv
        C = K.shape[1]
        su = torch.einsum("kgtd,kcd->kgtc", q.view(Hkv, G, T, hd), K) * scale
        col = torch.arange(C, device=q.device).view(1, C)
        row = torch.arange(T, device=q.device).view(T, 1)
        mask = (col > (C - T) + row).view(1, 1, T, C)
        s0 = su.masked_fill(mask, float("-inf"))
        s = s0
        if self.r2_on and self.r2["temp"]:
            # scale BEFORE masking: (-inf) · m would give d/dm = -inf · 0 = NaN in backward
            s = (su * self.R2[f"t{li}"].exp().view(Hkv, G, 1, 1)).masked_fill(mask, float("-inf"))
        return s0, s

    def _extra_read(self, li, q, K, V, T, scale):
        """heads = softmax(s(q))·V for an extra query (hop 2 / mq). With self.sub_ckpt (set by R2Reader under --mem_safe)
        and grad on, each extra read is checkpointed on its own, so in backward only ONE read's [Hkv,G,T,C]
        intermediates are alive at a time (R2_all OOMed on 90k-column CLIPPER operators with 4 reads per layer).
        Identical values and gradients either way (tests/test_r2.py)."""
        def f(q_, K_, V_):
            return torch.einsum("kgtc,kcd->kgtd", torch.softmax(self._scores(li, q_, K_, T, scale)[1], -1), V_)
        if getattr(self, "sub_ckpt", False) and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(f, q, K, V, use_reentrant=False)
        return f(q, K, V)

    def _query(self, att, xq, T, cos, sin):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        hd = att.head_dim
        q, gate = torch.chunk(att.q_proj(xq).view(T, -1, hd * 2), 2, dim=-1)
        q = att.q_norm(q).transpose(0, 1)
        qr = apply_rotary_pos_emb(q.unsqueeze(0), q.unsqueeze(0), cos.float(), sin.float())[0][0]
        return qr, gate.reshape(T, -1)

    def read(self, li, x, Kop, Vop, pos0: int, past=None, need_logits: bool = False, ext: dict | None = None):
        """Same signature / outputs as D13Interface.read (+ ext = {"h", "own", "g", "norm"} for hop, given by R2Reader).
        -> {"y" (mixed, written back), "y_base" (base-query read-out, for distillation), "o", "logits" (raw), "past"}."""
        if not self.r2_on:
            res = super().read(li, x, Kop, Vop, pos0, past=past, need_logits=need_logits)
            res["y_base"] = res["y"]
            return res
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        k_ = str(li)
        att = self.att[k_]
        hd = att.head_dim
        xe = self.map_x(li, x)
        T = xe.shape[0]
        px = past[2] if past is not None and len(past) > 2 else {}
        nx = {}
        xq = xe
        if self.r2["ctx"]:
            s_prev, n_prev = px.get("ctx", (torch.zeros(self.d_r, device=x.device, dtype=x.dtype), 0))
            cum = s_prev.unsqueeze(0) + x.cumsum(0)
            cnt = torch.arange(n_prev + 1, n_prev + T + 1, device=x.device, dtype=x.dtype).unsqueeze(-1)
            c = cum / cnt
            xq = xe + (c @ self.R2[f"cV{li}"].t()) @ self.R2[f"cU{li}"].t()
            nx["ctx"] = (cum[-1].detach(), n_prev + T)
        k = att.k_norm(att.k_proj(xe).view(T, -1, hd)).transpose(0, 1)
        v = att.v_proj(xe).view(T, -1, hd).transpose(0, 1)
        P = 0 if past is None else past[0].shape[1]
        pos = torch.arange(pos0 + P, pos0 + P + T, device=xe.device).view(1, 1, T).expand(3, 1, T)
        cos, sin = self.rot(xe.unsqueeze(0), pos)
        q, gate = torch.chunk(att.q_proj(xq).view(T, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(T, -1)
        q = att.q_norm(q).transpose(0, 1)
        qr, kr = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos.float(), sin.float())
        qr, kr = qr[0], kr[0]
        Kown = kr if past is None else torch.cat([past[0], kr], 1)
        Vown = v if past is None else torch.cat([past[1], v], 1)
        K = torch.cat([Kop.float(), Kown], 1)
        V = torch.cat([Vop.float(), Vown], 1)
        scale = att.scaling * self.read_scale_mult(Kop.shape[1])
        s0, s = self._scores(li, qr, K, T, scale)
        heads0 = torch.einsum("kgtc,kcd->kgtd", torch.softmax(s, -1), V)
        Hkv, G = heads0.shape[0], heads0.shape[1]
        sg = torch.sigmoid(gate)
        heads = heads0
        if self.r2["hop"]:
            assert ext is not None and ext.get("h") is not None, "R2 hop needs the R2Reader (h / own / g / norm)"
            y1 = self.Bmap(li, att.o_proj(_flat(heads0) * sg))
            own = ext["own"]
            h_mid = ext["h"] + own + ext["g"] * (y1 - own)
            nm = ext["norm"]
            x2 = nm._norm(h_mid.float()) * (1.0 + nm.weight.float())
            xq2 = self.map_x(li, x2)
            if self.r2["ctx"]:
                xq2 = xq2 + (c @ self.R2[f"cV{li}"].t()) @ self.R2[f"cU{li}"].t()
            q2, _ = self._query(att, xq2, T, cos, sin)
            heads2 = self._extra_read(li, q2, K, V, T, scale)
            heads = heads + self.R2[f"b{li}"].view(Hkv, G, 1, 1) * (heads2 - heads0)
        if self.r2["mq"]:
            for j in range(self.r2["mq"]):
                xej = xe + (x @ self.R2[f"mV{li}_{j}"].t()) @ self.R2[f"mU{li}_{j}"].t()
                if self.r2["ctx"]:
                    xej = xej + (c @ self.R2[f"cV{li}"].t()) @ self.R2[f"cU{li}"].t()
                qj, _ = self._query(att, xej, T, cos, sin)
                hj = self._extra_read(li, qj, K, V, T, scale)
                heads = heads + self.R2[f"mb{li}"][j].view(Hkv, G, 1, 1) * (hj - heads0)
        o = att.o_proj(_flat(heads) * sg)
        y = self.Bmap(li, o)
        y_base = y if heads is heads0 else self.Bmap(li, att.o_proj(_flat(heads0) * sg))
        return {"y": y, "y_base": y_base, "o": o, "logits": s0 if need_logits else None,
                "past": (Kown.detach(), Vown.detach()) + ((nx,) if nx else ())}


# ----------------------------------------------------------------------------- reader patch
class R2Reader(D13Reader):
    """D13Reader that passes the reader's residual input h, the stock attention output and the (token) gate to the
    R2 read (hop), applies the token gate on the write-back, and stores the UNMIXED base-query read-out as rec["y"]
    (the distillation regularizer's target side)."""

    def __init__(self, model, itf: R2Interface):
        super().__init__(model, itf)
        from deltaskill.backbone_utils import decoder_layers
        L = decoder_layers(model)
        self.h_in: dict = {}
        self.norms = {li: L[li].input_layernorm for li in itf.layers}
        self._hooks = []
        for li in itf.layers:
            def pre(mod, args, kwargs, li=li):
                self.h_in[li] = args[0] if args else kwargs["hidden_states"]
            self._hooks.append(L[li].register_forward_pre_hook(pre, with_kwargs=True))
        for li, m in self.mods.items():
            m.forward = types.MethodType(_r2_forward(self, li), m)

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        super().detach()


def _r2_forward(R: R2Reader, li: int):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        out, w = R._orig[li](hidden_states, position_embeddings, attention_mask,
                             past_key_values=past_key_values, **kwargs)
        if R.mode == "off":
            return out, w
        itf = R.itf
        e = itf.layer_map[li]
        K, V = R.op[e]
        x = hidden_states[0].float()
        own = out[0].float()
        h = R.h_in[li][0].float() if itf.r2_on and itf.r2["hop"] else None

        def ext_of(x_, own_, h_):
            if h_ is None:
                return None
            return {"h": h_, "own": own_, "g": itf.gate_t(li, x_), "norm": R.norms[li]}

        if R.mode == "student":
            res = itf.read(li, x, K, V, R.n_op, past=R.past.get(li), ext=ext_of(x, own, h))
            R.past[li] = res["past"]
            y = own + itf.gate_t(li, x) * (res["y"] - own)
        elif R.mode == "e2e":
            nl = bool(R.need_logits)
            itf.sub_ckpt = bool(getattr(R, "ckpt", False))          # --mem_safe: also checkpoint each extra read
            if torch.is_grad_enabled() and getattr(R, "ckpt", False):
                from torch.utils.checkpoint import checkpoint

                def _rd(x_, own_, h_):
                    r_ = itf.read(li, x_, K, V, R.n_op, need_logits=nl, ext=ext_of(x_, own_, h_))
                    return (r_["y"], r_["y_base"], r_["o"]) + ((r_["logits"],) if nl else ())
                t_ = checkpoint(_rd, x, own, h, use_reentrant=False)
                res = {"y": t_[0], "y_base": t_[1], "o": t_[2], "logits": t_[3] if nl else None}
            else:
                res = itf.read(li, x, K, V, R.n_op, need_logits=nl, ext=ext_of(x, own, h))
            R.rec[li] = {"y": res["y_base"], "o": res["o"], "logits": res["logits"], "y_mix": res["y"]}
            y = own + itf.gate_t(li, x) * (res["y"] - own)
        elif R.mode == "train":
            with torch.set_grad_enabled(R.grad):
                res = itf.read(li, x, K, V, R.n_op, need_logits=True, ext=ext_of(x, own, h))
            R.rec[li] = {"y": res["y_base"], "o": res["o"], "logits": res["logits"], "y_mix": res["y"]}
            y = res["y"].detach() if R.onpolicy else R.inject[li]
        else:                                                           # "teacher" (stage-1 ridge statistics)
            R.xcap[li] = x.detach()
            y = R.inject[li]
        return y.to(out.dtype).unsqueeze(0), w

    return forward


def upgrade_r2(itf: D13Interface, enc, r2: dict) -> R2Interface:
    """A loaded plain D13 base -> R2Interface with identical A / MLP / B / bias / gates and zero-init R2 mixing."""
    assert type(itf) is D13Interface, f"R2 read forms on a {type(itf).__name__} init are not supported"
    assert not itf.delta_rank
    B = None if itf.identity_B else {li: itf.B[str(li)].detach().cpu() for li in itf.layers}
    # constructing the interface runs nn.Linear inits (global CPU RNG); restore the RNG so a flag-on run consumes the
    # same random stream as its flag-off control (the R2 random tensors use a private generator)
    rng_state = torch.random.get_rng_state()
    try:
        new = R2Interface(enc, itf.layer_map, itf.d_r, itf.d_e, B=B, mlp_hidden=itf.mlp_hidden, r2=r2)
    finally:
        torch.random.set_rng_state(rng_state)
    sd = {k: v for k, v in itf.state_dict().items() if not k.startswith("D.")}
    missing, unexpected = new.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("R2.") for k in missing), missing
    if r2.get("mq"):
        # extra queries: U_j entries ~ N(0, 1/(d_e r)) and V_j x ~ N(0, mean x²) per entry, so
        # ‖U_j V_j x‖ ≈ ‖x‖_rms and ‖A x‖ ≈ ‖A‖_F ‖x‖_rms; scale U_j by mq_init·‖A‖_F -> ‖ΔA_j x‖ ≈ mq_init·‖A x‖
        with torch.no_grad():
            for li in new.layers:
                s = float(r2["mq_init"]) * float(new.A[str(li)].weight.norm())
                for j in range(r2["mq"]):
                    new.R2[f"mU{li}_{j}"].mul_(s)
    for k in ("read_tau", "read_logn_ref"):
        if hasattr(itf, k):
            setattr(new, k, getattr(itf, k))
    return new.to(itf.g.device)


@torch.no_grad()
def set_temp_init(itf: R2Interface, mult: float):
    """Exploration init (default 1.0 = off): start every learned head temperature at `mult` (θ = ln mult, still
    trainable). Step 0 then differs from the base BY DESIGN (e.g. E4's fixed x1.25)."""
    assert itf.r2.get("temp"), "--r2_temp_init needs --r2_temp 1"
    for li in itf.layers:
        itf.R2[f"t{li}"].fill_(math.log(float(mult)))


__all__ = ["R2Interface", "R2Reader", "parse_r2", "upgrade_r2", "set_temp_init", "FORMS"]
