"""D16: WHAT the reader-side query map A sees and WHICH keys U it addresses — a generalisation of D13's read
(xenc/d13_read.py, not modified). Same paper form: the operator is harvested once from the small encoder's frozen
prefill of the document (never modified); only the reader runs at inference; the read replaces the reader's attention
output at a few full-attention layers (gated: own + g·(read − own)). With the default cfg this IS the D13 read
(tests/test_d16.py checks equality).

Per reader layer ℓ (paired with encoder layer e = layer_map[ℓ]):
    x_in   = concat of features named in cfg["feats"]:   "x"  RMSNorm_ℓ(h_ℓ) (the layer input, D13's choice)
                                                           "L-k" RMSNorm_{ℓ-k}(h_{ℓ-k}) (earlier reader layers)
                                                           "q"  the reader's own q_norm(q_proj(x)) (pre-RoPE)
             (+ cfg["window"]=W: attention-pooled x_{t-1..t-W+1}, a small learned pool, concatenated)
    x̃      = A x_in (+ residual MLP, + per-row mixture of E low-rank experts)                A: d_in -> d_e
    q, gate = q_norm(W_q x̃) (or, cfg["qhead"], a per-head q map Q_h x_in decoupled from k/v/gate), W_g x̃
    keys   = U: the operator's post-RoPE encoder keys K_e (cfg rope=False: pre-RoPE keys and queries;
             cfg keymap: U = the encoder's residual states X_e at the document, keys k_norm(P X_e) with P trainable,
             init W_k — the operator stays the fixed harvest X_e);  cfg kcat {ℓ: e2 | [e2, e3, ...]}: add β_{j,h}·(q_j·K_ej) scores
             from further encoder layers' keys (adjacent layer, or all six: still the fixed harvest; values stay W_e)
    p      = softmax(q·[U ; k_own]ᵀ · scale / τ_h)         (cfg tau: learned per-head τ; tau/topk at call time)
    y      = B W_o (σ(gate) ⊙ p·[W ; v_own])                 B: D10's closed-form encoder-side post map
"""

from __future__ import annotations

import copy
import types

import torch
import torch.nn as nn

import xenc  # noqa: F401

FORMAT = "xenc.d16_read.v1"
DEFAULT_CFG = {"feats": ["x"], "mlp_hidden": 0, "window": 0, "moe": 0, "moe_rank": 256, "qhead": False,
               "tau": False, "keymap": False, "rope": True, "kcat": None}


def _rotary(model):
    from xenc.xattn_read import _rotary as r
    return r(model)


def feat_dim(name: str, d_r: int, d_q: int) -> int:
    return d_q if name == "q" else d_r


def lag_of(name: str) -> int:
    return int(name[2:]) if name.startswith("L-") else 0


class D16Interface(nn.Module):
    def __init__(self, enc, layer_map: dict, d_r: int, d_e: int, B: dict | None = None, cfg: dict | None = None,
                 d_q: int = 0):
        super().__init__()
        from deltaskill.backbone_utils import decoder_layers
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.cfg["feats"] = list(self.cfg["feats"])
        self.cfg["kcat"] = {int(k): ([int(x) for x in v] if isinstance(v, (list, tuple)) else [int(v)])
                            for k, v in (self.cfg["kcat"] or {}).items()} or None
        self.layer_map = {int(k): int(v) for k, v in layer_map.items()}
        self.layers = sorted(self.layer_map)
        self.d_r, self.d_e, self.d_q = int(d_r), int(d_e), int(d_q)
        c = self.cfg
        L = decoder_layers(enc)
        self.att = nn.ModuleDict({str(li): copy.deepcopy(L[e].self_attn).float().requires_grad_(False)
                                  for li, e in self.layer_map.items()})
        self.rot = copy.deepcopy(_rotary(enc)).float()
        a0 = self.att[str(self.layers[0])]
        self.hd, self.n_kv = int(a0.head_dim), int(a0.k_proj.out_features // a0.head_dim)
        self.n_q = int(a0.q_proj.out_features // (2 * a0.head_dim))
        d_in = self.d_in()
        self.A, self.M, self.Qh, self.Kp = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        self.R, self.EU, self.EV = nn.ModuleDict(), nn.ParameterDict(), nn.ParameterDict()
        self.win_u, self.win_b = nn.ParameterDict(), nn.ParameterDict()
        self.A2, self.att2 = nn.ModuleDict(), nn.ModuleDict()
        for li in self.layers:
            s = str(li)
            lin = nn.Linear(d_in, self.d_e)
            with torch.no_grad():
                lin.bias.zero_()
                lin.weight.copy_(torch.eye(self.d_e, d_in))
            self.A[s] = lin
            if c["mlp_hidden"]:
                m = nn.Sequential(nn.Linear(d_in, c["mlp_hidden"]), nn.GELU(), nn.Linear(c["mlp_hidden"], self.d_e))
                nn.init.zeros_(m[2].weight)
                nn.init.zeros_(m[2].bias)
                self.M[s] = m
            if c["moe"]:
                E, r = int(c["moe"]), int(c["moe_rank"])
                self.R[s] = nn.Linear(d_in, E)
                nn.init.zeros_(self.R[s].weight)
                nn.init.zeros_(self.R[s].bias)
                self.EU[s] = nn.Parameter(torch.randn(E, d_in, r) / d_in ** 0.5)
                self.EV[s] = nn.Parameter(torch.zeros(E, r, self.d_e))
            if c["window"]:
                self.win_u[s] = nn.Parameter(torch.zeros(self.d_r))
                self.win_b[s] = nn.Parameter(torch.zeros(int(c["window"]) - 1))
            if c["qhead"]:
                self.Qh[s] = nn.Linear(d_in, self.n_q * self.hd)
                self.init_qhead(li)
            if c["keymap"]:
                self.Kp[s] = copy.deepcopy(self.att[s].k_proj).float().requires_grad_(True)
            if c["kcat"]:
                for e2 in c["kcat"][li]:
                    k2 = f"{li}_{e2}"
                    self.att2[k2] = copy.deepcopy(L[e2].self_attn).float().requires_grad_(False)
                    l2 = nn.Linear(d_in, self.d_e)
                    with torch.no_grad():
                        l2.weight.copy_(lin.weight)
                        l2.bias.copy_(lin.bias)
                    self.A2[k2] = l2
        n = len(self.layers)
        nx = max((len(v) for v in c["kcat"].values()), default=0) if c["kcat"] else 0
        self.beta = nn.Parameter(torch.zeros(n, nx, self.n_q)) if c["kcat"] else None
        self.logtau = nn.Parameter(torch.zeros(n, self.n_q)) if c["tau"] else None
        self.identity_B = B is None
        self.B = nn.ParameterDict()
        if B is not None:
            for li in self.layers:
                self.B[str(li)] = nn.Parameter(B[li].float().clone(), requires_grad=False)
        self.g = nn.Parameter(torch.ones(n))
        self._gi = {li: i for i, li in enumerate(self.layers)}

    # ------------------------------------------------------------------ structure
    def d_in(self) -> int:
        d = sum(feat_dim(f, self.d_r, self.d_q) for f in self.cfg["feats"])
        return d + (self.d_r if self.cfg["window"] else 0)

    def lags(self) -> set:
        return {lag_of(f) for f in self.cfg["feats"] if f.startswith("L-")}

    def needs_q(self) -> bool:
        return "q" in self.cfg["feats"]

    def init_qhead(self, li):
        """Q_h := the composite W_q[q rows] · A (so qhead == shared at init)."""
        s = str(li)
        att = self.att[s]
        hd = self.hd
        rows = torch.cat([torch.arange(h * 2 * hd, h * 2 * hd + hd) for h in range(self.n_q)])
        dv = self.A[s].weight.device
        W = att.q_proj.weight[rows.to(att.q_proj.weight.device)].float().to(dv)
        assert W.device == self.A[s].weight.device == self.Qh[s].weight.device, "init_qhead: device mismatch"
        with torch.no_grad():
            self.Qh[s].weight.copy_(W @ self.A[s].weight)
            b = W @ self.A[s].bias
            if att.q_proj.bias is not None:
                b = b + att.q_proj.bias[rows.to(att.q_proj.bias.device)].float().to(dv)
            self.Qh[s].bias.copy_(b)

    def param_groups(self) -> dict:
        """{name: [params]} of the trainable interface (A family, extras, B, g)."""
        gr = {"A": list(self.A.parameters()), "M": list(self.M.parameters()),
              "moe": list(self.R.parameters()) + list(self.EU.parameters()) + list(self.EV.parameters()),
              "win": list(self.win_u.parameters()) + list(self.win_b.parameters()),
              "qhead": list(self.Qh.parameters()), "keymap": list(self.Kp.parameters()),
              "kcat": list(self.A2.parameters()), "beta": [self.beta] if self.beta is not None else [],
              "tau": [self.logtau] if self.logtau is not None else []}
        return {k: v for k, v in gr.items() if v}

    def n_params(self) -> int:
        """Trainable interface parameters excluding the fixed B / gates (the A-family 'read' parameters)."""
        return sum(p.numel() for v in self.param_groups().values() for p in v)

    def trainable(self, train_B: bool = False, train_g: bool = False):
        ps = [p for v in self.param_groups().values() for p in v]
        for p in ps:
            p.requires_grad_(True)
        for p in self.B.parameters():
            p.requires_grad_(bool(train_B))
        self.g.requires_grad_(bool(train_g))
        return ps + (list(self.B.parameters()) if train_B else []) + ([self.g] if train_g else [])

    def gate(self, li):
        return self.g[self._gi[li]]

    def config(self) -> dict:
        return {"layer_map": self.layer_map, "d_r": self.d_r, "d_e": self.d_e, "d_q": self.d_q, "cfg": self.cfg}

    def Bmap(self, li, o):
        return o if self.identity_B else o @ self.B[str(li)].t()

    # ------------------------------------------------------------------ read
    def build_in(self, li, feats: dict, hist=None):
        """feats {name: [T, d]} -> x_in [T, d_in], new window history."""
        xin = torch.cat([feats[f].float() for f in self.cfg["feats"]], -1)
        W = int(self.cfg["window"])
        if not W:
            return xin, None
        s = str(li)
        x = feats["x"].float()
        T = x.shape[0]
        full = x if hist is None else torch.cat([hist, x], 0)
        P = full.shape[0] - T
        idx = (P + torch.arange(T, device=x.device)).view(T, 1) - torch.arange(1, W, device=x.device).view(1, W - 1)
        valid = idx >= 0
        Xl = full[idx.clamp_min(0)]                                          # [T, W-1, d_r]
        sc = (Xl @ self.win_u[s]) + self.win_b[s].view(1, -1)
        sc = sc.masked_fill(~valid, float("-inf"))
        al = torch.nan_to_num(torch.softmax(sc, -1), nan=0.0)
        pooled = (al.unsqueeze(-1) * Xl).sum(1)
        return torch.cat([xin, pooled], -1), full[-(W - 1):].detach()

    def map_x(self, li, xin):
        s = str(li)
        y = self.A[s](xin)
        if self.cfg["mlp_hidden"]:
            y = y + self.M[s](xin)
        if self.cfg["moe"]:
            pi = torch.softmax(self.R[s](xin), -1)                            # [T, E]
            d = torch.einsum("td,edr->ter", xin, self.EU[s])
            y = y + torch.einsum("te,ter,erk->tk", pi, d, self.EV[s])
        return y

    def _qkv(self, att, xe, T):
        hd = att.head_dim
        q, gate = torch.chunk(att.q_proj(xe).view(T, -1, hd * 2), 2, dim=-1)
        k = att.k_norm(att.k_proj(xe).view(T, -1, hd)).transpose(0, 1)
        v = att.v_proj(xe).view(T, -1, hd).transpose(0, 1)
        return q, gate.reshape(T, -1), k, v

    def _rope(self, x, pos, ref):
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        p = pos.view(1, 1, -1).expand(3, 1, -1)
        cos, sin = self.rot(ref.unsqueeze(0), p)
        xr, _ = apply_rotary_pos_emb(x.unsqueeze(0), x.unsqueeze(0), cos.float(), sin.float())
        return xr[0]

    def op_keys(self, li, op: dict):
        e = self.layer_map[li]
        if self.cfg["keymap"]:
            X = op[e]["X"].float()
            N = X.shape[0]
            k = self.att[str(li)].k_norm(self.Kp[str(li)](X).view(N, -1, self.hd)).transpose(0, 1)
            if self.cfg["rope"]:
                k = self._rope(k, torch.arange(N, device=X.device), X)
            return k
        return (op[e]["K"] if self.cfg["rope"] else op[e]["Kpre"]).float()

    def read(self, li, feats: dict, op: dict, pos0: int, past=None, need_logits: bool = False,
             tau: float | None = None, topk: int = 0):
        """feats {name: [T, d]} reader-row features; op {enc_layer: {"K","Kpre","V"[, "X"]}}; pos0 = N.
        -> dict(y [T,d_r], o [T,d_e], logits [Hkv,G,T,C] (if asked), past)."""
        s = str(li)
        att = self.att[s]
        hd, scale = self.hd, att.scaling
        xin, hist = self.build_in(li, feats, None if past is None else past.get("hist"))
        xe = self.map_x(li, xin)
        T = xe.shape[0]
        q, gate, k, v = self._qkv(att, xe, T)
        if self.cfg["qhead"]:
            q = self.Qh[s](xin).view(T, -1, hd)
        q = att.q_norm(q).transpose(0, 1)                                    # [Hq,T,hd]
        P = 0 if past is None else past["K"].shape[1]
        pos = torch.arange(pos0 + P, pos0 + P + T, device=xe.device)
        if self.cfg["rope"]:
            q, k = self._rope(q, pos, xe), self._rope(k, pos, xe)
        Kown = k if past is None else torch.cat([past["K"], k], 1)
        Vown = v if past is None else torch.cat([past["V"], v], 1)
        e = self.layer_map[li]
        Kop, Vop = self.op_keys(li, op), op[e]["V"].float()
        N = Kop.shape[1]
        sc = _scores(q, torch.cat([Kop, Kown], 1), scale)
        newp = {"K": Kown.detach(), "V": Vown.detach(), "hist": hist}
        if self.cfg["kcat"]:
            for j, e2 in enumerate(self.cfg["kcat"][li]):
                k2n = f"{li}_{e2}"
                a2 = self.att2[k2n]
                xe2 = self.A2[k2n](xin)
                q2, _, k2, _ = self._qkv(a2, xe2, T)
                q2 = a2.q_norm(q2).transpose(0, 1)
                q2, k2 = self._rope(q2, pos, xe2), self._rope(k2, pos, xe2)
                K2own = k2 if past is None else torch.cat([past[f"K2_{e2}"], k2], 1)
                K2 = op[e2]["K"].float()
                assert K2.shape[1] == N
                sc = sc + self.beta[self._gi[li], j].view(self.n_kv, -1, 1, 1) * _scores(q2, torch.cat([K2, K2own], 1), a2.scaling)
                newp[f"K2_{e2}"] = K2own.detach()
        if self.logtau is not None:
            sc = sc * torch.exp(-self.logtau[self._gi[li]]).view(self.n_kv, -1, 1, 1)
        if tau is not None and tau != 1.0:
            sc = sc / float(tau)
        C = sc.shape[-1]
        col = torch.arange(C, device=sc.device).view(1, C)
        row = torch.arange(T, device=sc.device).view(T, 1)
        sc = sc.masked_fill((col > (C - T) + row).view(1, 1, T, C), float("-inf"))
        if topk and topk < N:
            thr = sc[..., :N].topk(int(topk), -1).values[..., -1:]
            drop = torch.zeros_like(sc, dtype=torch.bool)
            drop[..., :N] = sc[..., :N] < thr
            sc = sc.masked_fill(drop, float("-inf"))
        p = torch.softmax(sc, -1)
        Hq = q.shape[0]
        o = torch.einsum("kgtc,kcd->kgtd", p, torch.cat([Vop, Vown], 1)).reshape(Hq, T, hd).transpose(0, 1).reshape(T, Hq * hd)
        o = att.o_proj(o * torch.sigmoid(gate))
        return {"y": self.Bmap(li, o), "o": o, "logits": sc if need_logits else None, "past": newp}


def _scores(q, K, scale):
    """q [Hq,T,hd], K [Hkv,C,hd] -> [Hkv,G,T,C] (unmasked)."""
    Hq, T, hd = q.shape
    Hkv = K.shape[0]
    return torch.einsum("kgtd,kcd->kgtc", q.view(Hkv, Hq // Hkv, T, hd), K) * scale


def pair_logits(att, rot, xe, Kop, pos0: int, rope: bool = True, Kop_pre=None):
    """Ridge-pair read scores (no interface): encoder attention `att` applied to mapped rows xe [T,d_e] over the
    operator keys Kop [Hkv,N,hd] (post-RoPE) + own rows, causal. -> logits [Hkv,G,T,N+T]."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    T, hd = xe.shape[0], att.head_dim
    q, _ = torch.chunk(att.q_proj(xe).view(T, -1, hd * 2), 2, dim=-1)
    q = att.q_norm(q).transpose(0, 1)
    k = att.k_norm(att.k_proj(xe).view(T, -1, hd)).transpose(0, 1)
    if rope:
        pos = torch.arange(pos0, pos0 + T, device=xe.device).view(1, 1, T).expand(3, 1, T)
        cos, sin = rot(xe.unsqueeze(0), pos)
        qr, kr = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos.float(), sin.float())
        q, k = qr[0], kr[0]
    K = torch.cat([(Kop if rope else Kop_pre).float(), k], 1)
    sc = _scores(q, K, att.scaling)
    C = K.shape[1]
    col = torch.arange(C, device=sc.device).view(1, C)
    row = torch.arange(T, device=sc.device).view(T, 1)
    return sc.masked_fill((col > (C - T) + row).view(1, 1, T, C), float("-inf"))


# ----------------------------------------------------------------------------- encoder operator / teacher capture
class EncTeacher16:
    """Patches the encoder's attention at `layers`. rec = {} activates recording. mode "op": per layer
    {"K" post-RoPE, "Kpre", "V"} ([Hkv,T,hd]) + "X" (layer input [T,d]) for layers in x_layers;
    mode "rows": {"x", "q" (post-RoPE), "k", "v", "out"}."""

    def __init__(self, enc, layers, x_layers=()):
        from deltaskill.backbone_utils import decoder_layers
        L = decoder_layers(enc)
        self.layers = sorted(set(int(x) for x in layers))
        self.x_layers = {int(x) for x in x_layers}
        self.mods = {e: L[e].self_attn for e in self.layers}
        self.rec: dict | None = None
        self.mode = "op"
        self._orig = {}
        for e, m in self.mods.items():
            self._orig[e] = m.forward
            m.forward = types.MethodType(_teacher16_forward(self, e), m)

    def detach(self):
        for e, f in self._orig.items():
            self.mods[e].forward = f
        self._orig.clear()


def _teacher16_forward(Tc: EncTeacher16, e: int):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        if Tc.rec is not None:
            shp = hidden_states.shape[:-1]
            hs = (*shp, -1, self.head_dim)
            k = self.k_norm(self.k_proj(hidden_states).view(hs)).transpose(1, 2)
            v = self.v_proj(hidden_states).view(hs).transpose(1, 2)
            q, _ = torch.chunk(self.q_proj(hidden_states).view(*shp, -1, self.head_dim * 2), 2, dim=-1)
            q = self.q_norm(q.reshape(hs)).transpose(1, 2)
            cos, sin = position_embeddings
            qq, kk = apply_rotary_pos_emb(q, k, cos, sin)
            if Tc.mode == "op":
                r = {"K": kk[0].detach(), "Kpre": k[0].detach(), "V": v[0].detach()}
                if e in Tc.x_layers:
                    r["X"] = hidden_states[0].detach().half()
            else:
                r = {"x": hidden_states[0].detach(), "q": qq[0].detach(), "k": kk[0].detach(), "v": v[0].detach()}
            Tc.rec[e] = r
        out = type(self).forward(self, hidden_states, position_embeddings, attention_mask,
                                 past_key_values=past_key_values, **kwargs)
        if Tc.rec is not None and Tc.mode == "rows":
            Tc.rec[e]["out"] = out[0][0].detach()
        return out

    return forward


# ----------------------------------------------------------------------------- reader patch
class D16Reader:
    """Patches the reader's attention at itf.layers (same modes as D13Reader) and captures the extra features
    (earlier layers' input_layernorm outputs, the reader's own q) the interface asks for.
        "off" | "student" (eval/gen, R.past) | "train" (R.rec; inject R.inject or on-policy) |
        "teacher" (inject R.inject, record feats in R.fcap[li]) | "e2e" (gated read with grad flows downstream)
    `extra_lags` / `extra_q` widen the captured features beyond itf's (multi-variant screening)."""

    def __init__(self, model, itf: D16Interface, extra_lags=(), extra_q: bool = False):
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
        self.fcap: dict = {}
        self.onpolicy = False
        self.need_logits = False
        self.grad = True
        self.tau, self.topk = None, 0
        self.lags = set(itf.lags()) | {int(x) for x in extra_lags}
        self.want_q = itf.needs_q() or bool(extra_q)
        self._ln: dict = {}
        self._hooks = []
        src = sorted({li - k for li in itf.layers for k in self.lags if li - k >= 0})
        for j in src:
            self._hooks.append(L[j].input_layernorm.register_forward_hook(self._ln_hook(j)))
        self._orig = {}
        for li, m in self.mods.items():
            self._orig[li] = m.forward
            m.forward = types.MethodType(_reader16_forward(self, li), m)

    def _ln_hook(self, j):
        def h(mod, inp, out):
            if self.mode != "off":
                self._ln[j] = out[0]
        return h

    def load(self, op: dict, n_op: int):
        self.op, self.n_op, self.past = op, int(n_op), {}

    def feats(self, li, attn_mod, hidden_states) -> dict:
        x = hidden_states[0]
        f = {"x": x.float()}
        for k in self.lags:
            f[f"L-{k}"] = (self._ln[li - k] if li - k >= 0 else x).float()
        if self.want_q:
            T, hd = x.shape[0], attn_mod.head_dim
            q, _ = torch.chunk(attn_mod.q_proj(x).view(T, -1, hd * 2), 2, dim=-1)
            f["q"] = attn_mod.q_norm(q).reshape(T, -1).float()
        return f

    def detach(self):
        for li, f in self._orig.items():
            self.mods[li].forward = f
        self._orig.clear()
        for h in self._hooks:
            h.remove()
        self._hooks = []


def _reader16_forward(R: D16Reader, li: int):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        out, w = type(self).forward(self, hidden_states, position_embeddings, attention_mask,
                                    past_key_values=past_key_values, **kwargs)
        if R.mode == "off":
            return out, w
        itf = R.itf
        if R.mode == "teacher":
            with torch.no_grad():
                R.fcap[li] = {k: v.detach() for k, v in R.feats(li, self, hidden_states).items()}
            y = R.inject[li]
        elif R.mode == "student":
            with torch.no_grad():
                f = R.feats(li, self, hidden_states)
            res = itf.read(li, f, R.op, R.n_op, past=R.past.get(li), tau=R.tau, topk=R.topk)
            R.past[li] = res["past"]
            g = itf.gate(li)
            y = out[0].float() + g * (res["y"] - out[0].float())
        elif R.mode == "e2e":
            f = R.feats(li, self, hidden_states)
            res = itf.read(li, f, R.op, R.n_op, need_logits=R.need_logits)
            R.rec[li] = res
            g = itf.gate(li)
            y = out[0].float() + g * (res["y"] - out[0].float())
        else:
            with torch.no_grad():
                f = {k: v.detach() for k, v in R.feats(li, self, hidden_states).items()}
            with torch.set_grad_enabled(R.grad):
                res = itf.read(li, f, R.op, R.n_op, need_logits=True, tau=R.tau, topk=R.topk)
            R.rec[li] = res
            y = res["y"].detach() if R.onpolicy else R.inject[li]
        return y.to(out.dtype).unsqueeze(0), w

    return forward


# ----------------------------------------------------------------------------- save / load
_FROZEN = ("att.", "att2.", "rot.")


def save_d16(itf: D16Interface, path: str, extra: dict | None = None):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sd = {k: v.detach().cpu() for k, v in itf.state_dict().items() if not k.startswith(_FROZEN)}
    torch.save({"format": FORMAT, "config": itf.config(), "state_dict": sd, "extra": extra or {}}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_d16(path: str, enc, B: dict | None):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    c = blob["config"]
    itf = D16Interface(enc, c["layer_map"], c["d_r"], c["d_e"], B=B, cfg=c["cfg"], d_q=c.get("d_q", 0))
    sd = blob["state_dict"]
    if B is None:
        sd = {k: v for k, v in sd.items() if not k.startswith("B.")}
    missing, unexpected = itf.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith(_FROZEN) and not k.startswith("B.")]
    assert not unexpected and not bad, (bad, unexpected)
    return itf, blob.get("extra") or {}


def from_d13(path: str, enc, B: dict | None, cfg: dict | None = None, d_q: int = 0):
    """A D16 interface initialised from a D13 file (A [+ MLP] copied; cfg extras at their no-op init)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    c = blob["config"]
    cfg = {**(cfg or {})}
    cfg.setdefault("mlp_hidden", c.get("mlp_hidden", 0))
    itf = D16Interface(enc, c["layer_map"], c["d_r"], c["d_e"], B=B, cfg=cfg, d_q=d_q)
    sd = blob["state_dict"]
    with torch.no_grad():
        for li in itf.layers:
            s = str(li)
            W, b = sd[f"A.{s}.weight"].float(), sd[f"A.{s}.bias"].float()
            itf.A[s].weight.zero_()
            off = 0
            for f in itf.cfg["feats"]:
                d = feat_dim(f, itf.d_r, itf.d_q)
                if f == "x":
                    itf.A[s].weight[:, off:off + d].copy_(W)
                off += d
            itf.A[s].bias.copy_(b)
            if itf.cfg["mlp_hidden"] and itf.cfg["feats"] == ["x"] and not itf.cfg["window"] and f"M.{s}.0.weight" in sd:
                itf.M[s].load_state_dict({k[len(f"M.{s}."):]: v for k, v in sd.items() if k.startswith(f"M.{s}.")})
            if itf.cfg["qhead"]:
                itf.init_qhead(li)
            if itf.cfg["kcat"]:
                for e2 in itf.cfg["kcat"][li]:
                    itf.A2[f"{li}_{e2}"].weight.copy_(itf.A[s].weight)
                    itf.A2[f"{li}_{e2}"].bias.copy_(itf.A[s].bias)
        if "g" in sd:
            itf.g.copy_(sd["g"])
    return itf


__all__ = ["D16Interface", "D16Reader", "EncTeacher16", "pair_logits", "save_d16", "load_d16", "from_d13",
           "DEFAULT_CFG", "feat_dim"]
