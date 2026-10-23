"""D24: AMORTIZED registration of the paper-form read (D13 read, 4B encoder -> 9B reader) by a hypernetwork H.

User goal: a cost AND a performance advantage. At test time the LARGE reader never reads the document text (not even
once for registration). D20's per-document registration needs a 9B text teacher per test document; D24 replaces it
by ONE forward pass of a small network H, trained once on TRAINING documents only:

    test time:  4B encoder prefill of the document -> operator {e: (K_e, V_e)} (fixed, never modified)
                -> H(operator) -> per-document adapter (one forward, no optimisation, no 9B pass over the text)
                -> the 9B reader reads with base interface + adapter (the D13 read, unchanged otherwise)

Adapter (same family as D20's DocAdapter "dA + gate", optional "bias"), per read layer l:
    dA_l(doc)   = U_l · M_l(doc) · V_l                rank <= R delta on the base A_l  (U_l [d_e,R], V_l [R,d_r] shared
                                                      trainable bases; M_l(doc) [R,R] produced by H)
    g_l(doc)    = g0_l + dg_l(doc)                    gate offset
    b_l(doc)    = Bb0_l + Ub_l · c_l(doc)             optional write-back bias offset (Ub_l [d_r,Kb] shared)
H's output heads are ZERO-INITIALISED, so H(operator) at init == the base interface exactly (kv_H@0 == kv_base).

Operator summary (cheap, from the operator alone): per encoder read layer, the mean of V over the document positions
and n_q learned-query attention reads per kv head (softmax(q·K/√hd)·V over the operator), LayerNorm'd; a per-layer
projection, a shared trunk over all layers, per-layer heads. `input_mode="zero"` zeroes the summary (the control: H
then learns ONE shared adapter, i.e. "more base training", isolating the value of document conditioning).

Wiring: `attach(itf, H)` patches the D13Interface instance's map_x / Bmap / gate (instance attributes; the class and
the stored base parameters are untouched); `set_doc(itf, H(op))` / `set_doc(itf, None)` switch the adapter.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

FORMAT = "xenc.d24_hyper.v1"


class OpSummary(nn.Module):
    """{enc_layer: (K [Hkv,N,hd], V [Hkv,N,hd])} -> {enc_layer: feature [F]} (F = Hkv·hd·(1 + n_q))."""

    def __init__(self, enc_layers, hkv: int, hd: int, n_q: int = 8, seed: int = 0):
        super().__init__()
        self.enc_layers = sorted(int(e) for e in enc_layers)
        self.hkv, self.hd, self.n_q = int(hkv), int(hd), int(n_q)
        g = torch.Generator().manual_seed(seed + 101)
        self.Q = nn.ParameterDict({str(e): nn.Parameter(torch.randn(hkv, n_q, hd, generator=g) / hd ** 0.5)
                                   for e in self.enc_layers}) if n_q else nn.ParameterDict()
        self.dim = self.hkv * self.hd * (1 + self.n_q)

    def forward(self, op: dict) -> dict:
        out = {}
        for e in self.enc_layers:
            K, V = op[e]
            K, V = K.float(), V.float()
            f = [V.mean(1).reshape(-1)]                                       # [Hkv·hd]
            if self.n_q:
                q = self.Q[str(e)].to(K.device)
                p = torch.softmax(torch.einsum("hqd,hnd->hqn", q, K) / self.hd ** 0.5, -1)
                f.append(torch.einsum("hqn,hnd->hqd", p, V).reshape(-1))    # [Hkv·n_q·hd]
            x = torch.cat(f)
            out[e] = F.layer_norm(x, x.shape)
        return out


class D24Hyper(nn.Module):
    """H: operator -> per-document adapter for a D13Interface `itf` (see module doc)."""

    def __init__(self, itf, rank: int = 16, n_q: int = 8, d_proj: int = 256, hidden: int = 1024,
                 bias_k: int = 0, gate: bool = True, input_mode: str = "op", seed: int = 0,
                 u_scale: float = 1.0, b_scale: float = 0.05, db_rank: int = 0, arch: str = "v1",
                 stat_momentum: float = 0.01):
        super().__init__()
        assert input_mode in ("op", "zero"), input_mode
        assert arch in ("v1", "v2"), arch
        # v2 (fixes v1's collapse to a document-independent output, see outputs/D24_SUMMARY.md): the summary
        # features are STANDARDISED across documents (running per-coordinate mean / var buffers, updated only in
        # training forwards with grad), the trunk is pre-LayerNorm (no GELU saturation), and the document-independent
        # part of the adapter is a SEPARATE set of global parameters (gM, gdg, gc), so the heads only carry the
        # per-document residual.
        self.arch, self.stat_momentum = arch, float(stat_momentum)
        self.layers = list(itf.layers)
        self.layer_map = {int(k): int(v) for k, v in itf.layer_map.items()}
        self.d_r, self.d_e = int(itf.d_r), int(itf.d_e)
        att = itf.att[str(self.layers[0])]
        hd = int(att.head_dim)
        hkv = int(att.k_proj.out_features) // hd
        self.cfg = {"rank": int(rank), "n_q": int(n_q), "d_proj": int(d_proj), "hidden": int(hidden),
                    "bias_k": int(bias_k), "gate": bool(gate), "input_mode": input_mode, "seed": int(seed),
                    "u_scale": float(u_scale), "b_scale": float(b_scale), "db_rank": int(db_rank), "arch": arch,
                    "stat_momentum": float(stat_momentum), "hkv": hkv, "hd": hd}
        self.R, self.bias_k, self.use_gate, self.input_mode = int(rank), int(bias_k), bool(gate), input_mode
        self.summary = OpSummary(sorted(set(self.layer_map.values())), hkv, hd, n_q, seed)
        Fd = self.summary.dim
        self.proj = nn.ModuleDict({str(li): nn.Linear(Fd, d_proj) for li in self.layers})
        if arch == "v2":
            self.trunk = nn.Sequential(nn.Linear(len(self.layers) * d_proj, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                       nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU())
            for e in self.summary.enc_layers:
                self.register_buffer(f"fmu{e}", torch.zeros(Fd))
                self.register_buffer(f"fvar{e}", torch.ones(Fd))
            self.register_buffer("fstat_n", torch.zeros(()))
            nL = len(self.layers)
            self.gM = nn.Parameter(torch.zeros(nL, self.R, self.R))
            self.gdg = nn.Parameter(torch.zeros(nL))
            self.gc = nn.Parameter(torch.zeros(nL, max(int(bias_k), 1)))
        else:
            self.trunk = nn.Sequential(nn.Linear(len(self.layers) * d_proj, hidden), nn.GELU(),
                                       nn.Linear(hidden, hidden), nn.GELU())
        self.headM = nn.ModuleDict({str(li): nn.Linear(hidden, self.R * self.R) for li in self.layers})
        self.headG = nn.Linear(hidden, len(self.layers)) if self.use_gate else None
        self.headB = nn.ModuleDict({str(li): nn.Linear(hidden, self.bias_k) for li in self.layers}) if self.bias_k else None
        self.db_rank = int(db_rank) if not itf.identity_B else 0      # optional dB_l(doc) = UB_l N_l(doc) VB_l
        self.headN = nn.ModuleDict({str(li): nn.Linear(hidden, self.db_rank ** 2) for li in self.layers}) if self.db_rank else None
        for m in list(self.headM.values()) + ([self.headG] if self.headG is not None else []) + \
                (list(self.headB.values()) if self.headB is not None else []) + \
                (list(self.headN.values()) if self.headN is not None else []):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        g = torch.Generator().manual_seed(seed + 7)
        self.U = nn.ParameterDict()
        self.V = nn.ParameterDict()
        self.Ub = nn.ParameterDict()
        self.UB = nn.ParameterDict()
        self.VB = nn.ParameterDict()
        for li in self.layers:
            k = str(li)
            A0 = itf.A[k].weight.detach().float().cpu()
            sA = float(A0.norm()) / self.d_e ** 0.5 * float(u_scale)           # ~ rms of a row of A0 => rms of A0 x
            self.U[k] = nn.Parameter(torch.randn(self.d_e, self.R, generator=g) * sA / self.R ** 0.5)
            self.V[k] = nn.Parameter(torch.randn(self.R, self.d_r, generator=g) / self.d_r ** 0.5)
            if self.bias_k:
                self.Ub[k] = nn.Parameter(torch.randn(self.d_r, self.bias_k, generator=g) * b_scale / self.bias_k ** 0.5)
            if self.db_rank:
                B0 = itf.B[k].detach().float().cpu()
                sB = float(B0.norm()) / self.d_r ** 0.5
                self.UB[k] = nn.Parameter(torch.randn(self.d_r, self.db_rank, generator=g) * sB / self.db_rank ** 0.5)
                self.VB[k] = nn.Parameter(torch.randn(self.db_rank, self.d_e, generator=g) / self.d_e ** 0.5)

    def forward(self, op: dict) -> dict:
        """-> {"M": {li: [R,R]}, "dg": [L] | None, "c": {li: [Kb]} | None, "_norm": scalar}."""
        feats = self.summary(op)
        if self.arch == "v2":
            feats = self.standardise(feats)
        hs = []
        for li in self.layers:
            f = feats[self.layer_map[li]]
            if self.input_mode == "zero":
                f = torch.zeros_like(f)
            hs.append(self.proj[str(li)](f))
        h = self.trunk(torch.cat(hs))
        M = {li: self.headM[str(li)](h).view(self.R, self.R) for li in self.layers}
        dg = self.headG(h) if self.headG is not None else None
        c = {li: self.headB[str(li)](h) for li in self.layers} if self.headB is not None else None
        if self.arch == "v2":                      # global (document-independent) part + per-document residual
            out_doc = torch.stack([m.pow(2).mean() for m in M.values()]).mean()
            M = {li: self.gM[i] + M[li] for i, li in enumerate(self.layers)}
            dg = (self.gdg + dg) if dg is not None else None
            c = {li: self.gc[i, : self.bias_k] + c[li] for i, li in enumerate(self.layers)} if c is not None else None
        out = {"M": M, "dg": dg, "c": c,
               "N": {li: self.headN[str(li)](h).view(self.db_rank, self.db_rank) for li in self.layers}
               if self.headN is not None else None}
        out["_norm"] = torch.stack([m.pow(2).mean() for m in M.values()]).mean()
        if self.arch == "v2":
            out["_doc_norm"] = out_doc
        return out

    def standardise(self, feats: dict) -> dict:
        """(f − running mean) / running std per coordinate; the running statistics are updated (no grad) only in
        training-mode forwards with grad enabled (i.e. on training documents)."""
        upd = self.training and torch.is_grad_enabled()
        out = {}
        for e, f in feats.items():
            mu, var = getattr(self, f"fmu{e}"), getattr(self, f"fvar{e}")
            if upd:
                with torch.no_grad():
                    fd = f.detach()
                    m = 1.0 if float(self.fstat_n) == 0 else self.stat_momentum
                    mu.mul_(1 - m).add_(m * fd)
                    var.mul_(1 - m).add_(m * (fd - mu).pow(2)) if float(self.fstat_n) > 0 else None
            out[e] = ((f - mu) / (var + 1e-4).sqrt()).clamp(-6, 6)
        if upd:
            with torch.no_grad():
                self.fstat_n.add_(1)
        return out

    @torch.no_grad()
    def init_stats(self, ops: list):
        """Set the standardisation statistics from a list of TRAINING-document operators (before training)."""
        self.init_stats_feats([self.summary(op) for op in ops])

    @torch.no_grad()
    def init_stats_feats(self, fs: list):
        """Same, from precomputed self.summary(op) dicts (memory-light: no operators kept)."""
        for e in self.summary.enc_layers:
            X = torch.stack([f[e] for f in fs])
            getattr(self, f"fmu{e}").copy_(X.mean(0))
            getattr(self, f"fvar{e}").copy_(X.var(0, unbiased=False) if X.shape[0] > 1 else torch.ones_like(X[0]))
        self.fstat_n.fill_(len(fs))

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def adapter_numel(self) -> int:
        """Size of the per-document output (what H writes per document)."""
        return len(self.layers) * (self.R * self.R + (1 if self.use_gate else 0) + self.bias_k + self.db_rank ** 2)


# ----------------------------------------------------------------------------- wiring into a D13Interface
def attach(itf, H: D24Hyper):
    """Patch the interface INSTANCE so its read uses the current document adapter (itf._d24, None = base)."""
    assert not getattr(itf, "_d24_attached", False), "already attached"
    itf._d24, itf._d24_attached = None, True
    object.__setattr__(itf, "_d24_H", H)          # NOT a registered submodule (itf state_dict / .to unaffected)
    cls = type(itf)

    def map_x(self, li, x):
        y = cls.map_x(self, li, x)
        ad = self._d24
        if ad is not None:
            k = str(li)
            Hh = self._d24_H
            y = y + ((x @ Hh.V[k].t()) @ ad["M"][li].t()) @ Hh.U[k].t()
        return y

    def Bmap(self, li, o):
        y = cls.Bmap(self, li, o)
        ad = self._d24
        if ad is not None and ad.get("c") is not None and not self.identity_B:
            y = y + self._d24_H.Ub[str(li)] @ ad["c"][li]
        if ad is not None and ad.get("N") is not None and not self.identity_B:
            k, Hh = str(li), self._d24_H
            y = y + ((o @ Hh.VB[k].t()) @ ad["N"][li].t()) @ Hh.UB[k].t()
        return y

    def gate(self, li):
        g = cls.gate(self, li)
        ad = self._d24
        if ad is not None and ad.get("dg") is not None:
            g = g + ad["dg"][self._gi[li]]
        return g

    itf.map_x = types.MethodType(map_x, itf)
    itf.Bmap = types.MethodType(Bmap, itf)
    itf.gate = types.MethodType(gate, itf)
    return itf


def set_doc(itf, adapter):
    itf._d24 = adapter


def save_h(H: D24Hyper, path: str, extra: dict | None = None):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sd = {k: v.detach().cpu().clone() for k, v in H.state_dict().items()}
    torch.save({"format": FORMAT, "config": H.cfg, "state_dict": sd, "extra": extra or {}}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_h(path: str, itf) -> tuple:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    c = dict(blob["config"])
    c.pop("hkv", None)
    c.pop("hd", None)
    H = D24Hyper(itf, **c)
    H.load_state_dict(blob["state_dict"])
    return H, blob.get("extra") or {}


__all__ = ["D24Hyper", "OpSummary", "attach", "set_doc", "save_h", "load_h"]
