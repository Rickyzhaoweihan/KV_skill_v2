"""CMP: a per-token compression bottleneck on the D13 operator (agent CMP, 2026-10-23).

The D13 operator of a document is {enc_layer e: (K_e, V_e)}, the 4B encoder's post-RoPE keys / values at the read
layers (Qwen3.5-4B: Hkv=4, hd=256 -> 2·4·256 = 2048 numbers per token per layer; 5 layers -> 10240 per token,
20 KB/token in bf16). Hub Memory (HM, a colleague's method) stores 128 numbers per token instead. CMP puts a
per-token bottleneck of width d between the two:

    x_e,t   = [ unrope(K_e,t) ; V_e,t ] / s_e        (pre-RoPE keys: RoPE is re-applied at decode with the column's
                                                       own position, so the bottleneck never has to encode position)
    z_g,t   = E_g (x_S(g),t − μ_S(g))                  E_g: fixed PCA basis of the source layers S(g) of group g
                                                       (closed form from operator statistics, like HM's CCA map R)
    x̂_e,t   = U_e z_src(e),t + μ_e + b_e              U_e: per-target-layer decoder (closed-form init, TRAINABLE)
    (K̂, V̂)  = (rope(s_e·x̂_K), s_e·x̂_V)               -> the read uses (K̂, V̂) in place of (K, V)

Groups ("spec"):
    all:<d>         one shared hub over all read layers (HM-like: one vector per token for every depth)
    per:<d>         one hub per layer (storage 5·d)
    L<e>:<d>        hub of ONE encoder layer only (HM-like single-depth write); the other layers are decoded from it
                    by closed-form regression (cross-layer prediction)
    full            identity (no compression, no quantisation): the base operator
Storage per token = Σ_g d_g · bytes(quant) (+ fp16 group scales for int8 / int4); `bytes_per_token()`.
With d = full rank and own-group decoding the reconstruction is exact up to float rounding (tests/test_cmp.py).

z is quantised post hoc (`quant` = bf16 | fp16 | int8 | int4g64 | fp32) — the storage format; training uses bf16 z.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- RoPE helpers (partial rotary)
def _rot_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope_tables(rot, n: int, device, start: int = 0):
    """cos, sin [n, rotary_dim] fp32 for positions start..start+n-1 (text mRoPE: the 3 components are equal)."""
    pos = torch.arange(start, start + n, device=device).view(1, 1, n).expand(3, 1, n)
    x = torch.zeros(1, n, 1, device=device, dtype=torch.float32)
    cos, sin = rot(x, pos)
    return cos[0].float(), sin[0].float()


def rope(k, cos, sin):
    """k [H,N,hd] (pre-RoPE) -> post-RoPE, as transformers' qwen3_5 apply_rotary_pos_emb (partial rotary)."""
    rd = cos.shape[-1]
    kr, kp = k[..., :rd], k[..., rd:]
    return torch.cat((kr * cos + _rot_half(kr) * sin, kp), dim=-1)


def unrope(k, cos, sin):
    """Exact inverse of rope (rotation by −θ): x = y·cos − rot_half(y)·sin."""
    rd = cos.shape[-1]
    kr, kp = k[..., :rd], k[..., rd:]
    return torch.cat((kr * cos - _rot_half(kr) * sin, kp), dim=-1)


# ----------------------------------------------------------------------------- statistics
def token_features(op: dict, layers, cos, sin, idx=None) -> torch.Tensor:
    """[n, L·2F] fp32 raw features (pre-RoPE K ; V per layer, heads flattened) at token indices idx (default all)."""
    out = []
    for e in layers:
        K, V = op[e]
        if idx is not None:
            K, V, c, s = K[:, idx], V[:, idx], cos[idx], sin[idx]
        else:
            c, s = cos, sin
        k = unrope(K.float(), c, s)
        out.append(k.transpose(0, 1).reshape(k.shape[1], -1))
        out.append(V.float().transpose(0, 1).reshape(V.shape[1], -1))
    return torch.cat(out, -1)


class OpStats:
    """Streaming first / second moments of token_features over many operators (fp64)."""

    def __init__(self, layers, F: int, device="cpu"):
        self.layers = [int(e) for e in layers]
        self.F = int(F)
        D = len(self.layers) * 2 * self.F
        self.n = 0
        self.sum = torch.zeros(D, dtype=torch.float64, device=device)
        self.xx = torch.zeros(D, D, dtype=torch.float64, device=device)
        self.docs: list = []

    @torch.no_grad()
    def add(self, op: dict, rot, max_tokens: int = 0, seed: int = 0, doc_id: str = ""):
        e0 = self.layers[0]
        N = op[e0][0].shape[1]
        cos, sin = rope_tables(rot, N, op[e0][0].device)
        idx = None
        if max_tokens and N > max_tokens:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(N, generator=g)[:max_tokens].sort().values.to(op[e0][0].device)
        for a in range(0, (N if idx is None else idx.numel()), 4096):
            sub = (torch.arange(a, min(a + 4096, N), device=op[e0][0].device) if idx is None else idx[a:a + 4096])
            X = token_features(op, self.layers, cos, sin, sub).to(self.sum.device).double()
            self.n += X.shape[0]
            self.sum += X.sum(0)
            self.xx.addmm_(X.T, X)                                           # in place (no D x D temporary)
        self.docs.append(doc_id)

    @torch.no_grad()
    def add_features(self, X: torch.Tensor):
        """Precomputed raw features [n, D] in self.layers order (e.g. two models' features of the same tokens)."""
        X = X.to(self.sum.device).double()
        assert X.shape[1] == self.sum.numel(), (X.shape, self.sum.numel())
        self.n += X.shape[0]
        self.sum += X.sum(0)
        self.xx.addmm_(X.T, X)

    def state(self):
        return {"layers": self.layers, "F": self.F, "n": self.n, "sum": self.sum.cpu(), "xx": self.xx.cpu(),
                "docs": self.docs}


def parse_spec(spec: str, layers) -> dict:
    """'all:128' | 'per:256' | 'L7:128' | 'L7+L19:128' | 'full' -> {groups: [[e..]], dims: [d..]}."""
    layers = [int(e) for e in layers]
    spec = spec.strip()
    if spec in ("", "full", "none"):
        return {"groups": [], "dims": [], "spec": "full"}
    kind, d = spec.split(":")
    d = int(d)
    if kind == "all":
        groups = [list(layers)]
    elif kind == "per":
        groups = [[e] for e in layers]
    elif kind.startswith("L"):
        src = [int(x.lstrip("L")) for x in kind.split("+")]
        assert set(src) <= set(layers), (src, layers)
        groups = [src]
    else:
        raise ValueError(f"unknown CMP spec {spec!r}")
    return {"groups": groups, "dims": [d] * len(groups), "spec": spec}


# ----------------------------------------------------------------------------- the bottleneck
QUANT_BYTES = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "int8": 1.0, "int4g64": 0.5}


class OpCompressor(nn.Module):
    """Fixed encoders E_g (buffers) + trainable decoders U_e, b_e. See the module docstring."""

    def __init__(self, layers, F: int, Hkv: int, hd: int, groups, dims, spec: str = ""):
        super().__init__()
        self.layers = [int(e) for e in layers]
        self.F, self.Hkv, self.hd = int(F), int(Hkv), int(hd)
        assert self.F == self.Hkv * self.hd
        self.groups = [[int(e) for e in g] for g in groups]
        self.dims = [int(d) for d in dims]
        self.spec = spec
        self.quant = "bf16"
        P = 2 * self.F
        for gi, (g, d) in enumerate(zip(self.groups, self.dims)):
            self.register_buffer(f"E{gi}", torch.zeros(d, len(g) * P))
        self.src = {}
        for e in self.layers:
            own = [gi for gi, g in enumerate(self.groups) if e in g]
            self.src[e] = own[:1] if own else list(range(len(self.groups)))
        self.U = nn.ParameterDict({str(e): nn.Parameter(torch.zeros(P, sum(self.dims[gi] for gi in self.src[e])))
                                   for e in self.layers})
        self.b = nn.ParameterDict({str(e): nn.Parameter(torch.zeros(P)) for e in self.layers})
        self.register_buffer("mu", torch.zeros(len(self.layers), P))
        self.register_buffer("s", torch.ones(len(self.layers), 2))
        self.info: dict = {}

    # --------------------------------------------------------------- fit
    @classmethod
    @torch.no_grad()
    def from_stats(cls, st: dict, Hkv: int, hd: int, spec: str, ridge: float = 1e-6, device="cpu"):
        """spec may start with 'e1+e2+...|' to restrict the compressor to those encoder layers (the others pass through
        the wired read untouched), e.g. '23+27+31|all:128' = a compressed deep channel next to a raw shallow operator."""
        layers, F = [int(e) for e in st["layers"]], int(st["F"])
        P = 2 * F
        full_spec = spec
        if "|" in spec:
            sub, spec = spec.split("|")
            sub = [int(x) for x in sub.split("+")]
            assert set(sub) <= set(layers), (sub, layers)
            pos0 = {e: i for i, e in enumerate(layers)}
            ix = torch.cat([torch.arange(pos0[e] * P, (pos0[e] + 1) * P) for e in sub])
            st = {**st, "layers": sub, "sum": st["sum"][ix], "xx": st["xx"][ix][:, ix]}
            layers = sub
        sp = parse_spec(spec, layers)
        assert sp["groups"], "spec 'full' needs no compressor"
        m = cls(layers, F, Hkv, hd, sp["groups"], sp["dims"], full_spec)
        n = float(st["n"])
        mu = (st["sum"].to(device) / n)
        C = st["xx"].to(device) / n - torch.outer(mu, mu)                      # raw covariance (fp64)
        D = C.shape[0]
        diag = torch.diagonal(C)
        sec2 = torch.diagonal(st["xx"].to(device)) / n                         # second moment for the block scale
        scale = torch.ones(D, dtype=torch.float64, device=device)
        sblk = torch.ones(len(layers), 2, dtype=torch.float64)
        for i in range(len(layers)):
            for j in range(2):
                sl = slice(i * P + j * F, i * P + (j + 1) * F)
                r = float(sec2[sl].mean().clamp_min(1e-12).sqrt())
                sblk[i, j] = r
                scale[sl] = r
        Cs = C / torch.outer(scale, scale)                                    # standardised covariance
        mus = mu / scale
        pos = {e: i for i, e in enumerate(layers)}

        def cols(es):
            return torch.cat([torch.arange(pos[e] * P, (pos[e] + 1) * P, device=device) for e in es])
        Es, info = [], {"spec": full_spec, "groups": sp["groups"], "dims": sp["dims"], "n_tokens": int(n), "var_kept": []}
        for gi, (g, d) in enumerate(zip(sp["groups"], sp["dims"])):
            ci = cols(g)
            ev, V = torch.linalg.eigh(Cs[ci][:, ci])
            ev, V = ev.flip(0), V.flip(1)
            d_ = min(d, V.shape[1])
            E = V[:, :d_].T.contiguous()                                      # [d, |g|P]
            if d_ < d:                                                        # pad (never used for real specs)
                E = torch.cat([E, torch.zeros(d - d_, E.shape[1], dtype=E.dtype, device=device)])
            Es.append(E)
            info["var_kept"].append(float(ev[:d_].clamp_min(0).sum() / ev.clamp_min(0).sum()))
        for gi, E in enumerate(Es):
            getattr(m, f"E{gi}").copy_(E.float().cpu())
        for e in layers:
            ce = cols([e])
            srcs = m.src[e]
            if len(srcs) == 1 and e in sp["groups"][srcs[0]]:                 # own group: PCA reconstruction (exact)
                g = sp["groups"][srcs[0]]
                k = g.index(e)
                U = Es[srcs[0]][:, k * P:(k + 1) * P].T
            else:                                                             # cross-layer: ridge regression
                Ez = [Es[gi] for gi in srcs]
                gc = [cols(sp["groups"][gi]) for gi in srcs]
                Czz = torch.cat([torch.cat([Ez[a] @ Cs[gc[a]][:, gc[b]] @ Ez[b].T for b in range(len(srcs))], 1)
                                 for a in range(len(srcs))], 0)
                Cez = torch.cat([Cs[ce][:, gc[a]] @ Ez[a].T for a in range(len(srcs))], 1)
                lam = ridge * float(torch.diagonal(Czz).mean())
                U = torch.linalg.solve(Czz + lam * torch.eye(Czz.shape[0], dtype=Czz.dtype, device=device), Cez.T).T
                xe = Cs[ce][:, ce]
                r2 = 1 - float(torch.trace(xe - U @ Cez.T) / torch.trace(xe))
                info.setdefault("cross_r2", {})[int(e)] = r2
            m.U[str(e)].data.copy_(U.float().cpu())
        m.mu.copy_(mus.view(len(layers), P).float().cpu())
        m.s.copy_(sblk.float())
        m.info = info
        return m

    # --------------------------------------------------------------- apply
    def bytes_per_token(self, quant: str | None = None) -> float:
        q = quant or self.quant
        b = sum(self.dims) * QUANT_BYTES[q]
        if q == "int8":
            b += 2 * len(self.dims)                                           # fp16 absmax scale per token per group
        if q == "int4g64":
            b += 2 * sum(math.ceil(d / 64) for d in self.dims)
        return b

    def _quant(self, z):
        q = self.quant
        if q == "fp32":
            return z
        if q in ("bf16", "fp16"):
            dt = torch.bfloat16 if q == "bf16" else torch.float16
            return z + (z.to(dt).float() - z).detach()                        # storage rounding, straight-through
        from xenc.cache_compress import fake_quant
        bits, group = (8, 0) if q == "int8" else (4, 64)
        return z + (fake_quant(z.detach(), bits, dim=-1, group=group).float() - z).detach()

    @torch.no_grad()
    def encode(self, op: dict, cos, sin, chunk: int = 8192):
        """-> [z_g [N, d_g] fp32] (E is fixed: no grad)."""
        e0 = self.layers[0]
        N = op[e0][0].shape[1]
        dev = op[e0][0].device
        pos = {e: i for i, e in enumerate(self.layers)}
        zs = [torch.empty(N, d, device=dev) for d in self.dims]
        for a in range(0, N, chunk):
            idx = torch.arange(a, min(a + chunk, N), device=dev)
            for gi, g in enumerate(self.groups):
                X = token_features(op, g, cos, sin, idx)                       # raw
                s = torch.cat([self.s[pos[e]].repeat_interleave(self.F) for e in g]).to(dev)
                mu = torch.cat([self.mu[pos[e]] for e in g]).to(dev)
                zs[gi][a:a + len(idx)] = (X / s - mu) @ getattr(self, f"E{gi}").to(dev).T
        return zs

    def decode(self, zs, cos, sin, dtype) -> dict:
        pos = {e: i for i, e in enumerate(self.layers)}
        out = {}
        F, H, hd = self.F, self.Hkv, self.hd
        for e in self.layers:
            z = torch.cat([zs[gi] for gi in self.src[e]], -1)
            dev = z.device
            i = pos[e]
            x = z @ self.U[str(e)].to(dev).T + self.mu[i].to(dev) + self.b[str(e)].to(dev)
            N = x.shape[0]
            k = (x[:, :F] * self.s[i, 0].to(dev)).view(N, H, hd).transpose(0, 1)
            v = (x[:, F:] * self.s[i, 1].to(dev)).view(N, H, hd).transpose(0, 1)
            out[e] = (rope(k, cos, sin).to(dtype), v.to(dtype))
        return out

    def forward(self, op: dict, rot) -> dict:
        e0 = self.layers[0]
        K0 = op[e0][0]
        cos, sin = rope_tables(rot, K0.shape[1], K0.device)
        zs = [self._quant(z) for z in self.encode(op, cos, sin)]
        new = dict(op)
        new.update(self.decode(zs, cos, sin, K0.dtype))
        return new

    def trainable(self):
        return list(self.U.parameters()) + list(self.b.parameters())

    def config(self) -> dict:
        return {"layers": self.layers, "F": self.F, "Hkv": self.Hkv, "hd": self.hd, "groups": self.groups,
                "dims": self.dims, "spec": self.spec, "quant": self.quant}


def save_cmp(m: OpCompressor, path: str, extra: dict | None = None):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"format": "xenc.cmp_op.v1", "config": m.config(), "state_dict": {k: v.detach().cpu() for k, v in
                                                                                 m.state_dict().items()},
                "info": m.info, "extra": extra or {}}, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_cmp(path: str) -> OpCompressor:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    c = blob["config"]
    m = OpCompressor(c["layers"], c["F"], c["Hkv"], c["hd"], c["groups"], c["dims"], c.get("spec", ""))
    m.load_state_dict(blob["state_dict"])
    m.quant = c.get("quant", "bf16")
    m.info = blob.get("info", {})
    return m


def wire(rt, cmp: OpCompressor | None, log=print):
    """Wrap rt.R.load so every operator the reader reads goes through the bottleneck (with grad iff grad is enabled).
    The cached raw operator (rt._ops) is untouched, so the distillation teacher still sees the uncompressed columns."""
    if cmp is None:
        return rt
    R = rt.R
    rot = rt.itf.rot
    orig = R.load
    keys = set(cmp.layers)

    def load(op, n_op):
        assert keys <= set(k for k in op if not isinstance(k, str)), (sorted(keys), list(op))
        orig(cmp(op, rot), n_op)

    R.load = load
    rt.cmp = cmp
    log(f"CMP on: spec {cmp.spec} groups {cmp.groups} dims {cmp.dims} quant {cmp.quant} | "
        f"{cmp.bytes_per_token():.0f} B/token (full operator {len(cmp.layers) * 2 * cmp.F * 2} B/token bf16) | "
        f"decoder params {sum(p.numel() for p in cmp.trainable()) / 1e6:.2f}M | info {cmp.info}")
    return rt


__all__ = ["OpCompressor", "OpStats", "parse_spec", "rope", "unrope", "rope_tables", "token_features", "save_cmp",
           "load_cmp", "wire", "QUANT_BYTES"]
