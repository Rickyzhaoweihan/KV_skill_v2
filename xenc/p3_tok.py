"""P3: a TOKEN-IDENTITY channel for exact span copying (extension of the D13 read, xenc/d13_read.py).

Problem (outputs/D22_C3.md, outputs/E3_SUMMARY.md): on span tasks the read lands on the right REGION of the document
(first-row hit at reader layer 19 ≈ the text teacher's) but the reader cannot reproduce the exact string, because
the operator carries the 4B encoder's K/V, not the token identities.

Mechanism (document-side storage: the operator's column token ids, 4 bytes per column):
    F_s[j]   = scale · (E[id_{j+s}] − μ) P            E: the READER's own input-embedding table (resident; a table
                                                      lookup, never a reader pass over the document), P: d_r × r
                                                      low-rank projection shared by all documents (PCA of E; r = 0
                                                      means P = I), s ∈ shifts (±w neighbours: copying wants the
                                                      next token after the attended one as well)
    r_t      = Σ_j p_tj F[j]                          p: the read's OWN attention over the operator columns (the same
                                                      softmax as the D13 read; own-row columns are excluded), per kv
                                                      head (mean over its query group; "kv") or averaged over all
                                                      heads ("avg"); "topk" keeps only the k highest columns
    y        = y_D13 + γ_ℓ U_ℓ r_t                    U_ℓ zero-init (step 0 == the parent exactly), γ_ℓ gate (init 1)

Gradients reach U (and, through p, the read's addressing A / MLP). The stage-1 distillation regularizer keeps using
the UNmixed D13 read-out (rec["y"] = y_base), like E2. Columns: operator column c is text position c of the operator
prefix (E3, tests/test_e3.py), and the stored ids are exactly the ids fed to the encoder prefill (fit_d13.D13Rt.operator),
so alignment holds by construction; a cached operator without ids is re-tokenized and length-checked
(fit_d12p.CachedRt.operator).
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from xenc.d13_read import D13Interface, D13Reader


def parse_tok(spec_layers: str, rank: int = 256, heads: str = "kv", shifts: str = "-1,0,1", topk: int = 0,
              topk_norm: int = 0, temp: float = 1.0, src: str = "emb", adapter: str = "lin", mlp_hidden: int = 0,
              inj_layers: str = "", floor: float = 0.0) -> dict:
    """CLI -> tok config ({} when spec_layers is empty: channel off)."""
    layers = [int(x) for x in str(spec_layers or "").split(",") if x != ""]
    if not layers:
        return {}
    assert heads in ("kv", "avg", "kvavg"), heads
    assert not topk_norm or int(topk) > 0, "--tok_topk_norm needs --tok_topk > 0"
    assert float(temp) > 0, temp
    sh = [int(x) for x in str(shifts).split(",") if x != ""]
    assert sh and len(set(sh)) == len(sh), shifts
    if int(rank) == 0:
        assert int(topk) > 0, "full-rank (rank 0) token channel needs --tok_topk > 0 (dense full-rank is O(N·d_r) per doc)"
    assert src in ("emb", "unemb"), src
    ad = sorted(set(x for x in str(adapter).split("+") if x))
    assert ad and set(ad) <= {"lin", "mlp", "beta"}, adapter
    assert "mlp" not in ad or int(mlp_hidden) > 0, "--tok_adapter mlp needs --tok_mlp_hidden > 0"
    inj = [int(x) for x in str(inj_layers or "").split(",") if x != ""]
    if src == "unemb":
        assert int(rank) == 0 and int(topk) > 0 and heads in ("avg", "kvavg"), \
            "unembedding source: full rank (--tok_rank 0), top-k (--tok_topk > 0), --tok_heads avg|kvavg"
    if "beta" in ad:
        assert int(rank) == 0 and heads in ("avg", "kvavg"), "beta adapter needs full-rank head-averaged blocks"
    if inj:
        assert len(layers) == 1, "--tok_inj_layers: one read layer supplies the token read"
    return {"layers": layers, "rank": int(rank), "heads": heads, "shifts": sh, "topk": int(topk),
            "topk_norm": int(topk_norm), "temp": float(temp), "src": src, "adapter": "+".join(ad),
            "mlp_hidden": int(mlp_hidden), "inj_layers": inj, "floor": float(floor)}


def _attend_p(q, Kop, Vop, Kown, Vown, scale, need_logits=False):
    """xenc.d13_read._attend (identical arithmetic) that also returns the attention probabilities p [Hkv,G,T,C]."""
    Hq, T, hd = q.shape
    Hkv = Kop.shape[0]
    G = Hq // Hkv
    K = torch.cat([Kop, Kown], 1)
    V = torch.cat([Vop, Vown], 1)
    C = K.shape[1]
    qg = q.view(Hkv, G, T, hd)
    s = torch.einsum("kgtd,kcd->kgtc", qg, K) * scale
    first_own = C - T
    col = torch.arange(C, device=q.device).view(1, C)
    row = torch.arange(T, device=q.device).view(T, 1)
    s = s.masked_fill((col > first_own + row).view(1, 1, T, C), float("-inf"))
    p = torch.softmax(s, -1)
    o = torch.einsum("kgtc,kcd->kgtd", p, V).reshape(Hq, T, hd).transpose(0, 1).reshape(T, Hq * hd)
    out = {"o": o, "p": p}
    if need_logits:
        out["logits"] = s
    return out


def shift_table(F0: torch.Tensor, shifts) -> torch.Tensor:
    """F0 [N, r] -> [N, S·r] with block s holding F0[j + s] (zero outside 0..N-1)."""
    N = F0.shape[0]
    outs = []
    for s in shifts:
        Fs = torch.zeros_like(F0)
        if s >= 0:
            if s < N:
                Fs[: N - s] = F0[s:]
        else:
            if -s < N:
                Fs[-s:] = F0[: N + s]
        outs.append(Fs)
    return torch.cat(outs, -1)


def shifted_ids(ids: torch.Tensor, idx: torch.Tensor, s: int):
    """ids [N] LongTensor, idx [...] column indices -> (ids at idx+s clamped, valid mask)."""
    N = ids.shape[0]
    j = idx + s
    ok = (j >= 0) & (j < N)
    return ids[j.clamp(0, N - 1)], ok


class TokInterface(D13Interface):
    needs_ids = True

    def __init__(self, enc, layer_map: dict, d_r: int, d_e: int, B: dict | None = None, mlp_hidden: int = 0,
                 tok: dict | None = None):
        super().__init__(enc, layer_map, d_r, d_e, B=B, mlp_hidden=mlp_hidden)
        tok = dict(tok or {})
        assert tok.get("layers"), "TokInterface needs tok['layers']"
        self.tok_layers = sorted(int(x) for x in tok["layers"])
        assert set(self.tok_layers) <= set(self.layers), (self.tok_layers, self.layers)
        self.tok_rank = int(tok.get("rank", 256))
        self.tok_heads = str(tok.get("heads", "kv"))
        self.tok_shifts = [int(s) for s in tok.get("shifts", [0])]
        self.tok_topk = int(tok.get("topk", 0))
        self.tok_topk_norm = int(tok.get("topk_norm", 0))     # hard top-k: weights renormalized over the k columns
        self.tok_temp = float(tok.get("temp", 1.0))          # channel-only read temperature: p_tok = softmax(s·temp)
        if self.tok_rank == 0:
            assert self.tok_topk > 0, "full-rank channel only in top-k mode"
        a0 = self.att[str(self.layers[0])]
        self.tok_hkv = a0.k_proj.out_features // a0.head_dim
        r = self.tok_rank or self.d_r
        self.tok_nin = (self.tok_hkv if self.tok_heads == "kv" else 1) * len(self.tok_shifts) * r   # avg / kvavg: 1
        # P6 residual-stream extensions (defaults = the original P3 channel exactly)
        self.tok_src = str(tok.get("src", "emb"))                    # emb: E[id]; unemb: lm_head rows W_U[id] (+variants)
        self.tok_adapter = str(tok.get("adapter", "lin"))            # '+'-set of lin (U), mlp, beta (per-shift gains)
        self.tok_mlp_hidden = int(tok.get("mlp_hidden", 0))
        self.tok_inj_layers = [int(x) for x in tok.get("inj_layers", [])]   # residual injection layers (else: read layer)
        self.tok_floor = float(tok.get("floor", 0.0))                # token-gate floor (active once floor_on = 1)
        ad = set(self.tok_adapter.split("+"))
        tgt = [str(l) for l in (self.tok_inj_layers or self.tok_layers)]
        S = len(self.tok_shifts)
        mods = {
            "proj": nn.ParameterDict({
                "P": nn.Parameter(torch.zeros(self.d_r, self.tok_rank) if self.tok_rank else torch.zeros(0),
                                  requires_grad=False),
                "mu": nn.Parameter(torch.zeros(self.d_r), requires_grad=False),
                "scale": nn.Parameter(torch.ones(()), requires_grad=False)}),
            "U": nn.ParameterDict({k: nn.Parameter(torch.zeros(self.d_r, self.tok_nin)) for k in tgt} if "lin" in ad else {}),
            "gt": nn.ParameterDict({k: nn.Parameter(torch.ones(())) for k in tgt}),
        }
        if "beta" in ad:
            mods["beta"] = nn.ParameterDict({k: nn.Parameter(torch.zeros(S)) for k in tgt})
        if "mlp" in ad:
            g = torch.Generator().manual_seed(17)
            m1 = {}
            for k in tgt:
                lin = nn.Linear(self.tok_nin, self.tok_mlp_hidden)
                with torch.no_grad():
                    lin.weight.copy_(torch.randn(self.tok_mlp_hidden, self.tok_nin, generator=g) / self.tok_nin ** 0.5)
                    lin.bias.zero_()
                m1[k] = lin
            mods["M1"] = nn.ModuleDict(m1)
            mods["M2"] = nn.ModuleDict({k: nn.Linear(self.tok_mlp_hidden, self.d_r) for k in tgt})
            for m in mods["M2"].values():
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)
        if self.tok_floor > 0:
            mods["floor_on"] = nn.ParameterDict({"v": nn.Parameter(torch.zeros(()), requires_grad=False)})
        self.TOK = nn.ModuleDict(mods)
        self.tok_maps = None        # unemb: (nosp [V], gtm [V], gt_id) first-answer-token variants (set_tok_maps)
        self.tok_on = True          # register_d20 --tok_off: evaluate the same checkpoint with the channel disabled
        self.tok_log = False        # when True, read() records ‖γUr‖ / ‖y_D13‖ per layer in tok_stats
        self.tok_p_override = None  # EVAL-ONLY diagnostics (scripts/p3_diag.py): callable(li, p) -> p for the channel
        self.tok_stats: dict = {}

    # -------------------------------------------------------------- bookkeeping
    def config(self) -> dict:
        c = super().config()
        c["tok"] = {"layers": self.tok_layers, "rank": self.tok_rank, "heads": self.tok_heads,
                    "shifts": self.tok_shifts, "topk": self.tok_topk, "topk_norm": self.tok_topk_norm,
                    "temp": self.tok_temp, "src": self.tok_src, "adapter": self.tok_adapter,
                    "mlp_hidden": self.tok_mlp_hidden, "inj_layers": self.tok_inj_layers, "floor": self.tok_floor}
        return c

    # -------------------------------------------------------------- P6 residual-stream extensions
    def set_tok_maps(self, nosp: torch.Tensor, gtm: torch.Tensor, gt_id: int):
        """unemb source: a column contributes W_U of its id, its no-space variant and its '>'-merged variant (the
        first answer token is emitted as "Bob" / ">Bob" while the document holds " Bob"; xenc.p4_ptr.vocab_maps)."""
        self.tok_maps = (nosp.long(), gtm.long(), int(gt_id))

    def tok_gate(self, k: str):
        g = self.TOK["gt"][k]
        if self.tok_floor > 0 and float(self.TOK["floor_on"]["v"]) > 0:
            g = g.clamp_min(self.tok_floor)
        return g

    def adapt(self, k: str, r: torch.Tensor) -> torch.Tensor:
        """r [T, n_in] -> [T, d_r]: Σ of the configured adapters (all zero at init)."""
        out = None
        if k in self.TOK["U"]:
            out = r @ self.TOK["U"][k].t()
        if "beta" in self.TOK:
            T = r.shape[0]
            z = (r.view(T, len(self.tok_shifts), self.d_r) * self.TOK["beta"][k].view(1, -1, 1)).sum(1)
            out = z if out is None else out + z
        if "M1" in self.TOK:
            z = self.TOK["M2"][k](torch.nn.functional.gelu(self.TOK["M1"][k](r)))
            out = z if out is None else out + z
        return out

    def make_reader(self, model):
        return TokReader(model, self)

    def tok_params(self) -> dict:
        U = list(self.TOK["U"].parameters())
        for n in ("beta", "M1", "M2"):
            if n in self.TOK:
                U += list(self.TOK[n].parameters())
        return {"U": U, "gt": list(self.TOK["gt"].parameters())}

    def tok_state(self) -> dict:
        st = {"U_norm": {k: round(float(p.detach().norm()), 5) for k, p in self.TOK["U"].items()},
              "gt": {k: round(float(p.detach()), 4) for k, p in self.TOK["gt"].items()},
              "ratio": {k: round(v, 5) for k, v in self.tok_stats.items()}}
        if "beta" in self.TOK:
            st["beta"] = {k: [round(float(x), 5) for x in p.detach()] for k, p in self.TOK["beta"].items()}
        if "M2" in self.TOK:
            st["M2_norm"] = {k: round(float(m.weight.detach().norm()), 5) for k, m in self.TOK["M2"].items()}
        if "floor_on" in self.TOK:
            st["floor_on"] = float(self.TOK["floor_on"]["v"])
        return st

    @torch.no_grad()
    def init_proj(self, E: torch.Tensor, chunk: int = 16384):
        """P = top-r principal directions of the reader's embedding table E [V, d_r] (centered), scale so the
        projected features have unit mean variance; rank 0: P = I (not stored), μ = 0, scale = 1 / rms(E)."""
        dev = self.TOK["proj"]["mu"].device
        V, d = E.shape
        assert d == self.d_r, (E.shape, self.d_r)
        if not self.tok_rank:
            ss = torch.zeros((), dtype=torch.float64, device=E.device)
            for i in range(0, V, chunk):
                ss += E[i: i + chunk].double().pow(2).sum()
            self.TOK["proj"]["scale"].fill_(float((ss / (V * d)).rsqrt()))
            self.TOK["proj"]["mu"].zero_()
            return {"rank": 0, "rms": float((ss / (V * d)).sqrt())}
        mu = torch.zeros(d, dtype=torch.float64, device=E.device)
        for i in range(0, V, chunk):
            mu += E[i: i + chunk].double().sum(0)
        mu /= V
        C = torch.zeros(d, d, dtype=torch.float64, device=E.device)
        for i in range(0, V, chunk):
            X = E[i: i + chunk].double() - mu
            C += X.t() @ X
        C /= V
        ev, U = torch.linalg.eigh(C)
        top = ev.flip(0)[: self.tok_rank]
        P = U.flip(1)[:, : self.tok_rank]
        self.TOK["proj"]["P"].copy_(P.float().to(dev))
        self.TOK["proj"]["mu"].copy_(mu.float().to(dev))
        self.TOK["proj"]["scale"].fill_(float(top.mean().rsqrt()))
        return {"rank": self.tok_rank, "var_kept": float(top.sum() / ev.sum()), "top_ev": float(top[0]),
                "last_ev": float(top[-1])}

    @torch.no_grad()
    def features(self, ids: torch.Tensor, E: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        """ids [N] -> scale·(E[ids] − μ)P [N, r] fp32 (chunked gather: no [N, d_r] fp32 copy of the table)."""
        pr = self.TOK["proj"]
        P, mu, sc = pr["P"], pr["mu"], pr["scale"]
        dev = P.device if self.tok_rank else mu.device
        outs = []
        for i in range(0, ids.shape[0], chunk):
            e = E[ids[i: i + chunk].to(E.device)].to(dev).float()
            outs.append(((e - mu) @ P) * sc if self.tok_rank else e * sc)
        return torch.cat(outs, 0) if outs else torch.zeros(0, self.tok_rank or self.d_r, device=dev)

    @torch.no_grad()
    def doc_table(self, ids: torch.Tensor, E: torch.Tensor) -> dict:
        """Per-document token table for the read: dense low-rank -> F [N, S·r]; top-k -> F0 [N, r] (low rank) or
        the raw ids + E (full rank, gathered at read time)."""
        if self.tok_rank and not self.tok_topk:
            return {"F": shift_table(self.features(ids, E), self.tok_shifts), "N": int(ids.shape[0])}
        if self.tok_rank:
            return {"F0": self.features(ids, E), "N": int(ids.shape[0])}
        dev = self.TOK["proj"]["mu"].device
        out = {"ids": ids.to(dev), "E": E, "N": int(ids.shape[0])}
        if self.tok_src == "unemb":
            assert self.tok_maps is not None, "unembedding source: vocab maps not set (itf.set_tok_maps)"
            nosp, gtm, gt_id = self.tok_maps
            i_ = ids.to(nosp.device)
            out["nosp"] = nosp[i_].to(dev)
            gm = gtm[i_]
            out["gtm"] = torch.where(gm == gt_id, torch.full_like(gm, -1), gm).to(dev)
        return out

    # -------------------------------------------------------------- the read
    def tok_read(self, p: torch.Tensor, tok: dict) -> torch.Tensor:
        """p [Hkv,G,T,C] read attention (operator columns first) -> r [T, n_in]."""
        Hkv, G, T, C = p.shape
        N = tok["N"]
        assert N <= C - T, (N, C, T)
        if not self.tok_topk:
            F = tok["F"]
            ps = p[..., :N].reshape(Hkv, G * T, N)
            R = torch.matmul(ps, F).view(Hkv, G, T, -1).mean(1)                  # [Hkv,T,S·r]
            if self.tok_heads in ("avg", "kvavg"):
                R = R.mean(0, keepdim=True)
            return R.permute(1, 0, 2).reshape(T, -1)
        pk = p[..., :N].mean(1)                                                  # [Hkv,T,N]
        if self.tok_heads == "avg":
            pk = pk.mean(0, keepdim=True)
        v, idx = pk.topk(min(self.tok_topk, N), -1)                              # [H',T,k]
        if self.tok_topk_norm:                                                   # hard selection: Σ_k w = 1
            v = v / v.sum(-1, keepdim=True).clamp_min(1e-12)
        blocks = []
        for s in self.tok_shifts:
            if self.tok_rank:
                j = idx + s
                ok = ((j >= 0) & (j < N)).unsqueeze(-1)
                f = tok["F0"][j.clamp(0, N - 1)] * ok                            # [H',T,k,r]
            else:
                # full rank: gather the reader's own embedding rows in the table dtype (bf16 on GPU: no fp32
                # [H',T,k,d_r] copy) and contract with the attention weights by a batched matmul
                tid, ok = shifted_ids(tok["ids"], idx, s)
                pr = self.TOK["proj"]
                f = tok["E"][tid.to(tok["E"].device)].to(v.device)                       # [H',T,k,d_r]
                if self.tok_src == "unemb":       # mean of W_U over the id and its first-token variants
                    jn, _ = shifted_ids(tok["nosp"], idx, s)
                    jg, _ = shifted_ids(tok["gtm"], idx, s)
                    E_ = tok["E"]
                    vg = (jg >= 0)
                    f = (f.float() + E_[jn.to(E_.device)].to(v.device).float()
                         + E_[jg.clamp_min(0).to(E_.device)].to(v.device).float() * vg.unsqueeze(-1)) \
                        / (2.0 + vg.float()).unsqueeze(-1)
                w_ = (v * ok).to(f.dtype)
                blocks.append(torch.einsum("htk,htkd->htd", w_, f).float() * pr["scale"])
                continue
            blocks.append((v.unsqueeze(-1) * f).sum(2))                          # [H',T,r]
        R = torch.cat(blocks, -1)                                                # [H',T,S·r]
        if self.tok_heads == "kvavg":                                            # per-kv-head selection, mean read
            R = R.mean(0, keepdim=True)
        return R.permute(1, 0, 2).reshape(T, -1)

    def read(self, li, x, Kop, Vop, pos0: int, past=None, need_logits: bool = False, tok: dict | None = None):
        if not (self.tok_on and tok is not None and li in self.tok_layers):
            res = super().read(li, x, Kop, Vop, pos0, past=past, need_logits=need_logits)
            res["y_base"] = res["y"]
            return res
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
        tt = self.tok_temp != 1.0
        out = _attend_p(qr, Kop.float(), Vop.float(), Kown, Vown, att.scaling * self.read_scale_mult(Kop.shape[1]),
                        need_logits or tt)
        o = att.o_proj(out["o"] * torch.sigmoid(gate))
        y = self.Bmap(li, o)
        p_tok = torch.softmax(out["logits"] * self.tok_temp, -1) if tt else out["p"]
        if tt and not need_logits:
            out.pop("logits")
        if self.tok_p_override is not None:          # eval-only hook (oracle addressing / capture); None = off
            p_tok = self.tok_p_override(li, p_tok)
        r = self.tok_read(p_tok, tok)
        base = {"y_base": y, "o": o, "logits": out.get("logits"), "past": (Kown.detach(), Vown.detach()), "r": r}
        if self.tok_inj_layers:                      # P6: the token read is injected later, at tok_inj_layers
            return dict(base, y=y, z=None)
        if self.tok_adapter == "lin" and self.tok_floor == 0:
            z = r @ self.TOK["U"][str(li)].t()
            y_tok = self.TOK["gt"][str(li)] * z      # (the original P3 arithmetic, bit-identical)
        else:
            z = self.adapt(str(li), r)
            y_tok = self.tok_gate(str(li)) * z
        if self.tok_log:
            with torch.no_grad():
                self.tok_stats[str(li)] = float(y_tok.norm() / y.norm().clamp_min(1e-9))
        return dict(base, y=y + y_tok, z=z)


class TokReader(D13Reader):
    """D13Reader that builds the per-document token table at load() and passes it to the read at the channel layers.
    rec[li]["y"] is the UNmixed D13 read-out (distillation target side), rec[li]["y_mix"] the written-back one."""

    def __init__(self, model, itf: TokInterface):
        super().__init__(model, itf)
        self.tok = None
        self.tok_r = None            # P6: this forward's token read [T, n_in] (from the read layer), for injection
        self.tok_z: dict = {}        # P6: adapter outputs per target layer of this forward (logit-lens aux loss)
        for li, m in self.mods.items():
            m.forward = types.MethodType(_tok_forward(self, li), m)
        self._inj_hooks = []
        if itf.tok_inj_layers:
            from deltaskill.backbone_utils import decoder_layers
            L = decoder_layers(model)
            assert all(l > max(itf.tok_layers) for l in itf.tok_inj_layers), "injection layers must follow the read layer"
            for l in itf.tok_inj_layers:
                self._inj_hooks.append(L[l].register_forward_hook(_inj_hook(self, l)))

    def detach(self):
        for h in self._inj_hooks:
            h.remove()
        self._inj_hooks = []
        super().detach()

    def embed_table(self) -> torch.Tensor:
        if self.itf.tok_src == "unemb":
            return self.model.get_output_embeddings().weight
        return self.model.get_input_embeddings().weight

    def load(self, op: dict, n_op: int):
        super().load(op, n_op)
        self.tok, self.tok_r, self.tok_z = None, None, {}
        if not self.itf.tok_on:
            return
        assert "ids" in op, "token channel: the operator carries no column token ids (harvest with a TokInterface)"
        ids = op["ids"]
        assert int(ids.shape[0]) == int(n_op), f"token ids ({ids.shape[0]}) do not align with the operator ({n_op})"
        tb = op.get("_p3tok")
        if tb is None or tb.get("_itf") != id(self.itf):
            tb = self.itf.doc_table(ids, self.embed_table())
            tb["_itf"] = id(self.itf)
            op["_p3tok"] = tb                       # cached with the operator (LRU in D13Rt)
        self.tok = tb


def _tok_forward(R: TokReader, li: int):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        out, w = R._orig[li](hidden_states, position_embeddings, attention_mask,
                             past_key_values=past_key_values, **kwargs)
        if R.mode == "off":
            return out, w
        itf = R.itf
        K, V = R.op[itf.layer_map[li]]
        tk = R.tok if li in itf.tok_layers else None
        x = hidden_states[0].float()
        own = out[0].float()
        if R.mode == "student":
            res = itf.read(li, x, K, V, R.n_op, past=R.past.get(li), tok=tk)
            R.past[li] = res["past"]
            y = own + itf.gate(li) * (res["y"] - own)
            if tk is not None:
                R.tok_r = res.get("r")
                if res.get("z") is not None:
                    R.tok_z[str(li)] = res["z"]
        elif R.mode == "teacher":
            R.xcap[li] = x.detach()
            y = R.inject[li]
        elif R.mode == "e2e":
            nl = bool(R.need_logits)
            if torch.is_grad_enabled() and getattr(R, "ckpt", False):
                from torch.utils.checkpoint import checkpoint

                has_r = tk is not None and itf.tok_on
                has_z = has_r and not itf.tok_inj_layers

                def _rd(x_):
                    r_ = itf.read(li, x_, K, V, R.n_op, need_logits=nl, tok=tk)
                    return ((r_["y"], r_["y_base"], r_["o"]) + ((r_["logits"],) if nl else ())
                            + ((r_["r"],) if has_r else ()) + ((r_["z"],) if has_z else ()))
                t_ = checkpoint(_rd, x, use_reentrant=False)
                k_ = 4 if nl else 3
                res = {"y": t_[0], "y_base": t_[1], "o": t_[2], "logits": t_[3] if nl else None,
                       "r": t_[k_] if has_r else None, "z": t_[k_ + 1] if has_z else None}
            else:
                res = itf.read(li, x, K, V, R.n_op, need_logits=nl, tok=tk)
            if tk is not None:
                R.tok_r = res.get("r")
                if res.get("z") is not None:
                    R.tok_z[str(li)] = res["z"]
            R.rec[li] = {"y": res["y_base"], "o": res["o"], "logits": res["logits"], "y_mix": res["y"]}
            y = own + itf.gate(li) * (res["y"] - own)
        else:                                                           # "train" (stage-1 diagnostics)
            with torch.set_grad_enabled(R.grad):
                res = itf.read(li, x, K, V, R.n_op, need_logits=True, tok=tk)
            R.rec[li] = {"y": res["y_base"], "o": res["o"], "logits": res["logits"], "y_mix": res["y"]}
            y = res["y"].detach() if R.onpolicy else R.inject[li]
        return y.to(out.dtype).unsqueeze(0), w

    return forward


def _inj_hook(R: TokReader, l: int):
    """P6: add γ_l · adapt_l(r) to decoder layer l's OUTPUT (the residual stream) at every row of this forward, where r
    is the token read computed at the read layer earlier in the same forward (student / e2e modes only)."""
    def hook(mod, args, output):
        if R.mode not in ("student", "e2e") or R.tok_r is None or not R.itf.tok_on:
            return output
        h = output[0] if isinstance(output, tuple) else output
        r = R.tok_r
        if r.shape[0] != h.shape[1]:
            return output
        itf = R.itf
        z = itf.adapt(str(l), r.to(itf.TOK["gt"][str(l)].device))
        add = (itf.tok_gate(str(l)) * z).to(device=h.device, dtype=h.dtype)
        if R.mode == "e2e":
            R.tok_z[str(l)] = z
        if itf.tok_log:
            with torch.no_grad():
                itf.tok_stats[str(l)] = float(add.float().norm() / h[0].float().norm().clamp_min(1e-9))
        h2 = h + add.unsqueeze(0)
        return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2

    return hook


def upgrade_tok(itf: D13Interface, enc, tok: dict, E: torch.Tensor | None = None) -> TokInterface:
    """A loaded plain D13 base -> TokInterface with identical A / MLP / B / bias / gates and U = 0 (step 0 == base).
    E: the reader's input-embedding table (initializes the shared projection P)."""
    assert type(itf) is D13Interface, f"token channel on a {type(itf).__name__} init is not supported"
    assert not itf.delta_rank
    B = None if itf.identity_B else {li: itf.B[str(li)].detach().cpu() for li in itf.layers}
    new = TokInterface(enc, itf.layer_map, itf.d_r, itf.d_e, B=B, mlp_hidden=itf.mlp_hidden, tok=tok)
    sd = {k: v for k, v in itf.state_dict().items() if not k.startswith("D.")}
    missing, unexpected = new.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("TOK.") for k in missing), missing
    new = new.to(itf.g.device)
    info = new.init_proj(E) if E is not None else None
    new.tok_init_info = info
    return new


__all__ = ["TokInterface", "TokReader", "parse_tok", "upgrade_tok", "shift_table", "shifted_ids"]
