"""P5: a copy head with its OWN addressing, trained to copy (default off; scripts/p5_train.py).

Why (outputs/P3_SUMMARY.md, P4_SUMMARY.md, E3_SUMMARY.md): P3 / P4 reuse the D13 read's attention p at L19 to fetch
token identities. That attention is semantic, diffuse (≈ .30 gold-span mass; .48 with E3's gold loss), so the fetched
identity is a mixed bag (P4: P_copy(gold) ≈ .22, gate stays shut; P3: kv_tokoff ≈ kv). P5 gives the copy its own
pointer, queried by the reader's FINAL hidden state (the state that chooses the next token) and keyed by document-side
features that are stored with the operator, trained by a supervised span-copy loss.

Document side (stored with the operator; no reader pass over the document):
    ids[c]            the token id prefilled at operator column c (P3 / P4 format, op["ids"])
    hn_c              the 4B encoder's residual stream at layers --p5_enc_layers (default 19, 27) at column c, normed
                      by the encoder's own next-layer input RMSNorm (op["p5h"], bf16, [N, nL·d_e]; frozen harvest)
    k_c^h             = Wkh·hn_c + Wkt·rms(E_reader[ids[c]])        per head h (H heads of dim dc)   [trainable]
Reader side at row t (h_t = the reader's final post-norm hidden state, i.e. the lm_head input):
    s^h(t, j)         = <Wq_h·rms(h_t), k_j^h>·e^{logt}/√dc + pos_h[bucket(N−1−j)] + cr_h·log P_reader(t, tok_t(j))
    π(t, j)           = Σ_h softmax(hw)_h · softmax_j s^h(t, j)       over the document-body columns j ∈ [S, N)
    tok_t(j)          = nosp(ids[j]) on rows fed the ">" of "<answer>" (the first content token carries no leading
                      space: "<answer>Bob" vs the document's " Bob"); gtm(ids[j]) = id(">" + nosp string) on rows fed
                      the "answer" of "<answer>" (BPE merges the tag with the content: ">Bob"; '>' itself when no merged
                      token exists; --p5_merged 1, P4's finding); else ids[j]
  next-position tracking (--p5_chain 1, P4's rule with the space-variant match): mass that emitted the token fed at row t
  moves one column right:  m_t(j) = π'(t−1, j−1)·[x_t ∈ {ids[j−1], nosp(ids[j−1])}],  π'(t) = (π(t)+μ m_t)/(1+μ Σ m_t)
    P_copy(t, v)      = Σ_j π'(t, j)·[tok_t(j) = v]                   (sums to 1)
  gate  g_t = clamp(a, 0, a_max)·σ(gw·rms(h_t) + gf·f_t + gb), f_t = [log max π', H(π')/log N_b, Σ m_t, log max P_r,
                      log Σ_v P_r P_copy, first-row flag];  a = 0 at init  =>  step 0 == the parent EXACTLY
  fuse "mix": P = (1−g) P_reader + g P_copy      fuse "poe": P ∝ P_reader · (ε + P_copy)^g
Span head (candidate 2): an end query Wqe on the same keys; at the row after "<answer>", (s*, e*) = argmax
  log π(s) + log π_end(e) over 0 ≤ e − s < span_max; "span decode" emits "<answer>" + ids[s*..e*] + "</answer>".

Losses (scripts/p5_train.py): the stage-2 objective through the fused output (KL(text ‖ P) + 0.5·CE), plus
  span copy   −log Σ_o π'(r_k, o + k)  over the strict content rows r_k of items whose gold content tokens occur
              verbatim in the document (o = occurrence starts; first token matched up to the leading space)
  end         −log Σ_o π_end(r_0, o + L − 1)
  gate BCE    σ(z_t) vs 1[P_copy(gold_t) > P_reader(gold_t)]  on every answer row (the gate's selectivity learns while
              a ≈ 0)
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from xenc.d13_read import D13Interface, D13Reader

NF = 6           # gate features


def parse_p5(enc_layers: str = "19,27", heads: int = 4, dim: int = 128, fuse: str = "mix", chain: int = 1,
             prefix_ids=None, gt_id: int = -1, span_max: int = 48, nbkt: int = 18, eps: float = 1e-3,
             seed: int = 0, ans_id: int = -1, merged: int = 0) -> dict:
    """CLI -> p5 config. prefix_ids: chat-template ids before the document body (len = S). gt_id: the id of the ">"
    token that ends "<answer>" (the next row predicts the first answer-content token)."""
    L = [int(x) for x in str(enc_layers).split(",") if x != ""]
    assert L and fuse in ("mix", "poe"), (enc_layers, fuse)
    pre = [int(x) for x in (prefix_ids or [])]
    return {"enc_layers": L, "heads": int(heads), "dim": int(dim), "fuse": fuse, "chain": int(chain),
            "prefix": pre, "S": len(pre), "gt_id": int(gt_id), "span_max": int(span_max), "nbkt": int(nbkt),
            "eps": float(eps), "seed": int(seed),
            "ans_id": int(ans_id), "merged": int(merged)}


def nospace_table(tok, V: int) -> torch.Tensor:
    """[V] long: v -> id of v's string without its leading space marker ('Ġ') when that is one token, else v."""
    tab = torch.arange(V, dtype=torch.long)
    vocab = tok.get_vocab()
    for s, v in vocab.items():
        if v < V and len(s) > 1 and s[0] == "Ġ":
            w = vocab.get(s[1:])
            if w is not None and w < V:
                tab[v] = int(w)
    return tab


def space_maps(tok, V: int):
    """(nosp [V], gtm [V], id('>'), id('answer')): nosp[v] = v's string without its leading 'Ġ' (else v); gtm[v] = the
    token '>' + that string (else the '>' id). Same maps as xenc.p4_ptr.vocab_maps (P4's verified coverage .28 -> .87)."""
    vocab = tok.get_vocab()
    gt, ans = vocab[">"], vocab["answer"]
    nosp = torch.arange(V, dtype=torch.long)
    gtm = torch.full((V,), gt, dtype=torch.long)
    for st, v in vocab.items():
        if v >= V:
            continue
        b = st[1:] if len(st) > 1 and st[0] == "\u0120" else st
        if b is not st:
            w = vocab.get(b)
            if w is not None and w < V:
                nosp[v] = int(w)
        m = vocab.get(">" + b)
        if m is not None and m < V:
            gtm[v] = int(m)
    return nosp, gtm, int(gt), int(ans)


def _rms(x):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def _bucket(N: int, S: int, nb: int, dev) -> torch.Tensor:
    d = (N - 1 - torch.arange(S, N, device=dev)).clamp_min(0).float()
    return torch.floor(torch.log2(d + 1)).long().clamp_max(nb - 1)


class P5Mixin:
    needs_ids = True

    def _init_p5(self, p5: dict, d_e: int):
        c = dict(p5 or {})
        assert c.get("enc_layers"), "P5 needs p5['enc_layers']"
        self.p5 = c
        H, dc, nL = c["heads"], c["dim"], len(c["enc_layers"])
        g = torch.Generator().manual_seed(int(c.get("seed", 0)))

        def lin(i, o):
            return nn.Parameter(torch.randn(o, i, generator=g) / math.sqrt(i))
        self.P5 = nn.ParameterDict({
            "Wq": lin(self.d_r, H * dc), "Wqe": lin(self.d_r, H * dc),
            "Wkh": lin(nL * d_e, H * dc), "Wkt": lin(self.d_r, H * dc),
            "logt": nn.Parameter(torch.zeros(())), "hw": nn.Parameter(torch.zeros(H)),
            "pos": nn.Parameter(torch.zeros(H, c["nbkt"])), "cr": nn.Parameter(torch.zeros(H)),
            "rho": nn.Parameter(torch.ones(())),
            "a": nn.Parameter(torch.zeros(())),              # gate amplitude: 0 => exactly the parent
            "gw": nn.Parameter(torch.zeros(self.d_r)), "gf": nn.Parameter(torch.zeros(NF)),
            "gb": nn.Parameter(torch.zeros(())),
        })
        self.register_buffer("p5_nosp", torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer("p5_gtm", torch.zeros(0, dtype=torch.long), persistent=False)
        self.p5_on = True
        self._p5_ver = 0

    # ------------------------------------------------------------------ bookkeeping
    def config(self) -> dict:
        cfg = super().config()
        cfg["p5"] = dict(self.p5)
        return cfg

    def make_reader(self, model):
        self.__dict__["_E"] = model.get_input_embeddings().weight       # NOT registered (no copy, never saved)
        return P5Reader(model, self)

    def harvest_hooks(self, enc):
        return _P5Harvest(enc, self.p5["enc_layers"])

    def set_nosp(self, tab: torch.Tensor, gtm: torch.Tensor | None = None):
        self.p5_nosp = tab.to(self.P5["a"].device)
        if gtm is not None:
            self.p5_gtm = gtm.to(self.P5["a"].device)
        assert not self.p5.get("merged") or self.p5_gtm.numel() == self.p5_nosp.numel(), "P5 merged: gtm table missing"

    def p5_params(self) -> dict:
        P = self.P5
        return {"mat": [P[k] for k in ("Wq", "Wqe", "Wkh", "Wkt")],
                "ptr": [P[k] for k in ("logt", "hw", "pos", "cr", "rho")],
                "amp": [P["a"]], "gate": [P[k] for k in ("gw", "gf", "gb")]}

    def a_max(self) -> float:
        return 1.0 if self.p5["fuse"] == "mix" else 4.0

    def a_eff(self):
        a = self.P5["a"]
        return a + (a.clamp(0.0, self.a_max()) - a).detach()            # straight-through clamp

    def p5_state(self) -> dict:
        P = self.P5
        with torch.no_grad():
            return {"a": round(float(P["a"]), 4), "gb": round(float(P["gb"]), 4), "|gw|": round(float(P["gw"].norm()), 4),
                    "gf": [round(float(x), 3) for x in P["gf"]], "temp": round(float(P["logt"].exp()), 3),
                    "mu": round(float(P["rho"].exp()), 3),
                    "hw": [round(float(x), 3) for x in torch.softmax(P["hw"], -1)],
                    "cr": [round(float(x), 3) for x in P["cr"]], "|pos|": round(float(P["pos"].norm()), 3),
                    "|Wq|": round(float(P["Wq"].norm()), 2), "|Wkh|": round(float(P["Wkh"].norm()), 2)}

    # ------------------------------------------------------------------ document side
    def p5_tables(self, op: dict, N: int) -> dict:
        """Per-operator tables: body start S (common prefix with the chat template; 0 for a left-truncated doc), body
        ids, their no-space variants, distance-from-end buckets."""
        assert "ids" in op and "p5h" in op, "P5: the operator carries no ids / p5h (harvest with a P5Interface)"
        ids = op["ids"]
        assert int(ids.shape[0]) == N == int(op["p5h"].shape[0]), \
            f"P5 ids ({ids.shape[0]}) / p5h ({op['p5h'].shape[0]}) do not align with the operator ({N})"
        assert self.p5_nosp.numel() > 0, "P5: no-space table not set (itf.set_nosp)"
        pre = self.p5["prefix"]
        S, head = 0, ids[: len(pre)].tolist()
        while S < len(head) and head[S] == pre[S]:
            S += 1
        S = S if S < N - 1 else 0
        dev = self.P5["a"].device
        b = ids[S:N].to(dev)
        gtm = self.p5_gtm[b] if self.p5.get("merged") else b
        gtm_m = torch.where(gtm == int(self.p5["gt_id"]), torch.full_like(gtm, -1), gtm) if self.p5.get("merged") \
            else torch.full_like(b, -1)                      # chain match: only GENUINE merged tokens (not the bare '>')
        return {"S": S, "N": N, "ids": b, "nosp": self.p5_nosp[b], "gtm": gtm, "gtm_m": gtm_m,
                "bkt": _bucket(N, S, self.p5["nbkt"], dev), "op": op}

    def _key_chunk(self, hh, ii):
        assert "_E" in self.__dict__, "P5: call itf.make_reader(model) first (it attaches the reader's embedding table)"
        E = self.__dict__["_E"]
        e = _rms(E[ii.to(E.device)].to(hh.device))
        k = hh.float() @ self.P5["Wkh"].t() + e @ self.P5["Wkt"].t()
        return k.view(k.shape[0], self.p5["heads"], self.p5["dim"])

    def p5_keys(self, tab: dict, chunk: int = 8192) -> torch.Tensor:
        """-> [Nb, H, dc] fp32 (differentiable in the P5 key weights; per-chunk checkpointed under grad)."""
        from torch.utils.checkpoint import checkpoint
        op, S, N = tab["op"], tab["S"], tab["N"]
        dev = self.P5["a"].device
        hid, ids = op["p5h"][S:N].to(dev), op["ids"][S:N]
        out = []
        for a in range(0, N - S, chunk):
            b = min(N - S, a + chunk)
            if torch.is_grad_enabled():
                out.append(checkpoint(self._key_chunk, hid[a:b], ids[a:b], use_reentrant=False))
            else:
                out.append(self._key_chunk(hid[a:b], ids[a:b]))
        return torch.cat(out, 0)

    # ------------------------------------------------------------------ pointer
    def p5_scores(self, h, keys, tab, lp_tok=None, end: bool = False):
        """h [K,d_r] -> s [H,K,Nb]."""
        P = self.P5
        H, dc = self.p5["heads"], self.p5["dim"]
        q = (_rms(h) @ (P["Wqe"] if end else P["Wq"]).t()).view(h.shape[0], H, dc)
        s = torch.einsum("khd,nhd->hkn", q, keys) * (P["logt"].exp() / math.sqrt(dc))
        s = s + P["pos"][:, tab["bkt"]].unsqueeze(1)
        if lp_tok is not None:
            s = s + P["cr"].view(H, 1, 1) * lp_tok.unsqueeze(0)
        return s

    def p5_pi(self, s):
        w = torch.softmax(self.P5["hw"], -1)
        return torch.einsum("h,hkn->kn", w, torch.softmax(s, -1))

    def tok_idx(self, x, tab):
        """[K, Nb] token emitted by column j at each row: no-space variant on rows fed the '>' of '<answer>', merged
        '>'-variant on rows fed its 'answer' (--p5_merged); first-row flag [K]."""
        f1 = (x == int(self.p5["gt_id"])).view(-1, 1)
        ti = torch.where(f1, tab["nosp"].unsqueeze(0), tab["ids"].unsqueeze(0))
        if self.p5.get("merged"):
            f2 = (x == int(self.p5["ans_id"])).view(-1, 1)
            ti = torch.where(f2, tab["gtm"].unsqueeze(0), ti)
            f1 = f1 | f2
        return ti, f1.view(-1)

    def p5_forward(self, logits, h, x, keys, tab, state=None):
        """logits [K,V] reader logits, h [K,d_r] final hidden, x [K] ids fed at these rows, keys [Nb,H,dc];
        state = π'(previous row) or None (fresh). -> dict(logp [K,V] fp32, pi_raw, pi [K,Nb], Pc [K,V], g [K],
        sig [K] (σ of the gate logit), state)."""
        P = self.P5
        K, V = logits.shape
        x = x.to(keys.device).long()
        lsm = torch.log_softmax(logits.float(), -1).to(keys.device)
        ti, first = self.tok_idx(x, tab)
        lp_tok = lsm.gather(1, ti).clamp_min(-30.0).detach()
        pi = self.p5_pi(self.p5_scores(h.to(keys.device), keys, tab, lp_tok))
        mu = P["rho"].exp()
        outs, cm = [], []
        prev = state
        for t in range(K):
            p = pi[t]
            mt = None
            if self.p5["chain"] and prev is not None:
                match = ((tab["ids"] == x[t]) | (tab["nosp"] == x[t]) | (tab["gtm_m"] == x[t])).to(p.dtype)
                mt = F.pad((prev * match)[:-1], (1, 0))
                p = (p + mu * mt) / (1 + mu * mt.sum())
            cm.append(mt.sum() if mt is not None else torch.zeros((), device=p.device))
            outs.append(p)
            prev = p
        pi2 = torch.stack(outs)
        Pc = torch.zeros(K, V, device=pi2.device, dtype=pi2.dtype).scatter_add(1, ti, pi2)
        Pr = lsm.exp()
        Nb = pi2.shape[1]
        ent = -(pi2 * pi2.clamp_min(1e-30).log()).sum(-1) / math.log(max(Nb, 2))
        feat = torch.stack([pi2.max(-1).values.clamp_min(1e-30).log(), ent, torch.stack(cm),
                            lsm.max(-1).values, (Pr * Pc).sum(-1).clamp_min(1e-30).log(), first.float()], -1)
        z = _rms(h.to(keys.device)) @ P["gw"] + feat.detach() @ P["gf"] + P["gb"]
        sig = torch.sigmoid(z)
        g = self.a_eff() * sig
        if self.p5["fuse"] == "mix":
            Pm = (1 - g).double().unsqueeze(-1) * Pr.double() + g.double().unsqueeze(-1) * Pc.double()
            logp = Pm.clamp_min(1e-300).log().float()
        else:
            logp = torch.log_softmax(logits.float().to(keys.device) + g.unsqueeze(-1) * (self.p5["eps"] + Pc).log(), -1)
        return {"logp": logp, "pi_raw": pi, "pi": pi2, "Pc": Pc, "g": g, "sig": sig, "state": pi2[-1].detach(),
                "first": first}

    def p5_span(self, h_row, logits_row, keys, tab):
        """Span head at the row after '<answer>' (h_row [d_r], logits_row [V]) -> (s, e, logconf, pi_s, pi_e)."""
        x = torch.tensor([int(self.p5["gt_id"])], device=keys.device)
        lsm = torch.log_softmax(logits_row.float().view(1, -1), -1).to(keys.device)
        ti, _ = self.tok_idx(x, tab)
        lp_tok = lsm.gather(1, ti).clamp_min(-30.0)
        h = h_row.view(1, -1)
        ps = self.p5_pi(self.p5_scores(h, keys, tab, lp_tok))[0]
        pe = self.p5_pi(self.p5_scores(h, keys, tab, None, end=True))[0]
        s, e, c = best_span(ps, pe, int(self.p5["span_max"]))
        return s, e, c, ps, pe


def best_span(ps, pe, span_max: int, topk: int = 64):
    """argmax_{s <= e < s + span_max} log ps[s] + log pe[e] (over the top-k starts)."""
    Nb = ps.shape[0]
    ls, le = ps.clamp_min(1e-30).log(), pe.clamp_min(1e-30).log()
    st = ls.topk(min(topk, Nb)).indices
    off = torch.arange(span_max, device=ps.device)
    idx = st.view(-1, 1) + off.view(1, -1)
    ok = idx < Nb
    sc = ls[st].view(-1, 1) + le[idx.clamp_max(Nb - 1)]
    sc = sc.masked_fill(~ok, float("-inf"))
    flat = int(sc.argmax())
    i, d = flat // span_max, flat % span_max
    return int(st[i]), int(st[i]) + d, float(sc.view(-1)[flat])


class P5Interface(P5Mixin, D13Interface):
    def __init__(self, enc, layer_map: dict, d_r: int, d_e: int, B: dict | None = None, mlp_hidden: int = 0,
                 p5: dict | None = None):
        D13Interface.__init__(self, enc, layer_map, d_r, d_e, B=B, mlp_hidden=mlp_hidden)
        self._init_p5(p5, d_e)


def upgrade_p5(itf: D13Interface, enc, p5: dict) -> P5Interface:
    """A loaded plain D13 base -> P5Interface with identical A / MLP / B / bias / gates and a = 0 (step 0 == base)."""
    assert type(itf) is D13Interface, f"P5 on a {type(itf).__name__} init is not supported"
    assert not itf.delta_rank
    B = None if itf.identity_B else {li: itf.B[str(li)].detach().cpu() for li in itf.layers}
    new = P5Interface(enc, itf.layer_map, itf.d_r, itf.d_e, B=B, mlp_hidden=itf.mlp_hidden, p5=p5)
    sd = {k: v for k, v in itf.state_dict().items() if not k.startswith("D.")}
    missing, unexpected = new.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("P5.") for k in missing), missing
    return new.to(itf.g.device)


# ----------------------------------------------------------------------------- harvest
class _P5Harvest:
    """Forward hooks on the encoder's decoder layers: their output residual stream, normed by the NEXT layer's input
    RMSNorm (the encoder's own normalisation of that stream), bf16. finish() -> {"p5h": [N, nL·d_e]}."""

    def __init__(self, enc, layers):
        from deltaskill.backbone_utils import decoder_layers
        L = decoder_layers(enc)
        self.layers = list(layers)
        for li in self.layers:
            assert 0 <= li < len(L) - 1, f"P5 enc layer {li} needs a next layer (encoder has {len(L)})"
        self.norm = {li: L[li + 1].input_layernorm for li in self.layers}
        self.h: dict = {}
        self._hooks = [L[li].register_forward_hook(self._mk(li)) for li in self.layers]

    def _mk(self, li):
        def f(mod, inp, out):
            hs = out[0] if isinstance(out, (tuple, list)) else out
            with torch.no_grad():
                self.h[li] = self.norm[li](hs[0]).to(torch.bfloat16)
        return f

    def finish(self, dev) -> dict:
        for h in self._hooks:
            h.remove()
        out = {"p5h": torch.cat([self.h[li] for li in self.layers], -1).to(dev)}
        self.h = {}
        return out


# ----------------------------------------------------------------------------- reader
class P5Reader(D13Reader):
    """D13Reader + lm_head patch: in modes student / e2e with a P5 table loaded, the head returns the fused log P (or,
    with p5_capture, records (h, logits) and returns the parent logits unchanged)."""

    def __init__(self, model, itf):
        super().__init__(model, itf)
        self.head = model.get_output_embeddings()
        self._head_had = "forward" in self.head.__dict__
        self._head_orig = self.head.forward
        self.head.forward = types.MethodType(_head_forward(self), self.head)
        self._emb_hook = model.get_input_embeddings().register_forward_pre_hook(self._cap_ids)
        self.p5_x = None
        self.p5_tab = None
        self.p5_state = None
        self.p5_fresh = True
        self.p5_capture = False
        self.p5_h = None
        self.p5_last = None
        self.p5_gsum, self.p5_gn = 0.0, 0
        self._kc = (None, None, None)

    def _cap_ids(self, mod, args):
        if args and torch.is_tensor(args[0]):
            self.p5_x = args[0][0].detach()

    def load(self, op: dict, n_op: int):
        super().load(op, n_op)
        self.p5_state, self.p5_fresh, self.p5_last = None, True, None
        self.p5_tab = self.itf.p5_tables(op, int(n_op))

    def keys(self):
        """Cached (no grad) keys of the loaded operator for the current P5 weights (itf._p5_ver bumps on update)."""
        k = (id(self.p5_tab["op"]), self.p5_tab["N"], self.itf._p5_ver)
        if self._kc[0] != k:
            with torch.no_grad():
                self._kc = (k, self.itf.p5_keys(self.p5_tab), None)
        return self._kc[1]

    def detach(self):
        if getattr(self, "_head_orig", None) is not None:
            if self._head_had:
                self.head.forward = self._head_orig
            else:
                del self.head.forward
            self._head_orig = None
            self._emb_hook.remove()
        super().detach()


def _head_forward(R: P5Reader):
    def forward(self, h):
        out = R._head_orig(h)
        if R.mode not in ("student", "e2e") or R.p5_tab is None:
            return out
        if R.p5_capture:
            R.p5_h = h[0].detach()
            return out
        itf = R.itf
        if not itf.p5_on or float(itf.a_eff()) == 0.0:      # exactly the parent (step 0 / ablation)
            return out
        assert out.shape[0] == 1, "P5: batch size 1 only"
        K = h.shape[1]
        with torch.no_grad():
            res = itf.p5_forward(out[0], h[0], R.p5_x[-K:], R.keys(), R.p5_tab,
                                 state=None if R.p5_fresh else R.p5_state)
        R.p5_state, R.p5_fresh, R.p5_last = res["state"], False, res
        R.p5_gsum += float(res["g"].sum())
        R.p5_gn += K
        return res["logp"].to(out.device).unsqueeze(0)

    return forward


# ----------------------------------------------------------------------------- supervision helpers
def content_rows_strict(tok, target: str, open_tag: str = "<answer>", close_tag: str = "</answer>",
                        allow_merged: bool = False):
    """target "<answer>X</answer>" -> (ids, rows): rows = indices of the tokens lying strictly inside X, contiguous from
    the first content token; the first token may carry the tag's '>' (BPE merge ">Bob", matched through gtm) when
    allow_merged; [] if X is empty or (not allow_merged) the first content token is merged with the tag."""
    enc = tok(target, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = list(enc["input_ids"]), enc["offset_mapping"]
    if not (target.startswith(open_tag) and target.endswith(close_tag)):
        return ids, []
    a, b = len(open_tag), len(target) - len(close_tag)
    if b <= a:
        return ids, []
    over = [i for i, (s, e) in enumerate(off) if e > s and s < b and e > a]
    rows = []
    for i in over:
        s, e = off[i]
        if (s >= a or (allow_merged and not rows and s == a - 1)) and e <= b:
            if rows and i != rows[-1] + 1:
                break
            rows.append(i)
        elif rows:
            break
    if not over or not rows or rows[0] != over[0]:
        return ids, []
    return ids, rows


def occurrences(content: list[int], tab: dict, cap: int = 64) -> torch.Tensor:
    """Body-column starts o where the token sequence `content` occurs verbatim (first token up to its leading space)."""
    b, ns, gm = tab["ids"], tab["nosp"], tab.get("gtm_m", torch.full_like(tab["ids"], -1))
    L, Nb = len(content), int(b.shape[0])
    if L == 0 or L > Nb:
        return torch.zeros(0, dtype=torch.long, device=b.device)
    c0 = int(content[0])
    cand = torch.nonzero((b == c0) | (ns == c0) | (gm == c0)).view(-1)
    cand = cand[cand + L <= Nb]
    for k in range(1, L):
        if cand.numel() == 0:
            break
        cand = cand[b[cand + k] == int(content[k])]
    return cand[:cap]


__all__ = ["P5Interface", "P5Reader", "P5Mixin", "parse_p5", "nospace_table", "upgrade_p5", "best_span",
           "content_rows_strict", "occurrences"]
