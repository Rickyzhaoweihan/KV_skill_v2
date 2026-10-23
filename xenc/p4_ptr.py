"""P4: an OUTPUT-SPACE pointer / copy head on top of the D13 read (pointer-generator style; xenc/d13_read.py).

Problem (outputs/D22_C3.md, outputs/E3_SUMMARY.md): on span tasks kv answers are well-formed document spans but the
wrong ones / not the exact string. The read addresses the right region about as often as the text teacher (E3: first-row
hit .425 at reader layer 19), but the operator carries the 4B encoder's K/V, not the token identities, so the reader
must re-derive the exact string from a mixed value vector. P3 (xenc/p3_tok.py) adds the token identities to the
residual stream; P4 instead copies them straight into the OUTPUT distribution.

Document-side storage: the operator's column token ids (op["ids"], 4 bytes per column; the SAME format P3 uses, written
by fit_d13.D13Rt.operator when the interface has needs_ids). Column c is text position c of the operator prefix (E3,
tests/test_e3.py), so ids[c] is exactly the token the encoder prefilled at column c. No reader pass over the document.

At each reader row t (every forward in "student" = generation and "e2e" = training mode):
    p^l_h(t, ·)  = softmax(κ_l · s^l_h(t, ·))       s: the read's own attention logits at pointer layer l (operator
                                                    columns + own rows), κ_l = exp(logk_l) (init 1 = the read's p)
    π^l(t, j)    ∝ Σ_h w_{l,h} p^l_h(t, S + j)        body columns j = 0..N-S-1 (S = first document-body column: the
                                                    chat-template prefix / attention sink is never copied); w = softmax
                                                    of learned head logits (init 0 = head average)
    π(t)         = Σ_l ω_l π^l(t)                     ω = softmax of learned layer logits (init uniform)
  next-position pointer (chain, --ptr_chain 1): the mass of the previous row's pointer on the columns that emitted the
  token x_t actually fed at row t moves one column right, so an exact span is copied token by token:
    m_t(j)       = Σ_s β_s π'(t-1, j-1) · [ids[S+j-1+s] == x_t]
    π'(t)        = (π(t) + μ m_t) / (1 + μ Σ_j m_t(j))      μ = exp(rho)   (π'(0) = π(0); carried across decode steps)
  copy distribution over the vocabulary (offset mixture β = softmax(shift logits) over --ptr_shifts, e.g. s = 0 copies
  the attended token, s = 1 the token after it, induction-head style):
    P_copy(t, v) = Σ_s β_s Σ_j π'(t, j) · [ids[S+j+s] == v]       (renormalised: sums to 1)
  mixture with the reader's own next-token distribution:
    g_t          = clamp(a, 0, 1) · σ(w·rms(h_t) + b + c·log M_t)      h_t: the reader's final (post-norm) hidden state,
                                                                      M_t: the read's body mass; a init 0 => g = 0
    P(t)         = (1 − g_t) P_reader(t) + g_t P_copy(t)
The patched lm_head returns log P(t) (fp32). Every consumer (greedy argmax, KL(text ‖ ·), CE) is unchanged: log P is
already normalised, so log_softmax(log P) = log P. At init a = 0 => g = 0 => P = P_reader exactly (step 0 == parent).
a uses a straight-through clamp (forward clamp to [0, 1], backward identity) so it can come back after a negative step.
Gradients: a gets ∂L/∂g at step 1; w, b, c, head / layer / κ / β / μ and (through π) the read's addressing A / MLP
receive gradient once g > 0.

Cost per decode step: one extra softmax over the pointer layer's existing read logits, a scatter-add of N weights into
the vocabulary, and the O(N) chain update. No reader pass over the document.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from xenc.d13_read import D13Interface, D13Reader


def parse_ptr(spec_layers: str, shifts: str = "0,1", chain: int = 1, prefix_ids=None, topk: int = 0, nbr: int = 0,
              kappa0: float = 1.0, nosp: int = 0) -> dict:
    """CLI -> ptr config ({} when spec_layers is empty: pointer off). prefix_ids: the chat-template ids that precede
    the document body in every operator (template_ids); their length is S."""
    layers = [int(x) for x in str(spec_layers or "").split(",") if x != ""]
    if not layers:
        return {}
    sh = [int(x) for x in str(shifts).split(",") if x != ""]
    assert sh and len(set(sh)) == len(sh), shifts
    pre = [int(x) for x in (prefix_ids or [])]
    assert int(topk) >= 0 and int(nbr) >= 0 and float(kappa0) > 0
    return {"layers": layers, "shifts": sh, "chain": int(chain), "S": len(pre), "prefix": pre, "topk": int(topk),
            "nbr": int(nbr), "kappa0": float(kappa0), "nosp": int(nosp)}


def vocab_maps(tok, V: int):
    """Space-variant maps for the first answer-content token (P5's finding, verified on P4's val: only 28% of span answers'
    first content token id occurs among the document ids, 87% with these maps):
      nosp[v] = id of v's string without its leading 'Ġ' (else v)       rows fed '>'      ("<answer>Bob" vs doc " Bob")
      gtm[v]  = id of '>' + v's string without 'Ġ' (else the '>' id)     rows fed 'answer' ("<answer>" merged: ">Bob")
    -> (nosp [V], gtm [V], id('>'), id('answer'))."""
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


def template_ids(tok, hub: str, system: str) -> list[int]:
    """Ids of the document prefix with an EMPTY document (same rendering as D13Rt.pre / E3 body_start)."""
    from xenc.state_skill import doc_prefix_text
    return [int(x) for x in tok(doc_prefix_text(tok, hub, "", system), add_special_tokens=False)["input_ids"]]


class PtrMixin:
    needs_ids = True

    def _init_ptr(self, ptr: dict):
        ptr = dict(ptr or {})
        assert ptr.get("layers"), "pointer head needs ptr['layers']"
        self.ptr_layers = sorted(int(x) for x in ptr["layers"])
        assert set(self.ptr_layers) <= set(self.layers), (self.ptr_layers, self.layers)
        self.ptr_shifts = [int(s) for s in ptr.get("shifts", [0])]
        self.ptr_chain = int(ptr.get("chain", 1))
        self.ptr_prefix = [int(x) for x in ptr.get("prefix", [])]
        self.ptr_S = int(ptr.get("S", len(self.ptr_prefix)))
        self.ptr_topk = int(ptr.get("topk", 0))        # >0: keep each row's top-k body columns (+- nbr neighbours)
        self.ptr_nbr = int(ptr.get("nbr", 0))
        kappa0 = float(ptr.get("kappa0", 1.0))         # initial pointer sharpness (init only; then learned)
        self.ptr_nosp = int(ptr.get("nosp", 0))        # first-answer-token space variants (needs set_vocab_maps)
        self.ptr_maps = None
        a0 = self.att[str(self.layers[0])]
        self.ptr_hq = a0.q_proj.out_features // (2 * a0.head_dim)        # gated q_proj: [q ; gate] per head
        nl, ns = len(self.ptr_layers), len(self.ptr_shifts)
        self.PTR = nn.ParameterDict({
            "a": nn.Parameter(torch.zeros(())),            # gate amplitude (0 => exactly the base)
            "w": nn.Parameter(torch.zeros(self.d_r)),      # gate from the reader's final hidden state
            "b": nn.Parameter(torch.zeros(())),
            "c": nn.Parameter(torch.zeros(())),            # gate coefficient on log body mass
            "head": nn.Parameter(torch.zeros(nl, self.ptr_hq)),
            "layer": nn.Parameter(torch.zeros(nl)),
            "logk": nn.Parameter(torch.full((nl,), float(torch.tensor(kappa0).log()))),   # sharpness κ = exp(logk)
            "shift": nn.Parameter(torch.zeros(ns)),        # offset mixture β
            "rho": nn.Parameter(torch.zeros(())),          # chain strength μ = exp(rho)
        })
        self.ptr_on = True          # register_d20 --ptr_off / register_d13 val ablation: same checkpoint, pointer off

    # -------------------------------------------------------------- bookkeeping
    def config(self) -> dict:
        c = super().config()
        c["ptr"] = {"layers": self.ptr_layers, "shifts": self.ptr_shifts, "chain": self.ptr_chain, "S": self.ptr_S,
                    "prefix": self.ptr_prefix, "topk": self.ptr_topk, "nbr": self.ptr_nbr, "nosp": self.ptr_nosp}
        return c

    def make_reader(self, model):
        return PtrReader(model, self)

    def set_vocab_maps(self, nosp: torch.Tensor, gtm: torch.Tensor, gt_id: int, ans_id: int):
        """vocab_maps(tok, V) output; required when ptr_nosp (rebuilt from the tokenizer at build time, not saved)."""
        self.ptr_maps = {"nosp": nosp.long(), "gtm": gtm.long(), "gt": int(gt_id), "ans": int(ans_id)}

    def ptr_params(self) -> dict:
        P = self.PTR
        return {"w": [P["w"]], "scal": [P[k] for k in ("a", "b", "c", "head", "layer", "logk", "shift", "rho")]}

    def ptr_state(self) -> dict:
        P = self.PTR
        with torch.no_grad():
            hw = torch.softmax(P["head"], -1)
            return {"a": round(float(P["a"]), 4), "b": round(float(P["b"]), 4), "c": round(float(P["c"]), 4),
                    "|w|": round(float(P["w"].norm()), 4),
                    "kappa": [round(float(x), 3) for x in P["logk"].exp()],
                    "layer_w": [round(float(x), 3) for x in torch.softmax(P["layer"], -1)],
                    "head_w_max": [round(float(x), 3) for x in hw.max(-1).values],
                    "beta": {s: round(float(x), 3) for s, x in zip(self.ptr_shifts, torch.softmax(P["shift"], -1))},
                    "mu": round(float(P["rho"].exp()), 3)}

    # -------------------------------------------------------------- the pointer distribution at one layer
    def ptr_dist(self, li: int, s: torch.Tensor, N: int, S: int):
        """s [Hkv,G,T,C] read logits (operator columns first, -inf masked) -> (π [T, N-S], log body mass [T])."""
        i = self.ptr_layers.index(li)
        Hkv, G, T, C = s.shape
        k = self.PTR["logk"][i].exp()
        sf = s.reshape(Hkv * G, T, C).float()
        inf = torch.isinf(sf)                                   # causal -inf mask: keep it out of κ's gradient (no NaN)
        p = torch.softmax((sf.masked_fill(inf, 0.0) * k).masked_fill(inf, float("-inf")), -1)[..., S:N]   # [H,T,Nb]
        w = torch.softmax(self.PTR["head"][i], -1).to(p.dtype)
        mix = torch.einsum("h,htn->tn", w, p)                                           # [T,Nb]
        mass = mix.sum(-1)
        return mix / mass.clamp_min(1e-30).unsqueeze(-1), mass.clamp_min(1e-30).log()

    def read(self, li, x, Kop, Vop, pos0: int, past=None, need_logits: bool = False, ptr=None, **kw):
        """ptr = the body start S of the loaded operator (PtrReader passes it at pointer layers in student / e2e mode);
        None -> the plain parent read."""
        if ptr is None or not self.ptr_on or li not in self.ptr_layers:
            return super().read(li, x, Kop, Vop, pos0, past=past, need_logits=need_logits, **kw)
        res = super().read(li, x, Kop, Vop, pos0, past=past, need_logits=True, **kw)
        res["ptr_pi"], res["ptr_lmass"] = self.ptr_dist(li, res["logits"], int(Kop.shape[1]), int(ptr))
        if not need_logits:
            res["logits"] = None
        return res

    # -------------------------------------------------------------- the output mixture (called from the lm_head)
    def ptr_mix(self, logits: torch.Tensor, h: torch.Tensor, R) -> torch.Tensor:
        """logits [K,V] reader logits, h [K,d] final hidden -> log P [K,V] fp32 (mixture with the copy distribution)."""
        P = self.PTR
        K, V = logits.shape
        tab = R.ptr_tab
        lw = torch.softmax(P["layer"], -1)
        pis = [R.ptr_pi[li][0] for li in self.ptr_layers]
        pi = sum(lw[i] * pis[i] for i in range(len(pis)))                               # [T,Nb]
        lmass = sum(lw[i] * R.ptr_pi[li][1] for i, li in enumerate(self.ptr_layers))     # [T]
        T, Nb = pi.shape
        assert T >= K, (T, K)
        if self.ptr_topk and self.ptr_topk < Nb:       # sparse pointer: top-k columns per row and their +- nbr neighbours
            idx = pi.detach().topk(self.ptr_topk, -1).indices                       # [T,k]
            keep = torch.zeros_like(pi, dtype=torch.bool)
            for d in range(-self.ptr_nbr, self.ptr_nbr + 1):
                keep.scatter_(1, (idx + d).clamp(0, Nb - 1), True)
            pi = pi * keep
            pi = pi / pi.sum(-1, keepdim=True).clamp_min(1e-30)
        beta = torch.softmax(P["shift"], -1)
        ns = bool(self.ptr_nosp)
        if self.ptr_chain or ns:
            x = R.ptr_x.to(pi.device)
            assert x.shape[0] == T, (x.shape, T)
        if self.ptr_chain:
            mu = P["rho"].exp()

            def _hit(i):
                h = tab["tok"][i].unsqueeze(0) == x.unsqueeze(1)
                if ns:                                  # the fed token may be a space / '>'-merged variant of the column
                    h = h | (tab["tok_ns"][i].unsqueeze(0) == x.unsqueeze(1)) | (tab["tok_gm"][i].unsqueeze(0) == x.unsqueeze(1))
                return h & tab["valid"][i].unsqueeze(0)
            match = sum(beta[i] * _hit(i) for i in range(len(self.ptr_shifts)))          # [T,Nb]
            prev = R.ptr_state if (R.mode == "student" and not R.ptr_fresh) else None
            outs = []
            for t in range(T):
                p = pi[t]
                if prev is not None:
                    m = F.pad((prev * match[t])[:-1], (1, 0))                           # mass at j -> column j + 1
                    p = (p + mu * m) / (1 + mu * m.sum())
                outs.append(p)
                prev = p
            pi2 = torch.stack(outs)
            if R.mode == "student":
                R.ptr_state = pi2[-1].detach()
        else:
            pi2 = pi
        rows = pi2[-K:]                                                                  # [K,Nb]
        Pc = torch.zeros(K, V, device=rows.device, dtype=rows.dtype)
        if ns:
            f1 = (x[-K:] == self.ptr_maps["gt"]).unsqueeze(1)                          # rows fed '>'
            f2 = (x[-K:] == self.ptr_maps["ans"]).unsqueeze(1)                         # rows fed 'answer'
        for i in range(len(self.ptr_shifts)):
            src = rows * (beta[i] * tab["valid"][i].to(rows.dtype)).unsqueeze(0)
            idx = tab["tok"][i].unsqueeze(0).expand(K, Nb)
            if ns:
                idx = torch.where(f1, tab["tok_ns"][i].unsqueeze(0), torch.where(f2, tab["tok_gm"][i].unsqueeze(0), idx))
            Pc = Pc.scatter_add(1, idx, src)
        Pc = Pc / Pc.sum(-1, keepdim=True).clamp_min(1e-30)
        hf = h.float()
        hn = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + 1e-6)
        a = P["a"]
        a_eff = a + (a.clamp(0.0, 1.0) - a).detach()                                    # straight-through clamp
        g = a_eff * torch.sigmoid(hn @ P["w"] + P["b"] + P["c"] * lmass[-K:].to(hn.device))   # [K]
        g = g.to(logits.device)
        Pr = torch.softmax(logits.double(), -1)
        Pm = (1 - g).double().unsqueeze(-1) * Pr + g.double().unsqueeze(-1) * Pc.to(logits.device).double()
        R.ptr_out = {"g": g, "Pc": Pc}
        if R.mode == "student":
            R.ptr_gsum += float(g.detach().sum())
            R.ptr_gn += K
        return Pm.clamp_min(1e-300).log().float()


class PtrInterface(PtrMixin, D13Interface):
    def __init__(self, enc, layer_map: dict, d_r: int, d_e: int, B: dict | None = None, mlp_hidden: int = 0,
                 ptr: dict | None = None):
        D13Interface.__init__(self, enc, layer_map, d_r, d_e, B=B, mlp_hidden=mlp_hidden)
        self._init_ptr(ptr)


# ----------------------------------------------------------------------------- reader
class PtrReader(D13Reader):
    """D13Reader whose pointer-layer reads also return the pointer distribution, and whose reader lm_head returns the
    copy mixture log P while the read is active (modes student / e2e). Other modes and other layers are the parent's."""

    def __init__(self, model, itf):
        super().__init__(model, itf)
        for li in itf.ptr_layers:
            self.mods[li].forward = types.MethodType(_ptr_forward(self, li), self.mods[li])
        self.head = model.get_output_embeddings()
        self._head_had = "forward" in self.head.__dict__
        self._head_orig = self.head.forward
        self.head.forward = types.MethodType(_head_forward(self), self.head)
        self._emb_hook = model.get_input_embeddings().register_forward_pre_hook(self._cap_ids)
        self.ptr_pi: dict = {}
        self.ptr_x = None
        self.ptr_state = None
        self.ptr_fresh = True
        self.ptr_tab = None
        self.ptr_S_cur = 0
        self.ptr_out = None
        self.ptr_gsum, self.ptr_gn = 0.0, 0

    def _cap_ids(self, mod, args):
        if args and torch.is_tensor(args[0]):
            self.ptr_x = args[0][0].detach()

    def load(self, op: dict, n_op: int):
        super().load(op, n_op)
        self.ptr_pi, self.ptr_state, self.ptr_fresh, self.ptr_out, self.ptr_tab = {}, None, True, None, None
        itf = self.itf
        if not itf.ptr_on:
            return
        assert "ids" in op, "pointer head: the operator carries no column token ids (harvest with a PtrInterface)"
        ids = op["ids"]
        N = int(n_op)
        assert int(ids.shape[0]) == N, f"pointer ids ({ids.shape[0]}) do not align with the operator ({N})"
        # S = the common prefix of the operator ids and the chat-template ids (== len(template) for a normal
        # operator; ~0 when a long document was cut from the left, so every column is then body)
        S, head = 0, ids[: itf.ptr_S].tolist()
        while S < len(head) and head[S] == itf.ptr_prefix[S]:
            S += 1
        S = S if S < N - 1 else 0
        self.ptr_S_cur = S
        dev = itf.PTR["a"].device
        ids = ids.to(dev)
        j = torch.arange(S, N, device=dev)
        tok, val = [], []
        for s in itf.ptr_shifts:
            js = j + s
            ok = (js >= 0) & (js < N)
            tok.append(ids[js.clamp(0, N - 1)])
            val.append(ok)
        self.ptr_tab = {"tok": tok, "valid": val, "S": S, "N": N}
        if itf.ptr_nosp:
            assert itf.ptr_maps is not None, "pointer --ptr_nosp: vocab maps not set (itf.set_vocab_maps)"
            mp = itf.ptr_maps
            self.ptr_tab["tok_ns"] = [mp["nosp"].to(dev)[t_] for t_ in tok]
            self.ptr_tab["tok_gm"] = [mp["gtm"].to(dev)[t_] for t_ in tok]

    def detach(self):
        if getattr(self, "_head_orig", None) is not None:
            if self._head_had:
                self.head.forward = self._head_orig
            else:
                del self.head.forward
            self._head_orig = None
            self._emb_hook.remove()
        super().detach()


def _ptr_forward(R: PtrReader, li: int):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        out, w = R._orig[li](hidden_states, position_embeddings, attention_mask,
                             past_key_values=past_key_values, **kwargs)
        if R.mode == "off":
            return out, w
        itf = R.itf
        K, V = R.op[itf.layer_map[li]]
        x = hidden_states[0].float()
        own = out[0].float()
        S = R.ptr_S_cur if (itf.ptr_on and R.ptr_tab is not None) else None
        if R.mode == "student":
            R.ptr_fresh = R.past.get(li) is None
            res = itf.read(li, x, K, V, R.n_op, past=R.past.get(li), ptr=S)
            R.past[li] = res["past"]
            if S is not None:
                R.ptr_pi[li] = (res["ptr_pi"], res["ptr_lmass"])
            y = own + itf.gate(li) * (res["y"] - own)
        elif R.mode == "teacher":
            R.xcap[li] = x.detach()
            y = R.inject[li]
        elif R.mode == "e2e":
            R.ptr_fresh = True
            nl = bool(R.need_logits)
            if torch.is_grad_enabled() and getattr(R, "ckpt", False):
                from torch.utils.checkpoint import checkpoint

                def _rd(x_):
                    r_ = itf.read(li, x_, K, V, R.n_op, need_logits=nl, ptr=S)
                    o_ = (r_["y"], r_["o"])
                    if S is not None:
                        o_ = o_ + (r_["ptr_pi"], r_["ptr_lmass"])
                    return o_ + ((r_["logits"],) if nl else ())
                t_ = checkpoint(_rd, x, use_reentrant=False)
                res = {"y": t_[0], "o": t_[1], "logits": t_[-1] if nl else None}
                if S is not None:
                    res["ptr_pi"], res["ptr_lmass"] = t_[2], t_[3]
            else:
                res = itf.read(li, x, K, V, R.n_op, need_logits=nl, ptr=S)
            if S is not None:
                R.ptr_pi[li] = (res["ptr_pi"], res["ptr_lmass"])
            R.rec[li] = {"y": res["y"], "o": res["o"], "logits": res["logits"]}
            y = own + itf.gate(li) * (res["y"] - own)
        else:                                                           # "train" (stage-1 diagnostics): parent read
            with torch.set_grad_enabled(R.grad):
                res = itf.read(li, x, K, V, R.n_op, need_logits=True)
            R.rec[li] = res
            y = res["y"].detach() if R.onpolicy else R.inject[li]
        return y.to(out.dtype).unsqueeze(0), w

    return forward


def _head_forward(R: PtrReader):
    def forward(self, h):
        out = R._head_orig(h)
        if R.mode not in ("student", "e2e") or not R.itf.ptr_on or R.ptr_tab is None:
            return out
        assert set(R.ptr_pi) == set(R.itf.ptr_layers), f"pointer: read layers {sorted(R.ptr_pi)} ran in this forward"
        assert out.shape[0] == 1, "pointer head: batch size 1 only"
        lp = R.itf.ptr_mix(out[0], h[0], R)
        R.ptr_pi = {}
        return lp.unsqueeze(0)

    return forward


def upgrade_ptr(itf: D13Interface, enc, ptr: dict) -> PtrInterface:
    """A loaded plain D13 base -> PtrInterface with identical A / MLP / B / bias / gates and a = 0 (step 0 == base)."""
    assert type(itf) is D13Interface, f"pointer head on a {type(itf).__name__} init is not supported"
    assert not itf.delta_rank
    B = None if itf.identity_B else {li: itf.B[str(li)].detach().cpu() for li in itf.layers}
    new = PtrInterface(enc, itf.layer_map, itf.d_r, itf.d_e, B=B, mlp_hidden=itf.mlp_hidden, ptr=ptr)
    sd = {k: v for k, v in itf.state_dict().items() if not k.startswith("D.")}
    missing, unexpected = new.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("PTR.") for k in missing), missing
    return new.to(itf.g.device)


def copy_nll(Pc: torch.Tensor, gold: torch.Tensor, in_doc: torch.Tensor) -> torch.Tensor:
    """Optional copy supervision: mean −log P_copy(gold) over the rows whose gold token occurs in the document body."""
    lp = Pc.gather(-1, gold.view(-1, 1)).squeeze(-1).clamp_min(1e-12).log()
    w = in_doc.to(lp.dtype)
    return -(lp * w).sum() / w.sum().clamp_min(1.0)


__all__ = ["PtrInterface", "PtrReader", "PtrMixin", "parse_ptr", "template_ids", "upgrade_ptr", "copy_nll"]
