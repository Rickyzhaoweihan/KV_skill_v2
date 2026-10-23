"""ADDR / SEG: structure-aware keys and values for the D13 paper-form read (xenc/d13_read.py). Default off.

Problem (outputs/P3_SUMMARY.md error anatomy, span tasks): 17% of the kv answers come from the WRONG paragraph and 20%
are the wrong span inside the right paragraph; Shakespeare-CF "which act" questions are .66-.70 vs text .90. A column's
4B key / value describes its token in local context; nothing in it says WHICH paragraph / section / act it sits in, so
the read cannot score "the paragraph about X" or report "Act III" for the line it found.

Form (read-time only; the stored operator (U, W) is unchanged; no reader pass over the document). The document is cut
into paragraphs (blank-line separated, cut at 2*block tokens; documents without blank lines: blocks of `block` tokens)
and sections (header lines by level: markdown '#' / '##' / '###', MuSiQue '[Pk]', LastingBench 'Passage k:', plays
'ACT' / 'SCENE', 'Chapter'), from the document
text's character offsets (xenc.addr_seg.segment_ids; one tokenizer call per document at harvest). Per encoder layer e
and kv head h, with k = RoPE^-1(K) the pre-RoPE operator keys:
    kP_p = mean_{j in paragraph p} k_j        vP_p = mean V_j           (paragraph summary)
    kS_s = mean over the active header lines of s (e.g. 'ACT III' + 'Scene II.', '# title' + '## Results') of the
           mean k over each header line's first 24 tokens; vS_s likewise    (section header summary)
    K'_j = K_j + RoPE_j( kP_{p(j)} WkP_h + kS_{s(j)} WkS_h )            (RoPE is linear: zero W == K bit-exactly)
    V'_j = V_j + vP_{p(j)} WvP_h + vS_{s(j)} WvS_h
W (hd x hd per kv head and read layer) zero-init, so step 0 == the parent read exactly. The read then attends with
K' / V' exactly as D13 does (own rows unchanged). Columns outside the document (chat template) get no summary.
The distillation term (dist_w) sees the logits of the K' read (the encoder teacher scores the plain operator), so it
keeps pulling the augmented addressing toward the teacher; the task loss (KL to text + CE) trains W.
"""

from __future__ import annotations

import bisect
import re

import torch
import torch.nn as nn

from xenc.d13_read import D13Interface, D13Reader

# header lines by level: a level-1 header resets levels 2-3, a level-2 header resets level 3
HDR_LEVELS = (
    re.compile(r"^[ \t]*(#[ \t]+\S|\[P\d+\]|Passage \d+:|ACT [IVXLC]+\b|Chapter \w+|CHAPTER \w+|BOOK \w+|Book \w+"
               r"|PART \w+|Part [IVXLC\d]+\b)", re.M),
    re.compile(r"^[ \t]*(##[ \t]+\S|SCENE [IVXLC]+\b|Scene [IVXLC]+\b)", re.M),
    re.compile(r"^[ \t]*(#{3,6}[ \t]+\S)", re.M),
)
HDR = re.compile("|".join(f"(?:{r.pattern})" for r in HDR_LEVELS), re.M)
HEAD_CAP = 24


# ----------------------------------------------------------------------------- segment ids (CPU, pure)
def segment_ids(text: str, offsets, d0: int, d1: int, n: int, base: int = 0, block: int = 128) -> dict:
    """text = the full prefix text; offsets = (start, end) chars of every token of the full prefix; operator column c
    is full token base + c (c < n); [d0, d1) = the document's character span inside text.
    -> {"par": LongTensor[n], "sec": LongTensor[n], "head_id": LongTensor[n], "head": BoolTensor[n],
        "sec_mat": FloatTensor[n_sec, n_hdr], "n_par", "n_sec", "n_hdr", "par_mode"}
    par     paragraph index: blank-line separated paragraphs, each cut into pieces of at most 2*block tokens
            (documents without blank lines: blocks of `block` tokens); -1 outside the document.
    head_id header-line index for the first HEAD_CAP tokens of every header line present (-1 otherwise).
    sec     index of the column's ACTIVE HEADER TUPLE (the most recent header at each level, e.g. ACT III + Scene II,
            or '# title' + '## Section'); -1 when no header with tokens present is active / outside the document.
    sec_mat row s = uniform averaging weights over the header lines of tuple s (the section summary is the mean of
            its active header lines' summaries)."""
    starts = [int(offsets[base + c][0]) for c in range(n)]
    ends = [int(offsets[base + c][1]) for c in range(n)]
    # a token's position = its first non-whitespace character (BPE glues "\n\n" / spaces to the next word)
    starts = [a + (len(text[a:b]) - len(text[a:b].lstrip())) if b > a and text[a:b].strip() else a
              for a, b in zip(starts, ends)]
    ind = [(d0 <= s < d1) and e > s for s, e in zip(starts, ends)]
    doc = text[d0:d1]
    # paragraph starts: d0 and the first char after every run of >= 2 newlines (blank line)
    pstarts = [d0] + [d0 + m.end() for m in re.finditer(r"\n[ \t]*\n\s*", doc) if d0 + m.end() < d1]
    raw = [-1] * n
    mode = "blank" if len(pstarts) >= 2 else "block"
    cap = 2 * block if mode == "blank" else block
    for c in range(n):
        if ind[c]:
            raw[c] = bisect.bisect_right(pstarts, starts[c]) - 1 if mode == "blank" else 0
    par, cur, k, cnt_ = [-1] * n, None, -1, 0
    for c in range(n):                          # cut every paragraph into pieces of <= cap tokens
        if raw[c] < 0:
            continue
        if raw[c] != cur or cnt_ >= cap:
            k, cur, cnt_ = k + 1, raw[c], 0
        par[c] = k
        cnt_ += 1
    # header lines (start char, end char, level), in document order
    hl = {}
    for lev, rx in enumerate(HDR_LEVELS):
        for m in rx.finditer(doc):
            a_ = d0 + m.start()
            if a_ in hl:
                continue
            b_ = text.find("\n", a_)
            hl[a_] = (d1 if b_ < 0 or b_ > d1 else b_, lev)
    hs = sorted(hl)
    # columns of each header line (first HEAD_CAP tokens)
    head_id, hcount = [-1] * n, {}
    for c in range(n):
        if not ind[c] or not hs:
            continue
        h = bisect.bisect_right(hs, starts[c]) - 1
        if h >= 0 and starts[c] < hl[hs[h]][0] and hcount.get(h, 0) < HEAD_CAP:
            head_id[c] = h
            hcount[h] = hcount.get(h, 0) + 1
    present = sorted(hcount)                     # header lines with at least one column (truncation-safe)
    pidx = {h: i for i, h in enumerate(present)}
    # active header tuple per column (only headers present among the columns)
    sec, tuples, tid = [-1] * n, [], {}
    active = [None, None, None]
    hp = 0
    order = sorted((hs[h], h) for h in present)
    for c in range(n):
        if not ind[c]:
            continue
        while hp < len(order) and order[hp][0] <= starts[c]:
            h = order[hp][1]
            lev = hl[hs[h]][1]
            active[lev] = pidx[h]
            for l2 in range(lev + 1, 3):
                active[l2] = None
            hp += 1
        t = tuple(x for x in active if x is not None)
        if not t:
            continue
        if t not in tid:
            tid[t] = len(tuples)
            tuples.append(t)
        sec[c] = tid[t]
    n_hdr = len(present)
    sec_mat = torch.zeros(len(tuples), n_hdr)
    for i, t in enumerate(tuples):
        sec_mat[i, list(t)] = 1.0 / len(t)
    hid = [pidx[h] if h >= 0 else -1 for h in head_id]
    return {"par": torch.tensor(par, dtype=torch.long), "sec": torch.tensor(sec, dtype=torch.long),
            "head_id": torch.tensor(hid, dtype=torch.long), "head": torch.tensor([h >= 0 for h in hid]),
            "sec_mat": sec_mat, "n_par": k + 1, "n_sec": len(tuples), "n_hdr": n_hdr, "par_mode": mode}


def seg_for_doc(tok, hub: str, doc_text: str, system: str, pre_ids: list, n: int, block: int = 128) -> dict:
    """Segment ids of the operator columns of one document (pre_ids = the reader prefix ids, possibly left-truncated;
    the operator = pre_ids[:n])."""
    from xenc.state_skill import doc_prefix_text
    ptxt = doc_prefix_text(tok, hub, doc_text, system)
    enc = tok(ptxt, add_special_tokens=False, return_offsets_mapping=True)
    full = list(enc["input_ids"])
    base = len(full) - len(pre_ids)
    assert base >= 0 and full[base:] == list(pre_ids), "SEG: reader prefix ids do not match the prefix text"
    d0 = ptxt.rfind(doc_text) if doc_text else -1
    assert d0 >= 0, "SEG: document text not found in its prefix"
    return segment_ids(ptxt, enc["offset_mapping"], d0, d0 + len(doc_text), n, base, block)


def _seg_mean(X: torch.Tensor, ids: torch.Tensor, n: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    """X [H, N, d]; ids [N] (-1 = none) -> [H, n + 1, d] per-id means (row n = zeros, for id -1)."""
    H, N, d = X.shape
    keep = ids >= 0 if mask is None else (ids >= 0) & mask
    idx = torch.where(keep, ids, torch.full_like(ids, n))
    S = torch.zeros(H, n + 1, d, device=X.device, dtype=X.dtype).index_add_(1, idx, X)
    c = torch.zeros(n + 1, device=X.device, dtype=X.dtype).index_add_(0, idx, torch.ones_like(idx, dtype=X.dtype))
    S = S / c.clamp_min(1.0).view(1, -1, 1)
    S[:, n] = 0.0
    return S


def _sec_mean(X: torch.Tensor, sg: dict) -> torch.Tensor:
    """X [H, N, d] -> [H, n_sec + 1, d] section summaries: the mean over the active header lines of each header tuple of
    each header line's mean (row n_sec = zeros, for sec id -1)."""
    hid = sg["head_id"].to(X.device)
    nh, ns = int(sg["n_hdr"]), int(sg["n_sec"])
    Hm = _seg_mean(X, hid, nh)[:, :nh]                                   # [H, n_hdr, d]
    M = sg["sec_mat"].to(X.device, X.dtype)                              # [n_sec, n_hdr]
    out = torch.zeros(X.shape[0], ns + 1, X.shape[2], device=X.device, dtype=X.dtype)
    if ns and nh:
        out[:, :ns] = torch.einsum("sk,hkd->hsd", M, Hm)
    return out


# ----------------------------------------------------------------------------- interface
def parse_seg(on: int = 0, block: int = 128, parts: str = "kP,kS,vP,vS") -> dict:
    if not int(on):
        return {}
    ps = [p for p in str(parts).split(",") if p]
    assert ps and set(ps) <= {"kP", "kS", "vP", "vS"}, parts
    return {"block": int(block), "parts": sorted(ps)}


class SegInterface(D13Interface):
    """D13Interface + segment-summary key / value terms (zero-init W in self.SEG)."""
    needs_seg = True

    def __init__(self, enc, layer_map: dict, d_r: int, d_e: int, B: dict | None = None, mlp_hidden: int = 0,
                 seg: dict | None = None):
        super().__init__(enc, layer_map, d_r, d_e, B=B, mlp_hidden=mlp_hidden)
        self.seg = dict(seg or {})
        assert self.seg, "SegInterface needs a seg config (parse_seg)"
        a0 = self.att[str(self.layers[0])]
        hd = a0.head_dim
        hkv = a0.k_proj.out_features // hd
        self.SEG = nn.ParameterDict({f"{p}{li}": nn.Parameter(torch.zeros(hkv, hd, hd))
                                     for li in self.layers for p in self.seg["parts"]})
        self.seg_on = True                  # False: same checkpoint read on the plain operator (ablation)
        self._seg_op = None                 # the loaded operator dict (set by SegReader.load)
        self._seg_cache: dict = {}          # no-grad cache of (K', V') per layer for the loaded operator

    def config(self) -> dict:
        c = super().config()
        c["seg"] = dict(self.seg)
        return c

    def make_reader(self, model):
        return SegReader(model, self)

    def seg_params(self) -> list:
        return list(self.SEG.parameters())

    @torch.no_grad()
    def seg_state(self) -> dict:
        return {k: round(float(p.norm()), 5) for k, p in self.SEG.items()}

    def _summaries(self, li, Kop, Vop):
        """Per-document (no grad) pre-RoPE key / value summaries for this read layer's encoder layer, cached in the
        operator dict."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        op = self._seg_op
        e = self.layer_map[li]
        cache = op.setdefault("_segc", {})
        if e in cache and cache[e]["N"] == Kop.shape[1]:
            return cache[e]
        sg = op["seg"]
        K, V = Kop.float(), Vop.float()
        N = K.shape[1]
        assert int(sg["par"].shape[0]) == N, f"SEG: segment ids ({sg['par'].shape[0]}) do not align with the operator ({N})"
        with torch.no_grad():
            pos = torch.arange(N, device=K.device).view(1, 1, N).expand(3, 1, N)
            cos, sin = self.rot(K[:1].transpose(0, 1).reshape(1, N, -1), pos)
            cos, sin = cos.float(), sin.float()
            kpre = apply_rotary_pos_emb(K[None], K[None], cos, -sin)[1][0]
            par, sec = sg["par"].to(K.device), sg["sec"].to(K.device)
            c = {"N": N, "cos": cos, "sin": sin, "par": par, "sec": sec, "n_par": int(sg["n_par"]), "n_sec": int(sg["n_sec"]),
                 "kP": _seg_mean(kpre, par, int(sg["n_par"])), "vP": _seg_mean(V, par, int(sg["n_par"])),
                 "kS": _sec_mean(kpre, sg), "vS": _sec_mean(V, sg)}
        cache[e] = c
        return c

    def seg_kv(self, li, Kop, Vop):
        """(K', V') for read layer li (fp32)."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        K, V = Kop.float(), Vop.float()
        if not self.seg_on or self._seg_op is None:
            return K, V
        nog = not torch.is_grad_enabled()
        if nog and li in self._seg_cache:
            return self._seg_cache[li]
        c = self._summaries(li, Kop, Vop)
        idx = {"P": torch.where(c["par"] >= 0, c["par"], torch.full_like(c["par"], c["n_par"])),
               "S": torch.where(c["sec"] >= 0, c["sec"], torch.full_like(c["sec"], c["n_sec"]))}
        dk, dv = None, None
        for p in self.seg["parts"]:
            W = self.SEG[f"{p}{li}"]
            src = c[p]                                                        # [H, n+1, hd]
            g = torch.einsum("hnd,hde->hne", src, W).index_select(1, idx[p[1]])    # [H, N, hd]
            if p[0] == "k":
                dk = g if dk is None else dk + g
            else:
                dv = g if dv is None else dv + g
        if dk is not None:
            K = K + apply_rotary_pos_emb(dk[None], dk[None], c["cos"], c["sin"])[1][0]
        if dv is not None:
            V = V + dv
        if nog:
            self._seg_cache[li] = (K, V)
        return K, V

    def read(self, li, x, Kop, Vop, pos0: int, past=None, need_logits: bool = False):
        K, V = self.seg_kv(li, Kop, Vop)
        return super().read(li, x, K, V, pos0, past=past, need_logits=need_logits)


class SegReader(D13Reader):
    """D13Reader that hands the loaded operator dict (with its "seg" ids) to the interface."""

    def load(self, op: dict, n_op: int):
        super().load(op, n_op)
        itf = self.itf
        itf._seg_cache = {}
        if itf.seg_on:
            assert "seg" in op, "SEG: the operator carries no segment ids (harvest with a SegInterface)"
        itf._seg_op = op


def upgrade_seg(itf: D13Interface, enc, seg: dict) -> SegInterface:
    """A loaded plain D13 base -> SegInterface with identical A / MLP / B / bias / gates and W = 0."""
    assert type(itf) is D13Interface, f"SEG on a {type(itf).__name__} init is not supported"
    assert not itf.delta_rank
    B = None if itf.identity_B else {li: itf.B[str(li)].detach().cpu() for li in itf.layers}
    new = SegInterface(enc, itf.layer_map, itf.d_r, itf.d_e, B=B, mlp_hidden=itf.mlp_hidden, seg=seg)
    if getattr(itf, "mlp_B", 0):
        new.add_mlp_B(itf.mlp_B)
    sd = {k: v for k, v in itf.state_dict().items() if not k.startswith("D.")}
    missing, unexpected = new.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("SEG.") for k in missing), missing
    for k in ("read_tau", "read_logn_ref", "att_trained"):
        if hasattr(itf, k):
            setattr(new, k, getattr(itf, k))
    return new.to(itf.g.device)


__all__ = ["SegInterface", "SegReader", "parse_seg", "upgrade_seg", "segment_ids", "seg_for_doc", "HDR"]
