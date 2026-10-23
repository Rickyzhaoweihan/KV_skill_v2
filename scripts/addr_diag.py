"""ADDR diagnostic (EVAL-ONLY; nothing is trained or saved besides the report): on span tasks, where does the 9B TEXT
reader's advantage over the kv read live, and why does a read that HITS the gold span still decode a wrong answer?

Items: held-out VAL documents only (the P3 pilot val spec: Qasper / MuSiQue / LastingBench / LooGLE), D22 judged docs
and lever-1 near-duplicates excluded (asserted), untruncated prefixes only (the text pass and the operator see the
same document), so text position N+t == kv row t exactly (asserted per item).

(1) PATCHING DECOMPOSITION, teacher-forced on "<answer>gold</answer>" (gold = the answer string found whole-word in the
    operator when there is one, else the dataset gold). Every condition is one forward; metrics on the answer-content
    rows: content NLL, first-content-token NLL / top-1, all-content top-1 (tf_em = greedy would reproduce the gold
    given the gold prefix). Layer groups of the 9B reader: R = the read layers (3,7,11,15,19), D = the deep
    full-attention layers (23,27,31), Lin = every Gated-DeltaNet (linear-attention) layer.
      text        the 9B reads the document as text
      kv          the base read (D13, unregistered; e2e = inference condition)
      kv+X        kv, with the token-mixer OUTPUTS at the reader rows of group X replaced by the text pass's outputs
                  (X = R, R19, D, RD, Lin, R+Lin, D+Lin, all; kv+all must equal text: sanity row)
      text-X      text, with group X's outputs at the reader rows replaced by the kv pass's outputs (X = D, Lin)
      text*R      text, but at the read layers the rows' attention output is the D13 read of the operator computed
                  from the TEXT pass's own layer input (the read with text-quality queries); text*R19 at layer 19 only
      ora_tok@L   kv with the read's softmax at layers L replaced, on each answer-content row, by a one-hot on the
                  operator column holding the NEXT gold token (uniform over gold occurrences; P3's char-offset rule).
                  ora_span@19: uniform over every gold-span column on every content row.
(2) What the read writes back: cos / relative error of the kv write-back (gated) vs the text attention output at each
    read layer (question rows, content rows); logit lens (final norm + unembedding) of the residual stream after every
    layer at the first content row, text vs kv: where in depth the gold token emerges.
(3) Addressing (E3-comparable, base kv e2e logits): first-content-row hit (argmax within ±3 of the gold span) and gold
    mass at layers 15 / 19, head mixture over the body columns.
(4) Greedy decode (budget 32): kv and text answers, F1 / EM, and P3's error anatomy for both (correct / contains gold /
    same paragraph / other paragraph / not in doc / no answer), split by the read's first-row hit.

    python scripts/addr_diag.py --init outputs/D13_lever1_full/d13_best.pt --out outputs/ADDR_diag_L1f \
        --enc_device cuda:0 --device cuda:1
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import types
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401

SPAN_TASKS = ("qasper", "musique", "lastingbench", "loogle")
VAL_SPEC = "quality:5,musique:70,lastingbench:50,longhealth:0,qasper:90,contractnli:2,shakespeare_cf:1,loogle:0,clipper:1"
READ = (3, 7, 11, 15, 19)
DEEP = (23, 27, 31)
KV_PATCH = ("R", "R19", "D", "RD", "Lin", "R+Lin", "D+Lin", "all")


# ============================================================================ pure helpers (CPU-tested)
def tf_metrics(logits: torch.Tensor, tt: list, loc: list) -> dict:
    """logits [m+1, V] (window rows; row i predicts tt[i]); loc = content token indices -> per-item metrics."""
    lg = logits[:-1].float() if logits.shape[0] == len(tt) + 1 else logits.float()
    assert lg.shape[0] == len(tt), (lg.shape, len(tt))
    lp = torch.log_softmax(lg, -1)
    tgt = torch.tensor(tt, device=lp.device)
    nll = -lp.gather(-1, tgt.view(-1, 1)).view(-1)
    top = lp.argmax(-1) == tgt
    c = torch.tensor(loc, device=lp.device)
    return {"nll": float(nll[c].mean()), "nll_first": float(nll[c[0]]), "top1_first": float(top[c[0]]),
            "acc": float(top[c].float().mean()), "tf_em": float(top[c].all())}


def group_layers(name: str, full: list, lin: list) -> set:
    """Layer set of a patch group name ("R", "R19", "D", "RD", "Lin", "R+Lin", "D+Lin", "all")."""
    R, D = set(READ) & set(full), set(DEEP) & set(full)
    g = {"R": R, "R19": {19}, "D": D, "RD": R | D, "Lin": set(lin), "R+Lin": R | set(lin), "D+Lin": D | set(lin),
         "all": set(full) | set(lin)}
    return g[name]


def content_offset_chars(tok, target: str, loc: list, open_tag: str = "<answer>") -> list:
    """For each content token index in loc: number of content characters BEFORE that token (>= 0)."""
    off = tok(target, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    return [max(0, off[i][0] - len(open_tag)) for i in loc]


def stat(x):
    n = len(x)
    if not n:
        return float("nan"), float("nan")
    m = sum(x) / n
    return m, (math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1) / n) if n > 1 else float("nan"))


# ============================================================================ hooks
class Mixers:
    """Forward hooks on every decoder layer's token mixer (self_attn / linear_attn; chained after any earlier
    monkeypatch such as D13Reader) and on the decoder layer output. While self.T is set (last T rows of the forward):
      rec  {} -> records each mixer's (possibly substituted) output rows [T, d] (fp32)
      res  {} -> records each decoder layer's output rows [T, d] (fp32)
      sub  {i: [T, d]} replaces mixer i's output rows
      fn   {i: f(hs_rows, out_rows) -> new rows} computes the replacement from the mixer's input rows (text*R)."""

    def __init__(self, model):
        from deltaskill.backbone_utils import decoder_layers
        L = decoder_layers(model)
        self.full = [i for i, l in enumerate(L) if hasattr(l, "self_attn")]
        self.lin = [i for i, l in enumerate(L) if hasattr(l, "linear_attn")]
        self.T, self.rec, self.res, self.sub, self.fn = None, None, None, {}, {}
        self.h = []
        for i, l in enumerate(L):
            mod = l.self_attn if hasattr(l, "self_attn") else l.linear_attn
            self.h.append(mod.register_forward_hook(self._mix(i), with_kwargs=True))
            self.h.append(l.register_forward_hook(self._lay(i)))

    def set(self, T=None, rec=False, sub=None, fn=None):
        self.T = T
        self.rec = {} if rec else None
        self.res = {} if rec else None
        self.sub = sub or {}
        self.fn = fn or {}

    def _mix(self, i):
        def h(mod, args, kwargs, out):
            if self.T is None:
                return None
            o = out[0] if isinstance(out, tuple) else out
            T = self.T
            new = None
            if i in self.fn:
                hs = kwargs.get("hidden_states", args[0] if args else None)
                new = self.fn[i](hs[0, -T:].float(), o[0, -T:].float())
            elif i in self.sub:
                new = self.sub[i]
            if new is not None:
                o = o.clone()
                o[0, -T:] = new.to(o.dtype)
            if self.rec is not None:
                self.rec[i] = o[0, -T:].detach().float().clone()
            if new is None:
                return None
            return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
        return h

    def _lay(self, i):
        def h(mod, args, out):
            if self.T is None or self.res is None:
                return None
            o = out[0] if isinstance(out, tuple) else out
            self.res[i] = o[0, -self.T:].detach().float().clone()
            return None
        return h


# ============================================================================ GPU part
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--init", required=True)
    ap.add_argument("--val", default=VAL_SPEC)
    ap.add_argument("--val_q_per_doc", type=int, default=4)
    ap.add_argument("--val_seed", type=int, default=1)
    ap.add_argument("--val_fold", type=int, default=1)
    ap.add_argument("--exclude_sets", default=",".join(sorted(glob.glob("outputs/D22_sets/*.json"))))
    ap.add_argument("--exclude_docs", default="outputs/D13_lever1_leakage.json")
    ap.add_argument("--doc_len", default="loogle:32000:96000,clipper:32000:96000")
    ap.add_argument("--tasks", default=",".join(SPAN_TASKS))
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=49000, help="untruncated prefixes only: len(prefix) <= this - 1024")
    ap.add_argument("--lru", type=int, default=2)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--no_gen", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enc_device", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    import xenc.d13_read as d13
    from d22_c3_format import REFUSE
    from fit_d13 import D13Rt, build, select_items
    from p3_diag import anatomy, col_covering, occurrences, oracle_columns
    from register_d13 import doc_len_filter, excluded_docs
    from register_doc import SYSTEM_ANSWER, _get, log
    from register_state import _stop_ids
    from xenc.backbone import build_chat, decode, text_ids
    from xenc.doc_tasks import ctx_prompt, exact_match, extract_answer, f1_score, gold_text
    from xenc.e3_retrieval import answer_content_rows, dilate, mixture_logp
    from xenc.state_skill import doc_prefix_text
    t0 = time.time()
    reader, enc, tok, itf, lmap = build(a, init=a.init)
    assert type(itf).__name__ in ("D13Interface", "SegInterface"), f"plain D13 / SEG checkpoint expected, got {type(itf).__name__}"
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    ck_step = (ck.get("extra") or {}).get("step")
    rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens, lru=a.lru)
    val, docs = select_items(a.val, "val", {a.val_fold}, a.val_q_per_doc, a.val_seed)
    drop = excluded_docs(a.exclude_sets)
    if a.exclude_docs:
        X = json.load(open(a.exclude_docs))
        drop |= set(sum(X.values(), [])) if isinstance(X, dict) else set(X)
    keep = doc_len_filter(a.doc_len)
    tasks = set(a.tasks.split(","))
    val = [i for i in val if _get(i, "doc_id") not in drop and keep(_get(i, "doc_id"))
           and _get(i, "doc_id").split("_")[0] in tasks and not _get(i, "choices")]
    assert not ({_get(i, "doc_id") for i in val} & drop), "excluded doc in diagnostic items"
    rt.docs = docs
    S_body = len(tok(doc_prefix_text(tok, a.reader, "", SYSTEM_ANSWER), add_special_tokens=False)["input_ids"])
    M = Mixers(reader)
    assert set(READ) <= set(M.full) and set(DEEP) <= set(M.full), (M.full,)
    norm, head = reader.model.norm, reader.get_output_embeddings()
    gpu = torch.cuda.get_device_name(torch.device(a.device)) if torch.cuda.is_available() else "cpu"
    log(f"ADDR diag: init {a.init} (step {ck_step}) | encoder {a.encoder} | B {a.B} | layer map {lmap} | full-attn "
        f"{M.full} | linear {len(M.lin)} layers | candidate items {len(val)} | excluded docs {len(drop)} | body S={S_body} "
        f"| GPU {gpu}")

    # ---- oracle read: patch the module-level _attend (D13Interface.read resolves it at call time)
    CUR, ORA = [None], {"layers": set(), "rows": {}}
    _attend0 = d13._attend

    def _attend_ora(q, Kop, Vop, Kown, Vown, scale, need_logits=False):
        if CUR[0] not in ORA["layers"] or not ORA["rows"]:
            return _attend0(q, Kop, Vop, Kown, Vown, scale, need_logits)
        Hq, T, hd = q.shape
        Hkv = Kop.shape[0]
        G = Hq // Hkv
        K = torch.cat([Kop, Kown], 1)
        V = torch.cat([Vop, Vown], 1)
        C = K.shape[1]
        s = torch.einsum("kgtd,kcd->kgtc", q.view(Hkv, G, T, hd), K) * scale
        col = torch.arange(C, device=q.device).view(1, C)
        row = torch.arange(T, device=q.device).view(T, 1)
        s = s.masked_fill((col > (C - T) + row).view(1, 1, T, C), float("-inf"))
        p = torch.softmax(s, -1)
        for r, cols in ORA["rows"].items():
            p[:, :, r, :] = 0.0
            p[:, :, r, cols] = 1.0 / len(cols)
        o = torch.einsum("kgtc,kcd->kgtd", p, V).reshape(Hq, T, hd).transpose(0, 1).reshape(T, Hq * hd)
        return {"o": o, "logits": s} if need_logits else {"o": o}
    d13._attend = _attend_ora
    _read0 = itf.read

    def _read_w(li, *args, **kw):
        CUR[0] = li
        return _read0(li, *args, **kw)
    itf.read = _read_w

    R = rt.R
    stops = _stop_ids(tok)

    def fwd_text(tids, T, m, rec=False, sub=None, fn=None):
        R.mode = "off"
        M.set(T, rec, sub, fn)
        try:
            out = reader(input_ids=torch.tensor([tids], device=rt.dev), use_cache=False, logits_to_keep=m + 1)
            return out.logits[0, -(m + 1):].float(), M.rec, M.res
        finally:
            M.set(None)

    def fwd_kv(rows, m, op, n_op, rec=False, sub=None, ora=None, logits=False):
        R.load(op, n_op)
        R.mode, R.need_logits, R.rec = "e2e", bool(logits), {}
        ORA["layers"], ORA["rows"] = (set(ora[0]), ora[1]) if ora else (set(), {})
        M.set(len(rows), rec, sub)
        try:
            out = reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=False, logits_to_keep=m + 1)
            lg = {li: R.rec[li]["logits"] for li in lmap} if logits else None
            return out.logits[0, -(m + 1):].float(), M.rec, M.res, lg
        finally:
            R.mode, R.rec, R.need_logits, R.past = "off", {}, False, {}
            ORA["layers"], ORA["rows"] = set(), {}
            M.set(None)

    @torch.no_grad()
    def gen_text(prompt):
        R.mode = "off"
        out = reader(input_ids=torch.tensor([prompt], device=rt.dev), use_cache=True, logits_to_keep=1)
        new = []
        for _ in range(a.budget):
            nxt = int(out.logits[0, -1].argmax())
            new.append(nxt)
            if nxt in stops or "</answer>" in tok.decode(new[-12:], skip_special_tokens=True):
                break
            out = reader(input_ids=torch.tensor([[nxt]], device=rt.dev), past_key_values=out.past_key_values,
                         use_cache=True, logits_to_keep=1)
        txt = decode(tok, a.reader, new)
        return (txt.split("</answer>", 1)[0] + "</answer>") if "</answer>" in txt else txt

    def lens(res: dict, r: int, gid: int) -> list:
        with torch.no_grad():
            H = torch.stack([res[i][r] for i in sorted(res)])                     # [L, d]
            lp = torch.log_softmax(head(norm(H.to(head.weight.dtype))).float(), -1)
            return [round(float(x), 3) for x in lp[:, gid]]

    def cos_rows(x, y, rows_):
        if not rows_:
            return None
        xs, ys = x[rows_], y[rows_]
        c = torch.nn.functional.cosine_similarity(xs, ys, dim=-1).mean()
        re_ = ((xs - ys).pow(2).sum(-1) / ys.pow(2).sum(-1).clamp_min(1e-6)).mean()
        return [round(float(c), 4), round(float(re_), 4)]

    out_rows, skip = [], Counter()
    fo = open(os.path.join(a.out, "items.jsonl"), "w")
    for k, it in enumerate(val):
        if a.max_items and len(out_rows) >= a.max_items:
            break
        did = _get(it, "doc_id")
        doc = docs[did]
        ptxt = doc_prefix_text(tok, a.reader, doc, SYSTEM_ANSWER)
        enc_ = tok(ptxt, add_special_tokens=False, return_offsets_mapping=True)
        full = list(enc_["input_ids"])
        if full != rt.pre(did):
            skip["truncated_or_mismatch"] += 1
            continue
        N = len(full) - a.tail
        ends = [e for _, e in enc_["offset_mapping"]]
        ans, spans = occurrences(ptxt, doc, _get(it, "answers"), ends[N - 1])
        target = f"<answer>{ans}</answer>" if spans else gold_text(it)
        tt, loc = answer_content_rows(tok, target)
        if tt != text_ids(tok, target) or not loc:
            skip["target_fmt"] += 1
            continue
        m = len(tt)
        rows = rt.rows(did, it, tt)
        T = len(rows)
        prompt = build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, doc))
        tids = prompt + tt
        retok = False
        if (tids[-T:] != rows or len(tids) != N + T) and tids[:N] == full[:N]:
            # e.g. Qasper docs end with "\n", which BPE merges with "\n\n## Question" in the full prompt: use the
            # operator columns + the kv rows (one boundary token tokenized as in the kv path) so rows align exactly
            tids, retok = full[:N] + rows, True
            prompt = tids[:len(tids) - m]
        if tids[-T:] != rows or tids[:N] != full[:N] or len(tids) != N + T:
            skip["row_alignment"] += 1
            continue
        crow = [T - (m + 1) + i for i in loc]                  # absolute reader rows predicting the content tokens
        qrows = list(range(a.tail, T - (m + 1)))               # question rows (after the doc tail, before the answer)
        rec = {"key": str(_get(it, "key")), "task": did.split("_")[0], "n_op": N, "m": m, "gold_in_op": bool(spans),
               "target": target, "tf": {}, "text_retok": retok}
        try:
            with torch.no_grad():
                op, n_op, _ = rt.operator(did)
                assert n_op == N
                # --- base passes (recording)
                lt, recT, resT = fwd_text(tids, T, m, rec=True)
                lk, recK, resK, lgK = fwd_kv(rows, m, op, n_op, rec=True, logits=True)
                rec["tf"]["text"] = tf_metrics(lt, tt, loc)
                rec["tf"]["kv"] = tf_metrics(lk, tt, loc)
                gid = tt[loc[0]]
                rec["lens_text"] = lens(resT, crow[0], gid)
                rec["lens_kv"] = lens(resK, crow[0], gid)
                rec["wb"] = {str(li): {"q": cos_rows(recK[li], recT[li], qrows), "c": cos_rows(recK[li], recT[li], crow)}
                             for li in READ}
                rec["wb_deep"] = {str(li): {"q": cos_rows(recK[li], recT[li], qrows)} for li in DEEP}
                # --- kv with text outputs patched in
                for g in KV_PATCH:
                    Ls = group_layers(g, M.full, M.lin)
                    lg_, _, resP, _ = fwd_kv(rows, m, op, n_op, rec=(g == "R"), sub={i: recT[i] for i in Ls})
                    rec["tf"][f"kv+{g}"] = tf_metrics(lg_, tt, loc)
                    if g == "R":
                        rec["lens_kvR"] = lens(resP, crow[0], gid)
                # --- text with kv outputs patched in
                for g in ("D", "Lin"):
                    Ls = group_layers(g, M.full, M.lin)
                    lg_, _, _ = fwd_text(tids, T, m, sub={i: recK[i] for i in Ls})
                    rec["tf"][f"text-{g}"] = tf_metrics(lg_, tt, loc)

                # --- text-context read: the D13 read from the text pass's own layer input at the read layers
                def mk(li):
                    e = lmap[li]
                    K_, V_ = op[e]

                    def f(hs_rows, out_rows):
                        CUR[0] = li
                        r_ = itf.read(li, hs_rows, K_, V_, n_op)
                        g_ = itf.gate(li)
                        return out_rows + g_ * (r_["y"] - out_rows)
                    return f
                for nm, Ls in (("text*R", READ), ("text*R19", (19,))):
                    lg_, _, _ = fwd_text(tids, T, m, fn={li: mk(li) for li in Ls})
                    rec["tf"][nm] = tf_metrics(lg_, tt, loc)
                # --- addressing + oracle read (gold inside the operator only)
                if spans:
                    gm = torch.zeros(N, dtype=torch.bool)
                    for s_, e_ in spans:
                        for c in range(col_covering(ends, s_), min(col_covering(ends, e_ - 1) + 1, N)):
                            gm[c] = True
                    gmb = gm[S_body:].to(rt.dev)
                    gmw = dilate(gmb, 3)
                    addr = {}
                    for li in (15, 19):
                        lp = mixture_logp(lgK[li], [crow[0]], S_body, N)[0]
                        addr[str(li)] = {"hit": float(gmw[lp.argmax()]), "mass": float(lp.exp()[gmb].sum()),
                                         "top1_col": int(lp.argmax()) + S_body}
                    rec["addr"] = addr
                    Lc = content_offset_chars(tok, target, loc)
                    ora_tok = {}
                    for r_, L_ in zip(crow, Lc):
                        cols = oracle_columns(ends, spans, L_, N)
                        if cols:
                            ora_tok[r_] = cols
                    span_cols = gm.nonzero().flatten().tolist()
                    rec["ora_rows"] = len(ora_tok)
                    for nm, Ls, rowsmap in (("ora_tok@19", (19,), ora_tok), ("ora_tok@15,19", (15, 19), ora_tok),
                                            ("ora_tok@all", READ, ora_tok),
                                            ("ora_span@19", (19,), {r_: span_cols for r_ in crow})):
                        if not rowsmap:
                            continue
                        lg_, _, _, _ = fwd_kv(rows, m, op, n_op, ora=(Ls, rowsmap))
                        rec["tf"][nm] = tf_metrics(lg_, tt, loc)
                del recT, resT, recK, resK, lgK
            # --- greedy decodes + anatomy
            if not a.no_gen:
                gk, _ = rt.gen(did, it, a.budget)
                gtx = gen_text(prompt)
                d0 = ptxt.rfind(doc)
                gold_doc = [(s_ - d0, e_ - d0) for s_, e_ in spans]
                for nm, g in (("kv", gk), ("text", gtx)):
                    pred = extract_answer(g)
                    f1 = f1_score(pred, list(_get(it, "answers")))
                    em = exact_match(pred, list(_get(it, "answers")))
                    an = (anatomy(f1, em, pred, list(_get(it, "answers")), doc, gold_doc, lambda s_: bool(REFUSE.match(s_)))
                          if spans else ("correct" if em >= 1 else "gold_not_in_op"))
                    rec[f"gen_{nm}"] = {"gen": g, "f1": f1, "em": em, "anatomy": an}
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            rt._ops.clear()
            skip["oom"] += 1
            log(f"  OOM on {rec['key']} ({N} columns); item skipped")
            continue
        out_rows.append(rec)
        fo.write(json.dumps(rec) + "\n")
        fo.flush()
        if len(out_rows) % 10 == 0:
            log(f"  {len(out_rows)} items ({time.time() - t0:.0f}s) | skips {dict(skip)}")
    fo.close()
    summ = summarize(out_rows)
    summ.update({"init": a.init, "init_step": ck_step, "n": len(out_rows), "skips": dict(skip), "gpu": gpu,
                 "args": vars(a)})
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    md = to_markdown(summ)
    open(os.path.join(a.out, "summary.md"), "w").write(md)
    log(md)
    log(f"done: {len(out_rows)} items, skips {dict(skip)} ({time.time() - t0:.0f}s)")
    return 0


# ============================================================================ summaries (CPU; rerunnable from items.jsonl)
METRICS = ("nll", "nll_first", "top1_first", "acc", "tf_em")


def _paired(rows, a_, b_, met):
    d = [r["tf"][a_][met] - r["tf"][b_][met] for r in rows if a_ in r["tf"] and b_ in r["tf"]]
    return stat(d) + (len(d),)


def summarize(rows: list) -> dict:
    out = {"groups": {}}
    groups = {"all": rows, "gold_in_op": [r for r in rows if r["gold_in_op"]]}
    for r in rows:
        groups.setdefault(r["task"], []).append(r)
    hit = [r for r in rows if r.get("addr") and r["addr"]["19"]["hit"] >= 1]
    miss = [r for r in rows if r.get("addr") and r["addr"]["19"]["hit"] < 1]
    groups["hit19"], groups["miss19"] = hit, miss
    for g, rs in groups.items():
        if not rs:
            continue
        conds = sorted({c for r in rs for c in r["tf"]}, key=lambda c: (c not in ("text", "kv"), c))
        d = {"n": len(rs), "conds": {}}
        for c in conds:
            sub = [r for r in rs if c in r["tf"]]
            e = {"n": len(sub)}
            for met in METRICS:
                e[met] = stat([r["tf"][c][met] for r in sub])[0]
                if c != "kv":
                    e[f"d_kv_{met}"] = _paired(sub, c, "kv", met)
                if c != "text":
                    e[f"d_text_{met}"] = _paired(sub, c, "text", met)
            d["conds"][c] = e
        gk = [r for r in rs if "gen_kv" in r]
        if gk:
            d["gen"] = {nm: {"f1": stat([r[f"gen_{nm}"]["f1"] for r in gk])[0], "em": stat([r[f"gen_{nm}"]["em"] for r in gk])[0],
                             "anatomy": dict(Counter(r[f"gen_{nm}"]["anatomy"] for r in gk))} for nm in ("kv", "text")}
            d["gen"]["joint"] = dict(Counter(f"{r['gen_kv']['anatomy']}|{r['gen_text']['anatomy']}" for r in gk))
        lt = [r["lens_text"] for r in rs if "lens_text" in r]
        if lt:
            nl = len(lt[0])
            d["lens"] = {k_: [round(sum(x[i] for x in [r[k_] for r in rs if k_ in r]) / len([r for r in rs if k_ in r]), 3)
                              for i in range(nl)] for k_ in ("lens_text", "lens_kv", "lens_kvR")
                         if any(k_ in r for r in rs)}
        wb = {}
        for li in map(str, READ + DEEP):
            for part in ("q", "c"):
                v = [(r.get("wb", {}).get(li) or r.get("wb_deep", {}).get(li) or {}).get(part) for r in rs]
                v = [x for x in v if x]
                if v:
                    wb[f"{li}{part}"] = [round(sum(x[0] for x in v) / len(v), 4), round(sum(x[1] for x in v) / len(v), 4)]
        d["writeback_cos_relerr"] = wb
        ad = [r["addr"] for r in rs if r.get("addr")]
        if ad:
            d["addr"] = {li: {"hit": stat([x[li]["hit"] for x in ad])[0], "mass": stat([x[li]["mass"] for x in ad])[0]}
                         for li in ("15", "19")}
        out["groups"][g] = d
    return out


def to_markdown(s: dict) -> str:
    f2 = lambda t: f"{t[0]:+.3f} ± {t[1]:.3f}"   # noqa: E731
    L = [f"init `{s.get('init')}` (step {s.get('init_step')}) | n {s.get('n')} | skips {s.get('skips')} | {s.get('gpu')}", ""]
    for g in ("all", "gold_in_op", "hit19", "miss19", "qasper", "musique", "lastingbench", "loogle"):
        d = s["groups"].get(g)
        if not d:
            continue
        L += [f"### {g} (n={d['n']})", "",
              "| condition | n | content NLL | first NLL | first top-1 | content top-1 | tf_em | Δ first top-1 vs kv | Δ tf_em vs kv | Δ tf_em vs text |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c, e in d["conds"].items():
            L.append(f"| {c} | {e['n']} | {e['nll']:.3f} | {e['nll_first']:.3f} | {e['top1_first']:.3f} | {e['acc']:.3f} | "
                     f"{e['tf_em']:.3f} | {f2(e['d_kv_top1_first']) if 'd_kv_top1_first' in e else '—'} | "
                     f"{f2(e['d_kv_tf_em']) if 'd_kv_tf_em' in e else '—'} | {f2(e['d_text_tf_em']) if 'd_text_tf_em' in e else '—'} |")
        if d.get("gen"):
            G = d["gen"]
            L += ["", f"greedy F1 / EM: kv {G['kv']['f1']:.3f} / {G['kv']['em']:.3f} | text {G['text']['f1']:.3f} / "
                  f"{G['text']['em']:.3f}", f"anatomy kv {G['kv']['anatomy']}", f"anatomy text {G['text']['anatomy']}"]
        if d.get("addr"):
            L.append(f"addressing (first content row): L15 hit {d['addr']['15']['hit']:.3f} mass {d['addr']['15']['mass']:.3f} | "
                     f"L19 hit {d['addr']['19']['hit']:.3f} mass {d['addr']['19']['mass']:.3f}")
        if d.get("writeback_cos_relerr"):
            L.append("kv vs text mixer output at rows (cos, rel. err; q = question rows, c = content rows): "
                     + " ".join(f"L{k}: {v[0]:.3f}/{v[1]:.2f}" for k, v in d["writeback_cos_relerr"].items()))
        if d.get("lens"):
            for k_, v in d["lens"].items():
                L.append(f"{k_} (gold first-token log p after each layer): " + " ".join(f"{x:.1f}" for x in v))
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--resummarize":
        rows_ = [json.loads(x) for x in open(sys.argv[2])]
        S_ = summarize(rows_)
        S_["n"] = len(rows_)
        print(to_markdown(S_))
        sys.exit(0)
    sys.exit(main())
