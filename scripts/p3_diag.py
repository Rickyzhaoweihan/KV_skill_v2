"""P3 diagnostics (EVAL-ONLY; nothing is trained or saved besides the report): why does the token-identity channel
(xenc/p3_tok.py) not raise span F1?

On span val items (the P3 pilot val; held-out docs; Qasper / MuSiQue / LastingBench / LooGLE) whose gold answer occurs
inside the operator columns, with a TRAINED token-channel checkpoint (e.g. outputs/P3_b/d13_best.pt):

(A) ORACLE ADDRESSING. Greedy decode three ways on the same items:
      on      the checkpoint as trained (channel reads the read's own attention p)
      off     the same checkpoint, channel disabled (itf.tok_on = False)
      oracle  the channel's p is REPLACED (channel only; the V read keeps its own p) by a one-hot on the document column
              that holds the NEXT gold token: after "<answer>" has been generated and L content characters emitted, the
              column whose token covers gold-occurrence char start + L (uniform over all occurrences); the table's shift
              blocks then carry that token's ±1 neighbours. Rows before "<answer>" or past the span use the normal p.
    If oracle >> on, the channel can carry tokens and ADDRESSING is the bottleneck; if oracle ≈ on, the channel / its
    integration is (γU, the reader ignoring it, the embedding space).
(B) ERROR ANATOMY (normal model, channel on):
      - teacher-forced on "<answer>gold</answer>": at each content row, is the gold next token among the top-k (1/4/16)
        attended operator columns (head-averaged p at the channel layer)? Token identity uses the space variants
        (answer "Bob" vs document " Bob", merged ">Bob"); also positional hit (top-k column within ±3 of the gold span).
      - each wrong greedy answer: verbatim document substring? same paragraph (no newline between it and a gold
        occurrence)? -> "cannot reproduce" / "wrong place" / "right region, wrong span" / "contains gold (format)".

    python scripts/p3_diag.py --init outputs/P3_b/d13_best.pt --out outputs/P3_diag_b
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import math
import os
import re
import sys
import time
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


# ============================================================================ pure helpers (CPU-tested)
def occurrences(prefix_text: str, doc_text: str, answers, end_char: int, min_chars: int = 2):
    """-> (answer, [(s, e) char spans]) for the FIRST answer with a case-insensitive occurrence in the document part of
    prefix_text ending before end_char (= the end of the operator columns); (None, []) if none."""
    d0 = prefix_text.rfind(doc_text) if doc_text else -1
    if d0 < 0:
        return None, []
    for a in answers or []:
        a_ = (a or "").strip()
        if len(a_) < min_chars or a_.lower() in ("yes", "no", "unanswerable"):
            continue
        # whole-word, case-insensitive (a raw find matches "yes"/"no" inside "Bayes"/"monolingual")
        pat = re.compile(r"(?<!\w)" + re.escape(a_) + r"(?!\w)", re.IGNORECASE)
        spans = [(m.start(), m.end()) for m in pat.finditer(prefix_text, d0) if m.end() <= end_char]
        if spans:
            return a_, spans
    return None, []


def col_covering(ends: list, c: int) -> int:
    """ends = token end offsets (non-decreasing); -> index of the first token whose end > c (the token covering /
    following char c)."""
    return bisect.bisect_right(ends, c)


def oracle_columns(ends: list, spans, L: int, N: int) -> list:
    """Columns holding the next gold token after L emitted content chars, one per occurrence (< N, inside the span)."""
    out = []
    for s, e in spans:
        c = s + L
        if c >= e:
            continue
        j = col_covering(ends, c)
        if j < N:
            out.append(j)
    return sorted(set(out))


def strip_tok(t: str) -> str:
    """Token string without a merged leading '>' and the 'Ġ' space marker."""
    t = t[1:] if t.startswith(">") and len(t) > 1 else t
    return t[1:] if t.startswith("\u0120") else t


def tok_match(doc_tok: str, gold_tok: str, first: bool) -> bool:
    """Document token at an attended column vs the gold answer token (as the reader must emit it). Equal after
    stripping the space marker / merged '>'; on the FIRST content row the answer is tokenized without the leading
    space and BPE may split differently ("labeled" -> "l" + "abeled" vs document "Ġlabeled"), so a gold token that is
    a non-empty prefix of the stripped document token also counts."""
    d, g = strip_tok(doc_tok), strip_tok(gold_tok)
    if not g:
        return False
    return d == g or (first and d.startswith(g))


def space_variants(tok_str: str, vocab: dict) -> set:
    """Ids of the document-side equivalents of an answer token string: itself, with / without the leading 'Ġ' space
    marker, and with a merged leading '>' stripped ('>Bob' -> 'Bob' / 'ĠBob')."""
    out = set()
    base = tok_str
    if base.startswith(">") and len(base) > 1:
        base = base[1:]
    core = base[1:] if base.startswith("Ġ") else base
    for v in (tok_str, base, core, "Ġ" + core):
        if v in vocab:
            out.add(int(vocab[v]))
    return out


def same_paragraph(doc: str, pred: str, gold_spans_doc: list) -> tuple:
    """pred: predicted answer text; gold_spans_doc: gold (s, e) char spans IN doc. -> (verbatim, same_par, min_dist)
    verbatim = case-insensitive raw substring of doc; same_par = some occurrence of pred has no newline between it
    and some gold occurrence."""
    p = (pred or "").strip()
    if len(p) < 1:
        return False, False, None
    low, pl = doc.lower(), p.lower()
    occ, j = [], low.find(pl)
    while j >= 0 and len(occ) < 200:
        occ.append((j, j + len(p)))
        j = low.find(pl, j + 1)
    if not occ:
        return False, False, None
    best, same = None, False
    for s, e in occ:
        for gs, ge in gold_spans_doc:
            a, b = (e, gs) if e <= gs else (ge, s) if ge <= s else (s, s)
            d = max(0, b - a)
            best = d if best is None else min(best, d)
            if "\n" not in doc[min(a, b): max(a, b)]:
                same = True
    return True, same, best


def anatomy(f1: float, em: float, pred: str, answers, doc: str, gold_doc_spans, refuse) -> str:
    from xenc.doc_tasks import normalize_answer
    if em >= 1.0:
        return "correct"
    if not normalize_answer(pred) or refuse(normalize_answer(pred)):
        return "no_answer"
    if any(normalize_answer(a) and normalize_answer(a) in normalize_answer(pred) for a in answers):
        return "contains_gold"
    verb, same, _ = same_paragraph(doc, pred, gold_doc_spans)
    if not verb:
        return "not_in_doc" if normalize_answer(pred) not in normalize_answer(doc) else "in_doc_normalized_only"
    return "same_paragraph" if same else "other_paragraph"


def stat(x):
    n = len(x)
    if not n:
        return float("nan"), float("nan")
    m = sum(x) / n
    return m, (math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1) / n) if n > 1 else float("nan"))


# ============================================================================ GPU part
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--init", required=True, help="trained token-channel checkpoint (TokInterface)")
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
    ap.add_argument("--max_tokens", type=int, default=100000)
    ap.add_argument("--lru", type=int, default=2)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enc_device", default="")
    ap.add_argument("--zs_inj", default="", help="P6 zero-shot: build an UNTRAINED native-unembedding residual pointer on "
                    "a plain --init (read layer --zs_read, W_U rows of the attended columns, injected after these layers "
                    "with a fixed gain --zs_beta on shift 0); '' = load --init as a trained token-channel checkpoint")
    ap.add_argument("--zs_read", type=int, default=19)
    ap.add_argument("--zs_beta", type=float, default=4.0)
    ap.add_argument("--zs_topk", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    from d22_c3_format import REFUSE
    from fit_d13 import D13Rt, build, select_items
    from register_d13 import doc_len_filter, excluded_docs
    from register_doc import SYSTEM_ANSWER, _get, log
    from register_state import _stop_ids
    from xenc.backbone import decode, text_ids
    from xenc.doc_tasks import exact_match, extract_answer, f1_score
    from xenc.e3_retrieval import answer_content_rows
    from xenc.state_skill import doc_prefix_text
    t0 = time.time()
    reader, enc, tok, itf, lmap = build(a, init=a.init)
    if a.zs_inj:                                    # P6 zero-shot native unembedding pointer (nothing trained)
        from xenc.p3_tok import parse_tok, upgrade_tok
        from xenc.p4_ptr import vocab_maps
        cfg = parse_tok(str(a.zs_read), 0, "avg", "0", a.zs_topk, 1, 1.0, "unemb", "beta", 0, a.zs_inj, 0.0)
        itf = upgrade_tok(itf, enc, cfg, reader.get_output_embeddings().weight)
        itf.set_tok_maps(*vocab_maps(tok, reader.get_output_embeddings().weight.shape[0])[:3])
        with torch.no_grad():
            for p_ in itf.TOK["beta"].values():
                p_.fill_(a.zs_beta)
        itf.tok_log = True
        log(f"P6 zero-shot pointer: read L{a.zs_read} top-{a.zs_topk} (renormalized) -> W_U rows -> residual after "
            f"{itf.tok_inj_layers}, gain {a.zs_beta} | feature scale {float(itf.TOK['proj']['scale']):.4f}")
    assert hasattr(itf, "tok_layers"), f"{a.init} is not a token-channel checkpoint"
    li_tok = max(itf.tok_layers)
    ck_step = (torch.load(a.init, map_location="cpu", weights_only=False).get("extra") or {}).get("step")
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
    vocab = tok.get_vocab()
    S_body = len(tok(doc_prefix_text(tok, a.reader, "", SYSTEM_ANSWER), add_special_tokens=False)["input_ids"])
    log(f"P3 diag: init {a.init} (saved at step {ck_step}) | tok cfg {itf.config()['tok']} | channel layer {li_tok} | candidate items {len(val)} "
        f"| GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    R = rt.R
    stops = _stop_ids(tok)
    state = {"cols": None, "cap": None}

    def override(li, p):
        if li != li_tok:
            return p
        if state["cap"] is not None:                       # capture (teacher-forced pass)
            state["cap"].append(p[..., : state["N"]].detach())
            return p
        cols = state["cols"]
        if not cols:
            return p
        q = torch.zeros_like(p)
        q[..., -1, cols] = 1.0 / len(cols)                  # decode rows: T == 1
        return q

    @torch.no_grad()
    def gen(did, it, mode, it_ctx):
        rows = rt.rows(did, it)
        op, n_op, _ = rt.operator(did)
        itf.tok_on = mode != "off"
        itf.tok_p_override = override if mode == "oracle" else None
        state["cols"], state["cap"] = None, None
        R.load(op, n_op)
        R.mode = "student"
        new, oracle_steps = [], 0
        try:
            out = reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=True, logits_to_keep=1)
            for _ in range(a.budget):
                nxt = int(out.logits[0, -1].argmax())
                new.append(nxt)
                txt = tok.decode(new, skip_special_tokens=True)
                if nxt in stops or "</answer>" in txt[-14:]:
                    break
                state["cols"] = None
                if mode == "oracle" and "<answer>" in txt:
                    L = len(txt.split("<answer>", 1)[1])
                    state["cols"] = oracle_columns(it_ctx["ends"], it_ctx["spans"], L, n_op)
                    oracle_steps += bool(state["cols"])
                out = reader(input_ids=torch.tensor([[nxt]], device=rt.dev), past_key_values=out.past_key_values,
                             use_cache=True, logits_to_keep=1)
        finally:
            R.mode, R.past = "off", {}
            itf.tok_on, itf.tok_p_override, state["cols"] = True, None, None
        txt = decode(tok, a.reader, new)
        if "</answer>" in txt:
            txt = txt.split("</answer>", 1)[0] + "</answer>"
        return txt, oracle_steps

    @torch.no_grad()
    def teacher_forced(did, it, ans, it_ctx):
        target = f"<answer>{ans}</answer>"
        tt, loc = answer_content_rows(tok, target)
        assert tt == text_ids(tok, target), "target tokenization mismatch"
        m = len(tt)
        rows = rt.rows(did, it, tt)
        op, n_op, _ = rt.operator(did)
        R.load(op, n_op)
        R.mode = "student"
        state["cap"], state["N"] = [], n_op
        itf.tok_p_override = override
        try:
            reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=False, logits_to_keep=1)
        finally:
            R.mode, R.past = "off", {}
            itf.tok_p_override = None
        P = state["cap"][0]                                 # [Hkv,G,T,N]
        state["cap"] = None
        pw = P[..., -(m + 1):, :].float().mean((0, 1))     # head-averaged, (m+1)-window rows [m+1, N]
        ids = op["ids"]
        gm = torch.zeros(n_op, dtype=torch.bool)
        for s, e in it_ctx["spans"]:
            for c in range(col_covering(it_ctx["ends"], s), min(col_covering(it_ctx["ends"], e - 1) + 1, n_op)):
                gm[c] = True
        gpos = gm.nonzero().flatten().tolist()
        res = []
        for j_, i in enumerate(loc):                        # window row i predicts content token tt[i]
            gt_s = tok.convert_ids_to_tokens(int(tt[i]))
            var = space_variants(gt_s, vocab)
            pr = pw[i]
            r = {"i": i, "first": j_ == 0}
            for k in (1, 4, 16):
                top = pr.topk(min(k, n_op)).indices
                r[f"tok@{k}"] = bool(any(tok_match(tok.convert_ids_to_tokens(int(ids[c])), gt_s, j_ == 0)
                                         for c in top.tolist()))
                r[f"pos@{k}"] = bool(gpos) and bool(any(min(abs(c - g) for g in gpos) <= 3 for c in top.tolist()))
            r["gold_mass"] = float(pr[gm.to(pr.device)].sum()) if gpos else 0.0
            r["tok_mass"] = float(pr[torch.tensor([int(x) in var for x in ids.tolist()], device=pr.device)].sum())
            res.append(r)
        return res

    rows_out = []
    n_skip = Counter()
    for k, it in enumerate(val):
        if a.max_items and len(rows_out) >= a.max_items:
            break
        did = _get(it, "doc_id")
        doc = docs[did]
        ptxt = doc_prefix_text(tok, a.reader, doc, SYSTEM_ANSWER)
        enc_ = tok(ptxt, add_special_tokens=False, return_offsets_mapping=True)
        full = list(enc_["input_ids"])
        if full != rt.pre(did):                             # truncated or differently tokenized: alignment unsafe
            n_skip["prefix_mismatch"] += 1
            continue
        N = len(full) - a.tail
        ends = [e for _, e in enc_["offset_mapping"]]
        ans, spans = occurrences(ptxt, doc, _get(it, "answers"), ends[N - 1])
        if not spans:
            n_skip["gold_not_in_operator"] += 1
            continue
        d0 = ptxt.rfind(doc)
        ctx = {"ends": ends, "spans": spans}
        rec = {"key": str(_get(it, "key")), "task": did.split("_")[0], "n_op": N, "answer_used": ans,
               "n_occ": len(spans)}
        try:
            for mode in ("on", "off", "oracle"):
                g, n_or = gen(did, it, mode, ctx)
                pred = extract_answer(g)
                rec[mode] = {"gen": g, "f1": f1_score(pred, list(_get(it, "answers"))),
                             "em": exact_match(pred, list(_get(it, "answers")))}
                if mode == "oracle":
                    rec[mode]["oracle_steps"] = n_or
            rec["tf"] = teacher_forced(did, it, ans, ctx)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            rt._ops.clear()
            n_skip["oom"] += 1
            log(f"  OOM on {rec['key']} ({N} columns); item skipped")
            continue
        pred = extract_answer(rec["on"]["gen"])
        gold_doc = [(s - d0, e - d0) for s, e in spans]
        rec["anatomy"] = anatomy(rec["on"]["f1"], rec["on"]["em"], pred, list(_get(it, "answers")), doc, gold_doc,
                                 lambda s_: bool(REFUSE.match(s_)))
        rows_out.append(rec)
        if len(rows_out) % 20 == 0:
            log(f"  {len(rows_out)} items ({time.time() - t0:.0f}s) | skips {dict(n_skip)}")
    with open(os.path.join(a.out, "items.jsonl"), "w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    summ = summarize(rows_out)
    summ.update({"zs": {"inj": a.zs_inj, "beta": a.zs_beta, "topk": a.zs_topk, "read": a.zs_read} if a.zs_inj else None,
                 "tok_ratio_last": dict(itf.tok_stats)})
    summ.update({"init": a.init, "init_step": ck_step, "n": len(rows_out), "skips": dict(n_skip), "tok_cfg": itf.config()["tok"],
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "body_start": S_body})
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    md = to_markdown(summ)
    open(os.path.join(a.out, "summary.md"), "w").write(md)
    log(md)
    log(f"done: {len(rows_out)} items, skips {dict(n_skip)} ({time.time() - t0:.0f}s)")
    return 0


def summarize(rows: list) -> dict:
    out = {"by_task": {}}
    groups = {"all": rows}
    for r in rows:
        groups.setdefault(r["task"], []).append(r)
    for g, rs in groups.items():
        if not rs:
            continue
        d = {"n": len(rs)}
        for mode in ("on", "off", "oracle"):
            d[f"{mode}_f1"] = stat([r[mode]["f1"] for r in rs])
            d[f"{mode}_em"] = stat([r[mode]["em"] for r in rs])
        d["oracle-on_f1"] = stat([r["oracle"]["f1"] - r["on"]["f1"] for r in rs])
        d["on-off_f1"] = stat([r["on"]["f1"] - r["off"]["f1"] for r in rs])
        d["oracle_steps_mean"] = sum(r["oracle"]["oracle_steps"] for r in rs) / len(rs)
        first = [r["tf"][0] for r in rs if r["tf"]]
        allr = [x for r in rs for x in r["tf"]]
        for k in (1, 4, 16):
            d[f"first_tok@{k}"] = stat([float(x[f"tok@{k}"]) for x in first])[0]
            d[f"first_pos@{k}"] = stat([float(x[f"pos@{k}"]) for x in first])[0]
            d[f"all_tok@{k}"] = stat([float(x[f"tok@{k}"]) for x in allr])[0]
            d[f"all_pos@{k}"] = stat([float(x[f"pos@{k}"]) for x in allr])[0]
        d["first_gold_mass"] = stat([x["gold_mass"] for x in first])[0]
        d["first_tok_mass"] = stat([x["tok_mass"] for x in first])[0]
        d["anatomy"] = dict(Counter(r["anatomy"] for r in rs))
        wrong = [r for r in rs if r["anatomy"] != "correct"]
        d["wrong_first_tok@4"] = stat([float(r["tf"][0]["tok@4"]) for r in wrong if r["tf"]])[0]
        d["correct_first_tok@4"] = stat([float(r["tf"][0]["tok@4"]) for r in rs if r["anatomy"] == "correct" and r["tf"]])[0]
        out["by_task"][g] = d
    return out


def to_markdown(s: dict) -> str:
    f = lambda t: f"{t[0]:.3f} ± {t[1]:.3f}" if isinstance(t, (list, tuple)) else f"{t:.3f}"   # noqa: E731
    L = [f"init `{s.get('init')}` | n {s.get('n')} | skips {s.get('skips')} | {s.get('gpu')}", "",
         "**(A) oracle addressing (greedy F1 / EM, same items)**", "",
         "| set | n | on F1 | off F1 | oracle F1 | oracle − on | on − off | on EM | oracle EM | oracle steps/item |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for g, d in s["by_task"].items():
        L.append(f"| {g} | {d['n']} | {d['on_f1'][0]:.3f} | {d['off_f1'][0]:.3f} | {d['oracle_f1'][0]:.3f} | "
                 f"{f(d['oracle-on_f1'])} | {f(d['on-off_f1'])} | {d['on_em'][0]:.3f} | {d['oracle_em'][0]:.3f} | "
                 f"{d['oracle_steps_mean']:.1f} |")
    L += ["", "**(B) teacher-forced addressing at the channel layer (head-avg p): gold next token among top-k columns "
          "(tok@k, space variants) / top-k column within ±3 of the gold span (pos@k)**", "",
          "| set | first tok@1/4/16 | first pos@1/4/16 | all-rows tok@1/4/16 | first-row gold-span mass | first-row gold-token mass |",
          "|---|---|---|---|---|---|"]
    for g, d in s["by_task"].items():
        L.append(f"| {g} | {d['first_tok@1']:.2f}/{d['first_tok@4']:.2f}/{d['first_tok@16']:.2f} | "
                 f"{d['first_pos@1']:.2f}/{d['first_pos@4']:.2f}/{d['first_pos@16']:.2f} | "
                 f"{d['all_tok@1']:.2f}/{d['all_tok@4']:.2f}/{d['all_tok@16']:.2f} | {d['first_gold_mass']:.3f} | "
                 f"{d['first_tok_mass']:.3f} |")
    L += ["", "**(B) anatomy of greedy answers (channel on)**", "",
          "| set | correct | contains gold | same paragraph | other paragraph | not in doc | normalized-only | no answer | "
          "first tok@4: wrong / correct |", "|---|---|---|---|---|---|---|---|---|"]
    for g, d in s["by_task"].items():
        an = d["anatomy"]
        L.append(f"| {g} | {an.get('correct', 0)} | {an.get('contains_gold', 0)} | {an.get('same_paragraph', 0)} | "
                 f"{an.get('other_paragraph', 0)} | {an.get('not_in_doc', 0)} | {an.get('in_doc_normalized_only', 0)} | "
                 f"{an.get('no_answer', 0)} | {d['wrong_first_tok@4']:.2f} / {d['correct_first_tok@4']:.2f} |")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
