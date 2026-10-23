"""B1: benchmark-fit analysis (CPU only, existing per-item outputs).

Builds one per-item feature table over the D22 benchmarks (registered kv, D22_SUMMARY's kv_reg = es<=200 rule) and
correlates item features with the paired differences kv-text and kv-wrong_doc.

Features (all computed from the document, the question, the options and the gold; no model):
  doclen        Qwen3.5 tokens (docs.json)
  q_ovl         IDF-weighted share of the question's content words present in the best 40-word window of the doc
  q_pos         relative position of that best window
  lex_ok        MCQ: a sliding-window lexical baseline (question+option overlap; min for NOT/EXCEPT questions) picks gold
  opt_in_doc    MCQ: share of the correct option's content words that occur in the doc
  gold_cnt      span: occurrences of the normalized gold answer in the normalized doc (0 = not extractive)
  ev_n, ev_pos, ev_spread   gold evidence count / first relative position / max-min spread where the benchmark
                annotates it (LongHealth answer_location, ContractNLI evidence spans, MuSiQue supporting paragraphs,
                LooGLE evidence sentences, Shakespeare quoted line)
  hops          MuSiQue / LooGLE longdep / CLIPPER multi
  cb, ct        9B closed_book / closed_tail score (prior)
  t4b           4B encoder's own text score (D22_C1)

    python scripts/b1_benchmark_fit.py --items outputs/B1_items.jsonl --out outputs/B1_tables.md
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from xenc.doc_tasks import load_doc_split, load_docs, normalize_answer  # noqa: E402

# name -> (eval json, split dir, fmt, C1 file, in-domain for the base D13_4b_reg_w10)
RUNS = {
    "LongHealth": ("outputs/D20_4b_lh_f0/eval_test.json", "data/longhealth_split", "mcq5", "longhealth", "yes (f2-4, same template)"),
    "QuALITY": ("outputs/D20_4b_quality/eval_test_full.json", "data/quality_split", "mcq4", "quality", "yes (150 train docs)"),
    "ContractNLI": ("outputs/D22_contractnli/eval_test.json", "data/contractnli_split", "nli3", "contractnli", "no"),
    "Shakespeare-CF": ("outputs/D22_shakespeare_cf/eval_test.json", "data/shakespeare_cf_split", "mcq4-5", "shakespeare-cf", "no"),
    "Qasper": ("outputs/D22_qasper/eval_test.json", "data/qasper_split", "span+yn", "qasper", "no"),
    "MuSiQue": ("outputs/D22_musique/eval_test.json", "data/musique_split", "span", "musique", "yes (150 train docs)"),
    "LastingBench": ("outputs/D22_lastingbench/eval_test.json", "data/lastingbench_split", "span", "lastingbench", "yes (150 train docs)"),
    "LooGLE": ("outputs/D22_loogle/eval_test.json", "data/loogle_split", "span+cloze", "loogle", "no"),
    "LooGLE-long": ("outputs/D22_loogle_long/eval_test.json", "data/loogle_split", "span+cloze", "loogle-long", "no"),
    "CLIPPER": ("outputs/D22_clipper/eval_test.json", "data/clipper_split", "tf", "clipper", "no"),
}
STOP = set("""a an the of to in on at for by with from and or but not no nor is are was were be been being do does did has
have had this that these those it its as which what who whom whose when where why how there their they them he she his her
him i you we our your my me us any all each other than then so such into about over under after before during between
following based according regarding statement statements true false story document text agreement patient mr mrs ms
one can could would should will may might shall must also only very more most""".split())
NEG = re.compile(r"\b(NOT|EXCEPT|never|incorrect|false|not accurate|not part|not mentioned)\b")
_W = re.compile(r"[a-z0-9]+")


def words(s):
    return _W.findall((s or "").lower())


def content(s):
    return [w for w in words(s) if w not in STOP and len(w) > 1]


class DocIndex:
    """40-word windows (stride 20) with window-level IDF inside the document."""

    def __init__(self, text, W=40, S=20):
        self.text = text
        toks = [(m.group(0), m.start()) for m in _W.finditer(text.lower())]
        self.n_chars = max(len(text), 1)
        self.wins, self.pos = [], []
        for i in range(0, max(len(toks) - W, 0) + 1, S):
            seg = toks[i:i + W]
            self.wins.append(set(w for w, _ in seg))
            self.pos.append(seg[0][1] / self.n_chars if seg else 0.0)
        df = Counter(w for s in self.wins for w in s)
        N = len(self.wins)
        self.idf = {w: math.log((N + 1) / (c + 0.5)) for w, c in df.items()}
        self.vocab = set(df)
        self.norm = " " + normalize_answer(text) + " "
        self.jd = " ".join(w for w, _ in toks)

    def best(self, qw):
        qw = [w for w in set(qw)]
        if not qw:
            return 0.0, 0.0
        wt = {w: self.idf.get(w, math.log(len(self.wins) + 1)) for w in qw}
        tot = sum(wt.values())
        best, bp = -1.0, 0.0
        for s, p in zip(self.wins, self.pos):
            v = sum(wt[w] for w in qw if w in s)
            if v > best:
                best, bp = v, p
        return best / tot if tot else 0.0, bp


def mcq_lex(di, it):
    """Sliding-window lexical baseline (MCTest-style): score(option) = best window overlap of question+option words."""
    q = content(it.question)
    neg = bool(NEG.search(it.question))
    sc = []
    for c in it.choices:
        ow = content(c["text"])
        s, _ = di.best(q + ow) if ow else (0.0, 0.0)
        pres = (sum(w in di.vocab for w in set(ow)) / len(set(ow))) if ow else 0.0
        sc.append((s, pres))
    key = [p for _, p in sc] if it.task == "longhealth" else [s for s, _ in sc]   # LH options are entities: presence
    tgt = min(key) if neg else max(key)
    win = [c["label"] for c, k in zip(it.choices, key) if abs(k - tgt) < 1e-9]
    gold = set(it.meta.get("accept_labels") or [it.correct_label])
    lex = sum(w in gold for w in win) / len(win)
    gi = [c["label"] for c in it.choices].index(it.correct_label)
    return lex, sc[gi][1], neg


def find_all(hay, needle):
    out, i = [], hay.find(needle)
    while i >= 0 and needle:
        out.append(i)
        i = hay.find(needle, i + 1)
    return out


def evidence(name, it, doc, di):
    """(ev_n, ev_pos, ev_spread, hops) from the benchmark's own evidence annotation, None where absent."""
    m, L = it.meta or {}, max(len(doc), 1)
    if name == "LongHealth":
        parts = doc.split("\n\n---\n\n")
        offs, o = [], 0
        for p in parts:
            offs.append(o)
            o += len(p) + 7
        P = []
        for k, v in (m.get("answer_location") or {}).items():
            j = int(k.split("_")[-1])
            if j < len(parts):
                P += [(offs[j] + s * len(parts[j])) / L for s in v["start"]]
        n_letters = len(m.get("answer_location") or {})
        return (len(P), min(P) if P else None, (max(P) - min(P)) if P else None, n_letters)
    if name == "ContractNLI":
        sp = m.get("evidence_char_spans") or []
        P = [s / L for s, _ in sp]
        return (len(P), min(P) if P else None, (max(P) - min(P)) if P else None, None)
    if name == "MuSiQue":
        ps = m.get("para_spans") or []
        P = [ps[i][0] / L for i in m.get("supporting_idx") or [] if i < len(ps)]
        return (len(P), min(P) if P else None, (max(P) - min(P)) if P else None, len(m.get("supporting_idx") or []))
    if name.startswith("LooGLE"):
        ev = m.get("evidence")
        ev = [ev] if isinstance(ev, str) else (ev or [])
        P = []
        for e in ev:
            ws = words(e)[:8]
            if ws:
                h = find_all(di.jd, " ".join(ws))
                if h:
                    P.append(h[0] / max(len(di.jd), 1))
        hop = 2 if m.get("source") == "longdep_qa" else 1
        return (len(ev) if ev else None, min(P) if P else None, (max(P) - min(P)) if len(P) > 1 else (0.0 if P else None), hop)
    if name == "Shakespeare-CF":
        q = re.search(r'beginning: "(.*?)"', it.question)
        if q:
            ws = words(q.group(1))[:6]
            jd = di.jd
            h = find_all(jd, " ".join(ws))
            return (len(h), h[0] / max(len(jd), 1) if h else None, 0.0 if h else None, 1)
        return (None, None, None, None)
    if name == "CLIPPER":
        return (None, None, None, 2 if m.get("claim_type") == "multiple" else 1)
    return (None, None, None, None)


def load_rows(name, B=200):
    path, sd, fmt, c1, indom = RUNS[name]
    E = json.load(open(path))
    meta = {it.key: it for sp in ("train", "val", "test") if (Path(sd) / sp / "items.json").exists()
            for it in load_doc_split(sd, sp)}
    es = E.get("es_choice") or {}
    eb = sorted(int(b) for b in (E["args"].get("eval_budgets") or E["args"]["budgets"]).split(","))
    Bk = str(max(b for b in eb if b <= B))
    C1, C1g = {}, {}
    p1 = Path(f"outputs/D22_C1/{c1}.jsonl")
    if p1.exists():
        for line in open(p1):
            r = json.loads(line)
            C1[r["key"]] = r["score"]
            C1g[r["key"]] = r["gen"]
    rows = []
    for r in E["items"]:
        def pick(c, d, k="score"):
            b = es[d][Bk] if (es and d in es) else int(Bk)
            v = r.get(f"{c}@{b}")
            return None if v is None else v[k]
        rows.append({"bench": name, "key": r["key"], "doc_id": r["doc_id"], "item": meta[r["key"]],
                     "kv": pick("kv", r["doc_id"]), "wrong": pick("wrong_doc", r["wrong"]) if r.get("wrong") else None,
                     "text": r["text"]["score"], "ct": (r.get("closed_tail") or {}).get("score"),
                     "cb": (r.get("closed_book") or {}).get("score"), "kv0": r[f"kv@{eb[0]}"]["score"],
                     "t4b": C1.get(r["key"]), "g_kv": pick("kv", r["doc_id"], "gen"), "g_text": r["text"]["gen"],
                     "g_t4b": C1g.get(r["key"]), "g_ct": (r.get("closed_tail") or {}).get("gen"),
                     "g_wrong": pick("wrong_doc", r["wrong"], "gen") if r.get("wrong") else None})
    return rows, sd, fmt, indom


def featurize(rows, sd, name):
    docs = load_docs(sd)
    ntok = {d: int(m["n_tokens"]) for d, m in json.load(open(Path(sd) / "docs.json")).items()}
    cache = {}
    for r in rows:
        it, d = r["item"], r["doc_id"]
        if d not in cache:
            cache[d] = DocIndex(docs[d])
        di, doc = cache[d], docs[d]
        r["doclen"] = ntok[d]
        r["q_ovl"], r["q_pos"] = di.best(content(it.question))
        r["lex_ok"] = r["opt_in_doc"] = r["neg"] = r["gold_cnt"] = None
        if it.choices and name not in ("CLIPPER", "ContractNLI"):
            r["lex_ok"], r["opt_in_doc"], r["neg"] = mcq_lex(di, it)
            r["opt_len"] = float(np.mean([len(words(c["text"])) for c in it.choices]))
        if not it.choices:
            g = [normalize_answer(a) for a in it.answers if normalize_answer(a)]
            r["gold_cnt"] = max((len(find_all(di.norm, " " + a + " ")) for a in g), default=0)
            r["yesno"] = bool(g) and g[0] in ("yes", "no")
        r["ev_n"], r["ev_pos"], r["ev_spread"], r["hops"] = evidence(name, it, doc, di)
        m = it.meta or {}
        r["sub"] = (m.get("atype") or m.get("qtype") or m.get("kind") or m.get("claim_type") or m.get("nli_label")
                    or ("cf" if m.get("counterfactual") else ("orig" if "counterfactual" in m else None))
                    or ("hard" if m.get("hard") else ("easy" if "hard" in m else None)))
    return rows


# ---------------------------------------------------------------------------- statistics
def paired(rows, a, b):
    """Mean of a-b over items with naive paired se and doc-clustered se."""
    pr = [(r[a] - r[b], r["doc_id"]) for r in rows if r.get(a) is not None and r.get(b) is not None]
    if len(pr) < 2:
        return None
    d = np.array([x for x, _ in pr])
    mu, n = d.mean(), len(d)
    se = d.std(ddof=1) / n ** 0.5
    g = defaultdict(float)
    for (x, doc) in pr:
        g[doc] += x - mu
    G = len(g)
    cse = (sum(v * v for v in g.values()) / n ** 2 * G / max(G - 1, 1)) ** 0.5
    return {"gap": float(mu), "se": float(se), "cse": float(cse), "n": n, "G": G}


def fmt_gap(p):
    return "—" if not p else f"{p['gap']:+.3f}±{p['se']:.3f} ({p['cse']:.3f})"


def diff_of_gaps(r1, r2, a, b):
    """Interaction: gap(r1) - gap(r2) with independent-groups doc-clustered se."""
    p, q = paired(r1, a, b), paired(r2, a, b)
    if not p or not q:
        return None
    return {"gap": p["gap"] - q["gap"], "se": (p["se"] ** 2 + q["se"] ** 2) ** .5, "cse": (p["cse"] ** 2 + q["cse"] ** 2) ** .5}


def ols_cluster(X, y, groups):
    """OLS with cluster-robust (CR1) se. X includes the intercept / fixed-effect columns."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    gi = defaultdict(list)
    for i, g in enumerate(groups):
        gi[g].append(i)
    for idx in gi.values():
        s = X[idx].T @ e[idx]
        meat += np.outer(s, s)
    G, n, k = len(gi), len(y), X.shape[1]
    c = G / max(G - 1, 1) * (n - 1) / max(n - k, 1)
    V = c * XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.clip(np.diag(V), 0, None))


def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="outputs/B1_items.jsonl")
    ap.add_argument("--out", default="outputs/B1_tables.md")
    a = ap.parse_args(argv)
    R, info = {}, {}
    for name in RUNS:
        rows, sd, fmt, indom = load_rows(name)
        R[name] = featurize(rows, sd, name)
        info[name] = (fmt, indom)
        print(name, len(rows), file=sys.stderr)
    with open(a.items, "w") as f:
        for name, rows in R.items():
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if k != "item"}) + "\n")

    L = ["# B1 tables (generated by scripts/b1_benchmark_fit.py)", "",
         "Gaps: mean ± naive paired se (doc-clustered se). kv = registered kv (D22 es≤200 rule).", ""]
    # ---- 1. profile
    L += ["## 1. Benchmark profile", "",
          "| benchmark | format | in-domain base | n | docs | doclen med | q_ovl med | lex-baseline acc | gold-in-doc (span) | ev_n med | ev spread med | closed_book | closed_tail | text 4B | text 9B | kv | wrong | kv−text | kv−wrong | text−ct |",
          "|" + "---|" * 20]
    for name, rows in R.items():
        fmt, indom = info[name]
        med = lambda k: (f"{np.median([r[k] for r in rows if r.get(k) is not None]):.2f}"   # noqa: E731
                         if any(r.get(k) is not None for r in rows) else "—")
        gold_in = mean([float(r["gold_cnt"] > 0) for r in rows if r.get("gold_cnt") is not None and not r.get("yesno")])
        L.append(f"| {name} | {fmt} | {indom} | {len(rows)} | {len({r['doc_id'] for r in rows})} | "
                 f"{int(np.median([r['doclen'] for r in rows]))} | {med('q_ovl')} | {f3(mean([r['lex_ok'] for r in rows]))} | "
                 f"{f3(gold_in)} | {med('ev_n')} | {med('ev_spread')} | {f3(mean([r['cb'] for r in rows]))} | "
                 f"{f3(mean([r['ct'] for r in rows]))} | {f3(mean([r['t4b'] for r in rows]))} | {f3(mean([r['text'] for r in rows]))} | "
                 f"{f3(mean([r['kv'] for r in rows]))} | {f3(mean([r['wrong'] for r in rows]))} | "
                 f"{fmt_gap(paired(rows, 'kv', 'text'))} | {fmt_gap(paired(rows, 'kv', 'wrong'))} | {fmt_gap(paired(rows, 'text', 'ct'))} |")

    # ---- 2. encoder-information decomposition (C1 at item level)
    L += ["", "## 2. Item-level C1: does kv fail where the 4B encoder itself fails?", "",
          "Split by the 4B reading the document as text (score ≥ .5 = right). kv−text within each split.", "",
          "| benchmark | n 4B-right | kv | text9B | kv−text (4B right) | n 4B-wrong | kv | text9B | kv−text (4B wrong) | share of total kv−text from 4B-wrong items |",
          "|" + "---|" * 10]
    for name, rows in R.items():
        rr = [r for r in rows if r["t4b"] is not None]
        if not rr:
            L.append(f"| {name} | no C1 | | | | | | | | |")
            continue
        A = [r for r in rr if r["t4b"] >= .5]
        Bw = [r for r in rr if r["t4b"] < .5]
        tot = sum(r["kv"] - r["text"] for r in rr)
        sh = (sum(r["kv"] - r["text"] for r in Bw) / tot) if abs(tot) > 1e-9 else float("nan")
        L.append(f"| {name} | {len(A)} | {f3(mean([r['kv'] for r in A]))} | {f3(mean([r['text'] for r in A]))} | "
                 f"{fmt_gap(paired(A, 'kv', 'text'))} | {len(Bw)} | {f3(mean([r['kv'] for r in Bw]))} | "
                 f"{f3(mean([r['text'] for r in Bw]))} | {fmt_gap(paired(Bw, 'kv', 'text'))} | {sh:.2f} |")

    # ---- 3. feature splits per benchmark
    def split_table(title, feat, cut):
        out = ["", f"### {title}", "",
               "| benchmark | low: n | low kv−text | low kv−wrong | high: n | high kv−text | high kv−wrong | Δ(kv−text) high−low | Δ(kv−wrong) high−low |",
               "|" + "---|" * 9]
        for name, rows in R.items():
            rr = [r for r in rows if r.get(feat) is not None]
            if len(rr) < 20:
                continue
            c = cut(rr) if callable(cut) else cut
            lo = [r for r in rr if r[feat] <= c]
            hi = [r for r in rr if r[feat] > c]
            if len(lo) < 8 or len(hi) < 8:
                continue
            out.append(f"| {name} (cut {c:.2f}) | {len(lo)} | {fmt_gap(paired(lo, 'kv', 'text'))} | {fmt_gap(paired(lo, 'kv', 'wrong'))} | "
                       f"{len(hi)} | {fmt_gap(paired(hi, 'kv', 'text'))} | {fmt_gap(paired(hi, 'kv', 'wrong'))} | "
                       f"{fmt_gap(diff_of_gaps(hi, lo, 'kv', 'text'))} | {fmt_gap(diff_of_gaps(hi, lo, 'kv', 'wrong'))} |")
        return out
    medcut = lambda rr_, k: float(np.median([r[k] for r in rr_]))   # noqa: E731
    L += ["", "## 3. Feature splits (within benchmark; low ≤ cut < high)"]
    L += split_table("q_ovl: question–evidence lexical overlap (median split)", "q_ovl", lambda rr: medcut(rr, "q_ovl"))
    L += split_table("q_pos: position of the best-matching window (median split)", "q_pos", lambda rr: medcut(rr, "q_pos"))
    L += split_table("ev_pos: first annotated evidence position (median split)", "ev_pos", lambda rr: medcut(rr, "ev_pos"))
    L += split_table("ev_n: number of annotated evidence spans (1 vs >1)", "ev_n", 1)
    L += split_table("doclen (median split)", "doclen", lambda rr: medcut(rr, "doclen"))
    L += split_table("closed_tail prior right (0 vs 1; for spans F1 cut .5)", "ct", .5)
    L += split_table("closed_book prior right", "cb", .5)
    L += split_table("lexical baseline right (MCQ)", "lex_ok", .5)
    L += split_table("gold answer occurrences in doc (span; ≤1 vs >1)", "gold_cnt", 1)
    L += split_table("hops (1 vs >1; LongHealth = #letters holding evidence)", "hops", 1)
    L += split_table("MCQ mean option length in words (median split)", "opt_len", lambda rr: medcut(rr, "opt_len"))
    L += split_table("NOT/EXCEPT question (MCQ)", "neg", .5)

    # ---- 4. subsets
    L += ["", "## 4. Native subsets", "", "| benchmark: subset | n | text | kv | wrong | kv−text | kv−wrong |", "|---|---|---|---|---|---|---|"]
    for name, rows in R.items():
        subs = Counter(r["sub"] for r in rows if r.get("sub"))
        for s, c in sorted(subs.items()):
            if c < 15:
                continue
            rr = [r for r in rows if r["sub"] == s]
            L.append(f"| {name}: {s} | {c} | {f3(mean([r['text'] for r in rr]))} | {f3(mean([r['kv'] for r in rr]))} | "
                     f"{f3(mean([r['wrong'] for r in rr]))} | {fmt_gap(paired(rr, 'kv', 'text'))} | {fmt_gap(paired(rr, 'kv', 'wrong'))} |")

    # ---- 5. pooled regression with benchmark fixed effects
    L += ["", "## 5. Pooled item-level regression (benchmark fixed effects, doc-clustered CR1 se)", "",
          "Outcome = kv−text (or kv−wrong) per item. Continuous features z-scored within benchmark; binary as 0/1.", ""]
    feats = [("q_ovl", "z"), ("q_pos", "z"), ("log_doclen", "z"), ("ct", "b"), ("t4b", "b"), ("text", "b")]
    for outcome in (("kv", "text"), ("kv", "wrong")):
        use = [f for f in feats if not (outcome[1] == "text" and f[0] == "text")]
        rows_all = []
        for name, rows in R.items():
            for r in rows:
                r["log_doclen"] = math.log(r["doclen"])
                if all(r.get(f) is not None for f, _ in use) and r.get(outcome[1]) is not None and r.get("kv") is not None:
                    rows_all.append(r)
        benches = sorted({r["bench"] for r in rows_all})
        Z = {}
        for f, t in use:
            for b in benches:
                v = np.array([r[f] for r in rows_all if r["bench"] == b], float)
                Z[(f, b)] = (v.mean(), v.std() or 1.0)
        X, y, g = [], [], []
        for r in rows_all:
            x = [1.0 if r["bench"] == b else 0.0 for b in benches]
            for f, t in use:
                m_, s_ = Z[(f, r["bench"])]
                x.append((r[f] - m_) / s_ if t == "z" else float(r[f] >= .5))
            X.append(x)
            y.append(r["kv"] - r[outcome[1]])
            g.append(r["bench"] + "/" + r["doc_id"])
        beta, se = ols_cluster(X, y, g)
        L += [f"**kv−{outcome[1]}** (n={len(y)}, {len(set(g))} doc clusters, benchmarks: {', '.join(benches)})", "",
              "| feature | coef | cluster se | t |", "|---|---|---|---|"]
        for j, (f, t) in enumerate(use):
            k = len(benches) + j
            L.append(f"| {f} ({'per sd' if t == 'z' else '1 vs 0'}) | {beta[k]:+.3f} | {se[k]:.3f} | {beta[k] / se[k]:+.1f} |")
        L.append("")
    # ---- 6. additive decomposition kv - text9B = (kv - text4B) + (text4B - text9B)
    L += ["", "## 6. Gap decomposition: kv − text(9B) = [kv − text(4B encoder)] (read loss) + [text(4B) − text(9B)] (encoder deficit)", "",
          "| benchmark | n | text 4B | text 9B | kv | kv−text9B | **read loss** kv−text4B | **encoder deficit** text4B−text9B |",
          "|" + "---|" * 8]
    for name, rows in R.items():
        rr = [r for r in rows if r["t4b"] is not None]
        if not rr:
            L.append(f"| {name} | — | (no 4B text run) | {f3(mean([r['text'] for r in rows]))} | {f3(mean([r['kv'] for r in rows]))} | "
                     f"{fmt_gap(paired(rows, 'kv', 'text'))} | — | — |")
            continue
        L.append(f"| {name} | {len(rr)} | {f3(mean([r['t4b'] for r in rr]))} | {f3(mean([r['text'] for r in rr]))} | "
                 f"{f3(mean([r['kv'] for r in rr]))} | {fmt_gap(paired(rr, 'kv', 'text'))} | {fmt_gap(paired(rr, 'kv', 't4b'))} | "
                 f"{fmt_gap(paired(rr, 't4b', 'text'))} |")

    # ---- 7. MCQ answer agreement: whose reading does kv reproduce?
    from xenc.doc_tasks import extract_label
    L += ["", "## 7. MCQ letter agreement (whose answer does kv reproduce?)", "",
          "| benchmark | n | P(kv=4B) | P(kv=9B text) | P(4B=9B) | P(kv=closed_tail) | P(kv=wrong_doc) | items 4B≠9B: P(kv=4B) / P(kv=9B) (n) | both kv & 4B wrong: same letter (n) |",
          "|" + "---|" * 9]
    for name, rows in R.items():
        rr = [r for r in rows if r["item"].choices and r.get("g_t4b") is not None and r.get("g_kv") is not None]
        if not rr:
            continue
        lab = lambda r, k: extract_label(r[k] or "", "".join(c["label"] for c in r["item"].choices))   # noqa: E731
        ag = lambda k1, k2, xs: mean([float(lab(r, k1) == lab(r, k2)) for r in xs if r.get(k1) and r.get(k2)])   # noqa: E731
        dis = [r for r in rr if lab(r, "g_t4b") != lab(r, "g_text")]
        bw = [r for r in rr if r["kv"] < .5 and r["t4b"] < .5]
        L.append(f"| {name} | {len(rr)} | {f3(ag('g_kv', 'g_t4b', rr))} | {f3(ag('g_kv', 'g_text', rr))} | {f3(ag('g_t4b', 'g_text', rr))} | "
                 f"{f3(ag('g_kv', 'g_ct', rr))} | {f3(ag('g_kv', 'g_wrong', rr))} | {f3(ag('g_kv', 'g_t4b', dis))} / {f3(ag('g_kv', 'g_text', dis))} ({len(dis)}) | "
                 f"{f3(ag('g_kv', 'g_t4b', bw))} ({len(bw)}) |")
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
