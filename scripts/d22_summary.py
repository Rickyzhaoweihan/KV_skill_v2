"""D22: one format-aware table over every benchmark judged through per-document registration (register_d20 / _d22).

Per benchmark (an eval_<split>.json written by scripts/register_d20.py's eval phase) and per condition:
  closed_book, closed_tail, text (9B reads the document), kv_base (= kv@0, the base interface WITHOUT registration —
  a reference only), kv_reg (registered; per-document early stop on held-out SELF-question loss over budgets <= B,
  = register_d20's "es≤B"), wrong_doc (another document's REGISTERED operator + adapter, same early-stop rule).
Metric by answer format: MCQ / NLI / T-F -> accuracy; span -> token F1 (primary) and EM; numeric-gold items -> numeric
match (every number in the gold appears in the prediction, after removing thousands separators). Paired differences
(kv_reg − text, kv_reg − wrong_doc, kv_reg − kv_base, kv_reg − closed_tail) with se = sd/√n over items.

    python scripts/d22_summary.py --out outputs/D22_table.md [--B 200]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# name -> (eval json, split_dir, format, self-question file, subsets)
RUNS = {
    "LongHealth (f0)":  ("outputs/D20_4b_lh_f0/eval_test.json", "data/longhealth_split", "mcq", "outputs/D20_selfq/longhealth.jsonl"),
    "QuALITY":          ("outputs/D20_4b_quality/eval_test_full.json", "data/quality_split", "mcq", "outputs/D20_selfq/quality.jsonl"),
    "ContractNLI":      ("outputs/D22_contractnli/eval_test.json", "data/contractnli_split", "nli", "outputs/D20_selfq/contractnli.jsonl"),
    "Shakespeare-CF":   ("outputs/D22_shakespeare_cf/eval_test.json", "data/shakespeare_cf_split", "mcq", "outputs/D20_selfq/shakespeare_cf.jsonl"),
    "Qasper":           ("outputs/D22_qasper/eval_test.json", "data/qasper_split", "span", "outputs/D22_selfq/qasper.jsonl"),
    "MuSiQue":          ("outputs/D22_musique/eval_test.json", "data/musique_split", "span", "outputs/D22_selfq/musique.jsonl"),
    "LastingBench":     ("outputs/D22_lastingbench/eval_test.json", "data/lastingbench_split", "span", "outputs/D22_selfq/lastingbench.jsonl"),
    "LooGLE":           ("outputs/D22_loogle/eval_test.json", "data/loogle_split", "span", "outputs/D22_selfq/loogle.jsonl"),
    "LooGLE-long (≥48k)": ("outputs/D22_loogle_long/eval_test.json", "data/loogle_split", "span", "outputs/D22_selfq/loogle.jsonl"),
    "CLIPPER (T/F)":    ("outputs/D22_clipper/eval_test.json", "data/clipper_split", "mcq", "outputs/D22_selfq/clipper.jsonl"),
}
SUBSETS = {   # name -> {subset label: predicate(item meta)}
    "ContractNLI": {"clean (no near-dup train NDA)": lambda m: m.get("near_dup_train") is False},
    "Shakespeare-CF": {"speaker, conflict": lambda m: m.get("conflict") is True, "act (no conflict)": lambda m: m.get("kind") == "act"},
    "Qasper": {"extractive": lambda m: m.get("atype") == "extractive", "yes/no": lambda m: m.get("atype") == "boolean"},
    "LastingBench": {"counterfactual": lambda m: m.get("counterfactual") is True, "original": lambda m: m.get("counterfactual") is False},
    "LooGLE-long (≥48k)": {"span QA": lambda m: m.get("qtype") == "span", "cloze": lambda m: m.get("qtype") == "cloze"},
    "LooGLE": {"span QA": lambda m: m.get("qtype") == "span", "cloze": lambda m: m.get("qtype") == "cloze"},
}
LENGTH_BUCKETS = [(0, 8000), (8000, 24000), (24000, 48000), (48000, 64000), (64000, 96000), (96000, 10**9)]
CONDS = ("closed_book", "closed_tail", "text", "kv_base", "kv_reg", "wrong_doc")
_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def numbers(s: str) -> list[float]:
    return [float(x) for x in _NUM.findall((s or "").replace(",", ""))]


def num_match(item, gen: str) -> float:
    from xenc.doc_tasks import extract_answer
    g = numbers(item.answers[0])
    p = set(numbers(extract_answer(gen)))
    return float(bool(g) and all(any(abs(x - y) <= 1e-6 * max(1, abs(x)) for y in p) for x in g))


def is_numeric(item) -> bool:
    a = (item.answers or [""])[0].strip()
    if (item.meta or {}).get("qtype") == "order" or re.search(r"\d,\d{1,2}(\D|$)", a):   # LooGLE orderings "2,1,3"
        return False
    return bool(re.fullmatch(r"[\s$€£~<>]*[-+]?\d[\d,]*(\.\d+)?\s*(%|percent)?\s*", a))


def paired(xs, ys):
    d = [x - y for x, y in zip(xs, ys)]
    if not d:
        return None
    mu = sum(d) / len(d)
    sd = (sum((v - mu) ** 2 for v in d) / max(len(d) - 1, 1)) ** 0.5
    return {"gap": mu, "se": sd / len(d) ** 0.5, "n": len(d)}


def bench(name, path, split_dir, fmt, selfq, B):
    from xenc.doc_tasks import exact, load_doc_split
    E = json.load(open(path))
    meta = {it.key: it for sp in ("train", "val", "test") if (Path(split_dir) / sp / "items.json").exists()
            for it in load_doc_split(split_dir, sp)}
    ntok = {d: m["n_tokens"] for d, m in json.load(open(Path(split_dir) / "docs.json")).items()}
    es = E.get("es_choice") or {}
    eb = sorted(int(b) for b in (E["args"].get("eval_budgets") or E["args"]["budgets"]).split(","))
    Bk = str(max(b for b in eb if b <= B))
    rows = []
    for r in E["items"]:
        it = meta[r["key"]]
        def pick(c, d):
            b = es[d][Bk] if (es and d in es) else int(Bk)
            return r.get(f"{c}@{b}")
        g = {"closed_book": r.get("closed_book"), "closed_tail": r.get("closed_tail"), "text": r.get("text"),
             "kv_base": r.get(f"kv@{eb[0]}"), "kv_reg": pick("kv", r["doc_id"]),
             "wrong_doc": pick("wrong_doc", r["wrong"]) if r.get("wrong") else None}
        rows.append((it, {c: v for c, v in g.items() if v is not None}))

    def table(rs):
        out = {"n": len(rs)}
        for c in CONDS:
            v = [x[c] for _, x in rs if c in x]
            if v:
                out[c] = sum(y["score"] for y in v) / len(v)
                if fmt == "span":
                    out[c + "_em"] = sum(exact(it, x[c]["gen"]) for it, x in rs if c in x) / len(v)
        for o in ("text", "wrong_doc", "kv_base", "closed_tail"):
            pr = [(x["kv_reg"]["score"], x[o]["score"]) for _, x in rs if "kv_reg" in x and o in x]
            if pr:
                out[f"kv-{o}"] = paired(*zip(*pr))
        num = [(it, x) for it, x in rs if is_numeric(it)]
        if fmt == "span" and num:
            out["numeric"] = {c: sum(num_match(it, x[c]["gen"]) for it, x in num if c in x) / len(num) for c in CONDS
                              if all(c in x for _, x in num)}
            out["numeric"]["n"] = len(num)
        return out

    res = {"format": fmt, "all": table(rows), "subsets": {}, "length": {}}
    for lo, hi in LENGTH_BUCKETS:     # kv vs text by document length (Qwen tokens)
        sub = [(it, x) for it, x in rows if lo <= ntok[it.doc_id] < hi]
        if sub:
            res["length"][f"{lo // 1000}k–{hi // 1000 if hi < 10**9 else '∞'}k"] = table(sub)
    if name.startswith("CLIPPER"):    # pair accuracy: both claims of a (true, corrupted) pair right
        by = {}
        for it, x in rows:
            by.setdefault(it.meta["pair_id"], []).append(x)
        prs = [v for v in by.values() if len(v) == 2]
        res["pairs"] = {"n": len(prs), **{c: sum(all(y[c]["score"] == 1 for y in v) for v in prs) / max(len(prs), 1)
                                          for c in CONDS if all(c in y for v in prs for y in v)}}
        pk = [(float(all(y["kv_reg"]["score"] == 1 for y in v)), float(all(y[o]["score"] == 1 for y in v)), o)
              for v in prs for o in ("text", "wrong_doc") if all(o in y and "kv_reg" in y for y in v)]
        for o in ("text", "wrong_doc"):
            pp = [(p, q) for p, q, oo in pk if oo == o]
            if pp:
                res["pairs"][f"kv-{o}"] = paired(*zip(*pp))
    for lab, f in SUBSETS.get(name, {}).items():
        sub = [(it, x) for it, x in rows if f(it.meta or {})]
        if sub:
            res["subsets"][lab] = table(sub)
    c = E.get("cost", {})
    sq = [json.loads(line) for line in open(selfq)] if Path(selfq).exists() else []
    judged = {r["doc_id"] for r in E["items"]}
    sq_s = [x["seconds"] for x in sq if x["doc_id"] in judged]
    res["cost"] = {"docs": len(judged), "params": c.get("params_per_doc"), "reg_items": c.get("items_per_doc"),
                   "selfq_s": sum(sq_s) / len(sq_s) if sq_s else None, "teacher_s": c.get("teacher_s_mean"),
                   "reg_s": (c.get("reg_s_mean") or {}).get(Bk), "gpu": c.get("gpu"),
                   "es_budget_mean": (sum(es[d][Bk] for d in judged if d in es) / max(1, sum(d in es for d in judged)))
                   if es else None}
    return res


def fmt_row(name, fmt, t, cost=None, metric=None):
    f = lambda k: f"{t[k]:.3f}" if k in t else "—"   # noqa: E731
    g = lambda k: f"{t[k]['gap']:+.3f}±{t[k]['se']:.3f}" if t.get(k) else "—"   # noqa: E731
    met = metric or {"mcq": "acc", "nli": "acc", "span": "F1"}[fmt]
    s = (f"| {name} | {met} | {t['n']} | {f('closed_book')} | {f('closed_tail')} | {f('text')} | {f('kv_base')} | "
         f"**{f('kv_reg')}** | {f('wrong_doc')} | {g('kv-text')} | {g('kv-wrong_doc')} | {g('kv-kv_base')} |")
    if cost is not None:
        cs = "—"
        if cost.get("reg_s") is not None:
            cs = (f"{cost['docs']} docs; selfq {cost['selfq_s'] or 0:.0f}s + teacher {cost['teacher_s']:.0f}s + "
                  f"reg {cost['reg_s']:.0f}s; ES step {cost['es_budget_mean']:.0f}" if cost.get("es_budget_mean") is not None
                  else f"{cost['docs']} docs; teacher {cost['teacher_s']:.0f}s + reg {cost['reg_s']:.0f}s")
        s += f" {cs} |"
    return s


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=200, help="registration budget (per-doc early stop over budgets <= B)")
    ap.add_argument("--out", default="outputs/D22_table.md")
    ap.add_argument("--json", default="outputs/D22_table.json")
    a = ap.parse_args(argv)
    R = {}
    for name, (p, sd, fmt, sq) in RUNS.items():
        if Path(p).exists():
            R[name] = bench(name, p, sd, fmt, sq, a.B)
    H = ("| benchmark | metric | n | closed_book | closed_tail | text | kv_base (step 0) | **kv_reg** (ES≤{B}) | wrong_doc "
         "(reg.) | kv−text | kv−wrong | kv−kv_base |").format(B=a.B)
    L = [H + " registration cost / doc |", "|" + "---|" * 13]
    for name, r in R.items():
        L.append(fmt_row(name, r["format"], r["all"], r["cost"]))
    L += ["", "Subsets and secondary metrics (EM for spans; numeric-gold items: numeric match):", "",
          H, "|" + "---|" * 12]
    for name, r in R.items():
        for lab, t in r["subsets"].items():
            L.append(fmt_row(f"{name}: {lab}", r["format"], t))
        if r["format"] == "span":
            t = r["all"]
            em = {c[:-3]: t[c] for c in t if c.endswith("_em")}
            L.append(fmt_row(f"{name}: EM", "span", {"n": t["n"], **em}, metric="EM"))
            if "numeric" in t:
                L.append(fmt_row(f"{name}: numeric match", "span", t["numeric"], metric="num"))
    L += ["", "kv vs text by document length (Qwen3.5 tokens; all benchmarks pooled within a row's own metric):", "",
          H, "|" + "---|" * 12]
    for name, r in R.items():
        for lab, t in r["length"].items():
            L.append(fmt_row(f"{name}: {lab}", r["format"], t))
    for name, r in R.items():
        if "pairs" in r:
            L += ["", f"{name} pair accuracy (both claims of a true/corrupted pair right; chance 0.25):", "",
                  H, "|" + "---|" * 12, fmt_row(f"{name}: pairs", "mcq", r["pairs"], metric="pair acc")]
    md = "\n".join(L) + "\n"
    Path(a.out).write_text(md)
    json.dump(R, open(a.json, "w"), indent=1)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
