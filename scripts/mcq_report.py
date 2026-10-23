"""MCQ (agent MCQ): raw-score tables from the per-item judge files outputs/MCQ_<tag>_<set>/eval_test.json.

Per set and per task: n, closed_book, closed_tail, text, kv@0 (unregistered, MAIN), kv_reg (per-doc early stop <= 200,
D22's rule: kv@<es_choice[doc]['200']>), wrong_doc@0; paired SEs (sd/sqrt(n)) for kv@0 - text, kv_reg - kv@0,
kv@0 - wrong_doc@0. Recomputed independently from the items (not from summary.md).

    python -I scripts/mcq_report.py [--tag L1f] [--sets mcq_leval,mcq_bamboo,mcq_hallu,mcq_musr] [--out outputs/MCQ_tables.md]
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import OrderedDict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pse(d):
    n = len(d)
    if n < 2:
        return 0.0, 0.0
    m = sum(d) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)


def rows_of(path):
    E = json.load(open(path))
    es = E.get("es_choice") or {}
    out = []
    for r in E["items"]:
        b = (es.get(r["doc_id"]) or {}).get("200", 0)
        x = {"key": r["key"], "doc_id": r["doc_id"]}
        for c in ("closed_book", "closed_tail", "text"):
            if c in r:
                x[c] = r[c]["score"]
        x["kv0"] = r["kv@0"]["score"]
        x["kvreg"] = r[f"kv@{b}"]["score"]
        x["wrong0"] = r["wrong_doc@0"]["score"]
        out.append(x)
    return out


def table(name, R, task_of):
    groups = OrderedDict([("ALL", R)])
    for t in sorted({task_of[r["key"]] for r in R}):
        groups[t] = [r for r in R if task_of[r["key"]] == t]
    L = [f"### {name}", "",
         "| task | n | closed_book | closed_tail | **text** | **kv@0** | kv_reg | wrong_doc@0 | kv@0−text | kv_reg−kv@0 | kv@0−wrong |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g, X in groups.items():
        n = len(X)

        def m(c):
            v = [r[c] for r in X if c in r]
            return f"{sum(v) / len(v):.3f}" if v else "—"

        def d(a, b):
            if not all(a in r and b in r for r in X):
                return "—"
            mu, se = pse([r[a] - r[b] for r in X])
            return f"{mu:+.3f} ± {se:.3f}"
        L.append(f"| {g} | {n} | {m('closed_book')} | {m('closed_tail')} | {m('text')} | {m('kv0')} | {m('kvreg')} | "
                 f"{m('wrong0')} | {d('kv0', 'text')} | {d('kvreg', 'kv0')} | {d('kv0', 'wrong0')} |")
    flips = sum(r["kvreg"] != r["kv0"] for r in R)
    L += ["", f"kv_reg vs kv@0: {flips} of {len(R)} items change score.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="L1f")
    ap.add_argument("--sets", default="mcq_leval,mcq_bamboo,mcq_hallu,mcq_musr")
    ap.add_argument("--out", default="outputs/MCQ_tables.md")
    a = ap.parse_args(argv)
    os.chdir(REPO)
    parts = [f"# MCQ judge tables, tag {a.tag} (reader Qwen3.5-9B, encoder 4B; recomputed from per-item files)", ""]
    for s in a.sets.split(","):
        p = f"outputs/MCQ_{a.tag}_{s}/eval_test.json"
        if not os.path.exists(p):
            parts.append(f"### {s}\n\n(pending: {p} missing)\n")
            continue
        S = json.load(open(f"outputs/MCQ_sets/{s}.json"))
        task_of = {}
        for sp in S["splits"]:
            for it in json.load(open(os.path.join(S["split_dir"], sp, "items.json"))):
                task_of[it["key"]] = it["task"]
        R = rows_of(p)
        assert len(R) == S["n_items"] and {r["key"] for r in R} == set(S["item_keys"]), f"{s}: item set mismatch"
        parts.append(table(s, R, task_of))
    txt = "\n".join(parts)
    open(a.out, "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
