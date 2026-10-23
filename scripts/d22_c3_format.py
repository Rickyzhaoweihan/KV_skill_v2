"""D22 / BEAT_TEXT C3 (CPU, existing D22 outputs): is the span-F1 gap vs text CONTENT or FORMAT?

Per span benchmark and condition (text, kv_reg, kv_base, wrong_doc), over the judged items:
  sub_doc     predicted answer (SQuAD-normalized) is an exact substring of the normalized document
  has_gold    a gold answer is a substring of the prediction (content present, maybe verbose)
  len_ratio   mean / median of pred tokens / gold tokens (shortest gold)
  noans       empty, or a refusal ("unknown", "not mentioned", "not specified", "none", "n/a", ...)
  P / R / F1  token precision / recall / F1 (max over golds)
  F1_win      length-normalized F1: max F1 over every contiguous window of the prediction with the gold's length
              (removes the verbosity penalty; a too-short answer is not helped)
  F1_ans      F1 on items where the condition gave a non-refusal answer
Reading: if kv's F1_win / has_gold gap to text is much smaller than its F1 gap, the gap is FORMAT; if R and has_gold
are far below text, it is CONTENT.

    python scripts/d22_c3_format.py --out outputs/D22_C3.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SPAN = {"Qasper": ("outputs/D22_qasper/eval_test.json", "data/qasper_split"),
        "MuSiQue": ("outputs/D22_musique/eval_test.json", "data/musique_split"),
        "LooGLE": ("outputs/D22_loogle/eval_test.json", "data/loogle_split"),
        "LooGLE-long": ("outputs/D22_loogle_long/eval_test.json", "data/loogle_split"),
        "LastingBench": ("outputs/D22_lastingbench/eval_test.json", "data/lastingbench_split")}
REFUSE = re.compile(r"^(unknown|not (mentioned|specified|stated|provided|given|available|applicable|known)|none|n ?a|"
                    r"no answer|cannot (be )?determined?|unanswerable|not in (the )?(document|text)|insufficient.*)$")
CONDS = ("text", "kv_reg", "kv_base", "wrong_doc")


def prf(pred: list[str], gold: list[str]):
    n = sum((Counter(pred) & Counter(gold)).values())
    if not pred or not gold or n == 0:
        return 0.0, 0.0, 0.0
    p, r = n / len(pred), n / len(gold)
    return p, r, 2 * p * r / (p + r)


def rows_of(path, split_dir, B=200):
    from xenc.doc_tasks import load_doc_split, load_docs
    E = json.load(open(path))
    meta = {it.key: it for sp in ("train", "val", "test") if (Path(split_dir) / sp / "items.json").exists()
            for it in load_doc_split(split_dir, sp)}
    docs = load_docs(split_dir)
    es = E.get("es_choice") or {}
    eb = sorted(int(b) for b in (E["args"].get("eval_budgets") or E["args"]["budgets"]).split(","))
    Bk = str(max(b for b in eb if b <= B))
    out = []
    for r in E["items"]:
        pk = lambda c, d: r.get(f"{c}@{es[d][Bk] if (es and d in es) else int(Bk)}")   # noqa: E731
        g = {"text": r.get("text"), "kv_base": r.get(f"kv@{eb[0]}"), "kv_reg": pk("kv", r["doc_id"]),
             "wrong_doc": pk("wrong_doc", r["wrong"]) if r.get("wrong") else None}
        out.append((meta[r["key"]], docs[r["doc_id"]], {c: v for c, v in g.items() if v}))
    return out


def stats(rows, cond):
    from xenc.doc_tasks import extract_answer, normalize_answer
    acc = {k: [] for k in ("sub_doc", "has_gold", "len_ratio", "noans", "P", "R", "F1", "F1_win", "F1_ans")}
    ndoc = {}
    for it, doc, g in rows:
        if cond not in g:
            continue
        pa = normalize_answer(extract_answer(g[cond]["gen"]))
        golds = [normalize_answer(x) for x in it.answers if normalize_answer(x)]
        if not golds:
            continue
        key = id(doc)
        if key not in ndoc:
            ndoc[key] = " " + normalize_answer(doc) + " "
        pt = pa.split()
        refuse = (not pt) or bool(REFUSE.match(pa))
        best = max((prf(pt, gg.split()) for gg in golds), key=lambda x: x[2])
        win = 0.0
        for gg in golds:
            L = len(gg.split())
            for i in range(max(1, len(pt) - L + 1)):
                win = max(win, prf(pt[i:i + L], gg.split())[2])
        gl = min(len(gg.split()) for gg in golds)
        acc["sub_doc"].append(float(bool(pt) and f" {pa} " in ndoc[key]))
        acc["has_gold"].append(float(any(f" {gg} " in f" {pa} " for gg in golds)))
        acc["len_ratio"].append(len(pt) / max(gl, 1))
        acc["noans"].append(float(refuse))
        acc["P"].append(best[0])
        acc["R"].append(best[1])
        acc["F1"].append(best[2])
        acc["F1_win"].append(win)
        if not refuse:
            acc["F1_ans"].append(best[2])
    m = {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()}
    lr = sorted(acc["len_ratio"])
    m["len_med"] = lr[len(lr) // 2] if lr else float("nan")
    m["n"] = len(acc["F1"])
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/D22_C3.md")
    a = ap.parse_args(argv)
    L = ["| benchmark | cond | n | F1 | F1_win (len-norm) | P | R | has_gold | sub_doc | len ratio mean / med | no-answer | F1 on answered |",
         "|" + "---|" * 12]
    res = {}
    for name, (p, sd) in SPAN.items():
        if not Path(p).exists():
            continue
        rows = rows_of(p, sd)
        res[name] = {}
        for c in CONDS:
            m = stats(rows, c)
            res[name][c] = m
            L.append(f"| {name} | {c} | {m['n']} | {m['F1']:.3f} | {m['F1_win']:.3f} | {m['P']:.3f} | {m['R']:.3f} | "
                     f"{m['has_gold']:.3f} | {m['sub_doc']:.3f} | {m['len_ratio']:.2f} / {m['len_med']:.2f} | "
                     f"{m['noans']:.3f} | {m['F1_ans']:.3f} |")
    md = "\n".join(L) + "\n"
    Path(a.out).write_text(md)
    json.dump(res, open(Path(a.out).with_suffix(".json"), "w"), indent=1)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
