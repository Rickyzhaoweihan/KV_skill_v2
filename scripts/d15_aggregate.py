"""D15: paired gaps for eval_d13.py outputs: kv - wrong_doc, kv - closed_tail and kv - text (the text condition is
taken from a separate run on the same items, joined by item key), each mean ± paired se (n).

    python scripts/d15_aggregate.py --text_lh outputs/D15_ev_text_lh --text_q outputs/D15_ev_text_q outputs/D15_ev_* outputs/D13_s2w10_*
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def paired(rows, a, b, other=None):
    d = []
    for r in rows:
        if a not in r:
            continue
        if b in r:
            d.append(r[a]["score"] - r[b]["score"])
        elif other is not None and r["key"] in other:
            d.append(r[a]["score"] - other[r["key"]])
    if not d:
        return None
    mu = sum(d) / len(d)
    sd = (sum((x - mu) ** 2 for x in d) / max(len(d) - 1, 1)) ** 0.5
    return mu, sd / len(d) ** 0.5, len(d)


def text_map(path):
    if not path or not os.path.exists(os.path.join(path, "eval_test.json")):
        return {}
    return {r["key"]: r["text"]["score"] for r in json.load(open(os.path.join(path, "eval_test.json")))["items"] if "text" in r}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--text_lh", default="outputs/D15_ev_text_lh")
    ap.add_argument("--text_q", default="outputs/D15_ev_text_q")
    ap.add_argument("--json_out", default="")
    a = ap.parse_args(argv)
    T = {"lh": text_map(a.text_lh), "q": text_map(a.text_q)}
    out = []
    print(f"{'run':34s} {'n':>4s} {'closed_tail':>11s} {'text':>6s} {'kv':>6s} {'wrong':>6s} {'gen':>4s}  "
          f"{'kv-wrong':>15s} {'kv-closed_tail':>15s} {'kv-text':>15s}")
    for r in a.runs:
        p = os.path.join(r, "eval_test.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        rows = d["items"]
        task = "lh" if "longhealth" in d["args"]["split_dir"] else "q"
        if "kv" not in rows[0]:
            continue
        m = lambda c: sum(x[c]["score"] for x in rows if c in x) / max(sum(1 for x in rows if c in x), 1)  # noqa: E731
        tx = [T[task][x["key"]] for x in rows if x["key"] in T[task]]
        res = {"run": r, "task": task, "n": len(rows), "kv": m("kv"), "wrong_doc": m("wrong_doc"),
               "closed_tail": m("closed_tail"), "text": sum(tx) / len(tx) if tx else float("nan"),
               "gen": sum(x["kv"]["gen_tokens"] for x in rows) / len(rows),
               "kv_wrong": paired(rows, "kv", "wrong_doc"), "kv_tail": paired(rows, "kv", "closed_tail"),
               "kv_text": paired(rows, "kv", "text", T[task])}
        out.append(res)
        f = lambda g: f"{g[0]:+.3f}±{g[1]:.3f}" if g else "n/a"  # noqa: E731
        print(f"{os.path.basename(r.rstrip('/')):34s} {res['n']:4d} {res['closed_tail']:11.3f} {res['text']:6.3f} "
              f"{res['kv']:6.3f} {res['wrong_doc']:6.3f} {res['gen']:4.1f}  {f(res['kv_wrong']):>15s} "
              f"{f(res['kv_tail']):>15s} {f(res['kv_text']):>15s}")
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
