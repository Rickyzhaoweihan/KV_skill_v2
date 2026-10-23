"""D11: aggregate eval_encside runs (outputs/D11_ev/*) into paired tables.

Per row (config × task × tail): kv, wrong_doc, closed_tail, closed_book, text (reader-only baselines joined by item
key from base9b_<task>_t<tail>), mean generated tokens, kv − wrong_doc ± paired se (n), kv − closed_tail ± paired se,
and the encoder's own text accuracy (enc<2b|4b>_<task>). LongHealth rows concatenate the five per-fold runs.

    python scripts/d11_aggregate.py [--ev outputs/D11_ev] [--md outputs/D11_tables.md]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

PARTIAL: set = set()
TASKS = ("longhealth", "quality", "musique", "lastingbench")


def load_items(path_glob):
    rows = {}
    for p in sorted(glob.glob(path_glob)):
        f = os.path.join(p, "eval_test.json")
        if os.path.exists(f):
            for r in json.load(open(f))["items"]:
                rows[r["key"]] = r
        elif os.path.exists(os.path.join(p, "eval_test.ckpt.jsonl")):      # cancelled run: partial (flagged)
            PARTIAL.add(os.path.basename(p))
            for line in open(os.path.join(p, "eval_test.ckpt.jsonl")):
                if line.strip():
                    r = json.loads(line)
                    rows[r["key"]] = r
    return rows


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return float("nan"), float("nan"), n
    mu = sum(d) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1))
    return mu, sd / math.sqrt(n), n


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def row(ev, tag, task, tail):
    pat = f"{ev}/{tag}_lh_f[0-4]_t{tail}" if task == "longhealth" else f"{ev}/{tag}_{task}_t{tail}"
    kv = load_items(pat)
    if not kv:
        return None
    nf = len(glob.glob(pat)) if task == "longhealth" else 1
    base = load_items(f"{ev}/base9b_{task}_t{tail}")
    base64 = load_items(f"{ev}/base9b_{task}_t64")
    enc = "2b" if tag.startswith(("a2", "dg2")) else "4b"
    encr = load_items(f"{ev}/enc{enc}_{task}")
    keys = sorted(kv)
    S = lambda rows, k, c: rows[k][c]["score"]  # noqa: E731
    part = any(os.path.basename(p) in PARTIAL for p in glob.glob(pat))
    out = {"tag": tag, "task": task, "tail": tail, "n": len(keys), "folds": nf, "partial": part}
    out["kv"] = mean([S(kv, k, "kv") for k in keys])
    out["wrong"] = mean([S(kv, k, "wrong_doc") for k in keys])
    out["gen_kv"] = mean([kv[k]["kv"]["gen_tokens"] for k in keys])
    out["gen_wrong"] = mean([kv[k]["wrong_doc"]["gen_tokens"] for k in keys])
    out["gap"], out["gap_se"], _ = paired([S(kv, k, "kv") for k in keys], [S(kv, k, "wrong_doc") for k in keys])
    out["wins"] = sum(S(kv, k, "kv") > S(kv, k, "wrong_doc") for k in keys)
    out["losses"] = sum(S(kv, k, "kv") < S(kv, k, "wrong_doc") for k in keys)
    bk = [k for k in keys if k in base and "closed_tail" in base[k]]
    out["n_base"] = len(bk)
    out["ctail"] = mean([S(base, k, "closed_tail") for k in bk])
    out["gen_ctail"] = mean([base[k]["closed_tail"]["gen_tokens"] for k in bk])
    out["d_tail"], out["d_tail_se"], _ = paired([S(kv, k, "kv") for k in bk], [S(base, k, "closed_tail") for k in bk])
    b64 = [k for k in keys if k in base64 and "text" in base64[k]]
    out["closed"] = mean([S(base64, k, "closed_book") for k in b64])
    out["text"] = mean([S(base64, k, "text") for k in b64])
    out["gen_text"] = mean([base64[k]["text"]["gen_tokens"] for k in b64])
    ek = [k for k in keys if k in encr]
    out["enc_text"] = mean([S(encr, k, "text") for k in ek])
    out["enc_closed"] = mean([S(encr, k, "closed_book") for k in ek])
    return out


def fmt(r):
    f = lambda x: "—" if x != x else f"{x:.3f}"  # noqa: E731
    ok = "✓" if r["kv"] >= r["ctail"] else "✗"
    return (f"| {r['tag']}{' (partial)' if r.get('partial') else ''} | {r['task']}{' (' + str(r['folds']) + ' folds)' if r['task'] == 'longhealth' else ''} | {r['tail']} | {r['n']} "
            f"| {f(r['kv'])} ({r['gen_kv']:.1f}) | {f(r['wrong'])} ({r['gen_wrong']:.1f}) | {f(r['ctail'])} ({r['gen_ctail']:.1f}) "
            f"| {f(r['closed'])} | {f(r['text'])} | **{r['gap']:+.3f} ± {r['gap_se']:.3f}** ({r['wins']}/{r['losses']}) "
            f"| {r['d_tail']:+.3f} ± {r['d_tail_se']:.3f} {ok} | {f(r['enc_text'])} |")


HDR = ("| config | task | tail | n | kv (gen) | wrong_doc (gen) | closed_tail (gen) | closed_book | text | kv − wrong ± se (W/L) "
       "| kv − closed_tail ± se | encoder text |\n|---|---|---|---|---|---|---|---|---|---|---|---|")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ev", default="outputs/D11_ev")
    ap.add_argument("--md", default="outputs/D11_tables.md")
    ap.add_argument("--json", default="outputs/D11_rows.json")
    a = ap.parse_args()
    tags = sorted({re.sub(r"_(lh|longhealth|quality|musique|lastingbench)_.*$", "", os.path.basename(p))
                   for p in glob.glob(f"{a.ev}/*") if not os.path.basename(p).startswith(("base9b", "enc"))})
    rows = []
    for tag in tags:
        for task in TASKS:
            for tail in (64, 0, 256):
                r = row(a.ev, tag, task, tail)
                if r:
                    rows.append(r)
    lines = [HDR] + [fmt(r) for r in rows]
    open(a.md, "w").write("\n".join(lines) + "\n")
    json.dump(rows, open(a.json, "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
