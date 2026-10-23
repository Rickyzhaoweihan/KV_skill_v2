"""L1: per-length-bucket paired comparison of register_d13 runs on the SAME val items (needs --save_items 1).

usage: python scripts/l1_buckets.py outputs/L1_ctrl outputs/L1_logn [outputs/L1_chunk ...] [--last 3]

Each item's document length = docs.json n_tokens of its doc (the operator is the doc prefix minus the 64-token tail,
capped at --max_tokens − 1024). Buckets: ≤16k, 16–48k, ≥48k (and ≥16k pooled). For every bucket: control kv / variant
kv (mean of each item's kv over the last k val checks), wrong_doc and closed_tail of the control, the paired
variant − control difference ± SE, and the same at step 0 (must be 0 for zero-init forms). Item → doc is re-derived
from the control's args.json with register_d13's own selection / exclusion code (asserts the item set matches)."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _R)
sys.path.insert(0, os.path.join(_R, "scripts"))

BUCKETS = (("<=16k", 0, 16000), ("16-48k", 16001, 47999), (">=48k", 48000, 10 ** 9))


def item_docs(args: dict) -> dict:
    from fit_d13 import TASK_SPLIT, _get, select_items
    from register_d13 import doc_len_filter, excluded_docs
    val, _ = select_items(args["val"], "val", {args["val_fold"]}, args["val_q_per_doc"], args["val_seed"])
    drop = excluded_docs(args.get("exclude_sets", ""))
    if args.get("exclude_docs"):
        X = json.load(open(os.path.join(_R, args["exclude_docs"])))
        drop |= set(sum(X.values(), [])) if isinstance(X, dict) else set(X)
    keep_len = doc_len_filter(args.get("doc_len", ""))
    val = [i for i in val if _get(i, "doc_id") not in drop and keep_len(_get(i, "doc_id"))]
    ntok = {}
    for t, sd in TASK_SPLIT.items():
        d = json.load(open(os.path.join(_R, sd, "docs.json")))
        ntok.update({k: int(v.get("n_tokens", -1)) for k, v in d.items()})
    cap = int(args.get("max_tokens", 16000)) - 1024
    return {str(_get(i, "key")): (_get(i, "doc_id"), min(ntok[_get(i, "doc_id")], cap)) for i in val}


def load(d, last):
    ev = json.load(open(os.path.join(d, "history.json")))["evals"]
    assert all("items" in e for e in ev), f"{d}: needs --save_items 1"
    z = {r[0]: r for r in ev[0]["items"]}
    sel = [e for e in ev if e["step"] > 0][-last:]
    avg = {}
    for e in sel:
        for r in e["items"]:
            avg.setdefault(r[0], []).append(r[1])
    return z, {k: sum(v) / len(v) for k, v in avg.items()}, [e["step"] for e in sel]


def stat(x):
    n = len(x)
    if not n:
        return float("nan"), float("nan")
    mu = sum(x) / n
    se = math.sqrt(sum((v - mu) ** 2 for v in x) / (n - 1) / n) if n > 1 else float("nan")
    return mu, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ctrl")
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--last", type=int, default=3)
    x = ap.parse_args()
    args = json.load(open(os.path.join(x.ctrl, "args.json")))
    docs = item_docs(args)
    z0, mc, sc = load(x.ctrl, x.last)
    assert set(mc) == set(docs), f"control items != re-derived val items ({len(mc)} vs {len(docs)})"
    print(f"control {x.ctrl}: steps {sc}, n={len(mc)}")
    groups = [(n, [k for k, (_, L) in docs.items() if lo <= L <= hi]) for n, lo, hi in BUCKETS]
    groups.append(("all>=16k", [k for k, (_, L) in docs.items() if L > 16000]))
    groups.append(("all", list(docs)))
    for v in x.variants:
        zv, mv, sv = load(v, x.last)
        assert set(mv) == set(mc), f"{v}: different val items"
        a2 = json.load(open(os.path.join(v, "args.json")))
        for k in ("val", "val_seed", "val_q_per_doc", "seed", "steps", "eval_every", "init", "exclude_docs", "doc_len"):
            assert a2.get(k) == args.get(k), f"{v}: {k} differs ({a2.get(k)} vs {args.get(k)})"
        print(f"\n== {v} (steps {sv}) vs control")
        print("| bucket | n (docs) | ctrl kv | var kv | ctrl wrong / tail (last) | var − ctrl (paired) | step-0 var − ctrl |")
        print("|---|---|---|---|---|---|---|")
        for name, ks in groups:
            if not ks:
                continue
            c, w = stat([mc[k] for k in ks]), stat([mv[k] for k in ks])
            d, d0 = stat([mv[k] - mc[k] for k in ks]), stat([zv[k][1] - z0[k][1] for k in ks])
            nd = len({docs[k][0] for k in ks})
            wr = stat([z0[k][2] for k in ks if z0[k][2] is not None])[0]
            tl = stat([z0[k][3] for k in ks if z0[k][3] is not None])[0]
            flag = " **>2SE**" if abs(d[0]) > 2 * d[1] else ""
            print(f"| {name} | {len(ks)} ({nd}) | {c[0]:.3f} | {w[0]:.3f} | {wr:.3f} / {tl:.3f} (step 0) | "
                  f"{d[0]:+.3f} ± {d[1]:.3f}{flag} | {d0[0]:+.3f} |")


if __name__ == "__main__":
    main()
