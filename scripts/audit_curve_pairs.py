"""AUDIT (CPU, read-only): paired differences between checkpoints of a DS curve, same item set as scripts/ds_curve.py
(key intersection over the steps that exist, clean subset). Imports ds_curve.rows / ds_sweep_table, changes nothing.

    python scripts/audit_curve_pairs.py [--out outputs/DS_max8k_curve_pairs_audit.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds_curve as C  # noqa: E402
import ds_sweep_table as T  # noqa: E402


def paired(R, keys, a, b):
    d = [R[b][k] - R[a][k] for k in keys]
    n = len(d)
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / max(n - 1, 1)) / math.sqrt(n)
    return m, se, n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="0=ABbest,2000=m8k2000,4000=m8k4000,6000=m8k6000,8000=m8k8000")
    ap.add_argument("--leak", default="outputs/DS_judged_leak_max8k.json")
    ap.add_argument("--out", default="outputs/DS_max8k_curve_pairs_audit.json")
    a = ap.parse_args(argv)
    T.LEAK = a.leak
    steps = [(int(x.split("=")[0]), x.split("=")[1]) for x in a.steps.split(",")]
    res = {}
    print("| set | n | step0 | s6000 | s8000 | s6000−s0 ± SE | s8000−s0 ± SE | s8000−s6000 ± SE | has step 0 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in T.REFS:
        R = {st: C.rows(tag, s) for st, tag in steps}
        R = {st: r for st, r in R.items() if r}
        if not R:
            continue
        keys = set.intersection(*[set(r) for r in R.values()])
        ck = T.clean_keys(s)
        if ck is not None:
            keys &= ck
        keys = sorted(keys)
        row = {"n": len(keys), "partial": s in C.PARTIAL, "has_step0": 0 in R,
               "curve": {st: sum(R[st][k] for k in keys) / len(keys) for st in R}}
        for nm, x, y in (("s6000-s0", 0, 6000), ("s8000-s0", 0, 8000), ("s8000-s6000", 6000, 8000)):
            if x in R and y in R:
                row[nm] = paired(R, keys, x, y)
        res[s] = row
        f = lambda nm: f"{row[nm][0]:+.3f} ± {row[nm][1]:.3f}" if nm in row else "–"      # noqa: E731
        c = row["curve"]
        g = lambda st: f"{c[st]:.3f}" if st in c else "–"                                 # noqa: E731
        print(f"| {s}{' (PARTIAL)' if row['partial'] else ''} | {len(keys)} | {g(0)} | {g(6000)} | {g(8000)} | {f('s6000-s0')} | {f('s8000-s0')} | {f('s8000-s6000')} | {'yes' if 0 in R else 'NO'} |")
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
