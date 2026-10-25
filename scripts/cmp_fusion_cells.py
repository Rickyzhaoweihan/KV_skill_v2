"""CMP: does a fused / compressed arm recover HM-only and NEITHER items? (CPU, from per-item eval json)

For each set that has per-item HM rows on disk (scripts/cmp_route_oracle.HM_FILES; HM directory READ ONLY), the items
are split into 4 cells by our base read (outputs/DSsw_<base>_<set>) and HM: both correct / ours-only / HM-only /
neither (correct = score >= .5). For every CMP / HYB arm (outputs/CMPsw_<tag>_<set>) on the SAME items: its score,
the routing oracle max(ours, HM) (a SELECTION bound between the two fixed systems, not a fusion ceiling), and its
accuracy inside each cell (recovery of HM-only and neither items, retention of ours-only and both).

    python scripts/cmp_fusion_cells.py --tags a128,h512g1,orc --base ABbest [--out outputs/CMP_fusion_cells.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmp_route_oracle as RO  # noqa: E402
import ds_sweep_table as T  # noqa: E402

CELLS = ("both", "ours_only", "hm_only", "neither")


def hm_scores(path: str) -> dict:
    out = {}
    for line in open(os.path.join(RO.RX, path)):
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = float(r["xs"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--base", default="ABbest")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/CMP_fusion_cells.json")
    a = ap.parse_args(argv)
    T.LEAK = "outputs/DS_judged_leak_max8k.json"
    res, lines = {}, ["| set (HM row) | arm | n | ours | HM | routing oracle | arm | arm − routing | arm acc in cells "
                      "both / ours-only / HM-only / neither (cell sizes) |", "|" + "---|" * 9]
    for s, cands in RO.HM_FILES.items():
        bf = f"outputs/DSsw_{a.base}_{s}/eval_test.json"
        if a.base == "ABbest" and not os.path.exists(bf):
            bf = next((p.format(s=s) for p in T.OTHER["ABbest"] if os.path.exists(p.format(s=s))), bf)
        ours = T.scores(bf)
        if not ours:
            continue
        lab, path = cands[RO.PRIMARY.get(s) or 0]
        hm = hm_scores(path)
        ck = T.clean_keys(s)
        for tag in [t for t in a.tags.split(",") if t]:
            arm = T.scores(f"outputs/CMPsw_{tag}_{s}/eval_test.json")
            keys = set(ours) & set(hm) & set(arm)
            if ck is not None:
                keys &= ck
            if not keys:
                continue
            ok = lambda x: x >= a.thr          # noqa: E731
            cell = {k: ("both" if ok(ours[k]) and ok(hm[k]) else "ours_only" if ok(ours[k]) else
                        "hm_only" if ok(hm[k]) else "neither") for k in keys}
            n = len(keys)
            m = lambda d: sum(d[k] for k in keys) / n   # noqa: E731
            rout = sum(max(ours[k], hm[k]) for k in keys) / n
            per = {}
            for c in CELLS:
                kk = [k for k in keys if cell[k] == c]
                per[c] = {"n": len(kk), "arm_acc": (sum(ok(arm[k]) for k in kk) / len(kk)) if kk else None}
            r = {"hm_row": lab, "n": n, "ours": m(ours), "hm": m(hm), "routing_oracle": rout, "arm": m(arm),
                 "cells": per}
            res.setdefault(s, {})[tag] = r
            cc = " / ".join(f"{per[c]['arm_acc']:.2f}" if per[c]["arm_acc"] is not None else "–" for c in CELLS)
            cs = " / ".join(str(per[c]["n"]) for c in CELLS)
            lines.append(f"| {s} ({lab}) | {tag} | {n} | {r['ours']:.3f} | {r['hm']:.3f} | {rout:.3f} | {r['arm']:.3f} | "
                         f"{r['arm'] - rout:+.3f} | {cc} ({cs}) |")
    print("\n".join(lines))
    json.dump({"rows": res, "markdown": "\n".join(lines), "base": a.base, "thr": a.thr}, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
