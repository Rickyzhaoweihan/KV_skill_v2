"""D13 lever 1 (BEAT_TEXT): train/val documents that are near-duplicates of a JUDGED document.

For every task, the judged set = judged_docs + reg_docs + wrong-map docs of outputs/D22_sets/<task>.json (+ the
LongHealth fold 0 / QuALITY test convention is enforced elsewhere by split). A train/val doc t is dropped when the
word-8-gram containment c(t -> J) >= thr (J = union of judged-doc 8-grams), or when some judged doc j has
c(j -> t) >= thr. Output: {task: [doc_id, ...]} (json) for register_d13.py --exclude_docs.

    python scripts/d13_leakage.py --out outputs/D13_lever1_leakage.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def grams(text: str, n: int = 8) -> set:
    w = text.split()
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}


def contain(a: set, b: set) -> float:
    return len(a & b) / max(len(a), 1)


def main(argv=None) -> int:
    from register_d13 import excluded_docs
    from xenc.doc_tasks import load_docs
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="qasper,contractnli,shakespeare_cf,loogle,clipper,musique,lastingbench")
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res, rep = {}, {}
    for t in a.tasks.split(","):
        sd = f"data/{t}_split"
        D = load_docs(sd)
        sets = [f"outputs/D22_sets/{t}.json"] + (["outputs/D22_sets/loogle_long.json"] if t == "loogle" else [])
        J = excluded_docs(",".join(s for s in sets if os.path.exists(s)))
        cand = sorted({i["doc_id"] for s in ("train", "val") for i in json.load(open(f"{sd}/{s}/items.json"))} - J)
        gJ = {j: grams(D[j]) for j in sorted(J) if j in D}
        GJ = set().union(*gJ.values()) if gJ else set()
        drop = []
        for d in cand:
            g = grams(D[d])
            if contain(g, GJ) >= a.thr or any(contain(gj, g) >= a.thr for gj in gJ.values()):
                drop.append(d)
        res[t], rep[t] = drop, {"judged": len(J), "candidates": len(cand), "dropped": len(drop)}
        print(t, rep[t], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    json.dump(rep, open(a.out.replace(".json", "_report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
