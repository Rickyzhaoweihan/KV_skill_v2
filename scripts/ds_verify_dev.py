"""DS: check that a device change (e.g. 1-GPU vs 2-GPU eval_d13) leaves greedy decodes unchanged.

  make:    python scripts/ds_verify_dev.py make --ref outputs/DSsw_ABlast_musique/eval_test.json --set outputs/D22_sets/musique.json \
               --k 3 --out <dir>/set_musique.json      (a copy of the set restricted to the ref's first k item keys)
  compare: python scripts/ds_verify_dev.py compare --ref <ref eval_test.json> --new <new eval_test.json>
           exit 0 iff every common item has identical kv / wrong_doc generations (prints the per-item comparison)
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("make", "compare"))
    ap.add_argument("--ref", required=True)
    ap.add_argument("--set", default="")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--new", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    R = json.load(open(a.ref))["items"]
    if a.mode == "make":
        S = json.load(open(a.set))
        S["item_keys"] = [it["key"] for it in R[: a.k]]
        S["n_items"] = len(S["item_keys"])
        json.dump(S, open(a.out, "w"), indent=1)
        print(f"verify set {a.out}: {S['item_keys']}")
        return 0
    ref = {it["key"]: it for it in R}
    new = {it["key"]: it for it in json.load(open(a.new))["items"]}
    common = sorted(set(ref) & set(new))
    bad = 0
    for k in common:
        for c in ("kv", "wrong_doc"):
            g0 = ref[k].get(c, ref[k].get(f"{c}@0", {})).get("gen")
            g1 = new[k].get(c, {}).get("gen")
            same = g0 == g1
            bad += not same
            print(f"  {k} {c}: {'SAME' if same else 'DIFF'} | ref {g0!r} | new {g1!r}")
    print(f"verify: {len(common)} items, {bad} differing generations")
    return 0 if common and not bad else 3


if __name__ == "__main__":
    sys.exit(main())
