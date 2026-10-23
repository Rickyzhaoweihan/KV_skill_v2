"""[D19] Word-8-gram leakage check of the D19 datasets' TEST documents against
  (1) the D15 label-free pool (data/D15_selfq/docs.json, 5.6k docs),
  (2) the TRAIN(+val) documents of the existing splits (quality / longhealth / musique / lastingbench),
  (3) the TRAIN(+val) documents of every D19 dataset (its own and the others).
Fingerprints are exactly D15's (scripts/d15_build_pool.py::shingles: blake2b-hashed word 8-grams kept at mod 8).
Per test doc we report `max_frac` = largest share of its fingerprints found in ONE reference doc (D15's rule
flags a pair when shared >= 15% of the smaller doc and >= 5 fingerprints) and `union_frac` = share found anywhere
in that reference group. Stdlib only.

  python -I scripts/d19_leakage.py [--names shakespeare,shakespeare_cf,contractnli,loogle,clipper]
                                   [--out outputs/D19_leakage.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
WORD = re.compile(r"[a-z0-9]+")
OLD = ["quality", "longhealth", "musique", "lastingbench"]
D19 = ["shakespeare", "shakespeare_cf", "contractnli", "loogle", "clipper"]


def shingles(text: str, mod: int = 8, n: int = 8) -> set:
    w = WORD.findall(text.lower())
    out = set()
    for i in range(0, max(len(w) - n + 1, 0)):
        x = int.from_bytes(hashlib.blake2b(" ".join(w[i:i + n]).encode(), digest_size=8).digest(), "little")
        if x % mod == 0:
            out.add(x)
    return out


def split_docs(name: str, splits) -> dict[str, Path]:
    sd = DATA / f"{name}_split"
    index = json.load(open(sd / "docs.json"))
    ids = set()
    for s in splits:
        ids |= {it["doc_id"] for it in json.load(open(sd / s / "items.json"))}
    return {d: DATA / index[d]["path"] for d in sorted(ids)}


class Index:
    def __init__(self):
        self.post: dict[int, list[str]] = {}
        self.size: dict[str, int] = {}

    def add(self, did: str, sh: set):
        self.size[did] = len(sh)
        for x in sh:
            self.post.setdefault(x, []).append(did)

    def query(self, sh: set) -> dict:
        c, anyhit = Counter(), 0
        for x in sh:
            p = self.post.get(x)
            if p:
                anyhit += 1
                c.update(set(p))
        best, frac, flag = None, 0.0, None
        for did, k in c.most_common(5):
            f = k / max(1, len(sh))
            if f > frac:
                best, frac = did, f
            if flag is None and k >= 5 and k >= 0.15 * max(1, min(len(sh), self.size[did])):
                flag = did
        return {"max_frac": round(frac, 4), "max_doc": best, "union_frac": round(anyhit / max(1, len(sh)), 4),
                "d15_rule_hit": flag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=",".join(D19))
    ap.add_argument("--out", default=str(REPO / "outputs/D19_leakage.json"))
    args = ap.parse_args()
    names = [n for n in args.names.split(",") if (DATA / f"{n}_split").exists()]

    cache: dict[Path, set] = {}

    def sh_of(p: Path) -> set:
        if p not in cache:
            cache[p] = shingles(p.read_text(encoding="utf-8", errors="ignore"))
        return cache[p]

    groups: dict[str, Index] = {}
    pool = json.load(open(DATA / "D15_selfq/docs.json"))
    g = groups["D15_pool"] = Index()
    for did, m in pool.items():
        g.add(did, shingles((DATA / m["path"]).read_text(encoding="utf-8", errors="ignore")))
    print(f"D15 pool indexed: {len(g.size)} docs", flush=True)
    g = groups["old_splits_trainval"] = Index()
    for n in OLD:
        if (DATA / f"{n}_split").exists():
            for did, p in split_docs(n, ("train", "val")).items():
                if n == "longhealth":   # LongHealth splits are folds over all 20 patients: every patient is "train"
                    continue
                g.add(did, sh_of(p))
    for n in D19:
        if (DATA / f"{n}_split").exists():
            g = groups[f"{n}_trainval"] = Index()
            for did, p in split_docs(n, ("train", "val")).items():
                g.add(did, sh_of(p))
    print("reference groups:", {k: len(v.size) for k, v in groups.items()}, flush=True)

    report = {}
    for n in names:
        test = split_docs(n, ("test",))
        rows = {did: {gn: gi.query(sh_of(p)) for gn, gi in groups.items()} for did, p in test.items()}
        summ = {}
        for gn in groups:
            mf = [r[gn]["max_frac"] for r in rows.values()]
            uf = [r[gn]["union_frac"] for r in rows.values()]
            summ[gn] = {"n_test_docs": len(mf), "mean_max_frac": round(sum(mf) / len(mf), 4),
                        "max_max_frac": max(mf), "n_ge_0.15": sum(x >= 0.15 for x in mf),
                        "n_ge_0.50": sum(x >= 0.5 for x in mf), "mean_union_frac": round(sum(uf) / len(uf), 4),
                        "n_d15_rule_hits": sum(r[gn]["d15_rule_hit"] is not None for r in rows.values())}
        report[n] = {"summary": summ, "per_doc": rows}
        print(f"\n== {n} test ({len(test)} docs)")
        print(f"{'reference':<26} {'mean max':>9} {'max max':>8} {'>=.15':>6} {'>=.50':>6} {'D15 rule':>8} {'mean union':>10}")
        for gn, s in summ.items():
            print(f"{gn:<26} {s['mean_max_frac']:>9} {s['max_max_frac']:>8} {s['n_ge_0.15']:>6} {s['n_ge_0.50']:>6} "
                  f"{s['n_d15_rule_hits']:>8} {s['mean_union_frac']:>10}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
