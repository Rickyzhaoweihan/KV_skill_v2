"""[D19] Wrong-document map for eval_d13.py --wrong_map that never pairs near-duplicates.

Same deterministic rule as eval_doc.wrong_doc_id (doc_order[(i + shift) % n] over the sorted doc ids of the split),
but steps on while the candidate shares >= --max_frac of the smaller document's word-8-gram fingerprints (D15
fingerprints) with the own document. ContractNLI needs this: re-filed NDA templates are up to 100% identical.

  python -I scripts/d19_wrong_map.py --split_dir data/contractnli_split --split test \
      --out outputs/D19_wrongmap_contractnli_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from d19_leakage import shingles  # noqa: E402


def overlap(a: set, b: set) -> float:
    return len(a & b) / max(1, min(len(a), len(b)))


def wrong_map(doc_order: list[str], sh: dict[str, set], shift: int = 7, max_frac: float = 0.15) -> dict:
    n, out = len(doc_order), {}
    for i, d in enumerate(doc_order):
        for s in range(shift, shift + n):
            j = doc_order[(i + s) % n]
            if j != d and overlap(sh[d], sh[j]) < max_frac:
                out[d] = j
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--shift", type=int, default=7)
    ap.add_argument("--max_frac", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sd = Path(a.split_dir)
    index = json.load(open(sd / "docs.json"))
    items = json.load(open(sd / a.split / "items.json"))
    order = sorted({it["doc_id"] for it in items})
    sh = {d: shingles((sd.parent / index[d]["path"]).read_text(encoding="utf-8", errors="ignore")) for d in order}
    m = wrong_map(order, sh, a.shift, a.max_frac)
    moved = sum(1 for i, d in enumerate(order) if m.get(d) != order[(i + a.shift) % len(order)])
    near = Counter(sum(overlap(sh[d], sh[o]) >= a.max_frac for o in order if o != d) for d in order)
    json.dump(m, open(a.out, "w"), indent=1)
    print(f"{len(m)}/{len(order)} docs mapped; {moved} moved off the default shift; "
          f"docs by #near-duplicates in split: {dict(sorted(near.items()))} -> {a.out}")


if __name__ == "__main__":
    main()
