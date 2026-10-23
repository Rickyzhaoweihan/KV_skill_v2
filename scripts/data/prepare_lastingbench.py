"""Materialize LastingBench (LongBench-v1 originals + `_antifact` counterfactual twins).

Each original item is one document (its full `context`) with one question. Its counterfactual twin
(same `_id` in `<subset>_antifact.jsonl`) gets doc_id `<orig doc_id>_cf`, `meta.counterfactual=True`,
`meta.rewritten` = (twin context != original context), and `meta.answers_original` (the original gold).
Twins are written as separate docs even when not rewritten (identical text) so pairing is uniform.
Split 40/10/50 by ORIGINAL item, per subset, seed 42; twins always land in the same split.

  python scripts/data/prepare_lastingbench.py [--raw_dir data/raw_downloads/lastingbench] [--out_dir data]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from xenc.doc_tasks import load_qwen_tokenizer, print_summary, write_split  # noqa: E402

NAME = "lastingbench"
SUBSETS = ["hotpotqa", "2wikimqa", "musique", "multifieldqa_en"]
FRAC = (0.4, 0.1)  # train, val; rest test


def _rows(p: Path):
    with open(p) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def _answers(a):
    a = a if isinstance(a, list) else [a]
    return [str(x).strip() for x in a if str(x).strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(REPO / "data/raw_downloads/lastingbench"))
    ap.add_argument("--out_dir", default=str(REPO / "data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    raw = Path(args.raw_dir) / "data"

    splits = {"train": [], "val": [], "test": []}
    docs = {}
    rewritten = {}
    for sub in SUBSETS:
        orig = _rows(raw / f"{sub}.jsonl")
        cf = {r["_id"]: r for r in _rows(raw / f"{sub}_antifact.jsonl")}
        assert len(cf) == len(orig) and all(r["_id"] in cf for r in orig), sub
        ids = [r["_id"] for r in orig]
        assert len({i[:16] for i in ids}) == len(ids), f"_id prefix collision in {sub}"
        order = list(ids)
        random.Random(f"{args.seed}-{sub}").shuffle(order)
        n_tr, n_va = round(FRAC[0] * len(order)), round(FRAC[1] * len(order))
        split_of = {i: ("train" if k < n_tr else "val" if k < n_tr + n_va else "test") for k, i in enumerate(order)}
        rewritten[sub] = 0
        for r in orig:
            t = cf[r["_id"]]
            assert t["input"] == r["input"]
            base = f"{NAME}_{sub}_{r['_id'][:16]}"
            rw = t["context"] != r["context"]
            rewritten[sub] += rw
            a_orig, a_cf = _answers(r["answers"]), _answers(t["answers"])
            title = f"{sub}: {r['input'].strip()[:100]}"
            docs[base] = {"text": r["context"], "title": title}
            docs[base + "_cf"] = {"text": t["context"], "title": title + " [cf]"}
            common = {"subset": sub, "source_id": r["_id"], "pair_key": base, "length_words": r.get("length")}
            sp = split_of[r["_id"]]
            splits[sp].append({
                "key": base, "doc_id": base, "question": r["input"].strip(), "choices": None,
                "answers": a_orig, "correct_label": None, "task": NAME, "hop": None,
                "meta": {**common, "counterfactual": False, "rewritten": False},
            })
            splits[sp].append({
                "key": base + "_cf", "doc_id": base + "_cf", "question": t["input"].strip(), "choices": None,
                "answers": a_cf, "correct_label": None, "task": NAME, "hop": None,
                "meta": {**common, "counterfactual": True, "rewritten": rw, "answers_original": a_orig,
                         "answer_changed": a_cf != a_orig},
            })
    index = write_split(Path(args.out_dir), NAME, splits, docs, load_qwen_tokenizer())
    per_sub = {}
    for sub in SUBSETS:
        toks = sorted(m["n_tokens"] for d, m in index.items() if d.startswith(f"{NAME}_{sub}_") and d.endswith("_cf"))
        per_sub[sub] = f"rewritten {rewritten[sub]}/{len(toks)}  cf-doc tok med {toks[len(toks) // 2]} max {toks[-1]}"
    print_summary(NAME, splits, index, per_sub)


if __name__ == "__main__":
    main()
