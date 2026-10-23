"""Materialize MuSiQue-Ans dev (2,417 answerable items) into the xenc doc-QA schema.

The DOCUMENT is the item's 17–20 paragraphs concatenated as `[P{i}] {title}\n{text}` blocks
(blank line between). `meta.supporting_idx` = gold paragraph indices (`is_supporting`), `meta.para_spans`
= [start_char, end_char) of each block in the document, `hop` = int from the id prefix (`3hop1` -> 3).
Split 40/10/50 by item, seed 42. Capped at the dev set (train is not materialized).

  python scripts/data/prepare_musique.py [--raw_dir data/raw_downloads/musique] [--out_dir data]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from xenc.doc_tasks import load_qwen_tokenizer, print_summary, write_split  # noqa: E402

NAME = "musique"
FRAC = (0.4, 0.1)


def build_doc(paragraphs):
    parts, spans, pos = [], [], 0
    for i, p in enumerate(paragraphs):
        block = f"[P{i}] {p['title'].strip()}\n{p['paragraph_text'].strip()}"
        spans.append([pos, pos + len(block)])
        parts.append(block)
        pos += len(block) + 2
    return "\n\n".join(parts), spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(REPO / "data/raw_downloads/musique"))
    ap.add_argument("--out_dir", default=str(REPO / "data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = []
    with open(Path(args.raw_dir) / "musique_ans_v1.0_dev.jsonl") as f:  # streamed line by line
        for ln in f:
            if ln.strip():
                rows.append(json.loads(ln))
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    n_tr, n_va = round(FRAC[0] * len(rows)), round(FRAC[1] * len(rows))
    split_of = {j: ("train" if k < n_tr else "val" if k < n_tr + n_va else "test") for k, j in enumerate(order)}

    splits = {"train": [], "val": [], "test": []}
    docs = {}
    for j, r in enumerate(rows):
        paras = sorted(r["paragraphs"], key=lambda p: p["idx"])
        assert [p["idx"] for p in paras] == list(range(len(paras)))
        text, spans = build_doc(paras)
        did = f"{NAME}_{r['id']}"
        assert did not in docs
        docs[did] = {"text": text, "title": r["question"].strip()[:120]}
        hop_type = r["id"].split("__")[0]
        answers = [r["answer"].strip()] + [a.strip() for a in r.get("answer_aliases") or [] if a.strip()]
        splits[split_of[j]].append({
            "key": did, "doc_id": did, "question": r["question"].strip(), "choices": None,
            "answers": list(dict.fromkeys(answers)), "correct_label": None, "task": NAME,
            "hop": int(hop_type[0]),
            "meta": {
                "hop_type": hop_type, "n_paragraphs": len(paras),
                "supporting_idx": [p["idx"] for p in paras if p["is_supporting"]],
                "para_spans": spans, "para_titles": [p["title"] for p in paras],
                "decomposition": r.get("question_decomposition"), "answerable": r.get("answerable", True),
            },
        })
    index = write_split(Path(args.out_dir), NAME, splits, docs, load_qwen_tokenizer())
    print_summary(NAME, splits, index, {
        s: dict(sorted(Counter(it["hop"] for it in v).items())) for s, v in splits.items()
    } | {"paragraphs/item": dict(sorted(Counter(len(r["paragraphs"]) for r in rows).items()))})


if __name__ == "__main__":
    main()
