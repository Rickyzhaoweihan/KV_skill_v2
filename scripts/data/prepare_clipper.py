"""[D19] Materialize CLIPPER (chtmp223/CLIPPER, Apache-2.0; Pham et al. 2025) into the xenc doc-QA schema.

Public-domain Gutenberg novels (17k-127k o200k tokens) with Claude-written TRUE/FALSE claim PAIRS (a true claim and
a minimally corrupted false one, "single"-chapter or "multiple"-chapter scope). Each pair yields TWO 2-way MCQ items
(A True / B False) sharing `meta.pair_id`, so pair accuracy (both right) can be reported as in the paper.
Splits by BOOK: test = the official test books (held out of train/dev upstream); val = `--n_val` books drawn from
the train+dev books (seed 42); train = the remaining train+dev books (upstream dev is a claim-level split over the
train books, so it cannot be used as-is). Pairs are de-duplicated. Uses whatever parquet shards are present
(D19 downloaded test, dev and 3/12 train shards).

  python -I scripts/data/prepare_clipper.py [--raw_dir data/raw/clipper] [--out_dir data]
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
DEPS = "/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/SoftSkill/.deps_qwen35"
try:
    import pyarrow.parquet as pq
except ImportError:
    sys.path.append(DEPS)
    import pyarrow.parquet as pq

from xenc.doc_tasks import load_qwen_tokenizer, print_summary, write_split  # noqa: E402

NAME = "clipper"
COLS = ["id", "book_name", "book_title", "book_author", "book_length", "claim_type", "true_claim", "false_claim"]
CHOICES = [{"label": "A", "text": "True"}, {"label": "B", "text": "False"}]


def _book_text(user_msg: str) -> str:
    t = user_msg.split("<context>", 1)[1].rsplit("</context>", 1)[0]
    return t.lstrip("﻿").strip() + "\n"


def _read(files):
    """Yield (row-without-messages, book_text or None) streaming in small batches."""
    for f in files:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=16, columns=COLS + ["messages"]):
            for r in b.to_pylist():
                msgs = r.pop("messages")
                user = next(m["content"] for m in msgs if m["role"] == "user")
                yield r, user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(REPO / "data/raw/clipper"))
    ap.add_argument("--out_dir", default=str(REPO / "data"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_val", type=int, default=40)
    args = ap.parse_args()
    raw = Path(args.raw_dir) / "data"

    books: OrderedDict[str, dict] = OrderedDict()
    pairs: dict[str, OrderedDict] = {}          # book -> {(true, false): row}
    origin: dict[str, set] = {}
    for up in ("test", "dev", "train"):
        files = sorted(glob.glob(str(raw / f"{up}-*.parquet")))
        for r, user in _read(files):
            bn = r["book_name"]
            if bn not in books:
                books[bn] = {"text": _book_text(user), "title": f"{r['book_title']} — {r['book_author']}",
                             "o200k": r["book_length"]}
            origin.setdefault(bn, set()).add(up)
            pairs.setdefault(bn, OrderedDict()).setdefault((r["true_claim"], r["false_claim"]), r)
    test_b = sorted(b for b, o in origin.items() if "test" in o)
    assert all(origin[b] == {"test"} for b in test_b), "a test book also appears in train/dev"
    rest = sorted(b for b in origin if b not in set(test_b))
    random.Random(args.seed).shuffle(rest)
    part = {"test": test_b, "val": sorted(rest[:args.n_val]), "train": sorted(rest[args.n_val:])}

    splits = {}
    for s, bl in (("train", part["train"]), ("val", part["val"]), ("test", part["test"])):
        out = []
        for bn in bl:
            did = f"{NAME}_{bn}"
            for k, ((tc, fc), r) in enumerate(pairs[bn].items()):
                pid = f"{bn}_{k:03d}"
                for claim, truth in ((tc, True), (fc, False)):
                    out.append({
                        "key": f"{NAME}_{s}_{len(out):05d}", "doc_id": did,
                        "question": f"Is the following statement about the story true or false?\n\nStatement: {claim}",
                        "choices": CHOICES, "answers": ["True" if truth else "False"],
                        "correct_label": "A" if truth else "B", "task": NAME, "hop": None,
                        "meta": {"pair_id": pid, "truth": truth, "claim_type": r["claim_type"],
                                 "upstream_id": r["id"], "book_title": r["book_title"],
                                 "book_author": r["book_author"], "book_o200k": r["book_length"],
                                 "upstream_splits": sorted(origin[bn])},
                    })
        splits[s] = out
    docs = OrderedDict((f"{NAME}_{bn}", {"text": books[bn]["text"], "title": books[bn]["title"]}) for bn in books)
    index = write_split(Path(args.out_dir), NAME, splits, docs, load_qwen_tokenizer())
    print_summary(NAME, splits, index, {
        "pairs per split": {s: len(v) // 2 for s, v in splits.items()},
        "claim_type": {s: dict(Counter(it["meta"]["claim_type"] for it in v)) for s, v in splits.items()},
        "train shards used": len(glob.glob(str(raw / "train-*.parquet"))),
    })


if __name__ == "__main__":
    main()
