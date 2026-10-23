"""Materialize QuALITY (emozilla/quality parquet) into the xenc doc-QA schema.

train -> train; validation -> 50/50 by ARTICLE into val/test (seed 42; all questions of an article
stay together). doc_id = quality_<sha1(article)[:12]>. 4-way MCQ; `answer` is ZERO-indexed.

  python scripts/data/prepare_quality.py [--raw_dir data/raw_downloads/quality] [--out_dir data]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import random
import sys
from collections import OrderedDict
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

LAB = "ABCDE"
NAME = "quality"


def _parquet(raw_dir: Path, split: str) -> str:
    hits = sorted(glob.glob(str(raw_dir / "data" / f"{split}-*.parquet")))
    if hits:
        return hits[0]
    from huggingface_hub import snapshot_download  # fallback: HF cache copy
    snap = Path(snapshot_download("emozilla/quality", repo_type="dataset", local_files_only=True,
                                  cache_dir="/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home/hub"))
    return sorted(glob.glob(str(snap / "data" / f"{split}-*.parquet")))[0]


def _title(article: str) -> str:
    for ln in article.splitlines():
        if ln.strip():
            return ln.strip()[:120]
    return ""


def _group(rows):
    arts: OrderedDict[str, dict] = OrderedDict()
    for r in rows:
        did = f"{NAME}_{hashlib.sha1(r['article'].encode()).hexdigest()[:12]}"
        arts.setdefault(did, {"text": r["article"], "rows": []})["rows"].append(r)
    return arts


def _items(arts, split, orig_split):
    out = []
    for did, a in arts.items():
        for qi, r in enumerate(a["rows"]):
            opts = [str(o).strip() for o in r["options"]]
            gi = int(r["answer"])
            assert 0 <= gi < len(opts)
            out.append({
                "key": f"{NAME}_{split}_{len(out):04d}", "doc_id": did, "question": r["question"].strip(),
                "choices": [{"label": LAB[j], "text": o} for j, o in enumerate(opts)],
                "answers": [opts[gi]], "correct_label": LAB[gi], "task": NAME, "hop": None,
                "meta": {"hard": bool(r["hard"]), "orig_split": orig_split, "q_index_in_doc": qi},
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(REPO / "data/raw_downloads/quality"))
    ap.add_argument("--out_dir", default=str(REPO / "data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    raw = Path(args.raw_dir)

    tr = _group(pq.read_table(_parquet(raw, "train")).to_pylist())
    va = _group(pq.read_table(_parquet(raw, "validation")).to_pylist())
    assert not set(tr) & set(va), "article shared between HF train and validation"
    vids = sorted(va)
    random.Random(args.seed).shuffle(vids)
    half = len(vids) // 2
    val_ids, test_ids = set(vids[:half]), set(vids[half:])
    splits = {
        "train": _items(tr, "train", "train"),
        "val": _items(OrderedDict((k, v) for k, v in va.items() if k in val_ids), "val", "validation"),
        "test": _items(OrderedDict((k, v) for k, v in va.items() if k in test_ids), "test", "validation"),
    }
    docs = {did: {"text": a["text"], "title": _title(a["text"])} for d in (tr, va) for did, a in d.items()}
    index = write_split(Path(args.out_dir), NAME, splits, docs, load_qwen_tokenizer())
    hard = {s: sum(it["meta"]["hard"] for it in v) for s, v in splits.items()}
    print_summary(NAME, splits, index, {"hard Qs": hard})


if __name__ == "__main__":
    main()
