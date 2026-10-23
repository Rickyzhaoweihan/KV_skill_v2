"""[D19] Materialize LooGLE (bigai-nlco/LooGLE, CC BY-SA 4.0) into the xenc doc-QA schema.

The HF release is one row per question with the document repeated; documents are grouped by sha1(context)
(doc_id = loogle_<sha1[:12]>). Three question sources share documents:
  shortdep_qa    (Wikipedia, GPT-3.5 written + human checked): span; answers are short sentences -> token F1
  longdep_qa     (human written; comprehension / computation / timeline / multi-retrieval): span, except items whose
                 question lists numbered options and whose answer is one option number -> converted to MCQ, and
                 timeline-reorder items ("2,1,3") -> qtype "order" with comma/space variants as answers (grade EM)
  shortdep_cloze (movie/TV scripts): a summary of a script segment with named entities masked; ONE item per row
                 asking for <mask-0> (other masks stay masked) -> short entity span (EM/F1)
There is no official train/test split (the release is test-only): documents are split 65/10/25 train/val/test by
document (seed 42), all questions of a document stay together, and near-duplicate documents (two renderings of the
same screenplay; >= 30% shared word-8-gram fingerprints) are clustered and kept in the same split.

  python -I scripts/data/prepare_loogle.py [--raw_dir data/raw/loogle] [--out_dir data]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from xenc.doc_tasks import load_qwen_tokenizer, print_summary, write_split  # noqa: E402

NAME = "loogle"
LAB = "ABCDE"
OPT_RE = re.compile(r"\n\s*(\d)\.\s*([^\n]+)")


def _mcq(question: str, answer: str):
    """Numbered-option longdep question with a single-number answer -> (stem, options, gold index) or None."""
    opts = OPT_RE.findall(question)
    a = answer.strip().rstrip(".")
    if not (2 <= len(opts) <= 5 and a.isdigit()):
        return None
    if [int(n) for n, _ in opts] != list(range(1, len(opts) + 1)) or not 1 <= int(a) <= len(opts):
        return None
    stem = question[:question.index(OPT_RE.search(question).group(0))].strip()
    return stem, [t.strip() for _, t in opts], int(a) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(REPO / "data/raw/loogle"))
    ap.add_argument("--out_dir", default=str(REPO / "data"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.25)
    ap.add_argument("--dup_frac", type=float, default=0.3)
    args = ap.parse_args()
    raw = Path(args.raw_dir) / "data"

    docs: OrderedDict[str, dict] = OrderedDict()
    per_doc: dict[str, list[dict]] = {}
    for src in ("shortdep_qa", "longdep_qa", "shortdep_cloze"):
        for ln in open(raw / f"{src}.jsonl"):
            r = json.loads(ln)
            did = f"{NAME}_{hashlib.sha1(r['context'].encode()).hexdigest()[:12]}"
            docs.setdefault(did, {"text": r["context"], "title": str(r.get("title") or "")[:120]})
            q, a = r["question"].strip(), r["answer"]
            meta = {"source": src, "orig_id": r["id"], "title": r.get("title")}
            if src == "shortdep_cloze":
                ans = a if isinstance(a, list) else (ast.literal_eval(a) if str(a).startswith("[") else [a])
                if "<mask-0>" not in q or not ans or not str(ans[0]).strip():
                    continue
                item = {"question": ("Below is a summary of part of the document in which named entities are "
                                     "replaced by <mask-k> placeholders. Which entity is <mask-0>?\n\n" + q),
                        "choices": None, "answers": [str(ans[0]).strip()], "correct_label": None,
                        "meta": {**meta, "qtype": "cloze", "all_masks": [str(x) for x in ans]}}
            else:
                a = str(a).strip()
                if not a:
                    continue
                m = _mcq(q, a) if src == "longdep_qa" else None
                order = re.fullmatch(r"\d+(\s*,\s*\d+)+\.?", a)
                if order:  # timeline reorder: "2,1,3" -> accept "2,1,3" / "2, 1, 3" / "2 1 3" (use EM, not F1)
                    seq = re.findall(r"\d+", a)
                    item = {"question": q, "choices": None,
                            "answers": [",".join(seq), ", ".join(seq), " ".join(seq)], "correct_label": None,
                            "meta": {**meta, "qtype": "order", "evidence": r.get("evidence")}}
                elif m:
                    stem, opts, gi = m
                    item = {"question": stem, "choices": [{"label": LAB[j], "text": o} for j, o in enumerate(opts)],
                            "answers": [opts[gi]], "correct_label": LAB[gi], "meta": {**meta, "qtype": "mcq"}}
                else:
                    item = {"question": q, "choices": None, "answers": [a], "correct_label": None,
                            "meta": {**meta, "qtype": "span", "evidence": r.get("evidence")}}
            per_doc.setdefault(did, []).append(item)

    # Some films appear twice as different renderings of the same screenplay (e.g. "the-woman-king-2022" and
    # "The Woman King", 99.8% shared 8-grams): cluster documents sharing >= --dup_frac of the smaller one's word-8-gram
    # fingerprints (D15 fingerprints) and split by CLUSTER so near-duplicates never straddle train/test.
    from d19_leakage import shingles
    ids = sorted(per_doc)
    sh = {d: shingles(docs[d]["text"]) for d in ids}
    post: dict[int, list[str]] = {}
    for d in ids:
        for x in sh[d]:
            post.setdefault(x, []).append(d)
    parent = {d: d for d in ids}

    def find(d):
        while parent[d] != d:
            parent[d] = parent[parent[d]]
            d = parent[d]
        return d
    for d in ids:
        c = Counter(o for x in sh[d] for o in post[x] if o != d)
        for o, k in c.items():
            if k >= args.dup_frac * max(1, min(len(sh[d]), len(sh[o]))):
                parent[find(d)] = find(o)
    clusters: dict[str, list[str]] = {}
    for d in ids:
        clusters.setdefault(find(d), []).append(d)
    roots = sorted(clusters)
    random.Random(args.seed).shuffle(roots)
    n_te, n_va = round(args.test_frac * len(ids)), round(args.val_frac * len(ids))
    part, acc = {"test": [], "val": [], "train": []}, 0
    for r in roots:
        s = "test" if acc < n_te else "val" if acc < n_te + n_va else "train"
        part[s] += clusters[r]
        acc += len(clusters[r])
    n_dup = sum(len(v) for v in clusters.values() if len(v) > 1)
    splits = {}
    for s in ("train", "val", "test"):
        out = []
        for did in sorted(part[s]):
            for it in per_doc[did]:
                out.append({"key": f"{NAME}_{s}_{len(out):05d}", "doc_id": did, "question": it["question"],
                            "choices": it["choices"], "answers": it["answers"], "correct_label": it["correct_label"],
                            "task": NAME, "hop": None, "meta": it["meta"]})
        splits[s] = out
    docs = OrderedDict((d, docs[d]) for d in docs if d in per_doc)
    index = write_split(Path(args.out_dir), NAME, splits, docs, load_qwen_tokenizer())
    extra = {f"{s} qtypes": dict(Counter(it["meta"]["qtype"] for it in v)) for s, v in splits.items()}
    extra["near-duplicate docs kept together"] = f"{n_dup} docs in {sum(len(v) > 1 for v in clusters.values())} clusters"
    print_summary(NAME, splits, index, extra)


if __name__ == "__main__":
    main()
