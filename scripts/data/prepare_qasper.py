"""[D22] Materialize Qasper (Dasigi et al. 2021, NLP papers; CC BY 4.0) into the xenc doc-QA schema.

Official PAPER-level split (train 888 / dev 281 -> val / test 416), so documents are disjoint across splits.
Document = "# title / ## Abstract / ## <section> paragraphs" (figures and tables dropped).
Kept questions: answerable by the majority of annotators, with SHORT gradable answers only:
  extractive  spans joined by " " (all annotators' short extractive answers are gold aliases)
  boolean     "yes" / "no"
Free-form (abstractive) answers and answers longer than --max_words are dropped (token F1 on long free text is noise).
meta.atype in {extractive, boolean}; meta.numeric = the gold is a number (scored with D22's numeric normalizer too).

  python -I scripts/data/prepare_qasper.py [--raw_dir data/raw_downloads/qasper/s3] [--out_dir data]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from xenc.doc_tasks import load_qwen_tokenizer, print_summary, write_split  # noqa: E402

NAME = "qasper"
NUM = re.compile(r"^[\s$€£~≈<>]*[-+]?\d[\d,]*(\.\d+)?\s*(%|percent|k|m|million|billion)?\s*$", re.I)


def doc_text(p: dict) -> str:
    out = [f"# {p['title'].strip()}", "", "## Abstract", p["abstract"].strip(), ""]
    for sec in p["full_text"]:
        name = (sec.get("section_name") or "").strip()
        paras = [x.strip() for x in sec.get("paragraphs") or [] if x and x.strip()]
        if not paras:
            continue
        out += [f"## {name}" if name else "##", *[q + "\n" for q in paras]]
    return "\n".join(out).strip() + "\n"


def answer_of(a: dict, max_words: int):
    """(type, text) for one annotator's answer, or None (unanswerable / free-form / too long)."""
    a = a["answer"]
    if a.get("unanswerable"):
        return ("unanswerable", None)
    if a.get("yes_no") is not None:
        return ("boolean", "yes" if a["yes_no"] else "no")
    spans = [s.strip() for s in a.get("extractive_spans") or [] if s and s.strip()]
    if spans:
        t = " ".join(spans)
        return ("extractive", t) if len(t.split()) <= max_words else ("long", None)
    return ("abstractive", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(REPO / "data/raw_downloads/qasper/s3"))
    ap.add_argument("--out_dir", default=str(REPO / "data"))
    ap.add_argument("--max_words", type=int, default=20)
    args = ap.parse_args()
    raw = Path(args.raw_dir)
    files = {"train": "qasper-train-v0.3.json", "val": "qasper-dev-v0.3.json", "test": "qasper-test-v0.3.json"}
    docs, splits, drop = {}, {}, Counter()
    for split, fn in files.items():
        data = json.load(open(raw / fn))
        out = []
        for pid, p in sorted(data.items()):
            did = f"{NAME}_{pid.replace('.', '_')}"
            assert did not in docs, did
            items = []
            for qa in p["qas"]:
                ans = [answer_of(a, args.max_words) for a in qa["answers"]]
                n_unans = sum(t == "unanswerable" for t, _ in ans)
                if n_unans * 2 >= len(ans) or not ans:
                    drop["mostly_unanswerable"] += 1
                    continue
                good = [(t, x) for t, x in ans if x is not None]
                if not good:
                    drop["no_short_answer"] += 1
                    continue
                atype = Counter(t for t, _ in good).most_common(1)[0][0]
                gold = list(dict.fromkeys(x for t, x in good if t == atype))
                items.append({
                    "key": f"{NAME}_{qa['question_id'][:16]}", "doc_id": did, "question": qa["question"].strip(),
                    "choices": None, "answers": gold, "correct_label": None, "task": NAME, "hop": None,
                    "meta": {"atype": atype, "numeric": bool(NUM.match(gold[0])), "paper_id": pid,
                             "n_annotators": len(ans), "orig_split": split}})
            if items:
                docs[did] = {"text": doc_text(p), "title": p["title"].strip()}
                out += items
        splits[split] = out
    keys = [it["key"] for s in splits.values() for it in s]
    assert len(keys) == len(set(keys)), "duplicate item keys"
    index = write_split(Path(args.out_dir), NAME, splits, docs, load_qwen_tokenizer())
    extra = {"dropped": dict(drop)}
    for s, items in splits.items():
        extra[f"{s} atype"] = dict(Counter(it["meta"]["atype"] for it in items))
        extra[f"{s} numeric"] = sum(it["meta"]["numeric"] for it in items)
    print_summary(NAME, splits, index, extra)


if __name__ == "__main__":
    main()
