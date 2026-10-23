"""[DS Phase B] Materialize NarrativeQA (Kocisky et al. 2018) OFFICIAL TRAIN stories as a TRAIN-ONLY split `narrativeqa`
in the xenc doc-QA schema (span task, token F1 against the two reference answers, like LooGLE shortdep_qa).

Inputs: documents.csv / qaps.csv from the narrativeqa-master release (data/raw_downloads/narrativeqa/x/...) and the
HF `deepmind/narrativeqa` data/narrativeqa_full_text.zip (<document_id>.content, raw Gutenberg text or HTML script).
Document = the story cropped to [story_start, story_end] (as the release defines), HTML tags stripped (movie scripts),
blank-line runs squeezed. Kept: set == train, cropped length in [--min_words, --max_words] (default 3k-68k words,
about 4k-92k Qwen tokens, inside the 100k operator budget). Questions: every train question of the story, answers =
[answer1, answer2] (deduplicated).

NarrativeQA's own test stories are what LongBench / InfiniteBench reuse, and we take only `train`. Every story is
still checked with scripts/ds_leakage.py (strict: word-8-gram containment >= .15 either way vs EVERY val / test /
D22 document of every benchmark split, e.g. CLIPPER / QuALITY Gutenberg books, LooGLE scripts, Shakespeare plays;
text hash; normalized-question match). Failing stories are skipped; the first --n passing stories of a seed-42
shuffle are written.

    python -I scripts/data/prepare_narrativeqa_train.py --zip <narrativeqa_full_text.zip> [--n 300]
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from xenc.doc_tasks import load_qwen_tokenizer, print_summary, write_split  # noqa: E402

NAME = "narrativeqa"
_TAG = re.compile(r"<[^>]+>")


def _pat(phrase: str) -> str:
    return r"\s*".join(re.escape(x) for x in phrase.split())


def clean(raw: str, kind: str, start: str, end: str) -> str:
    t = raw
    if kind == "movie" or "<html" in t[:2000].lower() or "<pre" in t.lower():
        t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", t)
        t = html.unescape(_TAG.sub(" ", t))
    # story_start / story_end are TOKENIZED ("eBooks . ***FINIS***"): match token by token across any whitespace
    ms = re.search(_pat(start), t) if start.strip() else None
    me = None
    if end.strip():
        for me in re.finditer(_pat(end), t):
            pass
    if ms and me and me.end() > ms.start():
        t = t[ms.start():me.end()]
    t = t.replace("\r", "").replace("\xa0", " ")
    t = "\n".join(ln.rstrip() for ln in t.split("\n"))
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--meta_dir", default=str(REPO / "data/raw_downloads/narrativeqa/x/narrativeqa-master"))
    ap.add_argument("--out_dir", default=str(REPO / "data"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--min_words", type=int, default=3000)
    ap.add_argument("--max_words", type=int, default=68000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--ref_data_root", default="data", help="leakage reference: data/*_split (smoke tests only change it)")
    ap.add_argument("--ref_d22", default="outputs/D22_sets")
    args = ap.parse_args()
    import os
    os.chdir(REPO)
    from ds_leakage import Reference
    R = Reference(data_root=args.ref_data_root, d22_dir=args.ref_d22, thr=args.thr)
    meta = {r["document_id"]: r for r in csv.DictReader(open(Path(args.meta_dir) / "documents.csv"))}
    qaps: dict = {}
    for r in csv.DictReader(open(Path(args.meta_dir) / "qaps.csv")):
        if r["set"] == "train":
            qaps.setdefault(r["document_id"], []).append(r)
    ids = sorted(d for d, m in meta.items() if m["set"] == "train" and d in qaps)
    random.Random(args.seed).shuffle(ids)
    Z = zipfile.ZipFile(args.zip)
    names = set(Z.namelist())
    docs, items, why, n_seen, qdrop = {}, [], Counter(), 0, 0
    for d in ids:
        if len(docs) >= args.n:
            break
        m = meta[d]
        if f"{d}.content" not in names:
            why["no_text"] += 1
            continue
        n_seen += 1
        raw = Z.read(f"{d}.content").decode("utf-8", errors="replace")
        text = clean(raw, m["kind"], m["story_start"], m["story_end"])
        nw = len(text.split())
        if not (args.min_words <= nw <= args.max_words):
            why["length"] += 1
            continue
        did = f"{NAME}_{d}"
        reason = R.check(did, text)
        if reason:
            why[reason.split("=")[0]] += 1
            continue
        its = []
        for k, q in enumerate(qaps[d]):
            if R.q_hit(q["question"]):
                qdrop += 1
                continue
            ans = list(dict.fromkeys(a.strip() for a in (q["answer1"], q["answer2"]) if a and a.strip()))
            if not ans or not q["question"].strip():
                continue
            its.append({"key": f"{did}_q{k}", "doc_id": did, "question": q["question"].strip(), "choices": None,
                        "answers": ans, "correct_label": None, "task": NAME, "hop": None,
                        "meta": {"kind": m["kind"], "wiki_title": m["wiki_title"], "story_words": nw,
                                 "source": "narrativeqa train"}})
        if not its:
            continue
        docs[did] = {"text": text, "title": m["wiki_title"]}
        items += its
    print(f"[narrativeqa] {len(ids)} train stories; scanned {n_seen}; kept {len(docs)} stories / {len(items)} questions; "
          f"skips {dict(why)}; question-level drops {qdrop}; kinds {dict(Counter(meta[k.split('_', 1)[1]]['kind'] for k in docs))}")
    index = write_split(Path(args.out_dir), NAME, {"train": items, "val": [], "test": []}, docs, load_qwen_tokenizer())
    print_summary(NAME, {"train": items}, index)
    json.dump({"stories": len(ids), "scanned": n_seen, "kept_docs": len(docs), "kept_items": len(items), "skips": why,
               "question_drops": qdrop, "thr": args.thr},
              open(Path(args.out_dir) / f"{NAME}_split" / "leakage_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
