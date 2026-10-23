"""[D19] Materialize ContractNLI (Koreeda & Manning 2021; original Stanford release, CC BY 4.0) into the xenc
doc-QA schema.

607 NDAs x the same 17 hypotheses = 17 three-way MCQs per document (Entailed / Contradicted / Not mentioned),
with the OFFICIAL document-level split train 423 / dev 61 / test 123 (dev -> val). doc_id = contractnli_<id>.
`meta.evidence_char_spans` holds the gold evidence spans (character offsets into the document).
`meta.hyp_prior_label` is the majority label of that hypothesis on TRAIN: since every document is asked the same
17 hypotheses, "answer the per-hypothesis prior" is the closed-book ceiling to beat (report it next to closed_book).
The official split has near-duplicate NDAs across train and test (re-filed templates): val/test items carry
`meta.train_overlap` (max shared word-8-gram share with one train doc) and `meta.near_dup_train` (>= --dup_frac);
report test numbers on the `near_dup_train == False` subset too, and never draw wrong_doc from a near-duplicate.

  python -I scripts/data/prepare_contractnli.py [--raw_dir data/raw/contractnli/contract-nli] [--out_dir data]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from xenc.doc_tasks import load_qwen_tokenizer, print_summary, write_split  # noqa: E402

NAME = "contractnli"
OPTS = [("Entailment", "Entailed: the agreement states or implies this"),
        ("Contradiction", "Contradicted: the agreement states the opposite"),
        ("NotMentioned", "Not mentioned: the agreement does not address this")]
LAB = "ABC"


def _question(hyp: str) -> str:
    return ("Consider the following hypothesis about this non-disclosure agreement:\n"
            f"\"{hyp}\"\nWhat does the agreement say about the hypothesis?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(REPO / "data/raw/contractnli/contract-nli"))
    ap.add_argument("--out_dir", default=str(REPO / "data"))
    ap.add_argument("--dup_frac", type=float, default=0.5)
    args = ap.parse_args()
    raw = Path(args.raw_dir)
    data = {s: json.load(open(raw / f"{f}.json")) for s, f in (("train", "train"), ("val", "dev"), ("test", "test"))}
    labels = data["train"]["labels"]

    prior = {}
    for h in labels:
        c = Counter(d["annotation_sets"][0]["annotations"][h]["choice"] for d in data["train"]["documents"])
        prior[h] = c.most_common(1)[0][0]
    choices = [{"label": LAB[j], "text": t} for j, (_, t) in enumerate(OPTS)]
    lab_of = {k: LAB[j] for j, (k, _) in enumerate(OPTS)}

    docs, splits, seen = {}, {}, set()
    for split, d in data.items():
        out = []
        for doc in d["documents"]:
            did = f"{NAME}_{doc['id']:04d}"
            assert did not in seen, did
            seen.add(did)
            docs[did] = {"text": doc["text"], "title": doc["file_name"]}
            ann = doc["annotation_sets"][0]["annotations"]
            for h in sorted(labels, key=lambda x: int(x.split("-")[1])):
                a = ann[h]
                gl = lab_of[a["choice"]]
                out.append({
                    "key": f"{NAME}_{split}_{len(out):05d}", "doc_id": did,
                    "question": _question(labels[h]["hypothesis"]), "choices": choices,
                    "answers": [OPTS[LAB.index(gl)][1]], "correct_label": gl, "task": NAME, "hop": None,
                    "meta": {"hypothesis_id": h, "short_description": labels[h]["short_description"],
                             "nli_label": a["choice"], "hyp_prior_label": lab_of[prior[h]],
                             "evidence_char_spans": [doc["spans"][i] for i in a["spans"]],
                             "document_type": doc.get("document_type"), "orig_split": split},
                })
        splits[split] = out
    # The OFFICIAL split contains near-duplicate NDAs across train/test (the same filer's template re-filed; up to
    # 100% shared 8-grams). Keep the official split for comparability but flag every val/test document with its
    # max share of word-8-gram fingerprints (D15 fingerprints, relative to the smaller doc) found in ONE train doc.
    from d19_leakage import shingles
    train_ids = sorted({it["doc_id"] for it in splits["train"]})
    sh = {d: shingles(docs[d]["text"]) for d in docs}
    post: dict[int, list[str]] = {}
    for d in train_ids:
        for x in sh[d]:
            post.setdefault(x, []).append(d)
    ov = {}
    for s in ("val", "test"):
        for d in sorted({it["doc_id"] for it in splits[s]}):
            c = Counter(o for x in sh[d] for o in post.get(x, ()))
            ov[d] = max((k / max(1, min(len(sh[d]), len(sh[o]))) for o, k in c.items()), default=0.0)
        for it in splits[s]:
            it["meta"]["train_overlap"] = round(ov[it["doc_id"]], 4)
            it["meta"]["near_dup_train"] = ov[it["doc_id"]] >= args.dup_frac
    index = write_split(Path(args.out_dir), NAME, splits, docs, load_qwen_tokenizer())
    extra = {}
    for s, items in splits.items():
        extra[f"{s} label dist"] = dict(Counter(it["meta"]["nli_label"] for it in items))
        extra[f"{s} hypothesis-prior acc"] = round(
            sum(it["meta"]["hyp_prior_label"] == it["correct_label"] for it in items) / len(items), 4)
        if s != "train":
            nd = {it["doc_id"] for it in items if it["meta"]["near_dup_train"]}
            extra[f"{s} docs near-duplicate of a train doc (>= {args.dup_frac})"] = len(nd)
    print_summary(NAME, splits, index, extra)


if __name__ == "__main__":
    main()
