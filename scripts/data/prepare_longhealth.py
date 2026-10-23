"""Materialize LongHealth (benchmark_v5.json) into the xenc doc-QA schema.

20 patients = 20 documents (header + all clinical texts), 20 five-way MCQs each. Too few docs for a
held-out split, so patients are cross-validated: 5 folds of 4 patients (seed 42) written to
`longhealth_split/folds.json`; every split's items.json holds ALL 400 items, each with `meta.fold`.
Two raw questions list the correct option text twice; both labels go in `meta.accept_labels`.

  python scripts/data/prepare_longhealth.py [--raw_dir data/raw_downloads/longhealth] [--out_dir data]
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

LAB = "ABCDE"
NAME = "longhealth"
N_FOLDS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(REPO / "data/raw_downloads/longhealth"))
    ap.add_argument("--out_dir", default=str(REPO / "data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    raw = json.load(open(Path(args.raw_dir) / "data" / "benchmark_v5.json"))

    pids = sorted(raw)
    order = list(pids)
    random.Random(args.seed).shuffle(order)
    per = len(order) // N_FOLDS
    folds = [sorted(order[i * per:(i + 1) * per]) for i in range(N_FOLDS)]
    fold_of = {p: f for f, ps in enumerate(folds) for p in ps}

    docs, items = {}, []
    for pid in pids:
        p = raw[pid]
        did = f"{NAME}_{pid}"
        texts = p["texts"]
        texts = [texts[k] for k in sorted(texts, key=lambda k: int(k.split("_")[-1]))] if isinstance(texts, dict) else texts
        header = f"# Clinical record: {p['name']} (DOB {p['birthday']}), diagnosis: {p['diagnosis']}\n\n"
        docs[did] = {"text": header + "\n\n---\n\n".join(t.strip() for t in texts),
                     "title": f"{p['name']} — {p['diagnosis']}"}
        for q in p["questions"]:
            opts = [str(q[f"answer_{c}"]).strip() for c in "abcde" if q.get(f"answer_{c}") is not None]
            corr = str(q["correct"]).strip()
            labs = [LAB[j] for j, o in enumerate(opts) if o == corr]
            assert labs, (pid, q["No"])
            meta = {"fold": fold_of[pid], "patient": pid, "q_no": q["No"],
                    "answer_location": q.get("answer_location")}
            if len(labs) > 1:
                meta["accept_labels"] = labs
            items.append({
                "key": f"{NAME}_{pid}_q{q['No']:02d}", "doc_id": did, "question": q["question"].strip(),
                "choices": [{"label": LAB[j], "text": o} for j, o in enumerate(opts)],
                "answers": [opts[LAB.index(labs[0])]], "correct_label": labs[0], "task": NAME, "hop": None,
                "meta": meta,
            })
    splits = {s: items for s in ("train", "val", "test")}
    out = Path(args.out_dir)
    index = write_split(out, NAME, splits, docs, load_qwen_tokenizer())
    json.dump({"n_folds": N_FOLDS, "seed": args.seed,
               "folds": [[f"{NAME}_{p}" for p in ps] for ps in folds]},
              open(out / f"{NAME}_split" / "folds.json", "w"), indent=1)
    print_summary(NAME, splits, index, {
        "folds (doc_ids)": [[p.split("_")[-1] for p in ps] for ps in folds],
        "items with duplicate correct option": sum("accept_labels" in it["meta"] for it in items),
    })


if __name__ == "__main__":
    main()
