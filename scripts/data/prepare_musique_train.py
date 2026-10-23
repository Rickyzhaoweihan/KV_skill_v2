"""[DS Phase B] Materialize extra MuSiQue-Ans items from the OFFICIAL TRAIN file (19,938 items) as a TRAIN-ONLY split
`musiquetr`, built exactly like data/musique_split (prepare_musique.build_doc: the item's 20 paragraphs as
`[P{i}] {title}\\n{text}` blocks; answers = answer + aliases; meta as prepare_musique).

Our musique_split (val / test / D22-judged) comes from MuSiQue DEV, so the official train file is disjoint by id. Every
candidate is still checked with scripts/ds_leakage.py (strict: word-8-gram containment >= .15 either way vs EVERY
val / test / D22 document of every benchmark split, text hash, and normalized-question match vs every val / test
question); failures are skipped. The first --n passing items of a seed-42 shuffle are written (compact on turbo).

    python -I scripts/data/prepare_musique_train.py --raw <musique_ans_v1.0_train.jsonl> [--n 4000]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "scripts"), str(REPO / "scripts" / "data")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from prepare_musique import build_doc  # noqa: E402
from xenc.doc_tasks import load_qwen_tokenizer, print_summary, write_split  # noqa: E402

NAME = "musiquetr"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out_dir", default=str(REPO / "data"))
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--ref_data_root", default="data", help="leakage reference: data/*_split (smoke tests only change it)")
    ap.add_argument("--ref_d22", default="outputs/D22_sets")
    args = ap.parse_args()
    import os
    os.chdir(REPO)
    from ds_leakage import Reference
    R = Reference(data_root=args.ref_data_root, d22_dir=args.ref_d22, thr=args.thr)
    rows = [json.loads(ln) for ln in open(args.raw) if ln.strip()]
    rows = [r for r in rows if r.get("answerable", True)]
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    items, docs, why, n_seen = [], {}, Counter(), 0
    for j in order:
        if len(items) >= args.n:
            break
        r = rows[j]
        n_seen += 1
        paras = sorted(r["paragraphs"], key=lambda p: p["idx"])
        assert [p["idx"] for p in paras] == list(range(len(paras)))
        text, spans = build_doc(paras)
        did = f"{NAME}_{r['id']}"
        reason = R.check(did, text) or ("question" if R.q_hit(r["question"]) else "")
        if not reason:
            reason = R.check(f"musique_{r['id']}", text)        # same id as a dev-derived doc (should never happen)
        if reason:
            why[reason.split("=")[0]] += 1
            continue
        assert did not in docs
        docs[did] = {"text": text, "title": r["question"].strip()[:120]}
        hop_type = r["id"].split("__")[0]
        answers = [r["answer"].strip()] + [a.strip() for a in r.get("answer_aliases") or [] if a.strip()]
        items.append({
            "key": did, "doc_id": did, "question": r["question"].strip(), "choices": None,
            "answers": list(dict.fromkeys(answers)), "correct_label": None, "task": NAME,
            "hop": int(hop_type[0]),
            "meta": {
                "hop_type": hop_type, "n_paragraphs": len(paras),
                "supporting_idx": [p["idx"] for p in paras if p["is_supporting"]],
                "para_spans": spans, "para_titles": [p["title"] for p in paras],
                "decomposition": r.get("question_decomposition"), "answerable": r.get("answerable", True),
                "source": "musique_ans_v1.0_train",
            },
        })
    print(f"[musiquetr] {len(rows)} answerable train rows; scanned {n_seen}; kept {len(items)}; leakage drops {dict(why)} "
          f"({sum(why.values()) / max(n_seen, 1):.1%} of scanned)")
    splits = {"train": items, "val": [], "test": []}
    index = write_split(Path(args.out_dir), NAME, splits, docs, load_qwen_tokenizer())
    print_summary(NAME, {"train": items}, index, {"hops": dict(sorted(Counter(it["hop"] for it in items).items()))})
    json.dump({"rows": len(rows), "scanned": n_seen, "kept": len(items), "drops": why, "thr": args.thr},
              open(Path(args.out_dir) / f"{NAME}_split" / "leakage_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
