"""[D25] Knowledge-conflict sets in the xenc doc-QA schema (smart layer injection, step 1 diagnostics).

Short-passage conflict benchmarks are BUNDLED into multi-passage documents ("Passage 1: ... Passage k: ..."), so that
(i) one operator prefill serves several questions, (ii) wrong_doc is another bundle, and (iii) the 64-token tail the
reader prefills itself (kv protocol) never contains the evidence: questions are asked only about passages 1..k-1,
never about the LAST passage (which overlaps the tail).

  d25_cqa     ConflictQA-popQA-chatgpt (Xie et al., ICLR'24). 8 passages / doc. CF doc = counter_memory passages,
              paired ORIGINAL doc (`d25cqa_or_*`) = parametric_memory passages of the same entities in the same order.
              2-way MCQ: {memory_answer, counter_answer} (shuffled). CF items: gold = counter, meta.prior_label =
              memory. ORIG items (meta.variant == "orig"): same question / choices on the paired doc, gold = memory.
              Restricted to the top half by Wikipedia popularity (stronger priors).
  d25_fecf    FaithEval counterfactual (Ming et al., ICLR'25), 4-way MCQ (ARC science), 4 passages / doc.
              gold = the counterfactual answerKey; the prior is per reader (its closed-book choice).
  d25_nqswap  NQ-Swap (entity-substituted NQ; this release has 5 substituted contexts per question and ONLY the
              original answer). 3 items (5 contexts each) / doc, span answers. `answers` holds the ORIGINAL answer,
              i.e. the PRIOR (meta.answers_are_prior = True): never score it with doc_tasks.score as accuracy.
              Doc-adherence = the answer is not the prior and occurs in its passage (meta.span = char span).

Splits are by document (train / val / test); prior labels never come from the reader under test.

  python -I scripts/data/prepare_d25_conflict.py [--out_dir data]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from xenc.doc_tasks import load_qwen_tokenizer, print_summary, write_split  # noqa: E402

RAW = REPO / "data" / "raw_downloads"
SIZES = {"d25_cqa": (100, 30, 50), "d25_fecf": (150, 40, 60), "d25_nqswap": (300, 40, 100)}


def _bundle_text(passages):
    return "\n\n".join(f"Passage {k + 1}:\n{p.strip()}" for k, p in enumerate(passages))


def _spans(passages):
    """char spans of each passage body inside _bundle_text(passages)."""
    out, pos = [], 0
    for k, p in enumerate(passages):
        head = f"Passage {k + 1}:\n"
        body = p.strip()
        out.append((pos + len(head), pos + len(head) + len(body)))
        pos += len(head) + len(body) + 2
    return out


def _split_bundles(bundles, sizes):
    ntr, nva, nte = sizes
    assert len(bundles) >= ntr + nva + nte, (len(bundles), sizes)
    return {"test": bundles[:nte], "val": bundles[nte:nte + nva], "train": bundles[nte + nva:nte + nva + ntr]}


def build_cqa(rng):
    rows = [json.loads(l) for l in open(RAW / "conflictqa" / "conflictQA-popQA-chatgpt.json")]
    rows = [r for r in rows if r.get("counter_memory") and r.get("parametric_memory") and r["memory_answer"].strip()
            and r["counter_answer"].strip() and r["memory_answer"].strip() != r["counter_answer"].strip()
            and len(r["memory_answer"]) < 200 and len(r["counter_answer"]) < 200]
    pops = sorted(r["popularity"] for r in rows)
    med = pops[len(pops) // 2]
    rows = [r for r in rows if r["popularity"] >= med]
    rng.shuffle(rows)
    K = 8
    bundles, seen = [], set()
    cur = []
    for r in rows:
        if r["question"] in seen:
            continue
        seen.add(r["question"])
        cur.append(r)
        if len(cur) == K:
            bundles.append(cur)
            cur = []
    sp = _split_bundles(bundles, SIZES["d25_cqa"])
    docs, splits = {}, {}
    for split, bs in sp.items():
        items = []
        for b in bs:
            bid = f"{split}{len(docs) // 2:04d}"
            cf, orr = f"d25cqa_cf_{bid}", f"d25cqa_or_{bid}"
            docs[cf] = {"text": _bundle_text([r["counter_memory"] for r in b]), "title": f"ConflictQA CF bundle {bid}"}
            docs[orr] = {"text": _bundle_text([r["parametric_memory"] for r in b]), "title": f"ConflictQA orig bundle {bid}"}
            for k, r in enumerate(b[:-1]):
                opts = [("mem", r["memory_answer"].strip()), ("ctr", r["counter_answer"].strip())]
                rng.shuffle(opts)
                lab = {t: "AB"[i] for i, (t, _) in enumerate(opts)}
                ch = [{"label": "AB"[i], "text": s} for i, (_, s) in enumerate(opts)]
                base = {"question": r["question"], "choices": ch, "task": "d25_cqa", "hop": None}
                common = {"kind": "cqa", "passage": k + 1, "popularity": r["popularity"],
                          "ground_truth": r["ground_truth"], "bundle": bid}
                items.append({**base, "key": f"d25cqa_{split}_{len(items):05d}", "doc_id": cf,
                              "answers": [dict(opts)["ctr"]], "correct_label": lab["ctr"],
                              "meta": {**common, "variant": "cf", "conflict": True, "prior_label": lab["mem"],
                                       "pair_doc": orr}})
                items.append({**base, "key": f"d25cqa_{split}_{len(items):05d}", "doc_id": orr,
                              "answers": [dict(opts)["mem"]], "correct_label": lab["mem"],
                              "meta": {**common, "variant": "orig", "conflict": False, "prior_label": lab["mem"],
                                       "pair_doc": cf}})
        splits[split] = items
    return docs, splits


def build_fecf(rng):
    rows = [json.loads(l) for l in open(RAW / "faitheval_counterfactual" / "data" / "test.d25.jsonl")]
    rows = [r for r in rows if r.get("context") and r.get("answerKey")]
    rng.shuffle(rows)
    K = 4
    bundles = [rows[i:i + K] for i in range(0, len(rows) - K + 1, K)]
    sp = _split_bundles(bundles, SIZES["d25_fecf"])
    docs, splits = {}, {}
    for split, bs in sp.items():
        items = []
        for b in bs:
            bid = f"{split}{len(docs):04d}"
            did = f"d25fecf_{bid}"
            docs[did] = {"text": _bundle_text([r["context"] for r in b]), "title": f"FaithEval-CF bundle {bid}"}
            for k, r in enumerate(b[:-1]):
                labs, txts = list(r["choices"]["label"]), list(r["choices"]["text"])
                if labs != list("ABCDE"[:len(labs)]) or r["answerKey"] not in labs:
                    continue
                ch = [{"label": l, "text": str(t)} for l, t in zip(labs, txts)]
                items.append({"key": f"d25fecf_{split}_{len(items):05d}", "doc_id": did, "question": r["question"],
                              "choices": ch, "answers": [ch[labs.index(r["answerKey"])]["text"]],
                              "correct_label": r["answerKey"], "task": "d25_fecf", "hop": None,
                              "meta": {"kind": "fecf", "passage": k + 1, "conflict": True, "prior_label": None,
                                       "src_id": r["id"], "bundle": bid}})
        splits[split] = items
    return docs, splits


def build_nqswap(rng):
    rows = [json.loads(l) for l in open(RAW / "nqswap" / "data" / "train.d25.jsonl")]
    rows = [r for r in rows if r.get("context") and r.get("answer")]
    rng.shuffle(rows)
    K = 3
    bundles = [rows[i:i + K] for i in range(0, len(rows) - K + 1, K)]
    sp = _split_bundles(bundles, SIZES["d25_nqswap"])
    docs, splits = {}, {}
    for split, bs in sp.items():
        items = []
        for b in bs:
            bid = f"{split}{len(docs):04d}"
            did = f"d25nqs_{bid}"
            ps = ["\n".join(c.strip() for c in r["context"]) for r in b]
            docs[did] = {"text": _bundle_text(ps), "title": f"NQ-Swap bundle {bid}"}
            spans = _spans(ps)
            for k, r in enumerate(b[:-1]):
                q = r["question"].strip()
                q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
                items.append({"key": f"d25nqs_{split}_{len(items):05d}", "doc_id": did, "question": q,
                              "choices": None, "answers": [str(a) for a in r["answer"]], "correct_label": None,
                              "task": "d25_nqswap", "hop": None,
                              "meta": {"kind": "nqswap", "passage": k + 1, "conflict": True, "answers_are_prior": True,
                                       "prior_answers": [str(a) for a in r["answer"]], "span": list(spans[k]),
                                       "bundle": bid}})
        splits[split] = items
    return docs, splits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out_dir", default=str(REPO / "data"))
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--sets", default="d25_cqa,d25_fecf,d25_nqswap")
    a = ap.parse_args()
    tok = load_qwen_tokenizer()
    fns = {"d25_cqa": build_cqa, "d25_fecf": build_fecf, "d25_nqswap": build_nqswap}
    for name in a.sets.split(","):
        out = Path(a.out_dir) / f"{name}_split"
        if out.exists():
            print(f"{out} exists; skipping (delete it to rebuild)")
            continue
        docs, splits = fns[name](random.Random(a.seed))
        index = write_split(Path(a.out_dir), name, splits, docs, tok)
        print_summary(name, splits, index)


if __name__ == "__main__":
    main()
