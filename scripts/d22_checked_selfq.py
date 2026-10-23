"""D22 lever 3: turn a self-question file into CHECKED self-QA, whose fixed targets let registration exceed its
text teacher. The questions are written from the document only, by the ENCODER (d22_selfq.py --reader Qwen/Qwen3.5-4B).

register_d20 trains on meta.cont as the target when it is present: "<answer>{cont}</answer>", with KL to the 9B text
teacher's teacher-forced distribution on those tokens and CE on them. Without it, the target is the teacher's own
greedy answer, which caps registration at teacher level.

Per item:
  span (no choices)  kept only if the writer's answer (SQuAD-normalized) is a substring of the normalized document.
                     The target is then the answer's verbatim surface form in the document, when it can be found.
  mcq / mix-MC       the writer's intended option. It is pre-shuffled into --perm copies, each with its own
                     correct letter (so run register_d20 with --shuffle 0), and the letter prior stays balanced.
  nli                the writer's construction label (A supported / B contradicted / C not mentioned).
The leakage guard is unchanged: items stay self-questions (register_d20 asserts no task question is contained).

    python scripts/d22_checked_selfq.py --split_dir data/qasper_split --selfq outputs/D22_selfq/qasper_val_enc4b.jsonl \
        --out outputs/D22_selfq/qasper_val_enc4b_checked.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import zlib
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def verbatim(ans: str, doc: str) -> str | None:
    """The document's own surface form of `ans` (case/punct-insensitive match), else None."""
    words = re.findall(r"\w+", ans)
    if not words:
        return None
    pat = r"\b" + r"\W+".join(re.escape(w) for w in words) + r"\b"
    m = re.search(pat, doc, re.I)
    return m.group(0) if m else None


def convert(items, doc, perm: int, seed: int):
    from xenc.doc_tasks import normalize_answer
    out, st = [], Counter()
    ndoc = " " + normalize_answer(doc) + " "
    for it in items:
        tmpl = (it.get("meta") or {}).get("template")
        if not it.get("choices"):
            a = (it.get("answers") or [""])[0]
            na = normalize_answer(a)
            if not na or f" {na} " not in ndoc:
                st["span_dropped_not_in_doc"] += 1
                continue
            v = verbatim(a, doc) or a
            out.append(dict(it, meta=dict(it.get("meta") or {}, cont=v, d22_checked="span_in_doc")))
            st["span_kept"] += 1
            continue
        lab = it.get("correct_label")
        labs = [c["label"] for c in it["choices"]]
        if lab not in labs:
            st["mc_no_label"] += 1
            continue
        if tmpl == "nli":
            out.append(dict(it, meta=dict(it.get("meta") or {}, cont=lab, d22_checked="nli_construction")))
            st["nli_kept"] += 1
            continue
        gold_txt = it["choices"][labs.index(lab)]["text"]
        rng = random.Random(zlib.crc32(it["key"].encode()) + seed)
        for j in range(perm):
            p = list(it["choices"])
            rng.shuffle(p)
            ch = [{"label": L, "text": c["text"]} for L, c in zip(labs, p)]
            nl = labs[[c["text"] for c in p].index(gold_txt)]
            out.append(dict(it, key=f"{it['key']}_p{j}", choices=ch, correct_label=nl,
                            meta=dict(it.get("meta") or {}, cont=nl, d22_checked="mc_writer_label")))
        st["mc_kept"] += 1
    return out, st


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--selfq", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perm", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from xenc.doc_tasks import load_docs
    docs = load_docs(a.split_dir)
    tot = Counter()
    with open(a.out, "w") as f:
        for line in open(a.selfq):
            r = json.loads(line)
            items, st = convert(r["items"], docs[r["doc_id"]], a.perm, a.seed)
            tot.update(st)
            f.write(json.dumps({**r, "items": items, "d22_checked": dict(st)}) + "\n")
    print(f"{a.out}: {dict(tot)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
