"""F1 (ContractNLI fix): clause-level, task-format self-NLI for per-document registration. Additive wrapper around
scripts/d20_selfq.py (that file is not modified; TEMPLATES / parse are extended in-process).

Generate (GPU, the 9B reader writes from the DOCUMENT ONLY, like d20_selfq):
    python scripts/f1_nli_selfq.py gen --split_dir data/contractnli_split --docs val:20 --templates nlic --parts 8 \
        --out outputs/F1_selfq/contractnli_nlic.jsonl
  template nlic: 6 hypotheses about part k of S of the agreement, by construction
      H1 entailed (one clause, reworded)          H2 entailed (needs TWO different clauses)
      H3 contradicted (one clause)                H4 contradicted (a combination of two clauses)
      H5, H6 not mentioned (plausible terms for this kind of agreement that it does not address)
  rendered in the TASK's answer format (question frame and the 3 options read from the TRAIN split items, so the
  registration prompt has the judge's form; the hypotheses are the writer's own). Leakage: d20_selfq's
  drop_test_matches removes any item containing a task question or any quoted task hypothesis (>= 30 chars), and
  register_d20 asserts the same before registration.

Build arm files (CPU):
    python scripts/f1_nli_selfq.py build --base outputs/F1_selfq/contractnli_d20.jsonl \
        --nlic outputs/F1_selfq/contractnli_nlic.jsonl --n_nli 12 --ratio 5,2,5 --out <file>
  keeps the base file's mcq / mix items unchanged, replaces its nli items by n_nli nlic items drawn at the label ratio
  E,C,N (5,2,5 ~ the ContractNLI TRAIN prior .49/.11/.40 after the teacher check; 4,4,4 = balanced). Every kept item
  carries meta.intended = its construction letter, for register_d20 --check_label 1.
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

NLIC_TEMPLATE = (
    "Write 6 one-sentence hypotheses about what this agreement requires, permits or forbids, phrased as general "
    "statements about its terms, the way a lawyer checks a contract. Split the document into {S} equal parts and "
    "base them on part {k}. H1 and H2 are ENTAILED by the agreement: H1 restates one clause in different words; H2 "
    "follows only from combining two different clauses. H3 and H4 are CONTRADICTED by the agreement: it states the "
    "opposite (H4 contradicts what two clauses say together). H5 and H6 are NOT MENTIONED: plausible terms for this "
    "kind of agreement that it does not address at all. Use exactly this format and nothing else:\n"
    "H1: <hypothesis>\nH2: <hypothesis>\nH3: <hypothesis>\nH4: <hypothesis>\nH5: <hypothesis>\nH6: <hypothesis>")
NLIC_LAB = {1: "A", 2: "A", 3: "B", 4: "B", 5: "C", 6: "C"}
NLIC_KIND = {1: "E1", 2: "E2", 3: "C1", 4: "C2", 5: "N", 6: "N"}


def task_format(split_dir="data/contractnli_split"):
    """(question frame with {h}, choices) read from the TRAIN split items (format only, no train content)."""
    it = json.load(open(Path(split_dir) / "train" / "items.json"))[0]
    q = it["question"]
    m = re.search(r"\"(.+)\"", q, re.S)
    frame = q[: m.start()] + "\"{h}\"" + q[m.end():]
    return frame, [dict(c) for c in it["choices"]]


def parse_nlic(txt: str) -> list[dict]:
    import d20_selfq
    frame, choices = task_format()
    out = []
    for n in range(1, 7):
        m = re.search(rf"H{n}:\s*(.+?)\s*(?:\n|$)", txt)
        if m and len(m.group(1)) > 15:
            h = m.group(1).strip().replace('"', "'")
            lab = NLIC_LAB[n]
            out.append({"question": frame.format(h=h), "answers": [choices["ABC".index(lab)]["text"]],
                        "choices": [dict(c) for c in choices], "correct_label": lab, "_kind": NLIC_KIND[n]})
    return [q for q in out if d20_selfq._ok(q)]


def gen_main(argv):
    import d20_selfq
    _p20 = d20_selfq.parse
    d20_selfq.TEMPLATES["nlic"] = NLIC_TEMPLATE

    def parse(tmpl, txt):
        if tmpl != "nlic":
            return _p20(tmpl, txt)
        qs = parse_nlic(txt)
        return qs
    d20_selfq.parse = parse
    return d20_selfq.main(argv)


def build_main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="D20-style self-question file (mcq/mix/nli templates)")
    ap.add_argument("--nlic", required=True)
    ap.add_argument("--n_nli", type=int, default=12)
    ap.add_argument("--ratio", default="5,2,5", help="E,C,N counts per n_nli (scaled)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    base = {json.loads(x)["doc_id"]: json.loads(x) for x in open(a.base)}
    nl = {json.loads(x)["doc_id"]: json.loads(x) for x in open(a.nlic)}
    r = [int(x) for x in a.ratio.split(",")]
    want = {"A": r[0], "B": r[1], "C": r[2]}
    tot = sum(want.values())
    want = {k: round(v * a.n_nli / tot) for k, v in want.items()}
    st = Counter()
    with open(a.out, "w") as f:
        for did, rec in base.items():
            if did not in nl:
                continue
            keep = [x for x in rec["items"] if (x.get("meta") or {}).get("template") in ("mcq", "mix")]
            rng = random.Random(zlib.crc32(did.encode()))
            by = {"A": [], "B": [], "C": []}
            for x in nl[did]["items"]:
                by[x["correct_label"]].append(x)
            for L in by:
                # prefer a mix of single- and two-clause hypotheses: alternate kinds after a seeded shuffle
                rng.shuffle(by[L])
                ones = [x for x in by[L] if not (x["meta"].get("kind") or "").endswith("2")]
                twos = [x for x in by[L] if (x["meta"].get("kind") or "").endswith("2")]
                mixd = [y for pair in zip(ones, twos) for y in pair] + ones[len(twos):] + twos[len(ones):]
                by[L] = mixd
            nli = []
            for L, k in want.items():
                nli += by[L][:k]
                st[f"label_{L}"] += len(by[L][:k])
                st["short_" + L] += max(0, k - len(by[L]))
            rng.shuffle(nli)
            for x in nli:
                st[x["meta"].get("kind")] += 1
            # interleave so register_d20's [:max_items] cut never drops a whole template
            items = []
            mc = list(keep)
            while mc or nli:
                if mc:
                    items.append(mc.pop(0))
                if nli:
                    items.append(nli.pop(0))
            for j, x in enumerate(items):
                x = dict(x)
                x["meta"] = dict(x.get("meta") or {})
                if x["meta"].get("f1nlic"):
                    x["meta"]["intended"] = x["correct_label"]
                items[j] = x
            f.write(json.dumps({**{k: v for k, v in rec.items() if k != "items"}, "items": items,
                                "f1_build": {"ratio": want, "n_nli": a.n_nli}}) + "\n")
            st["docs"] += 1
            st["items"] += len(items)
    print(f"{a.out}: {dict(st)}")
    return 0


def tag_main(argv):
    """Post-process a generated nlic file: template name 'nli' (fixed option order in register_d20's shuffle),
    meta.kind from the parser, f1nlic marker."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    st = Counter()
    with open(a.out, "w") as f:
        for line in open(a.inp):
            rec = json.loads(line)
            for x in rec["items"]:
                kind = x.pop("_kind", None)
                x["meta"] = {**(x.get("meta") or {}), "template": "nli", "f1nlic": True, "kind": kind}
                st[kind] += 1
            f.write(json.dumps(rec) + "\n")
    print(f"{a.out}: {dict(st)}")
    return 0


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    sys.exit({"gen": gen_main, "build": build_main, "tag": tag_main}[cmd](rest))
