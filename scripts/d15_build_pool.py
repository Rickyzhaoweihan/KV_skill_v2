"""D15 step 1b: build the label-free training pool data/D15_selfq/ from the raw extracts (d15_extract_raw.py) plus
the TRAIN documents of data/{quality,musique,lastingbench}_split and LongHealth folds 2-4.

  * English 2k-14.5k-token documents (the reader runtime keeps the last ~15k prefix tokens); longer sources (books, reports) are cut into up to --max_windows[src] windows
    of 6k-14k tokens at line boundaries (native questions are dropped for windowed documents).
  * Leakage guard: sampled word-8-gram fingerprints of EVERY QuALITY val/test doc, LongHealth folds 0-1, MuSiQue
    and LastingBench val/test docs; a candidate sharing >= 15% of its (or their) fingerprints is rejected. The same
    test dedups the pool itself (e.g. LongBench musique vs the musique split).
  * Output (split-dir layout, loadable with xenc.doc_tasks.load_docs("data/D15_selfq")):
      docs.json      {doc_id: {"path", "n_tokens", "src", "windowed"}}
      native.json    native (doc, question) items (DocItem dicts; split-train items keep their keys)
      heldout.json   held-out pool docs (never trained on; selection set together with D13's val docs)
      order.json     self-question generation priority (held-out first, then a seeded shuffle)

    python scripts/d15_build_pool.py --raw data/D15_selfq/raw --out data/D15_selfq
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

MAXW = {"infbench": 4, "nqa": 3, "lbv2": 2, "aalcr": 2, "cuad": 2}
WORD = re.compile(r"[a-z0-9]+")


def shingles(text: str, mod: int = 8, n: int = 8) -> set:
    w = WORD.findall(text.lower())
    out = set()
    for i in range(0, max(len(w) - n + 1, 0)):
        x = int.from_bytes(hashlib.blake2b(" ".join(w[i:i + n]).encode(), digest_size=8).digest(), "little")
        if x % mod == 0:
            out.add(x)
    return out


class FP:
    def __init__(self):
        self.idx: dict = {}
        self.size: dict = {}

    def add(self, did, sh):
        self.size[did] = len(sh)
        for x in sh:
            self.idx.setdefault(x, did)

    def hit(self, sh, frac=0.15):
        c = collections.Counter(self.idx[x] for x in sh if x in self.idx)
        for did, k in c.most_common(3):
            if k >= 5 and k >= frac * max(1, min(len(sh), self.size[did])):
                return did
        return None


def english(text: str) -> bool:
    s = text[:20000]
    letters = sum(ch.isalpha() for ch in s)
    ascii_l = sum(ch.isascii() and ch.isalpha() for ch in s)
    return letters > 0 and ascii_l / letters > 0.95


def windows(tok, text, rng, k_max, lo=6000, hi=14000):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    n = len(off)
    if n <= 14500:                                  # D13Rt keeps the last ~15k prefix tokens
        return [(text, n, False)] if n >= 2000 else []
    k = min(k_max, max(1, n // lo))
    starts = [int(i * (n - hi) / max(k - 1, 1)) for i in range(k)] if k > 1 else [rng.randrange(0, n - hi)]
    out = []
    for s in starts:
        L = rng.randrange(lo, hi)
        a, b = off[s][0], off[min(s + L, n - 1)][1]
        a2 = text.rfind("\n", 0, a)
        a = a2 + 1 if a2 >= 0 and a - a2 < 3000 else a
        b2 = text.find("\n", b)
        b = b2 if 0 <= b2 - b < 3000 else b
        out.append((text[a:b], None, True))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/D15_selfq/raw")
    ap.add_argument("--out", default="data/D15_selfq")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--n_heldout", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    from xenc.doc_tasks import load_doc_split, load_docs, load_docs_index
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    rng = random.Random(a.seed)
    out = Path(a.out)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    data = _ROOT / "data"

    # ---- forbidden: every val/test document of the split tasks + LongHealth folds 0-1
    forb = FP()
    folds = json.load(open(data / "longhealth_split/folds.json"))["folds"]
    lh_bad = set(folds[0]) | set(folds[1])
    split_train = {}
    for task in ("quality", "musique", "lastingbench", "longhealth"):
        sd = str(data / f"{task}_split")
        docs = load_docs(sd)
        tr = {i.doc_id for i in load_doc_split(sd, "train")}
        bad = set()
        for sp in ("val", "test"):
            bad |= {i.doc_id for i in load_doc_split(sd, sp)}
        if task == "longhealth":
            bad, tr = lh_bad, set(docs) - lh_bad
        for d in bad:
            forb.add(d, shingles(docs[d]))
        split_train[task] = (sd, sorted(tr - bad), docs)
        print(f"forbidden {task}: {len(bad)} docs; train docs {len(tr - bad)}", flush=True)

    pool, native, seen = {}, [], FP()
    # ---- split train docs (D13's pool) with their native items
    for task, (sd, tr, docs) in split_train.items():
        idx = load_docs_index(sd)
        trs = set(tr)
        for d in tr:
            pool[d] = {"path": idx[d]["path"], "n_tokens": idx[d].get("n_tokens"), "src": task, "windowed": False}
            seen.add(d, shingles(docs[d]))
        for sp in ("train", "val", "test") if task == "longhealth" else ("train",):
            for it in load_doc_split(sd, sp):
                if it.doc_id in trs and (task != "longhealth" or sp == "train"):
                    native.append(it.to_dict())
    print(f"split-train docs {len(pool)}, native items {len(native)}", flush=True)

    # ---- raw sources
    stats = collections.Counter()
    for fn in sorted(os.listdir(a.raw)):
        if not fn.endswith(".jsonl"):
            continue
        src = fn[:-6]
        kmax = MAXW.get(src, 1)
        with open(os.path.join(a.raw, fn)) as f:
            for line in f:
                r = json.loads(line)
                text = r["text"].replace("\x00", "")
                if not english(text):
                    stats[f"{src}:non_en"] += 1
                    continue
                for wi, (t, n, win) in enumerate(windows(tok, text, rng, kmax)):
                    if n is None:
                        n = len(tok(t, add_special_tokens=False)["input_ids"])
                    if not 2000 <= n <= 20000:
                        stats[f"{src}:len"] += 1
                        continue
                    sh = shingles(t)
                    if forb.hit(sh):
                        stats[f"{src}:forbidden"] += 1
                        continue
                    if seen.hit(sh):
                        stats[f"{src}:dup"] += 1
                        continue
                    did = f"d15_{src}_{hashlib.sha1(str(r['sid']).encode()).hexdigest()[:10]}_{wi}"
                    p = out / "docs" / src
                    p.mkdir(exist_ok=True)
                    (p / f"{did}.txt").write_text(t)
                    pool[did] = {"path": f"{out.name}/docs/{src}/{did}.txt", "n_tokens": n, "src": src, "windowed": win}
                    seen.add(did, sh)
                    stats[f"{src}:kept"] += 1
                    if not win:
                        for j, q in enumerate(r["native"]):
                            native.append({"key": f"{did}_n{j}", "doc_id": did, "question": q["question"],
                                           "choices": q.get("choices"), "answers": q.get("answers") or [""],
                                           "correct_label": q.get("correct_label"), "task": src, "hop": None,
                                           "meta": {"native": True}})
        print(f"{src}: " + " ".join(f"{k.split(':')[1]} {v}" for k, v in stats.items() if k.startswith(src + ":")),
              flush=True)

    ext = sorted(d for d, m in pool.items() if d.startswith("d15_"))
    by_src = collections.defaultdict(list)
    for d in ext:
        by_src[pool[d]["src"]].append(d)
    held = []
    srcs = sorted(by_src)
    while len(held) < a.n_heldout and any(by_src.values()):
        for s in srcs:
            if by_src[s] and len(held) < a.n_heldout:
                held.append(by_src[s].pop(rng.randrange(len(by_src[s]))))
    rest = [d for d in sorted(pool) if d not in set(held)]
    rng.shuffle(rest)
    json.dump(pool, open(out / "docs.json", "w"), indent=0)
    json.dump(native, open(out / "native.json", "w"))
    json.dump(sorted(held), open(out / "heldout.json", "w"), indent=0)
    json.dump(held + rest, open(out / "order.json", "w"))
    c = collections.Counter(m["src"] for m in pool.values())
    toks = sorted(m["n_tokens"] or 0 for m in pool.values())
    print(f"POOL {len(pool)} docs {dict(c)} | native items {len(native)} | held-out {len(held)} | "
          f"tokens median {toks[len(toks) // 2]} p90 {toks[int(.9 * len(toks))]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
