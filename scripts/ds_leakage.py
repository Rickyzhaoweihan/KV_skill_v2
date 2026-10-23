"""DS (data scale-up) STRICT leakage check for new training documents / questions.

Reference set R (nothing new may overlap it):
  * every val AND test document of every materialized benchmark split data/*_split (LongHealth: folds 0 and 1, which
    are its test / val folds, since its split dirs list all 20 patients),
  * every document of every D22 set (judged_docs, reg_docs, both sides of the wrong map), outputs/D22_sets/*.json,
  * every val / test question (normalized text) of those splits.
A candidate train document t is DROPPED when (word-8-gram containment on lowercased alphanumeric words, as D13's
>= .15 rule but over ALL of R and across tasks):
  c(t -> R) >= thr      (share of t's 8-grams that occur anywhere in R), or
  c(r -> t) >= thr      for some single reference doc r (r largely contained in t), or
  t's doc id / normalized text hash equals a reference doc's.
A candidate ITEM is dropped when its normalized question equals a reference question (or its doc is dropped).

Library: Reference(...).check(doc_id, text) -> reason | "" ; Reference.q_hit(question) -> bool.
CLI (Phase A additions): python scripts/ds_leakage.py --cand_specs musique,lastingbench,loogle,clipper \
    --skip_used outputs/DS_audit_lever1.json --out outputs/DS_leakage_phaseA.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_W = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    return _W.findall(text.lower())


def grams(text: str, n: int = 8) -> set[int]:
    w = words(text)
    return {hash(" ".join(w[i:i + n])) for i in range(max(len(w) - n + 1, 0))}


def norm_q(q: str) -> str:
    return " ".join(words(q or ""))


def text_md5(text: str) -> str:
    return hashlib.md5(" ".join(words(text)).encode()).hexdigest()


def _split_docs(sd: str) -> dict:
    from xenc.doc_tasks import load_docs_index, _doc_path
    idx = load_docs_index(sd)
    return {d: _doc_path(Path(sd), m["path"]) for d, m in idx.items()}


class Reference:
    def __init__(self, data_root: str = "data", d22_dir: str = "outputs/D22_sets", thr: float = 0.15, log=print):
        t0 = time.time()
        self.thr = thr
        ref_paths: dict[str, Path] = {}
        self.questions: set[str] = set()
        all_paths: dict[str, Path] = {}
        for sd in sorted(glob.glob(os.path.join(data_root, "*_split"))):
            P = _split_docs(sd)
            all_paths.update(P)
            for sp in ("val", "test", "train"):
                f = Path(sd, sp, "items.json")
                if not f.exists():
                    continue
                for it in json.load(open(f)):
                    if os.path.basename(sd) == "longhealth_split":
                        if (it.get("meta") or {}).get("fold") not in (0, 1):
                            continue
                    elif sp == "train":
                        continue
                    ref_paths[it["doc_id"]] = P[it["doc_id"]]
                    self.questions.add(norm_q(it.get("question", "")))
        n_split = len(ref_paths)
        from register_d13 import excluded_docs
        sets = sorted(glob.glob(os.path.join(d22_dir, "*.json")))
        sets = [s for s in sets if not s.endswith(("_wrongmap.json", "_leakage.json"))]
        J = excluded_docs(",".join(sets))
        miss = [d for d in J if d not in all_paths]
        for d in J:
            if d in all_paths:
                ref_paths[d] = all_paths[d]
        self.ids = set(ref_paths) | set(J)
        self.questions.discard("")
        # inverted index: gram -> ref index (int) or tuple of indices; per-ref gram counts
        self.names = sorted(ref_paths)
        self.size, self.md5 = [], set()
        index: dict[int, object] = {}
        for k, d in enumerate(self.names):
            txt = ref_paths[d].read_text()
            self.md5.add(text_md5(txt))
            g = grams(txt)
            self.size.append(max(len(g), 1))
            for h in g:
                v = index.get(h)
                if v is None:
                    index[h] = k
                elif isinstance(v, int):
                    if v != k:
                        index[h] = (v, k)
                else:
                    index[h] = v + (k,)
        self.index = index
        log(f"[ds_leakage] reference: {len(self.names)} docs ({n_split} val/test split docs + D22 sets; "
            f"{len(miss)} D22 ids without text), {len(index) / 1e6:.1f}M 8-grams, {len(self.questions)} questions, "
            f"thr {thr} ({time.time() - t0:.0f}s)")

    def check(self, doc_id: str, text: str) -> str:
        if doc_id in self.ids:
            return "id"
        if text_md5(text) in self.md5:
            return "text-hash"
        g = grams(text)
        if not g:
            return ""
        hit, cnt = 0, Counter()
        for h in g:
            v = self.index.get(h)
            if v is None:
                continue
            hit += 1
            if isinstance(v, int):
                cnt[v] += 1
            else:
                cnt.update(v)
        if hit / len(g) >= self.thr:
            return f"c(t->R)={hit / len(g):.2f}"
        for k, c in cnt.most_common(50):
            if c / self.size[k] >= self.thr:
                return f"c(r->t)={c / self.size[k]:.2f} r={self.names[k]}"
        return ""

    def q_hit(self, q: str) -> bool:
        return norm_q(q) in self.questions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand_specs", default="musique,lastingbench,loogle,clipper",
                    help="tasks whose TRAIN-split docs are candidates (data/<task>_split)")
    ap.add_argument("--skip_used", default="", help="DS_audit json: docs lever-1 already trained on are reported, "
                    "not candidates (Phase A = lever-1 docs + new docs that pass)")
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--base_exclude", default="outputs/D13_lever1_leakage.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    R = Reference(thr=a.thr)
    used = set()
    if a.skip_used:
        used = {d for v in json.load(open(a.skip_used)).get("lever1_docs", {}).values() for d in v}
    base = json.load(open(a.base_exclude)) if a.base_exclude else {}
    out = {k: list(v) for k, v in base.items()} if isinstance(base, dict) else {"base": list(base)}
    rep = {}
    for t in a.cand_specs.split(","):
        sd = f"data/{t}_split"
        P = _split_docs(sd)
        its = json.load(open(Path(sd, "train", "items.json")))
        dids = sorted({i["doc_id"] for i in its})
        drop, reasons, n_used_flag = [], Counter(), 0
        qdrop = 0
        for d in dids:
            r = R.check(d, P[d].read_text())
            if d in used:
                n_used_flag += bool(r)
                continue
            if r:
                drop.append(d)
                reasons[r.split("=")[0]] += 1
        qs = [i for i in its if i["doc_id"] not in used and i["doc_id"] not in drop]
        qbad = {i["doc_id"] for i in qs if R.q_hit(i.get("question", ""))}
        qdrop = len(qbad)
        drop += sorted(qbad - set(drop))
        out[t] = sorted(set(out.get(t, [])) | set(drop))
        rep[t] = {"train_docs": len(dids), "lever1_used": len([d for d in dids if d in used]),
                  "lever1_used_flagged_by_strict_check": n_used_flag, "new_candidates": len([d for d in dids if d not in used]),
                  "new_dropped": len(set(drop)), "reasons": dict(reasons), "question_hits": qdrop}
        print(t, rep[t], flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    json.dump(rep, open(a.out.replace(".json", "_report.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
