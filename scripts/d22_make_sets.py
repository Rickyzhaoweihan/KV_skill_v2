"""D22: fix the judged item sample, the registered documents and the wrong_doc pairing for every D22 benchmark.

LEAKAGE FILTER: test documents whose word-8-gram fingerprints overlap ONE document of the D15 pool or of the base
interface's training splits (quality / musique / lastingbench train+val) by >= 0.15 (or hit D15's rule) are dropped
before sampling (outputs/D22_leakage.json, scripts/d19_leakage.py). This matters for MuSiQue (only 209 / 1208 test
paragraph-bundles are clean: MuSiQue reuses Wikipedia paragraphs across splits) and LastingBench (442 / 750 clean).
The judged items are then sampled like scripts/register_d20.py::judge_items (--max_docs first, then random.Random(0)
.sample(n)) and their KEYS are stored; scripts/register_d22.py judges exactly those keys. wrong_doc pairs stay INSIDE the judged document set
(so no extra registrations) and never pair near-duplicates (word-8-gram share >= --dup of the smaller doc, D15
fingerprints) — LooGLE has screenplay re-renderings, ContractNLI re-filed templates. ContractNLI keeps D19's map.

  python scripts/d22_make_sets.py            # -> outputs/D22_sets/<task>.json and <task>_wrongmap.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# task -> judge selection (register_d20 flags) + operator/teacher document cap (covers the whole judged documents
# except LooGLE's longest few, where the last --max_tokens tokens are read, as for the text teacher's targets)
BENCH = {
    "contractnli":    {"split_dir": "data/contractnli_split", "n": 300, "max_docs": 0, "max_tokens": 16000,
                       "wrong_map": "outputs/D19_wrongmap_contractnli_test.json", "selfq": "outputs/D20_selfq/contractnli.jsonl"},
    "qasper":         {"split_dir": "data/qasper_split", "n": 0, "max_docs": 120, "max_tokens": 16000},
    "musique":        {"split_dir": "data/musique_split", "n": 120, "max_docs": 0, "max_tokens": 16000},
    "lastingbench":   {"split_dir": "data/lastingbench_split", "n": 150, "max_docs": 0, "max_tokens": 18000},
    "loogle":         {"split_dir": "data/loogle_split", "n": 300, "max_docs": 0, "max_tokens": 128000},
    # VERY LONG documents (user, 2026-10-20: where kv might beat text). Nothing (base interface, B, A maps, D15 pool)
    # was ever trained on LooGLE, so its train/val documents are as unseen as its test documents: LooGLE-long = every
    # LooGLE document >= 48k Qwen tokens from all three splits (leakage-checked like the test docs).
    "loogle_long":    {"split_dir": "data/loogle_split", "splits": ["train", "val", "test"], "min_tokens": 48000,
                       "n": 300, "max_docs": 0, "max_tokens": 128000},
    # CLIPPER test books (48k-113k tokens): `spread` books evenly spaced over the length ranking, all their claims
    "clipper":        {"split_dir": "data/clipper_split", "spread": 12, "n": 0, "max_docs": 0, "max_tokens": 128000},
    "clipper_smoke":  {"split_dir": "data/clipper_split", "longest": 2, "n": 10, "max_docs": 0, "max_tokens": 128000},
    "shakespeare_cf": {"split_dir": "data/shakespeare_cf_split", "n": 0, "max_docs": 0, "max_tokens": 60000,
                       "selfq": "outputs/D20_selfq/shakespeare_cf.jsonl"},
}


_GROUPS = {}


def leaky(task, docs, split_dir, a) -> set:
    """Test-doc leakage from outputs/D22_leakage.json when present (d19_leakage.py), else computed here for `docs`
    with d19_leakage's fingerprints against the D15 pool and the old splits' train+val (same rule)."""
    L = json.load(open(a.leakage))
    if task in L:
        rows = L[task]["per_doc"]
    elif task.startswith("clipper") and "clipper" in json.load(open("outputs/D19_leakage.json")):
        rows = json.load(open("outputs/D19_leakage.json"))["clipper"]["per_doc"]
    else:
        from d19_leakage import DATA, OLD, Index, shingles, split_docs
        from xenc.doc_tasks import load_docs
        if not _GROUPS:
            g = _GROUPS["D15_pool"] = Index()
            for did, m in json.load(open(DATA / "D15_selfq/docs.json")).items():
                g.add(did, shingles((DATA / m["path"]).read_text(encoding="utf-8", errors="ignore")))
            g = _GROUPS["old_splits_trainval"] = Index()
            for n in OLD:
                if n != "longhealth":
                    for did, p in split_docs(n, ("train", "val")).items():
                        g.add(did, shingles(p.read_text(encoding="utf-8", errors="ignore")))
        texts = load_docs(split_dir)
        rows = {d: {gn: gi.query(shingles(texts[d])) for gn, gi in _GROUPS.items()} for d in docs}
        json.dump(rows, open(f"{a.out_dir}/{task}_leakage.json", "w"), indent=1)
    return {d for d, r in rows.items() if d in set(docs) and any(
        r[g]["max_frac"] >= a.leak_frac or r[g]["d15_rule_hit"] is not None for g in ("D15_pool", "old_splits_trainval"))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(BENCH))
    ap.add_argument("--dup", type=float, default=0.3)
    ap.add_argument("--out_dir", default="outputs/D22_sets")
    ap.add_argument("--leakage", default="outputs/D22_leakage.json")
    ap.add_argument("--leak_frac", type=float, default=0.15)
    a = ap.parse_args(argv)
    from d19_leakage import shingles
    from register_doc import _get, load_split_items
    from xenc.doc_tasks import load_docs
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for t in a.tasks.split(","):
        c = BENCH[t]
        splits = c.get("splits", ["test"])
        pool = [i for s in splits for i in load_split_items(c["split_dir"], s)]
        if c.get("min_tokens"):
            ntok = {d: m["n_tokens"] for d, m in json.load(open(Path(c["split_dir"]) / "docs.json")).items()}
            pool = [i for i in pool if ntok[_get(i, "doc_id")] >= c["min_tokens"]]
        bad = leaky(t, sorted({_get(i, "doc_id") for i in pool}), c["split_dir"], a)
        pool = [i for i in pool if _get(i, "doc_id") not in bad]
        if c.get("spread") or c.get("longest"):
            ntok = {d: m["n_tokens"] for d, m in json.load(open(Path(c["split_dir"]) / "docs.json")).items()}
            ds = sorted({_get(i, "doc_id") for i in pool}, key=lambda d: ntok[d])
            if c.get("longest"):
                pick = set(ds[-c["longest"]:])
            else:
                k = c["spread"]
                pick = {ds[round(j * (len(ds) - 1) / (k - 1))] for j in range(k)}
            pool = [i for i in pool if _get(i, "doc_id") in pick]
        docs_all = sorted({_get(i, "doc_id") for i in pool})
        if c["max_docs"]:
            keep = set(docs_all[: c["max_docs"]])
            pool = [i for i in pool if _get(i, "doc_id") in keep]
        items = random.Random(0).sample(pool, c["n"]) if c["n"] and len(pool) > c["n"] else pool
        judged = sorted({_get(i, "doc_id") for i in items})
        if c.get("wrong_map"):
            wmap = json.load(open(c["wrong_map"]))
            wmap = {d: wmap[d] for d in judged}
        else:
            texts = load_docs(c["split_dir"])
            sh = {d: shingles(texts[d]) for d in judged}

            def dup(x, y):
                k = len(sh[x] & sh[y])
                return k >= 5 and k >= a.dup * max(1, min(len(sh[x]), len(sh[y])))
            wmap, n = {}, len(judged)
            for i, d in enumerate(judged):
                for s in range(7 if n > 7 else 1, 7 + n):
                    w = judged[(i + s) % n]
                    if w != d and not dup(d, w):
                        wmap[d] = w
                        break
        reg = sorted(set(judged) | set(wmap.values()))
        out = {"task": t, **c, "splits": splits, "n_items": len(items), "n_leak_dropped_docs": len(bad), "judged_docs": judged,
               "reg_docs": reg, "item_keys": sorted(_get(i, "key") for i in items),
               "wrong_map_file": f"{a.out_dir}/{t}_wrongmap.json"}
        json.dump(out, open(f"{a.out_dir}/{t}.json", "w"), indent=1)
        json.dump(wmap, open(f"{a.out_dir}/{t}_wrongmap.json", "w"), indent=1)
        print(f"{t}: {len(items)} items, {len(judged)} judged docs, {len(reg)} registered docs "
              f"({len(bad)} leaky test docs excluded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
