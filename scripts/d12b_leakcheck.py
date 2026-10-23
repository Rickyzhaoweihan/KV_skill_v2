"""D12b: DS's STRICT judged-vs-training leakage check (scripts/ds_judged_leak.py rule: word-8-gram containment
c(j->T) >= thr or c(t->j) >= thr for a single training doc, same text hash, or the judged question equals a training
question) for EVERY 27B judge set against the union of the training documents of a run mix and its lineage runs.

    python scripts/d12b_leakcheck.py --mix_args <args.json of the new run, or --train/--extra_splits/... flags> \
        --lineage outputs/D12p_lever_full_s2/args.json,outputs/D12b_lv3_s2/args.json --out outputs/D12b_max8k_leak.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

KEYMAP = {"shakespeare": "shakespeare_cf"}


def run_docs(A: dict) -> set:
    import fit_d12p
    from register_doc import _get
    fit_d12p.add_extra_splits(A.get("extra_splits") or "")
    a = types.SimpleNamespace(exclude_sets=A.get("exclude_sets") or "", exclude_docs=A.get("exclude_docs") or "",
                              doc_len=A.get("doc_len") or "")
    tr, _ = fit_d12p.select_items(A["train"], "train", set(range(5)) - {A.get("test_fold", 0), A.get("val_fold", 1)},
                                  0, A.get("seed", 0))
    (tr,), _ = fit_d12p.lever_filter(a, tr)
    return {_get(i, "doc_id") for i in tr}


def texts_questions(dids: set):
    import fit_d12p
    from xenc.doc_tasks import load_docs
    from ds_leakage import norm_q
    by: dict = {}
    for d in dids:
        by.setdefault(d.split("_")[0], set()).add(d)
    texts, qs = {}, set()
    for t, ds in by.items():
        sd = fit_d12p.TASK_SPLIT[KEYMAP.get(t, t)]
        D = load_docs(sd)
        texts.update({d: D[d] for d in ds})
        for sp in ("train", "val", "test"):
            f = Path(sd, sp, "items.json")
            if f.exists():
                qs |= {norm_q(i["question"]) for i in json.load(open(f)) if i["doc_id"] in ds}
    return texts, qs


def judged_sets() -> dict:
    """{name: (split_dir, [item dicts])} for every 27B judge set."""
    out = {}
    for s in ("qasper", "loogle", "shakespeare_cf", "lastingbench"):
        S = json.load(open(f"outputs/D22_sets/{s}.json"))
        out[s] = (S["split_dir"], set(S["item_keys"]))
    for s in ("mcq_leval", "mcq_bamboo"):
        S = json.load(open(f"outputs/MCQ_sets/{s}.json"))
        out[s] = (S["split_dir"], set(S["item_keys"]))
    out["quality"] = ("data/quality_split", None)
    out["longhealth_f0"] = ("data/longhealth_split", "fold0")
    out["d25_cqacf"] = ("data/d25_cqacf_split", None)
    out["d25_fecf"] = ("data/d25_fecf_split", None)
    res = {}
    for n, (sd, keys) in out.items():
        its = [i for sp in ("train", "val", "test") if Path(sd, sp, "items.json").exists()
               for i in json.load(open(Path(sd, sp, "items.json")))]
        if keys == "fold0":
            its = [i for i in json.load(open(Path(sd, "test", "items.json"))) if (i.get("meta") or {}).get("fold") == 0]
        elif keys is None:
            its = json.load(open(Path(sd, "test", "items.json")))
        else:
            its = [i for i in its if i["key"] in keys]
        res[n] = (sd, its)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--extra_splits", default="")
    ap.add_argument("--exclude_sets", default="")
    ap.add_argument("--exclude_docs", default="")
    ap.add_argument("--doc_len", default="")
    ap.add_argument("--lineage", default="", help="comma list of args.json of earlier runs in this checkpoint's lineage")
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from ds_judged_leak import TrainIndex
    from xenc.doc_tasks import load_docs
    docs = run_docs(vars(a))
    print(f"new mix: {len(docs)} train docs", flush=True)
    for p in [x for x in a.lineage.split(",") if x]:
        d = run_docs(json.load(open(p)))
        print(f"lineage {p}: {len(d)} train docs ({len(d - docs)} not in the new mix)", flush=True)
        docs |= d
    texts, qs = texts_questions(docs)
    uniq = {}
    for d, t in texts.items():                        # aliased splits (nqamc / looglemc) share files: index once
        uniq.setdefault(hashlib.md5(t.encode()).hexdigest(), (d, t))
    print(f"training docs total {len(texts)} ({len(uniq)} distinct texts), {len(qs)} training questions", flush=True)
    IX = TrainIndex({d: t for d, t in uniq.values()}, qs)
    res = {"thr": a.thr, "n_train_docs": len(texts), "args": vars(a), "sets": {}}
    for n, (sd, its) in judged_sets().items():
        D = load_docs(sd)
        per = {i["key"]: IX.check(D[i["doc_id"]], i["question"], a.thr) for i in its}
        bad = [k for k, v in per.items() if v["bad"]]
        why = Counter(("c(j->T)" if per[k]["c_jT"] >= a.thr else "") + ("c(t->j)" if per[k]["c_tj"] >= a.thr else "")
                      + ("q" if per[k]["q"] else "") + ("hash" if per[k]["hash"] else "") for k in bad)
        res["sets"][n] = {"n_items": len(its), "flagged": len(bad), "reasons": dict(why), "flagged_keys": bad,
                          "worst": sorted(((v["c_jT"], v["c_tj"], v["t"]) for v in per.values()), reverse=True)[:5]}
        print(f"  {n:14s} n={len(its):5d} flagged {len(bad):4d} {dict(why)}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    tot = sum(v["flagged"] for v in res["sets"].values())
    print(f"LEAKCHECK {'CLEAN' if tot == 0 else f'FLAGGED {tot} judged items'} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
