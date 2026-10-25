"""SOFTCONF leakage check (agent SOFTCONF, 2026-10-25), CPU: every SOFTCONF document (the L0 / unedited variant of
each bundle; edits only change a few characters) and question vs the interface TRAINING documents, with the strict
rule and the training sets of scripts/ds_judged_leak.py (word-8-gram containment both ways >= thr, normalized text
hash, identical normalized question):
  lineage = lever-1 full checkpoint's replayed train docs + the old base D13_4b_reg_w10 docs
  ds_new  = the DS Phase A+B mix (outputs/DS_AB_full) docs not in the lineage
(the 27B lv4 interface D12b_lv4_s2 was trained on the same task mix; its docs are the lineage + DS sets).

    python scripts/softconf_leak.py --split_dirs <sc split dirs> --out outputs/SOFTCONF/leak.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split_dirs", required=True)
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--lever1_args", default="outputs/D13_lever1_full/args.json")
    ap.add_argument("--ds_args", default="outputs/DS_AB_full/args.json")
    ap.add_argument("--ds_extra", default="musiquetr=data/musiquetr_split,narrativeqa=data/narrativeqa_split")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import fit_d13
    from ds_judged_leak import TrainIndex, _texts_qs, base_docs, train_docs
    from xenc.doc_tasks import load_docs
    extra = dict(s.split("=") for s in a.ds_extra.split(",") if s)
    fit_d13.TASK_SPLIT.update(extra)
    l1_texts, l1_q = train_docs(a.lever1_args)
    bt, bq = _texts_qs(base_docs())
    lin_texts, lin_q = {**l1_texts, **bt}, l1_q | bq
    ds_texts, ds_q = train_docs(a.ds_args)
    ds_new = {d: t for d, t in ds_texts.items() if d not in lin_texts}
    print(f"train docs: lineage {len(lin_texts)}, DS-new {len(ds_new)}", flush=True)
    IX = {"lineage": TrainIndex(lin_texts, lin_q), "ds_new": TrainIndex(ds_new, ds_q - lin_q)}
    res = {"thr": a.thr}
    for sd in a.split_dirs.split(","):
        D = load_docs(sd)
        items = [it for sp in ("val", "test") if Path(sd, sp, "items.json").exists()
                 for it in json.load(open(Path(sd, sp, "items.json")))]
        per, bad = {}, Counter()
        for it in items:
            did = it["doc_id"]
            if not did.endswith(("_L0", "_L0A", "_L0B")):
                continue
            r = {n: ix.check(D[did], it["question"], a.thr) for n, ix in IX.items()}
            per[it["key"]] = r
            for n in IX:
                bad[n] += r[n]["bad"]
        cj = sorted(v["lineage"]["c_jT"] for v in per.values())
        res[sd] = {"n_checked_items": len(per), "fail": dict(bad),
                   "c_jT_quantiles": [cj[int(q * (len(cj) - 1))] for q in (0, .5, .9, 1)] if cj else [],
                   "bad_keys": [k for k, v in per.items() if any(x["bad"] for x in v.values())]}
        print(sd, {k: v for k, v in res[sd].items() if k != "bad_keys"}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
