"""DS (data scale-up) audit: per benchmark, train items / docs AVAILABLE vs USED by lever-1 full, and why each cap exists.

Replays register_d13.py's exact selection (select_items -> train_from_val -> exclude_sets / exclude_docs -> doc_len) for
the lever-1 full args (outputs/D13_lever1_full/args.json) and for an alternative spec (e.g. the DS Phase-A mix), so the
counts are the ones the training job sees. CPU only (no model).

    python scripts/ds_audit.py --args outputs/D13_lever1_full/args.json --out outputs/DS_audit.json \
        [--alt_train ... --alt_doc_len ... --alt_exclude_docs ...]
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


def replay(A: dict, train_spec: str, doc_len: str, exclude_docs: str, tr_folds: set, extra: dict | None = None):
    """-> (pre-filter items by task, kept items by task, drop reasons {task: Counter})."""
    import fit_d13
    from register_d13 import doc_len_filter, excluded_docs
    from register_doc import _get
    if extra:
        fit_d13.TASK_SPLIT.update(extra)
    train, d1 = fit_d13.select_items(train_spec, "train", tr_folds, 0, A.get("seed", 0))
    for spec in [x for x in A.get("train_from_val", "").split(",") if x]:
        name, skip = spec.split(":")
        ex, _ = fit_d13.select_items(f"{name}:0", "val", None, 0, A.get("seed", 0))
        keep = set(sorted({_get(i, "doc_id") for i in ex})[int(skip):])
        train += [i for i in ex if _get(i, "doc_id") in keep]
    J = excluded_docs(A.get("exclude_sets", ""))
    X = set()
    if exclude_docs:
        x = json.load(open(exclude_docs))
        X = set(sum(x.values(), [])) if isinstance(x, dict) else set(x)
    keep_len = doc_len_filter(doc_len)
    pre, kept, why = Counter(), Counter(), {}
    kdocs, pdocs = {}, {}
    for it in train:
        d = _get(it, "doc_id")
        t = d.split("_")[0]
        pre[t] += 1
        pdocs.setdefault(t, set()).add(d)
        r = "judged(D22 set)" if d in J else "near-dup(.15)" if d in X else "doc_len" if not keep_len(d) else ""
        if r:
            why.setdefault(t, Counter())[r] += 1
        else:
            kept[t] += 1
            kdocs.setdefault(t, set()).add(d)
    return pre, kept, why, {t: len(v) for t, v in kdocs.items()}, {t: len(v) for t, v in pdocs.items()}, kdocs


def available():
    """All train-split items / docs per task (LongHealth: folds 2-4 only = the train folds)."""
    from fit_d13 import TASK_SPLIT
    out = {}
    for t, sd in TASK_SPLIT.items():
        if not Path(sd, "train", "items.json").exists():
            continue
        its = json.load(open(Path(sd, "train", "items.json")))
        if t == "longhealth":
            its = [i for i in its if (i.get("meta") or {}).get("fold") in {2, 3, 4}]
        out[t] = (len(its), len({i["doc_id"] for i in its}))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--args", default="outputs/D13_lever1_full/args.json")
    ap.add_argument("--alt_train", default="")
    ap.add_argument("--alt_doc_len", default=None)
    ap.add_argument("--alt_exclude_docs", default=None)
    ap.add_argument("--extra_splits", default="", help="name=split_dir,... (as register_d13 --extra_splits)")
    ap.add_argument("--out", default="outputs/DS_audit.json")
    a = ap.parse_args(argv)
    A = json.load(open(a.args))
    tr_folds = set(range(5)) - {A.get("test_fold", 0), A.get("val_fold", 1)}
    extra = dict(s.split("=") for s in (a.extra_splits or A.get("extra_splits") or "").split(",") if s)
    if extra:
        import fit_d13
        fit_d13.TASK_SPLIT.update(extra)
    res = {"available": available()}
    pre, kept, why, kd, pd_, l1docs = replay(A, A["train"], A.get("train_doc_len") or A.get("doc_len", ""),
                                             A.get("exclude_docs", ""), tr_folds)
    res["lever1"] = {"pre": pre, "kept": kept, "why": why, "kept_docs": kd, "pre_docs": pd_}
    res["lever1_docs"] = {t: sorted(v) for t, v in l1docs.items()}
    print(f"lever-1 full ({a.args}): kept {sum(kept.values())} items / {sum(kd.values())} docs")
    print(f"{'task':<14}{'avail it/doc':>16}{'lever1 it/doc':>16}  drops")
    for t in sorted(set(res['available']) | set(pre)):
        av = res["available"].get(t if t != "shakespeare" else "shakespeare_cf", ("-", "-"))
        print(f"{t:<14}{str(av[0]) + '/' + str(av[1]):>16}{str(kept.get(t, 0)) + '/' + str(kd.get(t, 0)):>16}  {dict(why.get(t, {}))}")
    if a.alt_train:
        dl = A.get("doc_len", "") if a.alt_doc_len is None else a.alt_doc_len
        ex = A.get("exclude_docs", "") if a.alt_exclude_docs is None else a.alt_exclude_docs
        pre2, kept2, why2, kd2, pd2, alt_docs = replay(A, a.alt_train, dl, ex, tr_folds, extra)
        res["alt"] = {"train": a.alt_train, "doc_len": dl, "exclude_docs": ex, "pre": pre2, "kept": kept2, "why": why2,
                      "kept_docs": kd2, "pre_docs": pd2,
                      "superset_of_lever1": all(l1docs.get(t, set()) <= alt_docs.get(t, set()) for t in l1docs)}
        print(f"\nalt ({a.alt_train} | doc_len {dl} | exclude {ex}): kept {sum(kept2.values())} items / {sum(kd2.values())} docs;"
              f" superset of lever-1 docs: {res['alt']['superset_of_lever1']}")
        for t in sorted(pre2):
            print(f"{t:<14}{str(kept2.get(t, 0)) + '/' + str(kd2.get(t, 0)):>16}  (+{kept2.get(t, 0) - kept.get(t, 0)} items)"
                  f"  drops {dict(why2.get(t, {}))}")
    json.dump(res, open(a.out, "w"), indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else sorted(o))
    return 0


if __name__ == "__main__":
    sys.exit(main())
