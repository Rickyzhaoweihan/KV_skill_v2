"""DS: STRICT leakage check of the JUDGED D22 items (MuSiQue, LastingBench) against the TRAINING docs, and lever-1 full's
judge numbers recomputed on the strictly-clean vs contaminated subsets (CPU, from the existing per-item eval json).

Training sets checked:
  lineage  = docs the lever-1 full checkpoint ever trained on: the old base D13_4b_reg_w10 (stage 1 + 2 spec
             quality:150,musique:150,lastingbench:150,longhealth:0, no exclusions) ∪ lever-1 full's replayed train docs
  ds_ab    = the DS Phase A+B mix (outputs/DS_AB_full): replayed with its own args (adds musiquetr, narrativeqa)
Rule (scripts/ds_leakage.py, word-8-gram on lowercased alphanumeric words), for a judged doc j vs a training set T:
  c(j -> T) >= thr  (share of j's 8-grams found anywhere in T's docs), or
  c(t -> j) >= thr  for some single training doc t, or same normalized text hash, or
  the judged QUESTION equals (normalized) a training question.
Per-item recompute: kv@0, kv_reg = es<=200 (D22 rule: kv@{es_choice[doc]['200']}), wrong_doc@0, closed_tail, text, for
lever-1 full (outputs/D13L1f_<t>) and the D22 base (outputs/D22_<t>), on all / clean / contaminated items.

    python scripts/ds_judged_leak.py --out outputs/DS_judged_leak.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ds_leakage import grams, norm_q, text_md5  # noqa: E402


class TrainIndex:
    def __init__(self, texts: dict, questions: set):
        self.names = sorted(texts)
        self.size, self.md5, self.index = [], set(), {}
        for k, d in enumerate(self.names):
            t = texts[d]
            self.md5.add(text_md5(t))
            g = grams(t)
            self.size.append(max(len(g), 1))
            for h in g:
                v = self.index.get(h)
                if v is None:
                    self.index[h] = k
                elif isinstance(v, int):
                    if v != k:
                        self.index[h] = (v, k)
                else:
                    self.index[h] = v + (k,)
        self.questions = questions

    def check(self, text: str, question: str, thr: float) -> dict:
        g = grams(text)
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
        c_jT = hit / max(len(g), 1)
        best, bestd = 0.0, ""
        for k, c in cnt.most_common(200):
            r = c / self.size[k]
            if r > best:
                best, bestd = r, self.names[k]
        qhit = norm_q(question) in self.questions
        hashhit = text_md5(text) in self.md5
        bad = c_jT >= thr or best >= thr or qhit or hashhit
        return {"c_jT": round(c_jT, 4), "c_tj": round(best, 4), "t": bestd, "q": qhit, "hash": hashhit, "bad": bad}


def train_docs(args_json: str, extra: dict | None = None) -> tuple[dict, set]:
    """Replay register_d13's train selection for a run's args.json -> ({doc_id: text}, {normalized questions})."""
    import fit_d13
    from ds_audit import replay
    from xenc.doc_tasks import load_docs
    A = json.load(open(args_json))
    if extra is None and A.get("extra_splits"):
        extra = dict(s.split("=") for s in A["extra_splits"].split(",") if s)
    tr_folds = set(range(5)) - {A.get("test_fold", 0), A.get("val_fold", 1)}
    dl = A.get("train_doc_len") or A.get("doc_len") or ""
    *_, kdocs = replay(A, A["train"], dl, A.get("exclude_docs") or "", tr_folds, extra)
    return _texts_qs({d for v in kdocs.values() for d in v})


def _texts_qs(dids: set) -> tuple[dict, set]:
    import fit_d13
    from xenc.doc_tasks import load_docs
    texts, qs = {}, set()
    by = {}
    for d in dids:
        by.setdefault(d.split("_")[0], set()).add(d)
    key = {"shakespeare": "shakespeare_cf"}
    for t, ds in by.items():
        sd = fit_d13.TASK_SPLIT[key.get(t, t)]
        D = load_docs(sd)
        texts.update({d: D[d] for d in ds})
        for sp in ("train", "val", "test"):
            f = Path(sd, sp, "items.json")
            if f.exists():
                qs |= {norm_q(i["question"]) for i in json.load(open(f)) if i["doc_id"] in ds}
    return texts, qs


def base_docs() -> set:
    """Old base D13_4b_reg_w10 (and its stage-1 maps): default spec, first 150 docs, no exclusions."""
    import fit_d13
    A = json.load(open("outputs/D13_4b_reg_w10/args.json"))
    _, d = fit_d13.select_items(A["train"], "train", {2, 3, 4}, 0, 0)
    return set(d)


def stats(xs):
    n = len(xs)
    if not n:
        return None, None, 0
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(n - 1, 1))
    return m, sd / math.sqrt(n), n


def judge_rows(task: str, sub: dict) -> dict:
    out = {}
    text = {}
    for src in (f"outputs/D22_{task}/eval_test.json", f"outputs/D13L1f_{task}/eval_test.json"):
        E = json.load(open(src))
        for it in E["items"]:
            if "text" in it:
                text[it["key"]] = it["text"]["score"]
    for name, src in (("D22 base", f"outputs/D22_{task}/eval_test.json"), ("lever-1 full", f"outputs/D13L1f_{task}/eval_test.json")):
        E = json.load(open(src))
        rows = {}
        for it in E["items"]:
            es = E["es_choice"][it["doc_id"]]["200"]
            rows[it["key"]] = {"kv@0": it["kv@0"]["score"], "kv_reg": it[f"kv@{es}"]["score"],
                               "wrong_doc@0": it["wrong_doc@0"]["score"], "closed_tail": it["closed_tail"]["score"],
                               "text": text[it["key"]]}
        R = {}
        for s, keys in sub.items():
            ks = [k for k in keys if k in rows]
            R[s] = {"n": len(ks)}
            for c in ("text", "kv@0", "kv_reg", "wrong_doc@0", "closed_tail"):
                m, se, _ = stats([rows[k][c] for k in ks])
                R[s][c] = (round(m, 4), round(se, 4)) if m is not None else None
            m, se, _ = stats([rows[k]["kv@0"] - rows[k]["text"] for k in ks])
            R[s]["kv@0-text"] = (round(m, 4), round(se, 4)) if m is not None else None
        out[name] = R
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="musique,lastingbench")
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--lever1_args", default="outputs/D13_lever1_full/args.json")
    ap.add_argument("--ds_args", default="outputs/DS_AB_full/args.json")
    ap.add_argument("--ds_extra", default="musiquetr=data/musiquetr_split,narrativeqa=data/narrativeqa_split")
    ap.add_argument("--ds_train", default="quality:150,musique:0,lastingbench:0,longhealth:0,qasper:0,contractnli:0,"
                    "shakespeare_cf:0,loogle:0,clipper:0,musiquetr:0,narrativeqa:0")
    ap.add_argument("--out", default="outputs/DS_judged_leak.json")
    a = ap.parse_args(argv)
    import fit_d13
    from xenc.doc_tasks import load_docs
    extra = dict(s.split("=") for s in a.ds_extra.split(",") if s)
    fit_d13.TASK_SPLIT.update(extra)
    l1_texts, l1_q = train_docs(a.lever1_args)
    bt, bq = _texts_qs(base_docs())
    lin_texts, lin_q = {**l1_texts, **bt}, l1_q | bq
    print(f"lineage train docs: {len(lin_texts)} (lever-1 {len(l1_texts)}, base {len(bt)})", flush=True)
    if Path(a.ds_args).exists():
        ds_texts, ds_q = train_docs(a.ds_args)
    else:                                      # DS run not started yet: replay its intended spec from lever-1 args
        A = json.load(open(a.lever1_args))
        A.update({"train": a.ds_train, "doc_len": "loogle:0:96000,clipper:0:96000,narrativeqa:0:96000",
                  "exclude_docs": "outputs/DS_leakage_phaseA.json"})
        tmp = Path(a.out).with_suffix(".ds_args.json")
        json.dump(A, open(tmp, "w"))
        ds_texts, ds_q = train_docs(str(tmp), extra)
    ds_new = {d: t for d, t in ds_texts.items() if d not in lin_texts}
    print(f"DS A+B train docs: {len(ds_texts)} ({len(ds_new)} not in the lineage)", flush=True)
    IX = {"lineage": TrainIndex(lin_texts, lin_q), "ds_new": TrainIndex(ds_new, ds_q - lin_q)}
    res = {"thr": a.thr}
    for t in a.tasks.split(","):
        if t in ("quality", "longhealth"):          # judged by register_d20: QuALITY full test, LongHealth fold 0
            sd = f"data/{t}_split"
            its = json.load(open(Path(sd, "test", "items.json")))
            if t == "longhealth":
                its = [i for i in its if (i.get("meta") or {}).get("fold") == 0]
            S = {"split_dir": sd, "item_keys": [i["key"] for i in its]}
        else:
            S = json.load(open(f"outputs/MCQ_sets/{t}.json" if t.startswith("mcq_") else
                               f"outputs/OOD_sets/{t}.json" if t.startswith("d25_") else f"outputs/D22_sets/{t}.json"))
        sd = S["split_dir"]
        D = load_docs(sd)
        items = {i["key"]: i for sp in ("train", "val", "test")             # plain dicts (load_split_items -> DocItem)
                 if Path(sd, sp, "items.json").exists() for i in json.load(open(Path(sd, sp, "items.json")))}
        keys = S["item_keys"]
        per = {}
        for k in keys:
            it = items[k]
            per[k] = {n: ix.check(D[it["doc_id"]], it["question"], a.thr) for n, ix in IX.items()}
        bad_lin = [k for k in keys if per[k]["lineage"]["bad"]]
        bad_ds = [k for k in keys if per[k]["ds_new"]["bad"]]
        sub = {"all": keys, "clean (vs lineage)": [k for k in keys if k not in bad_lin],
               "contaminated (vs lineage)": bad_lin}
        cj = sorted(per[k]["lineage"]["c_jT"] for k in keys)
        res[t] = {"n_items": len(keys), "n_docs": len({items[k]["doc_id"] for k in keys}),
                  "fail_lineage": len(bad_lin), "fail_lineage_docs": len({items[k]["doc_id"] for k in bad_lin}),
                  "fail_ds_new": len(bad_ds),
                  "lineage_reasons": dict(Counter(("c(j->T)" if per[k]["lineage"]["c_jT"] >= a.thr else "") +
                                                  ("c(t->j)" if per[k]["lineage"]["c_tj"] >= a.thr else "") +
                                                  ("q" if per[k]["lineage"]["q"] else "") +
                                                  ("hash" if per[k]["lineage"]["hash"] else "") for k in bad_lin)),
                  "c_jT_quantiles": [cj[int(q * (len(cj) - 1))] for q in (0, .25, .5, .75, .9, 1)],
                  "rows": (judge_rows(t, sub) if Path(f"outputs/D13L1f_{t}/eval_test.json").exists()
                           and Path(f"outputs/D22_{t}/eval_test.json").exists() else None),
                  "bad_lineage_keys": bad_lin, "bad_ds_new_keys": bad_ds, "per_item": per}
        print(t, {k: v for k, v in res[t].items() if k not in ("per_item",)}, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
