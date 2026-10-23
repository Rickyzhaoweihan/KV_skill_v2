"""B1: why LongHealth reaches text parity (CPU; existing outputs).

1. Template overlap with the base's training pool: test-doc word-5-gram containment in the training docs, and each
   test question's best content-word Jaccard with any training question (same benchmark / whole pool).
2. Out-of-domain LongHealth (2B leave-LongHealth-out base, all 400 q, D13_noLH_lh400) vs in-domain (2B D13_s2w10 and
   4B D13_4b_s2 on fold 0): which item types does in-domain training fix?
3. LongHealth question categories (keyword rules) on fold 0 (registered 4B) and on the 400-item OOD run.

    python scripts/b1_lh_deep.py > outputs/B1_lh_tables.md
"""
import json
import re
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_benchmark_fit import DocIndex, content, mcq_lex, paired, fmt_gap, words  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xenc.doc_tasks import load_doc_split, load_docs  # noqa: E402

POOL = {"quality": 150, "musique": 150, "lastingbench": 150, "longhealth": 0}   # D13_4b_reg_w10 args.train


def train_pool():
    """Replicates fit_d13.select_items(spec, 'train', folds={2,3,4}) doc choice (sorted doc ids, first N)."""
    docs, qs = {}, defaultdict(list)
    for t, nd in POOL.items():
        sd = f"data/{t}_split"
        its = load_doc_split(sd, "train")
        if t == "longhealth":
            its = [i for i in its if i.meta.get("fold") in {2, 3, 4}]
        by = OrderedDict()
        for i in its:
            by.setdefault(i.doc_id, []).append(i)
        dids = sorted(by)[: nd] if nd else sorted(by)
        D = load_docs(sd)
        for d in dids:
            docs[d] = (t, D[d])
            qs[t] += [i.question for i in by[d]]
    return docs, qs


def ngrams(text, n=5):
    w = words(text)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


NAME = re.compile(r"\b(Mr|Mrs|Ms)\.?\s+\w+(\s+\w+)?")


def qset(q):
    return set(content(NAME.sub(" ", q)))


CATS = [("NOT/EXCEPT", r"\bNOT\b|EXCEPT|never|not accurate|incorrect"),
        ("order/chronology", r"order|chronolog|first|sequence|before|after"),
        ("date/time", r"\bwhen\b|date|month|year|how long|duration|\d{2}/\d{2}"),
        ("medication/dose", r"medicat|drug|dose|dosage|mg\b|therap|regimen|treatment|prescrib"),
        ("imaging/exam", r"MRI|CT\b|scan|imaging|ultrasound|x-ray|examination|PET"),
        ("lab/pathology", r"lab|patholog|histolog|immunohisto|mutation|molecular|biopsy|value|level"),
        ("diagnosis/other", r".")]


def cat(q):
    for n, p in CATS:
        if re.search(p, q, re.I):
            return n
    return "other"


def load(f):
    return {r["key"]: r for r in map(json.loads, open(f))}


def main():
    pool_docs, pool_q = train_pool()
    allq = [q for v in pool_q.values() for q in v]
    pool_ng = defaultdict(set)
    for d, (t, txt) in pool_docs.items():
        pool_ng[t] |= ngrams(txt)
    all_ng = set().union(*pool_ng.values())
    print("# B1 LongHealth deep-dive tables (scripts/b1_lh_deep.py)\n")
    print(f"Base training pool replicated: {len(pool_docs)} docs, {len(allq)} questions "
          f"({', '.join(f'{t} {sum(1 for x in pool_docs.values() if x[0] == t)}' for t in POOL)}).\n")
    print("## 1. Template overlap of judged test items with the base's training pool\n")
    print("| benchmark | test docs | doc 5-gram containment in same-task train docs (mean) | in whole pool | "
          "question best-Jaccard vs same-task train q (median) | share ≥ .5 | vs whole pool (median) |")
    print("|---|---|---|---|---|---|---|")
    B = [r for r in map(json.loads, open("outputs/B1_items.jsonl"))]
    TASK = {"LongHealth": "longhealth", "QuALITY": "quality", "MuSiQue": "musique", "LastingBench": "lastingbench"}
    from b1_benchmark_fit import RUNS
    qcache = [(q, qset(q)) for q in allq]
    for name in RUNS:
        rows = [r for r in B if r["bench"] == name]
        sd = RUNS[name][1]
        D = load_docs(sd)
        items = {i.key: i for sp in ("train", "val", "test") if (Path(sd) / sp / "items.json").exists()
                 for i in load_doc_split(sd, sp)}
        t = TASK.get(name)
        cs, ca = [], []
        for d in sorted({r["doc_id"] for r in rows}):
            g = ngrams(D[d])
            if g:
                ca.append(len(g & all_ng) / len(g))
                if t:
                    cs.append(len(g & pool_ng[t]) / len(g))
        js, ja = [], []
        sameq = [(q, s) for q, s in qcache if t and q in set(pool_q[t])] if t else []
        sameset = [s for _, s in sameq]
        allset = [s for _, s in qcache]
        rs = rows if len(rows) <= 400 else rows[::max(1, len(rows) // 400)]
        for r in rs:
            s = qset(items[r["key"]].question)
            if not s:
                continue
            jac = lambda S: max((len(s & x) / len(s | x) for x in S), default=0.0)   # noqa: E731
            ja.append(jac(allset))
            if t:
                js.append(jac(sameset))
        f = lambda v: f"{np.mean(v):.3f}" if v else "—"   # noqa: E731
        m = lambda v: f"{np.median(v):.2f}" if v else "—"   # noqa: E731
        sh = f"{np.mean([x >= .5 for x in js]):.2f}" if js else "—"
        print(f"| {name} | {len({r['doc_id'] for r in rows})} | {f(cs)} | {f(ca)} | {m(js)} | {sh} | {m(ja)} |")

    # ---- 2. in-domain vs out-of-domain on LongHealth
    lh_items = {i.key: i for i in load_doc_split("data/longhealth_split", "test")}
    lh_docs = load_docs("data/longhealth_split")
    di = {d: DocIndex(t) for d, t in lh_docs.items()}
    noLH = load("outputs/D13_noLH_lh400/eval_test.ckpt.jsonl")
    ind2 = load("outputs/D13_s2w10_lh/eval_test.ckpt.jsonl")
    ind4 = load("outputs/D13_4b_s2_lh/eval_test.ckpt.jsonl")
    cb = load("outputs/D15_ev_text_lh/eval_test.ckpt.jsonl")
    print("\n## 2. In-domain training on LongHealth (fold 0, same 80 items, all unregistered base reads)\n")
    print("| base | encoder | LH in training | kv | text | wrong | kv−text | kv−wrong |\n|---|---|---|---|---|---|---|---|")
    k0 = sorted(ind2)
    for lab, enc, ind, R in (("D13_noLH", "2B", "no", noLH), ("D13_s2w10", "2B", "folds 2–4", ind2), ("D13_4b_s2", "4B", "folds 2–4", ind4)):
        rr = [{"kv": R[k]["kv"]["score"], "text": (R[k].get("text") or ind4[k]["text"])["score"],
               "wrong": R[k]["wrong_doc"]["score"], "doc_id": R[k]["doc_id"]} for k in k0]
        print(f"| {lab} | {enc} | {ind} | {np.mean([r['kv'] for r in rr]):.3f} | {np.mean([r['text'] for r in rr]):.3f} | "
              f"{np.mean([r['wrong'] for r in rr]):.3f} | {fmt_gap(paired(rr, 'kv', 'text'))} | {fmt_gap(paired(rr, 'kv', 'wrong'))} |")
    rr = [{"a": ind2[k]["kv"]["score"], "b": noLH[k]["kv"]["score"], "doc_id": ind2[k]["doc_id"]} for k in k0]
    print(f"\nPaired in-domain effect (2B, same items): D13_s2w10 − D13_noLH = {fmt_gap(paired(rr, 'a', 'b'))}.\n")

    # ---- 3. categories / features on the OOD 400 and on fold 0 registered
    R22 = {r["key"]: r for r in B if r["bench"] == "LongHealth"}
    rows = []
    for k, r in noLH.items():
        it = lh_items[k]
        lex, oin, neg = mcq_lex(di[it.doc_id], it)
        ev = sum(len(v["start"]) for v in (it.meta.get("answer_location") or {}).values())
        rows.append({"doc_id": it.doc_id, "fold": it.meta["fold"], "cat": cat(it.question), "kv": r["kv"]["score"],
                     "text": r["text"]["score"], "wrong": r["wrong_doc"]["score"], "ct": r["closed_tail"]["score"],
                     "lex": lex, "ev_multi": ev > 1, "q_ovl": di[it.doc_id].best(content(it.question))[0],
                     "optlen": np.mean([len(words(c["text"])) for c in it.choices]), "key": k})
    print("## 3. LongHealth by question category\n")
    print("| category | OOD 2B noLH: n | kv | text | kv−text | kv−wrong | fold-0 registered 4B: n | kv−text | kv−wrong | 9B closed_book |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for c, _ in CATS:
        a = [r for r in rows if r["cat"] == c]
        b = [R22[r["key"]] for r in a if r["key"] in R22]
        cbv = [cb[r["key"]]["closed_book"]["score"] for r in a if r["key"] in cb]
        print(f"| {c} | {len(a)} | {np.mean([r['kv'] for r in a]):.3f} | {np.mean([r['text'] for r in a]):.3f} | "
              f"{fmt_gap(paired(a, 'kv', 'text'))} | {fmt_gap(paired(a, 'kv', 'wrong'))} | {len(b)} | "
              f"{fmt_gap(paired(b, 'kv', 'text'))} | {fmt_gap(paired(b, 'kv', 'wrong'))} | "
              f"{(f'{np.mean(cbv):.2f}' if cbv else '—')} |")
    print("\n## 4. OOD LongHealth (2B noLH, n=400): feature splits\n")
    print("| feature | low n | low kv−text | low kv−wrong | high n | high kv−text | high kv−wrong |\n|---|---|---|---|---|---|---|")
    for f, cut in (("q_ovl", float(np.median([r["q_ovl"] for r in rows]))), ("lex", .5), ("ev_multi", .5), ("ct", .5),
                   ("optlen", float(np.median([r["optlen"] for r in rows])))):
        lo = [r for r in rows if r[f] <= cut]
        hi = [r for r in rows if r[f] > cut]
        print(f"| {f} (cut {cut:.2f}) | {len(lo)} | {fmt_gap(paired(lo, 'kv', 'text'))} | {fmt_gap(paired(lo, 'kv', 'wrong'))} | "
              f"{len(hi)} | {fmt_gap(paired(hi, 'kv', 'text'))} | {fmt_gap(paired(hi, 'kv', 'wrong'))} |")
    cbk = [k for k in k0 if k in cb]
    print(f"\n9B closed_book on LH fold 0: {np.mean([cb[k]['closed_book']['score'] for k in cbk]):.3f} (n={len(cbk)}, 5 options, chance .20); "
          f"lexical presence baseline on the 400: {np.mean([r['lex'] for r in rows]):.3f}.")


if __name__ == "__main__":
    main()
