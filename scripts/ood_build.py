"""OOD (agent OOD, 2026-10-22): build the out-of-distribution judge suite for the lever-1 full base.

The base (outputs/D13_lever1_full/d13_best.pt) trained on QuALITY, MuSiQue, LastingBench, LongHealth, Qasper,
ContractNLI, Shakespeare-CF, LooGLE and CLIPPER (scripts/fit_d13.py TASK_SPLIT); agent DS adds MuSiQue official train
(musiquetr) and NarrativeQA train stories. Every OOD set below is a dataset / task type outside that mix:

  ood_cuad      legal span QA      CUAD v1 TEST contracts (<= 23k tokens); 4 short-answer clause fields (document
                                   name, parties, agreement date, effective date) rephrased as questions; token F1 vs
                                   every highlighted span (max).
  ood_infbench  long novel QA      InfiniteBench En.MC (4-option MCQ) + En.QA (span F1) on the novels that fit the
                                   128k operator (<= 126k tokens); entity-substituted classics.
  ood_lbv2      hard long MCQ      LongBench-v2 "short" items (<= 30k tokens), no code-repo items; 4-option MCQ.
  d25_nqswap    conflict span      NQ-Swap TEST (existing split, answer entity swapped in the passage).
  d25_fecf      conflict MCQ       FaithEval-counterfactual TEST (existing split).
  d25_cqacf     conflict MCQ       ConflictQA-popQA counterfactual TEST (existing split).

Only TEST items are written (train / val are empty lists). Prompts / scoring are xenc.doc_tasks' generic MCQ / span
ones, so the new task names need no new adapter code.

LEAKAGE (per OOD document, vs R = EVERY document and question of every lever-1 / old-base / DS split, all splits:
quality, musique, lastingbench, longhealth, qasper, contractnli, shakespeare_cf, shakespeare, loogle, clipper,
musiquetr, narrativeqa; the D15 label-free pool only with --with_d15 1, since neither base's lineage used it), with d19_leakage's blake2b word-8-gram fingerprints (mod 8):
  union containment  |sh(t) & R| / |sh(t)| >= .15                       (t's text largely seen anywhere), or
  pair rule          shared(t, r) >= 5 and >= .15 * min(|sh(t)|, |sh(r)|) for ANY single r (either direction), or
  normalized-text hash equal to a reference doc, or a question equal (normalized) to a reference question.
Flagged documents are dropped before sampling; the report goes to outputs/OOD_sets/leakage.json.

wrong_doc pairs stay inside the judged document set (d22_make_sets' shift-by-7 rule, never near-duplicates).

    python -I scripts/ood_build.py [--sets ood_cuad,ood_infbench,ood_lbv2,d25_nqswap,d25_fecf,d25_cqacf]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from d19_leakage import shingles  # noqa: E402
from xenc.doc_tasks import load_docs, load_qwen_tokenizer, print_summary, write_split  # noqa: E402

DATA = REPO / "data"
RAW = DATA / "raw_downloads"
OUT = REPO / "outputs" / "OOD_sets"
TRAIN_SPLITS = ["quality", "musique", "lastingbench", "longhealth", "qasper", "contractnli", "shakespeare_cf",
                "shakespeare", "loogle", "clipper", "musiquetr", "narrativeqa"]
WORD = re.compile(r"[a-z0-9]+")

# set -> (split dir, operator / teacher cap --max_tokens, doc token cap for selection)
SETS = {
    "ood_cuad":     {"split_dir": "data/ood_cuad_split", "max_tokens": 25000, "doc_cap": 23000, "max_docs": 80},
    "ood_infbench": {"split_dir": "data/ood_infbench_split", "max_tokens": 128000, "doc_cap": 126000, "max_docs": 0},
    "ood_lbv2":     {"split_dir": "data/ood_lbv2_split", "max_tokens": 32000, "doc_cap": 30000, "max_docs": 120},
    "d25_nqswap":   {"split_dir": "data/d25_nqswap_split", "max_tokens": 16000, "doc_cap": 15000, "max_docs": 0},
    "d25_fecf":     {"split_dir": "data/d25_fecf_split", "max_tokens": 16000, "doc_cap": 15000, "max_docs": 0},
    "d25_cqacf":    {"split_dir": "data/d25_cqacf_split", "max_tokens": 16000, "doc_cap": 15000, "max_docs": 0},
}


def nwords(t: str) -> str:
    return " ".join(WORD.findall((t or "").lower()))


def md5(t: str) -> str:
    return hashlib.md5(nwords(t).encode()).hexdigest()


def did_of(prefix: str, text: str) -> str:
    return f"{prefix}_{md5(text)[:12]}"


# ----------------------------------------------------------------------------------------------- reference index
class Ref:
    def __init__(self, log=print, with_d15: bool = False):
        t0 = time.time()
        self.post: dict[int, list[int]] = {}
        self.size: list[int] = []
        self.names: list[str] = []
        self.hashes: dict[str, str] = {}
        self.questions: set[str] = set()
        groups = {}
        for n in TRAIN_SPLITS:
            sd = DATA / f"{n}_split"
            if not (sd / "docs.json").exists():
                log(f"  [ref] {n}: missing, skipped")
                continue
            idx = json.load(open(sd / "docs.json"))
            for did, m in idx.items():
                self._add(f"{n}/{did}", (DATA / m["path"]).read_text(encoding="utf-8", errors="ignore"))
            nq = 0
            for s in ("train", "val", "test"):
                f = sd / s / "items.json"
                if f.exists():
                    for it in json.load(open(f)):
                        self.questions.add(nwords(it["question"]))
                        nq += 1
            groups[n] = (len(idx), nq)
        pool = DATA / "D15_selfq" / "docs.json"
        if with_d15 and pool.exists():
            k = 0
            for did, m in json.load(open(pool)).items():
                p = DATA / m["path"]
                if p.exists():
                    self._add(f"D15/{did}", p.read_text(encoding="utf-8", errors="ignore"))
                    k += 1
            groups["D15_pool"] = (k, 0)
        self.groups = groups
        log(f"  [ref] {len(self.names)} docs, {len(self.post)} fingerprints, {len(self.questions)} questions "
            f"({time.time() - t0:.0f}s): {groups}")

    def _add(self, name: str, text: str):
        i = len(self.names)
        self.names.append(name)
        sh = shingles(text)
        self.size.append(len(sh))
        self.hashes[md5(text)] = name
        for x in sh:
            self.post.setdefault(x, []).append(i)

    def check(self, text: str, questions=(), thr: float = 0.15) -> dict:
        sh = shingles(text)
        c, anyhit = Counter(), 0
        for x in sh:
            p = self.post.get(x)
            if p:
                anyhit += 1
                c.update(set(p))
        n = max(1, len(sh))
        union = anyhit / n
        pair, best, bestf = None, None, 0.0
        for j, k in c.items():
            f = k / n
            if f > bestf:
                best, bestf = self.names[j], f
            if pair is None and k >= 5 and k >= thr * max(1, min(len(sh), self.size[j])):
                pair = self.names[j]
        h = self.hashes.get(md5(text))
        qhit = [q for q in questions if nwords(q) in self.questions]
        reason = ("hash:" + h if h else f"union {union:.3f}" if union >= thr else "pair:" + pair if pair else
                  f"question:{qhit[0][:60]}" if qhit else "")
        return {"n_fp": len(sh), "union_frac": round(union, 4), "max_frac": round(bestf, 4), "max_doc": best,
                "pair_hit": pair, "hash_hit": h, "q_hits": len(qhit), "reason": reason}


# ----------------------------------------------------------------------------------------------- raw loaders
def build_cuad():
    d = json.load(open(RAW / "cuad/gh/x/test.json"))
    Q = {"Document Name": "What is the name (title) of this contract?",
         "Parties": "Name a party to this contract.",
         "Agreement Date": "What is the date of this contract (the agreement date)?",
         "Effective Date": "On what date does this contract become effective?"}
    docs, items = {}, []
    for a in d["data"]:
        for p in a["paragraphs"]:
            text = p["context"]
            did = did_of("cuad", text)
            docs[did] = {"text": text, "title": a["title"]}
            for q in p["qas"]:
                m = re.search(r'related to "(.+?)"', q["question"])
                cat = m.group(1) if m else None
                if cat not in Q or not q["answers"]:
                    continue
                ans = list(dict.fromkeys(x["text"].strip() for x in q["answers"] if x["text"].strip()))
                if not ans:
                    continue
                items.append({"key": "", "doc_id": did, "question": Q[cat], "choices": None, "answers": ans,
                              "correct_label": None, "task": "ood_cuad", "hop": None,
                              "meta": {"category": cat, "cuad_id": q["id"], "title": a["title"]}})
    return docs, items


def build_infbench():
    docs, items = {}, []
    for fn, task in (("longbook_choice_eng.jsonl", "ood_ibmc"), ("longbook_qa_eng.jsonl", "ood_ibqa")):
        for line in open(RAW / "infinitebench" / fn):
            r = json.loads(line)
            text = r["context"].strip()
            if len(text.split()) > 100000:          # cannot fit 126k tokens; skip before tokenizing
                continue
            did = did_of("ib", text)
            docs.setdefault(did, {"text": text, "title": ""})
            ans = [str(x).strip().strip('"').strip() for x in r["answer"] if str(x).strip()]
            if task == "ood_ibmc":
                opts = [str(o).strip() for o in r["options"]]
                if len(opts) != 4 or not ans or ans[0] not in opts or len(set(opts)) != 4:
                    continue
                lab = "ABCD"[opts.index(ans[0])]
                items.append({"key": "", "doc_id": did, "question": r["input"].strip(),
                              "choices": [{"label": "ABCD"[i], "text": o} for i, o in enumerate(opts)],
                              "answers": [ans[0]], "correct_label": lab, "task": task, "hop": None,
                              "meta": {"ib_id": r["id"], "sub": "En.MC"}})
            else:
                if not ans:
                    continue
                items.append({"key": "", "doc_id": did, "question": r["input"].strip(), "choices": None,
                              "answers": ans, "correct_label": None, "task": task, "hop": None,
                              "meta": {"ib_id": r["id"], "sub": "En.QA"}})
    return docs, items


def build_lbv2():
    docs, items = {}, []
    for r in json.load(open(RAW / "longbench_v2/data.json")):
        if r["length"] != "short" or r["domain"] == "Code Repository Understanding":
            continue
        if len(r["context"].split()) > 26000:
            continue
        text = r["context"].strip()
        did = f"lbv2_{r['_id']}"
        docs[did] = {"text": text, "title": r["sub_domain"]}
        opts = [str(r[f"choice_{x}"]).strip() for x in "ABCD"]
        items.append({"key": "", "doc_id": did, "question": r["question"].strip(),
                      "choices": [{"label": x, "text": o} for x, o in zip("ABCD", opts)],
                      "answers": [opts["ABCD".index(r["answer"])]], "correct_label": r["answer"], "task": "ood_lbv2",
                      "hop": None, "meta": {"lbv2_id": r["_id"], "domain": r["domain"], "sub_domain": r["sub_domain"],
                                            "difficulty": r["difficulty"]}})
    return docs, items


BUILDERS = {"ood_cuad": build_cuad, "ood_infbench": build_infbench, "ood_lbv2": build_lbv2}


# ----------------------------------------------------------------------------------------------- set assembly
def wrong_map(judged: list[str], texts: dict[str, str], dup: float = 0.3) -> dict:
    sh = {d: shingles(texts[d]) for d in judged}

    def isdup(x, y):
        k = len(sh[x] & sh[y])
        return k >= 5 and k >= dup * max(1, min(len(sh[x]), len(sh[y])))
    wm, n = {}, len(judged)
    for i, d in enumerate(judged):
        for s in range(7 if n > 7 else 1, 7 + n):
            w = judged[(i + s) % n]
            if w != d and not isdup(d, w):
                wm[d] = w
                break
    return wm


def verify(names) -> int:
    """register_d22's own lookup (judge_items) on every set: all keys found, wrong docs registered, 0 leaky docs."""
    from register_doc import _get, load_split_items
    rep = json.load(open(OUT / "leakage.json"))
    for name in names:
        S = json.load(open(OUT / f"{name}.json"))
        keys = set(S["item_keys"])
        items = [it for s in S["splits"] for it in load_split_items(S["split_dir"], s) if _get(it, "key") in keys]
        assert len(items) == len(keys), (name, len(items), len(keys))
        wm = json.load(open(S["wrong_map_file"]))
        assert set(wm) == set(S["judged_docs"]) and set(wm.values()) <= set(S["reg_docs"]), name
        assert not any(rep[name]["per_doc"][d]["reason"] for d in S["judged_docs"]), name
        assert all(w != d for d, w in wm.items()), name
        print(f"verify {name}: {len(items)} items found, {len(S['judged_docs'])} judged / {len(S['reg_docs'])} reg docs, "
              f"wrong map ok, leakage clean")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", type=int, default=0, help="1: only check the built sets")
    ap.add_argument("--with_d15", type=int, default=0, help="1: also treat the D15 label-free pool as training data "
                    "(it is NOT in either base's lineage: D13_4b_maps / D13_4b_reg_w10 / lever-1 / B xf01 never used it)")
    a = ap.parse_args(argv)
    if a.verify:
        return verify(a.sets.split(","))
    OUT.mkdir(parents=True, exist_ok=True)
    tok = load_qwen_tokenizer()
    R = Ref(with_d15=bool(a.with_d15))
    rep_path = OUT / "leakage.json"
    report = json.load(open(rep_path)) if rep_path.exists() else {}
    report["_reference"] = {"groups": R.groups, "n_docs": len(R.names), "thr": a.thr}
    for name in a.sets.split(","):
        c = SETS[name]
        sd = REPO / c["split_dir"]
        if name in BUILDERS:
            docs, items = BUILDERS[name]()
            used = {it["doc_id"] for it in items}
            docs = {d: v for d, v in docs.items() if d in used}
            ids = sorted(docs)
            from xenc.doc_tasks import count_tokens
            ntok = dict(zip(ids, count_tokens(tok, [docs[d]["text"] for d in ids], batch=4)))
            texts = {d: docs[d]["text"] for d in ids}
        else:
            items = json.load(open(sd / "test" / "items.json"))
            idx = json.load(open(sd / "docs.json"))
            texts = load_docs(str(sd))
            ntok = {d: idx[d]["n_tokens"] for d in idx}
        by = {}
        for it in items:
            by.setdefault(it["doc_id"], []).append(it)
        rows, bad, long_ = {}, set(), set()
        for d in sorted(by):
            if ntok[d] > c["doc_cap"]:
                long_.add(d)
                continue
            rows[d] = R.check(texts[d], [it["question"] for it in by[d]], a.thr)
            if rows[d]["reason"]:
                bad.add(d)
        keep = sorted(d for d in rows if d not in bad)
        if not keep:
            print(f"{name}: NO clean documents ({len(bad)} leaky, {len(long_)} too long): "
                  f"{Counter(r['reason'][:20] for r in rows.values())}", flush=True)
            report[name] = {"n_candidate_docs": len(by), "too_long": len(long_), "dropped": len(bad), "per_doc": rows}
            json.dump(report, open(rep_path, "w"), indent=1)
            continue
        if c["max_docs"] and len(keep) > c["max_docs"]:
            keep = sorted(random.Random(a.seed).sample(keep, c["max_docs"]))
        sel = [it for d in keep for it in by[d]]
        if name in BUILDERS:      # materialize the test-only split (only the kept docs)
            pre = name.replace("ood_", "ood")
            for j, it in enumerate(sel):
                it["key"] = f"{pre}_test_{j:05d}"
            index = write_split(DATA, f"{name}", {"train": [], "val": [], "test": sel},
                                {d: docs[d] for d in keep}, tok)
            if sel:
                print_summary(name, {"test": sel}, index)
        wm = wrong_map(keep, texts)
        reg = sorted(set(keep) | set(wm.values()))
        tasks = Counter(it["task"] for it in sel)
        S = {"task": name, "split_dir": c["split_dir"], "splits": ["test"], "max_tokens": c["max_tokens"],
             "doc_cap": c["doc_cap"], "n_items": len(sel), "tasks": dict(tasks),
             "n_leak_dropped_docs": len(bad), "n_too_long_docs": len(long_),
             "judged_docs": keep, "reg_docs": reg, "item_keys": sorted(it["key"] for it in sel),
             "wrong_map_file": f"outputs/OOD_sets/{name}_wrongmap.json",
             "doc_tokens": {"min": min(ntok[d] for d in keep), "med": sorted(ntok[d] for d in keep)[len(keep) // 2],
                            "max": max(ntok[d] for d in keep)}}
        json.dump(S, open(OUT / f"{name}.json", "w"), indent=1)
        json.dump(wm, open(OUT / f"{name}_wrongmap.json", "w"), indent=1)
        report[name] = {"n_candidate_docs": len(by), "too_long": len(long_), "dropped": len(bad),
                        "reasons": Counter(r["reason"].split(":")[0].split(" ")[0] for d, r in rows.items() if d in bad),
                        "max_union_kept": max([rows[d]["union_frac"] for d in keep] or [0]),
                        "max_pairfrac_kept": max([rows[d]["max_frac"] for d in keep] or [0]),
                        "per_doc": rows}
        json.dump(report, open(rep_path, "w"), indent=1)
        print(f"{name}: {len(sel)} items {dict(tasks)}, {len(keep)} judged docs, {len(reg)} registered; "
              f"{len(bad)} leaky docs dropped ({dict(report[name]['reasons'])}), {len(long_)} over {c['doc_cap']} tok; "
              f"kept max union {report[name]['max_union_kept']:.3f} max pair {report[name]['max_pairfrac_kept']:.3f}; "
              f"doc tokens {S['doc_tokens']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
