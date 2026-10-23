"""MCQ (agent MCQ, 2026-10-22): reading-comprehension MULTIPLE-CHOICE / JUDGMENT sets for the lever-1 base.

User (2026-10-22): "应该多搞这种阅读理解的选择题而不是单纯背书". Two products, built from small raw files under
$HF_HOME/mcq_raw (downloaded data, treated as untrusted data; parsed only, never executed):

TEST-ONLY judge sets (data/mcq_<set>_split/test, outputs/MCQ_sets/<set>.json + _wrongmap.json), prompts / scoring are
xenc.doc_tasks' generic MCQ ones (letter in <answer></answer>, exact label match), so no new scorer is needed:
  mcq_leval   L-Eval exam subsets (GPL-3.0): TOEFL-TPO lectures (4-way, task mcq_tpo), Coursera lectures (4-way,
              single-answer items only, task mcq_coursera), SFiction (story-world True/False, task mcq_sfic).
              L-Eval's QuALITY / GSM / CodeU / topic-retrieval subsets are excluded (QuALITY = our training source).
  mcq_bamboo  BAMBOO (MIT) paperqa_16k + meetingqa_16k: 4-way MCQ over 2023 NLP papers / committee meetings.
  mcq_hallu   BAMBOO senhallu_16k + abshallu_16k: judgment "is this hypothesis entailed by the paper?" (Yes / No).
  mcq_musr    MuSR (CC-BY-4.0): murder mysteries (2-way), team allocation (3-way), object placements (3–5-way).

TRAIN-ONLY splits (data/<name>_split/train) for comprehension-heavy base training (register_d13 --extra_splits):
  racemc     RACE-high train articles (human 4-way MCQ; short exam passages).
  multirc    MultiRC train (SuperGLUE): per candidate answer, judgment "is the proposed answer correct?" (2-way).
  nqamc      NarrativeQA train stories (DS's data/narrativeqa_split, long) -> 4-way MCQ: gold = reference answer,
             distractors = answers to OTHER questions about the SAME story with the same wh-type (labels are exact,
             no LLM). Doc paths reuse docs/narrativeqa (no copy).
  looglemc   LooGLE train docs (shortdep_qa / longdep_qa only; DS's exclusions + D22 sets + <= 96k tokens) -> 4-way
             MCQ, same construction. Doc paths reuse docs/loogle.

LEAKAGE.
  test sets : ood_build.Ref (every doc / question of every lever-1 / DS split, all splits) + every NEW train candidate
              text (racemc / multirc); OOD's rule (union >= .15, pair >= .15 of the smaller, text hash, question).
  train sets: ds_leakage.Reference (every val / test doc of every data/*_split, incl. the mcq_* / ood_* test splits
              written by --phase test, + every D22-set doc), DS's strict rule (c(t->R) >= .15, c(r->t) >= .15, id,
              text hash, question). nqamc / looglemc docs must ALSO be docs DS trains on (narrativeqa / loogle train,
              DS_leakage_phaseA + D22 exclusions, <= 96k tokens) -> they add no new document text.

    python -I scripts/mcq_build.py --phase test     # build + leakage-check the test sets (vs train + new train cands)
    python -I scripts/mcq_build.py --phase train    # build + leakage-check the train splits (vs every val/test)
    python -I scripts/mcq_build.py --phase verify   # replay register_d22's lookup on every test set
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RAW = Path(os.environ.get("MCQ_RAW", "/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home/mcq_raw"))
DATA = REPO / "data"
OUT = REPO / "outputs" / "MCQ_sets"
WORD = re.compile(r"[a-z0-9]+")

# set -> split dir, judge --max_tokens (doc prefix cap), doc cap for selection, max judged docs (0 = all)
SETS = {
    "mcq_leval":  {"max_tokens": 32000, "doc_cap": 30000, "max_docs": 0},
    "mcq_bamboo": {"max_tokens": 20000, "doc_cap": 19000, "max_docs": 0},
    "mcq_hallu":  {"max_tokens": 12000, "doc_cap": 11500, "max_docs": 120},
    "mcq_musr":   {"max_tokens": 4000, "doc_cap": 3800, "max_docs": 0},
}
SFIC_Q = "Is the following statement true or false in the world described in the document?\n\nStatement: {s}"
HALLU_Q = ("Consider this hypothesis about the paper:\n\"{h}\"\nIs the hypothesis entailed by the paper (no "
           "contradiction and no unsupported claim)?")
MRC_Q = "Question: {q}\nProposed answer: {a}\n\nIs the proposed answer correct according to the document?"
TF = [{"label": "A", "text": "True"}, {"label": "B", "text": "False"}]
YN = [{"label": "A", "text": "Yes"}, {"label": "B", "text": "No"}]
CI = [{"label": "A", "text": "Correct"}, {"label": "B", "text": "Incorrect"}]


def nwords(t: str) -> str:
    return " ".join(WORD.findall((t or "").lower()))


def md5(t: str) -> str:
    return hashlib.md5(nwords(t).encode()).hexdigest()


def did_of(prefix: str, text: str) -> str:
    return f"{prefix}_{md5(text)[:12]}"


def item(did, q, choices, gold_label, task, meta, answers=None):
    labs = [c["label"] for c in choices]
    assert gold_label in labs, (gold_label, labs)
    gold_txt = choices[labs.index(gold_label)]["text"]
    return {"key": "", "doc_id": did, "question": q.strip(), "choices": choices, "answers": [gold_txt] + (answers or []),
            "correct_label": gold_label, "task": task, "hop": None, "meta": meta}


def lettered(opts: list[str]) -> list[dict]:
    return [{"label": "ABCDE"[i], "text": o.strip()} for i, o in enumerate(opts)]


_OPT = re.compile(r"^\s*\(?([A-E])[.)]\s*(.*)$")


def split_mcq(text: str):
    """'question\\nA. x\\nB. y ...' -> (question, [options]) or None."""
    lines = [ln for ln in text.strip().split("\n")]
    qs, opts = [], []
    for ln in lines:
        m = _OPT.match(ln)
        if m and (not opts and m.group(1) == "A" or opts and m.group(1) == "ABCDE"[len(opts)]):
            opts.append(m.group(2).strip())
        elif opts:
            opts[-1] += " " + ln.strip()
        else:
            qs.append(ln)
    q = " ".join(x.strip() for x in qs).strip()
    q = re.sub(r"^Question\s*\d+\.\s*", "", q)
    return (q, opts) if q and len(opts) >= 2 else None


def jsonl(path):
    out = []
    for i, ln in enumerate(open(path, encoding="utf-8")):
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            print(f"  [warn] {path.name}: unparsable line {i} skipped")
    return out


# ============================================================================ test builders
def build_leval():
    docs, items = {}, []
    for fn, task in (("tpo", "mcq_tpo"), ("coursera", "mcq_coursera")):
        for r in jsonl(RAW / "leval" / f"{fn}.jsonl"):
            text = r["input"].strip()
            did = did_of(fn, text)
            docs[did] = {"text": text, "title": r.get("source", fn)}
            for ins, out in zip(r["instructions"], r["outputs"]):
                p = split_mcq(ins)
                gold = out.strip().upper()
                if not p or len(gold) != 1 or len(p[1]) != 4 or gold not in "ABCD":
                    continue                                     # Coursera multi-answer items ("BD") dropped
                items.append(item(did, p[0], lettered(p[1]), gold, task, {"src": f"leval/{fn}"}))
    for r in jsonl(RAW / "leval" / "sci_fi.jsonl"):
        text = r["input"].strip()
        did = did_of("sfic", text)
        docs[did] = {"text": text, "title": text.split("\n")[0][:80]}
        for ins, out in zip(r["instructions"], r["outputs"]):
            s = re.sub(r"^\s*\d+\.\s*", "", ins)
            s = re.sub(r",?\s*True or False\?.*$", "", s, flags=re.S).strip()
            story = out.strip().split()[0].strip("[],.").lower()      # "True [fact: False]" -> story-world answer
            if story not in ("true", "false") or not s:
                continue
            items.append(item(did, SFIC_Q.format(s=s), [dict(c) for c in TF], "A" if story == "true" else "B",
                              "mcq_sfic", {"src": "leval/sci_fi", "fact": out}))
    return docs, items


def build_bamboo():
    docs, items = {}, []
    for fn, task in (("paperqa_16k", "mcq_bpaper"), ("meetingqa_16k", "mcq_bmeet")):
        for r in jsonl(RAW / "bamboo" / f"{fn}.jsonl"):
            text = r["content"].strip()
            did = did_of("bamboo", text)
            docs[did] = {"text": text, "title": fn}
            opts = [re.sub(r"^\s*[A-D][.)]\s*", "", o).strip() for o in r["options"]]
            gold = str(r["answer"]).strip().upper()
            if len(opts) != 4 or gold not in ("A", "B", "C", "D") or len(set(opts)) != 4:
                continue
            items.append(item(did, r["question"], lettered(opts), gold, task, {"src": f"bamboo/{fn}"}))
    return docs, items


def build_hallu():
    docs, items = {}, []
    for fn, task in (("senhallu_16k", "mcq_senhallu"), ("abshallu_16k", "mcq_abshallu")):
        for r in jsonl(RAW / "bamboo" / f"{fn}.jsonl"):
            text = r["content"].strip()
            did = did_of("hallu", text)
            docs[did] = {"text": text, "title": r.get("title", "")}
            h = " ".join(str(r["hypothesis"]).split())
            if not h or not isinstance(r["answer"], bool):
                continue
            items.append(item(did, HALLU_Q.format(h=h), [dict(c) for c in YN], "A" if r["answer"] else "B", task,
                              {"src": f"bamboo/{fn}", "title": r.get("title", "")}))
    return docs, items


def build_musr(seed: int = 0, caps=(("murder_mystery", 80), ("team_allocation", 80), ("object_placements", 32))):
    csv.field_size_limit(10 ** 9)
    docs, items = {}, []
    for fn, cap in caps:
        rows = list(csv.DictReader(open(RAW / "musr" / f"{fn}.csv", encoding="utf-8")))
        by = {}
        for r in rows:
            by.setdefault(r["narrative"].strip(), []).append(r)
        keys = sorted(by, key=md5)
        keys = sorted(random.Random(seed).sample(keys, min(cap, len(keys))), key=md5)
        for text in keys:
            did = did_of("musr", text)
            docs[did] = {"text": text, "title": fn}
            for r in by[text]:
                opts = [str(x) for x in ast.literal_eval(r["choices"])]      # python list literal (data, not code)
                k = int(r["answer_index"])
                if not 2 <= len(opts) <= 5 or not 0 <= k < len(opts) or len(set(opts)) != len(opts):
                    continue
                items.append(item(did, r["question"], lettered(opts), "ABCDE"[k], f"mcq_musr_{fn.split('_')[0]}",
                                  {"src": f"musr/{fn}"}))
    return docs, items


TEST_BUILDERS = {"mcq_leval": build_leval, "mcq_bamboo": build_bamboo, "mcq_hallu": build_hallu, "mcq_musr": build_musr}


# ============================================================================ train builders
def build_racemc(n_articles: int = 2500):
    rows = jsonl(RAW / "race" / "high_train.jsonl")
    by = {}
    for r in rows:
        by.setdefault(r["article"].strip(), []).append(r)
    docs, items = {}, []
    for text in sorted(by, key=md5)[: n_articles * 2]:       # 2x candidates; the leakage pass trims to n_articles
        did = did_of("racemc", text)
        docs[did] = {"text": text, "title": by[text][0]["example_id"]}
        for r in by[text]:
            opts = [str(o) for o in r["options"]]
            if len(opts) != 4 or r["answer"] not in "ABCD" or len(set(opts)) != 4:
                continue
            items.append(item(did, r["question"], lettered(opts), r["answer"], "racemc", {"src": "race/high/train",
                                                                                         "race_id": r["example_id"]}))
    return docs, items, n_articles


def build_multirc(seed: int = 0, per_q: int = 3):
    rows = jsonl(RAW / "multirc" / "train.jsonl")
    rng = random.Random(seed)
    byq = {}
    for r in rows:
        byq.setdefault((r["paragraph"].strip(), r["question"].strip()), []).append(r)
    docs, items = {}, []
    for (para, q), cands in sorted(byq.items(), key=lambda x: md5(x[0][0] + x[0][1])):
        did = did_of("multirc", para)
        docs.setdefault(did, {"text": para, "title": "multirc"})
        pos = [c for c in cands if int(c["label"]) == 1]
        neg = [c for c in cands if int(c["label"]) == 0]
        pick = rng.sample(pos, min(len(pos), (per_q + 1) // 2)) + rng.sample(neg, min(len(neg), per_q // 2 + 1))
        for c in pick[:per_q]:
            a = " ".join(str(c["answer"]).split())
            items.append(item(did, MRC_Q.format(q=q, a=a), [dict(x) for x in CI], "A" if int(c["label"]) else "B",
                              "multirc", {"src": "superglue/multirc/train", "idx": str(c["idx"])}))
    return docs, items, 0


_WH = ("who", "whom", "whose", "where", "when", "why", "which", "what", "how")


def wh_type(q: str) -> str:
    w = WORD.findall(q.lower())[:4]
    for i, x in enumerate(w):
        if x in _WH:
            if x == "how" and i + 1 < len(w) and w[i + 1] in ("many", "much", "long", "old", "far", "often"):
                return "howq"
            return {"whom": "who", "whose": "who"}[x] if x in ("whom", "whose") else x
    return "other"


def _f1(a: str, b: str) -> float:
    from xenc.doc_tasks import f1_score
    return f1_score(a, [b])


def derived_mc(src_items, prefix: str, task: str, seed: int, ok_doc) -> tuple[dict, list]:
    """Long-doc 4-way MCQ from span items: distractors = other answers of the SAME doc with the same wh-type."""
    rng = random.Random(seed)
    by = {}
    for it in src_items:
        if ok_doc(it["doc_id"]):
            by.setdefault(it["doc_id"], []).append(it)
    out, used = [], set()
    for did in sorted(by):
        its = by[did]
        pool = {}
        for it in its:
            pool.setdefault(wh_type(it["question"]), []).append(it)
        for it in its:
            gold = it["answers"][0].strip()
            golds = [a.strip() for a in it["answers"]]
            gl = max(1, len(WORD.findall(gold.lower())))
            cands, seen = [], {nwords(g) for g in golds}
            for o in pool[wh_type(it["question"])]:
                if o is it:
                    continue
                a = o["answers"][0].strip()
                na = nwords(a)
                if not na or na in seen or max(_f1(a, g) for g in golds) >= 0.34:
                    continue
                if any(_f1(a, c) >= 0.5 for c in cands) or f" {na} " in f" {nwords(it['question'])} ":
                    continue
                seen.add(na)
                cands.append(a)
            # prefer distractors of similar length (no length cue)
            cands.sort(key=lambda a: abs(len(WORD.findall(a.lower())) - gl) / gl + rng.random() * 0.25)
            if len(cands) < 3 or not gold:
                continue
            opts = cands[:3] + [gold]
            rng.shuffle(opts)
            k = opts.index(gold)
            nid = f"{prefix}_{did.split('_', 1)[1]}"
            out.append(item(nid, it["question"], lettered(opts), "ABCD"[k], task,
                            {"src": f"{did} (derived MCQ, same-doc distractors)", "orig_key": it["key"],
                             "wh": wh_type(it["question"])}, answers=[g for g in golds[1:]]))
            used.add(did)
    return {did: None for did in used}, out


# ============================================================================ helpers
def wrong_map(judged, texts):
    from ood_build import wrong_map as wm
    return wm(judged, texts)


def write_derived(name: str, items: list, src_split: str, src_ids: dict):
    """train-only split whose docs.json reuses the source split's document files (no copy)."""
    src_idx = json.load(open(DATA / f"{src_split}_split" / "docs.json"))
    index = {}
    for it in items:
        orig = f"{src_split}_{it['doc_id'].split('_', 1)[1]}"
        index[it["doc_id"]] = dict(src_idx[orig])
    sd = DATA / f"{name}_split"
    from xenc.doc_tasks import validate_item
    for j, it in enumerate(items):
        it["key"] = f"{name}_train_{j:06d}"
        validate_item(it)
    for s, its in (("train", items), ("val", []), ("test", [])):
        (sd / s).mkdir(parents=True, exist_ok=True)
        json.dump(its, open(sd / s / "items.json", "w"), indent=1, ensure_ascii=False)
    json.dump(index, open(sd / "docs.json", "w"), indent=1, ensure_ascii=False)
    return index


# ============================================================================ phases
def phase_test(a):
    from ood_build import Ref
    from xenc.doc_tasks import count_tokens, load_qwen_tokenizer, print_summary, write_split
    OUT.mkdir(parents=True, exist_ok=True)
    tok = load_qwen_tokenizer()
    R = Ref()
    n0 = len(R.names)
    for fn in (build_racemc, build_multirc):            # NEW train candidates are training data too
        d, its, _ = fn()
        for did, v in d.items():
            R._add(f"newtrain/{did}", v["text"])
        R.questions |= {nwords(it["question"]) for it in its}
    print(f"[mcq] reference: {n0} split docs + {len(R.names) - n0} new-train candidate docs", flush=True)
    rep_path = OUT / "leakage.json"
    report = json.load(open(rep_path)) if rep_path.exists() else {}
    for name in a.sets.split(","):
        c = SETS[name]
        docs, items = TEST_BUILDERS[name]()
        by = {}
        for it in items:
            by.setdefault(it["doc_id"], []).append(it)
        ids = sorted(by)
        ntok = dict(zip(ids, count_tokens(tok, [docs[d]["text"] for d in ids], batch=4)))
        rows, bad, long_ = {}, set(), set()
        for d in ids:
            if ntok[d] > c["doc_cap"]:
                long_.add(d)
                continue
            # MuSR / templated questions repeat across docs by design: question hits only count for non-templated sets
            qs = [] if name in ("mcq_musr", "mcq_hallu") else [it["question"] for it in by[d]]
            rows[d] = R.check(docs[d]["text"], qs, a.thr)
            if rows[d]["reason"]:
                bad.add(d)
        keep = sorted(d for d in rows if d not in bad)
        if c["max_docs"] and len(keep) > c["max_docs"]:
            keep = sorted(random.Random(a.seed).sample(keep, c["max_docs"]))
        sel = [it for d in keep for it in by[d]]
        for j, it in enumerate(sel):
            it["key"] = f"{name}_test_{j:05d}"
        index = write_split(DATA, name, {"train": [], "val": [], "test": sel}, {d: docs[d] for d in keep}, tok)
        print_summary(name, {"test": sel}, index)
        texts = {d: docs[d]["text"] for d in keep}
        wm = wrong_map(keep, texts)
        reg = sorted(set(keep) | set(wm.values()))
        tasks = Counter(it["task"] for it in sel)
        S = {"task": name, "split_dir": f"data/{name}_split", "splits": ["test"], "max_tokens": c["max_tokens"],
             "doc_cap": c["doc_cap"], "n_items": len(sel), "tasks": dict(tasks), "n_leak_dropped_docs": len(bad),
             "n_too_long_docs": len(long_), "judged_docs": keep, "reg_docs": reg,
             "item_keys": sorted(it["key"] for it in sel), "wrong_map_file": f"outputs/MCQ_sets/{name}_wrongmap.json",
             "doc_tokens": {"min": min(ntok[d] for d in keep), "med": sorted(ntok[d] for d in keep)[len(keep) // 2],
                            "max": max(ntok[d] for d in keep)},
             "gold_label_dist": dict(Counter(it["correct_label"] for it in sel))}
        json.dump(S, open(OUT / f"{name}.json", "w"), indent=1)
        json.dump(wm, open(OUT / f"{name}_wrongmap.json", "w"), indent=1)
        reasons = Counter(rows[d]["reason"].split(":")[0].split(" ")[0] for d in bad)
        flagged = Counter((rows[d]["max_doc"] or "").split("/")[0] for d in bad)
        report[name] = {"n_candidate_docs": len(by), "too_long": len(long_), "dropped": len(bad),
                        "reasons": dict(reasons), "dropped_by_source": dict(flagged),
                        "max_union_kept": max([rows[d]["union_frac"] for d in keep] or [0]),
                        "max_pairfrac_kept": max([rows[d]["max_frac"] for d in keep] or [0]), "per_doc": rows}
        json.dump(report, open(rep_path, "w"), indent=1)
        print(f"{name}: {len(sel)} items {dict(tasks)}, {len(keep)} judged / {len(reg)} reg docs; {len(bad)} leaky docs "
              f"dropped {dict(reasons)} by source {dict(flagged)}, {len(long_)} over {c['doc_cap']} tok; kept max union "
              f"{report[name]['max_union_kept']:.3f} max pair {report[name]['max_pairfrac_kept']:.3f}; "
              f"tokens {S['doc_tokens']}; gold {S['gold_label_dist']}", flush=True)
    return 0


def phase_train(a):
    from ds_leakage import Reference
    from xenc.doc_tasks import load_qwen_tokenizer, print_summary, write_split, load_docs_index
    from register_d13 import excluded_docs
    os.chdir(REPO)
    tok = load_qwen_tokenizer()
    R = Reference()
    rep = {}
    # --- racemc / multirc: new text, DS's strict rule vs every val / test doc
    for name, fn in (("racemc", build_racemc), ("multirc", build_multirc)):
        docs, items, cap = fn()
        drop, why = set(), Counter()
        for d, v in docs.items():
            r = R.check(d, v["text"])
            if r:
                drop.add(d)
                why[r.split("=")[0].split(" ")[0]] += 1
        items = [it for it in items if it["doc_id"] not in drop]
        qhit = [it for it in items if R.q_hit(it["question"].split("\nProposed answer:")[0].replace("Question: ", ""))]
        items = [it for it in items if it not in qhit]
        keepd = sorted({it["doc_id"] for it in items}, key=lambda d: md5(docs[d]["text"]))
        if cap:
            keepd = set(keepd[:cap])
            items = [it for it in items if it["doc_id"] in keepd]
        keepd = sorted({it["doc_id"] for it in items})
        for j, it in enumerate(items):
            it["key"] = f"{name}_train_{j:06d}"
        index = write_split(DATA, name, {"train": items, "val": [], "test": []}, {d: docs[d] for d in keepd}, tok)
        print_summary(name, {"train": items}, index)
        rep[name] = {"candidate_docs": len(docs), "dropped_docs": len(drop), "reasons": dict(why),
                     "question_hits": len(qhit), "items": len(items), "docs": len(keepd),
                     "labels": dict(Counter(it["correct_label"] for it in items))}
        print(f"{name}: {rep[name]}", flush=True)
    # --- nqamc / looglemc: derived from docs DS already trains on (no new text); re-assert vs R anyway
    ex_sets = ",".join(sorted(str(p) for p in (REPO / "outputs" / "D22_sets").glob("*.json")
                              if not p.name.endswith(("_wrongmap.json", "_leakage.json"))))
    J = set(excluded_docs(ex_sets))
    X = json.load(open(REPO / "outputs" / "DS_leakage_phaseA.json"))
    J |= set(sum(X.values(), []))
    for name, src, keep_src, cap_tok in (("nqamc", "narrativeqa", None, 96000),
                                          ("looglemc", "loogle", ("shortdep_qa", "longdep_qa"), 96000)):
        sd = DATA / f"{src}_split"
        idx = load_docs_index(str(sd))
        src_items = json.load(open(sd / "train" / "items.json"))
        if keep_src:
            src_items = [it for it in src_items if (it.get("meta") or {}).get("source") in keep_src]
        texts = {}

        def ok_doc(d, idx=idx, sd=sd, texts=texts):
            if d in J or idx[d]["n_tokens"] > cap_tok:
                return False
            if d not in texts:
                texts[d] = (DATA / idx[d]["path"]).read_text()
                texts[d] = "" if R.check(d, texts[d]) else texts[d]
            return bool(texts[d])
        _, items = derived_mc(src_items, name, name, a.seed, ok_doc)
        items = [it for it in items if not R.q_hit(it["question"])]
        index = write_derived(name, items, src, {})
        print_summary(name, {"train": items}, index)
        rep[name] = {"source_items": len(src_items), "items": len(items), "docs": len(index),
                     "src_docs_rejected": sum(1 for v in texts.values() if not v), "excluded_ids": len(J),
                     "wh": dict(Counter(it["meta"]["wh"] for it in items)),
                     "labels": dict(Counter(it["correct_label"] for it in items))}
        print(f"{name}: {rep[name]}", flush=True)
    json.dump(rep, open(REPO / "outputs" / "MCQ_train_leakage.json", "w"), indent=1)
    return 0


def phase_verify(a):
    from register_doc import _get, load_split_items
    rep = json.load(open(OUT / "leakage.json"))
    for name in a.sets.split(","):
        S = json.load(open(OUT / f"{name}.json"))
        keys = set(S["item_keys"])
        items = [it for s in S["splits"] for it in load_split_items(S["split_dir"], s) if _get(it, "key") in keys]
        assert len(items) == len(keys), (name, len(items), len(keys))
        wm = json.load(open(S["wrong_map_file"]))
        assert set(wm) == set(S["judged_docs"]) and set(wm.values()) <= set(S["reg_docs"]), name
        assert not any(rep[name]["per_doc"][d]["reason"] for d in S["judged_docs"]), name
        assert all(w != d for d, w in wm.items()), name
        from xenc.doc_tasks import score, gold_text
        assert all(score(it, gold_text(it)) == 1.0 for it in items), f"{name}: gold text does not score 1"
        print(f"verify {name}: {len(items)} items, {len(S['judged_docs'])} judged / {len(S['reg_docs'])} reg docs, "
              f"wrong map ok, leakage clean, gold scores 1")
    # train splits: no doc id / text of any train split equals a judged doc of any MCQ / OOD / D22 set
    judged = set()
    for f in list(OUT.glob("mcq_*.json")) + list((REPO / "outputs" / "OOD_sets").glob("*.json")):
        if not f.name.endswith("_wrongmap.json") and f.name != "leakage.json":
            judged |= set(json.load(open(f)).get("reg_docs", []))
    for name in ("racemc", "multirc", "nqamc", "looglemc"):
        p = DATA / f"{name}_split" / "docs.json"
        if p.exists():
            ids = set(json.load(open(p)))
            assert not (ids & judged), name
            print(f"verify train {name}: {len(ids)} docs, none judged")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("test", "train", "verify"))
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    return {"test": phase_test, "train": phase_train, "verify": phase_verify}[a.phase](a)


if __name__ == "__main__":
    sys.exit(main())
