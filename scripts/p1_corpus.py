"""P1 (data scale): build the label-free distillation corpus manifest for the 4B->9B stage 2 (register_d13.py --p1_corpus).

Documents (train-only by construction, then leakage-filtered):
  (1) every benchmark's TRAIN split docs (QuALITY, MuSiQue, LastingBench, Qasper, ContractNLI, LooGLE, CLIPPER,
      Shakespeare, Shakespeare-CF, LongHealth folds 2-4) with their train questions (gold labels are kept but the
      P1 objective only uses the 9B text teacher);
  (2) the D15 pool (data/D15_selfq: LongBench v1/v2, CUAD, InfiniteBench, AA-LCR, NarrativeQA, Qasper-HF, ...) with
      its native + 9B self-questions;
  (3) new raw documents (data/raw_downloads/P1): PG-19 books (windowed), Wikipedia, PubMed (biomedical / clinical),
      arXiv papers, GovReport (government / legal). These get questions from scripts/p1_teacher.py --qgen.

Leakage guard (hard): reference set = every val AND test document of every data/*_split (LongHealth: folds 0,1),
every document of every outputs/D22_sets/*.json (judged, registered, wrong-map) and outputs/D13_lever1_leakage.json.
A candidate is dropped when its id is in the reference ids, its normalized-text md5 equals a reference doc's, its
word-8-gram containment in the union of reference 8-grams is >= --thr, or some reference doc j has c(j -> cand) >= --thr
(with >= --min_shared shared 8-grams). The surviving manifest is re-checked (assert) before it is written.

Output (--out, on /scratch): docs/<src>/<id>.md (new docs only), docs.json {doc_id: {path, n_tokens, src, md5,
group}}, questions.jsonl ({doc_id, items}), leakage_report.json, ref_md5.json.

    python scripts/p1_corpus.py --out /scratch/drjieliu_root/drjieliu/rickyhan/P1_corpus
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
import multiprocessing as mp
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SPLITS = ["quality", "musique", "lastingbench", "qasper", "contractnli", "loogle", "clipper", "shakespeare",
          "shakespeare_cf", "longhealth"]
RAW = "data/raw_downloads/P1"


def norm_md5(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", text.lower()).strip().encode()).hexdigest()


def grams(text: str, n: int = 8) -> set:
    w = text.split()
    return {hash(" ".join(w[i:i + n])) for i in range(max(len(w) - n + 1, 0))}


# --------------------------------------------------------------------------- references (never train)
def split_items(name: str, split: str) -> list:
    p = f"data/{name}_split/{split}/items.json"
    return json.load(open(p)) if os.path.exists(p) else []


def reference_docs() -> dict:
    """{doc_id: text} of every non-train document (val + test of every split, LongHealth folds 0/1, D22 sets)."""
    from register_d13 import excluded_docs
    from xenc.doc_tasks import load_docs
    ref, all_docs = {}, {}
    for sd in sorted(glob.glob("data/*_split")):
        name = Path(sd).name[:-6]
        D = load_docs(sd)
        all_docs.update(D)
        for split in ("val", "test"):
            for it in split_items(name, split):
                if name == "longhealth" and (it.get("meta") or {}).get("fold") not in (0, 1):
                    continue
                ref[it["doc_id"]] = D[it["doc_id"]]
    d22 = excluded_docs(",".join(f for f in sorted(glob.glob("outputs/D22_sets/*.json"))
                                 if "wrongmap" not in f and "leakage" not in f))
    miss = [d for d in d22 if d not in all_docs]
    for d in d22:
        if d in all_docs:
            ref[d] = all_docs[d]
    return ref, d22, miss


# --------------------------------------------------------------------------- candidates
def bench_train(lever1_drop: set) -> tuple[dict, dict]:
    from xenc.doc_tasks import load_docs
    docs, qs = {}, {}
    for name in SPLITS:
        D = load_docs(f"data/{name}_split")
        for it in split_items(name, "train"):
            if name == "longhealth" and (it.get("meta") or {}).get("fold") not in (2, 3, 4):
                continue
            d = it["doc_id"]
            if d in lever1_drop:
                continue
            docs[d] = {"text": D[d], "src": name, "group": "bench_train", "path": None}
            qs.setdefault(d, []).append({**it, "origin": "gold"})
    return docs, qs


def d15_pool() -> tuple[dict, dict]:
    from fit_d15 import load_pool
    from xenc.doc_tasks import _doc_path
    by_n, _, idx = load_pool("data/D15_selfq", use_selfq=False, use_native=True)
    by_s, _, _ = load_pool("data/D15_selfq", use_selfq=True, use_native=False)
    docs, qs = {}, {}
    for d, m in idx.items():
        p = _doc_path(Path("data/D15_selfq"), m["path"])
        docs[d] = {"text": p.read_text(), "src": "d15_" + str(m.get("src")), "group": "d15", "path": str(p.resolve())}
        qs[d] = [{**i, "origin": "native"} for i in by_n.get(d, [])] + [{**i, "origin": "selfq"} for i in by_s.get(d, [])]
    return docs, qs


def windows(text: str, words: int, max_w: int) -> list[str]:
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur, n = [], [], 0
    for p in paras:
        k = len(p.split())
        cur.append(p)
        n += k
        if n >= words:
            out.append("\n\n".join(cur))
            cur, n = [], 0
            if len(out) >= max_w:
                break
    if cur and n >= words // 2 and len(out) < max_w:
        out.append("\n\n".join(cur))
    return out


def _jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f]


def new_docs(a) -> dict:
    """New raw docs from --raw_jsonl (written by scripts/p1_parquet2jsonl.py; the qwen35 env has no pyarrow)."""
    rng = random.Random(0)
    docs = {}
    R = Path(a.raw_jsonl)
    for n in ("pg19", "wiki", "pubmed", "arxiv", "govreport"):          # fail loudly instead of a partial corpus
        assert (R / f"{n}.jsonl").stat().st_size > 0, f"missing raw input {R / n}.jsonl"

    def add(src, key, text):
        did = f"p1{src}_{hashlib.md5(key.encode()).hexdigest()[:12]}"
        docs[did] = {"text": text.strip(), "src": src, "group": "new", "path": None}

    # PG-19 books -> paragraph-aligned windows of ~--pg_words words (<= --pg_win per book)
    books = _jsonl(R / "pg19.jsonl")
    rng.shuffle(books)
    for b in books[: a.pg_books]:
        for j, w in enumerate(windows(b["text"], a.pg_words, a.pg_win)):
            add("pg19", f"{b['short_book_title']}#{j}", w)
    # Wikipedia (20231101.en shard 0): long articles only
    wk = [r for r in _jsonl(R / "wiki.jsonl") if len(r["text"].split()) >= a.min_words]
    rng.shuffle(wk)
    for r in wk[: a.n_wiki]:
        add("wiki", r["title"], f"# {r['title']}\n\n" + trunc(r["text"], a.max_words))
    for src, n in (("pubmed", a.n_pubmed), ("arxiv", a.n_arxiv), ("govreport", a.n_gov)):
        rows = [r for r in _jsonl(R / f"{src}.jsonl") if len(r["article"].split()) >= a.min_words]
        rng.shuffle(rows)
        for r in rows[:n]:
            add(src, r["article"][:2000], trunc(r["article"], a.max_words))
    return docs


def trunc(text: str, max_words: int) -> str:
    """Keep whole paragraphs up to max_words words."""
    out, n = [], 0
    for p in re.split(r"(\n+)", text):
        k = len(p.split())
        if n + k > max_words and n > 0:
            break
        out.append(p)
        n += k
    return "".join(out)


# --------------------------------------------------------------------------- leakage
_REF: dict = {}


def _init(ref_index, ref_sizes):
    _REF["idx"], _REF["size"] = ref_index, ref_sizes


def _check(arg):
    did, text = arg
    g = grams(text)
    hit = Counter()
    n_union = 0
    for x in g:
        r = _REF["idx"].get(x)
        if r is not None:
            n_union += 1
            if isinstance(r, tuple):
                hit.update(r)
            else:
                hit[r] += 1
    c_union = n_union / max(len(g), 1)
    best, best_j = 0.0, None
    for j, k in hit.items():
        c = k / max(_REF["size"][j], 1)
        if k >= 5 and c > best:
            best, best_j = c, j
    return did, len(g), c_union, best, best_j


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/scratch/drjieliu_root/drjieliu/rickyhan/P1_corpus")
    ap.add_argument("--thr", type=float, default=0.15)
    ap.add_argument("--raw_jsonl", default="/scratch/drjieliu_root/drjieliu/rickyhan/P1_corpus/raw_jsonl")
    ap.add_argument("--min_words", type=int, default=2500)
    ap.add_argument("--max_words", type=int, default=20000)
    ap.add_argument("--pg_books", type=int, default=700)
    ap.add_argument("--pg_words", type=int, default=12000)
    ap.add_argument("--pg_win", type=int, default=3)
    ap.add_argument("--n_wiki", type=int, default=1600)
    ap.add_argument("--n_pubmed", type=int, default=1300)
    ap.add_argument("--n_arxiv", type=int, default=900)
    ap.add_argument("--n_gov", type=int, default=900)
    ap.add_argument("--max_tokens", type=int, default=96000, help="drop docs longer than this (Qwen tokens)")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref, d22, miss = reference_docs()
    lever1 = json.load(open("outputs/D13_lever1_leakage.json"))
    lever1_drop = set(sum(lever1.values(), []))
    print(f"reference docs: {len(ref)} (val+test of every split, LH folds 0/1, D22 sets {len(d22)} ids, "
          f"{len(miss)} D22 ids not found in any split: {miss[:5]}); lever-1 near-dup drops {len(lever1_drop)}", flush=True)
    ref_ids = set(ref) | d22 | lever1_drop
    ref_md5 = {norm_md5(t) for t in ref.values()}
    # reference 8-gram index: gram -> ref doc id (or tuple of ids)
    idx, size = {}, {}
    for j, t in ref.items():
        g = grams(t)
        size[j] = len(g)
        for x in g:
            r = idx.get(x)
            if r is None:
                idx[x] = j
            elif isinstance(r, tuple):
                if j not in r:
                    idx[x] = r + (j,)
            elif r != j:
                idx[x] = (r, j)
    print(f"reference 8-grams: {len(idx)}", flush=True)

    bd, bq = bench_train(lever1_drop)
    pdocs, pq_ = d15_pool()
    nd = new_docs(a)
    cand, qs = {}, {}
    for D, Q in ((bd, bq), (pdocs, pq_), (nd, {})):
        for d, m in D.items():
            if d in cand:                      # D15 pool repeats split-train docs (same id): keep the first, merge questions
                qs[d] = qs.get(d, []) + [q for q in Q.get(d, []) if q.get("key") not in {x.get("key") for x in qs.get(d, [])}]
                continue
            cand[d] = m
            qs[d] = list(Q.get(d, []))
    print(f"candidates: {len(cand)} docs ({Counter(m['group'] for m in cand.values())})", flush=True)
    rep = {"n_ref_docs": len(ref), "n_ref_ids": len(ref_ids), "thr": a.thr, "dropped": {}, "kept": {}}
    drop = {}
    for d, m in cand.items():
        if d in ref_ids:
            drop[d] = "ref_id"
        elif not m["text"].strip():
            drop[d] = "empty"
        elif norm_md5(m["text"]) in ref_md5:
            drop[d] = "ref_md5"
    todo = [(d, m["text"]) for d, m in cand.items() if d not in drop]
    with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(idx, size)) as P:   # fork: same str-hash seed
        res = P.map(_check, todo, chunksize=16)
    ex = []
    for did, ng, cu, cb, j in res:
        if ng < 50:
            drop[did] = "too_short"
        elif cu >= a.thr:
            drop[did] = f"union_containment {cu:.3f}"
        elif cb >= a.thr:
            drop[did] = f"ref_in_cand {cb:.3f} ({j})"
        if did in drop and drop[did].startswith(("union", "ref_in")):
            ex.append((did, drop[did]))
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    keep = [d for d in cand if d not in drop]
    ntok = {}
    for i in range(0, len(keep), 64):
        b = keep[i:i + 64]
        for d, ids in zip(b, tok([cand[d]["text"] for d in b], add_special_tokens=False)["input_ids"]):
            ntok[d] = len(ids)
    for d in keep:
        if ntok[d] > a.max_tokens:
            drop[d] = f"too_long {ntok[d]}"
    keep = [d for d in keep if d not in drop]
    # re-check (hard): nothing kept is a reference doc by id or text
    assert not (set(keep) & ref_ids), "a reference (val/test/judged) doc id survived"
    assert not ({norm_md5(cand[d]["text"]) for d in keep} & ref_md5), "a reference doc text survived"
    docs_json = {}
    for d in keep:
        m = cand[d]
        path = m["path"]
        if path is None and m["group"] == "new":
            p = out / "docs" / m["src"] / f"{d}.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(m["text"])
            path = str(p)
        elif path is None:                      # split doc: resolve its file on turbo (no copy)
            from xenc.doc_tasks import _doc_path, load_docs_index
            sd = f"data/{m['src']}_split"
            path = str(_doc_path(Path(sd), load_docs_index(sd)[d]["path"]).resolve())
        docs_json[d] = {"path": path, "n_tokens": ntok[d], "src": m["src"], "group": m["group"], "md5": norm_md5(m["text"])}
    with open(out / "docs.json", "w") as f:
        json.dump(docs_json, f)
    nq = 0
    with open(out / "questions.jsonl", "w") as f:
        for d in keep:
            its = []
            for i in qs.get(d, []):
                its.append({k: i.get(k) for k in ("key", "doc_id", "question", "choices", "answers", "correct_label",
                                                  "task", "hop", "meta", "origin")})
            nq += len(its)
            f.write(json.dumps({"doc_id": d, "items": its}) + "\n")
    json.dump(sorted(ref_md5), open(out / "ref_md5.json", "w"))
    rep["dropped"] = Counter(v.split(" ")[0] for v in drop.values())
    rep["dropped_by_src"] = Counter(cand[d]["src"] for d in drop)
    rep["examples"] = ex[:200]
    rep["kept"] = Counter(docs_json[d]["src"] for d in keep)
    rep["kept_groups"] = Counter(docs_json[d]["group"] for d in keep)
    rep["n_docs"], rep["n_questions_existing"] = len(keep), nq
    rep["docs_without_questions"] = sum(1 for d in keep if not qs.get(d))
    rep["tokens"] = {"sum": sum(ntok[d] for d in keep), "median": sorted(ntok[d] for d in keep)[len(keep) // 2]}
    json.dump(rep, open(out / "leakage_report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "examples"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
