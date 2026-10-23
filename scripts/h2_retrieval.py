"""H2 / KV-RAG stage 0 (CPU): chunk the super-long documents, retrieve chunks per question, measure gold-evidence
chunk RECALL, and save the rankings that the GPU stage (scripts/register_h2.py) reads.

Chunks are token-aligned (Qwen3.5 tokenizer) windows of C tokens over the document text, leaving out the last `tail`
tokens (the reader prefills those itself, exactly as in the D13/D22 flat read). Every chunk size C is a multiple of
the 256-token passage grid, so passages nest inside chunks.

Retrievers (question text only; the gold answer / evidence is never used for retrieval):
  bm25    Okapi BM25 (k1 1.5, b .75) over chunk text, lowercase word tokens, stopwords removed
  dense   BAAI/bge-base-en-v1.5 (CLS, cosine) over 256-token passages; chunk score = max over its passages
  hybrid  reciprocal-rank fusion of bm25 and dense (k = 60)
  (latent router: GPU, scripts/register_h2.py --phase route)

Gold evidence (recall only):
  CLIPPER   the chapters named in the upstream true_reasoning ("Chapter 9: ..."), mapped to "CHAPTER <n>" headings
  LooGLE    meta.evidence strings (short-dep QA: one sentence; long-dep QA: a list), located by normalized exact match,
            else by the densest run of matching word 8-grams. Cloze items carry no evidence (recall n/a).
Queries: test items on their own doc AND on their wrong_doc (for the wrong_doc control), plus every D22 self-question
of the judged docs (registration selects chunks for those with the same retriever).

    python scripts/h2_retrieval.py --sets clipper,loogle_long --out outputs/H2_retrieval
"""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SETS = {"clipper": ("outputs/D22_sets/clipper.json", "outputs/D22_sets/clipper_wrongmap.json",
                    "outputs/D22_selfq/clipper.jsonl"),
        "loogle_long": ("outputs/D22_sets/loogle_long.json", "outputs/D22_sets/loogle_long_wrongmap.json",
                        "outputs/D22_selfq/loogle.jsonl"),
        # config selection set (VAL books, F1's pilot set): recall only unless its self-questions exist
        "clipper_val": ("outputs/F1_sets/clipper_val.json", "outputs/F1_sets/clipper_val_wrongmap.json",
                        "outputs/H2_selfq/clipper_val.jsonl")}
DEPS = "/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/SoftSkill/.deps_qwen35"
PASSAGE = 256
STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being it its this that
these those he she they them his her their we you i me my our your not no do does did done have has had having so than
then there here which who whom whose what when where why how all any each both few more most other some such only own
same too very can will just should would could may might must shall into over under again further once about above
below between through during before after up down out off also s t""".split())
TEMPLATE = [r"Is the following statement about the story true or false\?", r"Statement:",
            r"Below is a summary of part of the document in which named entities are replaced by <mask-k> placeholders\.",
            r"Which entity is <mask-\d+>\?", r"<mask-\d+>", r"Reorder the timeline of the following events:",
            r"Consider this statement about the document:", r"According to the document,?"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ============================================================================ items
def load_set(name):
    sp, wp, qp = SETS[name]
    S = json.load(open(sp))
    keys = set(S["item_keys"])
    items = [it for s in (S.get("splits") or [S.get("split", "test")]) for it in json.load(open(Path(S["split_dir"]) / s / "items.json"))
             if it["key"] in keys]
    assert len(items) == len(keys), (len(items), len(keys))
    wmap = json.load(open(wp))
    selfq = {}
    for line in (open(qp) if os.path.exists(qp) else []):
        r = json.loads(line)
        selfq[r["doc_id"]] = r["items"]
    return S, items, wmap, selfq


def query_text(it) -> str:
    q = str(it["question"])
    for t in TEMPLATE:
        q = re.sub(t, " ", q)
    q = q.replace('"', " ")
    opts = [c["text"] for c in (it.get("choices") or []) if len(str(c["text"])) > 15]
    return " ".join([q] + opts).strip()


# ============================================================================ chunks
def doc_token_starts(tok, text, tail):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    starts = [s for s, _ in enc["offset_mapping"]]
    n = len(starts) - tail                     # body tokens (the last `tail` are the reader's own rows)
    return starts, n


def chunk_spans(starts, n, C, text_len):
    """Char spans [cs, ce) of token windows of C tokens over body tokens [0, n)."""
    out = []
    for t in range(0, n, C):
        cs = 0 if t == 0 else starts[t]
        ce = starts[t + C] if t + C < n else starts[n] if n < len(starts) else text_len
        out.append((cs, ce))
    return out


# ============================================================================ gold evidence
def _roman(s):
    v = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    s, t = s.upper(), 0
    for a, b in zip(s, s[1:] + " "):
        t += -v[a] if b != " " and v.get(b, 0) > v[a] else v[a]
    return t


def _num(x):
    return int(x) if x.isdigit() else _roman(x)


def clipper_evidence(items, docs):
    sys.path.append(DEPS)
    import pyarrow.parquet as pq
    ids = {it["meta"]["upstream_id"] for it in items}
    R = {}
    for f in sorted(glob.glob(str(_ROOT / "data/raw/clipper/data/*.parquet"))):
        for b in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["id", "true_reasoning"]):
            for r in b.to_pylist():
                if r["id"] in ids:
                    R[r["id"]] = r["true_reasoning"]
    assert len(R) == len(ids), (len(R), len(ids))
    heads = {}
    for did in {it["doc_id"] for it in items}:
        t = docs[did]
        hs = [(m.start(), _num(m.group(1))) for m in re.finditer(r"(?im)^[ \t]*chapter[ \t]+([0-9]+|[ivxlcdm]+)\b", t)]
        cnt = Counter(n for _, n in hs)
        assert all(v == 1 for v in cnt.values()), f"duplicate chapter numbers in {did}"
        ends = [s for s, _ in hs[1:]] + [len(t)]
        heads[did] = {n: (s, e) for (s, n), e in zip(hs, ends)}
    ev = {}
    for it in items:
        txt = R[it["meta"]["upstream_id"]].split("<answer>")[0]
        nums = set()
        for m in re.finditer(r"Chapters?\s+((?:(?:[0-9]+|[IVXLC]+)(?:\s*(?:,|and|&|-)\s*)?)+)", txt):
            nums |= {_num(x) for x in re.findall(r"[0-9]+|[IVXLC]+", m.group(1))}
        H = heads[it["doc_id"]]
        ev[it["key"]] = [H[n] for n in sorted(nums) if n in H]
    return ev


def _norm_words(t):
    return [(m.group(0).lower(), m.start(), m.end()) for m in re.finditer(r"\w+", t)]


def locate(evid: str, doc: str, dwords, gram_index):
    """-> (cs, ce) char span of evidence in doc, or None."""
    k = doc.find(evid.strip())
    if k >= 0 and len(evid.strip()) > 20:
        return (k, k + len(evid.strip()))
    ew = [w for w, _, _ in _norm_words(evid)]
    if len(ew) < 8:
        g = tuple(ew)
        hits = gram_index.get(("short", g))
        return None if not hits else (dwords[hits[0]][1], dwords[hits[0] + len(ew) - 1][2])
    pos = []
    for i in range(len(ew) - 7):
        for p in gram_index.get(tuple(ew[i:i + 8]), ())[:50]:
            pos.append(p - i)                         # implied evidence start (word index)
    if not pos:
        return None
    c = Counter(pos)
    best = max(c, key=lambda p: sum(c[q] for q in range(p - 20, p + 21)))
    lo = max(0, best)
    hi = min(len(dwords) - 1, best + len(ew) - 1)
    return (dwords[lo][1], dwords[hi][2])


def loogle_evidence(items, docs):
    ev, stats = {}, Counter()
    by_doc = defaultdict(list)
    for it in items:
        by_doc[it["doc_id"]].append(it)
    for did, its in by_doc.items():
        doc = docs[did]
        dw = _norm_words(doc)
        gi = defaultdict(list)
        words = [w for w, _, _ in dw]
        for i in range(len(words) - 7):
            gi[tuple(words[i:i + 8])].append(i)
        for it in its:
            e = it["meta"].get("evidence")
            if not e:
                continue
            es = [e] if isinstance(e, str) else list(e)
            spans = []
            for x in es:
                ew = [w for w, _, _ in _norm_words(x)]
                if 0 < len(ew) < 8:
                    for i in range(len(words) - len(ew) + 1):
                        if words[i:i + len(ew)] == ew:
                            gi[("short", tuple(ew))].append(i)
                            break
                sp = locate(x, doc, dw, gi)
                stats["found" if sp else "missing"] += 1
                if sp:
                    spans.append(sp)
            ev[it["key"]] = spans
    return ev, stats


# ============================================================================ retrievers
def words(t):
    return [w for w in re.findall(r"\w+", t.lower()) if w not in STOP and len(w) > 1]


class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs_tokens]
        self.len = [len(d) for d in docs_tokens]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        N = len(docs_tokens)
        self.idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def scores(self, q):
        out = []
        qs = Counter(q)
        for c, L in zip(self.tf, self.len):
            s = 0.0
            for w, qn in qs.items():
                f = c.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out.append(s)
        return out


class Dense:
    def __init__(self, name="BAAI/bge-base-en-v1.5", threads=8):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModel.from_pretrained(name).eval()

    def embed(self, texts, bs=32, query=False):
        torch = self.torch
        pre = "Represent this sentence for searching relevant passages: " if query else ""
        out = []
        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                b = self.tok([pre + t for t in texts[i:i + bs]], padding=True, truncation=True, max_length=512,
                             return_tensors="pt")
                h = self.m(**b).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(h.float(), dim=-1))
        return torch.cat(out) if out else torch.zeros(0, 768)


def rrf(*rankings, k=60):
    s = defaultdict(float)
    for r in rankings:
        for i, c in enumerate(r):
            s[c] += 1.0 / (k + i + 1)
    return sorted(s, key=lambda c: -s[c])


# ============================================================================ recall
def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def recall_item(spans, chunks, sel):
    """-> (frac units hit, all hit, frac of evidence chars covered)."""
    hit = [any(overlap(u, chunks[c]) > 0 for c in sel) for u in spans]
    tot = sum(u[1] - u[0] for u in spans)
    cov = sum(sum(overlap(u, chunks[c]) for c in sel) for u in spans) / max(tot, 1)
    return sum(hit) / len(hit), float(all(hit)), min(cov, 1.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="clipper,loogle_long")
    ap.add_argument("--chunks", default="512,1024,2048,4096,8192")
    ap.add_argument("--ks", default="1,2,3,4,6,8,16,32")
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--topn", type=int, default=32, help="ranks saved per query")
    ap.add_argument("--dense", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--no_dense", type=int, default=0)
    ap.add_argument("--max_docs", type=int, default=0, help="smoke: first K judged docs only")
    ap.add_argument("--out", default="outputs/H2_retrieval")
    a = ap.parse_args(argv)
    from xenc.doc_tasks import load_docs, load_qwen_tokenizer
    os.makedirs(a.out, exist_ok=True)
    tok = load_qwen_tokenizer()
    Cs = [int(x) for x in a.chunks.split(",")]
    Ks = [int(x) for x in a.ks.split(",")]
    assert all(C % PASSAGE == 0 for C in Cs)
    dense = None if a.no_dense else Dense(a.dense)
    md = ["# H2 / KV-RAG stage 0: gold-evidence chunk recall (CPU)", ""]
    for name in a.sets.split(","):
        t0 = time.time()
        S, items, wmap, selfq = load_set(name)
        docs = load_docs(S["split_dir"])
        dids = sorted({it["doc_id"] for it in items})
        jd = S["judged_docs"]
        assert (len(jd) if isinstance(jd, list) else int(jd)) == len(dids), ("judged_docs mismatch", jd, len(dids))
        if a.max_docs:
            dids = dids[: a.max_docs]
            items = [it for it in items if it["doc_id"] in set(dids)]
        alld = sorted(set(dids) | {wmap[d] for d in dids})
        if name.startswith("clipper"):
            ev = clipper_evidence(items, docs)
            ev_stats = {"items_with_chapters": sum(bool(v) for v in ev.values()), "n": len(items)}
        else:
            ev, st = loogle_evidence(items, docs)
            ev_stats = dict(st, items_with_evidence=sum(bool(v) for v in ev.values()), n=len(items))
        log(name, "evidence", ev_stats)
        # queries: (qid, doc, text)
        Q = []
        for it in items:
            Q.append((f"{it['doc_id']}|{it['key']}", it["doc_id"], query_text(it)))
            w = wmap[it["doc_id"]]
            Q.append((f"{w}|{it['key']}", w, query_text(it)))
        for d in alld:
            for it in selfq.get(d, []):
                Q.append((f"{d}|{it['key']}", d, query_text(it)))
        Q = list(dict.fromkeys(Q))
        # passages (dense) once per doc
        tokinfo = {d: doc_token_starts(tok, docs[d], a.tail) for d in alld}
        pemb = {}
        if dense is not None:
            for d in alld:
                starts, n = tokinfo[d]
                ps = chunk_spans(starts, n, PASSAGE, len(docs[d]))
                pemb[d] = dense.embed([docs[d][s:e] for s, e in ps])
            log(name, f"passages embedded ({time.time() - t0:.0f}s)")
            qemb = dense.embed([q for _, _, q in Q], query=True)
        res = {"set": name, "tail": a.tail, "passage": PASSAGE, "evidence": {k: v for k, v in ev.items()},
               "ev_stats": ev_stats, "doc_tokens": {d: tokinfo[d][1] + a.tail for d in alld}, "chunks": {}, "rank": {}}
        for C in Cs:
            res["chunks"][C] = {d: chunk_spans(tokinfo[d][0], tokinfo[d][1], C, len(docs[d])) for d in alld}
            bm = {d: BM25([words(docs[d][s:e]) for s, e in res["chunks"][C][d]]) for d in alld}
            R = {"bm25": {}, "dense": {}, "hybrid": {}}
            for j, (qid, d, q) in enumerate(Q):
                nch = len(res["chunks"][C][d])
                sb = bm[d].scores(words(q))
                rb = sorted(range(nch), key=lambda c: -sb[c])
                R["bm25"][qid] = rb[: a.topn]
                if dense is not None:
                    sp = (pemb[d] @ qemb[j]).tolist()
                    per = C // PASSAGE
                    sc = [max(sp[c * per:(c + 1) * per]) for c in range(nch)]
                    rd = sorted(range(nch), key=lambda c: -sc[c])
                    R["dense"][qid] = rd[: a.topn]
                    R["hybrid"][qid] = rrf(rb, rd)[: a.topn]
            res["rank"][C] = {k: v for k, v in R.items() if v}
        json.dump(res, open(os.path.join(a.out, f"{name}.json"), "w"))
        # ---- recall table (own-doc test items with evidence)
        rng = random.Random(0)
        md += [f"## {name}: {ev_stats}", "",
               "| C (tokens) | chunks/doc | retriever | " + " | ".join(f"R@{k} (budget {k}×C)" for k in Ks) + " |",
               "|---|---|---|" + "---|" * len(Ks)]
        table = {}
        evit = [it for it in items if ev.get(it["key"])]
        for C in Cs:
            nchd = sum(len(res["chunks"][C][d]) for d in dids) / len(dids)
            for r in list(res["rank"][C]) + ["random"]:
                cells = []
                for k in Ks:
                    vals = []
                    for it in evit:
                        ch = res["chunks"][C][it["doc_id"]]
                        if r == "random":
                            sel = rng.sample(range(len(ch)), min(k, len(ch)))
                        else:
                            sel = res["rank"][C][r][f"{it['doc_id']}|{it['key']}"][:k]
                        vals.append(recall_item(ev[it["key"]], ch, sel))
                    m = [sum(v[i] for v in vals) / max(len(vals), 1) for i in range(3)]
                    table[f"{C}|{r}|{k}"] = {"unit": m[0], "all": m[1], "cov": m[2], "n": len(vals)}
                    cells.append(f"{m[0]:.2f}/{m[1]:.2f}/{m[2]:.2f}" + ("" if k * C <= 16384 else "*"))
                md.append(f"| {C} | {nchd:.0f} | {r} | " + " | ".join(cells) + " |")
        md += ["", "cell = unit recall / all-units recall / evidence-char coverage; * = k·C > 16k tokens (outside the base's operator regime)", ""]
        # subsets
        if name.startswith("clipper"):
            subs = {"single": [it for it in evit if it["meta"]["claim_type"] == "single"],
                    "multiple": [it for it in evit if it["meta"]["claim_type"] != "single"]}
        else:
            subs = {"shortdep": [it for it in evit if it["meta"]["source"] == "shortdep_qa"],
                    "longdep": [it for it in evit if it["meta"]["source"] == "longdep_qa"]}
        md += ["| subset | n | C | retriever | R@k at budget 8k | R@k at budget 16k |", "|---|---|---|---|---|---|"]
        for sn, its in subs.items():
            for C in Cs:
                for r in res["rank"][C]:
                    cells = []
                    for B in (8192, 16384):
                        k = max(1, B // C)
                        vals = [recall_item(ev[it["key"]], res["chunks"][C][it["doc_id"]],
                                            res["rank"][C][r][f"{it['doc_id']}|{it['key']}"][:k]) for it in its]
                        cells.append(f"k={k}: " + "/".join(f"{sum(v[i] for v in vals) / max(len(vals), 1):.2f}"
                                                           for i in range(3)))
                    md.append(f"| {sn} | {len(its)} | {C} | {r} | " + " | ".join(cells) + " |")
        md.append("")
        json.dump(table, open(os.path.join(a.out, f"{name}_recall.json"), "w"), indent=1)
        log(name, f"done ({time.time() - t0:.0f}s), {len(Q)} queries")
    open(os.path.join(a.out, "RECALL_" + a.sets.replace(",", "_") + ".md"), "w").write("\n".join(md))
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
