"""D15 step 1a: extract raw long documents (+ any native questions) from data/raw_downloads into
data/D15_selfq/raw/<source>.jsonl. Memory-lean (streams line by line); run on a compute node (the NarrativeQA
parquet needs pyarrow -> the kvskill env, or skip it with --skip narrativeqa).

One JSON line per source document: {"src", "sid", "text", "native": [{"question", "answers", "choices",
"correct_label"}]}. Native questions are only kept when they refer to the whole text (windowing in
d15_build_pool.py drops them for windowed docs). Never touches QuALITY / LongHealth (those come from the
splits' TRAIN docs in d15_build_pool.py, with fingerprint exclusion of every val/test document).

    python scripts/d15_extract_raw.py --out data/D15_selfq/raw
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import sys
import zipfile

RAW = "data/raw_downloads"
LB_QA = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "qmsum",
         "qasper_e", "multifieldqa_en_e", "hotpotqa_e", "2wikimqa_e"]
LB_NOQ = ["gov_report", "multi_news", "gov_report_e", "multi_news_e"]
LBV2_DOMAINS = {"Single-Document QA", "Multi-Document QA", "Long-dialogue History Understanding"}


def h(s: str) -> str:
    return hashlib.sha1(s[:4000].encode("utf-8", "ignore")).hexdigest()[:16]


def _ans(x):
    if isinstance(x, list):
        return [str(a) for a in x]
    try:
        v = ast.literal_eval(x)
        return [str(a) for a in v] if isinstance(v, (list, tuple)) else [str(v)]
    except Exception:
        return [str(x)]


class Writer:
    def __init__(self, out, src):
        self.f = open(os.path.join(out, f"{src}.jsonl"), "w")
        self.src, self.seen, self.n = src, {}, 0

    def add(self, text, native=None, sid=None):
        if not text or len(text) < 4000:
            return
        k = h(text)
        if k in self.seen:                      # same document, more native questions
            self.seen[k]["native"] += native or []
            return
        self.seen[k] = {"src": self.src, "sid": sid or k, "text": text, "native": list(native or [])}

    def close(self):
        for r in self.seen.values():
            self.f.write(json.dumps(r) + "\n")
            self.n += 1
        self.f.close()
        print(f"{self.src}: {self.n} docs, {sum(len(r['native']) for r in self.seen.values())} native q", flush=True)
        self.seen = {}


def longbench(out):
    z = zipfile.ZipFile(f"{RAW}/longbench/data.zip")
    for s in LB_QA + LB_NOQ:
        w = Writer(out, f"lb_{s}")
        with z.open(f"data/{s}.jsonl") as f:
            for line in io.TextIOWrapper(f, encoding="utf-8"):
                d = json.loads(line)
                if d.get("language", "en") != "en":
                    continue
                q = (d.get("input") or "").strip()
                nat = [{"question": q, "answers": _ans(d.get("answers")), "choices": None, "correct_label": None}] \
                    if (q and s in LB_QA) else []
                w.add(d["context"], nat)
        w.close()


def qasper(out):
    w = Writer(out, "qasper")
    for sp in ("train",):
        d = json.load(open(f"{RAW}/qasper/s3/qasper-{sp}-v0.3.json"))
        for pid, p in d.items():
            parts = [p.get("title", ""), "Abstract", p.get("abstract", "")]
            for sec in p.get("full_text", []):
                parts.append(sec.get("section_name") or "")
                parts += [x for x in sec.get("paragraphs", []) if x]
            nat = []
            for qa in p.get("qas", []):
                ans = []
                for a in qa.get("answers", []):
                    a = a.get("answer", {})
                    if a.get("unanswerable"):
                        continue
                    if a.get("extractive_spans"):
                        ans += a["extractive_spans"]
                    elif a.get("free_form_answer"):
                        ans.append(a["free_form_answer"])
                    elif a.get("yes_no") is not None:
                        ans.append("yes" if a["yes_no"] else "no")
                if ans:
                    nat.append({"question": qa["question"], "answers": ans[:3], "choices": None, "correct_label": None})
            w.add("\n\n".join(x for x in parts if x), nat, sid=pid)
        del d
    w.close()


def cuad(out):
    w = Writer(out, "cuad")
    z = zipfile.ZipFile(f"{RAW}/cuad/gh/data.zip")
    d = json.load(z.open("CUADv1.json"))
    for doc in d["data"]:
        for p in doc["paragraphs"]:
            w.add(p["context"], [], sid=h(doc["title"]))
    del d
    w.close()


def infinitebench(out):
    w = Writer(out, "infbench")
    for fn in ("longbook_qa_eng.jsonl", "longbook_choice_eng.jsonl"):
        with open(f"{RAW}/infinitebench/{fn}") as f:
            for line in f:
                d = json.loads(line)
                w.add(d["context"], [])           # whole-book questions are dropped (docs get windowed)
    w.close()


def lbv2(out):
    w = Writer(out, "lbv2")
    d = json.load(open(f"{RAW}/longbench_v2/data.json"))
    for x in d:
        if x.get("domain") not in LBV2_DOMAINS:
            continue
        ch = [{"label": L, "text": x[f"choice_{L}"]} for L in "ABCD"]
        w.add(x["context"], [{"question": x["question"], "answers": [x[f"choice_{x['answer']}"]], "choices": ch,
                              "correct_label": x["answer"]}], sid=x.get("_id"))
    del d
    w.close()


def aalcr(out):
    w = Writer(out, "aalcr")
    z = zipfile.ZipFile(f"{RAW}/aa_lcr/extracted_text/AA-LCR_extracted-text.zip")
    for i in z.infolist():
        if i.is_dir() or not i.filename.endswith(".txt"):
            continue
        w.add(z.read(i).decode("utf-8", "ignore"), [], sid=h(i.filename))
    w.close()


def narrativeqa(out):
    import pyarrow.parquet as pq
    w = Writer(out, "nqa")
    f = pq.ParquetFile(f"{RAW}/narrativeqa/data/test-00000-of-00008.parquet")
    for rg in range(f.metadata.num_row_groups):
        t = f.read_row_group(rg, columns=["document", "question", "answers"]).to_pylist()
        for r in t:
            doc = r["document"]
            w.add(doc["text"], [{"question": r["question"]["text"], "answers": [a["text"] for a in r["answers"]],
                                 "choices": None, "correct_label": None}], sid=doc["id"])
        del t
    w.close()


SOURCES = {"longbench": longbench, "qasper": qasper, "cuad": cuad, "infbench": infinitebench, "lbv2": lbv2,
           "aalcr": aalcr, "narrativeqa": narrativeqa}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/D15_selfq/raw")
    ap.add_argument("--only", default="")
    ap.add_argument("--skip", default="")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    only = [s for s in a.only.split(",") if s] or list(SOURCES)
    for s in only:
        if s in a.skip.split(","):
            continue
        try:
            SOURCES[s](a.out)
        except Exception as e:                       # one broken source must not kill the pool
            print(f"!! {s} failed: {type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
