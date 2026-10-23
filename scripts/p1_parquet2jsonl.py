"""P1: convert the downloaded parquet sources (data/raw_downloads/P1) to jsonl for p1_corpus.py (the qwen35 env has no
pyarrow; run this with an env that has it, e.g. envs/kvskill). Keeps only the text columns; docs shorter than
--min_words are dropped here already (p1_corpus applies the same floor).

    envs/kvskill/bin/python scripts/p1_parquet2jsonl.py --out /scratch/.../P1_corpus/raw_jsonl
"""
import argparse
import glob
import json
import os

import pyarrow.parquet as pq

RAW = "data/raw_downloads/P1"
SRC = {"pg19": ("emozilla__pg19/data/*.parquet", ["short_book_title", "text"]),
       "wiki": ("wikimedia__wikipedia/20231101.en/*.parquet", ["title", "text"]),
       "pubmed": ("ccdv__pubmed-summarization/document/*.parquet", ["article"]),
       "arxiv": ("ccdv__arxiv-summarization/document/*.parquet", ["article"]),
       "govreport": ("ccdv__govreport-summarization/document/*.parquet", ["report"])}   # written as "article"

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--min_words", type=int, default=2500)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
for name, (pat, cols) in SRC.items():
    n = k = 0
    with open(f"{a.out}/{name}.jsonl", "w") as f:
        for fn in sorted(glob.glob(f"{RAW}/{pat}")):
            for r in pq.read_table(fn, columns=cols).to_pylist():
                n += 1
                if "report" in r:
                    r = {"article": r["report"]}
                t = r.get("text") or r.get("article") or ""
                if len(t.split()) < a.min_words:
                    continue
                k += 1
                f.write(json.dumps(r) + "\n")
    print(name, "rows", n, "kept", k, flush=True)
