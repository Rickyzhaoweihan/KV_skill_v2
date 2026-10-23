#!/usr/bin/env python
"""Encode a split's documents with an ENCODER into the ρ-keyed operator cache (one .pt per doc).

    python scripts/encode_docs.py --encoder Qwen/Qwen3.5-2B --split_dir data/quality_split \
        --cache_dir /scratch/.../xenc_cache --chunk_tokens 512 --depths 0.25,0.5,0.75,1.0 --max_docs 0
    python scripts/encode_docs.py --tiny          # CPU, TinyLM + fake tokenizer + 3 synthetic docs

Files are named `cache_key(doc_text, encoder_short, chunk_tokens, depths).pt`; existing files are
skipped. Writes `<cache_dir>/index_<split>_<encoder_short>.json` mapping doc_id -> cache file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
import xenc  # noqa: E402
from xenc.depth_skill import (TinyTokenizer, cache_key, derive_by_depth,  # noqa: E402
                              load_doc_cache, save_doc_cache)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder", default="tiny", help="HF hub id, or 'tiny'")
    p.add_argument("--split_dir", default=None)
    p.add_argument("--cache_dir", default=None, help=f"default $XENC_CACHE or {xenc.DEFAULT_CACHE}")
    p.add_argument("--chunk_tokens", type=int, default=512)
    p.add_argument("--depths", default=",".join(str(r) for r in xenc.DEPTHS),
                   help="comma-separated normalized depths. Default = xenc.DEPTHS, what `register_doc.py "
                        "--selector 0` reads. For `register_doc.py --selector 1` encode at the CAND depths: "
                        "--depths " + ",".join(str(r) for r in xenc.CAND))
    p.add_argument("--splits", default="",
                   help="comma list of splits (e.g. train,val) whose docs to encode; --max_docs then applies "
                        "PER SPLIT to its sorted doc_ids (matches register_doc/eval_doc --max_docs). "
                        "Default: every doc in docs.json")
    p.add_argument("--max_docs", type=int, default=0, help="encode only the first N sorted doc_ids (0 = all)")
    p.add_argument("--harvest", choices=["resid", "attn", "attn_raw"], default="resid",
                   help="resid: residual stream (tied K=V); attn: encoder attention keys/values (untied, D3)")
    p.add_argument("--max_doc_tokens", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_mem", default=None)
    p.add_argument("--tiny", action="store_true", help="TinyLM + fake tokenizer + 3 fake docs, CPU")
    a = p.parse_args(argv)
    depths = sorted(float(x) for x in a.depths.split(","))

    if a.tiny:
        sys.path.insert(0, str(_ROOT / "tests"))
        from _tiny import TinyAttnLM, TinyLM
        model = TinyAttnLM(d=32, n_layers=8) if a.harvest.startswith("attn") else TinyLM(d=32, n_layers=8)
        tok = TinyTokenizer(vocab=50)
        enc_short = "tiny"
        rng = __import__("random").Random(0)
        words = [f"w{i}" for i in range(200)]
        docs = {f"tinydoc_{i}": " ".join(rng.choice(words) for _ in range(40 + 70 * i)) for i in range(3)}
        cache_dir = a.cache_dir or tempfile.mkdtemp(prefix="xenc_tiny_cache_")
        chunk_tokens = a.chunk_tokens if a.chunk_tokens != 512 else 32
        split_name = "tiny"
    else:
        if not a.split_dir:
            p.error("--split_dir is required unless --tiny")
        from xenc.doc_tasks import load_doc_split, load_docs   # agent [C]
        from xenc.backbone import load_encoder                  # agent [D]
        docs = load_docs(a.split_dir)
        if a.splits:
            sel: list[str] = []
            for sp in [x.strip() for x in a.splits.split(",") if x.strip()]:
                ds = sorted({it.doc_id for it in load_doc_split(a.split_dir, sp)})
                sel += ds[: a.max_docs] if a.max_docs else ds
            docs = {d: docs[d] for d in dict.fromkeys(sel) if d in docs}
            a.max_docs = 0                                       # already applied per split
        model, tok = load_encoder(a.encoder, device_map=a.device, max_mem=a.max_mem)
        enc_short = a.encoder.rstrip("/").split("/")[-1]
        cache_dir = a.cache_dir or xenc.DEFAULT_CACHE
        chunk_tokens = a.chunk_tokens
        split_name = Path(a.split_dir.rstrip("/")).name

    os.makedirs(cache_dir, exist_ok=True)
    ids = sorted(docs)
    if a.max_docs:
        ids = ids[: a.max_docs]
    index, n_new, n_skip, n_chunks, n_tok, t0 = {}, 0, 0, 0, 0, time.time()
    for k, doc_id in enumerate(ids):
        text = docs[doc_id]
        path = os.path.join(cache_dir, cache_key(text, enc_short, chunk_tokens, depths, harvest=a.harvest) + ".pt")
        index[doc_id] = os.path.basename(path)
        if os.path.exists(path):
            n_skip += 1
            continue
        d = derive_by_depth(model, tok, text, depths, chunk_tokens, a.max_doc_tokens,
                            doc_id=doc_id, encoder=a.encoder if not a.tiny else "tiny", harvest=a.harvest)
        save_doc_cache(d, path)
        n_new += 1
        n_chunks += len(d["chunks"])
        n_tok += sum(c["n_tokens"] for c in d["chunks"])
        print(f"[{k + 1}/{len(ids)}] {doc_id}: {sum(c['n_tokens'] for c in d['chunks'])} tok, "
              f"{len(d['chunks'])} chunks -> {os.path.basename(path)}", flush=True)

    hsuf = "" if a.harvest == "resid" else f"_h{a.harvest}"
    idx_path = os.path.join(cache_dir, f"index_{split_name}_{enc_short}{hsuf}.json")
    with open(idx_path, "w") as f:
        json.dump(index, f, indent=1)
    if a.tiny and index:   # round-trip sanity on one file
        d = load_doc_cache(os.path.join(cache_dir, next(iter(index.values()))))
        assert d["chunks"] and d["depths"] == depths
    import torch
    if torch.cuda.is_available():
        print(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"encoded {n_new} new, skipped {n_skip} existing, {n_chunks} chunks / {n_tok} tokens in "
          f"{time.time() - t0:.1f}s; cache_dir={cache_dir}; index={idx_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
