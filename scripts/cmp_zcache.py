"""CMP hybrid on cached-operator runtimes (27B): precompute the stored per-token codes z of every document a judge
set reads (own + wrong documents), with the ENCODER (9B) over the reader runtime's exact operator prefix.

z = E (x − μ) of the compressor file's encoder group (xenc/cmp_op.OpCompressor.encode; x = the encoder's pre-RoPE K/V
at the source layers), stored fp16 [N, d] per document under xenc.cmp_hyb.zpath(out, did) with N == the operator's
column count. The prefix is fit_d12p.D13Rt.pre (doc_prefix_text with the READER as hub, last max_tokens − 1024
tokens, minus the tail); where a cached operator (cache_d12p.py) stores token ids they are asserted identical.
Sets use eval_d12p's item selection (split / fold / n / seed / d22_set) and wrong-document rule (wrong_map, else
eval_doc.wrong_doc_id over the set's document order).

    python scripts/cmp_zcache.py --cmp outputs/CMP_fit27/cmp_...2048.pt --sets qasper,loogle,... --out $XENC_CACHE/CMP_z27
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402

D22 = ("qasper", "loogle", "loogle_long", "shakespeare_cf", "lastingbench", "musique", "contractnli", "clipper")


def set_spec(s: str) -> dict:
    """The D12b judge sets (slurm/D12b_max8k.sbatch SPEC)."""
    if s == "lh":
        return {"split_dir": "data/longhealth_split", "fold": 0, "n": 0}
    if s == "q300":
        return {"split_dir": "data/quality_split", "n": 300}
    if s in ("cqacf", "fecf"):
        return {"split_dir": f"data/d25_{s}_split", "n": 0}
    if s in D22:
        S = json.load(open(f"outputs/D22_sets/{s}.json"))
        return {"split_dir": S["split_dir"], "d22_set": f"outputs/D22_sets/{s}.json",
                "wrong_map": f"outputs/D22_sets/{s}_wrongmap.json", "splits": S.get("splits")}
    if s.startswith("mcq_"):
        S = json.load(open(f"outputs/MCQ_sets/{s}.json"))
        return {"split_dir": S["split_dir"], "d22_set": f"outputs/MCQ_sets/{s}.json",
                "wrong_map": f"outputs/MCQ_sets/{s}_wrongmap.json"}
    raise ValueError(s)


def set_docs(s: str, seed: int = 0):
    """-> (doc ids read by the set's kv and wrong_doc conditions, docs {id: text})."""
    from eval_doc import wrong_doc_id
    from xenc.doc_tasks import load_docs
    sp = set_spec(s)
    items = load_split_items(sp["split_dir"], "test")
    if isinstance(sp.get("splits"), list) and sp["splits"]:
        items = [i for s_ in sp["splits"] for i in load_split_items(sp["split_dir"], s_)]
    doc_order = sorted({_get(i, "doc_id") for i in items})
    if sp.get("fold") is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == sp["fold"]]
    if sp.get("d22_set"):
        keys = set(json.load(open(sp["d22_set"]))["item_keys"])
        items = [i for i in items if _get(i, "key") in keys]
    elif sp.get("n") and len(items) > sp["n"]:
        items = random.Random(seed).sample(items, sp["n"])
    wmap = json.load(open(sp["wrong_map"])) if sp.get("wrong_map") else {}
    need = set()
    for it in items:
        d = _get(it, "doc_id")
        need |= {d, wmap.get(d) or wrong_doc_id(d, doc_order)}
    return sorted(need), load_docs(sp["split_dir"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cmp", required=True, help="compressor file whose encoder group defines z (largest d)")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--reader", default="Qwen/Qwen3.5-27B", help="tokenizer / chat template of the operator prefix")
    ap.add_argument("--sets", required=True)
    ap.add_argument("--max_tokens", type=int, default=50000)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--enc_cache", default="", help="cache_d12p dirs (comma) whose stored token ids are checked")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    from fit_d12p import find_op
    from transformers import AutoTokenizer
    from xenc.backbone import load_encoder
    from xenc.cmp_hyb import zpath
    from xenc.cmp_op import load_cmp, rope_tables
    from xenc.d13_read import EncTeacher
    from xenc.state_skill import doc_prefix_text
    from xenc.xattn_read import _rotary
    cmp = load_cmp(a.cmp).to(a.device)
    assert len(cmp.groups) == 1, cmp.groups
    src = cmp.groups[0]
    tok = AutoTokenizer.from_pretrained(a.reader)
    enc = load_encoder(a.encoder, device_map=a.device)[0]
    rot = _rotary(enc)
    Tc = EncTeacher(enc, src)
    caches = [c for c in a.enc_cache.split(",") if c]
    todo: dict = {}
    for s in [x for x in a.sets.split(",") if x]:
        need, docs = set_docs(s)
        for d in need:
            todo.setdefault(d, docs[d])
        log(f"  {s}: {len(need)} documents (own + wrong)")
    log(f"CMP zcache: {len(todo)} documents, source layers {src}, d {cmp.dims[0]}, encoder {a.encoder}, prefix of "
        f"{a.reader} (max_tokens {a.max_tokens}, tail {a.tail}) -> {a.out}")
    t0, n_chk, n_new = time.time(), 0, 0
    for i, (did, text) in enumerate(sorted(todo.items())):
        p = zpath(a.out, did)
        if os.path.exists(p):
            continue
        ids = [int(x) for x in tok(doc_prefix_text(tok, a.reader, text, SYSTEM_ANSWER), add_special_tokens=False)["input_ids"]]
        ids = ids[-(a.max_tokens - 1024):]
        ids = ids[: -a.tail] if a.tail else ids
        if caches:
            cp = find_op(caches, did)
            if cp is not None:
                b = torch.load(cp, map_location="cpu", weights_only=False, mmap=True)
                assert int(b["n"]) == len(ids), f"{did}: cached operator N {b['n']} != prefix {len(ids)}"
                if b.get("ids") is not None:
                    assert list(b["ids"]) == ids, f"{did}: cached operator token ids differ from this prefix"
                n_chk += 1
        Tc.rec, Tc.kv_only = {}, True
        with torch.no_grad():
            enc(input_ids=torch.tensor([ids], device=a.device), use_cache=False, logits_to_keep=1)
            op = {e: (r["k"], r["v"]) for e, r in Tc.rec.items()}
            Tc.rec = None
            cos, sin = rope_tables(rot, len(ids), a.device)
            z = cmp.encode(op, cos, sin)[0]
        os.makedirs(os.path.dirname(p), exist_ok=True)
        torch.save({"z": z.half().cpu(), "n": len(ids), "cmp": a.cmp, "encoder": a.encoder}, p + ".tmp")
        os.replace(p + ".tmp", p)
        n_new += 1
        del op, z
        if i % 25 == 0:
            log(f"  [{i + 1}/{len(todo)}] {did} N={len(ids)} | {time.time() - t0:.0f}s")
    log(f"done: {n_new} new codes, {n_chk} prefixes checked against cached operators ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
