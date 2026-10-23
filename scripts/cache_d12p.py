"""D12p phase A: run ONLY the encoder and cache everything the 27B-reader stages need from it, so that phase B
(fit_d12p.py / register_d12p.py / eval_d12p.py with --enc_cache DIR) never holds the encoder on a GPU.

Cached (bf16, CPU tensors; rows built by the SAME D13Rt.pre/suf/rows code with the READER's tokenizer/template):
  op/<doc>.pt   {"kv": {enc_layer: (K, V) [Hkv,N,hd] post-RoPE}, "n": N}       operator of every needed document
  rec/<key>.pt  {"rows": rows + gold, "rec": {enc_layer: {q, k, out[, x]}}}     encoder teacher pass over the student
                rows WITH the gold answer appended; key = row_key(doc, rows WITHOUT the answer). Causal, so the
                question-only rows (stage 1, val attention) are the prefix of the same record (CachedRt.teacher).
Item sets reproduce fit_d12p (stage 1), register_d12p (stage 2) and eval_d12p (judge: operators only, incl. the
wrong-doc partner of every judged item).

    python scripts/cache_d12p.py --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B --out $XENC_CACHE/D12p/c9b
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import _get, load_split_items, log  # noqa: E402


def layer_map_from_hubs(reader_hub: str, enc, layers: str) -> dict:
    """Reader layer -> encoder layer by depth, from the reader's CONFIG only (the reader is never loaded)."""
    from transformers import AutoConfig
    from xenc.xattn_read import attn_dims, full_depth_map
    fake = types.SimpleNamespace(config=AutoConfig.from_pretrained(reader_hub))
    keep = {int(x) for x in layers.split(",")}
    return {k: v for k, v in full_depth_map(attn_dims(fake), attn_dims(enc)).items() if k in keep}


def eval_docs(spec: str, seed: int) -> set:
    """spec "data/longhealth_split:test:0:0;data/quality_split:test::300" (dir:split:fold:n) -> docs eval_d12p reads."""
    from eval_doc import wrong_doc_id
    need = set()
    for s in [x for x in spec.split(";") if x]:
        sd, split, fold, n = s.split(":")
        items = load_split_items(sd, split)
        order = sorted({_get(i, "doc_id") for i in items})
        if fold != "":
            items = [i for i in items if (_get(i, "meta") or {}).get("fold") == int(fold)]
        if int(n) and len(items) > int(n):
            items = random.Random(seed).sample(items, int(n))
        for it in items:
            d = _get(it, "doc_id")
            need.add((sd, d))
            w = wrong_doc_id(d, order)
            if w is not None:
                need.add((sd, w))
    return need


def _needs(path: str, need_x: bool) -> bool:
    """A record must be (re)written if it is missing, or if stage 1 needs the encoder layer input "x" and an earlier
    (stage-2-only) pass stored the record without it."""
    if not os.path.exists(path):
        return True
    if not need_x:
        return False
    b = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return any("x" not in r for r in b["rec"].values())


def by_doc_keys(want: dict) -> set:
    return {d for d, _, _ in want.values()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--layers", default="3,7,11,15,19,23,27,31")
    ap.add_argument("--s1_train", default="quality:120,musique:60,lastingbench:60,longhealth:0")
    ap.add_argument("--s1_val", default="quality:6,musique:12,lastingbench:12,longhealth:0")
    ap.add_argument("--q_per_doc", type=int, default=8)
    ap.add_argument("--val_q_per_doc", type=int, default=4)
    ap.add_argument("--s2_train", default="quality:120,musique:60,lastingbench:60,longhealth:0")
    ap.add_argument("--s2_q_per_doc", type=int, default=0, help="lever subset: max stage-2 train items per doc (0 = all)")
    ap.add_argument("--s2_val", default="quality:10,musique:20,lastingbench:20,longhealth:0")
    ap.add_argument("--s2_val_q_per_doc", type=int, default=4)
    ap.add_argument("--val_seed", type=int, default=1)
    ap.add_argument("--val_fold", type=int, default=1)
    ap.add_argument("--test_fold", type=int, default=0)
    ap.add_argument("--teacher_cache", default="", help="train_encside_B cache: restrict stage-2 train items to the "
                    "keys whose 27B text-teacher top-k is cached AND whose rows match")
    ap.add_argument("--evals_d22", default="", help="comma list of D22 set files: cache their judged + wrong-map docs")
    ap.add_argument("--exclude_sets", default="", help="lever 1 (as register_d12p)")
    ap.add_argument("--exclude_docs", default="")
    ap.add_argument("--doc_len", default="")
    ap.add_argument("--evals", default="data/longhealth_split:test:0:0;data/quality_split:test::300;"
                    "data/quality_split:test::200")
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra_splits", default="", help="D12b: 'name=data/<name>_split,...' train-only splits (DS data)")
    ap.add_argument("--s2_ops_only", type=int, default=0, help="D12b: 1 = stage-2 TRAIN items get operators only (no "
                    "encoder records; register_d12p --cached_only 2 trains them on KL+CE)")
    ap.add_argument("--skip_have", default="", help="D12b: comma list of other cache dirs; operators already there "
                    "(and docs with no record to write) are not recomputed")
    ap.add_argument("--val_extra", default="", help="D12b: operators (only) for register_d12p --val_extra docs")
    ap.add_argument("--val_extra_q", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        for k in ("s1_train", "s2_train", "s1_val", "s2_val"):
            setattr(a, k, ",".join(f"{s.split(':')[0]}:{2 if 'train' in k else 1}" for s in getattr(a, k).split(",")))
        a.q_per_doc, a.val_q_per_doc, a.s2_val_q_per_doc = 2, 2, 1
        a.evals = "data/longhealth_split:test:0:4"
    from fit_d12p import D13Rt, add_extra_splits, op_path, rec_path, row_key, select_items  # noqa: F401
    add_extra_splits(a.extra_splits)
    from transformers import AutoTokenizer
    from xenc.backbone import input_device, load_encoder, text_ids
    from xenc.d12p_read import EncTeacher
    from xenc.doc_tasks import gold_text, load_docs
    os.makedirs(os.path.join(a.out, "op"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "rec"), exist_ok=True)
    t0 = time.time()
    enc = load_encoder(a.encoder, device_map=a.device)[0]
    tok = AutoTokenizer.from_pretrained(a.reader)
    lmap = layer_map_from_hubs(a.reader, enc, a.layers)
    log(f"D12p phase A: encoder {a.encoder} for reader {a.reader} | layer map {lmap}")
    rt = object.__new__(D13Rt)                                    # encoder-only runtime (no reader, no interface)
    rt.reader, rt.enc, rt.tok, rt.hub, rt.itf = None, enc, tok, a.reader, None
    rt.tail, rt.max_tokens = a.tail, a.max_tokens
    rt.Tc = EncTeacher(enc, sorted(set(lmap.values())))
    rt.docs, rt._pre, rt._ops, rt.lru = {}, {}, __import__("collections").OrderedDict(), 2
    rt.dev = rt.edev = input_device(enc)

    tr_folds = set(range(5)) - {a.test_fold, a.val_fold}
    s1t, d1 = select_items(a.s1_train, "train", tr_folds, a.q_per_doc, a.seed)
    s1v, d2 = select_items(a.s1_val, "val", {a.val_fold}, a.val_q_per_doc, a.seed + 1)
    s2t, d3 = select_items(a.s2_train, "train", tr_folds, a.s2_q_per_doc, a.seed)
    s2v, d4 = select_items(a.s2_val, "val", {a.val_fold}, a.s2_val_q_per_doc, a.val_seed)
    from fit_d12p import lever_filter
    (s1t, s1v, s2t, s2v), _ = lever_filter(a, s1t, s1v, s2t, s2v)   # lever 1 (no-op unless the flags are set)
    rt.docs = {**d1, **d2, **d3, **d4}
    if a.teacher_cache:
        import glob
        tc = {}
        for p in sorted(glob.glob(os.path.join(a.teacher_cache, "train_*.pt"))):
            for r in torch.load(p, weights_only=False):
                tc[r["key"]] = list(r["sids"])
        keep = []
        for it in s2t:
            k = _get(it, "key")
            if k in tc and tc[k] == rt.rows(_get(it, "doc_id"), it, text_ids(tok, gold_text(it))):
                keep.append(it)
        log(f"teacher cache {a.teacher_cache}: {len(tc)} keys; stage-2 train items {len(s2t)} -> {len(keep)} "
            f"with matching cached 27B teacher rows")
        s2t = keep
    # unique records: (did, base rows) -> (item, need_x)
    want: dict = {}
    for its, need_x in ((s1t, True), (s1v, True), (s2t, False), (s2v, False)):
        if a.s2_ops_only and its is s2t:                          # D12b: operators only for the stage-2 train docs
            continue
        for it in its:
            did = _get(it, "doc_id")
            k = row_key(did, rt.rows(did, it))
            prev = want.get(k)
            want[k] = (did, it, need_x or (prev is not None and prev[2]))
    ops_needed = {d for d, _, _ in want.values()}
    if a.s2_ops_only:
        ops_needed |= {_get(i, "doc_id") for i in s2t}
    if a.val_extra:                                               # D12b: extra val docs (operators only)
        from fit_d12p import extra_val_items
        vx, dx = extra_val_items(a.val_extra, a.val_extra_q, a.val_seed, a.doc_len)
        (vx,), _ = lever_filter(a, vx)
        rt.docs.update(dx)
        ops_needed |= {_get(i, "doc_id") for i in vx}
        log(f"val_extra {a.val_extra}: {len(vx)} items / {len({_get(i, 'doc_id') for i in vx})} docs (operators only)")
    ev = eval_docs(a.evals, a.seed)
    for f in [x for x in a.evals_d22.split(",") if x]:            # D12p: D22 judge sets (+ their wrong-map docs)
        S = json.load(open(f))
        sd = S["split_dir"]
        ev |= {(sd, d) for d in S["judged_docs"]}
        wm = S.get("wrong_map_file") or f.replace(".json", "_wrongmap.json")
        if os.path.exists(wm):
            W = json.load(open(wm))
            ev |= {(sd, d) for d in W.values() if isinstance(d, str)}
    ev_docs: dict = {}
    for sd, d in ev:
        ev_docs.setdefault(sd, set()).add(d)
    for sd, ds in ev_docs.items():
        dd = load_docs(sd)
        for d in ds:
            rt.docs[d] = dd[d]
            ops_needed.add(d)
    if a.skip_have:                                               # D12b: operators already cached elsewhere
        others = [d for d in a.skip_have.split(",") if d]
        n0 = len(ops_needed)
        ops_needed = {d for d in ops_needed if d in by_doc_keys(want) or
                      not any(os.path.exists(op_path(c, d)) for c in others)}
        log(f"skip_have {others}: operators to consider {n0} -> {len(ops_needed)}")
    log(f"items: s1 train {len(s1t)} / val {len(s1v)}, s2 train {len(s2t)} / val {len(s2v)} -> {len(want)} records; "
        f"operators {len(ops_needed)} docs ({len(ev)} judge docs)")
    by_doc: dict = {}
    for k, (did, it, nx) in want.items():
        by_doc.setdefault(did, []).append((k, it, nx))
    n_op = n_rec = 0
    for i, did in enumerate(sorted(ops_needed)):
        po = op_path(a.out, did)
        recs = by_doc.get(did, [])
        todo = [(k, it, nx) for k, it, nx in recs if _needs(os.path.join(a.out, "rec", k + ".pt"), nx)]
        if os.path.exists(po) and not todo:
            continue
        op, n, _ = rt.operator(did)
        if not os.path.exists(po):
            ids = rt.pre(did)[: -rt.tail] if rt.tail else rt.pre(did)        # P3: column token ids (ids[c] = column c)
            assert len(ids) == int(n)
            torch.save({"kv": {int(e): (K.cpu(), V.cpu()) for e, (K, V) in op.items() if isinstance(e, int)},
                        "n": int(n), "ids": [int(x) for x in ids]}, po + ".tmp")
            os.replace(po + ".tmp", po)
            n_op += 1
        for k, it, nx in todo:
            rows = rt.rows(did, it, text_ids(tok, gold_text(it)))
            rec = rt.teacher(did, rows)
            keep = ("q", "k", "out", "x") if nx else ("q", "k", "out")
            blob = {"rows": rows, "rec": {int(e): {f: r[f].cpu() for f in keep} for e, r in rec.items()}}
            pr = os.path.join(a.out, "rec", k + ".pt")
            torch.save(blob, pr + ".tmp")
            os.replace(pr + ".tmp", pr)
            n_rec += 1
        rt._ops.clear()
        if (i + 1) % 20 == 0:
            log(f"  {i + 1}/{len(ops_needed)} docs | new operators {n_op}, records {n_rec} ({time.time() - t0:.0f}s)")
    if torch.cuda.is_available():
        log(f"PEAK MEM phase A: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    log(f"phase A done: {n_op} operators, {n_rec} records -> {a.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
