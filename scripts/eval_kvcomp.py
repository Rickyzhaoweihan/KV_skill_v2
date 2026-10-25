"""BL S5: question-agnostic compression of the READER's own document KV cache (agent BASE, 2026-10-23).

LABEL: "reader prefills the doc (ingest-cost-unmatched upper reference)". See xenc/kvcomp.py for the methods.

Per document: one reader prefill of [chat template + document] (no question) -> scores (SnapKV window = last 64
document tokens; KVzip reconstruction) -> keep masks per (method, ratio). Per question: greedy continuation of
[question suffix] from a copy of the cache, with the mask hook for compressed conditions.
Conditions: kvc_full (prefix cache, no compression: must reproduce the text row), and <method>@r<x> = keep 1/x of
the document positions per (full-attention layer, KV head) on average, for methods streaming | snapkv | kvzip.
Same items / prompt / greedy budget / scorer as the text row (eval_d13).

    python scripts/eval_kvcomp.py --reader Qwen/Qwen3.5-9B --split_dir data/longhealth_split --fold 0 --n 0 \
        --methods streaming,snapkv,kvzip --ratios 4,8,16 --out outputs/BL_kvc_9b_longhealth
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from eval_textbudget import log, select_items  # noqa: E402

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


LABEL = "reader prefills the doc (ingest-cost-unmatched upper reference)"


def summarize(rows, conds):
    out = {}
    for c in conds:
        rs = [r[c] for r in rows if c in r]
        if rs:
            out[c] = {"score": sum(x["score"] for x in rs) / len(rs), "n": len(rs),
                      "mean_gen_tokens": sum(x["gen_tokens"] for x in rs) / len(rs)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d22_set", default="")
    ap.add_argument("--keys_from", default="", help="restrict to exactly the items of this eval json (asserted)")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--methods", default="streaming,snapkv,kvzip")
    ap.add_argument("--ratios", default="4,8,16", help="compression factors x: keep 1/x of the document positions")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--kvzip_chunk", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="pilot: only the first N items (document order)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from register_doc import SYSTEM_ANSWER, _get
    from register_state import greedy
    from xenc.backbone import load_reader
    from xenc.doc_tasks import ctx_prompt, load_docs, score
    from xenc.kvcomp import (MaskHook, full_layers, keep_masks, kvzip_scores, prefill, snapkv_scores, split_prompt,
                             storage_bytes)
    os.makedirs(a.out, exist_ok=True)
    methods = [m for m in a.methods.split(",") if m]
    ratios = [float(x) for x in a.ratios.split(",") if x]
    conds = ["kvc_full"] + [f"{m}@r{int(x) if x == int(x) else x}" for m in methods for x in ratios]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    reader, tok = load_reader(a.reader, device_map=a.device)
    L = full_layers(reader)
    docs = load_docs(a.split_dir)
    items = sorted(select_items(a), key=lambda i: (_get(i, "doc_id"), _get(i, "key")))
    if a.limit:
        items = items[:a.limit]
    by = {}
    for it in items:
        by.setdefault(_get(it, "doc_id"), []).append(it)
    log(f"eval_kvcomp [{LABEL}]: reader {a.reader} on {gpu} | full-attn layers {L} | {len(items)} items, {len(by)} docs"
        f" | conds {conds}")
    ck_path = os.path.join(a.out, f"eval_{a.split}.ckpt.jsonl")
    done = {}
    if os.path.exists(ck_path):
        for line in open(ck_path):
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r
        log(f"  resume: {len(done)} items done")
    ck = open(ck_path, "a")
    rows, docinfo, t0 = [], {}, time.time()
    hdr = len("## Document\n")
    for did in sorted(by):
        its = by[did]
        todo = [it for it in its if not (_get(it, "key") in done and all(c in done[_get(it, "key")] for c in conds))]
        rows += [done[_get(it, "key")] for it in its if it not in todo]
        if not todo:
            continue
        doc = docs[did]
        sp = [split_prompt(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, doc), doc, hdr) for it in todo]
        ids0, ds, P = sp[0]
        for ids, ds_, P_ in sp:
            assert (ds_, P_) == (ds, P) and ids[:P] == ids0[:P], f"{did}: prefix is not question-agnostic"
        tdoc = time.time()
        _sync()
        tp = time.time()
        cache0, win = prefill(reader, ids0[:P], window=a.window)
        _sync()
        t_prefill = time.time() - tp
        n_kv = cache0.layers[L[0]].keys.shape[1]
        sc, t_score = {}, {}
        if "snapkv" in methods:
            ts = time.time()
            sc["snapkv"] = snapkv_scores(reader, cache0, win, P, window=a.window)
            t_score["snapkv"] = time.time() - ts
        if "kvzip" in methods:
            ts = time.time()
            sc["kvzip"] = kvzip_scores(reader, tok, a.reader, cache0, ids0, ds, P, chunk=a.kvzip_chunk)
            _sync()
            t_score["kvzip"] = time.time() - ts
        del win
        masks, minfo = {}, {}
        for m in methods:
            for x in ratios:
                c = f"{m}@r{int(x) if x == int(x) else x}"
                masks[c], minfo[c] = keep_masks(m, 1.0 / x, P, ds, n_kv, L, sc.get(m), window=a.window)
        kept_full = P * n_kv * len(L)
        stor = {"kvc_full": storage_bytes(reader, cache0, kept_full)}
        for c, mk in masks.items():
            stor[c] = storage_bytes(reader, cache0, sum(int(v.sum()) for v in mk.values()))
        docinfo[did] = {"P": P, "doc_start": ds, "N_doc": P - ds, "t_prefill_s": t_prefill, "t_score_s": t_score,
                        "storage": stor, "masks": minfo}
        for it, (ids, _, _) in zip(todo, sp):
            key = _get(it, "key")
            rec = done.get(key, {"key": key, "doc_id": did})
            for c in conds:
                if c in rec:
                    continue
                cache = copy.deepcopy(cache0)
                if c == "kvc_full":
                    g, n = greedy(reader, tok, a.reader, ids[P:], a.budget, cache=cache)
                else:
                    with MaskHook(reader, masks[c], P):
                        g, n = greedy(reader, tok, a.reader, ids[P:], a.budget, cache=cache)
                del cache
                rec[c] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n,
                          "kv_bytes": stor[c]["kv_bytes"], "linear_state_bytes": stor[c]["linear_state_bytes"]}
            rows.append(rec)
            ck.write(json.dumps(rec) + "\n")
            ck.flush()
        del cache0, sc, masks
        torch.cuda.empty_cache()
        s = summarize(rows, conds)
        log(f"  doc {did} N={P - ds} prefill {t_prefill:.1f}s score {({k: round(v, 1) for k, v in t_score.items()})} "
            f"doc total {time.time() - tdoc:.0f}s | [{len(rows)}/{len(items)}] "
            + " ".join(f"{c}={s[c]['score']:.3f}" for c in conds if c in s))
    ck.close()
    summ = summarize(rows, conds)
    json.dump({"summary": summ, "label": LABEL, "args": vars(a), "gpu": gpu, "docs": docinfo, "items": rows},
              open(os.path.join(a.out, f"eval_{a.split}.json"), "w"), indent=1)
    log(f"==== eval_kvcomp [{a.split}] n={len(rows)} on {gpu} -> {a.out}  ({time.time() - t0:.0f}s)")
    for c in conds:
        if c in summ:
            log(f"  {c:14s} {summ[c]['score']:.4f}  n={summ[c]['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
