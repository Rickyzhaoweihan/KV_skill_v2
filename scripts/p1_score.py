"""P1-hard: score the corpus subset with the BASE interface -> {key: KL(text teacher || kv read)} on the answer rows.

The same student pass register_d13.py trains (reader on doc tail + question + target with the D13 read of the 4B
operator), forward only, against the stored top-k teacher rows (p1_data.topk_kl). Items are grouped by document so
each operator is prefilled once. Output: --out json {key: kl}; register_d13.py --p1_weights then keeps the top
--p1_top_frac and samples proportional to the KL (hard-item weighting).

    python scripts/p1_score.py --init outputs/D13_lever1_full/d13_best.pt --corpus /scratch/.../P1_corpus \
        --span_only 1 --min_tokens 8000 --out outputs/P1_hard/base_kl.json
"""
from __future__ import annotations

import argparse
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--init", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--span_only", type=int, default=1)
    ap.add_argument("--min_tokens", type=int, default=8000)
    ap.add_argument("--groups", default="")
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=100000)
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--max_hours", type=float, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enc_device", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from fit_d13 import D13Rt, build
    from p1_data import load_p1, topk_kl
    from register_doc import log
    from register_state import kept_logits
    from xenc.backbone import text_ids
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    items, T, docs = load_p1(a.corpus, a.groups, 0.0, 0, 0, log, span_only=bool(a.span_only), min_tokens=a.min_tokens)
    items.sort(key=lambda i: (i["doc_id"], i["key"]))
    if a.max_items:
        items = items[: a.max_items]
    reader, enc, tok, itf, lmap = build(a, init=a.init)
    rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens, lru=1)
    R = rt.R
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    log(f"P1 score on {gpu}: {len(items)} items / {len({i['doc_id'] for i in items})} docs | init {a.init} | layers {lmap}")
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    t0, n, skip = time.time(), 0, 0
    for it in items:
        if it["key"] in out:
            continue
        if a.max_hours and time.time() - t0 > a.max_hours * 3600:
            log("time budget reached")
            break
        did = it["doc_id"]
        if did not in rt.docs:
            rt.docs = {did: open(docs[did]["path"]).read()}        # one doc at a time (items are sorted by doc)
            rt._pre = {}
        tt = text_ids(tok, it["target"])
        S = T[it["key"]]
        if tt != S["tgt"].tolist() or it["n_tids"] > a.max_tokens:
            skip += 1
            continue
        m = len(tt)
        try:
            with torch.no_grad():
                op, n_op, _ = rt.operator(did)
                R.load(op, n_op)
                R.mode = "student"
                s_log = kept_logits(reader, torch.tensor([rt.rows(did, it, tt)]), m + 1)
                out[it["key"]] = round(float(topk_kl(S["ids"], S["lp"], S["rest"], s_log[:-1])), 6)
                if n < 3:                       # correctness: the inference read == the training-mode (e2e) read
                    R.mode, R.past, R.need_logits, R.rec = "e2e", {}, False, {}
                    try:
                        e_log = kept_logits(reader, torch.tensor([rt.rows(did, it, tt)]), m + 1)
                        log(f"  [check {n + 1}] {it['key']}: KL student-mode {out[it['key']]:.5f} vs e2e-mode "
                            f"{float(topk_kl(S['ids'], S['lp'], S['rest'], e_log[:-1])):.5f} | max |d logit| "
                            f"{float((e_log.float() - s_log.float()).abs().max()):.4f}")
                    except Exception as ex:      # noqa: BLE001
                        log(f"  [check {n + 1}] e2e-mode comparison unavailable: {ex!r}")
        except torch.OutOfMemoryError:
            rt._ops.clear()
            torch.cuda.empty_cache()
            skip += 1
            continue
        finally:
            R.mode, R.past, R.rec = "off", {}, {}
        n += 1
        if n % 500 == 0:
            json.dump(out, open(a.out, "w"))
            v = sorted(out.values())
            log(f"  {n} scored ({skip} skipped), {(time.time() - t0) / n:.2f}s/item | KL median {v[len(v) // 2]:.4f} "
                f"p90 {v[int(.9 * len(v))]:.4f} max {v[-1]:.4f}")
    json.dump(out, open(a.out, "w"))
    v = sorted(out.values())
    if v:
        log(f"done: {len(out)} items scored ({skip} skipped) in {time.time() - t0:.0f}s | KL mean {sum(v) / len(v):.4f} "
            f"median {v[len(v) // 2]:.4f} p70 {v[int(.7 * len(v))]:.4f} p90 {v[int(.9 * len(v))]:.4f} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
