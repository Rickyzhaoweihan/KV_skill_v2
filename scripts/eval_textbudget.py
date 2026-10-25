"""BL S2 + S4: budget-matched TEXT baselines for the same reader (agent BASE, 2026-10-23).

Same items (eval_d13 selection: --split_dir/--fold/--n/--seed/--d22_set), same prompt (SYSTEM_ANSWER + ctx_prompt),
same greedy decoding / budget / scorer as the `text` row; only the document text the reader sees changes.

Conditions (--conds, comma list; K = reader tokens of document content, or r<x> = 1/x of the document):
  trunc_head@K  trunc_ht@K           first K tokens / first K/2 + last K/2                     (xenc/textbudget.py)
  bm25@K  dense@K  hybrid@K          top K//512 chunks of 512 tokens by the question text, document order
  lingua2@K  lingua2@r4             LLMLingua-2 (question-agnostic, once per document)        (xenc/lingua.py)
  llingua@K                          LongLLMLingua, question-aware, --scorer (Qwen3.5-4B) as the scoring LM
  text                               the full document (pilot sanity only: must reproduce the existing text rows)
A document with N <= K tokens is passed through unchanged (record field full=1).

    python scripts/eval_textbudget.py --reader Qwen/Qwen3.5-9B --split_dir data/qasper_split \
        --d22_set outputs/D22_sets/qasper.json --n 0 --conds trunc_ht@2048,bm25@2048,hybrid@2048 --out outputs/BL_9b_qasper
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
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def select_items(a):
    """eval_d13's item selection, verbatim semantics."""
    from register_doc import _get, load_split_items
    items = load_split_items(a.split_dir, a.split)
    _sp = json.load(open(a.d22_set)).get("splits") if a.d22_set else None
    if isinstance(_sp, list) and _sp:
        items = [i for s_ in _sp for i in load_split_items(a.split_dir, s_)]
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
    if a.d22_set:
        assert not a.n, "--d22_set fixes the sample: pass --n 0"
        keys = set(json.load(open(a.d22_set))["item_keys"])
        items = [i for i in items if _get(i, "key") in keys]
        assert len(items) == len(keys), (len(items), len(keys))
    if a.n and len(items) > a.n:
        items = random.Random(a.seed).sample(items, a.n)
    if getattr(a, "keys_from", ""):            # exact item set of an existing eval json (e.g. a kv@0 or text row)
        E = json.load(open(a.keys_from))
        want = {r["key"] for r in E["items"]}
        items = [i for i in items if _get(i, "key") in want]
        assert len(items) == len(want), f"--keys_from {a.keys_from}: {len(want)} keys, {len(items)} selected"
    return items


def summarize(rows, conds):
    out = {}
    for c in conds:
        rs = [r[c] for r in rows if c in r]
        if not rs:
            continue
        out[c] = {"score": sum(x["score"] for x in rs) / len(rs), "n": len(rs),
                  "mean_ctx_tokens": sum(x["ctx_tokens"] for x in rs) / len(rs),
                  "mean_content_tokens": sum(x.get("content_tokens", 0) for x in rs) / len(rs),
                  "frac_full": sum(x.get("full", 0) for x in rs) / len(rs),
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
    ap.add_argument("--conds", required=True)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--scorer", default="Qwen/Qwen3.5-4B", help="LongLLMLingua scoring LM (llingua@K conds)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="pilot: only the first N items (document order)")
    ap.add_argument("--dump", type=int, default=3, help="save the contexts of the first N items per cond (sanity)")
    ap.add_argument("--text_ref_gpu", default="", help="GPU type of the reused text row (L40S|H100|?): when this job's "
                    "GPU is a different type (or ?), the full-text row is computed in-job so text vs baseline is same-GPU")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from register_doc import SYSTEM_ANSWER, _get
    from register_state import greedy
    from xenc.backbone import build_chat, load_reader
    from xenc.doc_tasks import ctx_prompt, load_docs, score
    from xenc.textbudget import DocIndex, build_context, parse_cond, query_text
    os.makedirs(a.out, exist_ok=True)
    conds = [c for c in a.conds.split(",") if c]
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    if a.text_ref_gpu and "text" not in conds and (a.text_ref_gpu == "?" or a.text_ref_gpu not in gpu):
        conds = ["text"] + conds
        log(f"text row reused from {a.text_ref_gpu} but this job runs on {gpu}: computing text in-job")
    parsed = {c: (("text", None, None) if c == "text" else parse_cond(c)) for c in conds}
    reader, tok = load_reader(a.reader, device_map=a.device)
    dense = l2 = ll = None
    if any(p[0] in ("dense", "hybrid") for p in parsed.values()):
        from xenc.textbudget import Dense
        dense = Dense(device=a.device)
    if any(p[0] == "lingua2" for p in parsed.values()):
        from xenc.lingua import Lingua2
        l2 = Lingua2(tok, device=a.device)
    if any(p[0] == "llingua" for p in parsed.values()):
        from xenc.backbone import load_reader as _lr
        from xenc.lingua import LongLingua
        sm, stok = _lr(a.scorer, device_map=a.device)
        ll = LongLingua(sm, stok, tok, chunk=a.chunk)
    docs = load_docs(a.split_dir)
    items = select_items(a)
    items = sorted(items, key=lambda i: (_get(i, "doc_id"), _get(i, "key")))
    if a.limit:
        items = items[:a.limit]
    log(f"eval_textbudget: reader {a.reader} on {gpu} | {len(items)} items, {len({_get(i, 'doc_id') for i in items})} "
        f"docs | conds {conds} | split_dir {a.split_dir} set {a.d22_set or '-'} fold {a.fold}")
    ck_path = os.path.join(a.out, f"eval_{a.split}.ckpt.jsonl")
    done = {}
    if os.path.exists(ck_path):
        for line in open(ck_path):
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r
        log(f"  resume: {len(done)} items done")
    ck = open(ck_path, "a")
    dumps = {c: [] for c in conds}
    rows, t0 = [], time.time()
    cur_doc, idx, l2_cache = None, None, {}
    timing = {c: 0.0 for c in conds}
    for it in items:
        key, did = _get(it, "key"), _get(it, "doc_id")
        if key in done and all(c in done[key] for c in conds):
            rows.append(done[key])
            continue
        rec = done.get(key, {"key": key, "doc_id": did})
        if did != cur_doc:
            cur_doc, idx, l2_cache = did, DocIndex(tok, docs[did], chunk=a.chunk), {}
        q = query_text(it)
        for c in conds:
            if c in rec:
                continue
            meth, K, ratio = parsed[c]
            if ratio is not None:
                K = max(1, int(round(idx.N * ratio)))
            tc = time.time()
            if meth == "text":
                ctx, info = docs[did], {"full": 1, "doc_tokens": idx.N, "content_tokens": idx.N}
            elif meth == "lingua2":
                if K not in l2_cache:
                    l2_cache[K] = l2.compress(docs[did], target_token=K)
                ctx, info = l2_cache[K]
            elif meth == "llingua":
                ctx, info = ll.compress(docs[did], q, target_token=K)
            else:
                ctx, info = build_context(meth, K, idx, q, dense)
            timing[c] += time.time() - tc
            ids = build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, ctx))
            g, n = greedy(reader, tok, a.reader, ids, a.budget)
            rec[c] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n, "ctx_tokens": len(ids),
                      **{k: v for k, v in info.items() if k != "sel"}, "sel": info.get("sel")}
            if len(dumps[c]) < a.dump:
                dumps[c].append({"key": key, "question": str(_get(it, "question"))[:300], "query": q[:300],
                                 "context_head": ctx[:1500], "info": {k: v for k, v in info.items()}})
        rows.append(rec)
        ck.write(json.dumps(rec) + "\n")
        ck.flush()
        if len(rows) % 20 == 0:
            s = summarize(rows, conds)
            log(f"  [{len(rows)}/{len(items)}] " + " ".join(f"{c}={s[c]['score']:.3f}" for c in conds if c in s)
                + f" | {(time.time() - t0) / len(rows):.1f}s/item")
    ck.close()
    summ = summarize(rows, conds)
    json.dump({"summary": summ, "args": vars(a), "gpu": gpu, "compress_time_s": timing,
               "llingua_ppl_calls": (ll.pc.stats if ll else None), "items": rows},
              open(os.path.join(a.out, f"eval_{a.split}.json"), "w"), indent=1)
    json.dump(dumps, open(os.path.join(a.out, "ctx_dump.json"), "w"), indent=1)
    log(f"==== eval_textbudget [{a.split}] n={len(rows)} on {gpu} -> {a.out}")
    for c in conds:
        if c in summ:
            s = summ[c]
            log(f"  {c:16s} {s['score']:.4f}  n={s['n']}  ctx {s['mean_ctx_tokens']:.0f} tok  content "
                f"{s['mean_content_tokens']:.0f}  full {s['frac_full']:.2f}  compress {timing[c]:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
