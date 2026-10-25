"""D13 judge: the reader-query read (xenc/d13_read.py) of the encoder's operator, training-free at test.

Conditions (greedy, budget, SYSTEM_ANSWER; same items / seed as scripts/eval_encside.py):
  closed_book  chat(q_prompt)            closed_tail  doc tail (last --tail tokens) + question suffix
  text         chat(ctx_prompt)
  kv           reader prompt = own-doc tail + question; the D13 read of the own document's operator replaces the
               attention output at the paired layers (the encoder only prefills the document, never sees the question)
  wrong_doc    same with ANOTHER document (its tail in the prompt, its operator in the read)

    python scripts/eval_d13.py --interface outputs/D13_maps/d13_lin.pt --split_dir data/longhealth_split --fold 0 --n 0 \
        --out outputs/D13_eval_lin_lh
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

CONDS = ("closed_book", "closed_tail", "text", "kv", "wrong_doc")


def summarize(rows, conds) -> dict:
    m = lambda c: sum(r[c]["score"] for r in rows if c in r) / max(sum(1 for r in rows if c in r), 1)   # noqa: E731
    summ = {c: {"score": m(c), "n": sum(1 for r in rows if c in r),
                "mean_gen_tokens": sum(r[c]["gen_tokens"] for r in rows if c in r) / max(len(rows), 1)} for c in conds}
    for other in ("wrong_doc", "closed_tail", "text"):
        if "kv" in summ and other in summ:
            pair = [r["kv"]["score"] - r[other]["score"] for r in rows if "kv" in r and other in r]
            if not pair:
                continue
            mu = sum(pair) / len(pair)
            sd = (sum((p - mu) ** 2 for p in pair) / max(len(pair) - 1, 1)) ** 0.5
            summ[f"kv_minus_{other}"] = {"paired_gap": mu, "se": sd / len(pair) ** 0.5, "n_paired": len(pair),
                                         "wins": sum(p > 0 for p in pair), "losses": sum(p < 0 for p in pair)}
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--B", default="outputs/D10_maps/encB8_9b_2b_post.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--interface", default="", help="d13 file (fit_d13.py / register_d13.py); may be omitted only when "
                    "--conds is a subset of closed_book,text (C1: the reader alone, e.g. the 4B reading the document as text)")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--conds", default="closed_tail,kv,wrong_doc")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--share_encoder", type=int, default=0, help="1: C2, --encoder == --reader, one shared instance")
    ap.add_argument("--enc_device", default="", help="put the encoder (and its operator prefill) on this device, e.g. "
                    "cuda:0 with --device cuda:1 (2-GPU long-operator runs); default = --device")
    ap.add_argument("--out", required=True)
    ap.add_argument("--oom_skip", type=int, default=0, help="1: an item whose generation runs out of GPU memory is recorded in "
                    "<out>/oom_skipped.json and the process EXITS with code 75 (never continues in-process after a CUDA OOM); "
                    "a rerun with the same args resumes from eval_<split>.ckpt.jsonl, skipping finished and OOM items "
                    "(slurm/R2_seq.sbatch reruns exit-75 steps). Skipped items are absent from the eval json; default 0 = raise")
    ap.add_argument("--wrong_map", default=None,
                    help="[D19] JSON {doc_id: wrong_doc_id} overriding the default shift-7 wrong document "
                         "(e.g. scripts/d19_wrong_map.py, which never pairs near-duplicate documents)")
    ap.add_argument("--cmp", default="", help="CMP (xenc/cmp_op.py): compressor file; every operator the reader reads goes "
                    "through the per-token bottleneck (default '' = off: the plain operator)")
    ap.add_argument("--cmp_quant", default="", help="CMP: storage format of z (bf16|fp16|int8|int4g64|fp32); default = the "
                    "file's")
    ap.add_argument("--hyb", default="", help="CMP hybrid (xenc/cmp_hyb.py): deep-layer memory channel; a 6-layer compressor "
                    "file (closed form) or a saved hyb_*.pt state (default '' = off)")
    ap.add_argument("--hyb_oracle", type=int, default=0, help="CMP hybrid DIAGNOSTIC: memory = the reader's true K/V of a "
                    "reader pass over the document (not a method)")
    ap.add_argument("--hyb_g", type=float, default=None, help="CMP hybrid: override the deep gates (e.g. 1.0)")
    ap.add_argument("--d22_set", default="", help="restrict to the item_keys of a D22 set json (scripts/d22_make_sets.py; "
                    "leave --n 0); use with that set's --wrong_map")
    a = ap.parse_args(argv)
    from eval_doc import wrong_doc_id
    from fit_d13 import D13Rt, build
    from register_state import greedy
    from xenc.backbone import build_chat
    from xenc.doc_tasks import ctx_prompt, load_docs, q_prompt, score
    conds = [c for c in CONDS if c in a.conds.split(",")]
    os.makedirs(a.out, exist_ok=True)
    text_only = set(conds) <= {"closed_book", "text"}
    if text_only and not a.interface:
        from xenc.backbone import load_reader
        reader, tok = load_reader(a.reader, device_map=a.device)
        rt, itf, lmap = None, None, {}
    else:
        assert a.interface, "--interface is required unless --conds is a subset of closed_book,text"
        reader, enc, tok, itf, lmap = build(a, init=a.interface)
        rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens)
        if a.cmp:                                    # CMP operator bottleneck (default off)
            from xenc.cmp_op import load_cmp, wire
            _c = load_cmp(a.cmp).to(rt.dev)
            if a.cmp_quant:
                _c.quant = a.cmp_quant
            wire(rt, _c, log=log)
        if a.hyb or a.hyb_oracle:                    # CMP hybrid deep channel (default off)
            from xenc.cmp_hyb import build_hyb
            _b = torch.load(a.hyb, map_location="cpu", weights_only=False) if a.hyb else {}
            if isinstance(_b, dict) and _b.get("format") == "xenc.hyb_hm.v1":      # HYB-HM learned writer (xenc/hyb_hm.py)
                from xenc.hyb_hm import build_hm
                build_hm(rt, path=a.hyb, log=log)
            else:
                _isst = isinstance(_b, dict) and _b.get("format") == "xenc.cmp_hyb.v1"
                build_hyb(rt, "" if _isst else a.hyb, a.hyb if _isst else "", oracle=bool(a.hyb_oracle), g=a.hyb_g, log=log)
    docs = load_docs(a.split_dir)
    if rt is not None:
        rt.docs = docs
    items = load_split_items(a.split_dir, a.split)
    _sp = json.load(open(a.d22_set)).get("splits") if a.d22_set else None
    if isinstance(_sp, list) and _sp:                # LooGLE-long draws from every split (as register_d22); MCQ sets
        items = [i for s_ in _sp for i in load_split_items(a.split_dir, s_)]   # store an int count there (ignored)
    doc_order = sorted({_get(i, "doc_id") for i in items})
    wmap = json.load(open(a.wrong_map)) if a.wrong_map else {}
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
    if a.d22_set:
        assert not a.n, "--d22_set fixes the sample: pass --n 0"
        keys = set(json.load(open(a.d22_set))["item_keys"])
        items = [i for i in items if _get(i, "key") in keys]
        assert len(items) == len(keys), (len(items), len(keys))
    if a.n and len(items) > a.n:
        items = random.Random(a.seed).sample(items, a.n)
    log(f"D13 eval: {a.encoder} -> {a.reader} | layers {lmap} | interface {a.interface} | gates "
        f"{[round(float(x), 3) for x in itf.g] if itf is not None else None} | tail {a.tail} | {len(items)} items")
    rows, t0 = [], time.time()
    ck_path = os.path.join(a.out, f"eval_{a.split}.ckpt.jsonl")
    oom_path = os.path.join(a.out, "oom_skipped.json")
    done, oom_keys = {}, set()
    if a.oom_skip:                                   # resume after an exit-75 OOM restart (fresh process)
        if os.path.exists(ck_path):
            for line in open(ck_path):
                if line.strip():
                    r_ = json.loads(line)
                    done[r_["key"]] = r_
        if os.path.exists(oom_path):
            oom_keys = {x["key"] for x in json.load(open(oom_path))}
        if done or oom_keys:
            log(f"  --oom_skip resume: {len(done)} items done (from {ck_path}), {len(oom_keys)} OOM items skipped")
    ck = open(ck_path, "a" if a.oom_skip else "w")
    order = sorted(range(len(items)), key=lambda k: _get(items[k], "doc_id"))     # operator reuse; scores unaffected
    res = {}
    oom_skipped: list = json.load(open(oom_path)) if a.oom_skip and os.path.exists(oom_path) else []
    for k in order:
        it = items[k]
        did = _get(it, "doc_id")
        if _get(it, "key") in done:
            res[k] = done[_get(it, "key")]
            rows.append(res[k])
            continue
        if _get(it, "key") in oom_keys:
            continue
        w = wmap[did] if did in wmap else wrong_doc_id(did, doc_order)
        rec = {"key": _get(it, "key"), "doc_id": did, "wrong": w}
        try:
            for c in conds:
                if c == "closed_book":
                    g, n = greedy(reader, tok, a.reader, build_chat(tok, a.reader, SYSTEM_ANSWER, q_prompt(it)), a.budget)
                elif c == "text":
                    g, n = greedy(reader, tok, a.reader, build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, docs[did])), a.budget)
                elif c == "closed_tail":
                    g, n = rt.gen(did, it, a.budget, read=False)
                else:
                    g, n = rt.gen(w if c == "wrong_doc" else did, it, a.budget)
                rec[c] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
        except torch.OutOfMemoryError:
            if not a.oom_skip:
                raise
            oom_skipped.append({"key": rec["key"], "doc_id": did, "cond": c})
            json.dump(oom_skipped, open(oom_path, "w"), indent=1)
            ck.close()
            try:                                     # never continue in-process after a CUDA OOM (F1: later illegal
                torch.cuda.synchronize()             # memory access); exit 75, the job script reruns and resumes
                torch.cuda.empty_cache()
            except Exception as ex:                  # noqa: BLE001
                log(f"  (synchronize after OOM raised {type(ex).__name__})")
            log(f"  OOM on {rec['key']} ({did}, cond {c}); recorded in {oom_path} (total {len(oom_skipped)}); "
                f"EXIT 75 to resume in a fresh process (--oom_skip)")
            sys.exit(75)
        res[k] = rec
        rows.append(rec)
        ck.write(json.dumps(rec) + "\n")
        ck.flush()
        if len(rows) % 20 == 0:
            s = summarize(rows, conds)
            log(f"  [{len(rows)}/{len(items)}] " + " ".join(f"{c}={s[c]['score']:.3f}" for c in conds)
                + f" | {(time.time() - t0) / len(rows):.1f}s/item")
    rows = [res[k] for k in range(len(items)) if k in res]
    summ = summarize(rows, conds)
    json.dump({"summary": summ, "args": vars(a), "items": rows}, open(os.path.join(a.out, f"eval_{a.split}.json"), "w"), indent=1)
    log(f"==== eval_d13 [{a.split}] n={len(rows)} -> {a.out}")
    for c in conds:
        log(f"  {c:12s} {summ[c]['score']:.4f}  n={summ[c]['n']}  gen {summ[c]['mean_gen_tokens']:.1f}")
    for o in ("wrong_doc", "closed_tail", "text"):
        if f"kv_minus_{o}" in summ:
            v = summ[f"kv_minus_{o}"]
            log(f"  kv - {o}: {v['paired_gap']:+.4f} ± {v['se']:.4f} (n={v['n_paired']}, {v['wins']}/{v['losses']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
