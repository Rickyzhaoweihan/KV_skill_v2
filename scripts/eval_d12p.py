"""D12p (copy of D13 for the 27B reader). D13 judge: the reader-query read (xenc/d13_read.py) of the encoder's operator, training-free at test.

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

CONDS = ("closed_book", "closed_tail", "text", "kv", "wrong_doc", "kv_base", "kv_Hswap")   # last two: D12b --hyper_ckpt


def summarize(rows, conds) -> dict:
    m = lambda c: sum(r[c]["score"] for r in rows if c in r) / max(sum(1 for r in rows if c in r), 1)   # noqa: E731
    summ = {c: {"score": m(c), "n": sum(1 for r in rows if c in r),
                "mean_gen_tokens": sum(r[c]["gen_tokens"] for r in rows if c in r) / max(len(rows), 1)} for c in conds}
    for other in ("wrong_doc", "closed_tail", "kv_base", "kv_Hswap"):
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
    ap.add_argument("--interface", required=True, help="d13 file (fit_d13.py / register_d13.py)")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--conds", default="closed_tail,kv,wrong_doc")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enc_device", default=None, help="D12p: encoder device (e.g. cuda:1)")
    ap.add_argument("--reader_mem", default=None, help="D12p: shard the reader, per-GPU budgets '27GiB,27GiB'")
    ap.add_argument("--enc_cache", default=None, help="D12p phase B: cache_d12p.py dir")
    ap.add_argument("--d22_set", default="", help="judge exactly this D22 set's item keys (leakage-filtered test items)")
    ap.add_argument("--wrong_map", default="", help="D22 wrong-doc map json {doc_id: wrong_doc_id}")
    ap.add_argument("--itf_device", default=None, help="D12p: device of the interface + cached tensors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hyper_ckpt", default="", help="D12b: read through base + H(operator) (h_best.pt); conds kv_base "
                    "(no adapter) and kv_Hswap (own operator, wrong doc's adapter) become available")
    ap.add_argument("--oom_skip", type=int, default=0, help="DS: 1 = an item whose generation hits CUDA OOM (any cond) is "
                    "recorded in <out>/oom_skipped.json and the process EXITS 75; a rerun with the same args resumes from "
                    "eval_<split>.ckpt.jsonl and drops the recorded items from EVERY cond (same item set for text and kv). "
                    "Default 0 = raise (old behaviour)")
    ap.add_argument("--text_max_tokens", type=int, default=0, help="DS: >0 = the text cond reads only the LAST "
                    "(N - 1024) tokens of the document, the same window the operator gets with --max_tokens N (27B regime); "
                    "0 = full document (old behaviour)")
    ap.add_argument("--hyb", default="", help="CMP hybrid (xenc/cmp_hyb.py): deep-layer memory channel read natively by the "
                    "reader layers --hyb_layers; a cross-model compressor file (scripts/cmp_fit_xm.py) or a saved hyb state "
                    "(default '' = off)")
    ap.add_argument("--hyb_zcache", default="", help="CMP hybrid: dir of precomputed codes (scripts/cmp_zcache.py)")
    ap.add_argument("--hyb_zdim", type=int, default=0, help="CMP hybrid: keep the first d components of the cached codes")
    ap.add_argument("--hyb_oracle", type=int, default=0, help="CMP hybrid DIAGNOSTIC: memory = the reader's true K/V of a "
                    "reader pass over the document prefix (not a method)")
    ap.add_argument("--hyb_g", type=float, default=None, help="CMP hybrid: override the deep gates (e.g. 1.0)")
    ap.add_argument("--hyb_layers", default="51,55,59,63", help="CMP hybrid: reader layers that read the memory")
    ap.add_argument("--hyb_enc_layers", default="23,27,31", help="CMP hybrid: encoder source layers of the code")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from eval_doc import wrong_doc_id
    from fit_d12p import D13Rt, build, make_rt
    from register_state import greedy
    from xenc.backbone import build_chat
    from xenc.doc_tasks import ctx_prompt, load_docs, q_prompt, score
    conds = [c for c in CONDS if c in a.conds.split(",")]
    os.makedirs(a.out, exist_ok=True)
    reader, enc, tok, itf, lmap = build(a, init=a.interface)
    rt = make_rt(a, reader, enc, tok, itf)
    if a.hyper_ckpt:                                                # D12b hypernetwork on the frozen base
        from xenc.d12b_hyper import wire
        from xenc.d24_hyper import load_h
        if itf.delta_rank:
            itf.merge_delta()
        if torch.load(a.hyper_ckpt, map_location="cpu", weights_only=False, mmap=True).get("format") == "xenc.d12b_slots.v1":
            from xenc.d12b_slots import load_slots
            H, hx = load_slots(a.hyper_ckpt, itf)                   # D12b H-slots
        else:
            H, hx = load_h(a.hyper_ckpt, itf)
        H = H.to(itf.g.device).eval()
        assert hx.get("base") in (None, a.interface), f"H was trained on base {hx.get('base')}, not {a.interface}"
        wire(rt, H)
        if getattr(H, "kind", "") == "slots":
            itf._slot_stats = []                                     # slot share over every slot read of this judge
        log(f"D12b H: {a.hyper_ckpt} (step {hx.get('step')}, base {hx.get('base')})")
    else:
        assert not ({"kv_base", "kv_Hswap"} & set(conds)), "kv_base / kv_Hswap need --hyper_ckpt"
    if a.hyb or a.hyb_oracle:                                       # CMP hybrid deep channel (default off)
        from xenc.cmp_hyb import build_hyb
        _b = torch.load(a.hyb, map_location="cpu", weights_only=False) if a.hyb else {}
        _isst = isinstance(_b, dict) and _b.get("format") == "xenc.cmp_hyb.v1"
        _H = build_hyb(rt, "" if _isst else a.hyb, a.hyb if _isst else "", oracle=bool(a.hyb_oracle), g=a.hyb_g, log=log,
                       layers=[int(x) for x in a.hyb_layers.split(",")],
                       enc_layers=[int(x) for x in a.hyb_enc_layers.split(",")], zcache=a.hyb_zcache, zdim=a.hyb_zdim)
    docs = load_docs(a.split_dir)
    rt.docs = docs
    items = load_split_items(a.split_dir, a.split)
    _sp = json.load(open(a.d22_set)).get("splits") if a.d22_set else None
    if isinstance(_sp, list) and _sp:                # LooGLE-long draws from every split (as register_d22 / eval_d13);
        items = [i for s_ in _sp for i in load_split_items(a.split_dir, s_)]   # MCQ sets' int "splits" is ignored
    doc_order = sorted({_get(i, "doc_id") for i in items})
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
    if a.d22_set:                                                  # D12p: exactly a D22 judge set's item keys
        keys = set(json.load(open(a.d22_set))["item_keys"])
        items = [i for i in items if _get(i, "key") in keys]
        assert len(items) == len(keys), f"d22_set: {len(keys)} keys, {len(items)} found"
    elif a.n and len(items) > a.n:
        items = random.Random(a.seed).sample(items, a.n)
    wmap = json.load(open(a.wrong_map)) if a.wrong_map else {}
    log(f"D13 eval: {a.encoder} -> {a.reader} | layers {lmap} | interface {a.interface} | gates "
        f"{[round(float(x), 3) for x in itf.g]} | tail {a.tail} | {len(items)} items")
    rows, t0 = [], time.time()
    ck_path = os.path.join(a.out, f"eval_{a.split}.ckpt.jsonl")
    oom_path = os.path.join(a.out, "oom_skipped.json")
    done, oom_skipped = {}, []
    if a.oom_skip:                                   # DS: resume after an exit-75 OOM restart (fresh process)
        if os.path.exists(ck_path):
            for line in open(ck_path):
                if line.strip():
                    r_ = json.loads(line)
                    done[r_["key"]] = r_
        if os.path.exists(oom_path):
            oom_skipped = json.load(open(oom_path))
        if done or oom_skipped:
            log(f"  --oom_skip resume: {len(done)} items done, {len(oom_skipped)} OOM items skipped ({oom_path})")
    oom_keys = {x["key"] for x in oom_skipped}
    ck = open(ck_path, "a" if a.oom_skip else "w")
    order = sorted(range(len(items)), key=lambda k: _get(items[k], "doc_id"))     # operator reuse; scores unaffected
    res = {}
    _tdoc: dict = {}

    def text_doc(did):                               # DS --text_max_tokens: the operator's window of the document
        if not a.text_max_tokens:
            return docs[did]
        if did not in _tdoc:
            ids = tok(docs[did], add_special_tokens=False)["input_ids"]
            keep = max(a.text_max_tokens - 1024, 1)
            _tdoc[did] = docs[did] if len(ids) <= keep else tok.decode(ids[-keep:])
        return _tdoc[did]

    for k in order:
        it = items[k]
        did = _get(it, "doc_id")
        if _get(it, "key") in done:
            res[k] = done[_get(it, "key")]
            rows.append(res[k])
            continue
        if _get(it, "key") in oom_keys:
            continue
        w = wmap.get(did) or wrong_doc_id(did, doc_order)
        rec = {"key": _get(it, "key"), "doc_id": did, "wrong": w}
        try:
          for c in conds:
              if c == "closed_book":
                  g, n = greedy(reader, tok, a.reader, build_chat(tok, a.reader, SYSTEM_ANSWER, q_prompt(it)), a.budget)
              elif c == "text":
                  g, n = greedy(reader, tok, a.reader, build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, text_doc(did))), a.budget)
              elif c == "closed_tail":
                  g, n = rt.gen(did, it, a.budget, read=False)
              elif c in ("kv_base", "kv_Hswap"):
                  from xenc.d12b_hyper import adapter_of, override
                  with override(rt, False if c == "kv_base" else adapter_of(rt, w)):
                      g, n = rt.gen(did, it, a.budget)
              else:
                  g, n = rt.gen(w if c == "wrong_doc" else did, it, a.budget)
              rec[c] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
        except torch.OutOfMemoryError:
            if not a.oom_skip:
                raise
            oom_skipped.append({"key": rec["key"], "doc_id": did, "cond": c})
            json.dump(oom_skipped, open(oom_path, "w"), indent=1)
            ck.close()
            try:                                     # never continue in-process after a CUDA OOM; exit 75 -> rerun
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception as ex:                  # noqa: BLE001
                log(f"  (synchronize after OOM raised {type(ex).__name__})")
            log(f"  OOM on {rec['key']} ({did}, cond {c}); recorded in {oom_path} (total {len(oom_skipped)}); EXIT 75 (--oom_skip)")
            sys.exit(75)
        res[k] = rec
        rows.append(rec)
        ck.write(json.dumps(rec) + "\n")
        ck.flush()
        if len(rows) % 20 == 0:
            s = summarize(rows, conds)
            log(f"  [{len(rows)}/{len(items)}] " + " ".join(f"{c}={s[c]['score']:.3f}" for c in conds)
                + f" | {(time.time() - t0) / len(rows):.1f}s/item")
    rows = [res[k] for k in range(len(items)) if k in res]          # --oom_skip: OOM items absent from every cond
    if oom_skipped:
        log(f"  --oom_skip: {len(oom_skipped)} items dropped from every cond: {sorted(x['key'] for x in oom_skipped)}")
    summ = summarize(rows, conds)
    json.dump({"summary": summ, "args": vars(a), "items": rows}, open(os.path.join(a.out, f"eval_{a.split}.json"), "w"), indent=1)
    log(f"==== eval_d13 [{a.split}] n={len(rows)} -> {a.out}")
    for c in conds:
        log(f"  {c:12s} {summ[c]['score']:.4f}  n={summ[c]['n']}  gen {summ[c]['mean_gen_tokens']:.1f}")
    if getattr(itf, "_slot_stats", None):
        from xenc.d12b_slots import slot_stats_summary
        ss = slot_stats_summary(itf)
        summ["slots"] = ss
        log(f"  slots (kv / wrong_doc / kv_Hswap reads): share {ss['share']:.4f} entropy {ss['entropy']:.3f} "
            f"by layer {ss['share_by_layer']}")
        json.dump({"summary": summ, "args": vars(a), "items": rows}, open(os.path.join(a.out, f"eval_{a.split}.json"), "w"), indent=1)
    for o in ("wrong_doc", "closed_tail", "kv_base", "kv_Hswap"):
        if f"kv_minus_{o}" in summ:
            v = summ[f"kv_minus_{o}"]
            log(f"  kv - {o}: {v['paired_gap']:+.4f} ± {v['se']:.4f} (n={v['n_paired']}, {v['wins']}/{v['losses']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
