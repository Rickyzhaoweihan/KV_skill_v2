"""D10: judge a DeltaNet-state read interface (and the L/L prefix-cache anchors) on held-out documents.

Conditions (greedy, same --budget, stop on </answer>; system prompt = SYSTEM_ANSWER):
  closed_book    chat(q_prompt), no states
  text           chat(ctx_prompt): the document in context (the teacher / ceiling)
  kv             chat(q_prompt) + read branch over the item's own encoder states
  wrong_doc      chat(q_prompt) + branch over the states of another doc (eval_doc.wrong_doc_id, +7 shift)
  null           chat(q_prompt) + state branch over the MEAN DeltaNet state of the evaluated docs (content-free);
                 the attention branch is off in null (no mean K/V is defined)
L/L anchors (encoder == reader; need the full-attention K/V harvest, i.e. <doc>_kv.pt):
  prefix_full    DynamicCache(doc prefix: K/V + DeltaNet states + conv) ⊕ question suffix  (≡ text up to tokenization)
  prefix_state   DeltaNet states + conv only, no K/V                                        ⊕ question suffix
  prefix_kv      full-attention K/V only, DeltaNet layers start fresh                       ⊕ question suffix
  kv_kvprefix    prefix_kv + the read branch over the own states  (branch replaces the state carry-over)
  wrong_kvprefix prefix_kv (own K/V) + branch over the WRONG doc's states
--kl_text adds, per condition, KL(text ‖ cond) and the gold log-prob on the teacher-forced gold answer.

    python scripts/eval_state_xm.py --reader Qwen/Qwen3.5-9B --encoder same --identity --split_dir data/longhealth_split \
        --split test --fold 0 --n 0 --conds closed_book,text,prefix_full,prefix_state,prefix_kv,kv,wrong_doc,kv_kvprefix \
        --kl_text --out outputs/D10_anchor_lh
    python scripts/eval_state_xm.py --interface outputs/D10_q_SL/interface_best.pt --split_dir data/quality_split --n 300
    python scripts/eval_state_xm.py --tiny --n 4
"""

from __future__ import annotations

import argparse
import copy
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
from register_doc import _get, load_split_items, log  # noqa: E402
from register_state import js_item, student_prompt, set_tail, add_branch_args, build_reader, enc_dims, greedy, kept_logits, setup  # noqa: E402

BASE = ("closed_book", "closed_tail", "text", "kv", "wrong_doc", "null")
ANCHOR = ("prefix_full", "prefix_state", "prefix_kv", "kv_kvprefix", "wrong_kvprefix")
ALL = BASE + ANCHOR
BRANCH = {"kv": "own", "wrong_doc": "wrong", "null": "null", "kv_kvprefix": "own", "wrong_kvprefix": "wrong"}
CACHE = {"prefix_full": (True, True), "prefix_state": (False, True), "prefix_kv": (True, False),
         "kv_kvprefix": (True, False), "wrong_kvprefix": (True, False)}       # cond -> (keep_kv, keep_state)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default=None, help="default: from the interface args, else 'same'")
    ap.add_argument("--split_dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--state_root", default=None)
    ap.add_argument("--interface", default=None)
    ap.add_argument("--identity", action="store_true", help="L/L: A = B = I (structural), g = --gate_init (1.0)")
    ap.add_argument("--init", default=None, help="ridge maps: evaluate the un-registered ridge interface")
    add_branch_args(ap)
    ap.set_defaults(gate_init=1.0, tail=None)   # tail None = the interface's training --tail (else 0)
    ap.add_argument("--gain", type=float, default=1.0, help="multiplies every gate at eval time")
    ap.add_argument("--conds", default=",".join(BASE))
    ap.add_argument("--n", type=int, default=300, help="0 = all")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--kl_text", action="store_true")
    ap.add_argument("--kv_maps", default=None,
                    help="K/V-side maps (scripts/fit_kv_side.py): transform the encoder operator into the reader's K/V "
                         "space once and read it with the reader's own queries (attn branch, A = B = I)")
    ap.add_argument("--js", action="store_true",
                    help="addressing metric for kv (attn branch): JS between the reader's in-context attention over "
                         "document positions (text prompt) and the branch's read, over the shared prompt suffix rows")
    ap.add_argument("--harvest", type=int, default=1)
    ap.add_argument("--max_doc_tokens", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_mem", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="", help="suffix of eval_<split><tag>.json")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    if not a.tiny and not a.split_dir:
        ap.error("--split_dir is required unless --tiny")
    return a


def mean_states(rt, doc_ids, k: int = 16) -> dict | None:
    acc, n = None, 0
    for d in doc_ids[:k]:
        st = rt.states(d)
        if st is None:
            continue
        if acc is None:
            acc = {li: S.float().clone() for li, S in st["recurrent"].items()}
        else:
            for li, S in st["recurrent"].items():
                acc[li] += S.float()
        n += 1
    return None if acc is None else {"recurrent": {li: S / n for li, S in acc.items()}}


def st_own_for_js(rt, did):
    return rt.states(did)


def main(argv=None) -> int:
    a = parse_args(argv)
    from eval_doc import wrong_doc_id
    from xenc.backbone import build_chat, input_device, text_ids
    from xenc.doc_tasks import ctx_prompt, gold_text, q_prompt, score
    from xenc.losses import kl_teacher_student
    from xenc.state_interface import StateReader
    from xenc.state_skill import build_prefix_cache, question_suffix_ids

    conds = [c for c in ALL if c in {x.strip() for x in a.conds.split(",")}]
    extra, targs = {}, {}
    if a.interface:
        extra = torch.load(a.interface, map_location="cpu", weights_only=False).get("extra") or {}
        targs = extra.get("args") or {}
    a.encoder = a.encoder or targs.get("encoder") or "same"
    a.state_root = a.state_root or targs.get("state_root")
    if a.tail is None:
        a.tail = int(targs.get("tail", 0) or 0)
    a.grad_ckpt = 0
    out = a.out or (os.path.dirname(os.path.abspath(a.interface)) if a.interface else "outputs/D10_eval")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed)
    S = setup(a, need_train=False)
    model, tok, hub, docs, rt = S["model"], S["tok"], S["hub"], S["docs"], S["rt"]
    items = S["test"] if a.tiny else load_split_items(a.split_dir, a.split)
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
    if a.max_docs:
        keep = set(sorted({_get(i, "doc_id") for i in items})[: a.max_docs])
        items = [i for i in items if _get(i, "doc_id") in keep]
    items = [i for i in items if rt.has(_get(i, "doc_id"))]
    doc_order = sorted({_get(i, "doc_id") for i in (load_split_items(a.split_dir, a.split) if not a.tiny else items)
                        if rt.has(_get(i, "doc_id"))})
    if a.n and len(items) > a.n:
        items = random.Random(a.seed).sample(items, a.n)
    dev = input_device(model)

    R = None
    if any(c in BRANCH for c in conds):
        if a.interface:
            a.init = a.interface
        elif a.identity:
            a.a_mode, a.b_mode = "none", "none"
            a.attn_a_mode, a.attn_b_mode = "none", "none"
        rt.kv = "attn" in a.branches or bool(a.interface and "attn" in str(targs.get("branches", "")))
        sd, ad = enc_dims(S, rt, _get(items[0], "doc_id"))
        if a.kv_maps:
            from xenc.xattn_read import attn_dims
            ad = attn_dims(model)                 # the transformed operator lives in the reader's K/V space
            a.attn_a_mode, a.attn_b_mode, a.init = "none", "none", None
        R = build_reader(a, model, sd, ad, dev, record_mag=False)
        if a.kv_maps:
            from xenc.xattn_read import map_kv_states
            _km = torch.load(a.kv_maps, map_location="cpu", weights_only=False)
            _pairs = {int(k): int(v) for k, v in _km["layer_pairs"].items() if int(k) in R.attn_itf.layer_map}
            _orig_load = R.ar.load
            R.ar.load = lambda st, device=None: _orig_load(
                map_kv_states(st, _km, R.ar.rot, _pairs, device=dev) if (st and st.get("kv")) else st)
            log(f"K/V-side maps {a.kv_maps}: reader layers <- encoder layers {_pairs}")
        R.set_gain(a.gain)
        rt.reader_patch = R
        rt.kv = R.needs_kv
        set_tail(R, a.tail)
        log(f"D10 read: tail {a.tail} | branches {R.branches} gates {R.gates()} gain x{a.gain}")
    null_st = mean_states(rt, doc_order) if "null" in conds else None
    need_kv = any(c in CACHE for c in conds)
    sysp = rt.system

    ckpt = os.path.join(out, f"eval_{a.split}{a.tag}.ckpt.jsonl")
    rows, done = [], {}
    if os.path.exists(ckpt) and not a.no_resume:
        for ln in open(ckpt):
            if ln.strip():
                r = json.loads(ln)
                if all(c in r for c in conds):
                    done[r["key"]] = r
    rows = [done[_get(i, "key")] for i in items if _get(i, "key") in done]
    ck = open(ckpt, "w" if a.no_resume else "a")
    t0 = time.time()
    for k, it in enumerate(items):
        key, did = _get(it, "key"), _get(it, "doc_id")
        if key in done:
            continue
        w = wrong_doc_id(did, doc_order)
        rec = {"key": key, "doc_id": did, "wrong": w}
        st_kv = rt.states(did, kv=True) if need_kv else None
        target = gold_text(it)
        tlog = None
        for c in (["text"] if (a.kl_text and "text" in conds) else []) + [c for c in conds if not (a.kl_text and c == "text")]:
            # ---- prompt + optional prefix cache
            cache = None
            if c in CACHE:
                kk, ks = CACHE[c]
                cache = build_prefix_cache(model, st_kv, keep_kv=kk, keep_state=ks, keep_conv=ks)
                ids = question_suffix_ids(tok, hub, it, sysp)
            elif c == "text":
                ids = build_chat(tok, hub, sysp, ctx_prompt(it, docs[did]))
            elif c == "closed_tail":
                ids = student_prompt(rt, it, did, a.tail or 64)
            elif c in ("wrong_doc",) and a.tail and w:
                ids = student_prompt(rt, it, w, a.tail)
            elif c in ("kv", "null") and a.tail:
                ids = student_prompt(rt, it, did, a.tail)
            else:
                ids = build_chat(tok, hub, sysp, q_prompt(it))
            src = BRANCH.get(c)
            st = {"own": lambda: rt.states(did), "wrong": lambda: rt.states(w) if w else None,
                  "null": lambda: null_st}[src]() if src else None
            if src and st is None:
                rec[c] = {"skip": f"no {src} states"}
                continue
            if R is not None:
                R.load(st) if src else R.unload()
            try:
                kl_c = None
                if a.kl_text:
                    tg = text_ids(tok, target)
                    m = len(tg)
                    tc = copy.deepcopy(cache) if cache is not None else None
                    with torch.no_grad():
                        lg = kept_logits(model, torch.tensor([ids + tg]), m + 1, tc).float()
                    if c == "text":
                        tlog = lg
                    lp = torch.log_softmax(lg[:-1], -1)
                    gold_lp = float(lp[torch.arange(m), torch.tensor(tg, device=lp.device)].sum())
                    kl_c = (float(kl_teacher_student(tlog[:-1], lg[:-1])) if tlog is not None else None, gold_lp)
                g, ng = greedy(model, tok, hub, ids, a.budget, cache=cache)
            finally:
                if R is not None:
                    R.unload()
            rec[c] = {"gen": g, "score": float(score(it, g)), "gen_tokens": ng}
            if kl_c is not None:
                rec[c].update(kl_text=kl_c[0], gold_lp=kl_c[1])
            del cache
        if a.js and R is not None and R.ar is not None and st_own_for_js(rt, did) is not None:
            rec["js"] = js_item(a, R, rt, model, tok, hub, docs, it, did, sysp)
        rows.append(rec)
        ck.write(json.dumps(rec) + "\n")
        ck.flush()
        if (len(rows)) % 10 == 0 or a.tiny:
            log(f"  [{len(rows)}/{len(items)}] " + " ".join(f"{c}={_mean(rows, c, 'score'):.3f}" for c in conds)
                + f" | {(time.time() - t0) / max(len(rows) - len(done), 1):.1f}s/item")
    ck.close()

    summ = {}
    for c in conds:
        summ[c] = {"score": _mean(rows, c, "score"), "mean_gen_tokens": _mean(rows, c, "gen_tokens"),
                   "n": sum(1 for r in rows if "score" in r.get(c, {}))}
        if a.kl_text:
            summ[c].update(kl_text=_mean(rows, c, "kl_text"), gold_lp=_mean(rows, c, "gold_lp"))
    if a.js:
        js_rows = [r["js"] for r in rows if r.get("js")]
        if js_rows:
            agg = {}
            for r in js_rows:
                for li, d in r.items():
                    for k_, v in d.items():
                        agg.setdefault(li, {}).setdefault(k_, []).append(v)
            summ["addressing"] = {li: {k_: sum(v) / len(v) for k_, v in d.items()} for li, d in agg.items()}
            summ["addressing"]["n"] = len(js_rows)
            for li, d in summ["addressing"].items():
                if li != "n":
                    log(f"  addressing L{li}: JS {d['js']:.4f} (uniform {d['js_uniform']:.4f}) top5-overlap "
                        f"{d['top5_overlap']:.3f} mass teacher {d['mass_teacher']:.3f} branch {d['mass_branch']:.3f}")
    for x, y in (("kv", "wrong_doc"), ("kv_kvprefix", "wrong_kvprefix")):
        if x in summ and y in summ:
            pair = [r[x]["score"] - r[y]["score"] for r in rows if "score" in r.get(x, {}) and "score" in r.get(y, {})]
            if pair:
                mu = sum(pair) / len(pair)
                sd = (sum((p - mu) ** 2 for p in pair) / max(len(pair) - 1, 1)) ** 0.5
                summ[f"{x}_minus_{y}"] = {"paired_gap": mu, "se": sd / len(pair) ** 0.5, "n_paired": len(pair),
                                         "wins": sum(p > 0 for p in pair), "losses": sum(p < 0 for p in pair)}
    res = {"summary": summ, "n": len(rows), "conds": conds, "args": {k: v for k, v in vars(a).items()},
           "items": rows}
    path = os.path.join(out, f"eval_{a.split}{a.tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    log(f"==== eval_state [{a.split}{a.tag}] n={len(rows)} -> {path}")
    for c in conds:
        s = summ[c]
        kl = f"  KL(text||c) {s['kl_text']:.4f}  gold_lp {s['gold_lp']:.3f}" if a.kl_text else ""
        log(f"  {c:15s} {s['score']:.4f}  n={s['n']:4d}  gen {s['mean_gen_tokens']:.1f}{kl}")
    for k_, v in summ.items():
        if "_minus_" in k_:
            log(f"  {k_}: {v['paired_gap']:+.4f} ± {v['se']:.4f} (n={v['n_paired']}, {v['wins']}/{v['losses']})")
    if torch.cuda.is_available():
        log(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return 0


def _mean(rows, c, k):
    v = [r[c][k] for r in rows if r.get(c, {}).get(k) is not None]
    return sum(v) / len(v) if v else float("nan")


if __name__ == "__main__":
    sys.exit(main())
