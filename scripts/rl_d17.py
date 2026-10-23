"""D17: reward-learn the PAPER-FORM reader-query read (D13's read, xenc/d13_read.py) with the paper's GRPO recipe.

Setup (identical to D13): 2B encoder -> 9B reader; the operator is the encoder's frozen post-RoPE K/V of the document
prefix (fixed harvest, never trained); at the paired reader layers the reader's own state queries it through A (+MLP),
the encoder's frozen q/k/v/gate projections and B; the read replaces the attention output (gated). Only the reader
runs on the question (the encoder only prefills the document). Init = a D13 file (stage-1 distilled A, or a stage-2
registered one). Trainable: A (+MLP), optionally B and the gates.

Objective per step: P prompts (questions of MANY training documents: QuALITY / MuSiQue / LastingBench train docs +
LongHealth folds ∉ {test, val}); per prompt G samples at temperature T from the reader WITH the read of the question's
own document (batched read, xenc/d17_read.py); reward = xenc.doc_tasks.score (MCQ exact label; span token-F1) or
--reward em; group-normalised advantage (GRPO); token-level policy gradient with a truncated importance ratio
(sampler = cached incremental decode, trainer = full forward, both at temperature T); k3 KL to the REFERENCE policy
(the init read, frozen copy) after a KL-free warm-up; plus dist_w · (KL(p_enc-attn ‖ p_read) + rel_w · relMSE)
on the prompt rows against the encoder's own question pass (the D13 distillation target — keeps addressing from
drifting). Informative-group fraction tracked; temperature raised (up to --temp_max) when it collapses.
Selection: greedy kv reward on HELD-OUT DOCUMENTS (D13 stage-2's val set) every --eval_every steps.

    python scripts/rl_d17.py --init outputs/D13_maps/d13_mlp.pt --steps 2000 --out outputs/D17_rl_mlp
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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


# ============================================================================ sampling / log-probs
@torch.no_grad()
def sample_batched(model, R, tok, hub, ids: list[int], G: int, budget: int, temp: float, stops: set,
                   stop_str: str | None = "</answer>"):
    """G samples of one prompt with the D17 read (R loaded with the operator).
    -> (comp [G,L] long, mask [G,L] float, behav_lp [G,L] (log-probs at temperature temp), texts)."""
    from xenc.backbone import decode, input_device
    from xenc.reward import token_logprobs
    dev = input_device(model)
    x = torch.tensor([list(ids)] * G, dtype=torch.long, device=dev)
    prev = R.mode
    R.mode, R.past = "student", {}
    toks, lps = [], []
    nval = [budget] * G
    done = [False] * G
    try:
        out = model(input_ids=x, use_cache=True, logits_to_keep=1)
        for t in range(budget):
            lg = out.logits[:, -1].float() / max(float(temp), 1e-6)
            nxt = torch.multinomial(torch.softmax(lg, -1), 1)
            toks.append(nxt)
            lps.append(token_logprobs(lg.unsqueeze(1), nxt))
            rows = torch.cat(toks, 1).tolist()
            for g in range(G):
                if done[g]:
                    continue
                if rows[g][-1] in stops or (stop_str and tok is not None and
                                            stop_str in tok.decode(rows[g][-12:], skip_special_tokens=True)):
                    done[g], nval[g] = True, t + 1
            if all(done) or t == budget - 1:
                break
            out = model(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
    finally:
        R.mode, R.past = prev, {}
    comp = torch.cat(toks, 1)
    blp = torch.cat(lps, 1)
    L = comp.shape[1]
    mask = (torch.arange(L, device=comp.device).view(1, L) < torch.tensor(nval, device=comp.device).view(G, 1)).float()
    texts = []
    for g in range(G):
        if tok is None:
            texts.append("")
            continue
        txt = decode(tok, hub, comp[g, : nval[g]].tolist())
        if stop_str and stop_str in txt:
            txt = txt.split(stop_str, 1)[0] + stop_str
        texts.append(txt)
    return comp, mask, blp, texts


def policy_logprobs(model, R, ids: list[int], comp: torch.Tensor, temp: float) -> torch.Tensor:
    """Per-token log p_T(comp | prompt) under the current read (R.itf), full forward (grad if enabled): [G,L]."""
    from xenc.backbone import input_device
    from xenc.reward import token_logprobs
    dev = input_device(model)
    G, L = comp.shape
    full = torch.cat([torch.tensor([list(ids)] * G, dtype=torch.long, device=dev), comp.to(dev)], 1)
    prev = R.mode
    R.mode, R.need_logits = "e2e", False
    try:
        lg = model(input_ids=full, use_cache=False, logits_to_keep=L + 1).logits[:, :-1]
    finally:
        R.mode = prev
    return token_logprobs(lg.float() / max(float(temp), 1e-6), comp.to(lg.device))


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--B", default="outputs/D10_maps/encB8_9b_2b_post.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--init", required=True, help="d13 file (stage-1 distilled or stage-2 registered)")
    ap.add_argument("--train", default="quality:150,musique:150,lastingbench:150,longhealth:0")
    ap.add_argument("--task_p", default="quality:0.4,longhealth:0.2,musique:0.2,lastingbench:0.2")
    ap.add_argument("--val", default="quality:10,musique:20,lastingbench:20,longhealth:0")
    ap.add_argument("--val_q_per_doc", type=int, default=4)
    ap.add_argument("--val_fold", type=int, default=1)
    ap.add_argument("--test_fold", type=int, default=0)
    ap.add_argument("--n_val_attn", type=int, default=30)
    ap.add_argument("--train_probe", type=int, default=48,
                    help="greedy kv on this many fixed TRAINING questions at each eval (sharpening vs learning)")
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--prompts", type=int, default=4, help="prompt groups per step")
    ap.add_argument("--group", type=int, default=8, help="samples per prompt")
    ap.add_argument("--mb", type=int, default=8, help="micro-batch (rows) for the policy-gradient forward")
    ap.add_argument("--temp", type=float, default=1.2)
    ap.add_argument("--temp_max", type=float, default=1.6)
    ap.add_argument("--inf_floor", type=float, default=0.3,
                    help="raise temp by 0.1 when the informative-group fraction over the last 25 steps < this")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--reward", choices=["score", "em"], default="score")
    ap.add_argument("--adv", choices=["grpo", "drgrpo", "rloo"], default="grpo")
    ap.add_argument("--is_cap", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.02, help="k3 KL to the reference (init) read policy")
    ap.add_argument("--kl_warmup", type=int, default=100)
    ap.add_argument("--dist_w", type=float, default=1.0, help="KL to the encoder-teacher attention (+relMSE)")
    ap.add_argument("--rel_w", type=float, default=1.0)
    ap.add_argument("--anchor_w", type=float, default=0.0,
                    help="relative-norm L2 anchor to the init: Σ_group ||θ−θ0||² / ||θ0||² (groups A, MLP, B)")
    ap.add_argument("--lr", type=float, default=1e-5, help="A")
    ap.add_argument("--lr_mlp", type=float, default=5e-5)
    ap.add_argument("--lr_B", type=float, default=1e-5)
    ap.add_argument("--lr_g", type=float, default=1e-3)
    ap.add_argument("--train_B", type=int, default=0)
    ap.add_argument("--train_g", type=int, default=0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_budget", type=int, default=32)
    ap.add_argument("--patience", type=int, default=0, help="stop after K evals without a new best (0 = never)")
    ap.add_argument("--time_limit_h", type=float, default=0.0, help="stop cleanly after this many hours (0 = off)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train_from_val", default="", help="as register_d13: 'quality:10' adds that task's VAL docs after the "
                    "first 10 (sorted) to the RL training pool")
    ap.add_argument("--exclude_sets", default="", help="as register_d13: D22 set jsons whose judged/reg/wrong docs are dropped "
                    "from train AND val (hard assertion afterwards)")
    ap.add_argument("--exclude_docs", default="", help="as register_d13: json {task: [doc_id]} of extra docs to drop")
    ap.add_argument("--doc_len", default="", help="as register_d13: 'loogle:0:48000,clipper:0:48000'")
    ap.add_argument("--no_repeat", type=int, default=0, help="1: draw training prompts WITHOUT replacement per task pool "
                    "(reshuffle when a pool is exhausted; epochs per task logged)")
    ap.add_argument("--enc_device", default="", help="encoder (+ operator prefill) on this device, e.g. cuda:0 with "
                    "--device cuda:1 (2-GPU long-document RL); default = --device")
    ap.add_argument("--row_chunk", type=int, default=0, help=">0: no-grad sampling read over row chunks of this size")
    ap.add_argument("--mem_safe", type=int, default=0, help="1: checkpoint the grad-mode read (identical gradients)")
    ap.add_argument("--oom_skip", type=int, default=0, help="1: a CUDA OOM on one prompt group (train) or one val item is "
                    "caught, logged and skipped (val: the item is dropped from that eval and its key recorded)")
    ap.add_argument("--save_items", type=int, default=0, help="1: store per-item val scores (key, kv, wrong, tail) per eval")
    ap.add_argument("--val_fresh", default="", help="like --fresh, REPLACES --val (e.g. quality:val:0:10)")
    ap.add_argument("--save_every", type=int, default=0, help=">0: also save d17_step<N>.pt every N steps (for kv@0 "
                    "checkpoint sweeps; efficiency rule 3)")
    ap.add_argument("--select_from", type=int, default=1,
                    help="evals at steps < this are logged but never selected (hygiene: step 0 excluded); a checkpoint "
                         "is saved only if its val beats the step-0 val")
    ap.add_argument("--fresh", default="",
                    help="docs the init never trained on: 'task:split:start:end,...' (sorted doc ids [start,end)); "
                         "e.g. musique:train:150:450,quality:val:10:57. When set, these REPLACE --train except LongHealth")
    a = ap.parse_args(argv)
    if a.smoke:
        a.train = ",".join(f"{s.split(':')[0]}:2" for s in a.train.split(","))
        a.val = ",".join(f"{s.split(':')[0]}:1" for s in a.val.split(","))
        a.steps, a.eval_every, a.val_q_per_doc, a.n_val_attn, a.kl_warmup = 4, 2, 1, 2, 1
    from eval_d13 import summarize
    from eval_doc import wrong_doc_id
    from fit_d13 import TASK_SPLIT, D13Rt, build, fmt, mean_terms, select_items
    from register_state import _stop_ids
    from xenc.d13_read import save_d13
    from xenc.d17_read import D17Reader
    from xenc.doc_tasks import exact, score
    from xenc.reward import group_advantages, pg_loss
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    reader, enc, tok, itf, lmap = build(a, init=a.init)
    rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens)
    rt.R.detach()                                   # replace D13's single-row patch by the batched one
    R = D17Reader(reader, itf)
    R.mem_safe = bool(a.mem_safe)
    R.row_chunk = int(a.row_chunk)
    rt.R = R
    itf_ref = copy.deepcopy(itf).requires_grad_(False)
    tr_folds = set(range(5)) - {a.test_fold, a.val_fold}
    train, d1 = select_items(a.train, "train", tr_folds, 0, a.seed)
    if a.fresh:
        keep = [s for s in a.train.split(",") if s.startswith("longhealth")]
        train, d1 = select_items(",".join(keep), "train", tr_folds, 0, a.seed) if keep else ([], {})
        for s in a.fresh.split(","):
            name, sp, lo, hi = s.split(":")
            its, dd = select_items(f"{name}:0", sp, None, 0, a.seed)
            ids = sorted(dd)[int(lo):int(hi)]
            train += [i for i in its if _get(i, "doc_id") in set(ids)]
            d1.update({d: dd[d] for d in ids})
        log(f"fresh training docs: {a.fresh} -> {len(train)} items / {len(d1)} docs")
    for spec_ in [x for x in a.train_from_val.split(",") if x]:
        name_, skip_ = spec_.split(":")
        assert name_ != "longhealth", "LongHealth val docs are a fold"
        ex_, d3_ = select_items(f"{name_}:0", "val", None, 0, a.seed)
        keep_ = set(sorted({_get(i, "doc_id") for i in ex_})[int(skip_):])
        train += [i for i in ex_ if _get(i, "doc_id") in keep_]
        d1.update({k: v for k, v in d3_.items() if k in keep_})
    val, d2 = select_items(a.val, "val", {a.val_fold}, a.val_q_per_doc, a.seed + 1)
    if a.val_fresh:
        vrng = random.Random(a.seed + 1)
        val, d2 = [], {}
        for s_ in a.val_fresh.split(","):
            name, sp, lo, hi = s_.split(":")
            its, dd = select_items(f"{name}:0", sp, None, 0, a.seed)
            ids = set(sorted(dd)[int(lo):int(hi)])
            by: dict = {}
            for i in its:
                if _get(i, "doc_id") in ids:
                    by.setdefault(_get(i, "doc_id"), []).append(i)
            for d in sorted(by):
                qs = by[d]
                if a.val_q_per_doc and len(qs) > a.val_q_per_doc:
                    qs = vrng.sample(qs, a.val_q_per_doc)
                val += qs
            d2.update({d: dd[d] for d in ids})
        log(f"fresh val docs: {a.val_fresh} -> {len(val)} items / {len(d2)} docs")
    drop = set()
    if a.exclude_sets or a.exclude_docs or a.doc_len:
        from register_d13 import doc_len_filter, excluded_docs
        drop = excluded_docs(a.exclude_sets)
        if a.exclude_docs:
            X = json.load(open(a.exclude_docs))
            drop |= set(sum(X.values(), [])) if isinstance(X, dict) else set(X)
        keep_len = doc_len_filter(a.doc_len)
        n0 = (len(train), len(val))
        train = [i for i in train if _get(i, "doc_id") not in drop and keep_len(_get(i, "doc_id"))]
        val = [i for i in val if _get(i, "doc_id") not in drop and keep_len(_get(i, "doc_id"))]
        log(f"exclude_sets/exclude_docs ({len(drop)} docs) / doc_len {a.doc_len!r}: train {n0[0]} -> {len(train)}, "
            f"val {n0[1]} -> {len(val)}")
    assert not ({_get(i, "doc_id") for i in train + val} & drop), "an excluded (judged) document reached train/val"
    # LEAKAGE GUARD: no RL-train or val document / question may come from a TEST split (LongHealth: test fold docs)
    pref2task = {k.split("_")[0]: k for k in TASK_SPLIT}
    for pref in sorted({_get(i, "doc_id").split("_")[0] for i in train + val}):
        tname = pref2task[pref]
        t_items = load_split_items(TASK_SPLIT[tname], "test")
        if tname == "longhealth":
            t_items = [i for i in t_items if (_get(i, "meta") or {}).get("fold") == a.test_fold]
        t_docs = {_get(i, "doc_id") for i in t_items}
        t_q = {str(_get(i, "question")).strip() for i in t_items}
        templ = tname in ("longhealth", "contractnli")   # shared question strings by design (folds / 17 fixed hypotheses):
        if not templ:                     # for those the document-level assertion below is the guard
            # generic question strings ("What is the theme of the story?") also occur in test: drop those items
            nb = len(train) + len(val)
            keep_q = lambda i: _get(i, "doc_id").split("_")[0] != pref or str(_get(i, "question")).strip() not in t_q  # noqa: E731
            train, val = [i for i in train if keep_q(i)], [i for i in val if keep_q(i)]
            log(f"leakage guard {tname}: dropped {nb - len(train) - len(val)} items whose question string occurs in test")
        for i in train + val:
            if _get(i, "doc_id").split("_")[0] != pref:
                continue
            assert _get(i, "doc_id") not in t_docs, f"TEST DOC LEAK {_get(i, 'doc_id')}"
            if not templ:
                assert str(_get(i, "question")).strip() not in t_q, f"TEST QUESTION LEAK {_get(i, 'key')}"
    if a.smoke:
        val = val[:6]
    log(f"leakage guard passed: {len(train)} train + {len(val)} val items, none from a test split")
    rt.docs = {**d1, **d2}
    tr_docs, va_docs = {_get(i, "doc_id") for i in train}, {_get(i, "doc_id") for i in val}
    assert not (tr_docs & va_docs), "val docs overlap train docs"
    pools: dict = {}
    for it in train:
        pools.setdefault(_get(it, "doc_id").split("_")[0], []).append(it)
    tp = {k: float(v) for k, v in (s.split(":") for s in a.task_p.split(",")) if k in pools}
    tasks, weights = list(tp), [tp[t] for t in tp]
    val_order: dict = {}
    for d in sorted(va_docs):
        val_order.setdefault(d.split("_")[0], []).append(d)
    B0 = {li: itf.B[str(li)].detach().clone() for li in lmap}
    itf.trainable(train_B=bool(a.train_B), train_g=bool(a.train_g))
    groups = [{"params": list(itf.A.parameters()), "lr": a.lr}]
    if itf.mlp_hidden:
        groups.append({"params": list(itf.M.parameters()), "lr": a.lr_mlp})
    if a.train_B:
        groups.append({"params": list(itf.B.parameters()), "lr": a.lr_B})
    if a.train_g:
        groups.append({"params": [itf.g], "lr": a.lr_g})
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    anchor0 = [([p for p in g["params"]], [p.detach().clone() for p in g["params"]]) for g in groups
               if g["params"] and g["params"][0] is not itf.g]

    def anchor_loss():
        tot = 0.0
        for ps, p0 in anchor0:
            num = sum(((p - q) ** 2).sum() for p, q in zip(ps, p0))
            den = sum((q ** 2).sum() for q in p0).clamp_min(1e-6)
            tot = tot + num / den
        return tot
    stops = _stop_ids(tok)
    rfn = score if a.reward == "score" else exact
    log(f"D17 RL: {a.encoder} -> {a.reader} | layers {lmap} | init {a.init} | trainable "
        f"{sum(p.numel() for p in params) / 1e6:.1f}M | train {len(train)} items / {len(tr_docs)} docs "
        f"{ {t: len(v) for t, v in pools.items()} } | held-out val {len(val)} items / {len(va_docs)} docs | "
        f"P={a.prompts} G={a.group} T={a.temp} beta={a.beta} (warmup {a.kl_warmup}) dist_w={a.dist_w} anchor_w={a.anchor_w} reward={a.reward}")
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    hist = {"steps": [], "evals": []}
    best, n_bad, t0 = -1e9, 0, time.time()
    best_path = os.path.join(a.out, "d17_best.pt")
    ct_cache: dict = {}
    probe = random.Random(a.seed + 7).sample(train, min(a.train_probe, len(train)))
    queues: dict = {}
    epochs: dict = {}

    @torch.no_grad()
    def held_attn():
        rows_ = []
        for it in val[:: max(1, len(val) // max(a.n_val_attn, 1))][: a.n_val_attn]:
            did = _get(it, "doc_id")
            op, n_op, _ = rt.operator(did)
            rows = rt.rows(did, it)
            rec = rt.teacher(did, rows)
            R.load(op, n_op)
            R.mode, R.need_logits, R.rec = "e2e", True, {}
            try:
                reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=False, logits_to_keep=1)
                rows_.append(rt.attn_terms(rec, op, n_op, rt.y_teacher(rec), metrics=True))
            finally:
                R.mode, R.rec, R.need_logits = "off", {}, False
        return mean_terms(rows_)

    base_crit = [None]

    def run_eval(step):
        nonlocal best, n_bad
        rows, kept, skipped = [], [], []
        for it in val:
            did = _get(it, "doc_id")
            w = wrong_doc_id(did, val_order[did.split("_")[0]])
            rec = {}
            for c, d, rd in (("kv", did, True), ("wrong_doc", w, True), ("closed_tail", did, False)):
                if c == "closed_tail" and _get(it, "key") in ct_cache:
                    rec[c] = ct_cache[_get(it, "key")]
                    continue
                if d is None:
                    continue
                try:
                    g, n = rt.gen(d, it, a.eval_budget, read=rd)
                except torch.OutOfMemoryError:
                    if not a.oom_skip:
                        raise
                    R.mode, R.past = "off", {}
                    torch.cuda.empty_cache()
                    log(f"  OOM (val, skipped) {_get(it, 'key')} cond {c}")
                    rec = None
                    break
                rec[c] = {"score": float(score(it, g)), "gen_tokens": n}
                if c == "closed_tail":
                    ct_cache[_get(it, "key")] = rec[c]
            if rec is None:
                skipped.append(_get(it, "key"))
                continue
            rows.append(rec)
            kept.append(it)
        s = summarize(rows, ("closed_tail", "kv", "wrong_doc"))
        tp_sc = [float(score(it, rt.gen(_get(it, "doc_id"), it, a.eval_budget)[0])) for it in probe]
        att = held_attn()
        r = {"step": step, "kv": s["kv"]["score"], "wrong_doc": s["wrong_doc"]["score"],
             "closed_tail": s["closed_tail"]["score"],
             "gap": s.get("kv_minus_wrong_doc", {}).get("paired_gap", float("nan")),
             "gap_se": s.get("kv_minus_wrong_doc", {}).get("se", float("nan")), "gen": s["kv"]["mean_gen_tokens"],
             "gen_wrong": s["wrong_doc"]["mean_gen_tokens"], "attn": att,
             "gates": [round(float(x), 4) for x in itf.g], "n": len(rows), "t": round(time.time() - t0, 1),
             "train_probe_greedy": sum(tp_sc) / max(len(tp_sc), 1)}
        per_task: dict = {}
        for it, rec in zip(kept, rows):
            per_task.setdefault(_get(it, "doc_id").split("_")[0], []).append(rec)
        r["per_task"] = {t: {c: sum(x[c]["score"] for x in v if c in x) / len(v) for c in ("kv", "wrong_doc", "closed_tail")}
                         for t, v in per_task.items()}
        r["oom_skipped"] = skipped
        if a.save_items:
            r["items"] = [{"key": _get(it, "key"), "doc_id": _get(it, "doc_id"),
                           **{c: rec[c]["score"] for c in ("kv", "wrong_doc", "closed_tail") if c in rec}}
                          for it, rec in zip(kept, rows)]
        hist["evals"].append(r)
        crit = r["kv"] + (1e-3 * r["gap"] if math.isfinite(r["gap"]) else 0.0)
        tag = ""
        if step < a.select_from:
            base_crit[0] = crit if base_crit[0] is None else base_crit[0]
        if step >= a.select_from and crit > best and (base_crit[0] is None or crit > base_crit[0]):
            best, n_bad = crit, 0
            save_d13(itf, best_path, {"stage": "D17 rl", "step": step, "val": r, "args": vars(a)})
            tag = " *best*"
        else:
            n_bad += 1
        log(f"  [val @ {step}] held-out docs n={r['n']}: kv {r['kv']:.3f} wrong {r['wrong_doc']:.3f} closed_tail "
            f"{r['closed_tail']:.3f} gap {r['gap']:+.3f} ± {r['gap_se']:.3f} gen {r['gen']:.1f}/{r['gen_wrong']:.1f} | "
            f"train-probe greedy {r['train_probe_greedy']:.3f} (n={len(probe)}){tag}")
        log("            per task " + " ".join(f"{t}: {v['kv']:.2f}/{v['wrong_doc']:.2f}/{v['closed_tail']:.2f}"
                                            for t, v in r["per_task"].items()) + f" | attn {fmt(att)}")

    def dump():
        with open(os.path.join(a.out, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)

    run_eval(0)
    dump()
    temp = a.temp
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        beta = 0.0 if step <= a.kl_warmup else a.beta
        rec_s = {"step": step, "reward": 0.0, "n_signal": 0, "pg": 0.0, "kl": 0.0, "dist": 0.0, "is_trunc": 0.0,
                 "gen_tokens": 0.0, "temp": temp, "tasks": {}}
        for _ in range(a.prompts):
            t = rng.choices(tasks, weights)[0]
            if a.no_repeat:
                if not queues.get(t):
                    queues[t] = rng.sample(pools[t], len(pools[t]))
                    epochs[t] = epochs.get(t, 0) + 1
                it = queues[t].pop()
            else:
                it = pools[t][rng.randrange(len(pools[t]))]
            try:
                did = _get(it, "doc_id")
                rows = rt.rows(did, it)
                op, n_op, _ = rt.operator(did)
                R.load(op, n_op)
                comp, mask, blp, texts = sample_batched(reader, R, tok, a.reader, rows, a.group, a.budget, temp, stops)
                r = torch.tensor([float(rfn(it, x)) for x in texts])
                rec_s["reward"] += float(r.mean()) / a.prompts
                rec_s["gen_tokens"] += float(mask.sum(1).mean()) / a.prompts
                adv = group_advantages(r, a.adv)
                inf = float(adv.abs().max()) > 1e-8
                tt = rec_s["tasks"].setdefault(t, [0, 0, 0.0])
                tt[0] += 1
                tt[1] += int(inf)
                tt[2] += float(r.mean())
                if inf:
                    rec_s["n_signal"] += 1
                    for s0 in range(0, a.group, a.mb):
                        sl = slice(s0, s0 + a.mb)
                        ref = None
                        if beta > 0:
                            R.itf = itf_ref
                            try:
                                with torch.no_grad():
                                    ref = policy_logprobs(reader, R, rows, comp[sl], temp)
                            finally:
                                R.itf = itf
                        with torch.enable_grad():
                            cur = policy_logprobs(reader, R, rows, comp[sl], temp)
                            m_ = mask[sl].to(cur.device)
                            loss, met = pg_loss(cur, m_, adv[sl].to(cur.device), blp[sl].to(cur.device), ref, beta, a.is_cap)
                            frac = comp[sl].shape[0] / a.group
                            (loss * frac / a.prompts).backward()
                        rec_s["pg"] += float(loss) * frac
                        rec_s["kl"] += met.get("kl", 0.0) * frac
                        rec_s["is_trunc"] += met["is_trunc_frac"] * frac
                if a.dist_w > 0:                     # addressing anchor on the prompt rows (encoder teacher)
                    rec = rt.teacher(did, rows)
                    R.load(op, n_op)
                    R.mode, R.need_logits, R.rec = "e2e", True, {}
                    try:
                        with torch.enable_grad():
                            reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=False, logits_to_keep=1)
                            with torch.no_grad():
                                yref = {li: (rec[e]["out"].float() @ B0[li].t()) for li, e in lmap.items()}
                            terms = rt.attn_terms(rec, op, n_op, yref)
                            dist = torch.stack([x["kl"] + a.rel_w * x["rel"] for x in terms.values()]).mean()
                            (a.dist_w * dist / a.prompts).backward()
                    finally:
                        R.mode, R.rec, R.need_logits = "off", {}, False
                    rec_s["dist"] += float(dist) / a.prompts
            except torch.OutOfMemoryError:
                if not a.oom_skip:
                    raise
                R.mode, R.rec, R.need_logits, R.itf, R.past = "off", {}, False, itf, {}
                torch.cuda.empty_cache()
                rec_s["oom"] = rec_s.get("oom", 0) + 1
                log(f"  OOM (train, prompt group skipped) step {step} {_get(it, 'key')}")
        if a.anchor_w > 0:
            with torch.enable_grad():
                anc = anchor_loss()
                (a.anchor_w * anc).backward()
            rec_s["anchor"] = float(anc)
        ns = max(rec_s["n_signal"], 1)
        for k in ("pg", "kl", "is_trunc"):
            rec_s[k] /= ns
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.clip))
        if math.isfinite(gn):
            opt.step()
        rec_s.update(grad_norm=gn, frac_informative=rec_s["n_signal"] / a.prompts, t=round(time.time() - t0, 1))
        hist["steps"].append(rec_s)
        w = hist["steps"][-25:]
        inf25 = sum(x["frac_informative"] for x in w) / len(w)
        if len(hist["steps"]) >= 25 and step % 25 == 0 and inf25 < a.inf_floor and temp < a.temp_max - 1e-6:
            temp = round(min(a.temp_max, temp + 0.1), 3)
            log(f"  informative fraction {inf25:.2f} < {a.inf_floor}: temperature -> {temp}")
        if step % 10 == 0 or step <= 3:
            avg = {k: sum(x[k] for x in w) / len(w) for k in ("reward", "pg", "kl", "dist", "gen_tokens", "is_trunc")}
            agg: dict = {}
            for x in w:
                for tk, v in x["tasks"].items():
                    q = agg.setdefault(tk, [0, 0, 0.0])
                    for j in range(3):
                        q[j] += v[j]
            pt = " ".join(f"{tk} r{v[2] / max(v[0], 1):.2f} inf{v[1] / max(v[0], 1):.2f}" for tk, v in sorted(agg.items()))
            if a.no_repeat:
                pt += " | epochs " + " ".join(f"{tk}:{epochs.get(tk, 0) - 1 + 1 - len(queues.get(tk, [])) / max(len(pools[tk]), 1):.2f}"
                                              for tk in sorted(pools))
            log(f"  [step {step}/{a.steps}] (avg last {len(w)}) reward {avg['reward']:.3f} informative {inf25:.2f} "
                f"pg {avg['pg']:+.4f} kl {avg['kl']:.4f} dist {avg['dist']:.4f} trunc {avg['is_trunc']:.3f} "
                f"gen {avg['gen_tokens']:.1f} anc {rec_s.get('anchor', 0.0):.2e} |g| {gn:.3f} T {temp} | {pt} ({(time.time() - t0) / step:.1f}s/step)")
        if a.save_every and step % a.save_every == 0:
            save_d13(itf, os.path.join(a.out, f"d17_step{step}.pt"), {"stage": "D17 rl", "step": step, "args": vars(a)})
            log(f"  saved {a.out}/d17_step{step}.pt")
        if step % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        stop = bool(a.time_limit_h and time.time() - t0 > a.time_limit_h * 3600)
        if step % a.eval_every == 0 or step == a.steps or stop:
            run_eval(step)
            dump()
            if a.patience and n_bad >= a.patience:
                log(f"early stop: {n_bad} evals without a new best")
                break
        if stop:
            log("time limit reached")
            break
    save_d13(itf, os.path.join(a.out, "d17_last.pt"), {"stage": "D17 rl", "step": len(hist["steps"]), "args": vars(a)})
    dump()
    if torch.cuda.is_available():
        log(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    log(f"done: {len(hist['steps'])} steps, best val crit {best:.4f} -> {best_path} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
