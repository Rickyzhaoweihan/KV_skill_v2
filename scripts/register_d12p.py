"""D12p (copy of D13 for the 27B reader). D13 stage 2: end-to-end registration of the reader-query read (xenc/d13_read.py) with the paper's objective.

Same objective as register_doc.py / register_state.py (losses from xenc/losses.py): teacher = the reader with the
document in context (ctx_prompt + gold answer), student = the reader on its own doc tail + question + gold answer with
the D13 read of the encoder's operator replacing the attention output at the paired layers (gated: own + g·(read−own)):
    loss = kl·KL(p_text ‖ p_student)[answer tokens] + ce·CE(gold) + dist·(KL(p_enc-attn ‖ p_read) + relMSE(B o, B₀ o_enc))
The last term is stage 1's label-free attention distillation against the encoder's own pass over the same rows; it keeps
the read addressing the document (no collapse to a document-independent shift). A (+MLP), B, gates train; the operator
(the encoder's harvested K/V) and both backbones stay frozen. Init: stage-1 file (A) + D10's encoder-side ridge B.
Selection on HELD-OUT DOCUMENTS (QuALITY/MuSiQue/LastingBench val docs, LongHealth --val_fold): greedy kv / wrong_doc
every --eval_every steps.

    python scripts/register_d13.py --init outputs/D13_maps/d13_lin.pt --dist_w 1.0 --steps 2500 --out outputs/D13_reg_w1
"""

from __future__ import annotations

import argparse
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
from register_doc import SYSTEM_ANSWER, _get, log  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--B", default="outputs/D10_maps/encB8_9b_2b_post.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--init", required=True, help="stage-1 d13 file")
    ap.add_argument("--train", default="quality:150,musique:150,lastingbench:150,longhealth:0")
    ap.add_argument("--task_p", default="quality:0.4,longhealth:0.2,musique:0.2,lastingbench:0.2")
    ap.add_argument("--val", default="quality:10,musique:20,lastingbench:20,longhealth:0")
    ap.add_argument("--val_q_per_doc", type=int, default=4)
    ap.add_argument("--val_fold", type=int, default=1)
    ap.add_argument("--test_fold", type=int, default=0)
    ap.add_argument("--n_val_attn", type=int, default=30)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--ce", type=float, default=0.5)
    ap.add_argument("--dist_w", type=float, default=1.0)
    ap.add_argument("--rel_w", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5, help="A")
    ap.add_argument("--lr_mlp", type=float, default=5e-5)
    ap.add_argument("--lr_B", type=float, default=1e-5)
    ap.add_argument("--lr_g", type=float, default=1e-3)
    ap.add_argument("--train_B", type=int, default=1)
    ap.add_argument("--train_g", type=int, default=1)
    ap.add_argument("--delta_rank", type=int, default=0, help=">0: freeze the init A / B, train rank-r deltas on both "
                    "+ write-back bias + gates, with --reg·relative drift ‖Δ‖²/‖W0‖² (D12 recipe)")
    ap.add_argument("--reg", type=float, default=1.0)
    ap.add_argument("--lr_delta", type=float, default=5e-5)
    ap.add_argument("--train_mlp", type=int, default=-1, help="-1: yes unless --delta_rank")
    ap.add_argument("--train_from_val", default="", help="e.g. quality:10 — val-split docs AFTER the first 10 "
                    "(sorted; the first 10 are the held-out val docs) are added to training")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--teacher_cache", default="", help="D12p: train_encside_B.py cache dir (27B text-teacher top-k)")
    ap.add_argument("--teacher_only", type=int, default=0, help="D12p: train only on items whose 27B text teacher "
                    "is in --teacher_cache with matching rows")
    ap.add_argument("--itf_device", default=None, help="D12p: device of the interface + cached tensors")
    ap.add_argument("--skip_eval0", action="store_true")
    ap.add_argument("--select_min_step", type=int, default=0, help="D12p deep: only checkpoints at step >= this are "
                    "selectable (step 0 = the init)")
    ap.add_argument("--lr_g_deep", type=float, default=0.0, help="D12p deep: gate step size on --deep_layers")
    ap.add_argument("--bar_select", type=int, default=0, help="1: val points before --select_min_step set the bar that a "
                    "selected checkpoint must BEAT (register_d13 semantics); 0: they are only not selectable")
    ap.add_argument("--exclude_sets", default="", help="lever 1: D22 set files whose judged/reg/wrong-map docs are dropped")
    ap.add_argument("--exclude_docs", default="", help="lever 1: json {task: [doc_id]} of near-duplicate docs to drop")
    ap.add_argument("--doc_len", default="", help='lever 1: e.g. "loogle:32000:48000,clipper:32000:48000"')
    ap.add_argument("--dist_max_cols", type=int, default=0, help="D12p: drop the distillation term (keep KL+CE) on items "
                    "with operator tokens x rows above this (its [Hkv,G,T,C] attention logits are what runs out of memory)")
    ap.add_argument("--cached_only", type=int, default=0, help="1: train only on items whose encoder record is in "
                    "--enc_cache (subset cache); every val item must be cached")
    ap.add_argument("--save_items", type=int, default=0, help="1: per-item val scores in history.json (paired val)")
    ap.add_argument("--deep_layers", default="35,39,43,47")
    ap.add_argument("--budget", type=int, default=32)
    # D12b: train the D24 hypernetwork H on the 27B (base interface FROZEN, deltas merged; default off)
    ap.add_argument("--extra_splits", default="", help="D12b: 'name=data/<name>_split,...' train-only splits (DS data)")
    ap.add_argument("--op_alias", default="", help="D12b: 'nqamc=narrativeqa,looglemc=loogle' derived splits read the "
                    "source split's cached operator (same document file; prefix length asserted)")
    ap.add_argument("--hyper", type=int, default=0, help="D12b: 1 = train H (xenc/d24_hyper.py v2) instead of the base")
    ap.add_argument("--h_rank", type=int, default=64)
    ap.add_argument("--h_bias_k", type=int, default=16)
    ap.add_argument("--h_hidden", type=int, default=1024)
    ap.add_argument("--h_lr", type=float, default=2e-4)
    ap.add_argument("--h_lr_basis", type=float, default=2e-4)
    ap.add_argument("--h_warmup", type=int, default=50)
    ap.add_argument("--h_stat_docs", type=int, default=64)
    ap.add_argument("--h_input", default="op", choices=("op", "zero"))
    ap.add_argument("--h_kind", default="adapter", choices=("adapter", "slots"), help="D12b: adapter = D24 v2 query-map "
                    "adapter; slots = SlotH per-document memory slots (xenc/d12b_slots.py)")
    ap.add_argument("--h_slots", type=int, default=64)
    ap.add_argument("--h_blocks", type=int, default=2)
    ap.add_argument("--h_swap_w", type=float, default=0.0, help="D12b: >0 adds the anchored own-vs-swap contrast on gold "
                    "CE: w·[relu(margin − (CE_swap − CE_own)) + relu(CE_swap − CE_base − eps)] (swap = another training "
                    "doc of the same task; base = no adapter)")
    ap.add_argument("--h_swap_p", type=float, default=0.5, help="fraction of items that also get the contrast")
    ap.add_argument("--h_swap_margin", type=float, default=0.1)
    ap.add_argument("--h_anchor_eps", type=float, default=0.05)
    ap.add_argument("--val_extra", default="", help="D12b: extra val docs 'loogle:val:8,narrativeqa:train:-8' "
                    "(fit_d12p.extra_val_items); with --cached_only 2 they need cached operators only")
    ap.add_argument("--val_extra_q", type=int, default=4)
    ap.add_argument("--save_every", type=int, default=0, help="D12b: >0 saves the interface (deltas folded) to "
                    "<out>/ckpt_<step>.pt every N steps (checkpoint sweeps)")
    ap.add_argument("--select_balanced", type=int, default=0, help="D12b: 1 = select on the per-task MEAN val kv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enc_device", default=None, help="D12p: encoder device (e.g. cuda:1)")
    ap.add_argument("--reader_mem", default=None, help="D12p: shard the reader, per-GPU budgets '27GiB,27GiB'")
    ap.add_argument("--enc_cache", default=None, help="D12p phase B: cache_d12p.py dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_seed", type=int, default=1, help="fixed held-out val question sample (seed 0 runs used 1)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.train = ",".join(f"{s.split(':')[0]}:2" for s in a.train.split(","))
        a.val = ",".join(f"{s.split(':')[0]}:1" for s in a.val.split(","))
        a.steps, a.eval_every, a.val_q_per_doc, a.n_val_attn = 6, 3, 1, 2
    from eval_doc import wrong_doc_id
    from eval_d12p import summarize
    from fit_d12p import D13Rt, build, make_rt, fmt, mean_terms, select_items
    from register_state import kept_logits
    from xenc.backbone import build_chat, text_ids
    from xenc.d12p_read import save_d13
    from xenc.doc_tasks import ctx_prompt, gold_text, score
    from xenc.losses import answer_ce, kl_teacher_student
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    from fit_d12p import add_extra_splits, set_op_alias
    add_extra_splits(a.extra_splits)
    set_op_alias(a.op_alias)
    reader, enc, tok, itf, lmap = build(a, init=a.init)
    rt = make_rt(a, reader, enc, tok, itf)
    rt.R.gate_tf = True                     # D12p deep: diagnostics' teacher forcing honours the gates (gate-0 layers)
    tr_folds = set(range(5)) - {a.test_fold, a.val_fold}
    train, d1 = select_items(a.train, "train", tr_folds, 0, a.seed)
    val, d2 = select_items(a.val, "val", {a.val_fold}, a.val_q_per_doc, a.val_seed)
    if a.val_extra:                                                # D12b: per-task-balanced val (LooGLE, NarrativeQA)
        from fit_d12p import extra_val_items
        vx, dx = extra_val_items(a.val_extra, a.val_extra_q, a.val_seed, a.doc_len)
        val += vx
        d2.update(dx)
        log(f"  val_extra {a.val_extra}: +{len(vx)} items / {len(dx)} docs")
    for spec in [x for x in a.train_from_val.split(",") if x]:
        name, skip = spec.split(":")
        assert name != "longhealth", "LongHealth val docs are a fold; use --val_fold"
        ex, d3 = select_items(f"{name}:0", "val", None, 0, a.seed)
        keep = set(sorted({_get(i, "doc_id") for i in ex})[int(skip):])
        train += [i for i in ex if _get(i, "doc_id") in keep]
        d1.update({k: v for k, v in d3.items() if k in keep})
    from fit_d12p import lever_filter
    (train, val), _drop = lever_filter(a, train, val)              # D12p lever 1 (no-op unless the flags are set)
    if a.cached_only:                                              # D12p lever: only items with a cached encoder record
        assert hasattr(rt, "find"), "--cached_only needs --enc_cache"
        rt.docs = {**d1, **d2}
        n0 = len(train)
        if a.cached_only == 2:                                     # D12b: a cached OPERATOR suffices (record optional)
            from fit_d12p import op_path
            from fit_d12p import find_op
            has_op = lambda d: find_op(rt.cache_dirs, d) is not None   # noqa: E731
            train = [i for i in train if has_op(_get(i, "doc_id"))]
            n_rec = sum(rt.find(_get(i, "doc_id"), rt.rows(_get(i, "doc_id"), i)) is not None for i in train)
            log(f"  cached_only 2: {n_rec} of {len(train)} train items have an encoder record (distillation); "
                f"the rest train on KL+CE")
        else:
            train = [i for i in train if rt.find(_get(i, "doc_id"), rt.rows(_get(i, "doc_id"), i)) is not None]
        miss = [i for i in val if rt.find(_get(i, "doc_id"), rt.rows(_get(i, "doc_id"), i)) is None]
        if a.cached_only == 2:                                     # D12b: val needs operators; records only for diag
            from fit_d12p import op_path
            from fit_d12p import find_op
            nop = [i for i in val if find_op(rt.cache_dirs, _get(i, "doc_id")) is None]
            assert not nop, f"{len(nop)} val items have no cached operator"
            log(f"  cached_only 2: {len(miss)} of {len(val)} val items have no encoder record (no attention diagnostics)")
        else:
            assert not miss, f"{len(miss)} val items have no cached encoder record (val must be identical across arms)"
        log(f"  cached_only: train {n0} -> {len(train)} items with a cached encoder record")
    rt.docs = {**d1, **d2}
    tr_docs, va_docs = {_get(i, "doc_id") for i in train}, {_get(i, "doc_id") for i in val}
    assert not (tr_docs & va_docs), "val docs overlap train docs"
    pools: dict = {}
    for it in train:
        pools.setdefault(_get(it, "doc_id").split("_")[0], []).append(it)
    tp = {k: float(v) for k, v in (s.split(":") for s in a.task_p.split(",")) if k in pools}
    tasks, weights = list(tp), [tp[t] for t in tp]
    val_order = {}
    for d in sorted(va_docs):
        val_order.setdefault(d.split("_")[0], []).append(d)
    B0 = {li: itf.B[str(li)].detach().clone() for li in lmap}
    Bb0 = {li: itf.Bb[str(li)].detach().clone() for li in lmap}          # D12p: B2 maps carry a bias
    train_mlp = bool(a.train_mlp) if a.train_mlp >= 0 else not a.delta_rank
    H = None
    diag_val = val
    if a.cached_only == 2:
        diag_val = [i for i in val if rt.find(_get(i, "doc_id"), rt.rows(_get(i, "doc_id"), i)) is not None]
    if a.hyper and a.h_kind == "slots":                            # D12b H-slots: per-document memory slots
        from xenc.d12b_hyper import wire
        from xenc.d12b_slots import SlotH
        assert not a.train_g and not a.lr_g_deep and not a.delta_rank, "--hyper: pass --train_g 0 --delta_rank 0"
        if itf.delta_rank:
            itf.merge_delta()
        for p in itf.parameters():
            p.requires_grad_(False)
        H = SlotH(itf, n_slots=a.h_slots, n_blocks=a.h_blocks, seed=a.seed).to(itf.g.device)
        wire(rt, H)
        itf._slot_stats = []
        groups = [{"params": list(H.parameters()), "lr": a.h_lr}]
        train_mlp = False
        log(f"D12b H-slots: {H.n_params() / 1e6:.1f}M params, {a.h_slots} slots x {len(H.layers)} read layers "
            f"({H.adapter_numel()} numbers/doc), {a.h_blocks} blocks | base frozen (deltas merged) | lr {a.h_lr} "
            f"warmup {a.h_warmup} | swap contrast w {a.h_swap_w} p {a.h_swap_p} margin {a.h_swap_margin} eps {a.h_anchor_eps}")
    elif a.hyper:                                                  # D12b: frozen base + H(operator) per document
        from xenc.d12b_hyper import h_param_groups, init_stats, wire
        from xenc.d24_hyper import D24Hyper
        assert not a.train_g and not a.lr_g_deep and not a.delta_rank, "--hyper: pass --train_g 0 --delta_rank 0"
        if itf.delta_rank:
            itf.merge_delta()
        for p in itf.parameters():
            p.requires_grad_(False)
        H = D24Hyper(itf, rank=a.h_rank, hidden=a.h_hidden, bias_k=a.h_bias_k, gate=True, input_mode=a.h_input,
                     seed=a.seed, arch="v2").to(itf.g.device)
        wire(rt, H)
        init_stats(rt, H, tr_docs, a.h_stat_docs, log)
        groups = h_param_groups(H, a.h_lr, a.h_lr_basis)
        train_mlp = False
        log(f"D12b H: {H.n_params() / 1e6:.1f}M params, adapter {H.adapter_numel()} numbers/doc | rank {a.h_rank} "
            f"bias_k {a.h_bias_k} input {a.h_input} | base frozen (deltas merged) | lr {a.h_lr}/{a.h_lr_basis} "
            f"warmup {a.h_warmup}")
    elif a.delta_rank:
        itf.trainable(train_B=False, train_g=bool(a.train_g))
        for p in itf.A.parameters():
            p.requires_grad_(False)
        itf.add_delta(a.delta_rank, a.seed)
        groups = [{"params": list(itf.D.parameters()), "lr": a.lr_delta}]
        if a.train_B:
            for p in itf.Bb.parameters():
                p.requires_grad_(True)
            groups.append({"params": list(itf.Bb.parameters()), "lr": a.lr_delta})
    else:
        itf.trainable(train_B=bool(a.train_B), train_g=bool(a.train_g))
        groups = [{"params": list(itf.A.parameters()), "lr": a.lr}]
        if a.train_B:
            groups.append({"params": list(itf.B.parameters()) + list(itf.Bb.parameters()), "lr": a.lr_B})
    for p in itf.M.parameters():
        p.requires_grad_(train_mlp and not a.hyper)
    if itf.mlp_hidden and train_mlp:
        groups.append({"params": list(itf.M.parameters()), "lr": a.lr_mlp})
    sync_g = lambda: None                                          # noqa: E731
    if a.train_g and a.lr_g_deep > 0:
        # D12p deep: per-layer gate step size. g = g0 + s ⊙ θ, θ trained by Adam at lr_g; s = lr_g_deep / lr_g on the
        # --deep_layers, 1 elsewhere (Adam is invariant to gradient scale, so a separate step size needs this form).
        import types as _types
        deep = {int(x) for x in a.deep_layers.split(",") if x}
        g0 = itf.g.detach().clone()
        sc = torch.tensor([a.lr_g_deep / a.lr_g if li in deep else 1.0 for li in itf.layers], device=g0.device)
        theta = torch.nn.Parameter(torch.zeros_like(g0))
        itf.g.requires_grad_(False)
        itf.gate = _types.MethodType(lambda self, li: g0[self._gi[li]] + sc[self._gi[li]] * theta[self._gi[li]], itf)

        def sync_g():
            with torch.no_grad():
                itf.g.copy_(g0 + sc * theta)
        groups.append({"params": [theta], "lr": a.lr_g})
        log(f"gates: deep layers {sorted(deep)} step size {a.lr_g_deep}, others {a.lr_g}")
    elif a.train_g:
        groups.append({"params": [itf.g], "lr": a.lr_g})
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s_: min(1.0, (s_ + 1) / a.h_warmup)) if a.hyper else None
    log(f"D13 stage 2: {a.encoder} -> {a.reader} | layers {lmap} | init {a.init} | trainable "
        f"{sum(p.numel() for p in params) / 1e6:.1f}M | train {len(train)} items / {len(tr_docs)} docs "
        f"{ {t: len(v) for t, v in pools.items()} } | held-out val {len(val)} items / {len(va_docs)} docs | "
        f"kl {a.kl} ce {a.ce} dist {a.dist_w} | delta_rank {a.delta_rank} reg {a.reg} mlp {'train' if train_mlp else 'frozen'}")
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    hist = {"steps": [], "evals": []}
    best, t0 = -1e9, time.time()
    best_path = os.path.join(a.out, "d13_best.pt")
    ct_cache: dict = {}

    def run_eval(step):
        nonlocal best
        sync_g()
        if H is not None:
            H.eval()                                               # D12b: no stat updates on val docs
            if getattr(itf, "_slot_stats", None) is not None:
                itf._slot_stats = []
        rows = []
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
                g, n = rt.gen(d, it, a.budget, read=rd)
                rec[c] = {"score": float(score(it, g)), "gen_tokens": n}
                if c == "closed_tail":
                    ct_cache[_get(it, "key")] = rec[c]
            if H is not None and w is not None:                    # D12b: own operator, ANOTHER doc's H-adapter
                from xenc.d12b_hyper import adapter_of, override
                with override(rt, adapter_of(rt, w)):
                    g, n = rt.gen(did, it, a.budget)
                rec["kv_Hswap"] = {"score": float(score(it, g)), "gen_tokens": n}
            rows.append(rec)
        s = summarize(rows, ("closed_tail", "kv", "wrong_doc"))
        att = mean_terms([rt.s1_pass(_get(it, "doc_id"), it, onpolicy=True, grad=False, metrics=True)
                          for it in diag_val[:: max(1, len(diag_val) // max(a.n_val_attn, 1))][: a.n_val_attn]])
        r = {"step": step, "kv": s["kv"]["score"], "wrong_doc": s["wrong_doc"]["score"],
             "closed_tail": s["closed_tail"]["score"],
             "gap": s.get("kv_minus_wrong_doc", {}).get("paired_gap", float("nan")),
             "gap_se": s.get("kv_minus_wrong_doc", {}).get("se", float("nan")), "gen": s["kv"]["mean_gen_tokens"],
             "gen_wrong": s["wrong_doc"]["mean_gen_tokens"], "attn": att,
             "gates": [round(float(x), 4) for x in itf.g], "n": len(rows),
             "drift": float(itf.delta_reg()) if itf.delta_rank else None}
        per_task = {}
        for it, rec in zip(val, rows):
            per_task.setdefault(_get(it, "doc_id").split("_")[0], []).append(rec)
        r["per_task"] = {t: {c: sum(x[c]["score"] for x in v if c in x) / len(v) for c in ("kv", "wrong_doc", "closed_tail")}
                         for t, v in per_task.items()}
        if a.save_items:                                           # paired val across runs on the SAME items
            r["items"] = [[str(_get(it, "key")), rec["kv"]["score"], rec["wrong_doc"]["score"] if "wrong_doc" in rec else None,
                           rec["closed_tail"]["score"]] for it, rec in zip(val, rows)]
        if H is not None:
            from xenc.d12b_hyper import adapter_of
            pr = [x["kv"]["score"] - x["kv_Hswap"]["score"] for x in rows if "kv_Hswap" in x]
            mu = sum(pr) / max(len(pr), 1)
            se = (sum((q - mu) ** 2 for q in pr) / max(len(pr) - 1, 1)) ** 0.5 / max(len(pr), 1) ** 0.5
            ss = {}
            if getattr(itf, "_slot_stats", None) is not None:
                from xenc.d12b_slots import slot_stats_summary
                ss = slot_stats_summary(itf)
            ads = [adapter_of(rt, d) for d in sorted({_get(i, "doc_id") for i in val})[:12]]
            r["h"] = {"own_minus_swap": mu, "se": se, "wins": sum(q > 0 for q in pr), "losses": sum(q < 0 for q in pr),
                      "kv_Hswap": sum(x["kv_Hswap"]["score"] for x in rows if "kv_Hswap" in x) / max(len(pr), 1),
                      "M_norm": sum(float(x_["_norm"]) for x_ in ads) / len(ads),
                      "doc_norm": sum(float(x_.get("_doc_norm", 0.0)) for x_ in ads) / len(ads),
                      "dg": [round(float(v), 4) for v in ads[0]["dg"]] if ads[0].get("dg") is not None else None,
                      "slots": ss}
            if a.save_items:
                r["items_Hswap"] = [[str(_get(it, "key")), rec["kv_Hswap"]["score"]] for it, rec in zip(val, rows) if "kv_Hswap" in rec]
            log(f"  [H @ {step}] own-swap {mu:+.3f} ± {se:.3f} ({r['h']['wins']}/{r['h']['losses']}) kv_Hswap "
                f"{r['h']['kv_Hswap']:.3f} | |M|² {r['h']['M_norm']:.2e} doc part {r['h']['doc_norm']:.2e} | dg {r['h']['dg']}"
                + (f" | slot share {ss['share']:.4f} entropy {ss['entropy']:.3f} by layer {ss['share_by_layer']}" if ss else ""))
        hist["evals"].append(r)
        crit = r["kv"] + (1e-3 * r["gap"] if math.isfinite(r["gap"]) else 0.0)
        if a.select_balanced:                                      # D12b: per-task mean (every task counts equally)
            r["kv_balanced"] = sum(v["kv"] for v in r["per_task"].values()) / max(len(r["per_task"]), 1)
            crit = r["kv_balanced"]
            log(f"  [val @ {step}] per-task-balanced kv {crit:.4f} over {len(r['per_task'])} tasks")
        tag = ""
        if a.bar_select and step < a.select_min_step:              # D13 hygiene: early points only set the bar
            best = max(best, crit)
        elif crit > best and step >= a.select_min_step:
            best = crit
            if H is not None and a.h_kind == "slots":
                from xenc.d12b_slots import save_slots
                save_slots(H, os.path.join(a.out, "h_best.pt"), {"step": step, "val": r, "args": vars(a), "base": a.init})
            elif H is not None:                                    # D12b: the base is frozen; save H
                from xenc.d24_hyper import save_h
                save_h(H, os.path.join(a.out, "h_best.pt"), {"step": step, "val": r, "args": vars(a), "base": a.init})
            else:
                save_d13(itf, best_path, {"stage": "register", "step": step, "val": r, "args": vars(a)})
            tag = " *best*"
        log(f"  [val @ {step}] held-out docs n={r['n']}: kv {r['kv']:.3f} wrong {r['wrong_doc']:.3f} closed_tail "
            f"{r['closed_tail']:.3f} gap {r['gap']:+.3f} ± {r['gap_se']:.3f} gen {r['gen']:.1f}/{r['gen_wrong']:.1f} "
            f"gates {r['gates']}{tag}")
        log(f"            per task " + " ".join(f"{t}: {v['kv']:.2f}/{v['wrong_doc']:.2f}/{v['closed_tail']:.2f}"
                                             for t, v in r["per_task"].items()) + f" | attn {fmt(att)}")

    def dump():
        with open(os.path.join(a.out, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)

    tcache = {}                                                    # D12p: key -> (student ids, top-k logp, idx)
    if a.teacher_cache:
        import glob
        for pth in sorted(glob.glob(os.path.join(a.teacher_cache, "train_*.pt"))):
            for r in torch.load(pth, weights_only=False):
                tcache[r["key"]] = (list(r["sids"]), r["t_lp"], r["t_idx"])
        log(f"teacher cache: {len(tcache)} items from {a.teacher_cache}")
    n_tc = 0
    n_nodist = 0
    if a.teacher_only:
        for t in list(pools):
            pools[t] = [it for it in pools[t] if _get(it, "key") in tcache and
                        tcache[_get(it, "key")][0] == rt.rows(_get(it, "doc_id"), it, text_ids(tok, gold_text(it)))]
            if not pools[t]:
                del pools[t]
        tasks = [t for t in tasks if t in pools]
        weights = [tp[t] for t in tasks]
        log(f"teacher_only: train pools { {t: len(v) for t, v in pools.items()} }")
    if a.skip_eval0:
        log("skipping val @ 0")
    else:
        run_eval(0)
        dump()
    R = rt.R
    skipped = 0
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        if H is not None:
            H.train()
        acc = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "dist": 0.0, "n": 0}
        for _ in range(a.batch):
            t = rng.choices(tasks, weights)[0]
            it = pools[t][rng.randrange(len(pools[t]))]
            did = _get(it, "doc_id")
            target = gold_text(it)
            tt = text_ids(tok, target)
            m = len(tt)
            tids = build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, rt.docs[did])) + tt
            if len(tids) > a.max_tokens:
                skipped += 1
                continue
            R.mode = "off"
            rows = rt.rows(did, it, tt)
            tc = tcache.get(_get(it, "key"))
            if tc is not None and tc[0] == rows:                   # D12p: cached 27B text-teacher top-k
                t_log = None
                n_tc += 1
            else:
                try:
                    with torch.no_grad():
                        t_log = kept_logits(reader, torch.tensor([tids]), m + 1).detach()
                except torch.OutOfMemoryError:                 # D12p: online 27B text teacher too long for memory
                    torch.cuda.empty_cache()
                    skipped += 1
                    log(f"  OOM in the text teacher on {did} ({len(tids)} tokens); item skipped")
                    continue
            op, n_op, _ = rt.operator(did)
            use_dist = a.dist_w > 0 and not (a.dist_max_cols and n_op * len(rows) > a.dist_max_cols)
            if use_dist and a.cached_only == 2 and rt.find(did, rows) is None:     # D12b: no record -> KL+CE
                use_dist = False
            n_nodist += int(a.dist_w > 0 and not use_dist)                 # D12p: long docs train on KL+CE only
            rec = rt.teacher(did, rows) if use_dist else None
            R.load(op, n_op)
            R.mode, R.need_logits, R.rec = "e2e", use_dist, {}
            try:
                with torch.enable_grad():
                    s_log = kept_logits(reader, torch.tensor([rows]), m + 1)
                    if t_log is None:
                        sl = torch.log_softmax(s_log[:-1].float(), -1)
                        tl, ti = tc[1][:-1].to(sl.device).float(), tc[2][:-1].to(sl.device).long()
                        kl = (tl.exp() * (tl - sl.gather(-1, ti))).sum(-1).mean()
                    else:
                        kl = kl_teacher_student(t_log[:-1], s_log[:-1])
                    ce = answer_ce(s_log, rows[-m:])
                    loss = a.kl * kl + a.ce * ce
                    dist = torch.zeros((), device=kl.device)
                    if use_dist:
                        with torch.no_grad():
                            yref = {li: (rec[e]["out"].float() @ B0[li].t() + Bb0[li]) for li, e in lmap.items()}
                        terms = rt.attn_terms(rec, op, n_op, yref)
                        dist = torch.stack([x["kl"] + a.rel_w * x["rel"] for x in terms.values()]).mean()
                        loss = loss + a.dist_w * dist.to(loss.device)
                    if a.delta_rank and a.reg > 0:
                        loss = loss + a.reg * itf.delta_reg().to(loss.device)
                    if not math.isfinite(float(loss)):
                        skipped += 1
                        continue
                    (loss / a.batch).backward()
                    if H is not None and a.h_swap_w > 0 and rng.random() < a.h_swap_p:
                        # D12b anchored own-vs-swap contrast. Separate backward INSIDE the override: the checkpointed
                        # read recomputes with the adapter installed at backward time, so state must not change.
                        from xenc.d12b_hyper import override
                        cands = [x_ for x_ in pools[t] if _get(x_, "doc_id") != did]
                        if cands:
                            dsw = _get(cands[rng.randrange(len(cands))], "doc_id")
                            op_sw = rt.operator(dsw)[0]
                            R.rec, R.need_logits = {}, False
                            with torch.no_grad(), override(rt, False):
                                R.load(op, n_op)
                                ce_b = answer_ce(kept_logits(reader, torch.tensor([rows]), m + 1), rows[-m:])
                            ad_sw = H({k_: v_ for k_, v_ in op_sw.items() if k_ != "ids"})
                            with override(rt, ad_sw):
                                R.load(op, n_op)                   # OWN operator, ANOTHER doc's slots / adapter
                                ce_sw = answer_ce(kept_logits(reader, torch.tensor([rows]), m + 1), rows[-m:])
                                ce_o = float(ce.detach())
                                con = torch.relu(a.h_swap_margin - (ce_sw - ce_o)) + \
                                    torch.relu(ce_sw - float(ce_b) - a.h_anchor_eps)
                                (a.h_swap_w * con / a.batch).backward()
                            acc["swap_n"] = acc.get("swap_n", 0) + 1
                            acc["d_sw"] = acc.get("d_sw", 0.0) + float(ce_sw.detach()) - ce_o
                            acc["d_b"] = acc.get("d_b", 0.0) + float(ce_b) - ce_o
            except torch.OutOfMemoryError:                     # D12p: skip the rare over-long item, keep training
                R.rec, loss, kl, ce, dist, s_log, rec, terms = {}, None, None, None, None, None, None, None
                torch.cuda.empty_cache()
                skipped += 1
                log(f"  OOM on {did} ({len(rows)} rows, {n_op} operator tokens); item skipped")
                continue
            finally:
                R.mode, R.rec = "off", {}
            for k, v in (("loss", loss), ("kl", kl), ("ce", ce), ("dist", dist)):
                acc[k] += float(v)
            acc["n"] += 1
        if acc["n"]:
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            r = {k: (v / acc["n"] if k not in ("n", "swap_n", "d_sw", "d_b") else v) for k, v in acc.items()}
            r.update(step=step, grad_norm=float(gn), t=round(time.time() - t0, 1))
            hist["steps"].append(r)
            if step <= 3:
                from fit_d12p import peak_mem
                peak_mem(f"stage 2 step {step}")
            if step % 25 == 0 or step <= 3:
                sync_g()
                w = hist["steps"][-25:]
                avg = {k: sum(x[k] for x in w) / len(w) for k in ("loss", "kl", "ce", "dist")}
                if H is not None and getattr(itf, "_slot_stats", None) is not None:
                    from xenc.d12b_slots import slot_stats_summary
                    ss_ = slot_stats_summary(itf)
                    if ss_:
                        log(f"    slots (train reads, last 25 steps): share {ss_['share']:.4f} entropy {ss_['entropy']:.3f}")
                sw_ = [x_ for x_ in hist["steps"][-25:] if x_.get("swap_n")]
                if sw_:
                    nsw = sum(x_["swap_n"] for x_ in sw_)
                    log(f"    contrast ({nsw} items): CE_swap − CE_own {sum(x_['d_sw'] for x_ in sw_) / nsw:+.4f} | "
                        f"CE_base − CE_own {sum(x_['d_b'] for x_ in sw_) / nsw:+.4f}")
                log(f"  [cached teacher {n_tc}] [step {step}/{a.steps}] (avg last {len(w)}) loss {avg['loss']:.4f} kl {avg['kl']:.4f} "
                    f"ce {avg['ce']:.4f} dist {avg['dist']:.4f} |g| {r['grad_norm']:.3f} gates "
                    f"{[round(float(x), 3) for x in itf.g]} ({(time.time() - t0) / step:.2f}s/step)")
        if step % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if step % a.eval_every == 0 or step == a.steps:
            run_eval(step)
            dump()
        if a.save_every and step % a.save_every == 0 and H is None:       # D12b: checkpoint sweep
            sync_g()
            save_d13(itf, os.path.join(a.out, f"ckpt_{step}.pt"), {"stage": "register", "step": step, "args": vars(a)})
            log(f"  saved {a.out}/ckpt_{step}.pt")
    sync_g()
    if H is not None and a.h_kind == "slots":
        from xenc.d12b_slots import save_slots
        save_slots(H, os.path.join(a.out, "h_last.pt"), {"step": a.steps, "args": vars(a), "base": a.init})
    elif H is not None:
        from xenc.d24_hyper import save_h
        save_h(H, os.path.join(a.out, "h_last.pt"), {"step": a.steps, "args": vars(a), "base": a.init})
    else:
        save_d13(itf, os.path.join(a.out, "d13_last.pt"), {"stage": "register", "step": a.steps, "args": vars(a)})
    dump()
    from fit_d12p import peak_mem
    peak_mem("stage 2")
    log(f"distillation term dropped (--dist_max_cols {a.dist_max_cols}) on {n_nodist} long draws")
    log(f"done: {a.steps} steps, skipped {skipped}, best val crit {best:.4f} -> {best_path} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
