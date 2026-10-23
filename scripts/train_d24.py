"""D24: train the amortized-registration hypernetwork H (xenc/d24_hyper.py) on TRAINING documents only.

Student = the 9B reader reading the 4B encoder's operator through the frozen D13 base interface + H(operator) (the
per-document adapter, one forward of H). Objective = the D13 stage-2 objective (register_d13.py / register_d15.py):
    loss = kl·KL(p_text ‖ p_student)[answer tokens] + ce·CE(target) + dist·(KL(p_enc-attn ‖ p_read) + relMSE)
         + h_l2·mean‖M(doc)‖²   [+ base_anchor·‖θ_base − θ_base0‖²/‖θ_base0‖² when --base_lr > 0]
target = the gold answer (split-train items) or, for --pool_p pool items (D15 label-free self-questions), the 9B
text reader's own greedy answer (pseudo-label, confidence ≥ --pool_conf). The teacher reads the text at TRAINING time
only; at test time nothing but the 4B prefill + H forward runs before the 9B reads.
Selection on HELD-OUT documents (D13's val set: QuALITY/MuSiQue/LastingBench val docs + LongHealth --val_fold), greedy
kv with H every --eval_every steps; step 0 == the base (H heads are zero-initialised).

    python scripts/train_d24.py --init outputs/D13_4b_reg_w10/d13_best.pt --encoder Qwen/Qwen3.5-4B \
        --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --mem_safe 1 --out outputs/D24_h1
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


class HRun:
    """Thin wrapper: rt (D13Rt) + H; sets the document adapter from the operator of the document being READ."""

    def __init__(self, rt, H):
        from xenc.d24_hyper import set_doc
        self.rt, self.H, self.set_doc = rt, H, set_doc

    def adapter(self, did, grad=False):
        op = self.rt.operator(did)[0]
        with torch.set_grad_enabled(grad):
            return self.H(op)

    def gen(self, did, it, budget, read=True, use_h=True, adapter_from=None):
        """adapter_from: take the adapter from ANOTHER document's operator (diagnostic kv_Hswap: own operator read
        through a mismatched H-adapter; isolates what the document conditioning of H contributes)."""
        itf = self.rt.itf
        try:
            if read and use_h:
                self.set_doc(itf, self.adapter(adapter_from or did))
            return self.rt.gen(did, it, budget, read=read)
        finally:
            self.set_doc(itf, None)

    def s1_pass(self, did, it, use_h=True, **kw):
        itf = self.rt.itf
        try:
            if use_h:
                self.set_doc(itf, self.adapter(did))
            return self.rt.s1_pass(did, it, **kw)
        finally:
            self.set_doc(itf, None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--init", default="outputs/D13_4b_reg_w10/d13_best.pt", help="base D13 interface (frozen)")
    ap.add_argument("--train", default="quality:150,musique:150,lastingbench:150,longhealth:0")
    ap.add_argument("--train_from_val", default="", help="e.g. quality:10 — val docs AFTER the first 10 (sorted; the "
                    "first 10 are the held-out val docs) are added to training")
    ap.add_argument("--q_holdout", type=float, default=0.0, help="DIAGNOSTIC (in-sample headroom): hold out this "
                    "fraction of the questions of every --train_from_val document from training; after training, kv_base / "
                    "kv_H / kv_Hswap are judged on those held-out questions of SEEN documents (inq_eval.json)")
    ap.add_argument("--task_p", default="quality:0.4,longhealth:0.2,musique:0.2,lastingbench:0.2")
    ap.add_argument("--val", default="quality:10,musique:20,lastingbench:20,longhealth:0")
    ap.add_argument("--val_q_per_doc", type=int, default=4)
    ap.add_argument("--val_seed", type=int, default=1)
    ap.add_argument("--val_fold", type=int, default=1)
    ap.add_argument("--test_fold", type=int, default=0)
    ap.add_argument("--n_val_attn", type=int, default=30)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--ce", type=float, default=0.5)
    ap.add_argument("--dist_w", type=float, default=1.0)
    ap.add_argument("--rel_w", type=float, default=1.0)
    # H
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--n_q", type=int, default=8)
    ap.add_argument("--d_proj", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--bias_k", type=int, default=0)
    ap.add_argument("--db_rank", type=int, default=0, help=">0: H also writes a rank-r per-document delta on B")
    ap.add_argument("--arch", default="v1", choices=("v1", "v2"), help="v2: standardised summary, LN trunk, separate "
                    "global adapter part (v1 collapsed to a document-independent output)")
    ap.add_argument("--stat_docs", type=int, default=128, help="v2: training docs used to initialise the feature stats")
    ap.add_argument("--h_gate", type=int, default=1)
    ap.add_argument("--input_mode", default="op", choices=("op", "zero"))
    ap.add_argument("--u_scale", type=float, default=1.0)
    ap.add_argument("--lr_h", type=float, default=2e-4, help="H network (summary queries, proj, trunk, heads)")
    ap.add_argument("--lr_basis", type=float, default=1e-4, help="shared bases U, V (Ub)")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--h_l2", type=float, default=0.0)
    ap.add_argument("--h_init", default="", help="warm start H from a d24 file")
    # optional light co-training of the base
    ap.add_argument("--base_lr", type=float, default=0.0, help=">0: also train the base A/MLP/gates (anchored)")
    ap.add_argument("--base_anchor", type=float, default=1.0)
    ap.add_argument("--exclude_sets", default="", help="D22 set jsons: their judged/reg/wrong-map docs never enter "
                    "train or val (hard assertion)")
    ap.add_argument("--exclude_docs", default="", help="json {task: [doc_id]} or [doc_id] of extra excluded docs")
    ap.add_argument("--doc_len", default="", help="register_d13 doc_len filter, e.g. clipper:32000:96000")
    ap.add_argument("--group", type=int, default=0, help=">0: DOCUMENT-GROUPED steps: each step draws --batch documents "
                    "and up to G items of each; the 4B operator and the 9B text-teacher prefill are computed ONCE per "
                    "document (teacher = 9B over doc prefix + question suffix, as register_d20.targets). 0 = old path")
    ap.add_argument("--val_swap", type=int, default=0, help="1: val also reads the own operator through the WRONG "
                    "document's H-adapter (kv_Hswap) and logs kv − Hswap")
    ap.add_argument("--save_val_items", type=int, default=0, help="1: store per-item val scores (by key) in history.json")
    ap.add_argument("--select_min_step", type=int, default=0, help="val points before this step only set the bar "
                    "(never selected); later points must BEAT it to be saved as d24_best.pt")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enc_device", default="", help="put the 4B encoder (operator prefill + encoder teacher) on this "
                    "device, e.g. --device cuda:1 --enc_device cuda:0 on 2 GPUs (as register_d13)")
    ap.add_argument("--lru", type=int, default=3, help="operators kept in memory (long documents: 1-2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem_safe", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pool", default="data/D15_selfq")
    ap.add_argument("--pool_p", type=float, default=0.0)
    ap.add_argument("--pool_docs", type=int, default=1000)
    ap.add_argument("--pool_conf", type=float, default=0.8)
    a = ap.parse_args(argv)
    if a.smoke:
        a.train = ",".join(f"{s.split(':')[0]}:2" for s in a.train.split(","))
        a.val = ",".join(f"{s.split(':')[0]}:1" for s in a.val.split(","))
        a.steps, a.eval_every, a.val_q_per_doc, a.n_val_attn = 6, 3, 1, 2
        a.pool_docs = min(a.pool_docs, 4)
    from eval_doc import wrong_doc_id
    from eval_d13 import summarize
    from fit_d13 import D13Rt, build, fmt, mean_terms, select_items
    from register_state import greedy, kept_logits
    from xenc.backbone import build_chat, text_ids
    from xenc.d13_read import save_d13
    from xenc.d24_hyper import D24Hyper, attach, load_h, save_h, set_doc
    from xenc.doc_tasks import ctx_prompt, gold_text, score
    from xenc.losses import answer_ce, kl_teacher_student
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    reader, enc, tok, itf, lmap = build(a, init=a.init)
    if itf.delta_rank:
        itf.merge_delta()
    itf.requires_grad_(False)
    rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens, lru=a.lru)
    rt.ckpt = rt.R.ckpt = bool(a.mem_safe)
    log(f"devices: reader {rt.dev} encoder {rt.edev} | init {a.init} | B {a.B} | layers {lmap} | max_tokens {a.max_tokens} "
        f"lru {a.lru}")
    if a.h_init:
        H, _ = load_h(a.h_init, itf)
    else:
        H = D24Hyper(itf, rank=a.rank, n_q=a.n_q, d_proj=a.d_proj, hidden=a.hidden, bias_k=a.bias_k,
                     gate=bool(a.h_gate), input_mode=a.input_mode, seed=a.seed, u_scale=a.u_scale,
                     db_rank=a.db_rank, arch=a.arch)
    H = H.to(rt.dev)
    attach(itf, H)
    hr = HRun(rt, H)

    # ---------------- data (training documents only; LongHealth fold test_fold / val_fold excluded)
    tr_folds = set(range(5)) - {a.test_fold, a.val_fold}
    train, d1 = select_items(a.train, "train", tr_folds, 0, a.seed)
    val, d2 = select_items(a.val, "val", {a.val_fold}, a.val_q_per_doc, a.val_seed)
    inq: list = []
    for spec in [x for x in a.train_from_val.split(",") if x]:
        name, skip = spec.split(":")
        assert name != "longhealth", "LongHealth val docs are a fold; use --val_fold"
        ex, d3 = select_items(f"{name}:0", "val", None, 0, a.seed)
        keep = set(sorted({_get(i, "doc_id") for i in ex})[int(skip):])
        if a.q_holdout > 0:
            hrng = random.Random(4321)
            for d in sorted(keep):
                qs = [i for i in ex if _get(i, "doc_id") == d]
                hrng.shuffle(qs)
                nh = int(round(a.q_holdout * len(qs)))
                inq += qs[:nh]
                train += qs[nh:]
        else:
            train += [i for i in ex if _get(i, "doc_id") in keep]
        d1.update({k: v for k, v in d3.items() if k in keep})
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
        log(f"exclude_sets/exclude_docs ({len(drop)} docs) + doc_len '{a.doc_len}': train {n0[0]} -> {len(train)} items, "
            f"val {n0[1]} -> {len(val)}")
        assert not ({_get(i, "doc_id") for i in train} & drop) and not ({_get(i, "doc_id") for i in val} & drop), \
            "an excluded (judged) document reached train/val"
        tasks_ = {s.split(":")[0] for s in (a.train + "," + a.val).split(",") if s} - {"longhealth"}
        for name in sorted(tasks_):                     # hard guard: no TEST-split document of any task trains / validates
            exT, _ = select_items(f"{name}:0", "test", None, 0, 0)
            tdocs = {_get(i, "doc_id") for i in exT}
            bad = tdocs & ({_get(i, "doc_id") for i in train} | {_get(i, "doc_id") for i in val})
            assert not bad, f"TEST DOC LEAK ({name}): {sorted(bad)[:3]}"
        log(f"leakage guard: no test-split doc of {sorted(tasks_)} and no excluded doc in train/val")
    if a.smoke:
        inq = inq[:6]
    rt.docs = {**d1, **d2}
    tr_docs, va_docs = {_get(i, "doc_id") for i in train}, {_get(i, "doc_id") for i in val}
    assert not (tr_docs & va_docs), "val docs overlap train docs"
    for spl in ("test",):                                  # hard guard: no judged test document is a training doc
        for name in ("quality", "longhealth"):
            ex, _ = select_items(f"{name}:0", spl, None, 0, 0)
            tdocs = {_get(i, "doc_id") for i in ex if name != "longhealth" or (_get(i, "meta") or {}).get("fold") == a.test_fold}
            assert not (tdocs & tr_docs), f"TEST DOC LEAK ({name})"
    lh_test = {_get(i, "doc_id") for i in select_items("longhealth:0", "train", {a.test_fold}, 0, 0)[0]} | \
        {_get(i, "doc_id") for i in select_items("longhealth:0", "val", {a.test_fold}, 0, 0)[0]}
    assert not (lh_test & tr_docs), "LongHealth test-fold doc in training"
    if getattr(H, "arch", "v1") == "v2" and not a.h_init:
        sdocs = sorted(tr_docs)
        random.Random(99).shuffle(sdocs)
        fs_ = []
        for d in sdocs:
            if len(fs_) >= a.stat_docs:
                break
            try:
                with torch.no_grad():
                    fs_.append({e: v.detach() for e, v in H.summary(rt.operator(d)[0]).items()})
            except torch.OutOfMemoryError:
                log(f"  OOM on the stats prefill of {d}; skipped")
            finally:
                rt._ops.pop(d, None)                     # never keep long operators around
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        H.init_stats_feats(fs_)
        del fs_
        log(f"v2 feature statistics initialised on {min(a.stat_docs, len(sdocs))} training documents")
    pools: dict = {}
    for it in train:
        pools.setdefault(_get(it, "doc_id").split("_")[0], []).append(it)
    tp = {k: float(v) for k, v in (s.split(":") for s in a.task_p.split(",")) if k in pools}
    tasks, weights = list(tp), [tp[t] for t in tp]
    pool_items = []
    if a.pool_p > 0:
        from fit_d15 import load_pool
        from xenc.doc_tasks import _doc_path
        by, held_pool, pidx = load_pool(a.pool, use_selfq=True, use_native=False)
        pd_ = sorted(d for d in by if d not in held_pool and d not in va_docs and d.startswith("d15_"))
        random.Random(1234).shuffle(pd_)
        pd_ = pd_[: a.pool_docs]
        for d in pd_:
            rt.docs[d] = _doc_path(Path(a.pool), pidx[d]["path"]).read_text()
            pool_items += by[d]
        log(f"pool items (self-questions, reader-with-text targets): {len(pool_items)} from {len(pd_)} docs")
    ans_cache: dict = {}
    pool_stats = {"seen": 0, "low_conf": 0, "used": 0}
    val_order = {}
    for d in sorted(va_docs):
        val_order.setdefault(d.split("_")[0], []).append(d)
    B0 = {li: itf.B[str(li)].detach().clone() for li in lmap}

    # ---------------- optimiser
    basis = list(H.U.parameters()) + list(H.V.parameters()) + list(H.Ub.parameters()) + \
        list(H.UB.parameters()) + list(H.VB.parameters())
    bid = {id(p) for p in basis}
    hnet = [p for p in H.parameters() if id(p) not in bid]
    groups = [{"params": hnet, "lr": a.lr_h}, {"params": basis, "lr": a.lr_basis}]
    base_ps, base0 = [], {}
    if a.base_lr > 0:
        base_ps = list(itf.A.parameters()) + list(itf.M.parameters()) + [itf.g]
        for p in base_ps:
            p.requires_grad_(True)
        base0 = {id(p): p.detach().clone() for p in base_ps}
        groups.append({"params": base_ps, "lr": a.base_lr})
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) if a.warmup > 0 else 1.0)
    log(f"D24 H: {a.encoder} -> {a.reader} | layers {lmap} | base {a.init} (frozen{'' if not a.base_lr else ', co-trained'}) "
        f"| H {H.n_params() / 1e6:.2f}M params ({sum(p.numel() for p in basis) / 1e6:.2f}M bases), cfg {H.cfg} | "
        f"adapter {H.adapter_numel()} numbers/doc | train {len(train)} items / {len(tr_docs)} docs "
        f"{ {t: len(v) for t, v in pools.items()} } | held-out val {len(val)} items / {len(va_docs)} docs | "
        f"kl {a.kl} ce {a.ce} dist {a.dist_w} pool_p {a.pool_p}")
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    hist = {"steps": [], "evals": []}
    best, t0 = -1e9, time.time()
    best_path = os.path.join(a.out, "d24_best.pt")
    ct_cache: dict = {}

    def run_eval(step):
        nonlocal best
        rows = []
        for it in val:
            did = _get(it, "doc_id")
            w = wrong_doc_id(did, val_order[did.split("_")[0]])
            rec = {}
            if a.save_val_items:
                rec["key"] = _get(it, "key")
            if a.val_swap and w is not None:
                g, n = hr.gen(did, it, a.budget, adapter_from=w)
                rec["kv_Hswap"] = {"score": float(score(it, g)), "gen_tokens": n}
            for c, d, rd in (("kv", did, True), ("wrong_doc", w, True), ("closed_tail", did, False)):
                if c == "closed_tail" and _get(it, "key") in ct_cache:
                    rec[c] = ct_cache[_get(it, "key")]
                    continue
                if d is None:
                    continue
                g, n = hr.gen(d, it, a.budget, read=rd)
                rec[c] = {"score": float(score(it, g)), "gen_tokens": n}
                if c == "closed_tail":
                    ct_cache[_get(it, "key")] = rec[c]
            rows.append(rec)
        s = summarize(rows, ("closed_tail", "kv", "wrong_doc"))
        att = mean_terms([hr.s1_pass(_get(it, "doc_id"), it, onpolicy=True, grad=False, metrics=True)
                          for it in val[:: max(1, len(val) // max(a.n_val_attn, 1))][: a.n_val_attn]])
        with torch.no_grad():
            Ms = [float(hr.adapter(d)["_norm"]) for d in sorted(va_docs)[:12]]
        r = {"step": step, "kv": s["kv"]["score"], "wrong_doc": s["wrong_doc"]["score"],
             "closed_tail": s["closed_tail"]["score"],
             "gap": s.get("kv_minus_wrong_doc", {}).get("paired_gap", float("nan")),
             "gap_se": s.get("kv_minus_wrong_doc", {}).get("se", float("nan")), "gen": s["kv"]["mean_gen_tokens"],
             "gen_wrong": s["wrong_doc"]["mean_gen_tokens"], "attn": att, "M_norm": sum(Ms) / len(Ms), "n": len(rows)}
        per_task = {}
        for it, rec in zip(val, rows):
            per_task.setdefault(_get(it, "doc_id").split("_")[0], []).append(rec)
        r["per_task"] = {t: {c: sum(x[c]["score"] for x in v if c in x) / len(v) for c in ("kv", "wrong_doc", "closed_tail")}
                         for t, v in per_task.items()}
        if a.val_swap:
            pr = [x["kv"]["score"] - x["kv_Hswap"]["score"] for x in rows if "kv_Hswap" in x and "kv" in x]
            mu = sum(pr) / max(len(pr), 1)
            sd = (sum((q - mu) ** 2 for q in pr) / max(len(pr) - 1, 1)) ** 0.5
            r["kv_Hswap"], r["kv_minus_Hswap"], r["kv_minus_Hswap_se"] = (
                sum(x["kv_Hswap"]["score"] for x in rows if "kv_Hswap" in x) / max(len(pr), 1), mu, sd / max(len(pr), 1) ** 0.5)
            for t, v in r["per_task"].items():
                v["kv_Hswap"] = sum(x["kv_Hswap"]["score"] for x in per_task[t] if "kv_Hswap" in x) / len(per_task[t])
            log(f"            kv_Hswap {r['kv_Hswap']:.3f} | kv − Hswap {mu:+.3f} ± {r['kv_minus_Hswap_se']:.3f} | per task "
                + " ".join(f"{t}: {v['kv']:.3f}/{v['kv_Hswap']:.3f}" for t, v in r["per_task"].items()))
        if a.save_val_items:                        # per-item scores: paired H vs control / own vs swap across runs
            r["items"] = [{"key": x["key"], **{c: x[c]["score"] for c in ("kv", "kv_Hswap", "wrong_doc", "closed_tail")
                                               if c in x}} for x in rows]
        hist["evals"].append(r)
        crit = r["kv"] + (1e-3 * r["gap"] if math.isfinite(r["gap"]) else 0.0)
        tag = ""
        if step < a.select_min_step:              # sets the bar only (never selected)
            best = max(best, crit)
            crit = -1e9
        if crit > best:
            best = crit
            save_h(H, best_path, {"step": step, "val": r, "args": vars(a), "base": a.init})
            if base_ps:
                save_d13(itf, os.path.join(a.out, "d13_base_best.pt"), {"stage": "d24 co-trained base", "step": step})
            tag = " *best*"
        log(f"  [val @ {step}] held-out docs n={r['n']}: kv {r['kv']:.3f} wrong {r['wrong_doc']:.3f} closed_tail "
            f"{r['closed_tail']:.3f} gap {r['gap']:+.3f} ± {r['gap_se']:.3f} gen {r['gen']:.1f}/{r['gen_wrong']:.1f} "
            f"|M|² {r['M_norm']:.2e}{tag}")
        log(f"            per task " + " ".join(f"{t}: {v['kv']:.2f}/{v['wrong_doc']:.2f}/{v['closed_tail']:.2f}"
                                             for t, v in r["per_task"].items()) + f" | attn {fmt(att)}")

    def dump():
        with open(os.path.join(a.out, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)

    by_doc: dict = {}
    for it in train:
        by_doc.setdefault(_get(it, "doc_id"), []).append(it)

    def group_step(acc) -> int:
        """--group: one document, up to a.group of its items; operator + 9B teacher prefill once (see --group)."""
        import copy as _copy
        nskip = 0
        t = rng.choices(tasks, weights)[0]
        did = _get(pools[t][rng.randrange(len(pools[t]))], "doc_id")
        its = list(by_doc[did])
        rng.shuffle(its)
        its = its[: a.group]
        R.mode = "off"
        cache = None
        try:
            if a.kl > 0:
                with torch.no_grad():
                    out = reader(input_ids=torch.tensor([rt.pre(did)], device=rt.dev), use_cache=True, logits_to_keep=1)
                    cache = out.past_key_values
                    del out
            op, n_op, _ = rt.operator(did)
            for it in its:
                tt = text_ids(tok, gold_text(it))
                m = len(tt)
                rows = rt.rows(did, it, tt)
                t_log = None
                if cache is not None:
                    with torch.no_grad():
                        t_log = kept_logits(reader, torch.tensor([rt.suf(it) + tt]), m + 1, cache=_copy.deepcopy(cache)).detach()
                rec = rt.teacher(did, rows) if a.dist_w > 0 else None
                R.load(op, n_op)
                try:
                    with torch.enable_grad():
                        ad = H(op)
                        set_doc(itf, ad)
                        R.mode, R.need_logits, R.rec = "e2e", a.dist_w > 0, {}
                        s_log = kept_logits(reader, torch.tensor([rows]), m + 1)
                        kl = kl_teacher_student(t_log[:-1], s_log[:-1]) if t_log is not None else torch.zeros((), device=s_log.device)
                        ce = answer_ce(s_log, rows[-m:])
                        loss = a.kl * kl + a.ce * ce
                        dist = torch.zeros((), device=s_log.device)
                        if a.dist_w > 0:
                            with torch.no_grad():
                                yref = {li: (rec[e]["out"].float() @ B0[li].t()) for li, e in lmap.items()}
                            terms = rt.attn_terms(rec, op, n_op, yref)
                            dist = torch.stack([x["kl"] + a.rel_w * x["rel"] for x in terms.values()]).mean()
                            loss = loss + a.dist_w * dist.to(loss.device)
                        if a.h_l2 > 0:
                            loss = loss + a.h_l2 * ad["_norm"].to(loss.device)
                        if not math.isfinite(float(loss)):
                            nskip += 1
                            continue
                        (loss / (a.batch * len(its))).backward()
                    for k, v in (("loss", loss), ("kl", kl), ("ce", ce), ("dist", dist), ("hn", ad["_norm"])):
                        acc[k] += float(v)
                    acc["n"] += 1
                except torch.OutOfMemoryError:
                    nskip += 1
                    log(f"  OOM on {did} ({len(rows)} rows, {n_op} operator tokens); item skipped")
                finally:
                    R.mode, R.rec = "off", {}
                    set_doc(itf, None)
                    s_log = loss = kl = ce = dist = rec = terms = ad = None
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except torch.OutOfMemoryError:
            nskip += 1
            log(f"  OOM in the document pass of {did}; document skipped")
        finally:
            cache = None
            rt._ops.pop(did, None) if a.group and len(rt.pre(did)) > 32000 else None   # free long operators
        return nskip

    run_eval(0)
    dump()
    R = rt.R
    skipped = 0
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        acc = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "dist": 0.0, "hn": 0.0, "n": 0}
        if a.group > 0:
            for _ in range(a.batch):
                skipped += group_step(acc)
        for _ in range(0 if a.group > 0 else a.batch):
            is_pool = False
            if pool_items and rng.random() < a.pool_p:
                is_pool = True
                it = pool_items[rng.randrange(len(pool_items))]
                did = _get(it, "doc_id")
                if it["key"] not in ans_cache:
                    R.mode = "off"
                    g, _ = greedy(reader, tok, a.reader, build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, rt.docs[did])), 24)
                    ans_cache[it["key"]] = (g[g.index("<answer>"):].split("</answer>")[0] + "</answer>") if "<answer>" in g else None
                target = ans_cache[it["key"]]
                if ans_cache.get(it["key"] + "#conf", 1.0) < a.pool_conf:
                    pool_stats["low_conf"] += 1
                    continue
                if not target:
                    skipped += 1
                    continue
            else:
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
            with torch.no_grad():
                t_log = kept_logits(reader, torch.tensor([tids]), m + 1).detach()
            if is_pool:
                ck = it["key"] + "#conf"
                if ck not in ans_cache:
                    lp = torch.log_softmax(t_log[:-1].float(), -1).gather(-1, torch.tensor(tt, device=t_log.device).view(-1, 1))
                    ans_cache[ck] = float(lp.mean().exp())
                pool_stats["seen"] += 1
                if ans_cache[ck] < a.pool_conf:
                    pool_stats["low_conf"] += 1
                    continue
                pool_stats["used"] += 1
            rows = rt.rows(did, it, tt)
            op, n_op, _ = rt.operator(did)
            rec = rt.teacher(did, rows) if a.dist_w > 0 else None
            R.load(op, n_op)
            try:
                with torch.enable_grad():
                    ad = H(op)                                   # the amortized per-document adapter (with grad)
                    set_doc(itf, ad)
                    R.mode, R.need_logits, R.rec = "e2e", a.dist_w > 0, {}
                    s_log = kept_logits(reader, torch.tensor([rows]), m + 1)
                    kl = kl_teacher_student(t_log[:-1], s_log[:-1])
                    ce = answer_ce(s_log, rows[-m:])
                    loss = a.kl * kl + a.ce * ce
                    dist = torch.zeros((), device=kl.device)
                    if a.dist_w > 0:
                        with torch.no_grad():
                            yref = {li: (rec[e]["out"].float() @ B0[li].t()) for li, e in lmap.items()}
                        terms = rt.attn_terms(rec, op, n_op, yref)
                        dist = torch.stack([x["kl"] + a.rel_w * x["rel"] for x in terms.values()]).mean()
                        loss = loss + a.dist_w * dist.to(loss.device)
                    if a.h_l2 > 0:
                        loss = loss + a.h_l2 * ad["_norm"].to(loss.device)
                    if base_ps and a.base_anchor > 0:
                        anc = sum((p - base0[id(p)]).pow(2).sum() / base0[id(p)].pow(2).sum().clamp_min(1e-6)
                                  for p in base_ps) / len(base_ps)
                        loss = loss + a.base_anchor * anc.to(loss.device)
                    if not math.isfinite(float(loss)):
                        skipped += 1
                        continue
                    (loss / a.batch).backward()
            except torch.OutOfMemoryError:
                R.rec, loss, kl, ce, dist, s_log, rec, ad = {}, None, None, None, None, None, None, None
                torch.cuda.empty_cache()
                skipped += 1
                log(f"  OOM on {did} ({len(rows)} rows, {n_op} operator tokens); item skipped")
                continue
            finally:
                R.mode, R.rec = "off", {}
                set_doc(itf, None)
            for k, v in (("loss", loss), ("kl", kl), ("ce", ce), ("dist", dist), ("hn", ad["_norm"])):
                acc[k] += float(v)
            acc["n"] += 1
        if acc["n"]:
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            r = {k: (v / acc["n"] if k != "n" else v) for k, v in acc.items()}
            r.update(step=step, grad_norm=float(gn), t=round(time.time() - t0, 1))
            hist["steps"].append(r)
            if step % 25 == 0 or step <= 3:
                w = hist["steps"][-25:]
                avg = {k: sum(x[k] for x in w) / len(w) for k in ("loss", "kl", "ce", "dist", "hn")}
                log(f"  [step {step}/{a.steps}] (avg last {len(w)}) loss {avg['loss']:.4f} kl {avg['kl']:.4f} "
                    f"ce {avg['ce']:.4f} dist {avg['dist']:.4f} |M|² {avg['hn']:.2e} |g| {r['grad_norm']:.3f} "
                    f"({(time.time() - t0) / step:.2f}s/step)")
        if step % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if step % a.eval_every == 0 or step == a.steps:
            run_eval(step)
            dump()
    log(f"pool pseudo-label stats: {pool_stats}")
    save_h(H, os.path.join(a.out, "d24_last.pt"), {"step": a.steps, "args": vars(a), "base": a.init})
    if base_ps:
        save_d13(itf, os.path.join(a.out, "d13_base_last.pt"), {"stage": "d24 co-trained base", "step": a.steps})
    dump()
    if torch.cuda.is_available():
        log(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if not os.path.exists(best_path):
        log(f"WARNING: no val point after --select_min_step {a.select_min_step} beat the step-0 bar; NO d24_best.pt written")
    log(f"done: {a.steps} steps, skipped {skipped}, best val crit {best:.4f} -> {best_path} ({time.time() - t0:.0f}s)")
    if inq:
        inq_eval(a, hr, inq, H, best_path, summarize, score, _get, load_h)
    return 0


def inq_eval(a, hr, inq, H, best_path, summarize, score, _get, load_h):
    """In-sample headroom diagnostic: held-out QUESTIONS of documents H was trained on (never judged test docs)."""
    from eval_doc import wrong_doc_id
    order = sorted({_get(i, "doc_id") for i in inq})
    out = {}
    for tag in ("last", "best"):
        if tag == "best":
            Hb, _ = load_h(best_path, hr.rt.itf)
            H.load_state_dict(Hb.state_dict())
        rows = []
        for it in inq:
            did = _get(it, "doc_id")
            w = wrong_doc_id(did, order)
            rec = {"key": _get(it, "key"), "doc_id": did}
            for c, kw in (("kv_base", {"use_h": False}), ("kv", {}), ("kv_Hswap", {"adapter_from": w})):
                g, n = hr.gen(did, it, a.budget, **kw)
                rec[c] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
            rows.append(rec)
        s = summarize(rows, ("kv_base", "kv", "kv_Hswap"))
        for ref in ("kv_base", "kv_Hswap"):
            pair = [r["kv"]["score"] - r[ref]["score"] for r in rows]
            mu = sum(pair) / len(pair)
            sd = (sum((p - mu) ** 2 for p in pair) / max(len(pair) - 1, 1)) ** 0.5
            s[f"kv_minus_{ref}"] = {"paired_gap": mu, "se": sd / len(pair) ** 0.5, "n_paired": len(pair),
                                    "wins": sum(p > 0 for p in pair), "losses": sum(p < 0 for p in pair)}
        out[tag] = {"summary": s, "items": rows}
        log(f"[in-sample held-out questions, H {tag}] n={len(rows)} docs={len(order)}: kv_base {s['kv_base']['score']:.3f} "
            f"kv_H {s['kv']['score']:.3f} kv_Hswap {s['kv_Hswap']['score']:.3f} | kv−base "
            f"{s['kv_minus_kv_base']['paired_gap']:+.3f}±{s['kv_minus_kv_base']['se']:.3f} kv−swap "
            f"{s['kv_minus_kv_Hswap']['paired_gap']:+.3f}±{s['kv_minus_kv_Hswap']['se']:.3f}")
    with open(os.path.join(a.out, "inq_eval.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
