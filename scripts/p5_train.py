"""P5: train a dedicated copy head (xenc/p5_copy.py) on a FROZEN D13 base and judge it on held-out val docs.

The D13 base (A / MLP / B / gates) is frozen, so every val point compares the copy head ON vs OFF on the same frozen
base: kv_off == the base's step-0 kv exactly (checked at --off_check steps), and no plain-continuation control is
needed for the mechanism itself (the continuation controls P3_ctrl / P3_ctrl2000 measure the data effect, which is
additive). Training signal per item (lever-1 data mix, same items / RNG as register_d13.py for the same seed):
    kl·KL(text ‖ P_fused) + ce·CE(P_fused) + copy_w·span-copy NLL + end_w·span-end NLL + bce_w·gate switch BCE
    (+ tok_w·−log P_copy(gold) on content rows whose gold token is in the document)
only the P5 head trains (own Adam groups; the 9B reader forward runs under no_grad with the head capturing h).
Val (held-out docs; --save_items): kv (head on, greedy), wrong_doc (head on), closed_tail, kv_off (head off, same ckpt,
at --off_check steps), kv_span (span-head extraction after a forced "<answer>", non-MCQ items) + pointer diagnostics on
items whose gold answer occurs verbatim (first-row hit / gold mass / span EM).

    python scripts/p5_train.py <register_d13 data args> --init outputs/D13_lever1_pilot/d13_best.pt --steps 2000 --out outputs/P5_mix
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
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--init", required=True)
    ap.add_argument("--train", default="quality:150,musique:150,lastingbench:150,longhealth:0")
    ap.add_argument("--task_p", default="quality:0.4,longhealth:0.2,musique:0.2,lastingbench:0.2")
    ap.add_argument("--val", default="quality:10,musique:20,lastingbench:20,longhealth:0")
    ap.add_argument("--val_q_per_doc", type=int, default=4)
    ap.add_argument("--val_fold", type=int, default=1)
    ap.add_argument("--test_fold", type=int, default=0)
    ap.add_argument("--val_seed", type=int, default=1)
    ap.add_argument("--train_from_val", default="")
    ap.add_argument("--exclude_sets", default="")
    ap.add_argument("--exclude_docs", default="")
    ap.add_argument("--doc_len", default="")
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--ce", type=float, default=0.5)
    ap.add_argument("--copy_w", type=float, default=1.0, help="span-copy NLL (chain-tracked pointer at the strict content rows)")
    ap.add_argument("--end_w", type=float, default=0.5, help="span-end NLL (end query at the first content row)")
    ap.add_argument("--tok_w", type=float, default=0.0, help="-log P_copy(gold token) on content rows with the token in the doc")
    ap.add_argument("--bce_w", type=float, default=0.2, help="gate switch BCE: sigma(z) vs 1[P_copy(gold) > P_reader(gold)]")
    ap.add_argument("--p5_enc_layers", default="19,27")
    ap.add_argument("--p5_heads", type=int, default=4)
    ap.add_argument("--p5_dim", type=int, default=128)
    ap.add_argument("--p5_fuse", default="mix", choices=("mix", "poe"))
    ap.add_argument("--p5_chain", type=int, default=1)
    ap.add_argument("--p5_span_max", type=int, default=48)
    ap.add_argument("--p5_merged", type=int, default=1, help="1: rows fed the 'answer' of '<answer>' copy the merged "
                    "'>'+token variant (BPE '>Bob'; P4's finding) and span supervision accepts a first content token "
                    "merged with the tag; 0 = the no-space map only")
    ap.add_argument("--lr_mat", type=float, default=3e-4, help="query / key projections")
    ap.add_argument("--lr_ptr", type=float, default=1e-2, help="pointer scalars (temperature, head mix, position bias, "
                    "reader-logprob coefficient, chain strength)")
    ap.add_argument("--lr_amp", type=float, default=1e-2, help="gate amplitude a")
    ap.add_argument("--lr_gate", type=float, default=1e-3, help="gate net (gw, gf, gb)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--select_min_step", type=int, default=250)
    ap.add_argument("--off_check", default="250", help="comma steps (and the last) at which val also scores kv with the "
                    "head off on the same checkpoint (must equal step 0: the base is frozen)")
    ap.add_argument("--eval_span", type=int, default=1)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--save_items", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enc_device", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lru", type=int, default=2)
    ap.add_argument("--mem_safe", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.train = ",".join(f"{s.split(':')[0]}:2" for s in a.train.split(","))
        a.val = ",".join(f"{s.split(':')[0]}:1" for s in a.val.split(","))
        a.steps, a.eval_every, a.val_q_per_doc, a.select_min_step, a.off_check = 6, 3, 1, 3, "3"
    from eval_doc import wrong_doc_id
    from eval_d13 import summarize
    from fit_d13 import D13Rt, build, select_items
    from register_d13 import doc_len_filter, excluded_docs
    from register_state import kept_logits
    from xenc.backbone import build_chat, decode, text_ids
    from xenc.d13_read import save_d13
    from xenc.doc_tasks import ctx_prompt, gold_text, score
    from xenc.losses import answer_ce, kl_teacher_student
    from xenc.p4_ptr import template_ids
    from xenc.p5_copy import content_rows_strict, occurrences, parse_p5, space_maps, upgrade_p5
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    reader, enc, tok, base, lmap = build(a, init=a.init)
    for p in base.parameters():
        p.requires_grad_(False)
    V = int(reader.get_output_embeddings().weight.shape[0])
    nosp_t, gtm_t, gt_id, ans_id = space_maps(tok, V)
    assert text_ids(tok, "<answer>")[-2:] == [ans_id, gt_id], "P5: '<answer>' does not end in 'answer' '>' tokens"
    cfg = parse_p5(a.p5_enc_layers, a.p5_heads, a.p5_dim, a.p5_fuse, a.p5_chain,
                   template_ids(tok, a.reader, SYSTEM_ANSWER), gt_id, a.p5_span_max, seed=a.seed, ans_id=ans_id,
                   merged=a.p5_merged)
    itf = upgrade_p5(base, enc, cfg)
    itf.set_nosp(nosp_t, gtm_t)
    for n_, p in itf.named_parameters():
        p.requires_grad_(n_.startswith("P5."))
    rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens, lru=a.lru)
    R = rt.R
    cfg_s = dict(cfg, prefix=f"<{len(cfg['prefix'])} ids>")
    log(f"P5: init {a.init} | B {a.B} | layer map {lmap} | p5 {cfg_s} | "
        f"no-space remaps {int((itf.p5_nosp != torch.arange(V, device=itf.p5_nosp.device)).sum())} / merged '>' variants "
        f"{int((itf.p5_gtm != gt_id).sum())} of {V} | GPU "
        f"{torch.cuda.get_device_name(rt.dev) if torch.cuda.is_available() else 'cpu'}")
    # -------------------------------------------------- data (identical selection to register_d13.py)
    tr_folds = set(range(5)) - {a.test_fold, a.val_fold}
    train, d1 = select_items(a.train, "train", tr_folds, 0, a.seed)
    val, d2 = select_items(a.val, "val", {a.val_fold}, a.val_q_per_doc, a.val_seed)
    for spec in [x for x in a.train_from_val.split(",") if x]:
        name, skip = spec.split(":")
        assert name != "longhealth"
        ex, d3 = select_items(f"{name}:0", "val", None, 0, a.seed)
        keep = set(sorted({_get(i, "doc_id") for i in ex})[int(skip):])
        train += [i for i in ex if _get(i, "doc_id") in keep]
        d1.update({k: v for k, v in d3.items() if k in keep})
    drop = excluded_docs(a.exclude_sets)
    if a.exclude_docs:
        X = json.load(open(a.exclude_docs))
        drop |= set(sum(X.values(), [])) if isinstance(X, dict) else set(X)
    keep_len = doc_len_filter(a.doc_len)
    n0 = (len(train), len(val))
    train = [i for i in train if _get(i, "doc_id") not in drop and keep_len(_get(i, "doc_id"))]
    val = [i for i in val if _get(i, "doc_id") not in drop and keep_len(_get(i, "doc_id"))]
    rt.docs = {**d1, **d2}
    tr_docs, va_docs = {_get(i, "doc_id") for i in train}, {_get(i, "doc_id") for i in val}
    assert not (tr_docs & drop) and not (va_docs & drop), "an excluded (judged) document reached train/val"
    assert not (tr_docs & va_docs), "val docs overlap train docs"
    if a.exclude_sets:                                   # HARD: every D22 judged / reg / wrong-map doc is out
        assert drop, "--exclude_sets given but nothing excluded"
    pools: dict = {}
    for it in train:
        pools.setdefault(_get(it, "doc_id").split("_")[0], []).append(it)
    tp = {k: float(v) for k, v in (s.split(":") for s in a.task_p.split(",")) if k in pools}
    tasks, weights = list(tp), [tp[t] for t in tp]
    val_order: dict = {}
    for d in sorted(va_docs):
        val_order.setdefault(d.split("_")[0], []).append(d)
    log(f"  exclude_sets/docs ({len(drop)} docs) / doc_len: train {n0[0]} -> {len(train)} items, val {n0[1]} -> {len(val)} | "
        f"train {len(tr_docs)} docs {{{', '.join(f'{t}: {len(v)}' for t, v in pools.items())}}} | val {len(val)} items / "
        f"{len(va_docs)} docs | leakage assertion passed")
    # -------------------------------------------------- optimiser (P5 head only)
    pp = itf.p5_params()
    groups = [{"params": pp["mat"], "lr": a.lr_mat}, {"params": pp["ptr"], "lr": a.lr_ptr},
              {"params": pp["amp"], "lr": a.lr_amp}, {"params": pp["gate"], "lr": a.lr_gate}]
    plist = [p for g in groups for p in g["params"]]
    assert {id(p) for p in plist} == {id(p) for p in itf.parameters() if p.requires_grad}, "P5 param groups incomplete"
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    for g_ in opt.param_groups:
        g_["base_lr"] = g_["lr"]
    log(f"  trainable: P5 head only, {sum(p.numel() for p in plist) / 1e6:.2f}M params (base frozen: "
        f"{sum(p.numel() for n_, p in itf.named_parameters() if not n_.startswith('P5.')) / 1e6:.1f}M) | lr mat {a.lr_mat} "
        f"ptr {a.lr_ptr} amp {a.lr_amp} gate {a.lr_gate} warmup {a.warmup} | losses kl {a.kl} ce {a.ce} copy {a.copy_w} "
        f"end {a.end_w} tok {a.tok_w} bce {a.bce_w} | state {itf.p5_state()}")
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    open_ids = text_ids(tok, "<answer>")

    def gold_contents(it) -> list:
        """Strict content-token sequences of every gold answer (as the teacher-forced target would tokenize it)."""
        out = []
        for ans in (_get(it, "answers") or []):
            ids_, rows_ = content_rows_strict(tok, f"<answer>{ans}</answer>", allow_merged=bool(a.p5_merged))
            if rows_:
                out.append([ids_[i] for i in rows_])
        return out

    off_steps = {int(x) for x in a.off_check.split(",") if x} | {a.steps}
    hist = {"steps": [], "evals": []}
    best, t0 = -1e9, time.time()
    best_path = os.path.join(a.out, "d13_best.pt")
    ct_cache: dict = {}

    @torch.no_grad()
    def span_eval(it):
        """Span head after a forced '<answer>': (score, logconf, diag) ; diag on items with a verbatim gold occurrence."""
        did = _get(it, "doc_id")
        op, n_op, _ = rt.operator(did)
        R.load(op, n_op)
        R.mode, R.p5_capture = "student", True
        try:
            lg = kept_logits(reader, torch.tensor([rt.rows(did, it) + open_ids]), 1)
            h = R.p5_h[-1]
        finally:
            R.mode, R.p5_capture, R.past = "off", False, {}
        tab = R.p5_tab
        keys = R.keys()
        s, e, conf, ps, pe = itf.p5_span(h, lg[-1], keys, tab)
        sp = tab["ids"][s:e + 1].tolist()
        sp[0] = int(tab["nosp"][s])
        txt = "<answer>" + decode(tok, a.reader, sp).strip() + "</answer>"
        d = None
        occ = [(occurrences(c, tab), len(c)) for c in gold_contents(it)]
        occ = [(o, L) for o, L in occ if o.numel()]
        if occ:
            st = torch.cat([o for o, _ in occ])
            hit = any(int(o_) - 3 <= s <= int(o_) + L - 1 + 3 for o, L in occ for o_ in o)
            em = any(int(o_) == s and int(o_) + L - 1 == e for o, L in occ for o_ in o)
            d = {"hit": float(hit), "mass": float(ps[st].sum()), "em": float(em),
                 "end_mass": float(sum(pe[(o + L - 1).clamp_max(pe.shape[0] - 1)].sum() for o, L in occ))}
        return float(score(it, txt)), conf, d, txt

    def run_eval(step):
        nonlocal best
        rows = []
        R.p5_gsum, R.p5_gn = 0.0, 0
        itf._p5_ver += 1
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
                if c == "kv":
                    rec["kv_txt"] = g
                if c == "closed_tail":
                    ct_cache[_get(it, "key")] = rec[c]
            rows.append(rec)
        g_val = R.p5_gsum / max(R.p5_gn, 1)
        s_ = summarize(rows, ("closed_tail", "kv", "wrong_doc"))
        r = {"step": step, "kv": s_["kv"]["score"], "wrong_doc": s_["wrong_doc"]["score"],
             "closed_tail": s_["closed_tail"]["score"],
             "gap": s_.get("kv_minus_wrong_doc", {}).get("paired_gap", float("nan")),
             "gap_se": s_.get("kv_minus_wrong_doc", {}).get("se", float("nan")), "gen": s_["kv"]["mean_gen_tokens"],
             "n": len(rows), "g_val": round(g_val, 5), "p5": itf.p5_state()}
        if step in off_steps and step > 0:              # same checkpoint, head off (== base step 0: frozen base)
            itf.p5_on = False
            try:
                for it, rec in zip(val, rows):
                    g, n = rt.gen(_get(it, "doc_id"), it, a.budget, read=True)
                    rec["kv_off"] = {"score": float(score(it, g)), "gen_tokens": n}
            finally:
                itf.p5_on = True
            r["kv_off"] = sum(x["kv_off"]["score"] for x in rows) / len(rows)
            if hist["evals"]:
                z = hist["evals"][0]["items"]
                r["kv_off_eq_step0"] = sum(abs(x["kv_off"]["score"] - zz[1]) < 1e-9 for x, zz in zip(rows, z)) / len(rows)
        if step == 0:                                   # first-content-token coverage of the copy tables (val, non-MCQ)
            cov = {"n": 0, "plain": 0, "mapped": 0, "occ": 0, "later_in_doc": 0, "later_n": 0}
            for it in val:
                if _get(it, "choices"):
                    continue
                op, n_op, _ = rt.operator(_get(it, "doc_id"))
                tab = itf.p5_tables(op, n_op)
                tgt = gold_text(it)
                tt_, rows_ = content_rows_strict(tok, tgt, allow_merged=bool(a.p5_merged))
                cov["n"] += 1
                if not rows_:
                    continue
                c0 = tt_[rows_[0]]
                cov["plain"] += int(bool((tab["ids"] == c0).any()))
                x_ = torch.tensor([tt_[rows_[0] - 1]], device=tab["ids"].device)
                ti_, _ = itf.tok_idx(x_, tab)
                cov["mapped"] += int(bool((ti_[0] == c0).any()))
                cov["occ"] += int(occurrences([tt_[i] for i in rows_], tab).numel() > 0)
                for i in rows_[1:]:
                    cov["later_n"] += 1
                    cov["later_in_doc"] += int(bool((tab["ids"] == tt_[i]).any()))
            r["coverage"] = {k: (round(v / max(cov["n"], 1), 4) if k in ("plain", "mapped", "occ") else v)
                             for k, v in cov.items()}
            r["coverage"]["later_in_doc"] = round(cov["later_in_doc"] / max(cov["later_n"], 1), 4)
            log(f"  [coverage @ 0] val non-MCQ items {cov['n']}: first content token in doc ids plain "
                f"{r['coverage']['plain']:.3f} / with the row's variant map {r['coverage']['mapped']:.3f} | full verbatim "
                f"occurrence (span-supervisable) {r['coverage']['occ']:.3f} | later content tokens in doc "
                f"{r['coverage']['later_in_doc']:.3f} (n={cov['later_n']})")
        if a.eval_span:
            dg = []
            for it, rec in zip(val, rows):
                if _get(it, "choices"):
                    continue
                try:
                    sc, conf, d, txt = span_eval(it)
                except torch.OutOfMemoryError:
                    if not a.mem_safe:
                        raise
                    torch.cuda.empty_cache()
                    continue
                rec["kv_span"] = {"score": sc, "conf": conf, "txt": txt}
                if d is not None:
                    rec["span_diag"] = d
                    dg.append(d)
            sp = [x for x in rows if "kv_span" in x]
            r["span_n"] = len(sp)
            r["kv_span"] = sum(x["kv_span"]["score"] for x in sp) / max(len(sp), 1)
            r["kv_on_span_items"] = sum(x["kv"]["score"] for x in sp) / max(len(sp), 1)
            r["span_diag"] = {k: sum(d[k] for d in dg) / max(len(dg), 1) for k in ("hit", "mass", "em", "end_mass")}
            r["span_diag"]["n"] = len(dg)
        per_task: dict = {}
        for it, rec in zip(val, rows):
            per_task.setdefault(_get(it, "doc_id").split("_")[0], []).append(rec)
        r["per_task"] = {t: {c: sum(x[c]["score"] for x in v if c in x) / max(sum(1 for x in v if c in x), 1)
                             for c in ("kv", "wrong_doc", "closed_tail", "kv_off", "kv_span")
                             if any(c in x for x in v)} for t, v in per_task.items()}
        if a.save_items:
            r["items"] = [[str(_get(it, "key")), rec["kv"]["score"], rec["wrong_doc"]["score"] if "wrong_doc" in rec else None,
                           rec["closed_tail"]["score"], rec["kv_off"]["score"] if "kv_off" in rec else None,
                           rec["kv_span"]["score"] if "kv_span" in rec else None,
                           rec["kv_span"]["conf"] if "kv_span" in rec else None,
                           rec.get("span_diag"), rec.get("kv_txt"), rec["kv_span"]["txt"] if "kv_span" in rec else None]
                          for it, rec in zip(val, rows)]
        hist["evals"].append(r)
        crit = r["kv"]
        tag = ""
        if step < a.select_min_step:
            best = max(best, crit)
            tag = " (bar; not selectable)"
        elif crit > best:
            best = crit
            save_d13(itf, best_path, {"stage": "p5", "step": step, "args": vars(a),
                                      "val": {k: v for k, v in r.items() if k != "items"}})
            tag = " *best*"
        log(f"  [val @ {step}] n={r['n']}: kv {r['kv']:.3f} wrong {r['wrong_doc']:.3f} closed_tail {r['closed_tail']:.3f} "
            f"gap {r['gap']:+.3f} ± {r['gap_se']:.3f} gen {r['gen']:.1f} g_val {r['g_val']}{tag}"
            + (f" | kv_off {r['kv_off']:.3f} (== step0 on {r.get('kv_off_eq_step0', float('nan')):.3f} of items)"
               if "kv_off" in r else "")
            + (f" | span n={r['span_n']}: kv_span {r['kv_span']:.3f} vs kv {r['kv_on_span_items']:.3f} | diag "
               f"{ {k: round(v, 3) for k, v in r['span_diag'].items()} }" if "kv_span" in r else ""))
        log("            per task " + " ".join(f"{t}: " + "/".join(f"{c}={v:.2f}" for c, v in d.items())
                                             for t, d in r["per_task"].items()) + f" | p5 {r['p5']}")

    def dump():
        with open(os.path.join(a.out, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)

    run_eval(0)
    dump()
    skipped = 0
    st = {"items": 0, "span_items": 0, "extr_items": 0, "span": [], "end": [], "tok": [], "bce": [], "g": [], "hit": [],
          "pc_gold": [], "pr_gold": []}
    for step in range(1, a.steps + 1):
        f_ = min(1.0, step / max(a.warmup, 1))
        for g_ in opt.param_groups:
            g_["lr"] = g_["base_lr"] * f_
        opt.zero_grad(set_to_none=True)
        acc = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "n": 0}
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
            K = m + 1
            R.mode = "off"
            try:
                with torch.no_grad():
                    t_log = kept_logits(reader, torch.tensor([tids]), K).detach() if a.kl > 0 else None
                    op, n_op, _ = rt.operator(did)
                    rows = rt.rows(did, it, tt)
                    R.load(op, n_op)
                    R.mode, R.need_logits, R.rec, R.p5_capture = "e2e", False, {}, True
                    s_log = kept_logits(reader, torch.tensor([rows]), K).detach()
                    h = R.p5_h
                    R.mode, R.p5_capture, R.rec = "off", False, {}
                tab = R.p5_tab
                keys = itf.p5_keys(tab)
                x = torch.tensor(rows[-K:], device=keys.device)
                res = itf.p5_forward(s_log, h, x, keys, tab)
                logp = res["logp"]
                ce = answer_ce(logp, rows[-m:])
                kl = kl_teacher_student(t_log[:-1], logp[:-1]) if t_log is not None else torch.zeros((), device=logp.device)
                loss = a.kl * kl + a.ce * ce
                gold = torch.tensor(tt, device=logp.device)
                with torch.no_grad():
                    pr_g = torch.softmax(s_log[:-1].float(), -1).gather(1, gold.view(-1, 1).to(s_log.device)).view(-1).to(logp.device)
                pc_g = res["Pc"][:-1].gather(1, gold.view(-1, 1)).view(-1)
                ybce = (pc_g.detach() > pr_g).float()
                bce = torch.nn.functional.binary_cross_entropy(res["sig"][:-1].clamp(1e-6, 1 - 1e-6), ybce)
                loss = loss + a.bce_w * bce
                st["bce"].append(float(bce))
                st["items"] += 1
                if not _get(it, "choices"):
                    st["extr_items"] += 1
                    _, srows = content_rows_strict(tok, target, allow_merged=bool(a.p5_merged))
                    if srows:
                        content = [tt[i] for i in srows]
                        occ = occurrences(content, tab)
                        if occ.numel():
                            st["span_items"] += 1
                            Nb = tab["ids"].shape[0]
                            terms = []
                            for k_, r_ in enumerate(srows):
                                cols = occ + k_
                                cols = cols[cols < Nb]
                                terms.append(-res["pi"][r_, cols].sum().clamp_min(1e-30).log())
                            lspan = torch.stack(terms).mean()
                            hit = int(res["pi_raw"][srows[0]].argmax())
                            st["hit"].append(float(bool(((occ - 3 <= hit) & (hit <= occ + len(content) + 2)).any())))
                            pe = itf.p5_pi(itf.p5_scores(h[srows[0]:srows[0] + 1].to(keys.device), keys, tab, None, end=True))[0]
                            lend = -pe[(occ + len(content) - 1).clamp_max(Nb - 1)].sum().clamp_min(1e-30).log()
                            loss = loss + a.copy_w * lspan + a.end_w * lend
                            st["span"].append(float(lspan))
                            st["end"].append(float(lend))
                        if a.tok_w > 0:
                            rr = torch.tensor(srows, device=logp.device)
                            pcr = res["Pc"][rr].gather(1, gold[rr].view(-1, 1)).view(-1)
                            ind = pcr.detach() > 0
                            if bool(ind.any()):
                                ltok = -(pcr[ind].clamp_min(1e-30).log()).mean()
                                loss = loss + a.tok_w * ltok
                                st["tok"].append(float(ltok))
                        st["pc_gold"].append(float(pc_g[srows].mean()) if srows else 0.0)
                        st["pr_gold"].append(float(pr_g[srows].mean()) if srows else 0.0)
                st["g"].append(float(res["g"][:-1].mean()))
                if not math.isfinite(float(loss)):
                    skipped += 1
                    continue
                (loss / a.batch).backward()
            except torch.OutOfMemoryError:
                if not a.mem_safe:
                    raise
                R.mode, R.p5_capture, R.rec = "off", False, {}
                rt._ops.clear()
                torch.cuda.empty_cache()
                skipped += 1
                log(f"  OOM on {did} ({len(tids)} tokens); item skipped")
                continue
            finally:
                R.mode, R.p5_capture, R.rec = "off", False, {}
            acc["loss"] += float(loss)
            acc["kl"] += float(kl)
            acc["ce"] += float(ce)
            acc["n"] += 1
        if acc["n"]:
            gn = torch.nn.utils.clip_grad_norm_(plist, 1.0)
            if step <= 3 or step % 25 == 0:
                P_ = itf.P5
                gr = {k: (round(float(P_[k].grad.norm()), 6) if P_[k].grad is not None else None) for k in P_}
                L_ = lambda k: sum(st[k][-50:]) / max(len(st[k][-50:]), 1)   # noqa: E731
                log(f"  [p5 live @ {step}] grad |.| {gr} (pre-clip {float(gn):.4f}) | last-50 means: span {L_('span'):.3f} "
                    f"end {L_('end'):.3f} tok {L_('tok'):.3f} bce {L_('bce'):.3f} gate(ans rows) {L_('g'):.5f} first-row hit "
                    f"{L_('hit'):.3f} Pc(gold) {L_('pc_gold'):.3f} Pr(gold) {L_('pr_gold'):.3f} | items {st['items']} extractive "
                    f"{st['extr_items']} span-supervised {st['span_items']} | state {itf.p5_state()}")
            opt.step()
            itf._p5_ver += 1
            r = {k: (v / acc["n"] if k != "n" else v) for k, v in acc.items()}
            r.update(step=step, grad_norm=float(gn), t=round(time.time() - t0, 1))
            hist["steps"].append(r)
            if step % 25 == 0 or step <= 3:
                w = hist["steps"][-25:]
                log(f"  [step {step}/{a.steps}] (avg last {len(w)}) loss {sum(x['loss'] for x in w) / len(w):.4f} kl "
                    f"{sum(x['kl'] for x in w) / len(w):.4f} ce {sum(x['ce'] for x in w) / len(w):.4f} "
                    f"({(time.time() - t0) / step:.2f}s/step) skipped {skipped}")
        if step % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if step % a.eval_every == 0 or step == a.steps:
            run_eval(step)
            dump()
    save_d13(itf, os.path.join(a.out, "d13_last.pt"), {"stage": "p5", "step": a.steps, "args": vars(a)})
    hist["train_stats"] = {k: (v if isinstance(v, int) else None) for k, v in st.items()}
    dump()
    if torch.cuda.is_available():
        log(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    log(f"done: {a.steps} steps, skipped {skipped}, best val kv {best:.4f} -> {best_path} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
