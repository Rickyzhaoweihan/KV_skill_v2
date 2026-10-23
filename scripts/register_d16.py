"""D16 stage 2 (copy of register_d13.py for the generalised read xenc/d16_read.py): end-to-end registration with the
paper's objective on top of a D16 stage-1 (distilled) init. Everything below is as in D13 stage 2.

Same objective as register_doc.py / register_state.py (losses from xenc/losses.py): teacher = the reader with the
document in context (ctx_prompt + gold answer), student = the reader on its own doc tail + question + gold answer with
the D13 read of the encoder's operator replacing the attention output at the paired layers (gated: own + g·(read−own)):
    loss = kl·KL(p_text ‖ p_student)[answer tokens] + ce·CE(gold) + dist·(KL(p_enc-attn ‖ p_read) + relMSE(B o, B₀ o_enc))
The last term is stage 1's label-free attention distillation against the encoder's own pass over the same rows; it keeps
the read addressing the document (no collapse to a document-independent shift). A (+MLP), B, gates train; the operator
(the encoder's harvested K/V) and both backbones stay frozen. Init: stage-1 file (A) + D10's encoder-side ridge B.
Selection on HELD-OUT DOCUMENTS (QuALITY/MuSiQue/LastingBench val docs, LongHealth --val_fold): greedy kv / wrong_doc
every --eval_every steps.

    python scripts/register_d16.py --init outputs/D16_maps/d16_cat2.pt --dist_w 1.0 --steps 1500 --out outputs/D16_reg_cat2
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
    ap.add_argument("--lr_scale", type=float, default=1.0, help="× the stage-2 per-group LR table fit_d16.LR2 (= D13's)")
    ap.add_argument("--tail_n", type=int, default=0, help="unused (compat)")
    ap.add_argument("--lr_B", type=float, default=1e-5)
    ap.add_argument("--lr_g", type=float, default=1e-3)
    ap.add_argument("--train_B", type=int, default=1)
    ap.add_argument("--train_g", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.train = ",".join(f"{s.split(':')[0]}:2" for s in a.train.split(","))
        a.val = ",".join(f"{s.split(':')[0]}:1" for s in a.val.split(","))
        a.steps, a.eval_every, a.val_q_per_doc, a.n_val_attn = 6, 3, 1, 2
    from eval_doc import wrong_doc_id
    from eval_d13 import summarize
    from eval_d16 import build16, gen16
    from fit_d13 import fmt, mean_terms, select_items
    from fit_d16 import LR2
    from xenc.d13_read import d13_terms, teacher_logits
    from register_state import kept_logits
    from xenc.backbone import build_chat, text_ids
    from xenc.d16_read import save_d16
    from xenc.doc_tasks import ctx_prompt, gold_text, score
    from xenc.losses import answer_ce, kl_teacher_student
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    reader, enc, tok, itf, lmap, rt, _ = build16(a, a.init)

    def s1_terms(did, it):
        """On-policy held-out attention metrics (D13's s1_pass, D16 read)."""
        op, N, _ = rt.operator(did)
        return rt.onpolicy_terms(itf, did, it)

    def attn_terms(rec, op, N, yref):
        out = {}
        for li, e in lmap.items():
            r = rt.R.rec.get(li)
            if r is None or r.get("logits") is None:
                continue
            lt = teacher_logits(rec[e]["q"], op[e]["K"], rec[e]["k"], itf.att[str(li)].scaling)
            out[li] = d13_terms(r["logits"], lt, r["y"], yref[li], N)
        return out
    tr_folds = set(range(5)) - {a.test_fold, a.val_fold}
    train, d1 = select_items(a.train, "train", tr_folds, 0, a.seed)
    val, d2 = select_items(a.val, "val", {a.val_fold}, a.val_q_per_doc, a.seed + 1)
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
    groups = [{"params": ps, "lr": LR2[g] * a.lr_scale} for g, ps in itf.param_groups().items()]
    itf.trainable(train_B=bool(a.train_B), train_g=bool(a.train_g))
    if a.train_B:
        groups.append({"params": list(itf.B.parameters()), "lr": a.lr_B})
    if a.train_g:
        groups.append({"params": [itf.g], "lr": a.lr_g})
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    log(f"D16 stage 2: cfg {itf.cfg} | A-family {itf.n_params() / 1e6:.1f}M | {a.encoder} -> {a.reader} | layers {lmap} | init {a.init} | trainable "
        f"{sum(p.numel() for p in params) / 1e6:.1f}M | train {len(train)} items / {len(tr_docs)} docs "
        f"{ {t: len(v) for t, v in pools.items()} } | held-out val {len(val)} items / {len(va_docs)} docs | "
        f"kl {a.kl} ce {a.ce} dist {a.dist_w}")
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    hist = {"steps": [], "evals": []}
    best, t0 = -1e9, time.time()
    best_path = os.path.join(a.out, "d16_best.pt")
    ct_cache: dict = {}

    def run_eval(step):
        nonlocal best
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
                g, n = gen16(rt, d, it, a.budget, read=rd)
                rec[c] = {"score": float(score(it, g)), "gen_tokens": n}
                if c == "closed_tail":
                    ct_cache[_get(it, "key")] = rec[c]
            rows.append(rec)
        s = summarize(rows, ("closed_tail", "kv", "wrong_doc"))
        att = mean_terms([s1_terms(_get(it, "doc_id"), it)
                          for it in val[:: max(1, len(val) // max(a.n_val_attn, 1))][: a.n_val_attn]])
        r = {"step": step, "kv": s["kv"]["score"], "wrong_doc": s["wrong_doc"]["score"],
             "closed_tail": s["closed_tail"]["score"],
             "gap": s.get("kv_minus_wrong_doc", {}).get("paired_gap", float("nan")),
             "gap_se": s.get("kv_minus_wrong_doc", {}).get("se", float("nan")), "gen": s["kv"]["mean_gen_tokens"],
             "gen_wrong": s["wrong_doc"]["mean_gen_tokens"], "attn": att,
             "gates": [round(float(x), 4) for x in itf.g], "n": len(rows)}
        per_task = {}
        for it, rec in zip(val, rows):
            per_task.setdefault(_get(it, "doc_id").split("_")[0], []).append(rec)
        r["per_task"] = {t: {c: sum(x[c]["score"] for x in v if c in x) / len(v) for c in ("kv", "wrong_doc", "closed_tail")}
                         for t, v in per_task.items()}
        hist["evals"].append(r)
        crit = r["kv"] + (1e-3 * r["gap"] if math.isfinite(r["gap"]) else 0.0)
        tag = ""
        if crit > best:
            best = crit
            save_d16(itf, best_path, {"stage": "register", "step": step, "val": r, "args": vars(a)})
            tag = " *best*"
        log(f"  [val @ {step}] held-out docs n={r['n']}: kv {r['kv']:.3f} wrong {r['wrong_doc']:.3f} closed_tail "
            f"{r['closed_tail']:.3f} gap {r['gap']:+.3f} ± {r['gap_se']:.3f} gen {r['gen']:.1f}/{r['gen_wrong']:.1f} "
            f"gates {r['gates']}{tag}")
        log(f"            per task " + " ".join(f"{t}: {v['kv']:.2f}/{v['wrong_doc']:.2f}/{v['closed_tail']:.2f}"
                                             for t, v in r["per_task"].items()) + f" | attn {fmt(att)}")

    def dump():
        with open(os.path.join(a.out, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)

    run_eval(0)
    dump()
    R = rt.R
    skipped = 0
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
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
            with torch.no_grad():
                t_log = kept_logits(reader, torch.tensor([tids]), m + 1).detach()
            rows = rt.rows(did, it, tt)
            op, n_op, _ = rt.operator(did)
            rec = rt.teacher(did, rows) if a.dist_w > 0 else None
            R.load(op, n_op)
            R.mode, R.need_logits, R.rec = "e2e", a.dist_w > 0, {}
            try:
                with torch.enable_grad():
                    s_log = kept_logits(reader, torch.tensor([rows]), m + 1)
                    kl = kl_teacher_student(t_log[:-1], s_log[:-1])
                    ce = answer_ce(s_log, rows[-m:])
                    loss = a.kl * kl + a.ce * ce
                    dist = torch.zeros((), device=kl.device)
                    if a.dist_w > 0:
                        with torch.no_grad():
                            yref = {li: (rec[e]["out"].float() @ B0[li].t()) for li, e in lmap.items()}
                        terms = attn_terms(rec, op, n_op, yref)
                        dist = torch.stack([x["kl"] + a.rel_w * x["rel"] for x in terms.values()]).mean()
                        loss = loss + a.dist_w * dist.to(loss.device)
                    if not math.isfinite(float(loss)):
                        skipped += 1
                        continue
                    (loss / a.batch).backward()
            finally:
                R.mode, R.rec = "off", {}
            for k, v in (("loss", loss), ("kl", kl), ("ce", ce), ("dist", dist)):
                acc[k] += float(v)
            acc["n"] += 1
        if acc["n"]:
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            r = {k: (v / acc["n"] if k != "n" else v) for k, v in acc.items()}
            r.update(step=step, grad_norm=float(gn), t=round(time.time() - t0, 1))
            hist["steps"].append(r)
            if step % 25 == 0 or step <= 3:
                w = hist["steps"][-25:]
                avg = {k: sum(x[k] for x in w) / len(w) for k in ("loss", "kl", "ce", "dist")}
                log(f"  [step {step}/{a.steps}] (avg last {len(w)}) loss {avg['loss']:.4f} kl {avg['kl']:.4f} "
                    f"ce {avg['ce']:.4f} dist {avg['dist']:.4f} |g| {r['grad_norm']:.3f} gates "
                    f"{[round(float(x), 3) for x in itf.g]} ({(time.time() - t0) / step:.2f}s/step)")
        if step % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if step % a.eval_every == 0 or step == a.steps:
            run_eval(step)
            dump()
    save_d16(itf, os.path.join(a.out, "d16_last.pt"), {"stage": "register", "step": a.steps, "args": vars(a)})
    dump()
    if torch.cuda.is_available():
        log(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    log(f"done: {a.steps} steps, skipped {skipped}, best val crit {best:.4f} -> {best_path} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
