"""D18 direction A: multi-turn query map. Continues D13's paper-form read (outputs/D13_reg_w10/d13_best.pt) on
multi-turn questions built from TRAIN documents (xenc/d18_mt.py: deferred / distractor / referent, paraphrased
templates), mixed with single-turn items.

Reader rows (student) = the last `tail` document tokens + the WHOLE conversation (d18_context.segments conv_str) +
gold answer; the read replaces the attention output at the paired layers (D13Reader "e2e", gated).
Phase 1 (steps 1..--s1_steps), label-free: loss = dist = mean_ℓ [KL(p_enc-attn ‖ p_read) + rel_w·relMSE(B o, B₀ o_enc)],
  where the teacher is the ENCODER continuing over the same rows from its operator prefix, i.e. the encoder's view of
  the whole conversation (no answers used beyond teacher forcing the same rows; no text teacher).
Phase 2 (the rest): the paper objective, as register_d13.py:
  loss = kl·KL(p_text ‖ p_student)[answer rows] + ce·CE(gold) + dist_w·dist,
  text teacher = the reader with the document in the FIRST user message of the same conversation (d18_context.render).
Every draw: task by --task_p, then with prob --mt_p a multi-turn variant (uniform over the applicable ones), else the
single-turn item. Selection on HELD-OUT val documents (D13's val set: QuALITY/MuSiQue/LastingBench val docs, LongHealth
--val_fold) over --val_variants with the JUDGE wording (template 0): kv / wrong_doc / closed_tail; criterion = mean kv
over variants (+1e-3·mean gap). Judge documents (LongHealth --test_fold, QuALITY test) are never loaded.

    python scripts/register_d18mt.py --init outputs/D13_reg_w10/d13_best.pt --out outputs/D18A_mt
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
    ap.add_argument("--init", default="outputs/D13_reg_w10/d13_best.pt")
    ap.add_argument("--train", default="quality:150,musique:150,lastingbench:150,longhealth:0")
    ap.add_argument("--task_p", default="quality:0.4,longhealth:0.2,musique:0.2,lastingbench:0.2")
    ap.add_argument("--val", default="quality:10,musique:20,lastingbench:20,longhealth:0")
    ap.add_argument("--val_q_per_doc", type=int, default=4)
    ap.add_argument("--val_variants", default="orig,deferred,distractor")
    ap.add_argument("--val_fold", type=int, default=1)
    ap.add_argument("--test_fold", type=int, default=0)
    ap.add_argument("--mt_p", type=float, default=0.5, help="probability of a multi-turn draw")
    ap.add_argument("--mt_variants", default="deferred,distractor,referent")
    ap.add_argument("--judge_template_p", type=float, default=0.34, help="prob. of the judge wording (else paraphrase)")
    ap.add_argument("--s1_steps", type=int, default=300, help="label-free phase (dist only)")
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--steps", type=int, default=2000, help="total steps (phase 1 + phase 2)")
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
    ap.add_argument("--eval_every", type=int, default=400)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_seed", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.train = ",".join(f"{s.split(':')[0]}:2" for s in a.train.split(","))
        a.val = ",".join(f"{s.split(':')[0]}:1" for s in a.val.split(","))
        a.steps, a.s1_steps, a.eval_every, a.val_q_per_doc = 6, 2, 3, 1
    from eval_d18_context import ids_of, make_rt_class
    from eval_doc import wrong_doc_id
    from fit_d13 import build, select_items
    from register_state import kept_logits
    from xenc.backbone import text_ids
    from xenc.d13_read import save_d13
    from xenc.d18_context import render, segments
    from xenc.d18_mt import make_train_item
    from xenc.doc_tasks import gold_text, score
    from xenc.losses import answer_ce, kl_teacher_student
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    reader, enc, tok, itf, lmap = build(a, init=a.init)
    rt = make_rt_class()(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens)
    tr_folds = set(range(5)) - {a.test_fold, a.val_fold}
    train, d1 = select_items(a.train, "train", tr_folds, 0, a.seed)
    val, d2 = select_items(a.val, "val", {a.val_fold}, a.val_q_per_doc, a.val_seed)
    rt.docs = {**d1, **d2}
    tr_docs, va_docs = {_get(i, "doc_id") for i in train}, {_get(i, "doc_id") for i in val}
    assert not (tr_docs & va_docs), "val docs overlap train docs"
    for it in train + val:                                 # judge documents are never part of this run
        did = _get(it, "doc_id")
        if did.startswith("longhealth"):
            assert (_get(it, "meta") or {}).get("fold") != a.test_fold, "LongHealth test fold leaked"
    by_doc: dict = {}
    for it in train:
        by_doc.setdefault(_get(it, "doc_id"), []).append(it)
    mt_vars = [v for v in a.mt_variants.split(",") if v]
    pools: dict = {}
    for it in train:
        pools.setdefault(_get(it, "doc_id").split("_")[0], []).append(it)
    # which variants apply per item (template-independent: referent needs an entity, distractor a sibling question)
    from xenc.d18_mt import applicable
    ok = {v: {str(_get(i, "key")) for i in applicable(train, rt.docs, v)} for v in mt_vars}
    tp = {k: float(v) for k, v in (s.split(":") for s in a.task_p.split(",")) if k in pools}
    tasks, weights = list(tp), [tp[t] for t in tp]
    # held-out val conversations (judge wording, template 0)
    vby: dict = {}
    for it in val:
        vby.setdefault(_get(it, "doc_id"), []).append(it)
    val_order: dict = {}
    for d in sorted(va_docs):
        val_order.setdefault(d.split("_")[0], []).append(d)
    vrng = random.Random(a.val_seed)
    val_cis = []
    for var in [v for v in a.val_variants.split(",") if v]:
        for it in val:
            did = _get(it, "doc_id")
            ci = make_train_item(it, var, rt.docs[did], vby[did], vrng, template=0)
            if ci is not None:
                val_cis.append(ci)
    B0 = {li: itf.B[str(li)].detach().clone() for li in lmap}
    itf.trainable(train_B=bool(a.train_B), train_g=bool(a.train_g))
    groups = [{"params": list(itf.A.parameters()), "lr": a.lr}]
    if a.train_B:
        groups.append({"params": list(itf.B.parameters()) + list(itf.Bb.parameters()), "lr": a.lr_B})
    for p in itf.M.parameters():
        p.requires_grad_(True)
    if itf.mlp_hidden:
        groups.append({"params": list(itf.M.parameters()), "lr": a.lr_mlp})
    if a.train_g:
        groups.append({"params": [itf.g], "lr": a.lr_g})
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    log(f"D18A multi-turn registration: {a.encoder} -> {a.reader} | layers {lmap} | init {a.init} | trainable "
        f"{sum(p.numel() for p in params) / 1e6:.1f}M | train {len(train)} items / {len(tr_docs)} docs "
        f"{ {t: len(v) for t, v in pools.items()} } | mt applicable { {v: len(s) for v, s in ok.items()} } | "
        f"held-out val {len(val_cis)} conversations / {len(va_docs)} docs | s1 {a.s1_steps} / {a.steps} steps | "
        f"mt_p {a.mt_p} | kl {a.kl} ce {a.ce} dist {a.dist_w}")
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    hist = {"steps": [], "evals": []}
    best, t0 = -1e9, time.time()
    best_path = os.path.join(a.out, "d13_best.pt")
    ct_cache: dict = {}

    def conv_ids(ci):
        return ids_of(tok, segments(tok, a.reader, ci, SYSTEM_ANSWER)["conv_str"])

    def run_eval(step):
        nonlocal best
        per: dict = {}
        for ci in val_cis:
            did, it = ci["doc_id"], ci["item"]
            w = wrong_doc_id(did, val_order[did.split("_")[0]])
            sfx = {"_suffix": conv_ids(ci)}
            rec = {}
            for c, d, rd in (("kv", did, True), ("wrong_doc", w, True), ("closed_tail", did, False)):
                ck = (ci["variant"], ci["key"])
                if c == "closed_tail" and ck in ct_cache:
                    rec[c] = ct_cache[ck]
                    continue
                if d is None:
                    continue
                g, _ = rt.gen(d, sfx, a.budget, read=rd)
                rec[c] = float(score(it, g))
                if c == "closed_tail":
                    ct_cache[ck] = rec[c]
            per.setdefault(ci["variant"], []).append(rec)
        r = {"step": step, "variants": {}}
        for v, rows in per.items():
            m = {c: sum(x[c] for x in rows if c in x) / max(1, sum(c in x for x in rows))
                 for c in ("kv", "wrong_doc", "closed_tail")}
            m["gap"] = sum(x["kv"] - x["wrong_doc"] for x in rows if "wrong_doc" in x) / max(
                1, sum("wrong_doc" in x for x in rows))
            m["n"] = len(rows)
            r["variants"][v] = m
        V = r["variants"]
        r["kv"] = sum(m["kv"] for m in V.values()) / len(V)
        r["gap"] = sum(m["gap"] for m in V.values()) / len(V)
        r["gates"] = [round(float(x), 4) for x in itf.g]
        hist["evals"].append(r)
        crit = r["kv"] + 1e-3 * r["gap"]
        tag = ""
        if crit > best and step > 0:
            best = crit
            save_d13(itf, best_path, {"stage": "D18A", "step": step, "val": r, "args": vars(a)})
            tag = " *best*"
        log(f"  [val @ {step}] mean kv {r['kv']:.3f} gap {r['gap']:+.3f} | " + " | ".join(
            f"{v}: kv {m['kv']:.3f} wrong {m['wrong_doc']:.3f} tail {m['closed_tail']:.3f} (n={m['n']})"
            for v, m in V.items()) + f" | gates {r['gates']}{tag}")

    def dump():
        with open(os.path.join(a.out, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)

    run_eval(0)
    dump()
    R = rt.R
    skipped = 0
    for step in range(1, a.steps + 1):
        phase1 = step <= a.s1_steps
        opt.zero_grad(set_to_none=True)
        acc = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "dist": 0.0, "mt": 0, "n": 0}
        for _ in range(a.batch):
            t = rng.choices(tasks, weights)[0]
            it = pools[t][rng.randrange(len(pools[t]))]
            did = _get(it, "doc_id")
            var = "orig"
            if rng.random() < a.mt_p:
                vs = [v for v in mt_vars if str(_get(it, "key")) in ok[v]]
                if vs:
                    var = vs[rng.randrange(len(vs))]
            tmpl = 0 if rng.random() < a.judge_template_p else 1 + rng.randrange(2)
            ci = make_train_item(it, var, rt.docs[did], by_doc[did], rng, template=None if var == "orig" else tmpl)
            if ci is None:
                ci = make_train_item(it, "orig", rt.docs[did], by_doc[did], rng)
                var = "orig"
            tt = text_ids(tok, gold_text(it))
            m = len(tt)
            rows = rt.rows(did, {"_suffix": conv_ids(ci)}, tt)
            t_log = None
            if not phase1:
                tids = ids_of(tok, render(tok, a.reader, ci, SYSTEM_ANSWER, rt.docs[did])) + tt
                if len(tids) > a.max_tokens:
                    skipped += 1
                    continue
                R.mode = "off"
                with torch.no_grad():
                    t_log = kept_logits(reader, torch.tensor([tids]), m + 1).detach()
            op, n_op, _ = rt.operator(did)
            rec = rt.teacher(did, rows) if a.dist_w > 0 else None
            R.load(op, n_op)
            R.mode, R.need_logits, R.rec = "e2e", a.dist_w > 0, {}
            try:
                with torch.enable_grad():
                    s_log = kept_logits(reader, torch.tensor([rows]), m + 1)
                    zero = torch.zeros((), device=s_log.device)
                    kl, ce = zero, zero
                    if not phase1:
                        kl = kl_teacher_student(t_log[:-1], s_log[:-1])
                        ce = answer_ce(s_log, rows[-m:])
                    loss = a.kl * kl + a.ce * ce
                    dist = zero
                    if a.dist_w > 0:
                        with torch.no_grad():
                            yref = {li: (rec[e]["out"].float() @ B0[li].t()) for li, e in lmap.items()}
                        terms = rt.attn_terms(rec, op, n_op, yref)
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
            acc["mt"] += int(var != "orig")
            acc["n"] += 1
        if acc["n"]:
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            r = {k: (v / acc["n"] if k not in ("n", "mt") else v) for k, v in acc.items()}
            r.update(step=step, phase=1 if phase1 else 2, grad_norm=float(gn), t=round(time.time() - t0, 1))
            hist["steps"].append(r)
            if step % 25 == 0 or step <= 3:
                w = hist["steps"][-25:]
                avg = {k: sum(x[k] for x in w) / len(w) for k in ("loss", "kl", "ce", "dist")}
                log(f"  [step {step}/{a.steps} ph{r['phase']}] (avg last {len(w)}) loss {avg['loss']:.4f} kl "
                    f"{avg['kl']:.4f} ce {avg['ce']:.4f} dist {avg['dist']:.4f} mt-frac "
                    f"{sum(x['mt'] for x in w) / max(1, sum(x['n'] for x in w)):.2f} |g| {r['grad_norm']:.3f} "
                    f"({(time.time() - t0) / step:.2f}s/step)")
        if step % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if step % a.eval_every == 0 or step == a.steps:
            run_eval(step)
            dump()
    save_d13(itf, os.path.join(a.out, "d13_last.pt"), {"stage": "D18A", "step": a.steps, "args": vars(a)})
    dump()
    if torch.cuda.is_available():
        log(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    log(f"done: {a.steps} steps, skipped {skipped}, best val crit {best:.4f} -> {best_path} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
