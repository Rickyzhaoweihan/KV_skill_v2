"""D15: DATA SCALE for D13's stage 1 (reader-query map A of the paper's read, label-free distillation from the 2B
encoder's own question-pass attention + read-out; xenc/d13_read.py, D13Rt from scripts/fit_d13.py, unchanged).

Same losses / parameterization / B as fit_d13.py: ridge A (reader layer input -> encoder layer input), then lin and
lin+MLP A trained by KL(p_teacher || p_student) + rel_w * relMSE(B o_s, B o_t), on-policy mixing after the first
epoch-equivalent. What changes is the training set: --n_docs documents from the D15 pool (data/D15_selfq: split-train
docs + external long docs; questions = native + reader-written self-questions, scripts/d15_selfq.py; answers never
used). Teacher targets are computed online (exact; the operator prefill is needed for the student's softmax anyway).

Held-out selection set = D13's held-out set (QuALITY/MuSiQue/LastingBench val docs + LongHealth fold 1, identical
selection to fit_d13.py) + the pool's held-out documents (data/D15_selfq/heldout.json, self-questions). Early
stopping on held-out on-policy read-out cosine (D13's criterion); JS reported for both parts.

    python scripts/fit_d15.py --n_docs 1000 --out outputs/D15_maps_n1000
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import _get, log  # noqa: E402


def load_pool(pool: str, use_selfq=True, use_native=True):
    """-> (items_by_doc {did: [items]}, heldout set, doc index)."""
    from xenc.doc_tasks import load_docs_index
    idx = load_docs_index(pool)
    by: dict = {}
    if use_native:
        for it in json.load(open(os.path.join(pool, "native.json"))):
            if it["doc_id"] in idx:
                by.setdefault(it["doc_id"], []).append(it)
    if use_selfq:
        for fn in sorted(glob.glob(os.path.join(pool, "selfq", "shard*.jsonl"))):
            for line in open(fn):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r["doc_id"] in idx:
                    by.setdefault(r["doc_id"], []).extend(r["items"])
    held = set(json.load(open(os.path.join(pool, "heldout.json"))))
    return by, held, idx


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--B", default="outputs/D10_maps/encB8_9b_2b_post.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--pool", default="data/D15_selfq")
    ap.add_argument("--n_docs", type=int, default=0, help="training documents (nested seeded subsets); 0 = all")
    ap.add_argument("--only_split_docs", type=int, default=0, help="1 = restrict to D13's split-train docs")
    ap.add_argument("--use_selfq", type=int, default=1)
    ap.add_argument("--use_native", type=int, default=1)
    ap.add_argument("--val", default="quality:6,musique:12,lastingbench:12,longhealth:0")
    ap.add_argument("--val_fold", type=int, default=1)
    ap.add_argument("--val_q_per_doc", type=int, default=4)
    ap.add_argument("--pool_val_docs", type=int, default=60)
    ap.add_argument("--pool_val_q", type=int, default=2)
    ap.add_argument("--q_per_doc", type=int, default=8, help="questions sampled per doc per epoch")
    ap.add_argument("--ridge_q", type=int, default=2000, help="max questions for the ridge statistics")
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--variants", default="lin,mlp")
    ap.add_argument("--mlp_hidden", type=int, default=1024)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max_passes", type=int, default=40000)
    ap.add_argument("--min_passes", type=int, default=4000, help="small sets: at least this many passes (more epochs)")
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr_mlp", type=float, default=1e-4)
    ap.add_argument("--rel_w", type=float, default=1.0)
    ap.add_argument("--onpolicy_p", type=float, default=0.5)
    ap.add_argument("--init", default="", help="start every variant from this d13/d15 file instead of the ridge fit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.n_docs, a.max_passes, a.min_passes, a.eval_every, a.ridge_q = 6, 24, 0, 12, 12
        a.val, a.pool_val_docs, a.val_q_per_doc = "quality:1,musique:1,lastingbench:1,longhealth:0", 2, 1
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    from fit_d13 import D13Rt, build, fmt, mean_terms, select_items
    from fit_state_maps import Gram, ridge
    from xenc.d13_read import D13Interface, save_d13, load_d13
    from xenc.doc_tasks import _doc_path
    reader, enc, tok, itf, lmap = build(a)
    rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens)

    # ---------------- data
    by, held_pool, idx = load_pool(a.pool, bool(a.use_selfq), bool(a.use_native))
    split_src = {"quality", "musique", "lastingbench", "longhealth"}
    cand = sorted(d for d in by if d not in held_pool and by[d])
    if a.only_split_docs:
        cand = [d for d in cand if idx[d]["src"] in split_src]
    random.Random(1234).shuffle(cand)                       # fixed order -> nested subsets across --n_docs
    tr_docs = cand[: a.n_docs] if a.n_docs else cand
    val, dv = select_items(a.val, "val", {a.val_fold}, a.val_q_per_doc, a.seed + 1)
    pv = sorted(d for d in held_pool if by.get(d))[: a.pool_val_docs]
    pval = []
    for d in pv:
        qs = by[d][:]
        random.Random(7).shuffle(qs)
        pval += qs[: a.pool_val_q]
    pool_root = Path(a.pool)
    texts = {d: _doc_path(pool_root, idx[d]["path"]).read_text() for d in set(tr_docs) | set(pv)}
    rt.docs = {**texts, **dv}
    vdocs = {_get(i, "doc_id") for i in val} | set(pv)
    assert not (vdocs & set(tr_docs)), "held-out docs overlap train docs"
    nq = sum(min(len(by[d]), a.q_per_doc) for d in tr_docs)
    src_c: dict = {}
    for d in tr_docs:
        src_c[idx[d]["src"]] = src_c.get(idx[d]["src"], 0) + 1
    log(f"D15 stage 1: {a.encoder} -> {a.reader} | layers {lmap} | train {len(tr_docs)} docs ({src_c}) | "
        f"{nq} q/epoch (q_per_doc {a.q_per_doc}; pool q {sum(len(by[d]) for d in tr_docs)}) | held-out D13 "
        f"{len(val)} q / {len({_get(i, 'doc_id') for i in val})} docs + pool {len(pval)} q / {len(pv)} docs")
    t0 = time.time()

    def epoch_items(r):
        docs = tr_docs[:]
        r.shuffle(docs)
        out = []
        for d in docs:
            qs = by[d]
            out += [(d, q) for q in (r.sample(qs, a.q_per_doc) if len(qs) > a.q_per_doc else qs)]
        return out

    # ---------------- 1) ridge (or --init)
    d_r, d_e = itf.d_r, itf.d_e
    report = {"args": vars(a), "lmap": lmap, "n_train_docs": len(tr_docs), "q_per_epoch": nq, "src": src_c,
              "variants": {}}
    if a.init:
        itf0, _ = load_d13(a.init, enc, {li: itf.B[str(li)].detach().cpu() for li in lmap})
        init_sd = {k: v.detach().clone() for k, v in itf0.state_dict().items() if k.startswith(("A.", "M."))}
        del itf0
    else:
        G = {li: Gram(d_r + 1, d_e, rt.dev) for li in lmap}
        ritems = epoch_items(random.Random(a.seed + 11))
        if len(ritems) > a.ridge_q:
            keep = set(random.Random(a.seed + 12).sample(range(len(ritems)), a.ridge_q))
            ritems = [x for k, x in enumerate(ritems) if k in keep]
        ritems.sort(key=lambda x: x[0])
        for did, it in ritems:
            rows = rt.rows(did, it)
            op, n_op, _ = rt.operator(did)
            rec = rt.teacher(did, rows)
            R = rt.R
            R.load(op, n_op)
            R.mode, R.inject, R.xcap = "teacher", rt.y_teacher(rec), {}
            with torch.no_grad():
                reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=False, logits_to_keep=1)
            R.mode = "off"
            for li, e in lmap.items():
                X = R.xcap[li]
                G[li].add(torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1), rec[e]["x"].float())
        r2 = {}
        with torch.no_grad():
            for li in lmap:
                W = ridge(G[li], a.lam)
                r2[str(li)] = G[li].r2(W)
                itf.A[str(li)].weight.copy_(W[:-1].t().float())
                itf.A[str(li)].bias.copy_(W[-1].float())
        del G
        torch.cuda.empty_cache()
        report["ridge_r2_fit"] = r2
        log(f"ridge A on {len(ritems)} q: R² fit " + " ".join(f"L{li} {v:.3f}" for li, v in r2.items())
            + f" ({time.time() - t0:.0f}s)")
        init_sd = {k: v.detach().clone() for k, v in itf.state_dict().items() if k.startswith("A.")}

    def held():
        r1 = [rt.s1_pass(_get(it, "doc_id"), it, onpolicy=True, grad=False, metrics=True) for it in val]
        r2_ = [rt.s1_pass(_get(it, "doc_id"), it, onpolicy=True, grad=False, metrics=True) for it in pval]
        m1, m2, m = mean_terms(r1), mean_terms(r2_), mean_terms(r1 + r2_)
        return {"d13val": m1, "poolval": m2, "all": m}

    if not a.init:
        h0 = held()
        report["ridge_held"] = h0
        save_d13(itf, os.path.join(a.out, "d15_ridge.pt"), {"stage": "ridge", "held": h0["d13val"], "n_docs": len(tr_docs)})
        log(f"[ridge] held-out D13-val: {fmt(h0['d13val'])}")
        log(f"[ridge] held-out pool:    {fmt(h0['poolval'])}")

    # ---------------- 2) attention distillation
    total = int(min(max(a.epochs * nq, a.min_passes), a.max_passes))
    every = min(a.eval_every, max(500, total // 8)) if not a.smoke else a.eval_every
    log(f"pass budget {total} (eval every {every}, patience {a.patience})")
    for var in [v for v in a.variants.split(",") if v]:
        mh = a.mlp_hidden if var == "mlp" else 0
        if mh != itf.mlp_hidden:
            Bd = {li: itf.B[str(li)].detach().cpu() for li in lmap}
            itf = D13Interface(enc, lmap, d_r, d_e, B=Bd, mlp_hidden=mh).to(rt.dev)
            rt.itf = rt.R.itf = itf
        itf.load_state_dict({k: v for k, v in init_sd.items() if k in itf.state_dict()}, strict=False)
        if mh and not any(k.startswith("M.") for k in init_sd):
            for li in lmap:
                torch.nn.init.zeros_(itf.M[str(li)][2].weight)
                torch.nn.init.zeros_(itf.M[str(li)][2].bias)
        groups = [{"params": list(itf.A.parameters()), "lr": a.lr}]
        if mh:
            groups.append({"params": list(itf.M.parameters()), "lr": a.lr_mlp})
        itf.trainable()
        opt = torch.optim.AdamW(groups, weight_decay=0.0)
        params = [p for g in groups for p in g["params"]]
        bpath = os.path.join(a.out, f"d15_{var}.pt")
        h = held()
        best, bad, hist = h["all"]["cos"], 0, [{"passes": 0, "held": h}]
        save_d13(itf, bpath, {"stage": f"d15 attn-distill {var}", "passes": 0, "held": h["d13val"], "n_docs": len(tr_docs)})
        log(f"[{var} @0] held D13 {fmt(h['d13val'])}")
        log(f"          held pool {fmt(h['poolval'])}")
        passes, acc, ep, stop = 0, 0, 0, False
        tl = {"kl": 0.0, "rel": 0.0, "n": 0}
        opt.zero_grad(set_to_none=True)
        while passes < total and not stop:
            ep += 1
            for did, it in epoch_items(rng):
                p_on = a.onpolicy_p if passes >= min(nq, 8000) else 0.0
                terms = rt.s1_pass(did, it, onpolicy=rng.random() < p_on)
                if not terms:
                    continue
                kl = torch.stack([t["kl"] for t in terms.values()]).mean()
                rel = torch.stack([t["rel"] for t in terms.values()]).mean()
                loss = kl + a.rel_w * rel
                if not math.isfinite(float(loss)):
                    log(f"  non-finite loss on {did}; skipped")
                    continue
                (loss / a.accum).backward()
                tl["kl"] += float(kl)
                tl["rel"] += float(rel)
                tl["n"] += 1
                passes += 1
                acc += 1
                if acc == a.accum:
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    acc = 0
                if passes % every == 0 or passes == total:
                    h = held()
                    rec = {"passes": passes, "epoch": ep, "train_kl": tl["kl"] / max(tl["n"], 1),
                           "train_rel": tl["rel"] / max(tl["n"], 1), "held": h}
                    hist.append(rec)
                    tag = ""
                    if h["all"]["cos"] > best + 1e-4:
                        best, bad = h["all"]["cos"], 0
                        save_d13(itf, bpath, {"stage": f"d15 attn-distill {var}", "passes": passes,
                                              "held": h["d13val"], "n_docs": len(tr_docs)})
                        tag = " *best*"
                    else:
                        bad += 1
                    log(f"[{var} {passes}/{total} ep {ep}] train kl {rec['train_kl']:.4f} rel {rec['train_rel']:.4f} | "
                        f"held D13 {fmt(h['d13val'])}{tag} ({time.time() - t0:.0f}s)")
                    log(f"          held pool js {h['poolval']['js']:.3f} js_doc {h['poolval']['js_doc']:.3f} cos "
                        f"{h['poolval']['cos']:.3f} top5 {h['poolval']['top5']:.3f} | all cos {h['all']['cos']:.4f}")
                    tl = {"kl": 0.0, "rel": 0.0, "n": 0}
                    if bad >= a.patience:
                        log(f"  early stop ({a.patience} evals without improvement)")
                        stop = True
                if passes >= total or stop:
                    break
        report["variants"][var] = {"history": hist, "best_all_cos": best, "path": bpath, "passes": passes}
        with open(os.path.join(a.out, "report.json"), "w") as f:
            json.dump(report, f, indent=1)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    if torch.cuda.is_available():
        log(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    log(f"done ({time.time() - t0:.0f}s) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
