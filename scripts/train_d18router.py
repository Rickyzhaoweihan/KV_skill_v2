"""D18 direction B: train a library router (xenc/d18_router.py) with InfoNCE over per-document routing scores.

Objective (the ONLY loss): CE over documents of the router score, gold = the question's document, negatives = N−1
other documents of the SAME task drawn uniformly from the TRAIN documents (N per draw from --Ns, capped by the pool).
Reader rows = the doc-free chat header + the conversation (single-turn, or with prob --mt_p a multi-turn training
variant of xenc/d18_mt.py, paraphrased wording); the stock reader (no read) prefills them and the router reads the
self-attention inputs of the question rows (--row_sub random rows per draw). Operators = keys of the encoder's prefill
of the FULL document prefix at the paired encoder layers (exactly what eval_d18_library.py loads). D13's interface
(A, MLP, B, gates) and both backbones are frozen; only the router (rank-r residual on the mapped query, layer×head
aggregation weights, temperature) trains.
Selection on HELD-OUT val documents (QuALITY / MuSiQue / LastingBench val docs, LongHealth --val_fold), judge wording:
route top-1 accuracy at --val_Ns (libraries = gold + seeded N−1 same-task docs from train ∪ val), mean over N.
Judge documents (LongHealth --test_fold, QuALITY test) are never loaded.

    python scripts/train_d18router.py --interface outputs/D13_reg_w10/d13_best.pt --out outputs/D18B_router
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
    ap.add_argument("--interface", default="outputs/D13_reg_w10/d13_best.pt")
    ap.add_argument("--train", default="quality:150,musique:150,lastingbench:150,longhealth:0")
    ap.add_argument("--task_p", default="quality:0.4,longhealth:0.2,musique:0.2,lastingbench:0.2")
    ap.add_argument("--val", default="quality:20,musique:30,lastingbench:30,longhealth:0")
    ap.add_argument("--val_q_per_doc", type=int, default=3)
    ap.add_argument("--val_variants", default="orig,deferred")
    ap.add_argument("--val_fold", type=int, default=1)
    ap.add_argument("--test_fold", type=int, default=0)
    ap.add_argument("--Ns", default="5,20,50")
    ap.add_argument("--init_router", default=None, help="start from (or, with --steps 0, only evaluate) this router")
    ap.add_argument("--val_neg", default="mixed", choices=["mixed", "heldout"],
                    help="val library negatives: mixed = train ∪ val docs of the task; heldout = val docs only "
                         "(no document the router was trained on; removes the 'suppress seen documents' confound)")
    ap.add_argument("--val_Ns", default="5,20,50")
    ap.add_argument("--mt_p", type=float, default=0.5)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--row_sub", type=int, default=48)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3, help="U, V")
    ap.add_argument("--lr_agg", type=float, default=1e-2, help="aggregation weights, temperature")
    ap.add_argument("--eval_every", type=int, default=300)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_seed", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.train = ",".join(f"{s.split(':')[0]}:6" for s in a.train.split(",") if not s.startswith("longhealth")) + \
            ",longhealth:0"
        a.val = ",".join(f"{s.split(':')[0]}:2" for s in a.val.split(","))
        a.steps, a.eval_every, a.batch, a.val_q_per_doc = 6, 3, 2, 1
    from eval_d18_context import ids_of
    from fit_d13 import D13Rt, build, select_items
    from xenc.d18_context import segments
    from xenc.d18_mt import MT_VARIANTS, make_train_item
    from xenc.d18_router import D18Router, HiddenCap
    from xenc.state_skill import doc_prefix_text
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    a.tail = 0
    reader, enc, tok, itf, lmap = build(a, init=a.interface)
    for p in itf.parameters():
        p.requires_grad_(False)
    rt = D13Rt(reader, enc, tok, a.reader, itf, 0, a.max_tokens)
    tr_folds = set(range(5)) - {a.test_fold, a.val_fold}
    train, d1 = select_items(a.train, "train", tr_folds, 0, a.seed)
    val, d2 = select_items(a.val, "val", {a.val_fold}, a.val_q_per_doc, a.val_seed)
    rt.docs = {**d1, **d2}
    tr_docs, va_docs = sorted({_get(i, "doc_id") for i in train}), sorted({_get(i, "doc_id") for i in val})
    assert not (set(tr_docs) & set(va_docs))
    for it in train + val:
        if _get(it, "doc_id").startswith("longhealth"):
            assert (_get(it, "meta") or {}).get("fold") != a.test_fold, "LongHealth test fold leaked"

    # ---------------- operator keys (full prefix, as eval_d18_library.py), then drop the encoder
    header = ids_of(tok, doc_prefix_text(tok, a.reader, "", SYSTEM_ANSWER))
    Kd, skip, t_h = {}, len(header), time.time()
    n_trunc = 0
    for d in tr_docs + va_docs:
        ids = rt.pre(d)
        if len(ids) >= a.max_tokens - 1024:          # left-truncated by pre(): no chat header, all slots are content
            n_trunc += 1
        else:                                        # skip = shared header length over UNtruncated docs (as eval)
            skip = min(skip, next((k for k in range(min(len(ids), len(header))) if ids[k] != header[k]),
                                  min(len(ids), len(header))))
        rt.Tc.rec, rt.Tc.kv_only = {}, True
        with torch.no_grad():
            enc(input_ids=torch.tensor([ids], device=rt.edev), use_cache=False, logits_to_keep=1)
        Kd[d] = {e: r["k"].to(rt.dev, torch.bfloat16) for e, r in rt.Tc.rec.items()}
        rt.Tc.rec, rt.Tc.kv_only = None, False
    gib = sum(k.numel() * 2 for v in Kd.values() for k in v.values()) / 2**30
    log(f"harvested keys of {len(Kd)} docs in {time.time() - t_h:.0f}s ({gib:.2f} GiB), shared header skip={skip} ({n_trunc} left-truncated docs)")
    rt.Tc.detach()
    rt.enc = None
    del enc
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    by_doc: dict = {}
    for it in train:
        by_doc.setdefault(_get(it, "doc_id"), []).append(it)
    pools: dict = {}
    for it in train:
        pools.setdefault(_get(it, "doc_id").split("_")[0], []).append(it)
    task_docs = {t: sorted({_get(i, "doc_id") for i in v}) for t, v in pools.items()}
    tp = {k: float(v) for k, v in (s.split(":") for s in a.task_p.split(",")) if k in pools}
    tasks, weights = list(tp), [tp[t] for t in tp]
    Ns = [int(x) for x in a.Ns.split(",")]
    if a.init_router:
        R = D18Router.load(a.init_router, itf)[0].to(rt.dev)
    else:
        R = D18Router(itf, rank=a.rank, seed=a.seed).to(rt.dev)
    opt = torch.optim.AdamW([{"params": list(R.U.parameters()) + list(R.V.parameters()), "lr": a.lr},
                             {"params": [R.agg, R.logbeta], "lr": a.lr_agg}], weight_decay=0.0)
    cap = HiddenCap(reader, itf.layers)

    def conv_ids(ci):
        return ids_of(tok, segments(tok, a.reader, ci, SYSTEM_ANSWER)["conv_str"])

    @torch.no_grad()
    def hidden(rows):
        cap.h, cap.on = {}, True
        try:
            reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=False, logits_to_keep=1)
        finally:
            cap.on = False
        return {li: h[0].float() for li, h in cap.h.items()}

    def kdocs(lib):
        return {e: [Kd[d][e] for d in lib] for e in sorted(set(lmap.values()))}

    # ---------------- held-out val: fixed conversations and libraries (judge wording)
    vby: dict = {}
    for it in val:
        vby.setdefault(_get(it, "doc_id"), []).append(it)
    vrng = random.Random(a.val_seed)
    val_set = []
    for var in [v for v in a.val_variants.split(",") if v]:
        for it in val:
            did = _get(it, "doc_id")
            ci = make_train_item(it, var, rt.docs[did], vby[did], vrng, template=0)
            if ci is None:
                continue
            t = did.split("_")[0]
            heldout = {d for d in va_docs if d.startswith(t)}
            cand = sorted((heldout if a.val_neg == "heldout" else set(task_docs.get(t, [])) | heldout) - {did})
            random.Random(f"{a.val_seed}:{ci['key']}").shuffle(cand)
            libs = {N: [did] + cand[: N - 1] for N in (int(x) for x in a.val_Ns.split(","))}
            val_set.append((ci, libs))
    log(f"D18B router: {a.reader} | layers {lmap} | interface {a.interface} (frozen) | rank {a.rank} | "
        f"{sum(p.numel() for p in R.parameters()) / 1e6:.2f}M router params | train {len(train)} items / "
        f"{len(tr_docs)} docs { {t: len(v) for t, v in task_docs.items()} } | val {len(val_set)} conversations / "
        f"{len(va_docs)} docs | Ns {Ns} | mt_p {a.mt_p}")
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    hist = {"steps": [], "evals": []}
    best, t0 = -1e9, time.time()

    @torch.no_grad()
    def run_eval(step):
        nonlocal best
        acc: dict = {}
        for ci, libs in val_set:
            rows = header + conv_ids(ci)
            H = hidden(rows)
            qs = torch.arange(len(header), len(rows), device=rt.dev)
            xs = {li: H[li][len(header):] for li in itf.layers}
            t = ci["doc_id"].split("_")[0]
            for N, lib in libs.items():
                s = R(xs, qs, kdocs(lib), skip)
                p = torch.softmax(s, 0)
                rank = int((s > s[0]).sum())
                k = (t, ci["variant"], len(lib), N)
                acc.setdefault(k, []).append((float(rank == 0), rank, float(p[0]), 1.0 / len(lib)))
        res = {}
        for (t, v, n, N), xs_ in sorted(acc.items()):
            res[f"{t}/{v}/N{N}"] = {"top1": sum(x[0] for x in xs_) / len(xs_), "rank": sum(x[1] for x in xs_) / len(xs_),
                                    "mass": sum(x[2] for x in xs_) / len(xs_), "chance": xs_[0][3], "lib": n,
                                    "n": len(xs_)}
        byN: dict = {}
        for k, v in res.items():
            byN.setdefault(k.split("/N")[1], []).append(v["top1"])
        r = {"step": step, "per": res, "top1_by_N": {N: sum(v) / len(v) for N, v in byN.items()},
             "beta": float(R.logbeta.exp())}
        r["crit"] = sum(r["top1_by_N"].values()) / len(r["top1_by_N"])
        hist["evals"].append(r)
        tag = ""
        if r["crit"] > best and a.steps > 0:
            best = r["crit"]
            R.save(os.path.join(a.out, "router_best.pt"), {"step": step, "val": r, "args": vars(a), "skip": skip})
            tag = " *best*"
        log(f"  [val @ {step}] top1 by N { {N: round(v, 3) for N, v in r['top1_by_N'].items()} } crit {r['crit']:.3f} "
            f"beta {r['beta']:.2f}{tag}")
        log("            " + " | ".join(f"{k} {v['top1']:.2f}/{v['mass']:.2f} (ch {v['chance']:.2f})"
                                       for k, v in res.items()))

    def dump():
        with open(os.path.join(a.out, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)

    run_eval(0)
    dump()
    params = list(R.parameters())
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        tot, top1, n = 0.0, 0.0, 0
        for _ in range(a.batch):
            t = rng.choices(tasks, weights)[0]
            it = pools[t][rng.randrange(len(pools[t]))]
            did = _get(it, "doc_id")
            ci = None
            if rng.random() < a.mt_p:
                vs = list(MT_VARIANTS)
                rng.shuffle(vs)
                for v in vs:
                    ci = make_train_item(it, v, rt.docs[did], by_doc[did], rng)
                    if ci is not None:
                        break
            if ci is None:
                ci = make_train_item(it, "orig", rt.docs[did], by_doc[did], rng)
            rows = header + conv_ids(ci)
            if len(rows) > a.max_tokens:
                continue
            H = hidden(rows)
            q_idx = list(range(len(header), len(rows)))
            if a.row_sub and len(q_idx) > a.row_sub:
                q_idx = sorted(rng.sample(q_idx, a.row_sub))
            qi = torch.tensor(q_idx, device=rt.dev)
            xs = {li: H[li][qi] for li in itf.layers}
            N = min(rng.choice(Ns), len(task_docs[t]))
            neg = rng.sample([d for d in task_docs[t] if d != did], N - 1)
            lib = [did] + neg
            with torch.enable_grad():
                s = R(xs, qi, kdocs(lib), skip, ckpt=True)
                loss = torch.nn.functional.cross_entropy(s.unsqueeze(0), torch.zeros(1, dtype=torch.long,
                                                                                      device=s.device))
                (loss / a.batch).backward()
            tot += float(loss)
            top1 += float(int(s.argmax()) == 0)
            n += 1
        if not n:
            continue
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist["steps"].append({"step": step, "loss": tot / n, "top1": top1 / n, "grad_norm": float(gn),
                              "beta": float(R.logbeta.exp())})
        if step % 50 == 0 or step <= 3:
            w = hist["steps"][-50:]
            log(f"  [step {step}/{a.steps}] (avg last {len(w)}) InfoNCE {sum(x['loss'] for x in w) / len(w):.4f} "
                f"train top1 {sum(x['top1'] for x in w) / len(w):.3f} |g| {gn:.3f} beta {float(R.logbeta.exp()):.2f} "
                f"({(time.time() - t0) / step:.2f}s/step)")
        if step % a.eval_every == 0 or step == a.steps:
            run_eval(step)
            dump()
    if a.steps > 0:
        R.save(os.path.join(a.out, "router_last.pt"), {"step": a.steps, "args": vars(a), "skip": skip})
    dump()
    if torch.cuda.is_available():
        log(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    log(f"done: best val crit {best:.4f} -> {os.path.join(a.out, 'router_best.pt')} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
