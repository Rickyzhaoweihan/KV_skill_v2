"""D25 first method: learned per-layer read gates for knowledge conflict (M2) and conflict-triggered gates (M5).

Base interface frozen (A, MLP, B, bias); only per-layer gate parameters train, on TRAINING conflict documents
(doc label = the dataset's counterfactual answer) mixed with normal-task training items (gold label, do-no-harm).
  m2   g_l                         (dense learned gates, init = base gates; L2 to the base)
  m5   g_l + β_l · c(x)            c = total-variation distance between the closed-tail answer distribution (the
                                   reader's own prior from tail + question) and the base read's answer distribution,
                                   both at base gates (two extra no-grad passes); β init 0, L1 penalty.
Forced-choice CE on the answer letter after "<answer>" (MCQ only). Selection on the val split of the same sets
(never test). Eval: forced-choice accuracy / prior-adherence / kv−wrong per set at base vs learned gates.
kv numbers here are UNREGISTERED (step 0 of a registration curve) unless the interface was registered.

  python scripts/d25_gates.py --mode m5 --train data/d25_cqa_split:600,data/d25_fecf_split:400,data/shakespeare_cf_split:600,data/quality_split:600 \
      --val ...:split=val --test ... --out outputs/D25_m5_9b
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import log  # noqa: E402
from d25_sweep import NORMAL, set_items  # noqa: E402

ANS_PREFIX = "<answer>"


def extend_interface(itf, enc, reader, extra, rank, runtime):
    """New D13Interface over base ∪ extra reader layers (extra paired with encoder layers by depth). Base layers keep
    their A / MLP / B / bias / gate; each extra layer starts as a copy of the deepest base layer's A / MLP / B / bias
    with gate 0 (so step 0 == the base read), plus zero-init rank-r deltas (only the extra layers' deltas train)."""
    import copy as _copy
    if runtime == "d12p":
        from xenc.d12p_read import D13Interface
    else:
        from xenc.d13_read import D13Interface
    from xenc.xattn_read import attn_dims, full_depth_map
    full = full_depth_map(attn_dims(reader), attn_dims(enc))
    lmap = dict(itf.layer_map)
    src = max(itf.layers)
    for li in extra:
        lmap[li] = full[li]
    B = {li: (itf.B[str(li)] if li in itf.layer_map else itf.B[str(src)]).detach().cpu().clone() for li in lmap}
    new = D13Interface(enc, lmap, itf.d_r, itf.d_e, B=B, mlp_hidden=itf.mlp_hidden)
    with torch.no_grad():
        for li in new.layers:
            s_ = str(li if li in itf.layer_map else src)
            new.A[str(li)].load_state_dict(_copy.deepcopy(itf.A[s_].state_dict()))
            if itf.mlp_hidden:
                new.M[str(li)].load_state_dict(_copy.deepcopy(itf.M[s_].state_dict()))
            new.Bb[str(li)].copy_(itf.Bb[s_].detach().cpu())
            new.g[new._gi[li]] = float(itf.g[itf._gi[li]]) if li in itf.layer_map else 0.0
    new = new.to(itf.g.device)
    new.add_delta(rank)
    return new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--interface", required=True)
    ap.add_argument("--runtime", default="d13", choices=["d13", "d12p"])
    ap.add_argument("--mode", default="m5", choices=["m2", "m5"])
    ap.add_argument("--train", required=True, help="split_dir[:n[:q10]] list (train split)")
    ap.add_argument("--val", required=True, help="same format (val split)")
    ap.add_argument("--test", default="", help="same format (test split); evaluated once with the selected gates")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--l1", type=float, default=0.01)
    ap.add_argument("--l2", type=float, default=0.1)
    ap.add_argument("--normal_p", type=float, default=0.5, help="share of normal-task items per step")
    ap.add_argument("--eval_every", type=int, default=150)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=60000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enc_device", default=None)
    ap.add_argument("--reader_mem", default=None)
    ap.add_argument("--itf_device", default=None)
    ap.add_argument("--extra_layers", default="", help="M2-deep: add reader layers (e.g. 35,39,43,47 for the 27B "
                    "context-reading band) to the read; init = copy of the deepest base layer's A/MLP/B, gate 0, "
                    "rank-r deltas on A/B of the NEW layers train with the gates")
    ap.add_argument("--extra_rank", type=int, default=32)
    ap.add_argument("--lr_delta", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.runtime == "d12p":
        from fit_d12p import D13Rt, build
    else:
        from fit_d13 import D13Rt, build
    from eval_doc import wrong_doc_id
    from xenc.doc_tasks import load_docs
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    reader, enc, tok, itf, lmap = build(a, init=a.interface)
    extra = [int(x) for x in a.extra_layers.split(",") if x]
    if extra:
        itf = extend_interface(itf, enc, reader, extra, a.extra_rank, a.runtime)
    rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens)
    itf.requires_grad_(False)
    dparams = []
    if extra:
        for k, p_ in itf.D.items():
            if int(k[2:]) in extra:
                p_.requires_grad_(True)
                dparams.append(p_)
        log(f"M2-deep: extra layers {extra} -> layer_map {itf.layer_map}; {sum(p_.numel() for p_ in dparams)} delta params")
    ansp = [int(x) for x in tok(ANS_PREFIX, add_special_tokens=False)["input_ids"]]
    letter = {L: int(tok(L, add_special_tokens=False)["input_ids"][0]) for L in "ABCDE"}
    layers = list(itf.layers)
    dev = itf.g.device
    g0 = itf.g.detach().clone()
    g = torch.nn.Parameter(g0.clone())
    beta = torch.nn.Parameter(torch.zeros_like(g0))
    GE = {"v": g0}
    itf.gate = lambda li: GE["v"][itf._gi[li]]           # per-forward effective gates (instance override)

    def load(specs, split):
        out = []
        for s in [x for x in specs.split(",") if x]:
            name, sd, items = set_items(s, split, a.seed)
            out.append((name, sd, items))
        return out
    tr, va = load(a.train, "train"), load(a.val, "val")
    te = load(a.test, "test") if a.test else []
    docs = {}
    for name, sd, _ in tr + va + te:
        docs.update(load_docs(sd))
    rt.docs = docs

    def fwd(did, it, gates, grad=False):
        GE["v"] = gates
        R = rt.R
        read = bool((gates.detach().abs() > 0).any())
        if read:
            op, n_op, _ = rt.operator(did)
            R.load(op, n_op)
            R.mode = "student"
        try:
            with torch.set_grad_enabled(grad):
                rows = rt.rows(did, it) + ansp
                out = reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=False, logits_to_keep=1)
                labs = [c["label"] for c in it["choices"]]
                lg = out.logits[0, -1, [letter[x] for x in labs]].float()
        finally:
            R.mode, R.past = "off", {}
        return torch.log_softmax(lg, -1)

    cache_c = {}

    def conflict_score(did, it):
        k = (did, it["key"])
        if k not in cache_c:
            p0 = fwd(did, it, torch.zeros_like(g0)).exp()
            p1 = fwd(did, it, g0).exp()
            cache_c[k] = float(0.5 * (p0 - p1).abs().sum())
        return cache_c[k]

    def gates_for(did, it, grad=False):
        if a.mode == "m2":
            return g if grad else g.detach()
        c = conflict_score(did, it)
        v = g + beta * c
        return v if grad else v.detach()

    def evaluate(sets, tag, learned=True):
        res, rows = {}, []
        for name, sd, items in sets:
            order = sorted({i["doc_id"] for i in items})
            m = defaultdict(list)
            for it in sorted(items, key=lambda i: i["doc_id"]):
                did = it["doc_id"]
                w = wrong_doc_id(did, order) if len(order) > 1 else None
                labs = [c["label"] for c in it["choices"]]
                gv = gates_for(did, it) if learned else g0
                lp = fwd(did, it, gv)
                pred = labs[int(lp.argmax())]
                pri = (it.get("meta") or {}).get("prior_label")
                m["acc"].append(float(pred == it["correct_label"]))
                if pri:
                    m["prior"].append(float(pred == pri))
                if w is not None:
                    gw = gates_for(w, it) if learned else g0
                    m["wrong"].append(float(labs[int(fwd(w, it, gw).argmax())] == it["correct_label"]))
                rows.append({"set": name, "key": it["key"], "pred": pred, "gold": it["correct_label"], "prior": pri,
                             "c": cache_c.get((did, it["key"])), "gates": [round(float(x), 4) for x in gv]})
            res[name] = {k: sum(v) / len(v) for k, v in m.items()}
            res[name]["n"] = len(items)
        conf = [v["acc"] for k, v in res.items() if not k.startswith(NORMAL) and not k.endswith("_orig")]
        norm = [v["acc"] for k, v in res.items() if k.startswith(NORMAL)]
        res["_U"] = sum(conf) / max(len(conf), 1) + sum(norm) / max(len(norm), 1)
        # selection score: conflict (kv − wrong_doc) so a document-independent anti-prior bias does not pay
        cw = [v["acc"] - v.get("wrong", 0.0) for k, v in res.items()
              if not k.startswith(NORMAL) and not k.startswith("_") and not k.endswith("_orig")]
        res["_S"] = sum(cw) / max(len(cw), 1) + sum(norm) / max(len(norm), 1)
        log(f"  [{tag}] U={res['_U']:.4f} " + " | ".join(
            f"{k} acc {v['acc']:.3f} prior {v.get('prior', float('nan')):.3f} wrong {v.get('wrong', float('nan')):.3f}"
            for k, v in sorted(res.items()) if not k.startswith("_")))
        return res, rows

    params = [g] + ([beta] if a.mode == "m5" else [])
    opt = torch.optim.Adam([{"params": params, "lr": a.lr}] + ([{"params": dparams, "lr": a.lr_delta}] if dparams else []))
    conf_pool = [(n, it) for n, _, its in tr if not n.startswith(NORMAL) for it in its]
    norm_pool = [(n, it) for n, _, its in tr if n.startswith(NORMAL) for it in its]
    rng = random.Random(a.seed)
    hist = []
    base_val, _ = evaluate(va, "val base", learned=False)
    best = {"U": base_val["_U"], "S": base_val["_S"], "step": 0, "g": g0.tolist(), "beta": [0.0] * len(layers)}
    best_D = {k: p_.detach().clone() for k, p_ in itf.D.items()} if dparams else {}
    hist.append({"step": 0, "val": base_val})
    t0 = time.time()
    for step in range(1, a.steps + 1):
        pool = norm_pool if (norm_pool and rng.random() < a.normal_p) else conf_pool
        name, it = rng.choice(pool)
        labs = [c["label"] for c in it["choices"]]
        opt.zero_grad()
        try:
            gv = gates_for(it["doc_id"], it, grad=True)
            lp = fwd(it["doc_id"], it, gv, grad=True)
            loss = -lp[labs.index(it["correct_label"])] + a.l2 * (g - g0).pow(2).sum() + a.l1 * beta.abs().sum()
            if dparams:
                loss = loss + 0.1 * itf.delta_reg()
            loss.backward()
        except torch.OutOfMemoryError:                    # long-document item: skip it (logged)
            opt.zero_grad()
            loss = torch.zeros(())
            torch.cuda.empty_cache()
            log(f"  OOM skip step {step} ({name} {it['doc_id']})")
            continue
        opt.step()
        if step % 25 == 0:
            log(f"step {step} loss {float(loss):.4f} g {[round(float(x), 3) for x in g]} "
                f"beta {[round(float(x), 3) for x in beta]} | {(time.time() - t0) / step:.2f}s/step")
        if step % a.eval_every == 0 or step == a.steps:
            vr, _ = evaluate(va, f"val step {step}")
            hist.append({"step": step, "val": vr, "g": g.tolist(), "beta": beta.tolist()})
            if vr["_S"] > best["S"]:
                best = {"U": vr["_U"], "S": vr["_S"], "step": step, "g": g.detach().tolist(),
                        "beta": beta.detach().tolist()}
                best_D = {k: p_.detach().clone() for k, p_ in itf.D.items()} if dparams else {}
            json.dump({"args": vars(a), "best": best, "hist": hist}, open(os.path.join(a.out, "gates.json"), "w"), indent=1)
    with torch.no_grad():
        g.copy_(torch.tensor(best["g"], device=dev))
        beta.copy_(torch.tensor(best["beta"], device=dev))
        for k, v in best_D.items():                      # M2-deep: the deltas of the SELECTED step
            itf.D[k].copy_(v)
    torch.save({"best": best, "D": {k: v.cpu() for k, v in best_D.items()}, "layer_map": itf.layer_map},
               os.path.join(a.out, "best_state.pt"))
    log(f"selected step {best['step']} U={best['U']:.4f} g {best['g']} beta {best['beta']}")
    if a.mode == "m2" and not extra:                     # static learned gates -> a D13-format interface file
        from xenc.d13_read import save_d13
        del itf.gate                                     # back to the module's own gate()
        with torch.no_grad():
            itf.g.copy_(g.detach())
        save_d13(itf, os.path.join(a.out, "d13_m2.pt"), {"stage": "d25_m2", "base": a.interface, "best": best})
        itf.gate = lambda li: GE["v"][itf._gi[li]]
    out = {"args": vars(a), "best": best, "hist": hist}
    if te:
        out["test_base"], rb = evaluate(te, "test base", learned=False)
        out["test_learned"], rl = evaluate(te, "test learned")
        out["test_rows"] = {"base": rb, "learned": rl}
    json.dump(out, open(os.path.join(a.out, "gates.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
