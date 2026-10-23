"""D24 DIAGNOSTIC: headroom of the per-document adapter family (D20 dA + gate) on QuALITY when it is given the
RIGHT signal — real questions about the document (an oracle; never a method, never on judged test documents).

Documents: QuALITY val docs after the first 10 (sorted; D13's held-out val docs are the first 10), the same 47 docs and
the SAME question split as train_d24.py --q_holdout (random.Random(4321), per doc in sorted order). Each document is
registered with D20's runtime (xenc/d20_reg.py DocAdapter, register_d20.py D20.register: KL(9B text teacher ‖ student)
+ 0.5·CE(teacher answer) + 1.0·encoder-attention distillation) on its 70% "seen" real questions, then the 30% held-out
questions of the same document are judged at every budget (kv@0 = base). If even this oracle does not beat kv@0, an
amortized H in this family cannot close the QuALITY gap.

    python scripts/d24_oracle.py --out outputs/D24_oracle_q
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import _get, log  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--interface", default="outputs/D13_4b_reg_w10/d13_best.pt")
    ap.add_argument("--skip_docs", type=int, default=10)
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--q_holdout", type=float, default=0.3)
    ap.add_argument("--parts", default="dA,gate")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--budgets", default="0,50,100,200")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--target", default="teacher", choices=("teacher", "gold"),
                    help="CE target: the text teacher's greedy answer (D20) or the gold answer (stronger oracle)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    from register_d20 import D20
    from fit_d13 import select_items
    from xenc.doc_tasks import gold_text, load_docs, score
    os.makedirs(a.out, exist_ok=True)
    ns = argparse.Namespace(reader=a.reader, encoder=a.encoder, B=a.B, layers=a.layers, interface=a.interface,
                            runtime="d13", tail=64, max_tokens=16000, mem_safe=1, parts=a.parts, rank=a.rank, kv_rank=8,
                            seed=0, budgets=a.budgets, holdout=0.0, lr=a.lr, lr_gate=3e-3, lr_bias=None, warmup=25,
                            batch=2, clip=1.0, ans_budget=40, conf=0.0, dist_w=1.0, kl=1.0, ce=0.5, rel_w=1.0, l2=0.0,
                            device=a.device)
    S = D20(ns)
    S.rt.docs = load_docs("data/quality_split")
    ex, _ = select_items("quality:0", "val", None, 0, 0)
    keep = sorted({_get(i, "doc_id") for i in ex})[a.skip_docs:]
    if a.max_docs:
        keep = keep[: a.max_docs]
    hrng = random.Random(4321)                       # == train_d24.py --q_holdout split (same doc order)
    budgets = sorted({int(b) for b in a.budgets.split(",")})
    rows, regs = [], {}
    for k, d in enumerate(sorted({_get(i, "doc_id") for i in ex})[a.skip_docs:]):
        qs = [i for i in ex if _get(i, "doc_id") == d]
        hrng.shuffle(qs)
        nh = int(round(a.q_holdout * len(qs)))
        if d not in keep:
            continue
        held, seen = qs[:nh], qs[nh:]
        if a.target == "gold":                        # recitation-style path of D20.targets: fixed target text
            seen = [dict(i, meta=dict(i.get("meta") or {}, cont=gold_text(i).replace("<answer>", "").replace("</answer>", "")))
                    for i in seen]
        r = S.register(d, seen)
        regs[d] = {"n_items": r["n_items"], "t_reg_s": r["t_reg_s"], "t_teacher_s": r["t_teacher_s"]}
        for it in held:
            rec = {"key": _get(it, "key"), "doc_id": d}
            for b in budgets:
                g, n = S.gen(d, it, r["snaps"][b], 32)
                rec[f"kv@{b}"] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
            rows.append(rec)
        m = {b: sum(x[f"kv@{b}"]["score"] for x in rows) / len(rows) for b in budgets}
        log(f"  [{k + 1 - a.skip_docs if False else len(regs)}/{len(keep)}] {d}: {len(seen)} seen / {len(held)} held-out q | "
            f"running held-out kv " + " ".join(f"@{b} {v:.3f}" for b, v in m.items()))
    summ = {}
    for b in budgets:
        pr = [x[f"kv@{b}"]["score"] - x[f"kv@{budgets[0]}"]["score"] for x in rows]
        mu = sum(pr) / len(pr)
        sd = (sum((p - mu) ** 2 for p in pr) / max(len(pr) - 1, 1)) ** 0.5
        summ[str(b)] = {"kv": sum(x[f"kv@{b}"]["score"] for x in rows) / len(rows), "minus_base": mu,
                        "se": sd / len(pr) ** 0.5, "wins": sum(p > 0 for p in pr), "losses": sum(p < 0 for p in pr)}
    json.dump({"args": vars(a), "summary": summ, "regs": regs, "items": rows}, open(os.path.join(a.out, "oracle.json"), "w"), indent=1)
    log(f"==== D24 oracle (real seen questions -> held-out questions, same docs) n={len(rows)} docs={len(regs)}")
    for b, s in summ.items():
        log(f"  kv@{b}: {s['kv']:.4f}  minus kv@0 {s['minus_base']:+.4f} ± {s['se']:.4f} ({s['wins']}/{s['losses']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
