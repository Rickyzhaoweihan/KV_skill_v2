"""D25 D1: per-layer causal sweep of the paper-form read (base interface, UNREGISTERED = diagnostic / step 0).

The read replaces the reader's attention output at the interface layers with gate g_l (out + g_l·(read − out)).
A layer subset S is realised by g_l = g_base_l for l in S and 0 elsewhere (S = ∅ is closed_tail). Configs:
all, none, single {l}, leave-one-out, and greedy forward selection (on a per-set subsample) by the utility
U = mean(conflict doc-adherence) + accuracy on the normal task, each config scored on the own document and on a
wrong document. Scoring is forced-choice: the letter logits after the reader prompt (doc tail + question) +
"<answer>" (MCQ sets only). Also stores, for the "all" and "none" configs, the logit lens of the kv pass
(answer letters per layer), for the M5 conflict score.

  python scripts/d25_sweep.py --reader Qwen/Qwen3.5-9B --encoder Qwen/Qwen3.5-4B \
      --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19 --interface outputs/D13_4b_reg_w10/d13_best.pt \
      --sets data/shakespeare_cf_split,data/d25_cqa_split,data/d25_fecf_split,data/quality_split:300:q10 \
      --split val --out outputs/D25_sweep_9b
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

ANS_PREFIX = "<answer>"
NORMAL = ("quality", "longhealth")


def set_items(spec, split, seed):
    """split_dir[:n[:q10]] ; q10 = QuALITY-style: skip the first 10 sorted val docs (base selection docs)."""
    parts = spec.split(":")
    sd, n = parts[0], int(parts[1]) if len(parts) > 1 and parts[1] else 0
    items = json.load(open(os.path.join(sd, split, "items.json")))
    if len(parts) > 2 and parts[2] == "q10":
        docs = sorted({i["doc_id"] for i in items})[10:]
        items = [i for i in items if i["doc_id"] in set(docs)]
    want_orig = len(parts) > 2 and parts[2] == "orig"        # [D25] ORIGINAL-document control items only
    items = [i for i in items if i.get("choices")
             and (((i.get("meta") or {}).get("variant", "cf") == "orig") == want_orig)]
    if n and len(items) > n:
        items = random.Random(seed).sample(items, n)
    name = Path(sd.rstrip("/")).name.removesuffix("_split")
    return (name + "_orig" if want_orig else name), sd, items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--interface", required=True)
    ap.add_argument("--runtime", default="d13", choices=["d13", "d12p"])
    ap.add_argument("--sets", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=60000)
    ap.add_argument("--greedy_n", type=int, default=100, help="items per set for greedy forward selection")
    ap.add_argument("--no_greedy", action="store_true")
    ap.add_argument("--cfgs", default="", help="restrict the fixed configs (comma list; smoke tests)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enc_device", default=None)
    ap.add_argument("--reader_mem", default=None)
    ap.add_argument("--itf_device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.runtime == "d12p":
        from fit_d12p import D13Rt, build
    else:
        from fit_d13 import D13Rt, build
    from eval_doc import wrong_doc_id
    from xenc.d25_probe import LayerTap, lens_logits
    from xenc.doc_tasks import load_docs
    os.makedirs(a.out, exist_ok=True)
    reader, enc, tok, itf, lmap = build(a, init=a.interface)
    rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens)
    tap = LayerTap(reader)
    tap.on = False
    ansp = [int(x) for x in tok(ANS_PREFIX, add_special_tokens=False)["input_ids"]]
    letter = {L: int(tok(L, add_special_tokens=False)["input_ids"][0]) for L in "ABCDE"}
    g0 = itf.g.detach().clone()
    layers = list(itf.layers)
    log(f"D25 sweep: {a.encoder} -> {a.reader} | layers {lmap} | base gates {[round(float(x), 3) for x in g0]}")

    sets = [set_items(s, a.split, a.seed) for s in a.sets.split(",")]
    docs = {}
    for name, sd, _ in sets:
        docs.update(load_docs(sd))
    rt.docs = docs

    @torch.no_grad()
    def score(did, it, S, lens=False):
        """forced-choice letter log-probs with the read at layer subset S."""
        mask = torch.tensor([1.0 if li in S else 0.0 for li in layers], device=g0.device)
        itf.g.data.copy_(g0 * mask)
        R = rt.R
        if S:
            op, n_op, _ = rt.operator(did)
            R.load(op, n_op)
            R.mode = "student"
        tap.on = lens
        try:
            rows = rt.rows(did, it) + ansp
            out = reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=False, logits_to_keep=1)
        finally:
            R.mode, R.past = "off", {}
            tap.on = False
        labs = [c["label"] for c in it["choices"]]
        lg = out.logits[0, -1, [letter[x] for x in labs]].float()
        lp = torch.log_softmax(lg, -1)
        r = {"lp": [round(float(v), 4) for v in lp]}
        if lens:
            r["lens"] = [[round(float(v), 3) for v in row]
                         for row in lens_logits(reader, tap.stack(), [letter[x] for x in labs])]
        return r

    def metrics(rows, cfg):
        """per set: acc, doc-adh (= acc on conflict items), prior-adh, mean log p(gold), kv−wrong."""
        out = {}
        for name in {r["set"] for r in rows}:
            rs = [r for r in rows if r["set"] == name and cfg in r["cfg"]]
            if not rs:
                continue
            m = defaultdict(list)
            for r in rs:
                labs = r["labels"]
                c = r["cfg"][cfg]
                pred = labs[max(range(len(labs)), key=lambda k: c["lp"][k])]
                m["acc"].append(float(pred == r["gold"]))
                m["lp_gold"].append(c["lp"][labs.index(r["gold"])])
                if r.get("prior") in labs:
                    m["prior"].append(float(pred == r["prior"]))
                if "w" in c:
                    wp = labs[max(range(len(labs)), key=lambda k: c["w"][k])]
                    m["wrong"].append(float(wp == r["gold"]))
            out[name] = {k: sum(v) / len(v) for k, v in m.items()}
            out[name]["n"] = len(rs)
        return out

    def utility(mt):
        conf = [v["acc"] for k, v in mt.items() if not k.startswith(NORMAL)]
        norm = [v["acc"] for k, v in mt.items() if k.startswith(NORMAL)]
        return (sum(conf) / max(len(conf), 1)) + (sum(norm) / max(len(norm), 1))

    fixed = {"all": set(layers), "none": set()}
    for li in layers:
        fixed[f"only{li}"] = {li}
        fixed[f"loo{li}"] = set(layers) - {li}
    if a.cfgs:
        fixed = {k: v for k, v in fixed.items() if k in a.cfgs.split(",")}
    rows = []
    t0 = time.time()
    ck = open(os.path.join(a.out, f"sweep_{a.split}.ckpt.jsonl"), "w")
    for name, sd, items in sets:
        order = sorted({i["doc_id"] for i in items})
        by = defaultdict(list)
        for it in items:
            by[it["doc_id"]].append(it)
        for did in order:
            w = wrong_doc_id(did, order)
            for it in by[did]:
                m = it.get("meta") or {}
                r = {"key": it["key"], "set": name, "doc_id": did, "wrong": w, "gold": it["correct_label"],
                     "prior": m.get("prior_label"), "labels": [c["label"] for c in it["choices"]], "cfg": {}}
                for cfg, S in fixed.items():
                    c = score(did, it, S, lens=cfg in ("all", "none"))
                    if cfg != "none" and w is not None:
                        c["w"] = score(w, it, S)["lp"]
                    r["cfg"][cfg] = c
                rows.append(r)
                ck.write(json.dumps(r) + "\n")
                ck.flush()
            log(f"  {name} {did}: {len(rows)} rows | {(time.time() - t0) / len(rows):.2f}s/item")
    res = {cfg: metrics(rows, cfg) for cfg in fixed}
    greedy = []
    if not a.no_greedy:
        sub = []
        for name, _, items in sets:
            keys = {i["key"] for i in random.Random(a.seed + 1).sample(items, min(a.greedy_n, len(items)))}
            sub += [r for r in rows if r["set"] == name and r["key"] in keys]
        lookup = {(name, it["key"]): it for name, _, items in sets for it in items}
        cur, rem = [], list(layers)
        while rem:
            best = None
            for li in rem:
                S = set(cur) | {li}
                cfg = "g:" + ",".join(str(x) for x in sorted(S))
                for r in sub:
                    if cfg not in r["cfg"]:
                        r["cfg"][cfg] = score(r["doc_id"], lookup[(r["set"], r["key"])], S)
                u = utility(metrics(sub, cfg))
                if best is None or u > best[0]:
                    best = (u, li, cfg)
            cur.append(best[1])
            rem.remove(best[1])
            greedy.append({"add": best[1], "set": sorted(cur), "U": best[0], "metrics": metrics(sub, best[2])})
            log(f"  greedy +L{best[1]} -> {sorted(cur)} U={best[0]:.4f}")
        res["greedy_sub_all"] = metrics(sub, "all")
        res["greedy_sub_none"] = metrics(sub, "none")
    json.dump({"args": vars(a), "layers": layers, "base_gates": [float(x) for x in g0], "metrics": res,
               "greedy": greedy, "rows": rows}, open(os.path.join(a.out, f"sweep_{a.split}.json"), "w"))
    for cfg, mt in res.items():
        log(f"  {cfg:10s} " + " | ".join(f"{k}: acc {v['acc']:.3f} prior {v.get('prior', float('nan')):.3f} "
                                         f"wrong {v.get('wrong', float('nan')):.3f}" for k, v in sorted(mt.items())))
    itf.g.data.copy_(g0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
