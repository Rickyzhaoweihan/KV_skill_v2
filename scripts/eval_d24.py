"""D24 judge: amortized registration (H(operator) -> adapter, one forward) of the D13 read, 4B encoder -> 9B reader.

Same items / seed / prompts as scripts/eval_d13.py and scripts/register_d20.py (judge_items). Conditions (greedy):
  closed_tail  doc tail + question, no read                 text      9B with the document in context
  kv_base      base interface, no adapter (= D20 kv@0)       kv        base + H(own operator)       (kv_H)
  wrong_doc    ANOTHER document's operator AND its H-adapter (tail of that document too, as in eval_d13)
  kv_Hswap     (diagnostic) own operator read through the WRONG document's H-adapter: kv − kv_Hswap = the value
               of H's document conditioning
  kv_reg       (optional, --d20 <eval json>) D20's per-document registered kv rows on the same items (copied by key)
The 9B never reads the test document's text in kv / wrong_doc: before it reads, only the 4B encoder prefill and one H
forward run. Cost per document is timed (CUDA-synchronised): 4B prefill -> operator, H forward, and, for comparison,
a 9B prefill of the same document prefix (what the reader's own prefix cache would cost).

--reuse <eval json> copies closed_tail / text / kv_base rows (matched by item key; deterministic greedy) from an
earlier D24 or eval_d13 run (eval_d13's "kv" = kv_base when it used the same base interface).

    python scripts/eval_d24.py --h outputs/D24_h1/d24_best.pt --split_dir data/quality_split --n 300 --out outputs/D24_h1_q
"""

from __future__ import annotations

import argparse
import json
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
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--interface", default="outputs/D13_4b_reg_w10/d13_best.pt", help="base interface")
    ap.add_argument("--h", required=True, help="d24 file (train_d24.py)")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--conds", default="closed_tail,text,kv_base,kv,wrong_doc")
    ap.add_argument("--reuse", default="", help="eval json to copy closed_tail/text/kv_base rows from (by key)")
    ap.add_argument("--reuse_kv_as_base", type=int, default=1, help="an eval_d13 json's 'kv' row is kv_base")
    ap.add_argument("--d20", default="", help="D20 eval json (register_d20.py) -> kv_reg@<--d20_budget> rows by key")
    ap.add_argument("--d20_budget", default="200")
    ap.add_argument("--time_docs", type=int, default=20, help="documents timed for the cost table")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wrong_map", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from eval_d13 import summarize
    from eval_doc import wrong_doc_id
    from fit_d13 import D13Rt, build
    from register_state import greedy
    from train_d24 import HRun
    from xenc.backbone import build_chat
    from xenc.d24_hyper import attach, load_h
    from xenc.doc_tasks import ctx_prompt, load_docs, score
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    reader, enc, tok, itf, lmap = build(a, init=a.interface)
    if itf.delta_rank:
        itf.merge_delta()
    itf.requires_grad_(False)
    H, hx = load_h(a.h, itf)
    H = H.to(itf.g.device).eval().requires_grad_(False)
    attach(itf, H)
    rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens)
    hr = HRun(rt, H)
    docs = load_docs(a.split_dir)
    rt.docs = docs
    items = load_split_items(a.split_dir, a.split)
    doc_order = sorted({_get(i, "doc_id") for i in items})
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
    if a.n and len(items) > a.n:
        items = random.Random(a.seed).sample(items, a.n)
    wmap = json.load(open(a.wrong_map)) if a.wrong_map else {}
    # leak guard: H's training documents never include a judged document
    targs = (hx.get("args") or {})
    log(f"D24 eval: {a.encoder} -> {a.reader} | base {a.interface} | H {a.h} (step {hx.get('step')}, cfg {H.cfg}) | "
        f"{len(items)} items / {len({_get(i, 'doc_id') for i in items})} docs | H trained on {targs.get('train')} "
        f"+val {targs.get('train_from_val')} pool_p {targs.get('pool_p')}")
    conds = [c for c in a.conds.split(",") if c]
    reuse = {}
    if a.reuse:
        RJ = json.load(open(a.reuse))
        gpu_now = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        same_gpu = (RJ.get("cost") or {}).get("gpu") == gpu_now      # kv_base is paired with kv: same GPU type only
        log(f"reuse {a.reuse}: closed_tail/text rows; kv_base {'reused' if same_gpu else 'recomputed (other GPU type)'}")
        for r in RJ["items"]:
            x = {c: r[c] for c in ("closed_tail", "text", "closed_book") + (("kv_base",) if same_gpu else ()) if c in r}
            if "kv_base" not in x and a.reuse_kv_as_base and "kv" in r and "kv_base" not in r:
                x["kv_base"] = r["kv"]
            reuse[r["key"]] = x
    d20 = {}
    if a.d20:
        for r in json.load(open(a.d20))["items"]:
            if f"kv@{a.d20_budget}" in r:
                d20[r["key"]] = r[f"kv@{a.d20_budget}"]

    # ---------------- cost per document (timed on the first --time_docs judged documents)
    cost = {"enc_prefill_s": [], "h_forward_s": [], "reader_text_prefill_s": [], "n_doc_tokens": []}
    tdocs = sorted({_get(i, "doc_id") for i in items})
    tdocs = tdocs[:1] + tdocs[: a.time_docs]            # the first document once more, untimed: CUDA warm-up
    for j, did in enumerate(tdocs):
        rt._ops.pop(did, None)
        _sync()
        t = time.time()
        op, n_op, _ = rt.operator(did)
        _sync()
        cost["enc_prefill_s"].append(time.time() - t)
        t = time.time()
        with torch.no_grad():
            H(op)
        _sync()
        cost["h_forward_s"].append(time.time() - t)
        rt.R.mode = "off"
        t = time.time()
        with torch.no_grad():
            reader(input_ids=torch.tensor([rt.pre(did)], device=rt.dev), use_cache=True, logits_to_keep=1)
        _sync()
        cost["reader_text_prefill_s"].append(time.time() - t)
        cost["n_doc_tokens"].append(n_op)
        if j == 0:
            for v in cost.values():
                v.pop()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    cs = {k: sum(v) / max(len(v), 1) for k, v in cost.items()}
    cs.update({f"{k}_median": sorted(v)[len(v) // 2] for k, v in cost.items() if v})
    cs["per_doc"] = cost
    cs["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cs["n_timed"] = len(cost["n_doc_tokens"])
    cs["h_params"], cs["adapter_numbers"] = H.n_params(), H.adapter_numel()
    log(f"cost/doc: 4B prefill {cs['enc_prefill_s']:.3f}s + H {cs['h_forward_s'] * 1e3:.1f}ms vs 9B text prefill "
        f"{cs['reader_text_prefill_s']:.3f}s ({cs['n_doc_tokens']:.0f} tokens, {cs['gpu']})")

    rows, t0 = [], time.time()
    order = sorted(range(len(items)), key=lambda k: _get(items[k], "doc_id"))
    res = {}
    ck = open(os.path.join(a.out, f"eval_{a.split}.ckpt.jsonl"), "w")
    for k in order:
        it = items[k]
        did = _get(it, "doc_id")
        w = wmap[did] if did in wmap else wrong_doc_id(did, doc_order)
        rec = {"key": _get(it, "key"), "doc_id": did, "wrong": w}
        ru = reuse.get(rec["key"], {})
        for c in conds:
            if c in ru:
                rec[c] = ru[c]
                continue
            if c == "text":
                g, n = greedy(reader, tok, a.reader, build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, docs[did])), a.budget)
            elif c == "closed_tail":
                g, n = hr.gen(did, it, a.budget, read=False)
            elif c == "kv_base":
                g, n = hr.gen(did, it, a.budget, use_h=False)
            elif c == "kv":
                g, n = hr.gen(did, it, a.budget)
            elif c == "kv_Hswap":                 # own operator, the WRONG document's H-adapter (diagnostic)
                if w is None:
                    continue
                g, n = hr.gen(did, it, a.budget, adapter_from=w)
            elif c == "wrong_doc":
                if w is None:
                    continue
                g, n = hr.gen(w, it, a.budget)
            else:
                raise ValueError(c)
            rec[c] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
        if rec["key"] in d20:
            rec["kv_reg"] = d20[rec["key"]]
        res[k] = rec
        rows.append(rec)
        ck.write(json.dumps(rec) + "\n")
        ck.flush()
        if len(rows) % 20 == 0:
            s = summarize(rows, conds)
            log(f"  [{len(rows)}/{len(items)}] " + " ".join(f"{c}={s[c]['score']:.3f}" for c in conds)
                + f" | {(time.time() - t0) / len(rows):.1f}s/item")
    rows = [res[k] for k in range(len(items))]
    allc = conds + (["kv_reg"] if d20 else [])
    summ = summarize(rows, allc)
    for ref in ("kv_base", "kv_reg", "kv_Hswap"):   # paired kv_H − kv_base / kv_reg / kv_Hswap
        pair = [r["kv"]["score"] - r[ref]["score"] for r in rows if "kv" in r and ref in r]
        if pair:
            mu = sum(pair) / len(pair)
            sd = (sum((p - mu) ** 2 for p in pair) / max(len(pair) - 1, 1)) ** 0.5
            summ[f"kv_minus_{ref}"] = {"paired_gap": mu, "se": sd / len(pair) ** 0.5, "n_paired": len(pair),
                                       "wins": sum(p > 0 for p in pair), "losses": sum(p < 0 for p in pair)}
    for ref in ("text", "wrong_doc", "closed_tail"):   # the same gaps for kv_base (and kv_reg), paired
        for base in ("kv_base", "kv_reg"):
            pair = [r[base]["score"] - r[ref]["score"] for r in rows if base in r and ref in r]
            if pair:
                mu = sum(pair) / len(pair)
                sd = (sum((p - mu) ** 2 for p in pair) / max(len(pair) - 1, 1)) ** 0.5
                summ[f"{base}_minus_{ref}"] = {"paired_gap": mu, "se": sd / len(pair) ** 0.5, "n_paired": len(pair),
                                               "wins": sum(p > 0 for p in pair), "losses": sum(p < 0 for p in pair)}
    json.dump({"summary": summ, "cost": cs, "args": vars(a), "h_extra": {"step": hx.get("step"), "val": hx.get("val")},
               "items": rows}, open(os.path.join(a.out, f"eval_{a.split}.json"), "w"), indent=1)
    log(f"==== eval_d24 [{a.split}] n={len(rows)} -> {a.out}")
    for c in allc:
        if c in summ:
            log(f"  {c:12s} {summ[c]['score']:.4f}  n={summ[c]['n']}  gen {summ[c]['mean_gen_tokens']:.1f}")
    for k_, v in summ.items():
        if "_minus_" in k_:
            log(f"  {k_}: {v['paired_gap']:+.4f} ± {v['se']:.4f} (n={v['n_paired']}, {v['wins']}/{v['losses']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
