"""D8 phase 2: accuracy vs retained bytes of query-independent column pruning of the reader's native document KV.

Built on D7's native-state evaluator (scripts/eval_state.py; xenc/native_state.py: harvest_states,
build_prefix_cache, greedy_continue). The document is prefilled ONCE as raw tokens; the reader continues with
the closed-book chat prompt from a cache holding a compressed part of its own document memory.

Reference conditions: closed_book, text, all (full KV + DeltaNet states), kv_only, wrong_kv, wrong_all.
Pruned conditions  `<regime>_<layers>_<score><pct>` and `wrong_<...>` (another document, pruned the same way):
  regime : kvs = KV + DeltaNet recurrent + conv states (the 0.81 regime)  | kv = KV only (the 0.575 regime)
           kvc = KV + conv states (no recurrent)
  layers : E = early full-attention layers only (first half; D7: late-layer KV carries nothing), A = all 8
  score  : acc (attention received during the document prefill) | mean (acc / #queries seeing it) | rand
  pct    : retained % of document positions per kept layer and KV head (100 = no column pruning)
Retained bytes per condition = KV columns actually readable (Σ_ℓ b_ℓ · n_kv · hd · 2 · bytes) + state bytes.

    python scripts/eval_prune.py --split_dir data/longhealth_split --split test --fold 0 --out outputs/D8_prune_lh
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
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
import torch  # noqa: E402

import xenc  # noqa: E402,F401

REGIMES = {"kvs": (True, True), "kv": (False, False), "kvc": (False, True)}    # (keep_state, keep_conv)


def parse_cond(c: str):
    """-> dict(wrong, regime, layers, score, pct) for pruned conditions, None for reference ones."""
    w = c.startswith("wrong_")
    body = c[6:] if w else c
    parts = body.split("_")
    if len(parts) != 3 or parts[0] not in REGIMES or parts[1] not in ("E", "A"):
        return None
    sp = parts[2]
    score = next((s for s in ("acc", "mean", "rand") if sp.startswith(s)), None)
    if score is None:
        if sp != "100":
            return None
        score, pct = "acc", 100.0
    else:
        pct = float(sp[len(score):])
    return {"wrong": w, "regime": parts[0], "layers": parts[1], "score": score, "pct": pct}


def default_conds(regimes, layers, scores, pcts) -> list[str]:
    out = ["closed_book", "text", "all", "kv_only", "wrong_kv", "wrong_all"]
    for r in regimes:
        for L in layers:
            for w in ("", "wrong_"):
                out.append(f"{w}{r}_{L}_100")
            for p in pcts:
                for s in scores:
                    for w in ("", "wrong_"):
                        out.append(f"{w}{r}_{L}_{s}{p:g}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--regimes", default="kvs,kv")
    ap.add_argument("--layers", default="E")
    ap.add_argument("--scores", default="acc,mean,rand")
    ap.add_argument("--pcts", default="25,10,5")
    ap.add_argument("--conds", default=None, help="explicit comma list (overrides the grid)")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--max_doc_tokens", type=int, default=0)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--doc_cache", type=int, default=6, help="documents whose full memory stays resident")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from register_doc import _get, load_split_items, system_prompt
    from xenc.backbone import build_chat, load_reader
    from xenc.doc_tasks import ctx_prompt, load_docs, q_prompt, score
    from xenc.kv_prune import ColumnBudget, harvest_with_importance, prune_states
    from xenc.native_state import build_prefix_cache, greedy_continue, state_nbytes

    pcts = [float(x) for x in a.pcts.split(",") if x.strip()]
    conds = ([c.strip() for c in a.conds.split(",") if c.strip()] if a.conds else
             default_conds(a.regimes.split(","), a.layers.split(","), a.scores.split(","), pcts))
    docs = load_docs(a.split_dir)
    items = load_split_items(a.split_dir, a.split)
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
    doc_ids = sorted({_get(i, "doc_id") for i in items if _get(i, "doc_id") in docs})
    if a.max_docs:
        doc_ids = doc_ids[: a.max_docs]
    items = [i for i in items if _get(i, "doc_id") in set(doc_ids)]
    if a.n and len(items) > a.n:
        items = random.Random(a.seed).sample(items, a.n)
    # same wrong-doc rule as eval_state.py
    wrong = {d: doc_ids[(k + 7) % len(doc_ids)] if doc_ids[(k + 7) % len(doc_ids)] != d
             else doc_ids[(k + 1) % len(doc_ids)] for k, d in enumerate(doc_ids)}
    # group items by document so each document is harvested once (own) and reused as a wrong doc
    items.sort(key=lambda i: (_get(i, "doc_id"), str(_get(i, "key"))))
    print(f"{len(items)} items over {len(doc_ids)} docs; {len(conds)} conds: {conds}", flush=True)

    model, tok = load_reader(a.reader, device_map=a.device)
    hub = a.reader
    mem, pruned, sizes = {}, {}, {}

    def memory(d):
        if d not in mem:
            st, imp = harvest_with_importance(model, tok, docs[d], max_tokens=a.max_doc_tokens, block=a.block)
            mem[d] = (st, imp)
            sizes[d] = {**state_nbytes(st), "n_doc": st["n_doc"]}
            while len(mem) > a.doc_cache:
                old = next(iter(mem))
                mem.pop(old)
                for k_ in [k_ for k_ in pruned if k_[0] == old]:
                    pruned.pop(k_)
        return mem[d]

    def pruned_mem(d, spec):
        key = (d, spec["layers"], spec["score"], spec["pct"])
        if key not in pruned:
            st, imp = memory(d)
            full = sorted(st["kv"])
            late = full[len(full) // 2:] if spec["layers"] == "E" else []
            budgets = {li: 0.0 for li in late}
            stp, info = prune_states(model, st, imp, frac=spec["pct"] / 100.0, score=spec["score"],
                                     budgets=budgets or None, seed=a.seed, min_keep=0)
            pruned[key] = (stp, info)
        return pruned[key]

    sysp = system_prompt("answer")
    os.makedirs(a.out, exist_ok=True)
    ck_path = os.path.join(a.out, "eval_prune.ckpt.jsonl")
    done = {}
    if os.path.exists(ck_path):
        for ln in open(ck_path):
            if ln.strip():
                r = json.loads(ln)
                if all(c in r for c in conds):
                    done[r["key"]] = r
    ck = open(ck_path, "a")
    rows, t0 = [], time.time()
    bytes_of: dict = {}
    for k, it in enumerate(items):
        if _get(it, "key") in done:
            rows.append(done[_get(it, "key")])
            continue
        did = _get(it, "doc_id")
        rec = {"key": _get(it, "key"), "doc_id": did, "wrong_doc": wrong[did]}
        q_ids = build_chat(tok, hub, sysp, q_prompt(it))
        for c in conds:
            spec = parse_cond(c)
            budget_ctx = None
            if c == "closed_book":
                cache, ids = build_prefix_cache(model, None), q_ids
            elif c == "text":
                cache, ids = build_prefix_cache(model, None), build_chat(tok, hub, sysp, ctx_prompt(it, docs[did]))
            elif c in ("all", "kv_only", "wrong_kv", "wrong_all"):
                st, _ = memory(wrong[did] if c.startswith("wrong") else did)
                s = c in ("all", "wrong_all")
                cache, ids = build_prefix_cache(model, st, True, s, s), q_ids
                nb = state_nbytes(st)
                bytes_of.setdefault(c, []).append(nb["kv"] + (nb["recurrent"] + nb["conv"] if s else 0))
            elif spec is not None:
                src = wrong[did] if spec["wrong"] else did
                stp, info = pruned_mem(src, spec)
                ks, kc = REGIMES[spec["regime"]]
                cache, ids = build_prefix_cache(model, stp, True, ks, kc), q_ids
                budget_ctx = ColumnBudget(model, info["col_budget"], info["b_max"])
                K0 = next(iter(stp["kv"].values()))[0]
                col_bytes = K0.shape[1] * K0.shape[3] * 2 * K0.element_size()
                nb = state_nbytes(stp)
                bytes_of.setdefault(c, []).append(sum(info["col_budget"].values()) * col_bytes
                                                  + (nb["recurrent"] if ks else 0) + (nb["conv"] if kc else 0))
                rec.setdefault("_kept_frac", {})[c] = info["kept_frac"]
            else:
                raise SystemExit(f"unknown condition {c}")
            if budget_ctx is not None:
                with budget_ctx:
                    g, n = greedy_continue(model, tok, cache, ids, a.budget, hub=hub)
            else:
                g, n = greedy_continue(model, tok, cache, ids, a.budget, hub=hub)
            rec[c] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
            del cache
        rows.append(rec)
        ck.write(json.dumps(rec) + "\n")
        ck.flush()
        if (k + 1) % 10 == 0:
            show = [c for c in conds if not c.startswith("wrong_")][:12]
            print(f"  [{k + 1}/{len(items)}] " + " ".join(
                f"{c}={sum(r[c]['score'] for r in rows) / len(rows):.3f}" for c in show)
                + f" | {(time.time() - t0) / max(1, len(rows)):.1f}s/item", flush=True)
    ck.close()

    def paired(a_, b_):
        d = [r[a_]["score"] - r[b_]["score"] for r in rows]
        m = sum(d) / len(d)
        se = (sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / len(d)) ** 0.5
        return {"gap": m, "se": se}

    summ = {c: {"score": sum(r[c]["score"] for r in rows) / len(rows),
                "gen_tokens": sum(r[c]["gen_tokens"] for r in rows) / len(rows), "n": len(rows),
                "bytes": (sum(bytes_of[c]) / len(bytes_of[c])) if bytes_of.get(c) else None} for c in conds}
    gaps = {}
    for c in conds:
        if c.startswith("wrong_") and c[6:] in summ:
            gaps[f"{c[6:]}-{c}"] = paired(c[6:], c)
    gaps["kv_only-wrong_kv"] = paired("kv_only", "wrong_kv") if "kv_only" in summ and "wrong_kv" in summ else None
    gaps["all-wrong_all"] = paired("all", "wrong_all") if "all" in summ and "wrong_all" in summ else None
    if "text" in summ:
        for c in conds:
            if c not in ("text", "closed_book") and not c.startswith("wrong_"):
                gaps[f"{c}-text"] = paired(c, "text")
    mean_sz = {k: sum(v[k] for v in sizes.values()) / max(1, len(sizes)) for k in ("recurrent", "conv", "kv", "n_doc")}
    res = {"summary": summ, "gaps": gaps, "mean_full_bytes": mean_sz, "n": len(rows), "conds": conds, "args": vars(a),
           "items": rows}
    json.dump(res, open(os.path.join(a.out, "eval_prune.json"), "w"), indent=1)
    print(f"==== n={len(rows)} ({len(doc_ids)} docs) -> {a.out}/eval_prune.json")
    print(f"  full memory: kv {mean_sz['kv'] / 2**20:.1f} MiB, recurrent {mean_sz['recurrent'] / 2**20:.1f} MiB, "
          f"conv {mean_sz['conv'] / 2**20:.2f} MiB, n_doc {mean_sz['n_doc']:.0f}")
    for c in conds:
        if c.startswith("wrong_"):
            continue
        s = summ[c]
        w = f"wrong_{c}" if f"wrong_{c}" in summ else ("wrong_kv" if c == "kv_only" else ("wrong_all" if c == "all" else None))
        gap = paired(c, w) if w else None
        mb = "" if s["bytes"] is None else f"{s['bytes'] / 2**20:8.2f} MiB"
        print(f"  {c:16s} {s['score']:.4f} {mb}" + (f"  wrong {summ[w]['score']:.4f}  gap {gap['gap']:+.4f} ± {gap['se']:.4f}"
                                                    if gap else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
