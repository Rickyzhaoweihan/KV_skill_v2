"""H2 / KV-RAG summary — an INDEPENDENT recompute from raw per-item outputs (not the training job's own summary):
  KV-RAG    <h2>/eval_<split>.ckpt.jsonl (per-item generations' scores) + per-document early stop recomputed from
            <h2>/adapters/<doc>.pt ho_loss (register_d20 rule: the budget <= 200 with the lowest held-out self-q loss)
  flat kv / text / closed_book   the D22 run on the same items (<d22>/eval_<split>.json items + its adapters)
  text-RAG  <h2>/eval_textrag_*.jsonl
  recall    gold-evidence hit of the chunks KV-RAG actually read (<h2>/selections_eval.json) per item
Tables: kv-rag vs text / flat kv / text-RAG / wrong_doc (paired ± se), CLIPPER pair accuracy, length buckets,
evidence-hit vs miss.

    python scripts/h2_summary.py --set outputs/D22_sets/clipper.json --h2 outputs/H2_clipper --d22 outputs/D22_clipper \
        --retrieval outputs/H2_retrieval/clipper.json --name CLIPPER
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def es_choice(out_dir, docs, budgets, cap=200):
    import torch
    ch = {}
    for d in docs:
        ad = torch.load(os.path.join(out_dir, "adapters", f"{d}.pt"), map_location="cpu", weights_only=False)
        ho = {int(k): float(v) for k, v in (ad.get("ho_loss") or {}).items() if int(k) in budgets and int(k) <= cap}
        ch[d] = min(sorted(ho), key=lambda k: ho[k]) if ho else cap
    return ch


def load_rows(out_dir, split):
    p = os.path.join(out_dir, f"eval_{split}.ckpt.jsonl")
    rows = {}
    for line in open(p):
        r = json.loads(line)
        rows[r["key"]] = r
    return rows


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = sum(d) / n
    sd = (sum((x - mu) ** 2 for x in d) / max(n - 1, 1)) ** 0.5
    return mu, sd / math.sqrt(n), n


def fmt(mu, se):
    return f"{mu:+.3f}±{se:.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--h2", required=True, nargs="+", help="one or more KV-RAG run dirs (arms)")
    ap.add_argument("--d22", default=None, help="flat-kv (/ text) run on the same items; use the SAME GPU type as --h2 "
                    "(GPU-type matching rule): e.g. the D22 adapters re-evaluated on L40S")
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--name", default="")
    ap.add_argument("--budgets", default="0,25,50,100,200")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from h2_retrieval import recall_item
    S = json.load(open(a.set))
    keys = list(S["item_keys"])
    split_dir = S["split_dir"]
    items = {}
    for s in (S.get("splits") or [a.split]):
        for it in json.load(open(Path(split_dir) / s / "items.json")):
            if it["key"] in set(keys):
                items[it["key"]] = it
    assert len(items) == len(keys)
    idx = json.load(open(Path(split_dir) / "docs.json"))
    R = json.load(open(a.retrieval))
    eb = [int(x) for x in a.budgets.split(",")]
    docs_all = sorted({it["doc_id"] for it in items.values()})
    L = [f"## KV-RAG {a.name or S.get('task')} (n={len(keys)}; independent recompute, scripts/h2_summary.py)", ""]
    cols = {}
    # flat (D22)
    if a.d22:
        d22 = {r["key"]: r for r in json.load(open(os.path.join(a.d22, f"eval_{a.split}.json")))["items"]}
        wdocs = sorted({r["wrong"] for r in d22.values()} | set(docs_all))
        esf = es_choice(a.d22, wdocs, eb)
        for c in ("text", "closed_book", "closed_tail"):
            if all(c in d22[k] for k in keys):
                cols[c] = [float(d22[k][c]["score"]) for k in keys]
        cols["flat kv@0"] = [float(d22[k]["kv@0"]["score"]) for k in keys]
        cols["flat kv_reg"] = [float(d22[k][f"kv@{esf[d22[k]['doc_id']]}"]["score"]) for k in keys]
        cols["flat wrong"] = [float(d22[k][f"wrong_doc@{esf[d22[k]['wrong']]}"]["score"]) for k in keys]
    sel_info = {}
    for h in a.h2:
        tag = Path(h).name
        rows = load_rows(h, a.split)
        miss = [k for k in keys if k not in rows]
        if miss:
            L.append(f"**{tag}: {len(miss)} of {len(keys)} items missing — partial run, skipped**")
            continue
        wd = sorted({rows[k]["wrong"] for k in keys} | set(docs_all))
        es = es_choice(h, wd, eb)
        cols[f"{tag} kv@0"] = [float(rows[k]["kv@0"]["score"]) for k in keys]
        cols[f"{tag} kv_reg"] = [float(rows[k][f"kv@{es[rows[k]['doc_id']]}"]["score"]) for k in keys]
        cols[f"{tag} wrong"] = [float(rows[k][f"wrong_doc@{es[rows[k]['wrong']]}"]["score"]) for k in keys]
        for c in ("closed_tail", "text", "closed_book"):     # same-job (same GPU type) baseline rows
            if all(c in rows[k] for k in keys):
                cols[f"{tag} {c}"] = [float(rows[k][c]["score"]) for k in keys]
        try:
            import torch
            g = sorted({str(torch.load(os.path.join(h, "adapters", f"{d}.pt"), map_location="cpu",
                                       weights_only=False).get("gpu")) for d in docs_all[:3]})
            L.append(f"- {tag}: registration GPU {g}")
        except Exception:
            pass
        for p in sorted(glob.glob(os.path.join(h, "eval_textrag_*.jsonl"))):
            tr = {}
            for line in open(p):
                r = json.loads(line)
                tr[r["key"]] = r
            if all(k in tr for k in keys):
                cols[f"{tag} {Path(p).stem.replace('eval_', '')}"] = [float(tr[k]["text_rag"]["score"]) for k in keys]
        sp = os.path.join(h, "selections_eval.json")
        if os.path.exists(sp):
            sj = json.load(open(sp))
            sel_info[tag] = (sj.get("meta", {}), sj.get("sel", sj))
    # ---- main table
    L += ["| condition | score | n |", "|---|---|---|"]
    for c, v in cols.items():
        L.append(f"| {c} | {sum(v) / len(v):.3f} | {len(v)} |")
    L.append("")
    L += ["| comparison (paired) | gap ± se | n |", "|---|---|---|"]
    for h in a.h2:
        tag = Path(h).name
        if f"{tag} kv_reg" not in cols:
            continue
        kv = cols[f"{tag} kv_reg"]
        for other in [f"{tag} text", "text", "flat kv_reg", f"{tag} closed_book", "closed_book", f"{tag} wrong",
                      f"{tag} kv@0"] + \
                [c for c in cols if c.startswith(f"{tag} textrag")]:
            if other in cols:
                mu, se, n = paired(kv, cols[other])
                L.append(f"| {tag} kv_reg − {other} | {fmt(mu, se)} | {n} |")
        if "closed_tail" in cols and f"{tag} closed_tail" in cols:
            same = sum(x == y for x, y in zip(cols["closed_tail"], cols[f"{tag} closed_tail"]))
            L.append(f"| sanity: closed_tail H2 vs D22 identical items | {same}/{len(keys)} | |")
    L.append("")
    # ---- CLIPPER pair accuracy
    if str(S.get("task", "")).startswith("clipper"):
        pairs = {}
        for i, k in enumerate(keys):
            pairs.setdefault(items[k]["meta"]["pair_id"], []).append(i)
        pp = [v for v in pairs.values() if len(v) == 2]
        L += [f"**pair accuracy** ({len(pp)} pairs, chance .25)", "", "| condition | pair acc |", "|---|---|"]
        for c, v in cols.items():
            L.append(f"| {c} | {sum(v[i] * v[j] for i, j in pp) / len(pp):.3f} |")
        L.append("")
    # ---- length buckets
    bk = [(0, 64000, "<64k"), (64000, 96000, "64–96k"), (96000, 10 ** 9, "≥96k")]
    L += ["| bucket | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for lo, hi, nm in bk:
        ii = [i for i, k in enumerate(keys) if lo <= idx[items[k]["doc_id"]].get("n_tokens", 0) < hi]
        if ii:
            L.append(f"| {nm} | {len(ii)} | " + " | ".join(f"{sum(cols[c][i] for i in ii) / len(ii):.3f}" for c in cols)
                     + " |")
    L.append("")
    # ---- evidence hit vs miss of the chunks actually read
    ev = R.get("evidence", {})
    for tag, (meta, sel) in sel_info.items():
        C = meta.get("C")
        if C is None or str(C) not in R["chunks"]:
            continue
        ch = R["chunks"][str(C)]
        hit = {}
        for k in keys:
            q = f"{items[k]['doc_id']}|{k}"
            if ev.get(k) and q in sel:
                u, al, cov = recall_item(ev[k], ch[items[k]["doc_id"]], sel[q])
                hit[k] = (u, al, cov)
        if not hit:
            continue
        hk = [k for k in keys if k in hit and hit[k][0] > 0]
        mk = [k for k in keys if k in hit and hit[k][0] == 0]
        L += [f"**{tag}: gold evidence in the read chunks** ({meta}) — unit recall {sum(v[0] for v in hit.values()) / len(hit):.3f}, "
              f"all-units {sum(v[1] for v in hit.values()) / len(hit):.3f}, coverage {sum(v[2] for v in hit.values()) / len(hit):.3f} "
              f"over {len(hit)} items with evidence", "",
              "| subset | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        for nm, ks in (("evidence hit", hk), ("evidence missed", mk)):
            ii = [keys.index(k) for k in ks]
            if ii:
                L.append(f"| {nm} | {len(ii)} | " + " | ".join(f"{sum(cols[c][i] for i in ii) / len(ii):.3f}" for c in cols)
                         + " |")
        L.append("")
    md = "\n".join(L)
    print(md)
    if a.out:
        open(a.out, "w").write(md)
        json.dump({"keys": keys, "cols": cols}, open(a.out.replace(".md", ".json"), "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
