"""D7: fit per-depth linear maps from an ENCODER's raw attention K/V space to a READER's (cross-model interface).

    U' = M_ρ U + c_ρ  (encoder keys  [n_kv_e·hd] -> reader keys  [n_kv_r·hd])
    W' = N_ρ W + d_ρ  (encoder values                -> reader values)

by ridge regression on token-aligned slot pairs: Qwen3.5 sizes share the tokenizer and both caches are one
whole-document prefill chunked identically, so slot i is the same token in both. Pairs are streamed one
document at a time (only XᵀX and XᵀY are held). Fit quality = cosine(pred, true) per KV head (and R²)
on held-out documents (--eval_split).

    python scripts/fit_kv_map.py --split_dir data/quality_split --encoder Qwen/Qwen3.5-2B --reader Qwen/Qwen3.5-9B \
        --fit_split train --max_docs 40 --eval_split test --out outputs/D7_maps/kv_2B_to_9B.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

import xenc  # noqa: E402
from xenc.depth_skill import _key, cache_key, load_doc_cache  # noqa: E402


def _paths(split_dir, split, enc_short, rd_short, depths, cache_dir, chunk, max_docs, docs):
    from xenc.doc_tasks import load_doc_split
    ids = sorted({it.doc_id for it in load_doc_split(split_dir, split)})
    if max_docs:
        ids = ids[:max_docs]
    out = []
    for d in ids:
        pe = os.path.join(cache_dir, cache_key(docs[d], enc_short, chunk, depths, harvest="attn_raw") + ".pt")
        pr = os.path.join(cache_dir, cache_key(docs[d], rd_short, chunk, depths, harvest="attn_raw") + ".pt")
        if os.path.exists(pe) and os.path.exists(pr):
            out.append((d, pe, pr))
    return out


def _pairs(pe, pr, rho, which, max_slots, g):
    ce, cr = load_doc_cache(pe), load_doc_cache(pr)
    X = torch.cat([c[which][_key(c[which], rho)] for c in ce["chunks"]]).float()
    Y = torch.cat([c[which][_key(c[which], rho)] for c in cr["chunks"]]).float()
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"token counts differ ({X.shape[0]} vs {Y.shape[0]}): tokenizers not aligned")
    if max_slots and X.shape[0] > max_slots:
        i = torch.randperm(X.shape[0], generator=g)[:max_slots]
        X, Y = X[i], Y[i]
    return X, Y


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split_dir", default="data/quality_split")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--cache_dir", default=os.environ.get("XENC_CACHE", xenc.DEFAULT_CACHE))
    ap.add_argument("--depths", default=",".join(str(r) for r in xenc.DEPTHS))
    ap.add_argument("--chunk_tokens", type=int, default=512)
    ap.add_argument("--fit_split", default="train")
    ap.add_argument("--max_docs", type=int, default=40)
    ap.add_argument("--eval_split", default="test")
    ap.add_argument("--eval_docs", type=int, default=10)
    ap.add_argument("--slots_per_doc", type=int, default=0, help="subsample slots per doc (0 = all)")
    ap.add_argument("--lam", type=float, default=1e-3, help="ridge lambda relative to mean diag of XᵀX")
    ap.add_argument("--head_dim", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from xenc.doc_tasks import load_docs
    docs = load_docs(a.split_dir)
    depths = sorted(float(x) for x in a.depths.split(","))
    es, rs = a.encoder.split("/")[-1], a.reader.split("/")[-1]
    fit = _paths(a.split_dir, a.fit_split, es, rs, depths, a.cache_dir, a.chunk_tokens, a.max_docs, docs)
    ev = _paths(a.split_dir, a.eval_split, es, rs, depths, a.cache_dir, a.chunk_tokens, a.eval_docs, docs)
    print(f"fit docs {len(fit)} ({a.fit_split}), eval docs {len(ev)} ({a.eval_split})", flush=True)
    if not fit:
        print("no paired caches; encode both models with --harvest attn_raw", file=sys.stderr)
        return 2
    g = torch.Generator().manual_seed(0)
    maps, report = {}, {}
    for rho in depths:
        maps[rho] = {}
        for which, nm in (("K", "k"), ("V", "v")):
            XtX = XtY = None
            n = 0
            for _, pe, pr in fit:
                X, Y = _pairs(pe, pr, rho, which, a.slots_per_doc, g)
                X = torch.cat([X, torch.ones(X.shape[0], 1)], 1).double()
                Y = Y.double()
                XtX = X.T @ X if XtX is None else XtX + X.T @ X
                XtY = X.T @ Y if XtY is None else XtY + X.T @ Y
                n += X.shape[0]
            lam = a.lam * float(XtX.diagonal()[:-1].mean())
            reg = torch.eye(XtX.shape[0], dtype=XtX.dtype) * lam
            reg[-1, -1] = 0.0                                    # do not shrink the bias
            Wb = torch.linalg.solve(XtX + reg, XtY)              # [d_e+1, d_r]
            W, b = Wb[:-1].T.float().contiguous(), Wb[-1].float().contiguous()
            # held-out quality
            cos_h, cos_full, sse, sst, mu_y, nn_ = [], [], 0.0, 0.0, None, 0
            for _, pe, pr in (ev or fit[-5:]):
                X, Y = _pairs(pe, pr, rho, which, 0, g)
                P = X @ W.T + b
                H = Y.shape[1] // a.head_dim
                ph, yh = P.view(-1, H, a.head_dim), Y.view(-1, H, a.head_dim)
                cos_h.append(torch.nn.functional.cosine_similarity(ph, yh, dim=-1).mean(0))      # [H]
                cos_full.append(torch.nn.functional.cosine_similarity(P, Y, dim=-1).mean())
                sse += float(((P - Y) ** 2).sum())
                sst += float(((Y - Y.mean(0)) ** 2).sum())
                nn_ += Y.shape[0]
            ch = torch.stack(cos_h).mean(0)
            st = {"n_fit_pairs": n, "cos_head_mean": float(ch.mean()), "cos_per_head": [round(float(x), 4) for x in ch],
                  "cos_concat": float(torch.stack(cos_full).mean()), "r2": 1 - sse / max(sst, 1e-12),
                  "n_eval_pairs": nn_, "d_in": int(W.shape[1]), "d_out": int(W.shape[0])}
            maps[rho][f"W{nm}"], maps[rho][f"b{nm}"] = W, b
            maps[rho][f"{nm}_stats"] = st
            report[f"rho={rho}/{which}"] = st
            print(f"rho {rho} {which}: {W.shape[1]}->{W.shape[0]}  held-out cos/head {st['cos_head_mean']:.4f} "
                  f"{st['cos_per_head']}  cos {st['cos_concat']:.4f}  R2 {st['r2']:.4f}  ({n} fit pairs)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"maps": maps, "encoder": a.encoder, "reader": a.reader, "depths": depths, "lam": a.lam,
                "fit_docs": [d for d, _, _ in fit], "eval_docs": [d for d, _, _ in ev]}, a.out)
    json.dump(report, open(os.path.splitext(a.out)[0] + "_report.json", "w"), indent=1)
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
