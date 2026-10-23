"""Fit the interface query map A (d_reader -> d_encoder) by ridge regression on token-aligned slot pairs.

Both encoders use the same tokenizer and chunking, so slot i of a document encoded by the READER and
slot i encoded by the ENCODER are the same token. We fit  A · L2(h_reader) ≈ L2(h_encoder - mu)  (mu =
the encoder's centering vector when --center is given), pooled over depths, streaming one document at a
time (login-node safe: only X^T X [d_r,d_r] and X^T Y [d_r,d_e] are held).

    python scripts/fit_query_map.py --reader_short Qwen3.5-9B --encoder_short Qwen3.5-2B --task quality \
        --center outputs/centers/center_Qwen3.5-2B.pt --out outputs/centers/A_Qwen3.5-9B_to_Qwen3.5-2B.pt
For L/L pass the same short name twice (then A maps the reader's raw states onto its centered states).
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xenc  # noqa: F401
import torch


def _l2(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


def load_index(cache_dir, task, short):
    p = os.path.join(cache_dir, f"index_{task}_split_{short}.json")
    if os.path.exists(p):
        idx = json.load(open(p))
        return {k: (v if os.path.isabs(v) else os.path.join(cache_dir, os.path.basename(v))) for k, v in idx.items()}
    # fallback: glob by name and read doc_id from each file (slow)
    out = {}
    for f in glob.glob(os.path.join(cache_dir, f"*_{short}_c*_d*.pt")):
        try:
            out[torch.load(f, map_location="cpu", weights_only=True)["doc_id"]] = f
        except Exception:
            pass
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default=xenc.DEFAULT_CACHE)
    ap.add_argument("--reader_short", required=True); ap.add_argument("--encoder_short", required=True)
    ap.add_argument("--task", default="quality")
    ap.add_argument("--center", default=None, help="encoder centering file; targets become L2(h - mu)")
    ap.add_argument("--lam", type=float, default=1e-2, help="ridge lambda (relative to mean diag of X^T X)")
    ap.add_argument("--max_docs", type=int, default=40); ap.add_argument("--slots_per_doc", type=int, default=1500)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ri = load_index(a.cache_dir, a.task, a.reader_short); ei = load_index(a.cache_dir, a.task, a.encoder_short)
    docs = sorted(set(ri) & set(ei))[: a.max_docs]
    if not docs:
        print("no paired documents found", file=sys.stderr); return 2
    center = None
    if a.center:
        center = {float(k): v for k, v in torch.load(a.center, map_location="cpu", weights_only=True)["center"].items()}
    XtX = XtY = None; n = 0; g = torch.Generator().manual_seed(0)
    for i, did in enumerate(docs):
        cr = torch.load(ri[did], map_location="cpu", weights_only=True); ce = torch.load(ei[did], map_location="cpu", weights_only=True)
        if len(cr["chunks"]) != len(ce["chunks"]):
            print(f"skip {did}: chunk count differs ({len(cr['chunks'])} vs {len(ce['chunks'])})"); continue
        for rho in sorted(float(r) for r in cr["depths"]):
            Hr = torch.cat([c["K"][rho] for c in cr["chunks"]], 0).float()
            He = torch.cat([c["K"][rho] for c in ce["chunks"]], 0).float()
            m = min(Hr.shape[0], He.shape[0]); Hr, He = Hr[:m], He[:m]
            if a.slots_per_doc and m > a.slots_per_doc:
                idx = torch.randperm(m, generator=g)[: a.slots_per_doc]; Hr, He = Hr[idx], He[idx]
            X = _l2(Hr)                                            # reader states (unit rows, like RMSNorm up to scale)
            Y = _l2(He - center[rho]) if center is not None else He  # encoder keys as the Library sees them
            XtX = X.T @ X if XtX is None else XtX + X.T @ X
            XtY = X.T @ Y if XtY is None else XtY + X.T @ Y
            n += X.shape[0]
        del cr, ce
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(docs)}] pairs {n}", flush=True)
    lam = a.lam * float(XtX.diagonal().mean())
    A = torch.linalg.solve(XtX + lam * torch.eye(XtX.shape[0]), XtY)   # [d_r, d_e]
    W = A.T.contiguous()                                                # nn.Linear weight: [d_e, d_r]
    # fit quality on the training pairs (cosine between prediction and target)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"A_weight": W, "d_reader": W.shape[1], "d_encoder": W.shape[0], "n_pairs": n, "docs": docs,
                "reader_short": a.reader_short, "encoder_short": a.encoder_short, "center": a.center, "lam": a.lam}, a.out)
    print(f"wrote {a.out}: A {tuple(W.shape)} from {n} pairs over {len(docs)} docs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
