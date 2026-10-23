"""D24 diagnostic: how document-dependent is H's output? For each H file, over N documents (QuALITY test + LH test:
operators only, no reader pass, no labels): between-document relative spread of M_l(doc) (‖M − mean_doc M‖ / ‖mean M‖),
the pairwise cosine of the operator summary features, and of the trunk code h.

    python scripts/d24_hvar.py --h outputs/D24_h1/d24_best.pt,outputs/D24_h3/d24_best.pt --out outputs/D24_hvar.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--interface", default="outputs/D13_4b_reg_w10/d13_best.pt")
    ap.add_argument("--h", required=True)
    ap.add_argument("--n_docs", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from fit_d13 import D13Rt, build
    from register_doc import log
    from xenc.d24_hyper import load_h
    from xenc.doc_tasks import load_docs
    reader, enc, tok, itf, lmap = build(a, init=a.interface)
    rt = D13Rt(reader, enc, tok, a.reader, itf, 64, 16000, lru=1)
    docs = load_docs("data/quality_split")
    dids = sorted(docs)[: a.n_docs]
    rt.docs = docs
    ops = {}
    Hs = {p: load_h(p, itf)[0].to(itf.g.device).eval() for p in a.h.split(",")}
    res = {p: {"M": [], "f": [], "h": []} for p in Hs}
    for d in dids:
        op = rt.operator(d)[0]
        with torch.no_grad():
            for p, H in Hs.items():
                out = H(op)
                res[p]["M"].append(torch.cat([out["M"][li].flatten() for li in H.layers]).cpu())
                feats = H.summary(op)
                res[p]["f"].append(torch.cat([feats[e] for e in sorted(feats)]).cpu())
                hs = [H.proj[str(li)](feats[H.layer_map[li]]) for li in H.layers]
                res[p]["h"].append(H.trunk(torch.cat(hs)).cpu())
                if a.debug:
                    f0 = feats[H.layer_map[H.layers[0]]]
                    pre = H.trunk[0](torch.cat(hs))
                    log(f"  [debug {d}] feat0 norm {float(f0.norm()):.3g} max|.| {float(f0.abs().max()):.3g} "
                        f"finite {bool(torch.isfinite(f0).all())} | proj0 norm {float(hs[0].norm()):.4g} first {hs[0][:3].tolist()} | "
                        f"trunk0 pre mean {float(pre.mean()):.3g} std {float(pre.std()):.3g} | M0[:3] {out['M'][H.layers[0]].flatten()[:3].tolist()}")
        rt._ops.clear()
    rep = {}
    for p, r in res.items():
        M = torch.stack(r["M"])
        mu = M.mean(0)
        spread = float((M - mu).norm(dim=1).mean() / mu.norm().clamp_min(1e-12))
        def pcos(X):
            Xn = torch.nn.functional.normalize(X - 0 * X.mean(0), dim=1)
            C = Xn @ Xn.t()
            n = C.shape[0]
            return float((C.sum() - n) / (n * (n - 1)))
        def pcos_c(X):
            Xc = X - X.mean(0)
            return float(Xc.norm(dim=1).mean() / X.mean(0).norm().clamp_min(1e-12))
        D = M - mu                                            # per-document deviation H(op_i) − mean_doc H(op)
        Dn = torch.nn.functional.normalize(D, dim=1)
        Cd = Dn @ Dn.t()
        nd = Cd.shape[0]
        Mn = torch.nn.functional.normalize(M, dim=1)
        Cm = Mn @ Mn.t()
        rep_dev = {"dev_norm_mean": float(D.norm(dim=1).mean()), "dev_norm_max": float(D.norm(dim=1).max()),
                   "dev_pairwise_cos": float((Cd.sum() - nd) / (nd * (nd - 1))),
                   "M_pairwise_cos": float((Cm.sum() - nd) / (nd * (nd - 1))),
                   "dev_norms": [round(float(x), 6) for x in D.norm(dim=1)]}
        rep[p] = {**rep_dev, "M_between_doc_rel_spread": spread, "M_mean_norm": float(mu.norm()),
                  "feat_pairwise_cos": pcos(torch.stack(r["f"])), "feat_rel_spread": pcos_c(torch.stack(r["f"])),
                  "code_pairwise_cos": pcos(torch.stack(r["h"])), "code_rel_spread": pcos_c(torch.stack(r["h"])),
                  "n_docs": len(dids)}
        log(f"{p}: {rep[p]}")
    json.dump(rep, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
