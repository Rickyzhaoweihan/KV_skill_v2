"""D12: non-token-wise write-back B for the encoder-side read — closed-form ridge from CONTEXT features.

For each reader full-attention layer ℓ (depth-matched encoder layer e), the target is the reader's own in-context
attention output a_ℓ[t] (level post: attention-module output) on a training document; the features are encoder
attention outputs from several encoder layers and/or a causal window of tokens (xenc/encside_ctx.py):
    m     : o_e[t]                                     (= D10's token-wise B, with intercept)
    n3    : o_{e'}[t] for the 3 encoder full-attention layers nearest e in depth order (e and its neighbours)
    w3    : o_e[t], o_e[t-1], o_e[t-2]
    n3w3  : n3 + o_e[t-1], o_e[t-2]
    a8    : o_{e'}[t] for EVERY encoder full-attention layer
Label-free (no answers used); fitted on train-split documents only (LongHealth folds in --exclude_folds removed).
Pass 1 prefills reader + encoder on each document and keeps the last --rows token rows (CPU, fp16); pass 2 frees
the models and solves one ridge per (layer, config), λ selected per layer on held-out-document R².
Each source block is RMS-normalised before the ridge (folded back into W). Output: one blob per config,
"xenc.encside_B2.v1", loadable by eval_encside.py --maps.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402


def config_src(cfg: str, enc_full: list, e: int) -> list:
    from xenc.encside_ctx import neighbour_src
    if cfg == "m":
        return [(e, 0)]
    if cfg == "n3":
        return neighbour_src(enc_full, e, 3)
    if cfg == "n2":
        return neighbour_src(enc_full, e, 2)
    if cfg == "w3":
        return [(e, 0), (e, 1), (e, 2)]
    if cfg == "n3w3":
        return neighbour_src(enc_full, e, 3, match_lags=[1, 2])
    if cfg == "a8":
        return [(e, 0)] + [(x, 0) for x in enc_full if x != e]
    raise ValueError(cfg)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--splits", default="quality:60,longhealth:12,musique:40,lastingbench:30")
    ap.add_argument("--exclude_folds", default="0,1")
    ap.add_argument("--layers", default="", help="reader full-attention layers (default: all)")
    ap.add_argument("--configs", default="m,n3,w3,n3w3")
    ap.add_argument("--max_tokens", type=int, default=4000)
    ap.add_argument("--rows", type=int, default=2048, help="last token rows kept per document")
    ap.add_argument("--heldout_frac", type=float, default=0.1)
    ap.add_argument("--lams", default="0.0003,0.003,0.03,0.3")
    ap.add_argument("--level", default="post", choices=["pre", "post"])
    ap.add_argument("--out_prefix", required=True, help="writes <prefix>_<cfg>.pt (+ _report.json)")
    ap.add_argument("--report", default=None, help="json with every (layer, cfg, λ) fidelity number")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    from xenc.backbone import build_chat, input_device, load_encoder, load_reader
    from xenc.doc_tasks import ctx_prompt, load_docs
    from xenc.encside_read import EncCapture
    from xenc.xattn_read import attn_dims, full_depth_map
    reader, tok = load_reader(a.reader, device_map=a.device)
    enc = load_encoder(a.encoder, device_map=a.device)[0]
    ra, ea = attn_dims(reader), attn_dims(enc)
    enc_full = list(ea["full"])
    dmap = full_depth_map(ra, ea)
    rl = [int(x) for x in a.layers.split(",")] if a.layers else list(ra["full"])
    amap = {li: dmap[li] for li in rl}
    cfgs = [c for c in a.configs.split(",") if c]
    log(f"D12 fit: {a.encoder} -> {a.reader} | pairs {amap} | configs {cfgs}")
    rcap, ecap = EncCapture(reader, rl, a.level), EncCapture(enc, enc_full, a.level)
    excl = {int(x) for x in a.exclude_folds.split(",") if x != ""}
    data = []           # [(split, {e: [n+2, d] fp16}, {li: [n, d] fp16})]
    for spec in a.splits.split(","):
        name, nd = spec.split(":")
        sd = f"data/{name}_split"
        docs = load_docs(sd)
        items = [i for i in load_split_items(sd, "train") if (_get(i, "meta") or {}).get("fold") not in excl]
        first = {}
        for it in items:
            first.setdefault(_get(it, "doc_id"), it)
        dids = sorted(first)[: int(nd)]
        n_held = max(1, int(round(len(dids) * a.heldout_frac)))
        for k, did in enumerate(dids):
            s = "held" if k >= len(dids) - n_held else "fit"
            ids = build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(first[did], docs[did]))[: a.max_tokens]
            T = len(ids)
            p0 = max(2, T - a.rows)
            x = torch.tensor([ids])
            rcap.start()
            ecap.start()
            with torch.no_grad():
                reader(input_ids=x.to(input_device(reader)), use_cache=False, logits_to_keep=1)
                enc(input_ids=x.to(input_device(enc)), use_cache=False, logits_to_keep=1)
                r, e = rcap.get(), ecap.get()
            data.append((s, {j: e[j][0, p0 - 2:].half().cpu() for j in enc_full},
                         {li: r[li][0, p0:].half().cpu() for li in rl}))
        log(f"  {name}: {len(dids)} docs ({n_held} held out)")
    rcap.detach()
    ecap.detach()
    del reader, enc, rcap, ecap
    gc.collect()
    torch.cuda.empty_cache()
    dev = torch.device(a.device)
    lams = [float(x) for x in a.lams.split(",")]
    out, rep_all = solve(data, rl, amap, enc_full, cfgs, lams, dev, a.level, vars(a))
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
    for c in cfgs:
        p = f"{a.out_prefix}_{c}.pt"
        torch.save(out[c], p)
        with open(f"{a.out_prefix}_{c}_report.json", "w") as f:
            json.dump(out[c]["report"], f, indent=1)
        log(f"saved {p}")
    if a.report:
        with open(a.report, "w") as f:
            json.dump(rep_all, f, indent=1)
    log("==== held-out fidelity (R² / cos) per layer")
    log("layer  " + "  ".join(f"{c:>13s}" for c in cfgs))
    for li in rl:
        log(f"L{li:<5d} " + "  ".join(f"{out[c]['report'][li]['r2_held']:.3f}/{out[c]['report'][li]['cos_held']:.3f}"
                                      .rjust(13) for c in cfgs))
    return 0


def solve(data, rl, amap, enc_full, cfgs, lams, dev, level, args):
    """data: [("fit"|"held", {enc_layer: [n+2, d_e]}, {reader_layer: [n, d_r]})] -> ({cfg: blob}, report)."""
    fit = [d for d in data if d[0] == "fit"]
    held = [d for d in data if d[0] == "held"]
    log(f"rows: fit {sum(d[2][rl[0]].shape[0] for d in fit)} ({len(fit)} docs), "
        f"held {sum(d[2][rl[0]].shape[0] for d in held)} ({len(held)} docs)")
    # per-encoder-layer RMS scale (fit rows)
    scale = {}
    for j in enc_full:
        ss = sum(float((d[1][j][2:].float() ** 2).sum()) for d in fit)
        nn_ = sum(d[1][j][2:].numel() for d in fit)
        scale[j] = (ss / nn_) ** 0.5
    srcs = {(li, c): config_src(c, enc_full, amap[li]) for li in rl for c in cfgs}
    out = {c: {"format": "xenc.encside_B2.v1", "level": level, "config": c, "layers": {}, "report": {},
               "args": args} for c in cfgs}
    rep_all = {}

    def feats(d, src, n):
        return torch.cat([d[1][j][2 - g: 2 - g + n].to(dev).float() / scale[j] for j, g in src], dim=-1)

    for li in rl:
        ys = sum(d[2][li].to(dev).float().sum(0) for d in held)     # SST around the held-out mean
        yy = sum(float((d[2][li].to(dev).float() ** 2).sum()) for d in held)
        nh_all = sum(d[2][li].shape[0] for d in held)
        sst = yy - float((ys * ys).sum()) / nh_all
        for c in cfgs:
            src = srcs[(li, c)]
            D, dy = None, None
            xx = xy = sx = sy = None
            n = 0
            for d in fit:
                Y = d[2][li].to(dev).float()
                X = feats(d, src, Y.shape[0])
                if xx is None:
                    D, dy = X.shape[1], Y.shape[1]
                    xx = torch.zeros(D, D, dtype=torch.float64, device=dev)
                    xy = torch.zeros(D, dy, dtype=torch.float64, device=dev)
                    sx = torch.zeros(D, dtype=torch.float64, device=dev)
                    sy = torch.zeros(dy, dtype=torch.float64, device=dev)
                xx += (X.t() @ X).double()
                xy += (X.t() @ Y).double()
                sx += X.sum(0).double()
                sy += Y.sum(0).double()
                n += X.shape[0]
                del X, Y
            mx, my = sx / n, sy / n
            Cxx = xx - n * torch.outer(mx, mx)
            Cxy = xy - n * torch.outer(mx, my)
            del xx, xy
            tr = torch.trace(Cxx) / D
            res, Ws = {}, {}
            for lm in lams:
                A = Cxx + lm * tr * torch.eye(D, dtype=Cxx.dtype, device=dev)
                W = torch.linalg.solve(A, Cxy).float()
                del A
                sse = cs = 0.0
                nh = 0
                for d in held:
                    Y = d[2][li].to(dev).float()
                    P = (feats(d, src, Y.shape[0]) - mx.float()) @ W + my.float()
                    sse += float(((P - Y) ** 2).sum())
                    cs += float(torch.nn.functional.cosine_similarity(P, Y, dim=-1).sum())
                    nh += Y.shape[0]
                    del P, Y
                res[lm] = {"r2_held": 1 - sse / sst, "cos_held": cs / nh}
                Ws[lm] = W
            best = max(lams, key=lambda lm: res[lm]["r2_held"])
            W = Ws[best]
            svec = torch.cat([torch.full((ea_dim,), scale[j], device=dev) for j, _ in src
                              for ea_dim in [d_in_of(data, j)]])
            out[c]["layers"][str(li)] = {"src": [[int(j), int(g)] for j, g in src], "match": int(amap[li]),
                                         "W": (W / svec[:, None]).t().contiguous().to(torch.bfloat16).cpu(),
                                         "mx": (mx.float() * svec).cpu(), "my": my.float().cpu()}
            out[c]["report"][li] = {"lam": best, **res[best], "D": D}
            rep_all[f"{li}:{c}"] = {str(k): v for k, v in res.items()}
            log(f"  L{li}<-E{amap[li]} [{c:5s} D={D}] λ {best:g}: held R² {res[best]['r2_held']:.3f} "
                f"cos {res[best]['cos_held']:.3f}  (grid " + " ".join(f"{k:g}:{v['r2_held']:.3f}" for k, v in res.items()) + ")")
            del Ws, W, Cxx, Cxy
            torch.cuda.empty_cache()
    return out, rep_all


def d_in_of(data, j):
    return int(data[0][1][j].shape[1])


if __name__ == "__main__":
    sys.exit(main())
