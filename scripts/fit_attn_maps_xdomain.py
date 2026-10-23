"""D10: multi-domain ridge fit of the cross-model ATTENTION-read maps (2B -> 9B), with a cross-domain R² matrix.

For every domain (a data/<name>_split dir) the 9B reader and the 2B encoder read the same `text` prompts
(identical tokenizers, token-aligned). At reader full-attention layer ℓ (depth-paired encoder layer ℓ') and reader
q-head h (encoder q-head h' = h·Hq_e//Hq_r):  A_ℓ,h : q9_h -> q2_h' (pre-RoPE),  B_ℓ,h : o2_h' -> o9_h
(attention outputs, pre-gate/o_proj). Gram matrices per (domain, fit|held). Reports per layer the held-out R² of
(i) the pooled fit on every domain and (ii) each single-domain fit on every domain (in- vs cross-domain), then saves
the POOLED maps in the fit_state_maps format (attn_A / attn_B) for register_state.py / eval_state_xm.py --init.

    python scripts/fit_attn_maps_xdomain.py --splits quality:40,longhealth:12,musique:30,lastingbench:30 \
        --exclude_folds 0,1 --layers 3,7,11,15 --out outputs/D10_maps/maps_9b_2b_xdomain.pt
"""

from __future__ import annotations

import argparse
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
from fit_state_maps import FORMAT, Gram, ridge  # noqa: E402
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402


def merged(gs):
    g0 = gs[0]
    out = Gram(g0.xx.shape[0], g0.xy.shape[1], g0.xx.device)
    for g in gs:
        out.xx += g.xx
        out.xy += g.xy
        out.yy += g.yy
        out.ys += g.ys
        out.n += g.n
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--splits", default="quality:40,longhealth:12,musique:30,lastingbench:30",
                    help="name:max_docs list over data/<name>_split (train split)")
    ap.add_argument("--exclude_folds", default="0,1", help="LongHealth folds kept out (test / val)")
    ap.add_argument("--layers", default="3,7,11,15")
    ap.add_argument("--max_tokens", type=int, default=9000)
    ap.add_argument("--heldout_frac", type=float, default=0.15)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    from xenc.backbone import build_chat, input_device, load_encoder, load_reader
    from xenc.doc_tasks import ctx_prompt, load_docs
    from xenc.xattn_read import AttnReader, attn_dims, full_depth_map, make_attn_interface

    reader, tok = load_reader(a.reader, device_map=a.device)
    enc = load_encoder(a.encoder, device_map=a.device)[0]
    ra, ea = attn_dims(reader), attn_dims(enc)
    keep = {int(x) for x in a.layers.split(",")} if a.layers else set(ra["full"])
    amap = {k: v for k, v in full_depth_map(ra, ea).items() if k in keep}
    Ar = AttnReader(reader, make_attn_interface(reader, ra, layers=sorted(amap)), record_mag=False)
    Ae = AttnReader(enc, make_attn_interface(enc, ea, layers=sorted(set(amap.values()))), record_mag=False)
    hmap = [h * ea["n_q"] // ra["n_q"] for h in range(ra["n_q"])]
    dev = input_device(reader)
    hd, H = ra["hd"], ra["n_q"]
    excl = {int(x) for x in a.exclude_folds.split(",") if x != ""}
    domains, G, ntok = [], {}, {}
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
        domains.append(name)
        G[name] = {s: {li: [(Gram(hd, hd, dev), Gram(hd, hd, dev)) for _ in range(H)] for li in amap}
                   for s in ("fit", "held")}
        ntok[name] = {"fit": 0, "held": 0, "docs": len(dids), "held_docs": n_held}
        for k, did in enumerate(dids):
            s = "held" if k >= len(dids) - n_held else "fit"
            ids = build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(first[did], docs[did]))[: a.max_tokens]
            x = torch.tensor([ids])
            Ar.capture, Ae.capture = {}, {}
            with torch.no_grad():
                reader(input_ids=x.to(dev), use_cache=False, logits_to_keep=1)
                enc(input_ids=x.to(input_device(enc)), use_cache=False, logits_to_keep=1)
            for li, e in amap.items():
                qr, orr = Ar.capture[li]
                qe, oe = Ae.capture[e]
                for h, h2 in enumerate(hmap):
                    ga, gb = G[name][s][li][h]
                    ga.add(qr[0, h].float().to(dev), qe[0, h2].float().to(dev))
                    gb.add(oe[0, h2].float().to(dev), orr[0, h].float().to(dev))
            Ar.capture = Ae.capture = None
            ntok[name][s] += len(ids)
        log(f"  {name}: {ntok[name]}")
    eye = torch.eye(hd, dtype=torch.float64, device=dev)
    rep = {"tokens": ntok, "layers": {}}
    AA, AB = {}, {}
    for li in amap:
        WA, WB = [], []
        r = {"pooled": {}, "single": {}}
        fitsA = {d: [] for d in domains}
        fitsB = {d: [] for d in domains}
        for h in range(H):
            pa = merged([G[d]["fit"][li][h][0] for d in domains])
            pb = merged([G[d]["fit"][li][h][1] for d in domains])
            wa, wb = ridge(pa, a.lam), ridge(pb, a.lam)
            WA.append(wa.t().float().cpu())
            WB.append(wb.t().float().cpu())
            for d in domains:
                fitsA[d].append(ridge(G[d]["fit"][li][h][0], a.lam))
                fitsB[d].append(ridge(G[d]["fit"][li][h][1], a.lam))
            for d in domains:
                ha, hb = G[d]["held"][li][h]
                r["pooled"].setdefault(d, []).append((ha.r2(wa), hb.r2(wb)))
        for d in domains:
            v = r["pooled"][d]
            r["pooled"][d] = {"A": sum(x[0] for x in v) / H, "B": sum(x[1] for x in v) / H}
        for src in domains:          # single-domain fit -> every held-out domain
            r["single"][src] = {}
            for d in domains:
                ra_ = [G[d]["held"][li][h][0].r2(fitsA[src][h]) for h in range(H)]
                rb_ = [G[d]["held"][li][h][1].r2(fitsB[src][h]) for h in range(H)]
                r["single"][src][d] = {"A": sum(ra_) / H, "B": sum(rb_) / H}
        AA[str(li)], AB[str(li)] = torch.stack(WA), torch.stack(WB)
        rep["layers"][li] = {"enc_layer": amap[li], **r}
        log(f"  L{li}<-E{amap[li]} pooled held R² " + " ".join(
            f"{d}: A {r['pooled'][d]['A']:.3f} B {r['pooled'][d]['B']:.3f}" for d in domains))
        for src in domains:
            log(f"      fit on {src:12s} -> " + " ".join(
                f"{d}: A {r['single'][src][d]['A']:.3f} B {r['single'][src][d]['B']:.3f}" for d in domains))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"format": FORMAT, "A": {}, "B": {}, "layer_map": {}, "attn_A": AA, "attn_B": AB, "attn_map": amap,
                "report": rep, "args": vars(a)}, a.out)
    with open(os.path.splitext(a.out)[0] + "_report.json", "w") as f:
        json.dump(rep, f, indent=1)
    log(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
