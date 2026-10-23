"""D16 copy of D10's scripts/fit_encside_B.py (post level only) over a GRID of (reader layer, encoder layer) pairs:
for every reader layer in --layers and every encoder full-attention layer, the closed-form token-aligned ridge
B: a2_e[t] -> a9_ℓ[t] and its held-out R². Also writes one encside_B blob for a chosen pairing (--pairs "3:3,7:3,...").
Original D10 docstring follows.

D10 (encoder-side addressing): ridge B_ℓ mapping the ENCODER's attention output to the READER's, token-aligned.

Both models prefill the same `text` prompts (identical tokenizers). At reader full-attention layer ℓ (depth-paired
encoder layer ℓ'):  level pre:  o9_pre[t] (16·256, pre-gate per-head concat) ≈ B o2_pre[t] (8·256)
                    level post: a9[t] (attention module output, 4096) ≈ B a2[t] (2048)
Held-out R² per layer. Output (--out_pre / --out_post) loads via eval_encside.py --maps.
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
from fit_state_maps import Gram, ridge  # noqa: E402
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--splits", default="quality:60,longhealth:12,musique:40,lastingbench:30")
    ap.add_argument("--exclude_folds", default="0,1")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--pairs", default="", help="pairing(s) to export: 'name=3:3,7:3,...;name2=...'")
    ap.add_argument("--out_dir", default="outputs/D16_Bgrid")
    ap.add_argument("--max_tokens", type=int, default=9000)
    ap.add_argument("--heldout_frac", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--lams", default="", help="comma list: per layer pick the ridge λ with the best held-out R²")
    ap.add_argument("--out_pre", default=None)
    ap.add_argument("--out_post", default=None)
    ap.add_argument("--levels", default="pre,post")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mlp_hidden", type=int, default=0, help=">0: also fit a 2-layer residual MLP on top of ridge B")
    ap.add_argument("--mlp_rows", type=int, default=1024, help="rows per doc cached for the MLP fit")
    ap.add_argument("--mlp_epochs", type=int, default=6)
    ap.add_argument("--mlp_level", default="pre")
    a = ap.parse_args(argv)
    from xenc.backbone import build_chat, input_device, load_encoder, load_reader
    from xenc.doc_tasks import ctx_prompt, load_docs
    from xenc.encside_read import EncCapture
    from xenc.state_skill import text_config
    from xenc.xattn_read import attn_dims, full_depth_map
    reader, tok = load_reader(a.reader, device_map=a.device)
    enc = load_encoder(a.encoder, device_map=a.device)[0]
    ra, ea = attn_dims(reader), attn_dims(enc)
    keep = {int(x) for x in a.layers.split(",")}
    efull = sorted(int(x) for x in ea["full"])
    pairs = [(li, e) for li in sorted(keep) for e in efull]
    levels = ["post"]
    caps = {"post": (EncCapture(reader, sorted(keep), "post"), EncCapture(enc, efull, "post"))}
    dev = input_device(reader)
    de, dr = int(text_config(enc).hidden_size), int(text_config(reader).hidden_size)
    G = {"post": {s: {p: Gram(de, dr, dev) for p in pairs} for s in ("fit", "held")}}
    excl = {int(x) for x in a.exclude_folds.split(",") if x != ""}
    rows: dict = {}
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
            x = torch.tensor([ids])
            for lv in levels:
                caps[lv][0].start()
                caps[lv][1].start()
            with torch.no_grad():
                reader(input_ids=x.to(dev), use_cache=False, logits_to_keep=1)
                enc(input_ids=x.to(input_device(enc)), use_cache=False, logits_to_keep=1)
                for lv in levels:
                    r9, r2 = caps[lv][0].get(), caps[lv][1].get()
                    for (li, e) in pairs:
                        G[lv][s][(li, e)].add(r2[e][0].float().to(dev), r9[li][0].float())
        log(f"  {name}: {len(dids)} docs ({n_held} held out)")
    os.makedirs(a.out_dir, exist_ok=True)
    W, rep = {}, {}
    for (li, e) in pairs:
        W[(li, e)] = ridge(G["post"]["fit"][(li, e)], a.lam)
        rep[f"{li}-{e}"] = {"r2_fit": G["post"]["fit"][(li, e)].r2(W[(li, e)]),
                            "r2_held": G["post"]["held"][(li, e)].r2(W[(li, e)])}
    log("B grid held-out R² (reader layer <- encoder layer, post):")
    for li in sorted(keep):
        log(f"  L{li}: " + " ".join(f"E{e} {rep[f'{li}-{e}']['r2_held']:.3f}" for e in efull))
    with open(os.path.join(a.out_dir, "Bgrid_report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    torch.save({"W": {f"{li}-{e}": W[(li, e)].t().float().cpu() for (li, e) in pairs}, "report": rep},
               os.path.join(a.out_dir, "Bgrid_all.pt"))
    for spec in [x for x in a.pairs.split(";") if x]:
        name, pm = spec.split("=")
        amap = {int(k): int(v) for k, v in (t.split(":") for t in pm.split(","))}
        blob = {"format": "xenc.encside_B.v1", "level": "post",
                "enc_B": {str(li): W[(li, e)].t().float().cpu() for li, e in amap.items()}, "mlp": {}, "mlp_hidden": 0,
                "layer_pairs": amap, "report": {li: rep[f"{li}-{e}"] for li, e in amap.items()}, "args": vars(a)}
        path = os.path.join(a.out_dir, f"encB_{name}.pt")
        torch.save(blob, path)
        log(f"saved {path} pairs {amap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
