"""D10: KEY/VALUE-side affine maps (encoder K/V -> reader K/V) at the early full-attention layers, token-aligned.

At reader layer ℓ ∈ --layers and its depth-paired encoder layer ℓ', both models prefill the same `text` prompts;
pre-RoPE keys (post k_norm) and values of all KV heads are concatenated per token:
    K9[t] ≈ M_ℓ [K2[t]; 1]      V9[t] ≈ N_ℓ [V2[t]; 1]        (ridge, affine)
The reader then reads the transformed operator (U' = M K2, W' = N V2, re-RoPE'd) with its OWN queries
(xenc.xattn_read.map_kv_states; eval_state_xm.py --kv_maps). Held-out R² and per-head cosine are reported.

    python scripts/fit_kv_side.py --out outputs/D10_maps/kvside_9b_2b.pt
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
    ap.add_argument("--layers", default="3,7,11,15")
    ap.add_argument("--max_tokens", type=int, default=9000)
    ap.add_argument("--heldout_frac", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    from xenc.backbone import build_chat, input_device, load_encoder, load_reader
    from xenc.doc_tasks import ctx_prompt, load_docs
    from xenc.xattn_read import AttnReader, attn_dims, full_depth_map, make_attn_interface, rotate_keys
    reader, tok = load_reader(a.reader, device_map=a.device)
    enc = load_encoder(a.encoder, device_map=a.device)[0]
    ra, ea = attn_dims(reader), attn_dims(enc)
    keep = {int(x) for x in a.layers.split(",")}
    amap = {k: v for k, v in full_depth_map(ra, ea).items() if k in keep}
    Ar = AttnReader(reader, make_attn_interface(reader, ra, layers=sorted(amap)), record_mag=False)
    Ae = AttnReader(enc, make_attn_interface(enc, ea, layers=sorted(set(amap.values()))), record_mag=False)
    dev = input_device(reader)
    hd = ra["hd"]
    dx, dy = ea["n_kv"] * hd + 1, ra["n_kv"] * hd
    G = {s: {li: (Gram(dx, dy, dev), Gram(dx, dy, dev)) for li in amap} for s in ("fit", "held")}
    cos_acc = {li: {"K": [], "V": []} for li in amap}
    excl = {int(x) for x in a.exclude_folds.split(",") if x != ""}
    held_batches = []
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
            Ar.capture_kv, Ae.capture_kv = {}, {}
            with torch.no_grad():
                reader(input_ids=x.to(dev), use_cache=False, logits_to_keep=1)
                enc(input_ids=x.to(input_device(enc)), use_cache=False, logits_to_keep=1)
                for li, e in amap.items():
                    K9, V9 = Ar.capture_kv[li]
                    K2, V2 = Ae.capture_kv[e]
                    T = K9.shape[2]
                    k9 = rotate_keys(Ar.rot, K9[0], -1.0).transpose(0, 1).reshape(T, -1)
                    k2 = rotate_keys(Ar.rot, K2[0].to(dev), -1.0).transpose(0, 1).reshape(T, -1)
                    v9 = V9[0].float().transpose(0, 1).reshape(T, -1)
                    v2 = V2[0].to(dev).float().transpose(0, 1).reshape(T, -1)
                    one = torch.ones(T, 1, device=dev)
                    G[s][li][0].add(torch.cat([k2, one], 1), k9)
                    G[s][li][1].add(torch.cat([v2, one], 1), v9)
                    if s == "held":
                        held_batches.append((li, torch.cat([k2, one], 1).half().cpu(), k9.half().cpu(),
                                             torch.cat([v2, one], 1).half().cpu(), v9.half().cpu()))
            Ar.capture_kv = Ae.capture_kv = None
        log(f"  {name}: {len(dids)} docs ({n_held} held out)")
    maps = {"kv_K": {}, "kv_V": {}}
    rep = {}
    for li in amap:
        WK, WV = ridge(G["fit"][li][0], a.lam), ridge(G["fit"][li][1], a.lam)
        maps["kv_K"][str(li)], maps["kv_V"][str(li)] = WK.t().float().cpu(), WV.t().float().cpu()
        ck, cv = [], []
        for (l2, xk, yk, xv, yv) in held_batches:
            if l2 != li:
                continue
            pk = (xk.to(dev).float() @ WK.float()).view(-1, ra["n_kv"], hd)
            pv = (xv.to(dev).float() @ WV.float()).view(-1, ra["n_kv"], hd)
            ck.append(torch.nn.functional.cosine_similarity(pk, yk.to(dev).float().view_as(pk), dim=-1).mean(0))
            cv.append(torch.nn.functional.cosine_similarity(pv, yv.to(dev).float().view_as(pv), dim=-1).mean(0))
        rep[li] = {"enc_layer": amap[li], "r2_K_held": G["held"][li][0].r2(WK), "r2_V_held": G["held"][li][1].r2(WV),
                   "cos_K_held_per_head": [round(float(x), 3) for x in torch.stack(ck).mean(0)],
                   "cos_V_held_per_head": [round(float(x), 3) for x in torch.stack(cv).mean(0)]}
        log(f"  L{li}<-E{amap[li]}: K R² {rep[li]['r2_K_held']:.3f} cos {rep[li]['cos_K_held_per_head']} | "
            f"V R² {rep[li]['r2_V_held']:.3f} cos {rep[li]['cos_V_held_per_head']}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"format": "xenc.kv_side.v1", **maps, "layer_pairs": amap, "report": rep, "args": vars(a)}, a.out)
    with open(os.path.splitext(a.out)[0] + "_report.json", "w") as f:
        json.dump(rep, f, indent=1)
    log(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
