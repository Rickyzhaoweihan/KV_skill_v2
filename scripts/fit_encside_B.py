"""D10 (encoder-side addressing): ridge B_ℓ mapping the ENCODER's attention output to the READER's, token-aligned.

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
    ap.add_argument("--layers", default="3,7,11,15")
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
    amap = {k: v for k, v in full_depth_map(ra, ea).items() if k in keep}
    levels = [x for x in a.levels.split(",") if x]
    for lv in levels:
        if (a.out_pre if lv == "pre" else a.out_post) is None:
            ap.error(f"--out_{lv} is required when --levels includes {lv}")
    caps = {lv: (EncCapture(reader, sorted(amap), lv), EncCapture(enc, sorted(set(amap.values())), lv)) for lv in levels}
    dev = input_device(reader)
    dims = {"pre": (ea["n_q"] * ea["hd"], ra["n_q"] * ra["hd"]),
            "post": (int(text_config(enc).hidden_size), int(text_config(reader).hidden_size))}
    G = {lv: {s: {li: Gram(dims[lv][0], dims[lv][1], dev) for li in amap} for s in ("fit", "held")} for lv in levels}
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
                    for li, e in amap.items():
                        G[lv][s][li].add(r2[e][0].float().to(dev), r9[li][0].float())
                        if a.mlp_hidden and lv == a.mlp_level:
                            T = r9[li].shape[1]
                            idx = torch.randperm(T)[: a.mlp_rows].to(dev)
                            rows.setdefault((s, li), []).append((r2[e][0][idx.to(r2[e].device)].half().cpu(),
                                                                 r9[li][0][idx].half().cpu()))
        log(f"  {name}: {len(dids)} docs ({n_held} held out)")
    for lv, path in [(lv, a.out_pre if lv == "pre" else a.out_post) for lv in levels]:
        B, rep = {}, {}
        lam_grid = [float(x) for x in a.lams.split(",")] if a.lams else [a.lam]
        for li in amap:
            cands = [(G[lv]["held"][li].r2(ridge(G[lv]["fit"][li], lm)), lm) for lm in lam_grid]
            best_lam = max(cands)[1]
            if len(lam_grid) > 1:
                log(f"  [{lv}] L{li} λ grid held R² " + " ".join(f"{lm:g}:{r:.3f}" for r, lm in cands) + f" -> {best_lam:g}")
            W = ridge(G[lv]["fit"][li], best_lam)
            B[str(li)] = W.t().float().cpu()
            rep[li] = {"enc_layer": amap[li], "r2_fit": G[lv]["fit"][li].r2(W), "r2_held": G[lv]["held"][li].r2(W)}
            log(f"  [{lv}] L{li}<-E{amap[li]}: R² fit {rep[li]['r2_fit']:.3f} held {rep[li]['r2_held']:.3f}")
        mlp = {}
        if a.mlp_hidden and lv == a.mlp_level:
            for li in amap:
                Xf = torch.cat([x for x, _ in rows[("fit", li)]]).float()
                Yf = torch.cat([y for _, y in rows[("fit", li)]]).float()
                Xh = torch.cat([x for x, _ in rows[("held", li)]]).float().to(dev)
                Yh = torch.cat([y for _, y in rows[("held", li)]]).float().to(dev)
                Bw = B[str(li)].to(dev)
                net = torch.nn.Sequential(torch.nn.Linear(Xf.shape[1], a.mlp_hidden), torch.nn.GELU(),
                                          torch.nn.Linear(a.mlp_hidden, Yf.shape[1])).to(dev)
                torch.nn.init.zeros_(net[2].weight)
                torch.nn.init.zeros_(net[2].bias)
                opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
                sst = float(((Yh - Yh.mean(0)) ** 2).sum())
                r2_of = lambda: 1 - float(((Xh @ Bw.t() + net(Xh) - Yh) ** 2).sum()) / sst   # noqa: E731
                with torch.no_grad():
                    r2_0 = r2_of()
                best, best_sd = r2_0, {k: v.clone() for k, v in net.state_dict().items()}
                for ep in range(a.mlp_epochs):
                    perm = torch.randperm(Xf.shape[0])
                    for i in range(0, Xf.shape[0], 1024):
                        j = perm[i:i + 1024]
                        x, y = Xf[j].to(dev), Yf[j].to(dev)
                        loss = ((x @ Bw.t() + net(x) - y) ** 2).mean()
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    with torch.no_grad():
                        r2 = r2_of()
                    if r2 > best:
                        best, best_sd = r2, {k: v.clone() for k, v in net.state_dict().items()}
                    log(f"  [{lv} mlp] L{li} epoch {ep + 1}: held R² {r2:.4f} (ridge rows {r2_0:.4f})")
                mlp[str(li)] = {k: v.cpu() for k, v in best_sd.items()}
                rep[li]["r2_held_rows_ridge"], rep[li]["r2_held_rows_mlp"] = r2_0, best
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"format": "xenc.encside_B.v1", "level": lv, "enc_B": B, "mlp": mlp, "mlp_hidden": a.mlp_hidden if mlp else 0, "layer_pairs": amap, "report": rep,
                    "args": vars(a)}, path)
        with open(os.path.splitext(path)[0] + "_report.json", "w") as f:
            json.dump(rep, f, indent=1)
        log(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
