"""D7: real-model harvest check. Re-compute K/V for the first --n_tok tokens of a document through the
reader's attention modules (k_norm(k_proj(x)), v_proj(x) from the captured attention input of a prefix
prefill) and compare with the attn_raw cache (whole-doc prefill, fp16). Causal ⇒ identical up to kernel
numerics (bf16 model ⇒ expect ~1e-2 relative, cosine ≈ 1).

    python scripts/check_native_harvest.py --model Qwen/Qwen3.5-9B --split_dir data/longhealth_split \
        --depths 0.09375,...  --n_tok 256
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import xenc
from xenc.depth_skill import _key, attn_layer_map, cache_key, load_doc_cache


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B"); ap.add_argument("--split_dir", required=True)
    ap.add_argument("--depths", required=True); ap.add_argument("--n_tok", type=int, default=256)
    ap.add_argument("--cache_dir", default=os.environ.get("XENC_CACHE", xenc.DEFAULT_CACHE))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from deltaskill.backbone_utils import decoder_layers
    from xenc.backbone import load_reader
    from xenc.doc_tasks import load_docs
    from xenc.native_attn import attn_geometry
    depths = sorted(float(x) for x in a.depths.split(","))
    docs = load_docs(a.split_dir)
    m, tok = load_reader(a.model, device_map="cuda")
    short = a.model.split("/")[-1]
    did = next(d for d in sorted(docs) if os.path.exists(os.path.join(a.cache_dir, cache_key(docs[d], short, 512, depths, "attn_raw") + ".pt")))
    cache = load_doc_cache(os.path.join(a.cache_dir, cache_key(docs[did], short, 512, depths, "attn_raw") + ".pt"))
    ids = tok(docs[did], return_tensors="pt")["input_ids"][:, : a.n_tok].to(m.device)
    amap = attn_layer_map(m, depths); blocks = decoder_layers(m); got = {}; hs = []
    for li in amap.values():
        def pre(mod, args, kwargs, li=li):
            got[li] = kwargs.get("hidden_states", args[0] if args else None).detach()
        hs.append(blocks[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    with torch.no_grad():
        m(input_ids=ids, use_cache=False)
    for h in hs:
        h.remove()
    rep = {"doc_id": did, "n_tok": int(ids.shape[1]), "layers": {}}
    for rho, li in amap.items():
        att = blocks[li].self_attn; g = attn_geometry(att); x = got[li]; T = x.shape[1]
        with torch.no_grad():
            k = att.k_norm(att.k_proj(x).view(1, T, g["n_kv"], g["hd"]))[0].reshape(T, -1).float()
            v = att.v_proj(x)[0].float()
        Kc = torch.cat([c["K"][_key(c["K"], rho)] for c in cache["chunks"]])[:T].float().to(k.device)
        Vc = torch.cat([c["V"][_key(c["V"], rho)] for c in cache["chunks"]])[:T].float().to(k.device)
        r = {}
        for nm, A, B in (("K", Kc, k), ("V", Vc, v)):
            r[f"{nm}_maxabs"] = float((A - B).abs().max()); r[f"{nm}_rel"] = float((A - B).norm() / B.norm())
            r[f"{nm}_cos_min"] = float(torch.nn.functional.cosine_similarity(A, B, dim=-1).min())
            r[f"{nm}_absmax"] = float(B.abs().max())
        rep["layers"][li] = r
        print(f"layer {li} (rho {rho}): " + " ".join(f"{k_}={v_:.2e}" for k_, v_ in r.items()), flush=True)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
