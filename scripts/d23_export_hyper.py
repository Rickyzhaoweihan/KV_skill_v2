"""D23: export D24 hypernetwork adapters in the vLLM plugin's per-document adapter format (envs/qwen35, CPU is fine).

For every operator in <export>/ops: ad = H(operator), converted to the plugin rows
    D.Ua{l} = U_l · M_l(doc)   [d_e, R]     (dA_l = U_l M_l V_l = Ua_l Va_l)
    D.Va{l} = V_l              [R, d_r]
    g       = g0 + dg(doc)     [n_layers]   (the full gate, as D20 snapshots store it)
    Bb{l}   = Ub_l · c_l(doc)  [d_r]        (write-back bias offset; only when H has bias_k > 0)
A self-check asserts that the converted rows reproduce xenc.d24_hyper.attach(itf, H) (map_x / Bmap / gate) on random
inputs. Also writes the HF reference generations for the plugin's validation items (D24 judged `kv` = base + H).

    python scripts/d23_export_hyper.py --export $XENC_CACHE/D23_vllm
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


def to_rows(H, itf, ad) -> dict:
    out = {"g": (itf.g.detach().float() + (ad["dg"] if ad["dg"] is not None else 0)).cpu()}
    for li in H.layers:
        k = str(li)
        out[f"D.Ua{li}"] = (H.U[k] @ ad["M"][li]).detach().float().cpu()
        out[f"D.Va{li}"] = H.V[k].detach().float().cpu()
        if ad.get("c") is not None:
            out[f"Bb{li}"] = (H.Ub[k] @ ad["c"][li]).detach().float().cpu()
    return out


@torch.no_grad()
def self_check(H, itf, ad, rows, seed=0):
    """rows (plugin form) == attach(itf, H) with set_doc(ad), on random x / o."""
    import copy
    from xenc.d24_hyper import attach, set_doc
    it2 = copy.deepcopy(itf)
    attach(it2, H)
    set_doc(it2, ad)
    g = torch.Generator().manual_seed(seed)
    for li in H.layers:
        x = torch.randn(3, itf.d_r, generator=g)
        o = torch.randn(3, itf.d_e, generator=g)
        ref_x = it2.map_x(li, x)
        got_x = itf.map_x(li, x) + (x @ rows[f"D.Va{li}"].t()) @ rows[f"D.Ua{li}"].t()
        assert torch.allclose(got_x, ref_x, atol=1e-4, rtol=1e-4), (li, float((got_x - ref_x).abs().max()))
        ref_y = it2.Bmap(li, o)
        got_y = itf.Bmap(li, o) + (rows[f"Bb{li}"] if f"Bb{li}" in rows else 0)
        assert torch.allclose(got_y, ref_y, atol=1e-4, rtol=1e-4), (li, float((got_y - ref_y).abs().max()))
        assert abs(float(it2.gate(li)) - float(rows["g"][itf._gi[li]])) < 1e-5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", required=True)
    ap.add_argument("--h9", default="outputs/D24_h3/d24_best.pt")
    ap.add_argument("--tag9", default="adapters_h3")
    ap.add_argument("--tag27", default="adapters_h27r")
    ap.add_argument("--ref_eval", default="outputs/D24_h3_lh/eval_test.json", help="HF judged kv_H generations")
    ap.add_argument("--seed27", type=int, default=0)
    a = ap.parse_args(argv)
    from xenc.d23_vllm_plugin import build_interface
    from xenc.d24_hyper import D24Hyper, load_h
    ops_dir = os.path.join(a.export, "ops")
    itf9, _ = build_interface(os.path.join(a.export, "itf_9b.pt"), "cpu")
    H9, _ = load_h(a.h9, itf9)
    H9.eval()
    itf27, _ = build_interface(os.path.join(a.export, "itf_27b.pt"), "cpu")
    c = dict(H9.cfg)
    c.pop("hkv", None)
    c.pop("hd", None)
    torch.manual_seed(a.seed27)
    H27 = D24Hyper(itf27, **c).eval()
    with torch.no_grad():                                  # random H with non-zero (small) heads: timing only
        for p in H27.parameters():
            if p.dim() > 1:
                p.normal_(0, 0.02)
    prompts = json.load(open(os.path.join(a.export, "prompts.json")))
    bench_docs = {b["doc"] for b in prompts["bench"]}
    for tag, H, itf, docs in ((a.tag9, H9, itf9, None), (a.tag27, H27, itf27, bench_docs)):
        od = os.path.join(a.export, tag)
        os.makedirs(od, exist_ok=True)
        for f in sorted(os.listdir(ops_dir)):
            did = f[:-3]
            if docs is not None and did not in docs:
                continue
            blob = torch.load(os.path.join(ops_dir, f), map_location="cpu", weights_only=False)
            op = {int(e): kv for e, kv in blob["op"].items() if int(e) in set(itf.layer_map.values())}
            with torch.no_grad():
                ad = H(op)
            rows = to_rows(H, itf, ad)
            self_check(H, itf, ad, rows)
            torch.save(rows, os.path.join(od, f"{did}.pt"))
            print(f"{tag} {did}: n_op {blob['n_op']} rank {H.R} bias {H.bias_k} |M| "
                  f"{float(torch.stack([m.norm() for m in ad['M'].values()]).mean()):.4f} gate {rows['g'].tolist()}",
                  flush=True)
    judged = {r["key"]: r for r in json.load(open(a.ref_eval))["items"]}
    ref = {v["key"]: judged[v["key"]]["kv"]["gen"] for v in prompts["val"] if v["key"] in judged}
    json.dump(ref, open(os.path.join(a.export, "val_ref_h3.json"), "w"), indent=1)
    print(f"val references (D24 h3 judged kv = base + H): {len(ref)} / {len(prompts['val'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
