"""Merge encoder-side post-B files (xenc.encside_B.v1) over disjoint reader layers into one file (lever 2a).

    python scripts/d13_merge_B.py --inputs outputs/D11_maps/encB5_9b_4b_post_xf01.pt,outputs/D13_deep/encB_deep_post.pt \
        --out outputs/D13_deep/encB8_9b_4b_post.pt
"""
import argparse
import sys

import torch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = None
    for p in a.inputs.split(","):
        b = torch.load(p, map_location="cpu", weights_only=False)
        assert b.get("level") == "post" and not b.get("mlp"), p
        if out is None:
            out = {"format": b["format"], "level": "post", "enc_B": {}, "mlp": {}, "mlp_hidden": 0, "layer_pairs": {},
                   "report": {}, "args": {"merged_from": a.inputs}}
        for k, v in b["enc_B"].items():
            assert k not in out["enc_B"], f"layer {k} in two inputs"
            out["enc_B"][k] = v
        out["layer_pairs"].update(b.get("layer_pairs", {}))
        out["report"].update(b.get("report", {}))
    torch.save(out, a.out)
    print("merged layers", sorted(int(k) for k in out["enc_B"]), "pairs", out["layer_pairs"], "->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
