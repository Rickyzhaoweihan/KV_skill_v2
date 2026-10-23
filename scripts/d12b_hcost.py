"""D12b: cost of one H-slots forward per document (GPU, CUDA-synchronised) vs operator length, from cached operators.
    python scripts/d12b_hcost.py --h outputs/D12b_slots_s1/h_best.pt --ops $XENC_CACHE/D12p/c9b_judge/op --n 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", required=True)
    ap.add_argument("--ops", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    from xenc.d12b_slots import SlotH
    blob = torch.load(a.h, map_location="cpu", weights_only=False)
    sd = blob["state_dict"]
    layers = sorted(int(k.split(".")[1]) for k in sd if k.startswith("Wk."))
    lm = {int(k): int(v) for k, v in json.loads(json.dumps(blob["extra"].get("layer_map") or {})).items()} or \
        {3: 3, 7: 3, 11: 7, 15: 7, 19: 11, 23: 11, 27: 15, 31: 15, 35: 19, 39: 19, 43: 23, 47: 23}
    D = sd[f"Wk.{layers[0]}.weight"].shape[0]
    hd = 256
    att = types.SimpleNamespace(head_dim=hd, k_proj=types.SimpleNamespace(out_features=D))
    fake = types.SimpleNamespace(layers=layers, layer_map=lm, att={str(li): att for li in layers})
    H = SlotH(fake, **blob["config"]).cuda().eval()
    H.load_state_dict(sd)
    fs = sorted(os.listdir(a.ops))[:: max(1, len(os.listdir(a.ops)) // a.n)][: a.n]
    rows = []
    for f in fs:
        b = torch.load(os.path.join(a.ops, f), map_location="cpu", weights_only=False)
        op = {int(e): (K.cuda(), V.cuda()) for e, (K, V) in b["kv"].items()}
        with torch.no_grad():
            H(op)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(3):
                H(op)
            torch.cuda.synchronize()
        rows.append({"doc": f[:-3], "n": int(b["n"]), "ms": (time.time() - t) / 3 * 1e3})
        print(f"  {f[:-3]:60s} N={b['n']:6d}  H forward {rows[-1]['ms']:.1f} ms", flush=True)
    print(f"H-slots forward: mean {sum(r['ms'] for r in rows) / len(rows):.1f} ms over {len(rows)} docs "
          f"(mean N {sum(r['n'] for r in rows) / len(rows):.0f}); {torch.cuda.get_device_name(0)}")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
