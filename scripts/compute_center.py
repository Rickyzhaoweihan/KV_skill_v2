"""Per-depth mean slot (centering vector) over an encoder's operator cache.

    python scripts/compute_center.py --cache_dir $XENC_CACHE --encoder_short Qwen3.5-2B --out outputs/centers/center_Qwen3.5-2B_quality.pt [--index index_quality_split_Qwen3.5-2B.json]

Streams one cache file at a time (login-node safe). Output: {"center": {rho: mu[d_s] fp32}, "n_slots", "files"}.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xenc  # noqa: F401
import torch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default=xenc.DEFAULT_CACHE)
    ap.add_argument("--encoder_short", required=True)
    ap.add_argument("--index", default=None, help="index_<split>_<enc>.json in cache_dir to restrict files (default: glob all for the encoder)")
    ap.add_argument("--chunk_tokens", type=int, default=512)
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.index:
        idx = json.load(open(os.path.join(a.cache_dir, a.index)))
        files = sorted({os.path.join(a.cache_dir, os.path.basename(v)) if not os.path.isabs(v) else v for v in idx.values()})
    else:
        files = sorted(f for f in glob.glob(os.path.join(a.cache_dir, f"*_{a.encoder_short}_c{a.chunk_tokens}_d*.pt")) if "index" not in f)
    if a.max_files:
        files = files[: a.max_files]
    if not files:
        print("no cache files found", file=sys.stderr); return 2
    sums: dict[float, torch.Tensor] = {}; n = 0
    for i, f in enumerate(files):
        c = torch.load(f, map_location="cpu", weights_only=True)
        for ch in c["chunks"]:
            for rho, K in ch["K"].items():
                r = float(rho); s = K.float().sum(0)
                sums[r] = sums.get(r, 0) + s
            n += next(iter(ch["K"].values())).shape[0]
        del c
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(files)}] slots {n}", flush=True)
    center = {r: (s / n) for r, s in sums.items()}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"center": center, "n_slots": n, "files": len(files), "encoder_short": a.encoder_short,
                "norms": {r: float(m.norm()) for r, m in center.items()}}, a.out)
    print(f"wrote {a.out}: {len(files)} files, {n} slots, ||mu|| = " + ", ".join(f"{r}:{float(m.norm()):.3f}" for r, m in sorted(center.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
