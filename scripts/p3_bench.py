"""P3: memory / latency / storage of the token-identity channel variants (xenc/p3_tok.py) on ONE GPU.

Synthetic operator of --n_op columns (random post-RoPE K/V at the paired 4B layers, random column ids), the 9B reader,
the lever-1 interface. For each variant: per-document table build time and bytes, prefill + per-token decode latency
of the read (student mode, --rows prompt rows, --decode tokens), peak extra memory, and one training-mode e2e
forward+backward (--train_rows rows, checkpointed read as --mem_safe 1) peak memory and time. U is random (non-zero)
so the channel is computed. Content-independent: these are cost numbers only.

    python scripts/p3_bench.py --init outputs/D13_lever1_pilot/d13_best.pt --out outputs/P3_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401

VARIANTS = {
    "base": None,
    "b_dense_kv_r256_s3_L19": dict(layers="19", rank=256, heads="kv", shifts="-1,0,1", topk=0),
    "b_dense_avg_r256_s3_L19": dict(layers="19", rank=256, heads="avg", shifts="-1,0,1", topk=0),
    "c_dense_kv_r256_s3_L15_19": dict(layers="15,19", rank=256, heads="kv", shifts="-1,0,1", topk=0),
    "a_topk64_kv_r256_s3_L19": dict(layers="19", rank=256, heads="kv", shifts="-1,0,1", topk=64),
    "a_topk32_avg_full_s3_L19": dict(layers="19", rank=0, heads="avg", shifts="-1,0,1", topk=32),
    "a_topk64_avg_full_s5_L19": dict(layers="19", rank=0, heads="avg", shifts="-2,-1,0,1,2", topk=64),
}


def sync():
    torch.cuda.synchronize()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--init", default="outputs/D13_lever1_pilot/d13_best.pt")
    ap.add_argument("--n_op", default="16000,63000")
    ap.add_argument("--rows", type=int, default=160)
    ap.add_argument("--decode", type=int, default=32)
    ap.add_argument("--train_rows", type=int, default=300)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out", default="outputs/P3_bench.json")
    a = ap.parse_args(argv)
    from register_doc import log
    from xenc.backbone import load_encoder, load_reader
    from xenc.d13_read import D13Reader, load_B, load_d13
    from xenc.p3_tok import TokReader, parse_tok, upgrade_tok
    dev = "cuda"
    reader, tok = load_reader(a.reader, device_map=dev)
    enc = load_encoder(a.encoder, device_map="cpu")[0]
    lmap = {int(k): int(v) for k, v in torch.load(a.init, map_location="cpu", weights_only=False)["config"]["layer_map"].items()}
    B = load_B(a.B, sorted(lmap))
    base, _ = load_d13(a.init, enc, B)
    base = base.to(dev)
    E = reader.get_input_embeddings().weight
    gpu = torch.cuda.get_device_name(0)
    a0 = base.att[str(base.layers[0])]
    hkv, hd = a0.k_proj.out_features // a0.head_dim, a0.head_dim
    log(f"P3 bench on {gpu}: init {a.init} | layer_map {lmap} | hkv {hkv} hd {hd}")
    res = {"gpu": gpu, "init": a.init, "rows": a.rows, "decode": a.decode, "train_rows": a.train_rows, "runs": []}
    g = torch.Generator(device="cpu").manual_seed(0)
    for N in [int(x) for x in a.n_op.split(",")]:
        op = {e: (torch.randn(hkv, N, hd, generator=g).to(dev, torch.bfloat16),
                  torch.randn(hkv, N, hd, generator=g).to(dev, torch.bfloat16)) for e in sorted(set(lmap.values()))}
        op["ids"] = torch.randint(0, 248000, (N,), generator=g).to(dev)
        kv_bytes = sum(2 * hkv * N * hd * 2 for e in set(lmap.values()))     # bf16 K + V at the paired layers
        rows = torch.randint(1000, 100000, (1, a.rows), generator=g).to(dev)
        for name in a.variants.split(","):
            cfg = VARIANTS[name]
            torch.cuda.empty_cache()
            if cfg is None:
                itf, Rc = base, D13Reader
            else:
                itf = upgrade_tok(base, enc, parse_tok(cfg["layers"], cfg["rank"], cfg["heads"], cfg["shifts"], cfg["topk"]), E)
                with torch.no_grad():
                    for p in itf.TOK["U"].values():
                        p.normal_(0, 1e-3)
                Rc = TokReader
            R = Rc(reader, itf)
            opx = dict(op)
            sync()
            m0 = torch.cuda.memory_allocated()
            t0 = time.time()
            R.load(opx, N)
            sync()
            t_table = time.time() - t0
            table_bytes = torch.cuda.memory_allocated() - m0
            torch.cuda.reset_peak_memory_stats()
            m1 = torch.cuda.memory_allocated()
            R.mode = "student"
            with torch.no_grad():
                sync()
                t0 = time.time()
                out = reader(input_ids=rows, use_cache=True, logits_to_keep=1)
                sync()
                t_pre = time.time() - t0
                t0 = time.time()
                nxt = rows[:, -1:]
                for _ in range(a.decode):
                    out = reader(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
                    nxt = out.logits[:, -1:].argmax(-1)
                sync()
                t_dec = (time.time() - t0) / a.decode
            peak_inf = torch.cuda.max_memory_allocated() - m1
            del out
            R.mode, R.past = "off", {}
            # training-mode e2e forward + backward (checkpointed read, as --mem_safe 1)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            m2 = torch.cuda.memory_allocated()
            ps = [p for p in itf.parameters() if p.dtype.is_floating_point]
            req = [p.requires_grad for p in ps]
            itf.trainable(train_B=True, train_g=True)
            if cfg is not None:
                for p in list(itf.TOK["U"].parameters()):
                    p.requires_grad_(True)
            R.load(opx, N)
            R.ckpt, R.mode, R.need_logits, R.rec = True, "e2e", True, {}
            trows = torch.randint(1000, 100000, (1, a.train_rows), generator=g).to(dev)
            t_tr, peak_tr, err = None, None, None
            try:
                sync()
                t0 = time.time()
                with torch.enable_grad():
                    lo = reader(input_ids=trows, logits_to_keep=40).logits.float().logsumexp(-1).mean()
                    lo.backward()
                sync()
                t_tr = time.time() - t0
                peak_tr = torch.cuda.max_memory_allocated() - m2
            except torch.OutOfMemoryError as ex:
                err = f"OOM: {str(ex)[:120]}"
            finally:
                R.mode, R.rec = "off", {}
                for p in ps:
                    p.grad = None
                for p, rq in zip(ps, req):
                    p.requires_grad_(rq)
                lo = None
            R.detach()
            r = {"variant": name, "cfg": cfg, "n_op": N, "kv_bytes_per_doc": kv_bytes, "ids_bytes_per_doc": 4 * N,
                 "ids_overhead_pct": 100.0 * 4 * N / kv_bytes, "table_bytes_gpu": int(table_bytes),
                 "t_table_s": round(t_table, 4), "t_prefill_s": round(t_pre, 4), "t_decode_ms_per_tok": round(1e3 * t_dec, 3),
                 "peak_extra_inference_MiB": round(peak_inf / 2**20, 1),
                 "t_train_fwd_bwd_s": None if t_tr is None else round(t_tr, 3),
                 "peak_extra_train_MiB": None if peak_tr is None else round(peak_tr / 2**20, 1), "error": err,
                 "U_params": (sum(p.numel() for p in itf.TOK["U"].parameters()) if cfg else 0)}
            res["runs"].append(r)
            log(json.dumps(r))
            if cfg is not None:
                del itf
            torch.cuda.empty_cache()
        del op
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
