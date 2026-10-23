"""D23: per-document cost of the HYPERNETWORK registration path (D24's H) vs the 4B harvest, the reader's own prefill
and per-document registration. Benchmark only (no accuracy). One GPU (H100 for comparability with D23's tables).

Per document length L (the D23 documents, 2k / 8k / 16k / 32k / 63k):
  harvest   4B encoder prefill truncated after the deepest read layer (9B map: layer 19; 27B map: layer 15)
  H         one forward of H on that operator -> per-document adapter (trained h1 / h3 for the 9B reader;
            randomly initialised H of the same shapes for the 27B reader, timing only)
  rows      conversion of H's output into the vLLM plugin's static adapter rows (Ua = U·M, Va = V, gate)
Reported: median time over --reps (after warm-up, torch.cuda.synchronize), FLOPs (FlopCounterMode), peak working
memory, the per-document adapter size, and H's own (shared, one-time) parameter count.

    python scripts/bench_d23_hyper.py --export $XENC_CACHE/D23_vllm --out outputs/D23_hyper
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, log  # noqa: E402

DOCS = ("2k:quality_380c2df08912,8k:quality_67c8a9cb38f1,16k:loogle_b9faf4fc6659,32k:loogle_cbeab30e24d7,"
        "64k:loogle_78d3ed194af9")


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, warm, reps):
    for _ in range(warm):
        fn()
    sync()
    ts, peak, base = [], 0, 0
    for k in range(reps):
        if k == 0 and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        t = time.perf_counter()
        r = fn()
        sync()
        ts.append(time.perf_counter() - t)
        if k == 0 and torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()
        del r
    return float(st.median(ts)), (peak - base) / 2 ** 30


def flops(fn) -> float:
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        fn()
    return float(fc.get_total_flops())


def adapter_rows(H, ad):
    """H output -> the plugin's static adapter rows: per layer Ua = U·M [d_e,R], Va = V [R,d_r] (shared), gate dg."""
    return {li: H.U[str(li)] @ ad["M"][li] for li in H.layers}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--export", required=True, help="scripts/d23_export_vllm.py output (itf_9b.pt / itf_27b.pt)")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--h9", default="h1:outputs/D24_h1/d24_best.pt,h3:outputs/D24_h3/d24_best.pt")
    ap.add_argument("--docs", default=DOCS)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max_tokens", type=int, default=72000)
    ap.add_argument("--device", default="cuda", help="cpu: smoke test only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    from bench_d23 import Bench, task_of
    from xenc.backbone import load_encoder
    from xenc.d13_read import EncTeacher
    from xenc.d23_vllm_plugin import build_interface
    from xenc.d24_hyper import D24Hyper, load_h
    from xenc.doc_tasks import load_docs
    from xenc.state_skill import doc_prefix_text
    os.makedirs(a.out, exist_ok=True)
    rows_f = open(os.path.join(a.out, "rows.jsonl"), "a")

    def row(**kw):
        rows_f.write(json.dumps(kw) + "\n")
        rows_f.flush()
        log("ROW " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in kw.items()})[:400])

    dev = a.device
    enc = load_encoder(a.encoder, device_map=dev)[0]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    readers = {}
    itf9, _ = build_interface(os.path.join(a.export, "itf_9b.pt"), dev)
    Hs9 = {}
    for spec in a.h9.split(","):
        name, path = spec.split(":")
        H, _ = load_h(path, itf9)
        Hs9[name] = H.to(dev).eval()
    readers["9b"] = (itf9, Hs9)
    itf27, _ = build_interface(os.path.join(a.export, "itf_27b.pt"), dev)
    Hs27 = {}
    for name, H9 in Hs9.items():                       # same shapes, random init (timing only)
        c = dict(H9.cfg)
        c.pop("hkv", None)
        c.pop("hd", None)
        H = D24Hyper(itf27, **c).to(dev).eval()
        for p in H.parameters():                       # non-zero heads so the forward does real work
            torch.nn.init.normal_(p, 0, 0.02) if p.dim() > 1 else None
        Hs27[name + "_random"] = H
    readers["27b"] = (itf27, Hs27)
    row(phase="env", gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", torch=torch.__version__, encoder=a.encoder,
        H={k: {"params": H.n_params(), "adapter_numel": H.adapter_numel(), "cfg": H.cfg}
           for rd in readers.values() for k, H in rd[1].items()},
        itf_layers={k: rd[0].layer_map for k, rd in readers.items()}, slurm_job=os.environ.get("SLURM_JOB_ID"))
    docs = {}
    for spec in a.docs.split(","):
        tag, did = spec.split(":")
        if did not in docs:
            docs.update(load_docs(f"data/{task_of(did)}_split"))
        ids = [int(x) for x in tok(doc_prefix_text(tok, "Qwen/Qwen3.5-9B", docs[did], SYSTEM_ANSWER),
                                   add_special_tokens=False)["input_ids"]][-(a.max_tokens - 1024):]
        for rname, (itf, Hs) in readers.items():
            enc_layers = sorted(set(itf.layer_map.values()))
            Tc = EncTeacher(enc, enc_layers)
            fake = types.SimpleNamespace(rt=types.SimpleNamespace(pre=lambda d, ids=ids: ids, Tc=Tc), enc=enc)
            with torch.no_grad():
                hv = lambda: Bench.harvest(fake, did, trunc=True)  # noqa: E731
                t_h, m_h = timed(hv, a.warmup, a.reps)
                f_h = flops(hv)
                op, n = hv()
                for hname, H in Hs.items():
                    fwd = lambda H=H: H(op)  # noqa: E731
                    t_H, m_H = timed(fwd, a.warmup, a.reps)
                    f_H = flops(fwd)
                    ad = fwd()
                    conv = lambda H=H, ad=ad: adapter_rows(H, ad)  # noqa: E731
                    t_c, _ = timed(conv, a.warmup, a.reps)
                    row(phase="hyper", L=tag, doc=did, N=len(ids), n_op=n, reader=rname, H=hname,
                        enc_layers=enc_layers, harvest_s=t_h, harvest_peak_GiB=m_h, harvest_flops=f_h,
                        H_s=t_H, H_peak_GiB=m_H, H_flops=f_H, rows_s=t_c,
                        adapter_numel=H.adapter_numel(), adapter_bytes_fp32=4 * H.adapter_numel(),
                        rows_bytes_fp32=4 * sum(v.numel() for v in adapter_rows(H, ad).values()),
                        H_params=H.n_params(),
                        op_bytes=sum(K.nbytes + V.nbytes for K, V in op.values()))
                del op
            Tc.detach()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    log("D23_HYPER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
