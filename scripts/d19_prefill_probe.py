"""[D19] Cost probe for book-length documents: encoder prefill time, peak memory and operator size of the D13
operator (same runtime as eval_d13.py, no truncation), plus the reader's own full-text prefill for comparison.
One median-length TEST document per split. GPU only; no generation, no training.

  python scripts/d19_prefill_probe.py --interface outputs/D13_reg_w10/d13_best.pt \
      --splits data/loogle_split,data/clipper_split --out outputs/D19_prefill_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--B", default="outputs/D10_maps/encB8_9b_2b_post.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--interface", required=True)
    ap.add_argument("--splits", default="data/loogle_split,data/clipper_split")
    ap.add_argument("--max_tokens", type=int, default=200000)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from fit_d13 import D13Rt, build
    from register_doc import SYSTEM_ANSWER
    from xenc.backbone import input_device
    from xenc.doc_tasks import load_docs
    from xenc.state_skill import doc_prefix_text

    reader, enc, tok, itf, lmap = build(a, init=a.interface)
    rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens, lru=1)
    res = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "layer_map": lmap, "docs": []}
    for sd in a.splits.split(","):
        index = json.load(open(Path(sd) / "docs.json"))
        test = {it["doc_id"] for it in json.load(open(Path(sd) / "test" / "items.json"))}
        ids = sorted(test, key=lambda d: index[d]["n_tokens"])
        did = ids[len(ids) // 2]
        rt.docs = load_docs(sd)
        rt._ops.clear()
        rec = {"split": sd, "doc_id": did, "n_tokens_doc": index[did]["n_tokens"]}
        torch.cuda.reset_peak_memory_stats()
        _sync()
        t0 = time.time()
        op, n_op, _ = rt.operator(did)
        _sync()
        rec["encoder_prefill_s"] = round(time.time() - t0, 2)
        rec["encoder_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        rec["operator_positions"] = int(n_op)
        nbytes = sum(t.numel() * t.element_size() for kv in op.values() for t in kv)
        rec["operator_mb"] = round(nbytes / 2**20, 1)
        rec["operator_kb_per_token"] = round(nbytes / 2**10 / max(1, n_op), 2)
        rec["operator_shapes"] = {str(e): [list(kv[0].shape), str(kv[0].dtype)] for e, kv in op.items()}
        rt._ops.clear()
        torch.cuda.empty_cache()
        ids_txt = tok(doc_prefix_text(tok, a.reader, rt.docs[did], SYSTEM_ANSWER), add_special_tokens=False)["input_ids"]
        torch.cuda.reset_peak_memory_stats()
        try:
            _sync()
            t0 = time.time()
            with torch.no_grad():
                reader(input_ids=torch.tensor([ids_txt], device=input_device(reader)), use_cache=True, logits_to_keep=1)
            _sync()
            rec["reader_text_prefill_s"] = round(time.time() - t0, 2)
            rec["reader_text_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        except torch.cuda.OutOfMemoryError:
            rec["reader_text_prefill_s"] = "OOM"
        torch.cuda.empty_cache()
        res["docs"].append(rec)
        print(json.dumps(rec), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
