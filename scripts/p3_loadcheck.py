"""Load check before a registered judge (P3 / P4 / P5 token-identity family): the checkpoint loads through the SAME
entry point the judge uses (fit_d13.build -> load_d13 dispatch + the family's build-time setup) and reproduces the
training run's own held-out val kv on a few span items at the checkpoint's step (same GPU type: L40S).

Reads <run>/args.json (val selection exactly as register_d13 / p5_train) and <run>/history.json (per-item kv at the
checkpoint's step, --save_items 1). Exit 1 if fewer than --min_match of the re-generated scores equal the saved ones.

    python scripts/p3_loadcheck.py --run outputs/P3_bL --ckpt outputs/P3_bL/d13_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tasks", default="qasper,musique,lastingbench")
    ap.add_argument("--per_task", type=int, default=4)
    ap.add_argument("--min_match", type=float, default=0.8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enc_device", default="")
    a = ap.parse_args(argv)
    from fit_d13 import D13Rt, build, select_items
    from register_d13 import doc_len_filter, excluded_docs
    from register_doc import _get, log
    from xenc.doc_tasks import score
    A = json.load(open(f"{a.run}/args.json"))
    H = json.load(open(f"{a.run}/history.json"))
    blob = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    step = (blob.get("extra") or {}).get("step")
    ev = next((e for e in H["evals"] if e["step"] == step), None)
    assert ev is not None and ev.get("items"), f"no saved val items at step {step} in {a.run}/history.json"
    saved = {r[0]: r[1] for r in ev["items"]}
    ns = types.SimpleNamespace(reader=A.get("reader", "Qwen/Qwen3.5-9B"), encoder=A["encoder"], B=A["B"],
                               layers=A.get("layers", "3,7,11,15,19"), device=a.device, enc_device=a.enc_device)
    reader, enc, tok, itf, lmap = build(ns, init=a.ckpt)
    rt = D13Rt(reader, enc, tok, ns.reader, itf, A.get("tail", 64), A.get("max_tokens", 16000), lru=2)
    val, docs = select_items(A["val"], "val", {A.get("val_fold", 1)}, A.get("val_q_per_doc", 4), A.get("val_seed", 1))
    drop = excluded_docs(A.get("exclude_sets", ""))
    if A.get("exclude_docs"):
        X = json.load(open(A["exclude_docs"]))
        drop |= set(sum(X.values(), [])) if isinstance(X, dict) else set(X)
    keep = doc_len_filter(A.get("doc_len", ""))
    val = [i for i in val if _get(i, "doc_id") not in drop and keep(_get(i, "doc_id"))]
    assert [str(_get(i, "key")) for i in val] == [r[0] for r in ev["items"]], "val items differ from the run's history"
    rt.docs = docs
    pick = []
    for t in a.tasks.split(","):
        pick += [i for i in val if _get(i, "doc_id").split("_")[0] == t][: a.per_task]
    log(f"load check: {a.ckpt} (step {step}, {type(itf).__name__}, cfg keys {sorted(itf.config())}) on {len(pick)} val "
        f"items | GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    n_ok = 0
    for it in pick:
        g, _ = rt.gen(_get(it, "doc_id"), it, A.get("budget", 32), read=True)
        s = float(score(it, g))
        ok = abs(s - saved[str(_get(it, "key"))]) < 1e-6
        n_ok += ok
        log(f"  {_get(it, 'key')}: regen {s:.3f} saved {saved[str(_get(it, 'key'))]:.3f} {'ok' if ok else 'DIFF'} | {g[:80]!r}")
    frac = n_ok / max(len(pick), 1)
    log(f"load check: {n_ok}/{len(pick)} reproduce the saved val scores ({frac:.2f}; need >= {a.min_match})")
    return 0 if frac >= a.min_match else 1


if __name__ == "__main__":
    sys.exit(main())
