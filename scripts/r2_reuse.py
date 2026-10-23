"""R2 dedup: before computing an eval_d13 row, look for an EXISTING item-identical row computed on the same GPU type and
reuse it (efficiency rule 5: never re-run what exists). Exit 0 = reused (writes <target>/REUSED.json and symlinks
<target>/eval_<split>.json to the source); exit 1 = nothing reusable, the caller computes it (`r2_reuse.py ... || eval_d13.py ...`).

A candidate is any outputs/*/eval_<split>.json written by scripts/eval_d13.py (args has "conds", no "budgets") with
  - the same interface checkpoint (realpath) — or, for --text, no interface needed and the "text" cond present;
  - the same item keys as the target set (D22 set item_keys, or the full split for --split_dir runs without a set);
  - every requested cond present on every item, except items listed in the candidate's oom_skipped.json (≤ --max_oom);
  - the same budget / tail / max_tokens;
  - produced on the required GPU type (the slurm log that ran `--out <candidate>` names the GPU at its top).

    python scripts/r2_reuse.py --target outputs/R2s_DSAB_qasper --interface outputs/DS_AB_full/d13_best.pt \
        --d22_set outputs/D22_sets/qasper.json --conds kv,wrong_doc --max_tokens 16000 || python scripts/eval_d13.py ...
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gpu_of(out_dir: str) -> str | None:
    """GPU name from the slurm log whose command wrote `out_dir` (lanes: the main log of that job)."""
    pat = re.compile(r"--out\s+" + re.escape(out_dir.rstrip("/")) + r"(\s|$|;)")
    for f in sorted(glob.glob(os.path.join(ROOT, "slurm/logs/*.out")), key=os.path.getmtime, reverse=True):
        try:
            txt = open(f, errors="ignore").read()
        except OSError:
            continue
        if not pat.search(txt):
            continue
        main = re.sub(r"_lane\d+\.out$", ".out", f)
        head = open(main, errors="ignore").read(2000) if os.path.exists(main) else txt[:2000]
        for g in ("L40S", "H100", "A100", "V100"):
            if g in head:
                return g
    return None


def expected_keys(a) -> set:
    if a.d22_set:
        return set(json.load(open(a.d22_set))["item_keys"])
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from register_doc import _get, load_split_items
    return {str(_get(i, "key")) for i in load_split_items(a.split_dir, a.split)}


def check(cand: str, a, want: set) -> str | None:
    """None if `cand` (an eval json path) is reusable, else the reason."""
    try:
        E = json.load(open(cand))
    except Exception as ex:                              # noqa: BLE001
        return f"unreadable ({type(ex).__name__})"
    g = E.get("args") or {}
    if "conds" not in g or "budgets" in g:
        return "not an eval_d13 output"
    if a.text:
        if "text" not in str(g.get("conds", "")).split(","):
            return "no text cond"
    elif not g.get("interface") or os.path.realpath(os.path.join(ROOT, g["interface"])) != os.path.realpath(
            os.path.join(ROOT, a.interface)):
        return f"interface {g.get('interface')}"
    for k, v in (("budget", a.budget), ("tail", a.tail), ("max_tokens", a.max_tokens)):
        if int(g.get(k, -1)) != int(v):
            return f"{k} {g.get(k)} != {v}"
    conds = ["text"] if a.text else [c for c in a.conds.split(",") if c]
    items = E.get("items") or []
    have = {str(r["key"]): r for r in items}
    oom = set()
    op = os.path.join(os.path.dirname(cand), "oom_skipped.json")
    if os.path.exists(op):
        oom = {str(x["key"]) for x in json.load(open(op))}
    if set(have) - want:
        return f"{len(set(have) - want)} items outside the target set"
    missing = want - set(have)
    if missing - oom or len(missing) > a.max_oom:
        return f"{len(missing)} target items missing ({len(missing - oom)} not OOM-skipped)"
    for r in items:
        if any(c not in r for c in conds):
            return "a requested cond is missing on some item"
    gpu = gpu_of(os.path.relpath(os.path.dirname(cand), ROOT))
    if gpu != a.gpu:
        return f"GPU {gpu} != {a.gpu}"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--interface", default="")
    ap.add_argument("--text", type=int, default=0, help="1: a text-only row (checkpoint-independent)")
    ap.add_argument("--d22_set", default="")
    ap.add_argument("--split_dir", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conds", default="kv,wrong_doc")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, required=True)
    ap.add_argument("--gpu", default="L40S")
    ap.add_argument("--max_oom", type=int, default=10)
    ap.add_argument("--glob", default="outputs/*/eval_{split}.json")
    a = ap.parse_args(argv)
    assert a.text or a.interface, "--interface or --text 1"
    want = expected_keys(a)
    tgt = os.path.abspath(os.path.join(ROOT, a.target))
    log = []
    for cand in sorted(glob.glob(os.path.join(ROOT, a.glob.format(split=a.split)))):
        if os.path.abspath(os.path.dirname(cand)) == tgt:
            continue
        why = check(cand, a, want)
        if why is None:
            os.makedirs(tgt, exist_ok=True)
            json.dump({"reused_from": os.path.relpath(cand, ROOT), "gpu": a.gpu, "n_target": len(want)},
                      open(os.path.join(tgt, "REUSED.json"), "w"), indent=1)
            link = os.path.join(tgt, f"eval_{a.split}.json")
            if not os.path.lexists(link):
                os.symlink(os.path.abspath(cand), link)
            print(f"[r2_reuse] {a.target}: REUSED {os.path.relpath(cand, ROOT)}", flush=True)
            return 0
        log.append((os.path.relpath(cand, ROOT), why))
    rel = [x for x in log if "interface" not in x[1] and "not an eval_d13" not in x[1] and "no text" not in x[1]]
    print(f"[r2_reuse] {a.target}: nothing reusable among {len(log)} eval jsons"
          + (f"; near misses: {rel[:5]}" if rel else ""), flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
