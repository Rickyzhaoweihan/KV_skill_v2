"""DS: per-set TEST curve over checkpoints of one run (kv@0 from eval_d13 sweeps), with val next to each step, and two
selection-bias-safe reports (user 2026-10-23: val is used but not fully trusted; testing decides):
  (a) the FINAL-step checkpoint;
  (b) split-half pick: each set's clean items are split 50/50 by a fixed hash of the item key; the step is chosen on
      half A (per set, and once globally on the mean over sets) and reported on half B (and vice versa, averaged).
Also reports the val-best step and whether it agrees with the test pick.

    python scripts/ds_curve.py --run outputs/DS_max8k --steps 0=ABbest,2000=m8k2000,4000=m8k4000,6000=m8k6000,8000=m8k8000 \
        --leak outputs/DS_judged_leak_max8k.json --out outputs/DS_max8k_curve.json
Step tags resolve to outputs/DSsw_<tag>_<set>/eval_test.json (ABbest also from R2s_DSAB_* / ADDRsw_DSAB_*).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds_sweep_table as T  # noqa: E402


def half(key: str, seed: int) -> int:
    return int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest(), 16) % 2


PARTIAL: set = set()


def rows(tag: str, s: str) -> dict:
    """kv@0 per item. An unfinished sweep (no eval_test.json, e.g. its OOM-rerun cap was hit) is read from its ckpt
    jsonl, and any set with OOM-skipped items is flagged PARTIAL (those items are absent from kv AND wrong_doc and,
    through the key intersection over steps, from every step)."""
    d = f"outputs/DSsw_{tag}_{s}"
    p = f"{d}/eval_test.json"
    if os.path.exists(f"{d}/oom_skipped.json") and json.load(open(f"{d}/oom_skipped.json")):
        PARTIAL.add(s)
    if not os.path.exists(p) and os.path.exists(f"{d}/eval_test.ckpt.jsonl"):
        PARTIAL.add(s)
        out = {}
        for line in open(f"{d}/eval_test.ckpt.jsonl"):
            if line.strip():
                r = json.loads(line)
                if "kv" in r and "wrong_doc" in r:
                    out[r["key"]] = r["kv"]["score"]
        return out
    if not os.path.exists(p) and tag in T.OTHER:
        p = next((q.format(s=s) for q in T.OTHER[tag] if os.path.exists(q.format(s=s))), p)
    return T.scores(p)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/DS_max8k")
    ap.add_argument("--steps", default="0=ABbest,2000=m8k2000,4000=m8k4000,6000=m8k6000,8000=m8k8000")
    ap.add_argument("--leak", default="outputs/DS_judged_leak_max8k.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/DS_max8k_curve.json")
    a = ap.parse_args(argv)
    T.LEAK = a.leak
    steps = [(int(x.split("=")[0]), x.split("=")[1]) for x in a.steps.split(",")]
    val = {}
    hp = os.path.join(a.run, "history.json")
    if os.path.exists(hp):
        for e in json.load(open(hp)).get("evals", []):
            val[int(e.get("step", -1))] = e.get("kv")
    res, lines = {"steps": [s for s, _ in steps], "val": {s: val.get(s) for s, _ in steps}, "sets": {}}, []
    have = [(st, tag) for st, tag in steps]
    lines.append("| set | n | text | " + " | ".join(f"step {st}" for st, _ in have) + " | final | split-half pick (A→B / B→A) |")
    lines.append("|" + "---|" * (len(have) + 5))
    lines.append("| val (n=321) | | | " + " | ".join(f"{val[st]:.3f}" if val.get(st) is not None else "–" for st, _ in have) + " | | |")
    glob_half = {0: {st: [] for st, _ in have}, 1: {st: [] for st, _ in have}}
    for s, (_, (tf, tg)) in T.REFS.items():
        R = {st: rows(tag, s) for st, tag in have}
        R = {st: r for st, r in R.items() if r}
        if not R:
            continue
        keys = set.intersection(*[set(r) for r in R.values()])
        ck = T.clean_keys(s)
        if ck is not None:
            keys &= ck
        text = T.scores(tf, "text")
        curve = {st: mean([R[st][k] for k in keys]) for st in R}
        H = {h: [k for k in keys if half(k, a.seed) == h] for h in (0, 1)}
        pick = {}
        for h in (0, 1):
            ch = max(R, key=lambda st: mean([R[st][k] for k in H[h]]))
            pick[h] = (ch, mean([R[ch][k] for k in H[1 - h]]))
            if len(R) == len(have):              # the global pick uses only sets that have every step
                for st in R:
                    glob_half[h][st].append(mean([R[st][k] for k in H[h]]))
        fin = max(R)
        tx = mean([text[k] for k in keys if k in text]) if text else float("nan")
        res["sets"][s] = {"partial": s in PARTIAL, "n": len(keys), "text": tx, "text_gpu": tg, "curve": curve, "final_step": fin,
                          "final": curve[fin], "split_pick": {str(h): pick[h] for h in (0, 1)},
                          "split_pick_mean": (pick[0][1] + pick[1][1]) / 2}
        lines.append(f"| {s}{' (PARTIAL: OOM-skipped or unfinished items excluded)' if s in PARTIAL else ''} | {len(keys)} | {tx:.3f} | " + " | ".join(f"{curve[st]:.3f}" if st in curve else "–" for st, _ in have)
                     + f" | {curve[fin]:.3f} (step {fin}) | {pick[0][0]}→{pick[0][1]:.3f} / {pick[1][0]}→{pick[1][1]:.3f} |")
    # one global step: chosen on the mean over sets of half A, reported per set on half B (and vice versa)
    g = {h: max(glob_half[h], key=lambda st: mean([x for x in glob_half[h][st] if x == x]) if glob_half[h][st] else -1)
         for h in (0, 1)}
    vbest = max((st for st, _ in have if val.get(st) is not None and st > 0), key=lambda st: val[st], default=None)
    res["global_split_pick"] = g
    res["val_best_step"] = vbest
    lines.append(f"\nGlobal split-half pick (mean over sets): step {g[0]} (chosen on half 0) / step {g[1]} (half 1); "
                 f"val-best step (excl. 0): {vbest}" + ("" if vbest in (g[0], g[1]) else "  ← DISAGREES with the test pick"))
    print("\n".join(lines))
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
