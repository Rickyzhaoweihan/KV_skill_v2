"""D12b: per-set kv@0 / wrong_doc on ALL vs CLEAN items (judged items flagged by scripts/d12b_leakcheck.py removed),
for any eval_d12p output prefix. Recomputed from the per-item ckpt jsonl (independent of the judge's own summary).

    python scripts/d12b_clean_report.py --prefix outputs/D12b_max8k_c2000_ --flagged outputs/D12b_max8k_flagged.json \
        [--base_prefix outputs/D12b_lv4_]
"""
from __future__ import annotations

import argparse
import json
import math
import os


def load(p):
    f = os.path.join(p, "eval_test.ckpt.jsonl")
    return {json.loads(x)["key"]: json.loads(x) for x in open(f)} if os.path.exists(f) else None


def ms(xs):
    n = len(xs)
    m = sum(xs) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / max(n - 1, 1)) / math.sqrt(n)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--flagged", default="outputs/D12b_max8k_flagged.json")
    ap.add_argument("--base_prefix", default="", help="paired comparison (e.g. outputs/D12b_lv4_; MCQ sets: outputs/D12b_mcq_lv4_)")
    a = ap.parse_args(argv)
    F = json.load(open(a.flagged))["sets"]
    print("| set | subset | n | kv | wrong_doc | kv − base (paired) |")
    print("|---|---|---|---|---|---|")
    for s, v in F.items():
        E = load(a.prefix + s)
        if E is None:
            continue
        bad = set(v["flagged_keys"])
        bp = a.base_prefix + s
        if a.base_prefix and s.startswith("mcq_") and "mcq" not in a.base_prefix:
            bp = a.base_prefix.replace("D12b_", "D12b_mcq_") + s
        B = load(bp) if a.base_prefix else None
        for name, keys in (("all", list(E)), ("clean", [k for k in E if k not in bad])):
            if not keys:
                continue
            kv = sum(E[k]["kv"]["score"] for k in keys) / len(keys)
            wr = sum(E[k]["wrong_doc"]["score"] for k in keys if "wrong_doc" in E[k]) / len(keys)
            d = ""
            if B:
                ks = [k for k in keys if k in B]
                m, se = ms([E[k]["kv"]["score"] - B[k]["kv"]["score"] for k in ks])
                d = f"{m:+.3f} ± {se:.3f}"
            print(f"| {s} | {name} | {len(keys)} | {kv:.3f} | {wr:.3f} | {d} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
