"""RECON: kv@0 sweep table (CPU, from per-item eval json). Per judge set, raw kv@0 of the 9B reader (L40S) for DS_AB_full
(last = DSsw_ABlast_*, best where a row exists) and the RECON arm's checkpoints (DSsw_RC<ARM>_s<step>_*), the matched
no-RECON control DS_max8k at the same step when its sweep exists (DSsw_m8k<step>_*), and the text row (GPU noted).
Items = the clean subset (outputs/DS_judged_leak_max8k.json) ∩ every row shown; paired per-item SE (sd/√n).

    python scripts/recon_table.py --arm A [--out outputs/RECON_A_table.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import ds_sweep_table as D

SETS = ["musique", "lastingbench", "qasper", "loogle", "quality", "shakespeare_cf", "mcq_hallu", "mcq_leval", "mcq_bamboo"]
BEST = {"qasper": "outputs/ADDRsw_DSAB_qasper/eval_test.json"}
BEST.update({s: f"outputs/DSsw_ABbest_{s}/eval_test.json" for s in ("mcq_hallu", "mcq_leval", "mcq_bamboo")})


def gold_no(path: str, keys) -> tuple | None:
    """mcq_hallu: kv@0 accuracy on gold-No (B) items (labels from the set's test split)."""
    if not path or not os.path.exists(path):
        return None
    lab = {it["key"]: it["correct_label"] for it in json.load(open("data/mcq_hallu_split/test/items.json"))}
    sc = D.scores(path)
    return D.ms([sc[k] for k in keys if k in sc and lab.get(k) == "B"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A", help="comma list; later arms also get a paired diff vs the FIRST arm")
    ap.add_argument("--steps", default="1000,2000,3000")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    D.LEAK = "outputs/DS_judged_leak_max8k.json"
    steps = [int(x) for x in a.steps.split(",")]
    res, lines = {}, []
    arms = a.arm.split(",")
    cols = ["ABlast", "ABbest"] + [f"RC{x} s{s}" for x in arms for s in steps] + [f"m8k s{s}" for s in steps]
    for s in SETS:
        _, (tf, tg) = D.REFS[s]
        rows = {"ABlast": D.scores(f"outputs/DSsw_ABlast_{s}/eval_test.json"),
                "ABbest": D.scores(BEST[s]) if s in BEST else {}}
        for st in steps:
            for x in arms:
                rows[f"RC{x} s{st}"] = D.scores(f"outputs/DSsw_RC{x}_s{st}_{s}/eval_test.json")
            rows[f"m8k s{st}"] = D.scores(f"outputs/DSsw_m8k{st}_{s}/eval_test.json")
        text = D.scores(tf, "text")
        rows = {k: v for k, v in rows.items() if v}
        if not rows.get("ABlast") or not any(k.startswith("RC") for k in rows):
            continue
        keys = set.intersection(*[set(v) for v in rows.values()]) & set(text)
        ck = D.clean_keys(s)
        sub = "all"
        if ck is not None:
            keys &= ck
            sub = "clean"
        r = {"subset": sub, "n": len(keys), "text": D.ms([text[k] for k in keys])[:2], "text_gpu": tg, "rows": {}}
        for c, v in rows.items():
            m = D.ms([v[k] for k in keys])
            d = D.ms([v[k] - rows["ABlast"][k] for k in keys]) if c != "ABlast" else None
            r["rows"][c] = {"kv": m[:2], "minus_ABlast": d and d[:2]}
        for st in steps:                                  # matched no-RECON control at the same step
            for x in arms:
                c, k0, k1 = f"RC{x} s{st}", f"m8k s{st}", f"RC{arms[0]} s{st}"
                if c in rows and k0 in rows:
                    r["rows"][c]["minus_m8k"] = D.ms([rows[c][k] - rows[k0][k] for k in keys])[:2]
                if x != arms[0] and c in rows and k1 in rows:   # arm vs the first arm, same step
                    r["rows"][c]["minus_arm0"] = D.ms([rows[c][k] - rows[k1][k] for k in keys])[:2]
        if s == "mcq_hallu":
            paths = {"ABlast": f"outputs/DSsw_ABlast_{s}/eval_test.json", "ABbest": BEST[s]}
            for st in steps:
                for x in arms:
                    paths[f"RC{x} s{st}"] = f"outputs/DSsw_RC{x}_s{st}_{s}/eval_test.json"
                paths[f"m8k s{st}"] = f"outputs/DSsw_m8k{st}_{s}/eval_test.json"
            r["gold_no"] = {c: (gold_no(paths[c], keys) or (None,))[0] for c in rows}
            r["gold_no_n"] = (gold_no(paths["ABlast"], keys) or (0, 0, 0))[2]
        res[s] = r
    hdr = "| set (subset n) | text 9B (GPU) | " + " | ".join(c for c in cols if any(c in v["rows"] for v in res.values())) + " |"
    shown = [c for c in cols if any(c in v["rows"] for v in res.values())]
    lines += [hdr, "|" + "---|" * (2 + len(shown))]
    for s, r in res.items():
        cells = []
        for c in shown:
            x = r["rows"].get(c)
            if not x:
                cells.append("–")
                continue
            t = f"{x['kv'][0]:.3f}"
            if x.get("minus_ABlast"):
                t += f" ({x['minus_ABlast'][0]:+.3f}±{x['minus_ABlast'][1]:.3f})"
            if x.get("minus_arm0"):
                t += f" {{vs {arms[0]} {x['minus_arm0'][0]:+.3f}±{x['minus_arm0'][1]:.3f}}}"
            if x.get("minus_m8k"):
                t += f" [m8k {x['minus_m8k'][0]:+.3f}±{x['minus_m8k'][1]:.3f}]"
            cells.append(t)
        lines.append(f"| {s} ({r['subset']} {r['n']}) | {r['text'][0]:.3f} ({r['text_gpu']}) | " + " | ".join(cells) + " |")
    print("\n".join(lines))
    if "mcq_hallu" in res:
        print("mcq_hallu kv@0 on gold-No items: " + ", ".join(f"{c} {v:.3f}" for c, v in res["mcq_hallu"]["gold_no"].items()
                                                            if v is not None))
    json.dump(res, open(a.out or f"outputs/RECON_{''.join(arms)}_table.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
