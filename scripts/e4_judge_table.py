"""E4: independent recompute of the judge table (CPU) from raw per-item generations.

For each benchmark: the baseline eval file (D22 / D20) and E4 eval files (same adapters or E4 registrations). Scores
are recomputed from the stored generations with xenc.doc_tasks.score (not the job's own summary). kv = es≤200
(per-document early stop by held-out self-question loss, choice read from each file's es_choice); wrong_doc likewise;
text from the baseline file. Paired se = sd/√n over the shared item keys.

    python scripts/e4_judge_table.py            -> prints markdown (also outputs/E4_judge_table.md)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BENCH = {   # name: (split_dir, baseline eval file, {row label: E4 eval file})
    "MuSiQue": ("data/musique_split", "outputs/D22_musique/eval_test.json",
                {"E4 base (repro, D22 adapters)": "outputs/E4_ev_base_musique/eval_test.json",
                 "E4 temp1.5 (D22 adapters, eval-only)": "outputs/E4_ev_temp1.5_musique/eval_test.json",
                 "E4 base REGISTERED (L40S, same job)": "outputs/E4_reg_base_musique/eval_test.json",
                 "E4 temp1.25 REGISTERED (val-chosen)": "outputs/E4_reg_temp1.25_musique/eval_test.json"}),
    "LastingBench": ("data/lastingbench_split", "outputs/D22_lastingbench/eval_test.json",
                     {"E4 base (repro, D22 adapters)": "outputs/E4_ev_base_lastingbench/eval_test.json",
                      "E4 temp1.5 (D22 adapters, eval-only)": "outputs/E4_ev_temp1.5_lastingbench/eval_test.json",
                      "E4 base REGISTERED (L40S, same job)": "outputs/E4_reg_base_lastingbench/eval_test.json",
                 "E4 temp1.25 REGISTERED (val-chosen)": "outputs/E4_reg_temp1.25_lastingbench/eval_test.json"}),
    "QuALITY": ("data/quality_split", "outputs/D20_4b_quality/eval_test_full.json",
                {"E4 base (repro, D20 adapters)": "outputs/E4_ev_base_quality/eval_test_full.json",
                 "E4 temp1.5 (D20 adapters, eval-only)": "outputs/E4_ev_temp1.5_quality/eval_test_full.json"}),
}
ES = 200
# same-GPU-type reference for the Δ column (GPU-type matching rule): temp rows vs the base row of the SAME job
REF = {"E4 temp1.5 (D22 adapters, eval-only)": "E4 base (repro, D22 adapters)",
       "E4 temp1.5 (D20 adapters, eval-only)": "E4 base (repro, D20 adapters)",
       "E4 temp1.25 REGISTERED (val-chosen)": "E4 base REGISTERED (L40S, same job)"}


def ms(xs):
    n = len(xs)
    m = sum(xs) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / max(n - 1, 1)) / math.sqrt(n)


def rescored(path, items_by_key, ckpt_gens=None):
    """{key: {"kv": s, "wrong": s, "text": s?}} at es≤ES, recomputed from generations."""
    from xenc.doc_tasks import score
    E = json.load(open(path))
    es = E.get("es_choice") or {}
    out = {}
    for r in E["items"]:
        it = items_by_key[r["key"]]
        b = int(es[r["doc_id"]][str(ES)]) if es else ES
        row = {}
        g = (ckpt_gens or {}).get(r["key"], r)
        if f"kv@{b}" in g and "gen" in g[f"kv@{b}"]:
            row["kv"] = float(score(it, g[f"kv@{b}"]["gen"]))
        if r.get("wrong") is not None:
            bw = int(es[r["wrong"]][str(ES)]) if es else ES
            if f"wrong_doc@{bw}" in g and "gen" in g[f"wrong_doc@{bw}"]:
                row["wrong"] = float(score(it, g[f"wrong_doc@{bw}"]["gen"]))
        if "text" in g and "gen" in g["text"]:
            row["text"] = float(score(it, g["text"]["gen"]))
        out[r["key"]] = row
    return out


def ckpt(path):
    p = path.replace(".json", ".ckpt.jsonl")
    if not os.path.exists(p):
        return None
    return {json.loads(line)["key"]: json.loads(line) for line in open(p)}


def main() -> int:
    from register_doc import _get, load_split_items
    lines = ["# E4 judge table (independent recompute from raw generations; kv/wrong = es≤200)", "",
             "| benchmark | row | GPU | n | kv | wrong_doc | text | kv−text ± se | kv−wrong ± se | Δkv vs baseline ± se (W/L) | "
             "identical-gen rate vs baseline | Δkv vs same-job base ± se (W/L) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, (sd, base, rows) in BENCH.items():
        if not os.path.exists(base):
            continue
        items = {_get(i, "key"): i for s in ("test", "val", "train") for i in load_split_items(sd, s)}
        bg = ckpt(base)
        B = rescored(base, items, bg)
        cands = [("baseline", base)] + [(k, v) for k, v in rows.items() if os.path.exists(v)]
        done = {}
        for lab, f in cands:
            g = ckpt(f)
            R = B if lab == "baseline" else rescored(f, items, g)
            done[lab] = R
            gpu = ",".join(x.replace("NVIDIA ", "").replace(" 80GB HBM3", "") for x in (json.load(open(f)).get("cost") or {}).get("gpu", []))
            ref = done.get(REF.get(lab, ""))
            if ref is not None:
                rr = [R[k]["kv"] - ref[k]["kv"] for k in R if "kv" in R[k] and "kv" in ref.get(k, {})]
                m_, s_ = ms(rr)
                d_ref = f"{m_:+.3f} ± {s_:.3f} ({sum(x > 0 for x in rr)}/{sum(x < 0 for x in rr)}; n={len(rr)})"
            else:
                d_ref = "—"
            keys = [k for k in R if "kv" in R[k] and "text" in B.get(k, {})]
            kv = [R[k]["kv"] for k in keys]
            tx = [B[k]["text"] for k in keys]
            wk = [k for k in keys if "wrong" in R[k]]
            d_t = ms([R[k]["kv"] - B[k]["text"] for k in keys])
            d_w = ms([R[k]["kv"] - R[k]["wrong"] for k in wk]) if wk else (float("nan"),) * 2
            if lab == "baseline":
                d_b, wl, ident = "—", "", "—"
            else:
                dd = [R[k]["kv"] - B[k]["kv"] for k in keys if "kv" in B.get(k, {})]
                m, s = ms(dd)
                d_b, wl = f"{m:+.3f} ± {s:.3f}", f" ({sum(x > 0 for x in dd)}/{sum(x < 0 for x in dd)})"
                same = 0
                tot = 0
                E = json.load(open(f))
                BE = {r["key"]: r for r in json.load(open(base))["items"]}
                gsrc, bsrc = g or {r["key"]: r for r in E["items"]}, bg or BE
                for k in keys:
                    for c in [c for c in gsrc[k] if c.startswith("kv@")]:
                        if c in bsrc.get(k, {}) and "gen" in gsrc[k][c] and "gen" in bsrc[k][c]:
                            tot += 1
                            same += gsrc[k][c]["gen"] == bsrc[k][c]["gen"]
                ident = f"{same / max(tot, 1):.3f} ({tot})"
            lines.append(f"| {name} | {lab} | {gpu} | {len(keys)} | {sum(kv) / len(kv):.3f} | "
                         f"{(sum(R[k]['wrong'] for k in wk) / len(wk)) if wk else float('nan'):.3f} | {sum(tx) / len(tx):.3f} | "
                         f"{d_t[0]:+.3f} ± {d_t[1]:.3f} | {d_w[0]:+.3f} ± {d_w[1]:.3f} | {d_b}{wl} | {ident} | {d_ref} |")
    txt = "\n".join(lines)
    print(txt)
    open("outputs/E4_judge_table.md", "w").write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
