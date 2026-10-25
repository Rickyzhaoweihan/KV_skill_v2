"""DS: kv@0 sweep table (CPU, from per-item eval json). For each judge set: a DS checkpoint's unregistered kv@0
(eval_d13 sweep outputs/DSsw_<tag>_<set>, or another agent's row of the same checkpoint) vs lever-1 full's kv@0 rows
(each labelled with its GPU type) and text, paired per item on the common keys, with MuSiQue / LastingBench restricted
to the strictly-clean subset (outputs/DS_judged_leak.json: items failing vs the lever-1 lineage OR the DS-new docs).

    python scripts/ds_sweep_table.py --tag ABlast [--out outputs/DS_sweep_ABlast.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

H, L = "H100", "L40S"
REFS = {   # set -> lever-1 full kv@0 rows [(label, file, gpu)], text source (file, gpu)
    "musique": ([("L1f s0", "outputs/D13L1f_musique/eval_test.json", H), ("L1f s1", "outputs/D13L1s1_musique/eval_test.json", H)],
                ("outputs/D22_musique/eval_test.json", H)),
    "qasper": ([("L1f s0", "outputs/D13L1f_qasper/eval_test.json", H), ("L1f s1", "outputs/D13L1s1_qasper/eval_test.json", H)],
               ("outputs/D22_qasper/eval_test.json", H)),
    "lastingbench": ([("L1f s0", "outputs/D13L1f_lastingbench/eval_test.json", H),
                      ("L1f s1", "outputs/D13L1s1_lastingbench/eval_test.json", H)], ("outputs/D13L1f_lastingbench/eval_test.json", H)),
    "loogle": ([("L1f s0", "outputs/D13L1f_loogle/eval_test.json", H), ("L1f s1", "outputs/D13L1s1_loogle/eval_test.json", H)],
               ("outputs/D22_loogle/eval_test.json", H)),
    "loogle_long": ([("L1f s0", "outputs/D13L1f_loogle_long/eval_test.json", H),
                     ("L1f s1", "outputs/D13L1s1_loogle_long/eval_test.json", H),
                     ("L1f s0", "outputs/H2_L1f_loogle_long/eval_test.json", L)], ("outputs/H2_L1f_loogle_long/eval_test.json", L)),
    "quality": ([("L1f s0", "outputs/D13L1f_quality/eval_test_full.json", H),
                 ("L1f s1", "outputs/D13L1s1_quality/eval_test_full.json", H)], ("outputs/D13L1f_quality/eval_test_full.json", H)),
    "longhealth": ([("L1f s0", "outputs/D13L1f_lh_f0/eval_test.json", H), ("L1f s1", "outputs/D13L1s1_lh_f0/eval_test.json", H),
                    ("L1f s0", "outputs/DSsw_L1f_longhealth/eval_test.json", L)], ("outputs/D13L1f_lh_f0/eval_test.json", H)),
    "contractnli": ([("L1f s0", "outputs/F1L1f_cnli_s0/eval_test.json", L)], ("outputs/F1L1f_cnli_s0/eval_test.json", L)),
    "shakespeare_cf": ([("L1f s0", "outputs/ADDRj_L1f_scf/eval_test.json", L)], ("outputs/ADDRj_L1f_scf/eval_test.json", L)),
    "clipper": ([("L1f s0", "outputs/H2_L1f_clipper/eval_test.json", L)], ("outputs/H2_L1f_clipper/eval_test.json", L)),
}
for _m in ("mcq_leval", "mcq_bamboo", "mcq_hallu", "mcq_musr"):
    REFS[_m] = ([("L1f s0", f"outputs/MCQ_L1f_{_m}/eval_test.json", L), ("L1f s0", f"outputs/DSsw_L1f_{_m}/eval_test.json", L)],
                (f"outputs/MCQ_L1f_{_m}/eval_test.json", L))
OTHER = {"ABbest": ["outputs/R2s_DSAB_{s}/eval_test.json", "outputs/ADDRsw_DSAB_{s}/eval_test.json"]}


def scores(path: str, cond: str = "kv") -> dict:
    if not os.path.exists(path):
        return {}
    E = json.load(open(path))
    out = {}
    for it in E["items"]:
        c = cond
        if cond == "kv":
            c = "kv@0" if "kv@0" in it else "kv"
        if c in it and isinstance(it[c], dict):
            out[it["key"]] = it[c]["score"]
    return out


def ms(xs):
    n = len(xs)
    if not n:
        return None
    m = sum(xs) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / max(n - 1, 1)) / math.sqrt(n), n


LEAK = "outputs/DS_judged_leak_all.json"     # strict judged-vs-training check over all sets (falls back to MuSiQue/LB file)


def _bad(v: dict, thr: float, use_q: bool) -> bool:
    return v["c_jT"] >= thr or v["c_tj"] >= thr or v["hash"] or (use_q and v["q"])


def clean_keys(task: str):
    """Items clean vs the lever-1 lineage AND the DS-new docs. ContractNLI asks the SAME 17 hypotheses of every NDA, so
    a question match is true by construction there: only the text criteria count for it."""
    f = LEAK if os.path.exists(LEAK) else "outputs/DS_judged_leak.json"
    if not os.path.exists(f):
        return None
    R = json.load(open(f))
    if task not in R:
        return None
    thr, P = R.get("thr", 0.15), R[task]["per_item"]
    use_q = task != "contractnli"
    ck = {k for k, v in P.items() if not _bad(v["lineage"], thr, use_q) and not _bad(v["ds_new"], thr, use_q)}
    return None if len(ck) == len(P) else ck


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ABlast")
    ap.add_argument("--out", default="")
    ap.add_argument("--leak", default="", help="judged-leakage json for the clean subsets (default DS_judged_leak_all; "
                    "use outputs/DS_judged_leak_max8k.json for DS_max8k checkpoints)")
    a = ap.parse_args(argv)
    global LEAK
    if a.leak:
        LEAK = a.leak
    res, lines = {}, []
    hdr = (f"| set | subset n | text (GPU) | **DS {a.tag} kv@0** (L40S) | lever-1 full kv@0 rows (GPU) | "
           f"DS − L1f (paired, per row) | DS kv − text | DS wrong_doc |")
    lines += [hdr, "|" + "---|" * 8]
    for s, (refs, (tf, tg)) in REFS.items():
        dsp = f"outputs/DSsw_{a.tag}_{s}/eval_test.json"
        if not os.path.exists(dsp) and a.tag in OTHER:
            dsp = next((p.format(s=s) for p in OTHER[a.tag] if os.path.exists(p.format(s=s))), dsp)
        ds = scores(dsp)
        if not ds:
            continue
        wr = scores(dsp, "wrong_doc")
        text = scores(tf, "text")
        keys = set(ds)
        ck = clean_keys(s)
        sub = "all"
        if ck is not None:
            keys &= ck
            sub = "clean"
        keys &= set(text) if text else keys
        m = ms([ds[k] for k in keys])
        t = ms([text[k] for k in keys]) if text else None
        dt = ms([ds[k] - text[k] for k in keys]) if text else None
        w = ms([wr[k] for k in keys if k in wr])
        rr, dd = [], []
        for lab, f, g in refs:
            r = scores(f)
            kk = keys & set(r)
            if not kk:
                continue
            rr.append(f"{lab} {ms([r[k] for k in kk])[0]:.3f} ({g}{'' if len(kk) == len(keys) else f', n={len(kk)}'})")
            d = ms([ds[k] - r[k] for k in kk])
            dd.append(f"{d[0]:+.3f} ± {d[1]:.3f} vs {lab} {g}")
            res.setdefault(s, {}).setdefault("refs", []).append({"label": lab, "gpu": g, "file": f, "n": len(kk),
                                                                  "ref_kv": ms([r[k] for k in kk])[0], "diff": d[:2]})
        res.setdefault(s, {}).update({"ds_file": dsp, "subset": sub, "n": len(keys), "ds_kv": m[:2], "text": t and t[:2],
                                      "text_gpu": tg, "kv_minus_text": dt and dt[:2], "wrong": w and w[:2]})
        lines.append(f"| {s} | {sub} {len(keys)} | {t[0]:.3f} ({tg}) | **{m[0]:.3f}** | {'; '.join(rr)} | {'; '.join(dd)} | "
                     f"{dt[0]:+.3f} ± {dt[1]:.3f} | {w[0]:.3f} |" if t else f"| {s} | {sub} {len(keys)} | – | **{m[0]:.3f}** | "
                     f"{'; '.join(rr)} | {'; '.join(dd)} | – | {w[0]:.3f} |")
    print("\n".join(lines))
    json.dump(res, open(a.out or f"outputs/DS_sweep_{a.tag}.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
