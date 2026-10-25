"""BL tables (CPU): budget-matched text (S2), LLMLingua (S4) and reader-KV compression (S5) rows next to the same
reader's full text row and our kv@0, on the SAME item keys (intersection), raw scores only (no difference columns).

Cost columns:
  reader tokens / query  mean prompt tokens the reader prefills per question (text: the whole document)
  storage / doc          S5: kept K,V of the reader's full-attention layers (bf16) + DeltaNet state (as held);
                         ours: encoder operator = 20,480 B / token (9B map, 5 layers x K,V x 4 heads x 256 x bf16)

    python scripts/bl_table.py --prefix BL_pilot --sets mcq_bamboo,clipper           # pilot dirs
    python scripts/bl_table.py --prefix BL_9b                                          # full 9B runs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

H, L = "H100", "L40S"
# set -> (text row (file, cond, gpu), our kv@0 row (file, gpu), metric)
REF = {
    "longhealth": (("outputs/D13L1f_lh_f0/eval_test.json", "text", H), ("outputs/DSsw_ABbest_longhealth/eval_test.json", L), "acc"),
    "quality": (("outputs/D13L1f_quality/eval_test_full.json", "text", H), ("outputs/DSsw_ABlast_quality/eval_test.json", L), "acc"),
    "contractnli": (("outputs/F1L1f_cnli_s0/eval_test.json", "text", L), ("outputs/DSsw_ABlast_contractnli/eval_test.json", L), "acc"),
    "shakespeare_cf": (("outputs/ADDRj_L1f_scf/eval_test.json", "text", L), ("outputs/DSsw_ABlast_shakespeare_cf/eval_test.json", L), "acc"),
    "clipper": (("outputs/H2_L1f_clipper/eval_test.json", "text", L), ("outputs/DSsw_ABlast_clipper/eval_test.json", L), "acc"),
    "qasper": (("outputs/D22_qasper/eval_test.json", "text", H), ("outputs/DSsw_ABlast_qasper/eval_test.json", L), "F1"),
    "musique": (("outputs/D22_musique/eval_test.json", "text", H), ("outputs/DSsw_ABlast_musique/eval_test.json", L), "F1"),
    "lastingbench": (("outputs/D13L1f_lastingbench/eval_test.json", "text", H), ("outputs/DSsw_ABlast_lastingbench/eval_test.json", L), "F1"),
    "loogle": (("outputs/D22_loogle/eval_test.json", "text", H), ("outputs/DSsw_ABlast_loogle/eval_test.json", L), "F1"),
    "loogle_long": (("outputs/H2_L1f_loogle_long/eval_test.json", "text", L), ("outputs/DSsw_ABlast_loogle_long/eval_test.json", L), "F1"),
    "mcq_leval": (("outputs/MCQ_L1f_mcq_leval/eval_test.json", "text", L), ("outputs/DSsw_ABbest_mcq_leval/eval_test.json", L), "acc"),
    "mcq_bamboo": (("outputs/MCQ_L1f_mcq_bamboo/eval_test.json", "text", L), ("outputs/DSsw_ABbest_mcq_bamboo/eval_test.json", L), "acc"),
    "mcq_hallu": (("outputs/MCQ_L1f_mcq_hallu/eval_test.json", "text", L), ("outputs/DSsw_ABbest_mcq_hallu/eval_test.json", L), "acc"),
    "mcq_musr": (("outputs/MCQ_L1f_mcq_musr/eval_test.json", "text", L), ("outputs/DSsw_ABbest_mcq_musr/eval_test.json", L), "acc"),
    "d25_fecf": (("outputs/D25_reg_fecf_base/eval_test.json", "text", "?"), ("", "?"), "acc"),
    "d25_cqacf": (("outputs/D25_diag_9b/d25_cqa_test.ckpt.jsonl", "text", "?"), ("", "?"), "acc"),
}
OP_BYTES_PER_TOKEN = 20480


def rows_of(path):
    if not path or not os.path.exists(path):
        return {}
    if path.endswith(".jsonl"):
        return {r["key"]: r for r in (json.loads(x) for x in open(path) if x.strip())}
    return {r["key"]: r for r in json.load(open(path))["items"]}


def sc(r, cond):
    if cond == "kv" and "kv" not in r and "kv@0" in r:
        cond = "kv@0"
    v = r.get(cond)
    return v["score"] if isinstance(v, dict) and "score" in v else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def fmt(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="BL_9b")
    ap.add_argument("--sets", default=",".join(REF))
    ap.add_argument("--json", default="")
    ap.add_argument("--alias", default="", help="set=dirtag pairs, e.g. mcq_bamboo=bamboo (pilot dir names)")
    ap.add_argument("--kv_extra", default="", help="label=path_template with {set}, extra kv@0 rows (e.g. s6000=outputs/DSsw_m8k6000_{set}/eval_test.json)")
    ap.add_argument("--no_ref_kv", action="store_true", help="27B runs: drop the 9B DS_AB kv@0 row (use --kv_extra for the 27B kv@0 refs, "
                    "e.g. c5000=outputs/D12b_max13k_c5000_{set}/eval_test.json,cont21k=...)")
    ap.add_argument("--op_bytes", type=int, default=OP_BYTES_PER_TOKEN, help="operator bytes/token; 27B pipeline (9B encoder, unique "
                    "read layers 3,7,11,15,19,23 = 6 x K,V x 4 heads x 256 x bf16) = 24576")
    a = ap.parse_args(argv)
    extras = [x.split("=", 1) for x in a.kv_extra.split(",") if x]
    out, md = {}, []
    alias = dict(x.split("=") for x in a.alias.split(",") if x)
    for s in [x for x in a.sets.split(",") if x]:
        (tf, tc, tg), (kf, kg), metric = REF[s]
        tag = alias.get(s, s)
        tbp, kvp = f"outputs/{a.prefix}_tb_{tag}/eval_test.json", f"outputs/{a.prefix}_kvc_{tag}/eval_test.json"
        tb, kvc = rows_of(tbp), rows_of(kvp)
        if not tb and not kvc:
            continue
        text, kv = rows_of(tf), rows_of(kf)
        keys = set(tb or kvc) & (set(kvc) if kvc and tb else set(tb or kvc))
        keys &= set(text) if text else keys
        keys = sorted(keys)
        tbc = [c for c in (next(iter(tb.values())) if tb else {}) if isinstance((next(iter(tb.values())))[c], dict)]
        kvcc = [c for c in (next(iter(kvc.values())) if kvc else {}) if isinstance((next(iter(kvc.values())))[c], dict)]
        tbg = json.load(open(tbp)).get("gpu", "?") if tb else "?"
        kvg = json.load(open(kvp)).get("gpu", "?") if kvc else "?"
        if tb and all("text" in tb[k] for k in tb):          # in-job text row (same GPU as the BL rows)
            text, tc, tg = tb, "text", f"in-job {tbg}"
            keys = sorted(set(tb) & (set(kvc) if kvc else set(tb)))
        res = {"n": len(keys), "metric": metric, "text_ref": (mean(sc(text[k], tc) for k in keys), tg),
               "kv@0": (mean(sc(kv[k], "kv") for k in keys if k in kv), kg, sum(k in kv for k in keys))}
        for c in tbc:
            res[c] = {"score": mean(sc(tb[k], c) for k in keys),
                      "reader_tokens": mean(tb[k][c]["ctx_tokens"] for k in keys),
                      "frac_full": mean(tb[k][c].get("full", 0) for k in keys)}
        for c in kvcc:
            res[c] = {"score": mean(sc(kvc[k], c) for k in keys),
                      "storage_MB": mean((kvc[k][c]["kv_bytes"] + kvc[k][c]["linear_state_bytes"]) / 2 ** 20 for k in keys)}
        if "text" in tbc:
            same = sum(tb[k]["text"]["gen"] == text[k]["text"]["gen"] for k in keys if "text" in text[k])
            res["sanity_text_identical_gen"] = f"{same}/{len(keys)}"
        if "kvc_full" in kvcc and "text" in tbc:
            same = sum(kvc[k]["kvc_full"]["gen"] == tb[k]["text"]["gen"] for k in keys)
            res["sanity_kvc_full_vs_text_identical_gen"] = f"{same}/{len(keys)}"
        dtok = mean(tb[k][tbc[0]].get("doc_tokens") for k in keys) if tbc else None
        res["mean_doc_tokens"] = dtok
        out[s] = res
        md.append(f"\n### {s} (n={len(keys)}, {metric}; reader Qwen3.5-9B; S2/S4 on {tbg}; S5 on {kvg})")
        md.append("| row | score | reader tokens / query | storage / doc |")
        md.append("|---|---|---|---|")
        md.append(f"| text, full document ({tg}) | {fmt(res['text_ref'][0])} | {fmt(dtok, 0)} | doc text |")
        op = f"{dtok * a.op_bytes / 2 ** 20:.0f} MB" if dtok else "–"
        if not a.no_ref_kv:
            md.append(f"| **ours kv@0** ({kg}{'' if res['kv@0'][2] == len(keys) else ', n=' + str(res['kv@0'][2])}) "
                      f"| **{fmt(res['kv@0'][0])}** | 64-token tail + question | {op} (operator) |")
        for lab, tpl in extras:
            ex = rows_of(tpl.format(set=s))
            if ex:
                md.append(f"| kv@0 {lab} (n={sum(k in ex for k in keys)}) | {fmt(mean(sc(ex[k], 'kv') for k in keys if k in ex))} | 64-token tail + question | {op} (operator) |")
        for c in tbc:
            if c == "text":
                continue
            r = res[c]
            md.append(f"| {c} | {fmt(r['score'])} | {fmt(r['reader_tokens'], 0)} (full {r['frac_full']:.2f}) | doc text |")
        for c in kvcc:
            r = res[c]
            md.append(f"| S5 {c} [reader prefills doc] | {fmt(r['score'])} | question only | {r['storage_MB']:.0f} MB |")
        if "sanity_text_identical_gen" in res:
            md.append(f"\nsanity: BL text re-run vs the existing text row, identical generations "
                      f"{res['sanity_text_identical_gen']}; kvc_full vs BL text {res.get('sanity_kvc_full_vs_text_identical_gen', '–')}")
    print("\n".join(md))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
