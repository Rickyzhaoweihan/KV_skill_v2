"""CMP: compression design-space table (CPU, from per-item eval json).

For every CMP arm (outputs/CMPsw_<tag>_<set>) and the uncompressed base (outputs/DSsw_<base>_<set>, or R2s_DSAB_ /
ADDRsw_DSAB_ for ABbest): kv@0 per set on the clean subsets (outputs/DS_judged_leak_max8k.json), paired vs the base and
vs text (text rows reused from existing item-identical runs, GPU named, scripts/ds_sweep_table.REFS), grouped into
categories; storage per token from the compressor file; HM's published rows (shared HM artifact) as the other endpoint.

    python scripts/cmp_table.py --arms all_128=outputs/CMP_fit/cmp_all_128.pt,... --base ABbest --out outputs/CMP_table.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ds_sweep_table as T  # noqa: E402

CATS = {"comprehension": ["mcq_leval", "mcq_bamboo", "quality", "longhealth"],
        "judgment": ["mcq_hallu", "shakespeare_cf", "contractnli"],
        "extractive": ["qasper", "loogle"],
        "multihop": ["musique", "lastingbench"]}
# HM ours-of-record (xiangzhg/kvshill/universal/hm_R3_report.py:33-37, as relayed in outputs/HM_vs_ours.md; their n / protocol:
# H200, BM25 latent-RAG on loogle / clipper, hub-RAG on SCF / LH, full sets incl. items our clean subsets drop)
HM = {"loogle": .575, "qasper": .575, "lastingbench": .450, "musique": .472, "shakespeare_cf": .660, "quality": .656,
      "longhealth": .688, "contractnli": .693}   # LH: HM_vs_ours.md .688 (hm_R3_report.py has .713)
HM_BPT = 256.0                      # 128 fp16 numbers per token
FULL_BPT = 5 * 2 * 4 * 256 * 2      # 5 read layers x (K,V) x 4 kv heads x 256 x bf16 = 20480
REF_TOK = 60000                     # reference document length for MB/doc (HM ~15 MB, ours ~1.2 GB)


def bpt(path: str, quant: str = "") -> float:
    if path in ("", "none", "full"):
        return float(FULL_BPT)
    from xenc.cmp_op import load_cmp
    m = load_cmp(path)
    return m.bytes_per_token(quant or None)


def base_file(base: str, s: str) -> str:
    cands = [f"outputs/DSsw_{base}_{s}/eval_test.json"]
    if base == "ABbest":
        cands += [f"outputs/R2s_DSAB_{s}/eval_test.json", f"outputs/ADDRsw_DSAB_{s}/eval_test.json"]
    return next((c for c in cands if os.path.exists(c)), cands[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="tag=compressor_file[@quant],...  (file 'none' = uncompressed rows of "
                    "outputs/CMPsw_<tag>_<set>)")
    ap.add_argument("--base", default="ABbest")
    ap.add_argument("--leak", default="outputs/DS_judged_leak_max8k.json")
    ap.add_argument("--out", default="outputs/CMP_table.json")
    a = ap.parse_args(argv)
    T.LEAK = a.leak
    sets = [s for c in CATS.values() for s in c]
    base = {s: T.scores(base_file(a.base, s)) for s in sets}
    text = {s: T.scores(T.REFS[s][1][0], "text") for s in sets}
    keys = {}
    for s in sets:
        k = set(base[s]) & set(text[s])
        ck = T.clean_keys(s)
        keys[s] = k & ck if ck is not None else k
    arms = [("base", "none", "")] + [(p.split("=")[0], *(p.split("=", 1)[1].split("@") + [""])[:2])
                                     for p in a.arms.split(",") if p]
    res = {"sets": {}, "arms": {}, "cats": CATS, "hm": HM, "ref_tokens": REF_TOK}
    for s in sets:
        res["sets"][s] = {"n": len(keys[s]), "text": T.ms([text[s][k] for k in keys[s]])[:2] if keys[s] else None,
                          "text_gpu": T.REFS[s][1][1], "base_file": base_file(a.base, s)}
    lines = []
    hdr = "| arm | B/token | MB / 60k-tok doc | " + " | ".join(sets) + " | " + " | ".join(CATS) + " |"
    lines += [hdr, "|" + "---|" * (3 + len(sets) + len(CATS))]
    for tag, f, q in arms:
        sc = base if tag == "base" else {s: T.scores(f"outputs/CMPsw_{tag}_{s}/eval_test.json") for s in sets}
        b = bpt(f, q)
        row = {"bytes_per_token": b, "MB_per_ref_doc": b * REF_TOK / 1e6, "sets": {}, "cats": {}}
        cells = []
        for s in sets:
            kk = keys[s] & set(sc[s])
            if not kk or len(kk) < len(keys[s]):
                cells.append("–" if not kk else f"({len(kk)}/{len(keys[s])})")
                continue
            m = T.ms([sc[s][k] for k in kk])
            d = T.ms([sc[s][k] - base[s][k] for k in kk])
            dt = T.ms([sc[s][k] - text[s][k] for k in kk])
            row["sets"][s] = {"kv": m[:2], "minus_base": d[:2], "minus_text": dt[:2], "n": len(kk)}
            cells.append(f"{m[0]:.3f}" + ("" if tag == "base" else f" ({d[0]:+.3f})"))
        for c, ss in CATS.items():
            got = [row["sets"][s] for s in ss if s in row["sets"]]
            if len(got) == len(ss):
                row["cats"][c] = {"kv": sum(g["kv"][0] for g in got) / len(got),
                                  "minus_base": [sum(g["minus_base"][0] for g in got) / len(got),
                                                 math.sqrt(sum(g["minus_base"][1] ** 2 for g in got)) / len(got)]}
        res["arms"][tag] = {"file": f, "quant": q, **row}
        cc = [f"{row['cats'][c]['kv']:.3f}" + ("" if tag == "base" else f" ({row['cats'][c]['minus_base'][0]:+.3f})")
              if c in row["cats"] else "–" for c in CATS]
        lines.append(f"| {tag} | {b:.0f} | {b * REF_TOK / 1e6:.1f} | " + " | ".join(cells) + " | " + " | ".join(cc) + " |")
    tl = [f"{res['sets'][s]['text'][0]:.3f}" if res["sets"][s]["text"] else "–" for s in sets]
    tc = []
    for c, ss in CATS.items():
        v = [res["sets"][s]["text"][0] for s in ss if res["sets"][s]["text"]]
        tc.append(f"{sum(v) / len(v):.3f}" if len(v) == len(ss) else "–")
    lines.append("| text (9B reads doc) | – | – | " + " | ".join(tl) + " | " + " | ".join(tc) + " |")
    hc = []
    for c, ss in CATS.items():
        v = [HM[s] for s in ss if s in HM]
        hc.append(f"{sum(v) / len(v):.3f} ({len(v)}/{len(ss)})" if v else "–")
    lines.append(f"| HM published (own protocol) | {HM_BPT:.0f} | {HM_BPT * REF_TOK / 1e6:.1f} | "
                 + " | ".join(f"{HM[s]:.3f}" if s in HM else "–" for s in sets) + " | " + " | ".join(hc) + " |")
    lines.append("")
    lines.append("n per set (clean): " + ", ".join(f"{s} {res['sets'][s]['n']}" for s in sets))
    print("\n".join(lines))
    res["markdown"] = "\n".join(lines)
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
