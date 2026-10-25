"""CMP: offline per-question ROUTING ORACLE between our kv@0 (DSsw_ABlast / DSsw_ABbest) and Hub Memory (HM) per-item rows.

NOTE: this script only READS the HM directory (/nfs/turbo/umms-drjieliu/usr/xiangzhg/kvshill, another user's tree);
it never writes there. CPU only.

Item matching: HM row "id" == our item "key" (HM's data/xenc/<set>.json is built from our eval_test.json rows).
HM per-item score = field "xs" (our xenc.doc_tasks.score on the generated "resp"); re-scored here as a consistency check.

    PYTHONPATH=$PWD:$PWD/../delta-skill python scripts/cmp_route_oracle.py [--out outputs/CMP_route_oracle.json]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
HM = "/nfs/turbo/umms-drjieliu/usr/xiangzhg/kvshill"            # READ ONLY
RX = f"{HM}/runs/xenc"
import ds_sweep_table as dst  # noqa: E402
from xenc.doc_tasks import score as our_score  # noqa: E402

dst.LEAK = os.path.join(REPO, "outputs/DS_judged_leak_max8k.json")
os.chdir(REPO)

SETS = ["musique", "qasper", "lastingbench", "quality", "longhealth", "contractnli", "loogle", "loogle_long",
        "shakespeare_cf", "clipper", "mcq_leval", "mcq_bamboo", "mcq_hallu", "mcq_musr"]
FORMAT_OURS = {"quality", "longhealth", "contractnli", "shakespeare_cf", "clipper",
               "mcq_leval", "mcq_bamboo", "mcq_hallu", "mcq_musr"}   # choices / judgment sets -> ours; span sets -> HM
# HM "of record" numbers (outputs/HM_vs_ours.md; hm_R3_report.py OURS table; hm_R3_tax.py READOUT)
OFFICIAL = {"loogle": .575, "loogle_long": .546, "qasper": .575, "lastingbench": .450, "musique": .472, "clipper": .772,
            "shakespeare_cf": .660, "quality": .656, "longhealth": .713, "contractnli": .693}
# candidate per-item HM files that exist on disk; (label, path relative to RX)
HM_FILES = {
    "quality": [("memf1 mem (official .656)", "memf1-9b-quality/gen_mem.jsonl"),
                ("lhval mem (replicate)", "lhval-9b-quality/gen_mem.jsonl"),
                ("ragf1 hub_8 (alt)", "W2/ragf1-9b-quality/gen_mem_hub_8.jsonl")],
    "longhealth": [("ragf1 hub_8 (official .713)", "W2/ragf1-9b-longhealth/gen_mem_hub_8.jsonl"),
                   ("ragf1 bm25_8 (alt .688)", "W2/ragf1-9b-longhealth/gen_mem_bm25_8.jsonl")],
    "contractnli": [("ragf1 hub_16 (PARTIAL n=45, not the n=300 official)", "W2/ragf1-9b-contractnli/gen_mem_hub_16.jsonl")],
    "loogle": [("W1 ragfull-9bE bm25_8 (NOT the official .575 row)", "W1/ragfull-9bE-loogle/gen_mem_bm25_8.jsonl")],
    "shakespeare_cf": [("memf1 mem (NOT the official .660 row)", "memf1-9b-shakespeare_cf/gen_mem.jsonl"),
                       ("W1 ragfull-9bE bm25_8 (NOT official)", "W1/ragfull-9bE-shakespeare_cf/gen_mem_bm25_8.jsonl")],
}
# index of the candidate used as "the" HM row per set (None -> no usable official-config row)
PRIMARY = {"quality": 0, "longhealth": 0, "contractnli": 0, "loogle": None, "shakespeare_cf": None}


def mse(xs):
    n = len(xs)
    if not n:
        return None
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(n - 1, 1))
    return {"mean": m, "se": sd / math.sqrt(n), "n": n}


def load_hm(rel):
    p = f"{RX}/{rel}"
    if not os.path.exists(p):
        return None
    out = {}
    for line in open(p):
        r = json.loads(line)
        out.setdefault(r["id"], r)          # resumed runs append duplicates: keep the first row (as HM's own scripts do)
    return out


def hm_items(task):
    p = f"{HM}/data/xenc/{task}.json"
    return {r["id"]: r for r in json.load(open(p))} if os.path.exists(p) else {}


def route_stats(ours, hm, text, keys):
    keys = sorted(keys)
    o = [ours[k] for k in keys]
    h = [hm[k] for k in keys]
    orc = [max(a, b) for a, b in zip(o, h)]
    res = {"n": len(keys), "ours": mse(o), "hm": mse(h), "oracle": mse(orc),
           "oracle_minus_ours": mse([x - y for x, y in zip(orc, o)]), "hm_minus_ours": mse([y - x for x, y in zip(o, h)])}
    if text:
        res["text"] = mse([text[k] for k in keys])
    res["mean_min"] = sum(min(a, b) for a, b in zip(o, h)) / len(keys)
    n = len(keys)
    both = sum(a >= .5 and b >= .5 for a, b in zip(o, h)) / n
    oo = sum(a >= .5 and b < .5 for a, b in zip(o, h)) / n
    ho = sum(a < .5 and b >= .5 for a, b in zip(o, h)) / n
    res["agree"] = {"both": both, "ours_only": oo, "hm_only": ho, "neither": 1 - both - oo - ho}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "outputs/CMP_route_oracle.json"))
    a = ap.parse_args(argv)
    R = {"hm_dir_read_only": HM, "official": OFFICIAL, "item_id_mapping": "HM row['id'] == our item['key']",
         "leak_file": dst.LEAK, "sets": {}}
    for s in SETS:
        refs, (tf, _g) = dst.REFS[s]
        text = dst.scores(tf, "text")
        ck = dst.clean_keys(s)
        entry = {"text_file": tf, "clean_subset": ck is not None, "n_clean_keys": len(ck) if ck else None,
                 "format_route_uses": "ours" if s in FORMAT_OURS else "hm", "hm_candidates": [], "ours_rows": {}}
        ours = {}
        for tag in ("ABlast", "ABbest"):
            p = f"outputs/DSsw_{tag}_{s}/eval_test.json"
            sc = dst.scores(p)
            if sc:
                ours[tag] = sc
                entry["ours_rows"][tag] = {"file": p, "n_all": len(sc), "mean_all": sum(sc.values()) / len(sc)}
        entry["official_hm"] = OFFICIAL.get(s)
        cands = []
        for label, rel in HM_FILES.get(s, []):
            h = load_hm(rel)
            if h is None:
                continue
            xs = {k: r["xs"] for k, r in h.items() if "xs" in r}
            mean_xs = sum(xs.values()) / len(xs)
            c = {"label": label, "file": f"{RX}/{rel}", "n_rows": len(h), "mean_xs": mean_xs,
                 "matches_official": OFFICIAL.get(s) is not None and abs(mean_xs - OFFICIAL[s]) < .002
                 and len(h) >= 0.9 * (len(ours.get("ABlast", {})) or 1)}
            items = hm_items(s)
            if items:   # consistency: re-score the stored generations with our scorer
                rs = {k: our_score(items[k], r["resp"]) for k, r in h.items() if k in items}
                c["rescore_mean"] = sum(rs.values()) / len(rs)
                c["rescore_identical_frac"] = sum(abs(rs[k] - xs[k]) < 1e-9 for k in rs) / len(rs)
            c["_xs"] = xs
            cands.append(c)
        for i, c in enumerate(cands):
            xs = c.pop("_xs")
            c["primary"] = PRIMARY.get(s) == i
            c["by_ours"] = {}
            for tag, o in ours.items():
                keys = set(o) & set(xs)
                if ck is not None:
                    keys &= ck
                if text:
                    keys &= set(text)
                if not keys:
                    continue
                st = route_stats(o, xs, text, keys)
                st["format_route"] = mse([(o[k] if s in FORMAT_OURS else xs[k]) for k in sorted(keys)])
                c["by_ours"][tag] = st
            entry["hm_candidates"].append(c)
        if not cands:
            entry["hm_status"] = ("NO per-item HM file readable on disk for this set (HM runs were on Lighthouse/armis2 "
                                  "and are not synced here; glf1/ragf1 jobs still pending as of 2026-10-23)")
        for tag, o in ours.items():
            keys = set(o) & (ck if ck is not None else set(o))
            if text:
                keys &= set(text)
            entry["ours_rows"][tag]["clean_n"] = len(keys)
            entry["ours_rows"][tag]["clean_mean"] = mse([o[k] for k in keys])
            if text:
                entry["ours_rows"][tag]["text_mean_same_items"] = mse([text[k] for k in keys])
        R["sets"][s] = entry
    json.dump(R, open(a.out, "w"), indent=1)
    print("| set/ours | HM file | n | text | ours | HM | oracle | oracle-ours +- SE | HM-ours +- SE | format-route | both/ours-only/HM-only/neither |")
    print("|" + "---|" * 11)
    for s, e in R["sets"].items():
        for c in e["hm_candidates"]:
            for tag, st in c["by_ours"].items():
                f = lambda d: f"{d['mean']:.3f}"  # noqa: E731
                om, hmo, ag = st["oracle_minus_ours"], st["hm_minus_ours"], st["agree"]
                print(f"| {s} {tag} | {c['label'][:30]} | {st['n']} | {f(st['text']) if 'text' in st else '-'} | {f(st['ours'])} | "
                      f"{f(st['hm'])} | {f(st['oracle'])} | {om['mean']:+.3f} +- {om['se']:.3f} | {hmo['mean']:+.3f} +- {hmo['se']:.3f} | "
                      f"{st['format_route']['mean']:.3f} | {ag['both']:.2f}/{ag['ours_only']:.2f}/{ag['hm_only']:.2f}/{ag['neither']:.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
