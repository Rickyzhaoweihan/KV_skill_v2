"""CAD tables (agent CAD, CPU only): alpha selection on VALIDATION splits, then test tables from scripts/eval_cad.py.

Selection (one alpha per (mode, main, ref), as hm_A_cad.py:55-63 but on OUR val sets; FaithEval val is never used):
  val sets   v_cqacf (ConflictQA-CF val, acc), v_cqaor (ConflictQA ORIGINAL val, acc: do-no-harm), v_nqswap (NQ-Swap
             val, doc adherence), v_quality (QuALITY val docs 1-20, acc), v_qasper (Qasper val docs 1-16, span F1)
  criterion  mean over the val sets that have the mode (FC: MCQ sets only), ties -> smaller alpha.
Test rows: kv@0 vs kv+CAD (selected alpha) vs text vs text+CAD, raw scores, paired SE (sd/sqrt n); plus the test-only
alpha curve, labelled as a sensitivity analysis (never used for selection). Latency from the per-item timings.

    python scripts/cad_table.py --tag 9b
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

VAL = ["v_cqacf", "v_cqaor", "v_nqswap", "v_quality", "v_qasper"]
TEST = ["fecf", "cqacf", "nqswap", "scf", "quality", "mcq_leval", "mcq_hallu", "qasper", "musique"]
FC_AS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
NAMES = {"fecf": "FaithEval-CF", "cqacf": "ConflictQA-CF", "nqswap": "NQ-Swap (doc-adh)", "scf": "Shakespeare-CF",
         "quality": "QuALITY", "mcq_leval": "mcq_leval", "mcq_hallu": "mcq_hallu", "qasper": "Qasper",
         "musique": "MuSiQue"}
FLAG_KEY = {"fecf": "fecf", "cqacf": "cqacf", "qasper": "qasper", "scf": "shakespeare_cf", "mcq_leval": "mcq_leval",
            "quality": "q300"}


def ms(xs):
    n = len(xs)
    if not n:
        return None
    m = sum(xs) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / max(n - 1, 1)) / math.sqrt(n), n


def load(tag, s):
    for sp in ("val", "test"):
        p = f"outputs/CAD_{tag}_{s}/eval_{sp}.json"
        if os.path.exists(p):
            return json.load(open(p))
    return None


def gen_metric(rec, cond):
    g = rec["gen"].get(cond)
    if g is None:
        return None
    return g["doc_adh"] if "doc_adh" in g else g["score"]


def fc_metric(rec, main, ref, a):
    from xenc.cad import cad_pick
    fc = rec.get("fc")
    if not fc or main not in fc["lp"]:
        return None
    i = cad_pick(fc["lp"][main], fc["lp"][ref], a) if a else cad_pick(fc["lp"][main], fc["lp"][main], 0.0)
    return float(fc["labels"][i] in fc["gold"])


def gen_cond(main, ref, a):
    if a == 0:
        return main
    return f"{main}_cad_{'ct' if ref == 'closed_tail' else 'cb'}_{a:g}"


def gen_alphas(items, main, ref):
    rs = "ct" if ref == "closed_tail" else "cb"
    out = {0.0}
    for r in items:
        for c in r["gen"]:
            if c.startswith(f"{main}_cad_{rs}_"):
                out.add(float(c.rsplit("_", 1)[1]))
    return sorted(out)


def score_set(items, mode, main, ref, a, keys=None):
    xs = []
    for r in items:
        if keys is not None and r["key"] not in keys:
            continue
        v = fc_metric(r, main, ref, a) if mode == "fc" else gen_metric(r, gen_cond(main, ref, a))
        if v is not None:
            xs.append(v)
    return ms(xs)


def paired(items, f1, f2, keys=None):
    d = []
    for r in items:
        if keys is not None and r["key"] not in keys:
            continue
        x, y = f1(r), f2(r)
        if x is not None and y is not None:
            d.append(x - y)
    return ms(d)


def clean_keys(tag, s, items):
    """27B: outputs/D12b_max8k_flagged.json; 9B: the DS judged-leak file for qasper / musique (scripts/ds_sweep_table)
    and the 27B flags for the conflict sets (conservative; the 9B DS mix has no d25 data)."""
    ks = {r["key"] for r in items}
    fl = json.load(open("outputs/D12b_max8k_flagged.json"))["sets"]
    if not tag.startswith("27") and s in ("qasper", "musique"):
        import ds_sweep_table
        ck = ds_sweep_table.clean_keys(s)
        return None if ck is None else ks & ck
    fk = FLAG_KEY.get(s)
    if fk and fk in fl:
        bad = set(fl[fk]["flagged_keys"])
        return (ks - bad) if ks & bad else None
    return None


def select(tag, mode, main, ref):
    V = {s: load(tag, s) for s in VAL}
    V = {s: v["items"] for s, v in V.items() if v}
    grid = FC_AS if mode == "fc" else sorted(set.union(*[set(gen_alphas(v, main, ref)) for v in V.values()])) if V else [0.0]
    curve = {}
    for a in grid:
        per = {}
        for s, items in V.items():
            m = score_set(items, mode, main, ref, a)
            if m is not None:
                per[s] = m[0]
        if per:
            curve[a] = (sum(per.values()) / len(per), per)
    if not curve:
        return 0.0, {}
    best = max(curve, key=lambda a: (round(curve[a][0], 10), -a))
    return best, curve


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="9b")
    ap.add_argument("--reader", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    reader = a.reader or ("Qwen3.5-27B" if a.tag.startswith("27") else "Qwen3.5-9B")
    configs = [("fc", "kv", "closed_tail"), ("fc", "kv", "closed_book"), ("fc", "text", "closed_book"),
               ("gen", "kv", "closed_tail"), ("gen", "kv", "closed_book"), ("gen", "text", "closed_book")]
    sel, out = {}, {"tag": a.tag, "reader": reader, "selection": {}, "test": {}}
    print(f"# CAD {a.tag} (reader {reader}) — alpha selection on VAL ({', '.join(VAL)}; FaithEval val NOT used)")
    for cfg in configs:
        best, curve = select(a.tag, *cfg)
        sel[cfg] = best
        out["selection"]["|".join(cfg)] = {"alpha": best, "curve": {str(k): v for k, v in curve.items()}}
        print(f"  {cfg[0]:3s} {cfg[1]:4s} vs {cfg[2]:11s}: selected a = {best:g}   val mean " +
              " ".join(f"{k:g}:{v[0]:.3f}" for k, v in curve.items()))
    gpu = set()
    lat = {"gen_plain": [], "gen_cad": [], "fc_plain": [], "fc_cad": []}
    a_gct = sel[("gen", "kv", "closed_tail")]
    print("\n| set | subset | n | text | text+CAD (gen) | kv@0 | kv+CAD gen (ct) | kv+CAD gen (cb) | kv FC | kv FC+CAD (ct) "
          "| kv FC+CAD (cb) | text FC | text FC+CAD | wrong_doc | Δ gen ct vs kv | Δ FC ct vs kv FC | kv+CAD(ct) − text |")
    print("|" + "---|" * 17)
    for s in TEST:
        E = load(a.tag, s)
        if not E:
            continue
        gpu.add(E.get("gpu", "?"))
        items = E["items"]
        subs = [("all", None)]
        ck = clean_keys(a.tag, s, items)
        if ck is not None:
            subs.append(("clean", ck))
        out["test"][s] = {}
        for sub, keys in subs:
            def g(main, ref, al, mode="gen"):
                m = score_set(items, mode, main, ref, al, keys)
                return m
            row = {
                "text": g("text", "closed_book", 0.0),
                "text_cad": g("text", "closed_book", sel[("gen", "text", "closed_book")]),
                "kv": g("kv", "closed_tail", 0.0),
                "kv_cad_ct": g("kv", "closed_tail", a_gct),
                "kv_cad_cb": g("kv", "closed_book", sel[("gen", "kv", "closed_book")]),
                "fc_kv": g("kv", "closed_tail", 0.0, "fc"),
                "fc_kv_cad_ct": g("kv", "closed_tail", sel[("fc", "kv", "closed_tail")], "fc"),
                "fc_kv_cad_cb": g("kv", "closed_book", sel[("fc", "kv", "closed_book")], "fc"),
                "fc_text": g("text", "closed_book", 0.0, "fc"),
                "fc_text_cad": g("text", "closed_book", sel[("fc", "text", "closed_book")], "fc"),
                "wrong_doc": g("wrong_doc", "closed_tail", 0.0),
            }
            c_ct = gen_cond("kv", "closed_tail", a_gct)
            row["d_gen_ct"] = paired(items, lambda r: gen_metric(r, c_ct), lambda r: gen_metric(r, "kv"), keys)
            afc = sel[("fc", "kv", "closed_tail")]
            row["d_fc_ct"] = paired(items, lambda r: fc_metric(r, "kv", "closed_tail", afc),
                                    lambda r: fc_metric(r, "kv", "closed_tail", 0.0), keys)
            row["d_ct_text"] = paired(items, lambda r: gen_metric(r, c_ct), lambda r: gen_metric(r, "text"), keys)
            row["d_fc_ct_text"] = paired(items, lambda r: fc_metric(r, "kv", "closed_tail", afc),
                                         lambda r: fc_metric(r, "text", "closed_book", 0.0), keys)
            # test-only sensitivity curves (NOT used for selection)
            row["sens_gen_ct"] = {f"{x:g}": score_set(items, "gen", "kv", "closed_tail", x, keys)
                                  for x in gen_alphas(items, "kv", "closed_tail")}
            row["sens_gen_cb"] = {f"{x:g}": score_set(items, "gen", "kv", "closed_book", x, keys)
                                  for x in gen_alphas(items, "kv", "closed_book")}
            row["sens_fc_ct"] = {f"{x:g}": score_set(items, "fc", "kv", "closed_tail", x, keys) for x in FC_AS}
            row["sens_fc_cb"] = {f"{x:g}": score_set(items, "fc", "kv", "closed_book", x, keys) for x in FC_AS}
            out["test"][s][sub] = row
            f = lambda m: "" if m is None else f"{m[0]:.3f}"                               # noqa: E731
            d = lambda m: "" if m is None else f"{m[0]:+.3f} ± {m[1]:.3f}"                 # noqa: E731
            n = row["kv"][2] if row["kv"] else 0
            print(f"| {NAMES[s]} | {sub} | {n} | {f(row['text'])} | {f(row['text_cad'])} | {f(row['kv'])} | "
                  f"{f(row['kv_cad_ct'])} | {f(row['kv_cad_cb'])} | {f(row['fc_kv'])} | {f(row['fc_kv_cad_ct'])} | "
                  f"{f(row['fc_kv_cad_cb'])} | {f(row['fc_text'])} | {f(row['fc_text_cad'])} | {f(row['wrong_doc'])} | "
                  f"{d(row['d_gen_ct'])} | {d(row['d_fc_ct'])} | {d(row['d_ct_text'])} |")
        c_lat = c_ct if a_gct else "kv_cad_ct_1"           # latency: the selected config (a = 1 if a = 0 was selected)
        for r in items:
            t, gg = r.get("t", {}), r["gen"]
            if "kv" in gg and "t" in gg["kv"] and c_lat in gg and "t_loop" in gg[c_lat] and "pre_kv" in t:
                lat["gen_plain"].append(gg["kv"]["t"])
                lat["gen_cad"].append(t["pre_kv"] + t["pre_closed_tail"] + gg[c_lat]["t_loop"])
            fc = r.get("fc")
            if fc and "t" in fc and "pre_kv" in t:
                lat["fc_plain"].append(t["pre_kv"] + fc["t"]["kv"])
                lat["fc_cad"].append(t["pre_kv"] + fc["t"]["kv"] + t["pre_closed_tail"] + fc["t"]["closed_tail"])
    print("\nTest-only alpha sensitivity (NOT used for selection), kv+CAD vs closed_tail:")
    for s, subs in out["test"].items():
        r = subs["all"]
        gl = " ".join(f"{k}:{v[0]:.3f}" for k, v in r["sens_gen_ct"].items() if v)
        fl = " ".join(f"{k}:{v[0]:.3f}" for k, v in r["sens_fc_ct"].items() if v)
        print(f"  {NAMES[s]:18s} gen [{gl}]" + (f"  FC [{fl}]" if fl else ""))
    L = {k: (sum(v) / len(v) if v else None) for k, v in lat.items()}
    out["latency"] = {**L, "n_gen": len(lat["gen_plain"]), "n_fc": len(lat["fc_plain"])}
    if L["gen_plain"]:
        print(f"\nLatency per item (reader only; operator built once per doc, excluded): plain kv gen {L['gen_plain']:.3f}s "
              f"-> kv+CAD(ct, a={a_gct or 1:g}) {L['gen_cad']:.3f}s (x{L['gen_cad'] / L['gen_plain']:.2f})")
    if L["fc_plain"]:
        print(f"  forced choice: {L['fc_plain']:.3f}s -> {L['fc_cad']:.3f}s (x{L['fc_cad'] / L['fc_plain']:.2f})")
    out["gpu"] = sorted(gpu)
    m0 = [(load(a.tag, s) or {}).get("summary", {}).get("cad0_match") for s in VAL + TEST]
    m0 = [x for x in m0 if x]
    out["cad0_match"] = {"n": sum(x["n"] for x in m0), "match": sum(x["match"] for x in m0)}
    print(f"a=0 interleaving check: {out['cad0_match']['match']}/{out['cad0_match']['n']} identical to the plain kv decode;"
          f" GPU {sorted(gpu)}")
    json.dump(out, open(a.out or f"outputs/CAD_{a.tag}_table.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
