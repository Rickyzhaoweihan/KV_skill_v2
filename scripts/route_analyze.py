#!/usr/bin/env python
"""ROUTE: evaluate label-free per-document signals as a register/don't-register router (CPU, stdlib only).

Input: outputs/ROUTE/docs.jsonl (scripts/route_extract.py). One row per (run, doc).
Policies per document: base = kv@0 (no registration), reg = D22 rule (es<=200, which can itself pick 0),
router = reg iff signal crosses tau (direction and tau fit on TRAIN docs only), oracle = per-doc max(kv0, reg).
Scores are item-weighted within a benchmark and macro-averaged over benchmarks.

usage: python scripts/route_analyze.py [--docs outputs/ROUTE/docs.jsonl] [--out outputs/ROUTE/analysis.json]
"""
import argparse
import json
import math
import random
import zlib
from collections import defaultdict

# one primary run per (base, bench): avoids counting the same docs twice. Replicates are used separately.
PRIMARY = {
    "L1f_9b": ["D13L1f_qasper", "D13L1f_loogle", "D13L1f_loogle_long", "D13L1f_musique", "D13L1f_quality",
               "D13L1f_lh_f0", "F1L1f_cnli_s0", "D24_head_scf_rf", "D13L1f_lastingbench", "D24_head_clip_rf"],
    "L1p_9b": ["D13L1p_qasper", "D13L1p_loogle", "D13L1p_loogle_long", "D13L1p_musique", "D13L1p_quality",
               "D13L1p_lh_f0", "D13L1p_lastingbench", "F1L1_cnli", "F1L1_scf"],
    "old_9b": ["D22_qasper", "D22_loogle", "D22_loogle_long", "D22_musique", "D20_4b_quality", "D20_4b_lh_f0",
               "D22_contractnli", "D22_shakespeare_cf", "D22_clipper", "D22_lastingbench", "D25_reg_fecf_base"],
    "L1f_27b": ["D12b_reg_fecf", "D12b_reg_loogle", "D12b_reg_qasper", "D12b_reg_shakespeare_cf"],
    "old_27b": ["D12h_reg_qas_tt", "D12h_reg_rf_q300", "D12h_reg_tt_lh"],
}


def attach_probe(R, pattern):
    """join scripts/route_probe.py rows (by run + doc); the text probe is shared by docs with identical probe items."""
    import glob
    P, T = {}, {}
    for f in glob.glob(pattern):
        for line in open(f):
            x = json.loads(line)
            P[(x["run"].rstrip("/"), x["doc_id"])] = x
            if x.get("probe_text") is not None:
                T[(x["doc_id"], tuple(i["key"] for i in x["items"]))] = x["probe_text"]
    n = 0
    for r in R:
        x = P.get((r["run"].rstrip("/"), r["doc_id"]))
        if not x or x.get("probe_kv") is None:
            continue
        n += 1
        r["probe_kv"], r["probe_tail"] = x["probe_kv"], x.get("probe_tail")
        r["probe_text"] = x.get("probe_text")
        if r["probe_text"] is None:
            r["probe_text"] = T.get((r["doc_id"], tuple(i["key"] for i in x["items"])))
        r["probe_t_s"] = x.get("t_s")
    print(f"probe rows joined: {n}")


def make_pools(R):
    """PRIMARY pools + OOD pools (agent OOD's register_d22 runs outputs/OOD_<L1f|old>_<set>, one per set)."""
    runname = lambda r: r["run"].rstrip("/").split("/")[-1]  # noqa: E731
    pools = {}
    for base, runs in PRIMARY.items():
        rs = [r for r in R if r["base"] == base and runname(r) in runs]
        if rs:
            pools[base] = rs
    for r in R:
        if runname(r).startswith("OOD_"):
            pools.setdefault(r["base"] + "_ood", []).append(r)
    return pools


def ho(r, b):
    h = r["ho"]
    return h.get(str(b), h.get(b))


def add_signals(r):
    h0 = ho(r, 0)
    later = {int(k): v for k, v in r["ho"].items() if int(k) > 0}
    s = {}
    pre = "rf_" if r.get("recipe") == "rf" else ""     # reader-free runs: ho = attention-distillation loss, not KL
    if h0 is not None and pre:
        s["rf_ho0"] = h0
        h0 = None
    if h0 is not None:
        s["ho0"] = h0                                           # KL(text||kv@0)+.5CE on held-out self-q, no reg steps
        s["log_ho0"] = math.log(h0 + 1e-4)
        if later:
            m = min(later.values())
            s["hodrop_rel"] = (h0 - m) / (h0 + 1e-6)              # needs the full registration
            s["hodrop_abs"] = h0 - m
            b1 = min(later)                                      # first budget (25 or 50): a short probe registration
            s["probe_rel"] = (h0 - later[b1]) / (h0 + 1e-6)
            s["probe_b"] = b1
    for k in ("kl1", "ce1", "dist1", "dist5", "kl5", "ce5", "conf_mean", "conf_lo"):
        if r.get(k) is not None:
            s[k] = r[k]
    if r.get("probe_kv") is not None:
        s["probe_kv"] = r["probe_kv"]
        if r.get("probe_tail") is not None:
            s["probe_use"] = r["probe_kv"] - r["probe_tail"]
        if r.get("probe_text") is not None:
            s["probe_gap"] = r["probe_text"] - r["probe_kv"]
    if r.get("tq") is not None:
        s["tq"] = r["tq"]                                       # text-teacher targets vs self-q writer agreement
    if r.get("n_op"):
        s["log_nop"] = math.log(r["n_op"])
    r["sig"] = s
    reg = r["kv_es"]
    r["gain"] = reg - r["kv0"]
    return r


SIGNALS = ["probe_kv", "probe_use", "probe_gap", "ho0", "log_ho0", "tq", "rf_ho0", "kl1", "ce1", "dist1", "dist5", "conf_mean", "conf_lo", "log_nop", "probe_rel",
           "hodrop_rel", "hodrop_abs"]
COST_CLASS = {   # what computing the signal needs at deployment (beyond the 4B harvest that kv@0 needs anyway)
    "ho0": "self-q + 9B text-teacher pass + held-out kv pass (no reg steps)",
    "log_ho0": "same as ho0",
    "kl1": "self-q + 9B text teacher on 2 items", "ce1": "same as kl1",
    "dist1": "self-q + 4B encoder attention + 1 kv pass (no text teacher)", "dist5": "same as dist1 (5 steps)",
    "conf_mean": "self-q + 9B text teacher", "conf_lo": "same as conf_mean",
    "log_nop": "free (token count)",
    "probe_kv": "K=12 self-q: kv@0 answers vs the writer's answers (K short kv decodes; no text pass)",
    "probe_use": "probe_kv - probe_tail (+K tail-only decodes): is the read using the document?",
    "probe_gap": "probe_text - probe_kv (analysis: needs K reader TEXT passes)",
    "tq": "self-q + 9B text-teacher answers (registration targets) vs the writer's answers",
    "rf_ho0": "self-q + encoder attention + held-out kv pass (reader-free recipe only)",
    "probe_rel": "self-q + teacher + first 25/50 reg steps + 2 held-out passes",
    "hodrop_rel": "full 200-step registration (= always register)", "hodrop_abs": "full registration",
}


def bench_score(rows, choose):
    """item-weighted mean score of a policy inside one benchmark; choose(r) -> True = registered."""
    n = sum(r["n_items"] for r in rows)
    return sum((r["kv_es"] if choose(r) else r["kv0"]) * r["n_items"] for r in rows) / n


def macro(rows_by_bench, choose):
    v = [bench_score(rs, choose) for rs in rows_by_bench.values()]
    return sum(v) / len(v)


def fit_tau(train_by_bench, sig):
    """Pick direction and threshold maximising the macro score on train. Returns (dir, tau) or None."""
    vals = sorted({r["sig"][sig] for rs in train_by_bench.values() for r in rs if sig in r["sig"]})
    if len(vals) < 3:
        return None
    qs = [vals[int(i * (len(vals) - 1) / 40)] for i in range(41)]
    cands = [(+1, -1e18), (+1, 1e18)] + [(d, t) for t in qs for d in (+1, -1)]
    best, bv = None, -1e9
    for d, t in cands:
        ch = (lambda r, d=d, t=t: sig in r["sig"] and d * (r["sig"][sig] - t) > 0)
        v = macro(train_by_bench, ch)
        if v > bv + 1e-12:
            best, bv = (d, t), v
    return best


def router(sig, dt):
    d, t = dt
    return lambda r: sig in r["sig"] and d * (r["sig"][sig] - t) > 0


def by_bench(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["bench"]].append(r)
    return dict(g)


def frac_reg(rows, ch):
    return sum(ch(r) for r in rows) / max(len(rows), 1)


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                rk[o[k]] = (i + j) / 2
            i = j + 1
        return rk
    if len(x) < 4:
        return None
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    c = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return c / (vx * vy) if vx > 0 and vy > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="outputs/ROUTE/docs.jsonl")
    ap.add_argument("--out", default="outputs/ROUTE/analysis.json")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--probe", default="outputs/ROUTE_probe/*/probe.jsonl")
    a = ap.parse_args()
    R = [json.loads(x) for x in open(a.docs)]
    attach_probe(R, a.probe)
    R = [add_signals(r) for r in R]
    runname = lambda r: r["run"].split("/")[-1]  # noqa: E731
    pools = make_pools(R)
    out = {"pools": {}, "corr": {}, "halves": {}, "lobo": {}, "transfer": {}}

    # ---- 1. per-benchmark base / reg / oracle
    for p, rs in pools.items():
        bb = by_bench(rs)
        out["pools"][p] = {b: {"docs": len(x), "items": sum(r["n_items"] for r in x),
                               "base": bench_score(x, lambda r: False), "reg": bench_score(x, lambda r: True),
                               "oracle": bench_score(x, lambda r: r["gain"] > 0),
                               "es0_frac": sum(r["es200"] == 0 for r in x) / len(x),
                               "kv200": sum(r["kv200"] * r["n_items"] for r in x) / sum(r["n_items"] for r in x)}
                           for b, x in sorted(bb.items())}

    # ---- 2. Spearman(signal, gain): doc level, within benchmark (bench-demeaned ranks via per-bench corr avg)
    for p, rs in pools.items():
        bb = by_bench(rs)
        out["corr"][p] = {}
        for s in SIGNALS:
            per = {}
            for b, x in bb.items():
                xs = [r for r in x if s in r["sig"]]
                c = spearman([r["sig"][s] for r in xs], [r["gain"] for r in xs])
                if c is not None:
                    per[b] = round(c, 3)
            # across-benchmark: bench-mean signal vs bench-mean gain
            mb = [(sum(r["sig"][s] for r in x if s in r["sig"]) / max(1, sum(s in r["sig"] for r in x)),
                   bench_score(x, lambda r: True) - bench_score(x, lambda r: False)) for x in bb.values()
                  if any(s in r["sig"] for r in x)]
            cb = spearman([m[0] for m in mb], [m[1] for m in mb]) if len(mb) >= 4 else None
            out["corr"][p][s] = {"within": per, "across_bench": None if cb is None else round(cb, 3)}

    # ---- 3. random half split within pool (doc-hash halves, repeated)
    for p, rs in pools.items():
        res = {}
        for s in SIGNALS:
            acc = defaultdict(list)
            for rep in range(a.reps):
                rnd = random.Random(rep)
                hk = {r["doc_id"]: rnd.random() < 0.5 for r in rs}
                tr = by_bench([r for r in rs if hk[r["doc_id"]]])
                te = by_bench([r for r in rs if not hk[r["doc_id"]]])
                te = {b: x for b, x in te.items() if x}
                dt = fit_tau(tr, s)
                if dt is None or not te:
                    continue
                ch = router(s, dt)
                acc["router"].append(macro(te, ch))
                acc["base"].append(macro(te, lambda r: False))
                acc["reg"].append(macro(te, lambda r: True))
                acc["oracle"].append(macro(te, lambda r: r["gain"] > 0))
                acc["frac"].append(frac_reg([r for x in te.values() for r in x], ch))
            if acc:
                m = {k: sum(v) / len(v) for k, v in acc.items()}
                d = [x - max(y, z) for x, y, z in zip(acc["router"], acc["base"], acc["reg"])]
                m["router_minus_best_fixed"] = sum(d) / len(d)
                m["p_router_beats_best_fixed"] = sum(x > 1e-9 for x in d) / len(d)
                res[s] = {k: round(v, 4) for k, v in m.items()}
        out["halves"][p] = res

    # ---- 4. leave-one-benchmark-out within pool (a NEW task type at deployment)
    for p, rs in pools.items():
        bb = by_bench(rs)
        res = {}
        for s in SIGNALS:
            rows = {}
            for b in bb:
                tr = {k: v for k, v in bb.items() if k != b}
                dt = fit_tau(tr, s)
                if dt is None:
                    continue
                ch = router(s, dt)
                rows[b] = {"router": round(bench_score(bb[b], ch), 4), "frac": round(frac_reg(bb[b], ch), 3),
                           "tau": dt}
            if rows:
                res[s] = {"per_bench": rows,
                          "macro_router": round(sum(v["router"] for v in rows.values()) / len(rows), 4)}
        out["lobo"][p] = {"base": round(macro(bb, lambda r: False), 4), "reg": round(macro(bb, lambda r: True), 4),
                          "oracle": round(macro(bb, lambda r: r["gain"] > 0), 4), "signals": res}

    # ---- 5. transfer across bases: fit on old_9b (+L1p_9b), apply to L1f_9b / L1f_27b and the reverse
    pairs = [("old_9b+L1p_9b+L1f_9b", "L1f_9b_ood"), ("L1f_9b", "L1f_9b_ood"), ("old_9b+L1p_9b+L1f_9b", "old_9b_ood"),
             ("old_9b", "L1f_9b"), ("old_9b+L1p_9b", "L1f_9b"), ("L1f_9b", "old_9b"), ("old_9b+L1f_9b", "L1f_27b"),
             ("L1f_9b", "L1f_27b"), ("old_9b+L1p_9b+L1f_9b", "L1f_27b"), ("old_9b+L1p_9b+L1f_9b", "old_27b")]
    for src, dst in pairs:
        srcs = [x for x in src.split("+") if x in pools]
        if not srcs or dst not in pools:
            continue
        tr = by_bench([r for x in srcs for r in pools[x]])
        tr = {f"{b}": v for b, v in tr.items()}
        # each source pool's benchmarks stay separate
        tr = {}
        for x in srcs:
            for b, v in by_bench(pools[x]).items():
                tr[f"{x}:{b}"] = v
        te = by_bench(pools[dst])
        res = {"base": round(macro(te, lambda r: False), 4), "reg": round(macro(te, lambda r: True), 4),
               "oracle": round(macro(te, lambda r: r["gain"] > 0), 4), "signals": {}}
        for s in SIGNALS:
            dt = fit_tau(tr, s)
            if dt is None:
                continue
            ch = router(s, dt)
            res["signals"][s] = {"router": round(macro(te, ch), 4), "frac": round(frac_reg(pools[dst], ch), 3),
                                 "tau": dt, "per_bench": {b: round(bench_score(x, ch), 4) for b, x in te.items()}}
        out["transfer"][f"{src}->{dst}"] = res

    # ---- 6. replicate-seed oracle: oracle chosen on one registration seed, scored on the other
    rep = {}
    for a1, a2 in (("F1L1f_cnli_s0", "F1L1f_cnli_s1"), ("D20_4b_cnli", "D20_4b_cnli_rf"),
                   ("D22_contractnli", "F1L0_cnli"), ("D22_shakespeare_cf", "F1L0_scf"),
                   ("D12h_reg_qas_tt", "D12h_reg_qas_rf")):
        A = {r["doc_id"]: r for r in R if runname(r) == a1}
        B = {r["doc_id"]: r for r in R if runname(r) == a2}
        ks = sorted(set(A) & set(B))
        if not ks:
            continue
        n = sum(B[k]["n_items"] for k in ks)
        sc = lambda f: sum(f(k) * B[k]["n_items"] for k in ks) / n  # noqa: E731
        rep[f"{a1}->{a2}"] = {"docs": len(ks), "base": round(sc(lambda k: B[k]["kv0"]), 4),
                              "reg": round(sc(lambda k: B[k]["kv_es"]), 4),
                              "oracle_own": round(sc(lambda k: max(B[k]["kv0"], B[k]["kv_es"])), 4),
                              "oracle_from_other_seed": round(sc(lambda k: B[k]["kv_es"] if A[k]["gain"] > 0
                                                                 else B[k]["kv0"]), 4),
                              "gain_corr_across_seeds": spearman([A[k]["gain"] for k in ks],
                                                                 [B[k]["gain"] for k in ks])}
    out["replicate_oracle"] = rep
    out["cost_class"] = COST_CLASS
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("pools", "replicate_oracle")}, indent=1))


if __name__ == "__main__":
    main()
