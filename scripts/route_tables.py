#!/usr/bin/env python
"""ROUTE: markdown evidence tables (per benchmark: base / always-register / router / oracle + cost) from
outputs/ROUTE/docs.jsonl (+ probe rows). Routers are fit WITHOUT the scored benchmark (leave-one-benchmark-out inside
the pool) or on other pools (transfer). Writes outputs/ROUTE/tables.md.

usage: python scripts/route_tables.py [--signals ho0,probe_kv,probe_use] [--out outputs/ROUTE/tables.md]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import route_analyze as RA  # noqa: E402

SELFQ_S = 40.0      # self-question generation per doc (D22 table: 21-63 s on H100), not logged per doc


def wm(rows, f):
    n = sum(r["n_items"] for r in rows)
    return sum(f(r) * r["n_items"] for r in rows) / n


def se_paired(rows, ch):
    """doc-clustered SE of (router - base) item-weighted mean."""
    import math
    n = sum(r["n_items"] for r in rows)
    d = [(r["kv_es"] - r["kv0"]) * (1.0 if ch(r) else 0.0) for r in rows]
    w = [r["n_items"] for r in rows]
    m = sum(x * y for x, y in zip(d, w)) / n
    v = sum((y * (x - m)) ** 2 for x, y in zip(d, w)) / (n ** 2)
    return math.sqrt(v * len(rows) / max(len(rows) - 1, 1))


def reg_cost(r):
    t = r.get("t_reg") or {}
    return (r.get("t_teacher") or 0) + float(t.get("200", t.get(200, 0)) or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="outputs/ROUTE/docs.jsonl")
    ap.add_argument("--probe", default="outputs/ROUTE_probe/*/probe.jsonl")
    ap.add_argument("--signals", default="ho0,hodrop_rel,probe_kv,probe_use")
    ap.add_argument("--out", default="outputs/ROUTE/tables.md")
    a = ap.parse_args()
    R = [json.loads(x) for x in open(a.docs)]
    RA.attach_probe(R, a.probe)
    R = [RA.add_signals(r) for r in R]
    runname = lambda r: r["run"].split("/")[-1]  # noqa: E731
    pools = RA.make_pools(R)
    sigs = [s for s in a.signals.split(",")]
    L = []
    for p, rs in pools.items():
        bb = RA.by_bench(rs)
        L.append(f"\n### pool {p} (reader {rs[0]['reader']}), routers fit leave-one-benchmark-out inside the pool\n")
        hdr = "| benchmark | docs | items | base kv@0 | always-reg (es≤200) | " + " | ".join(
            f"router `{s}` (frac reg)" for s in sigs) + " | oracle | reg cost/doc (self-q ≈40 s + logged teacher + 200 steps) |"
        L.append(hdr)
        L.append("|" + "---|" * (hdr.count("|") - 1))
        macro = {"base": [], "reg": [], "oracle": [], **{s: [] for s in sigs}}
        paired = {s: [] for s in sigs}       # (router, base, reg) on the benchmarks where the signal exists
        fracs = {s: [] for s in sigs}
        for b, x in sorted(bb.items()):
            base, reg = wm(x, lambda r: r["kv0"]), wm(x, lambda r: r["kv_es"])
            orc = wm(x, lambda r: max(r["kv0"], r["kv_es"]))
            cells = []
            for s in sigs:
                tr = {k: v for k, v in bb.items() if k != b}
                if not any(s in r["sig"] for r in x):
                    cells.append("—")
                    continue
                dt = RA.fit_tau(tr, s)
                if dt is None:
                    cells.append("—")
                    continue
                ch = RA.router(s, dt)
                v = RA.bench_score(x, ch)
                fr = RA.frac_reg(x, ch)
                macro[s].append(v)
                paired[s].append((v, base, reg))
                fracs[s].append(fr)
                cells.append(f"{v:.3f} ({fr:.2f})")
            macro["base"].append(base)
            macro["reg"].append(reg)
            macro["oracle"].append(orc)
            c = sum(reg_cost(r) for r in x) / len(x)
            L.append(f"| {b} | {len(x)} | {sum(r['n_items'] for r in x)} | {base:.3f} | {reg:.3f} | " + " | ".join(cells)
                     + f" | {orc:.3f} | {c + SELFQ_S:.0f} s ({x[0]['gpu'][0].replace('NVIDIA ', '')[:4]}) |")
        mm = lambda k: (sum(macro[k]) / len(macro[k])) if macro[k] else float("nan")  # noqa: E731
        def pcell(s):
            if not paired[s]:
                return "—"
            k = len(paired[s])
            v, b0, r0 = (sum(x[i] for x in paired[s]) / k for i in range(3))
            return f"**{v:.3f}** (Δbase {v - b0:+.3f}, Δreg {v - r0:+.3f}; {k} sets)"
        L.append(f"| **macro** | | | **{mm('base'):.3f}** | **{mm('reg'):.3f}** | " + " | ".join(pcell(s) for s in sigs)
                 + f" | {mm('oracle'):.3f} | |")
    # transfer: fit on all 9B pools except the target, apply to the target pool
    L.append("\n### transfer: router fit on the OTHER pools (all their benchmarks), applied to the target pool\n")
    L.append("| target pool | base | always-reg | " + " | ".join(f"`{s}` (frac)" for s in sigs) + " | oracle |")
    L.append("|" + "---|" * (4 + len(sigs)))
    for tgt in pools:
        tr = {}
        for src, rs in pools.items():
            if src == tgt or src.endswith("_ood") or (tgt.startswith("L1") and src.startswith("L1") and "27b" not in tgt + src
                                                     and not tgt.endswith("_ood")):
                continue                      # L1p and L1f share the training mix: not a held-out base for each other
            for b, v in RA.by_bench(rs).items():
                tr[f"{src}:{b}"] = v
        te = RA.by_bench(pools[tgt])
        cells = []
        for s in sigs:
            dt = RA.fit_tau(tr, s) if tr else None
            if dt is None or not any(s in r["sig"] for r in pools[tgt]):
                cells.append("—")
                continue
            ch = RA.router(s, dt)
            cells.append(f"{RA.macro(te, ch):.3f} ({RA.frac_reg(pools[tgt], ch):.2f})")
        L.append(f"| {tgt} (from {','.join(sorted({k.split(':')[0] for k in tr}))}) | {RA.macro(te, lambda r: False):.3f} | "
                 f"{RA.macro(te, lambda r: True):.3f} | " + " | ".join(cells)
                 + f" | {RA.macro(te, lambda r: r['gain'] > 0):.3f} |")
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
