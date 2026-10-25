"""Conflict-gated CAD (agent CAD, CPU only): a per-item alpha from the stored eval_cad rows, no GPU.

Gate signal (HM, hm_A_cad.py:37-38): TV_i = 1/2 sum_letters |p_kv(l) − p_ref(l)| over the option letters after "<answer>"
(stored forced-choice log-probs; ref = closed_tail or closed_book). Variants, each selected on VAL only
(v_cqacf, v_cqaor, v_nqswap, v_quality, v_qasper — FaithEval val never used), ties -> the smaller alpha / larger tau:
  fc_tv    HM exact: forced choice with alpha_i = a·TV_i
  fc_thr   forced choice with alpha_i = a if TV_i > tau else 0
  gen_thr  per-step CAD generation row kv_cad_<ref>_<a> if TV_i > tau else the plain kv@0 row. Items without option
           letters (span sets: NQ-Swap, Qasper, MuSiQue) have no gate signal and keep kv@0 (gate closed).
  gen_prior / fc_prior  same, but the gate is the PRIOR strength: max_l p_closed_book(l) > tau (a confident closed-book
           answer = the setting where a document that disagrees is a knowledge conflict). Not in HM; costs the closed_book
           letter pass (free when ref = closed_book).

    python scripts/cad_gate.py --tag 27lv4
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

from cad_table import NAMES, TEST, VAL, clean_keys, load, ms  # noqa: E402

FC_A = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0]
PTAUS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
TAUS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def tv(rec, ref):
    fc = rec.get("fc")
    if not fc or ref not in fc["lp"]:
        return None
    return 0.5 * sum(abs(math.exp(x) - math.exp(y)) for x, y in zip(fc["lp"]["kv"], fc["lp"][ref]))


def prior(rec):
    fc = rec.get("fc")
    if not fc or "closed_book" not in fc["lp"]:
        return None
    return max(math.exp(x) for x in fc["lp"]["closed_book"])


def sig(rec, var, ref):
    return prior(rec) if var.endswith("prior") else tv(rec, ref)


def fc_pick(rec, ref, alpha):
    fc = rec["fc"]
    s = [(1 + alpha) * x - alpha * y for x, y in zip(fc["lp"]["kv"], fc["lp"][ref])]
    i = max(range(len(s)), key=lambda j: s[j])
    return float(fc["labels"][i] in fc["gold"])


def metric(g):
    return g["doc_adh"] if "doc_adh" in g else g["score"]


def value(rec, var, ref, a, tau):
    """per-item score of a variant, or None if the item does not enter this mode."""
    t = sig(rec, var, ref)
    if var == "fc_tv":
        return None if t is None else fc_pick(rec, ref, a * t)
    if var in ("fc_thr", "fc_prior"):
        return None if t is None else fc_pick(rec, ref, a if t > tau else 0.0)
    rs = "ct" if ref == "closed_tail" else "cb"
    c = f"kv_cad_{rs}_{a:g}" if (t is not None and t > tau) else "kv"
    g = rec["gen"].get(c)
    return None if g is None else metric(g)


def gate_rate(items, ref, tau, keys=None, var="gen_thr"):
    xs = [float(sig(r, var, ref) > tau) for r in items
          if sig(r, var, ref) is not None and (keys is None or r["key"] in keys)]
    return sum(xs) / len(xs) if xs else None


def mean_on(items, var, ref, a, tau, keys=None):
    xs = [v for r in items if keys is None or r["key"] in keys for v in [value(r, var, ref, a, tau)] if v is not None]
    return ms(xs)


def select(V, var, ref, gen_as):
    grid = []
    for a in (FC_A if var.startswith("fc") else gen_as):
        for tau in ([0.0] if var == "fc_tv" else PTAUS if var.endswith("prior") else TAUS):
            per = [m[0] for items in V.values() for m in [mean_on(items, var, ref, a, tau)] if m is not None]
            if per:
                grid.append((sum(per) / len(per), -a, tau, a))
    best = max(grid)
    return best[3], best[2], best[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="27lv4")
    # [AUDIT 10-25] default-off: report each prior mode's OWN tau/open rate, and write to a different json.
    ap.add_argument("--open_per_mode", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    V = {s: load(a.tag, s)["items"] for s in VAL if load(a.tag, s)}
    gen_as = sorted({float(c.rsplit("_", 1)[1]) for items in V.values() for r in items for c in r["gen"]
                     if c.startswith("kv_cad_ct_")} - {0.0})
    out = {"tag": a.tag, "selection": {}, "test": {}}
    print(f"# conflict-gated CAD, {a.tag}: selection on VAL {sorted(V)} (FaithEval val NOT used)")
    sel = {}
    for var in ("fc_tv", "fc_thr", "gen_thr", "fc_prior", "gen_prior"):
        for ref in ("closed_tail", "closed_book"):
            al, tau, vm = select(V, var, ref, gen_as)
            sel[(var, ref)] = (al, tau)
            out["selection"][f"{var}|{ref}"] = {"alpha": al, "tau": tau, "val_mean": vm}
            print(f"  {var:7s} vs {ref:11s}: a = {al:g}, tau = {tau:g} (val mean {vm:.3f})")
    cols = [("gen_thr", "closed_tail"), ("gen_thr", "closed_book"), ("gen_prior", "closed_tail"),
            ("gen_prior", "closed_book"), ("fc_tv", "closed_tail"), ("fc_tv", "closed_book"),
            ("fc_thr", "closed_tail"), ("fc_thr", "closed_book"), ("fc_prior", "closed_tail"), ("fc_prior", "closed_book")]
    print("\n| set | subset | n | text | kv@0 | kv FC | " + " | ".join(f"{v} {r[7:] if 'tail' in r else 'cb'}".replace("closed_", "")
                                                                   for v, r in cols) + " | open thr ct | open gen_prior cb (tau_gen_prior; col was mislabelled 'open prior cb', NOT fc_prior's rate) |"
          + (" open fc_prior cb (tau_fc_prior) |" if a.open_per_mode else "") + " Δ gen_prior cb − kv |")
    for s in TEST:
        E = load(a.tag, s)
        if not E:
            continue
        items = E["items"]
        subs = [("all", None)]
        ck = clean_keys(a.tag, s, items)
        if ck is not None:
            subs.append(("clean", ck))
        out["test"][s] = {}
        for sub, keys in subs:
            ks = [r for r in items if keys is None or r["key"] in keys]
            row = {"text": ms([metric(r["gen"]["text"]) for r in ks if "text" in r["gen"]]),
                   "kv": ms([metric(r["gen"]["kv"]) for r in ks]),
                   "kv_fc": ms([fc_pick(r, "closed_tail", 0.0) for r in ks if r.get("fc")])}
            for v, r_ in cols:
                al, tau = sel[(v, r_)]
                row[f"{v}|{r_}"] = mean_on(items, v, r_, al, tau, keys)
            al, tau = sel[("gen_thr", "closed_tail")]
            row["gate_open_ct"] = gate_rate(items, "closed_tail", tau, keys)
            al2, tau2 = sel[("gen_prior", "closed_book")]
            row["gate_open_prior"] = gate_rate(items, "closed_book", tau2, keys, "gen_prior")
            if a.open_per_mode:
                _, tau_fc = sel[("fc_prior", "closed_book")]
                row["gate_open_gen_prior_cb"] = row["gate_open_prior"]
                row["tau_gen_prior_cb"] = tau2
                row["gate_open_fc_prior_cb"] = gate_rate(items, "closed_book", tau_fc, keys, "fc_prior")
                row["tau_fc_prior_cb"] = tau_fc
            row["d_gen_thr_ct"] = ms([value(r, "gen_thr", "closed_tail", al, tau) - metric(r["gen"]["kv"]) for r in ks])
            row["d_gen_prior_cb"] = ms([value(r, "gen_prior", "closed_book", al2, tau2) - metric(r["gen"]["kv"]) for r in ks])
            out["test"][s][sub] = row
            f = lambda m: "" if not m else f"{m[0]:.3f}"                               # noqa: E731
            go, gp = row["gate_open_ct"], row["gate_open_prior"]
            print(f"| {NAMES[s]} | {sub} | {row['kv'][2]} | {f(row['text'])} | {f(row['kv'])} | {f(row['kv_fc'])} | "
                  + " | ".join(f(row[f"{v}|{r_}"]) for v, r_ in cols)
                  + f" | {'' if go is None else f'{go:.2f}'} | {'' if gp is None else f'{gp:.2f}'} | "
                  + (f"{'' if row['gate_open_fc_prior_cb'] is None else format(row['gate_open_fc_prior_cb'], '.2f')} | " if a.open_per_mode else "")
                  +
                  f"{row['d_gen_prior_cb'][0]:+.3f} ± {row['d_gen_prior_cb'][1]:.3f} |")
    json.dump(out, open(a.out or f"outputs/CAD_{a.tag}_gate.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
