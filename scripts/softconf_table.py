"""SOFTCONF tables (agent SOFTCONF, 2026-10-25), CPU only: graded-conflict behaviour of text / kv / kv+CAD variants from
eval_cad (or softconf_eval) jsons + the SOFTCONF items + the reader's prior probe.

Per item and condition the predicted option is mapped to its ROLE (meta.roles) and classified:
  doc      = the value / wording / framing printed in THIS document (gold; doc-adherence)
  orig     = the original (unedited) value                         (num / ent / hedge)
  prior    = the reader's closed-book choice (softconf_prior.py, else the closed_book FC argmax)
  over     = beyond the doc value in the edit direction (num: magnitude order; hedge: marker ladder) = OVERSHOOT
  under    = strictly between orig and doc (num / hedge)
At L0 (no edit) doc == orig: accuracy there = collateral damage of a method on unedited documents.
Strata: conflict = prior stable over option orders AND prior == orig (the reader holds the original value, so every
edit contradicts it); weak = everything else (no stable prior / prior on another option); clean = (num) the original
value does not also appear, unedited, elsewhere in the passage (meta.orig_elsewhere == 0).

Conditions: generation rows present in the json (closed_book, text, kv, kv_cad_cb_*, kv_cad_ct_*, wrong_doc, ...)
and forced-choice recomputes from the stored letter log-probs:
  fc_kv, fc_text, fc_cb;   fc_cad_<a> (kv vs closed_book);   fc_gate_<a>_<tau> (prior gate, CAD_SUMMARY 5.2);
  fc_plaus_<a>_<b> (CAD + plausibility cut b);   fc_jsd_<c> (AdaCAD: a = c·JSD2);   fc_cj_<c>_<b> (a = c·JSD2·max p_cb,
  + plausibility b: conflict strength x prior confidence, guarded).
Hyper-parameters of the continuous methods are chosen on the SOFTCONF *val* json (--val) by mean doc-adherence over
all levels (L0 included: collateral counts); the test tables then use the chosen values.

    python scripts/softconf_table.py --evals outputs/SC_27lv4_num/eval_test.json --val outputs/SC_27lv4_num/eval_val.json \
        --split_dir <softconf_num_split> --prior outputs/SOFTCONF/prior_27b.json --out outputs/SOFTCONF/table_27lv4_num
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LEVELS = ("L0", "L1", "L2", "L3", "L4")


# ---------------------------------------------------------------------------- forced-choice arbitration rules
def _p(lp):
    return [math.exp(x) for x in lp]


def jsd2(p, q) -> float:
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(a, b):
        return sum(x * (math.log(max(x, 1e-30)) - math.log(max(y, 1e-30))) for x, y in zip(a, b) if x > 0)
    return max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m)) / math.log(2)


def fc_pick(lpm, lpr, alpha: float, plaus: float = 0.0) -> int:
    """argmax (1+a)·lp_main − a·lp_ref, optionally restricted to options with p_main >= plaus·max p_main."""
    s = [(1 + alpha) * x - alpha * y for x, y in zip(lpm, lpr)] if alpha else list(lpm)
    if plaus > 0:
        pm = _p(lpm)
        mx = max(pm)
        s = [v if pm[i] >= plaus * mx else -1e30 for i, v in enumerate(s)]
    return max(range(len(s)), key=lambda i: s[i])


def fc_method(name: str, lpm, lpr) -> int:
    """name: kv | cad_<a> | gate_<a>_<tau> | plaus_<a>_<b> | jsd_<c> | cj_<c>_<b>."""
    parts = name.split("_")
    kind, xs = parts[0], [float(x) for x in parts[1:]]
    if kind == "kv":
        return fc_pick(lpm, lpr, 0.0)
    if kind == "cad":
        return fc_pick(lpm, lpr, xs[0])
    if kind == "gate":
        return fc_pick(lpm, lpr, xs[0] if max(_p(lpr)) >= xs[1] else 0.0)
    if kind == "plaus":
        return fc_pick(lpm, lpr, xs[0], xs[1])
    pm, pr = _p(lpm), _p(lpr)
    if kind == "jsd":
        return fc_pick(lpm, lpr, xs[0] * jsd2(pm, pr))
    if kind == "cj":
        return fc_pick(lpm, lpr, xs[0] * jsd2(pm, pr) * max(pr), xs[1])
    raise ValueError(name)


GRIDS = {"cad": [[a] for a in (0.25, 0.5, 1, 2, 3)],
         "gate": [[a, t] for a in (0.5, 1, 3) for t in (0.6, 0.9, 0.99, 0.995, 0.999)],
         "plaus": [[a, b] for a in (0.5, 1, 3, 5) for b in (0.05, 0.1, 0.3)],
         "jsd": [[c] for c in (1, 2, 5, 10, 20)],
         "cj": [[c, b] for c in (2, 5, 10, 20, 50) for b in (0.0, 0.1, 0.3)]}


def mname(kind, xs):
    return kind + "_" + "_".join(f"{x:g}" for x in xs)


# ---------------------------------------------------------------------------- classification
def classify(meta: dict, pred_label, prior_role) -> dict:
    roles = meta["roles"]
    r = roles.get(pred_label) if pred_label else None
    doc_r = roles[meta["doc_label"]]
    orig_r = roles.get(meta["orig_label"]) if meta.get("orig_label") else None
    out = {"doc": float(r == doc_r), "orig": float(orig_r is not None and r == orig_r),
           "prior": float(prior_role is not None and r == prior_role), "none": float(r is None)}
    order = meta.get("order")                     # labels ordered from the original towards the most extreme edit
    if order and meta["axis"] in ("num", "hedge") and orig_r is not None and r is not None:
        pos = {roles[lab]: i for i, lab in enumerate(order)}
        if meta["axis"] == "hedge":               # ladder m3 m2 m1 m0: position = marker level, direction by the edit
            o, d = 3 - meta["orig_marker_level"], 3 - meta["doc_marker_level"]
        else:
            o, d = 0, pos[doc_r]
        p = pos[r]
        sgn = (d > o) - (d < o)
        out["over"] = float(sgn != 0 and (p - d) * sgn > 0)
        out["under"] = float(sgn != 0 and 0 <= (p - o) * sgn < abs(d - o) and p != d)
    return out


def ms(xs):
    n = len(xs)
    if not n:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / max(n - 1, 1)) / math.sqrt(n), n


# ---------------------------------------------------------------------------- loading
def load_items(split_dir: str) -> dict:
    out = {}
    for sp in ("val", "test"):
        f = Path(split_dir, sp, "items.json")
        if f.exists():
            out.update({it["key"]: it for it in json.load(open(f))})
    return out


def predictions(rec: dict, item: dict, fc_methods: list[str]) -> dict:
    """condition -> predicted label (None = unparseable)."""
    from xenc.doc_tasks import extract_label
    valid = "".join(c["label"] for c in item["choices"])
    out = {}
    for c, g in rec.get("gen", {}).items():
        out[c] = extract_label(g.get("gen", ""), valid)
    fc = rec.get("fc")
    if fc:
        labs, lp = fc["labels"], fc["lp"]
        for nm in ("kv", "text", "closed_book", "closed_tail", "wrong_doc"):
            if nm in lp:
                out[f"fc_{nm}"] = labs[fc_pick(lp[nm], lp[nm], 0.0)]
        if "kv" in lp and "closed_book" in lp:
            for m in fc_methods:
                out[f"fc_{m}"] = labs[fc_method(m, lp["kv"], lp["closed_book"])]
        if "text" in lp and "closed_book" in lp:
            out["fc_text_cad_1"] = labs[fc_pick(lp["text"], lp["closed_book"], 1.0)]
    return out


def prior_of(items: dict, recs: dict, prior: dict | None) -> dict:
    """qid -> {"role", "stable", "pmax"} (probe if given, else the closed_book FC of the L0 / first item)."""
    out = {}
    if prior:
        for qid, v in prior["q"].items():
            out[qid] = {"role": v["prior_role"], "stable": bool(v["stable"]), "pmax": v["pmax"]}
    for k, rec in recs.items():
        it = items.get(k)
        if not it or it["meta"]["qid"] in out or not rec.get("fc") or "closed_book" not in rec["fc"]["lp"]:
            continue
        lp = rec["fc"]["lp"]["closed_book"]
        j = max(range(len(lp)), key=lambda i: lp[i])
        out[it["meta"]["qid"]] = {"role": it["meta"]["roles"][rec["fc"]["labels"][j]], "stable": True,
                                  "pmax": math.exp(lp[j]), "from": "closed_book_fc"}
    return out


def rows_for(eval_paths: list[str], items: dict, prior: dict | None, fc_methods: list[str]):
    recs = {}
    for p in eval_paths:
        for r in json.load(open(p))["items"]:
            recs[r["key"]] = r
    pri = prior_of(items, recs, prior)
    rows = []
    for k, rec in recs.items():
        it = items.get(k)
        if it is None:
            continue
        m = it["meta"]
        pq = pri.get(m["qid"], {})
        orig_role = m["roles"].get(m["orig_label"]) if m.get("orig_label") else None
        stratum = "conflict" if (pq.get("stable") and orig_role is not None and pq.get("role") == orig_role) else "weak"
        if m["axis"] == "persp":          # conflict = the doc frames a side the reader does not prefer
            stratum = "conflict" if pq.get("stable") and pq.get("role") not in (m["roles"][m["doc_label"]],) else "weak"
        preds = predictions(rec, it, fc_methods)
        rows.append({"key": k, "meta": m, "stratum": stratum, "prior_role": pq.get("role"),
                     "clean": m.get("orig_elsewhere", 0) == 0,
                     "cls": {c: classify(m, lab, pq.get("role")) for c, lab in preds.items()}})
    return rows


# ---------------------------------------------------------------------------- selection + tables
def select(rows, kinds=("cad", "gate", "plaus", "jsd", "cj")) -> dict:
    """Per method kind, the grid point with the best mean doc-adherence over ALL val rows (all levels)."""
    best = {}
    for kind in kinds:
        sc = []
        for xs in GRIDS[kind]:
            nm = "fc_" + mname(kind, xs)
            v = [r["cls"][nm]["doc"] for r in rows if nm in r["cls"]]
            if v:
                sc.append((sum(v) / len(v), -sum(xs), nm))
        if sc:
            best[kind] = max(sc)[2]
    return best


def table(rows, conds, axis_filter=None, stratum=None) -> dict:
    T = defaultdict(dict)
    for lev in LEVELS:
        rs = [r for r in rows if r["meta"]["level"] == lev and (axis_filter is None or r["meta"]["axis"] == axis_filter)
              and (stratum is None or (r["clean"] if stratum == "clean" else r["stratum"] == stratum))]
        if not rs:
            continue
        for c in conds:
            cs = [r for r in rs if c in r["cls"]]
            if not cs:
                continue
            cell = {}
            for met in ("doc", "prior", "over", "under", "none"):
                xs = [r["cls"][c][met] for r in cs if met in r["cls"][c]]
                if xs:
                    cell[met] = ms(xs)[:2]
            base = "fc_kv" if c.startswith("fc_") else "kv"
            pd = [r["cls"][c]["doc"] - r["cls"][base]["doc"] for r in cs if base in r["cls"]]
            if pd and c != base:
                cell["d_doc_vs_" + base] = ms(pd)[:2]
            cell["n"] = len(cs)
            T[lev][c] = cell
    return T


def fmt_table(T, conds, title) -> str:
    out = [f"\n#### {title}", "", "| level | n | " + " | ".join(conds) + " |", "|---|---|" + "---|" * len(conds)]
    for lev, row in T.items():
        n = max(v["n"] for v in row.values())
        cells = []
        for c in conds:
            v = row.get(c)
            if not v:
                cells.append("")
                continue
            s = f"{v['doc'][0]:.3f}"
            if "over" in v:
                s += f" (o{v['over'][0]:.2f})"
            cells.append(s)
        out.append(f"| {lev} | {n} | " + " | ".join(cells) + " |")
    out.append("cells: doc-adherence (= accuracy vs the document); (oX) = overshoot rate. Levels: L1 = subtlest edit (num +-10%), L4 = most blatant (x10 / x0.1); L0 = unedited.")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--evals", required=True, help="comma list of eval_test.json (same split_dir)")
    ap.add_argument("--val", default="", help="comma list of eval_val.json for hyper-parameter selection")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--prior", default="")
    ap.add_argument("--reader", default="", help="reader name for the table titles (e.g. 27B)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    items = load_items(a.split_dir)
    prior = json.load(open(a.prior)) if a.prior else None
    allm = [mname(k, xs) for k, g in GRIDS.items() for xs in g]
    sel = {}
    if a.val:
        vrows = rows_for(a.val.split(","), items, prior, allm)
        sel = select(vrows)
    rows = rows_for(a.evals.split(","), items, prior, allm)
    gen = [c for c in ("closed_book", "text", "kv", "kv_cad_cb_0.25", "kv_cad_cb_1", "kv_cad_ct_0.25", "wrong_doc")
           if any(c in r["cls"] for r in rows)]
    gen += sorted({c for r in rows for c in r["cls"] if c.startswith("kv_cad") and c not in gen})
    fcs = ["fc_closed_book", "fc_text", "fc_kv", "fc_cad_0.25", "fc_cad_1", "fc_gate_0.5_0.995"] + \
        [f"fc_{v[3:] if v.startswith('fc_') else v}" for v in sel.values()]
    fcs = [c for c in dict.fromkeys(fcs) if any(c in r["cls"] for r in rows)]
    conds = gen + fcs
    axes = sorted({r["meta"]["axis"] for r in rows})
    res = {"selected_on_val": sel, "reader": a.reader, "tables": {}}
    md = [f"# SOFTCONF table ({a.reader}) — {a.evals}", f"val-selected: {sel}"]
    for ax in axes:
        for st in (None, "conflict", "weak", "clean"):
            T = table(rows, conds, ax, st)
            res["tables"][f"{ax}|{st or 'all'}"] = T
            md.append(fmt_table(T, conds, f"{ax} — stratum {st or 'all'} — reader {a.reader}"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    Path(a.out + ".md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
