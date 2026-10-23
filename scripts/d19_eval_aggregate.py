"""[D19] Aggregate eval_d13.py outputs on the D19 splits into markdown tables.

  * every run: per-condition accuracy ± se and paired kv − {wrong_doc, closed_tail, text} ± se, overall and per subset
  * ContractNLI: subsets all / clean (meta.near_dup_train False) + the per-hypothesis majority baseline
  * Shakespeare(-CF): subsets by meta.kind (speaker / act); on CF speaker items also document-adherence
    (pred == gold = CF label) vs prior-adherence (pred == meta.prior_label = the memorized speaker) vs other

  python scripts/d19_eval_aggregate.py --eval outputs/D19_eval_contractnli/eval_test.json \
      --split_dir data/contractnli_split [--md_out outputs/D19_eval_contractnli/summary.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from xenc.doc_tasks import extract_label  # noqa: E402

CONDS = ("closed_book", "closed_tail", "text", "kv", "wrong_doc")


def mse(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if not n:
        return float("nan"), float("nan")
    mu = sum(xs) / n
    sd = (sum((x - mu) ** 2 for x in xs) / max(n - 1, 1)) ** 0.5
    return mu, sd / n ** 0.5


def table(rows: list[dict], items: dict, title: str, baseline: str | None = None) -> list[str]:
    conds = [c for c in CONDS if any(c in r for r in rows)]
    out = [f"**{title}** (n={len(rows)}, docs={len({r['doc_id'] for r in rows})})", "",
           "| " + " | ".join(conds + [f"kv−{o}" for o in ("wrong_doc", "closed_tail", "text") if o in conds and "kv" in conds])
           + (" | hyp-prior" if baseline else "") + " |",
           "|" + "---|" * (len(conds) + sum(o in conds and "kv" in conds for o in ("wrong_doc", "closed_tail", "text"))
                          + (1 if baseline else 0))]
    cells = []
    for c in conds:
        mu, se = mse([r[c]["score"] for r in rows if c in r])
        cells.append(f"{mu:.3f}±{se:.3f}")
    for o in ("wrong_doc", "closed_tail", "text"):
        if o in conds and "kv" in conds:
            mu, se = mse([r["kv"]["score"] - r[o]["score"] for r in rows if "kv" in r and o in r])
            cells.append(f"{mu:+.3f}±{se:.3f}")
    if baseline:
        mu, se = mse([float(items[r["key"]]["meta"][baseline] == items[r["key"]]["correct_label"]) for r in rows])
        cells.append(f"{mu:.3f}±{se:.3f}")
    out.append("| " + " | ".join(cells) + " |")
    return out + [""]


def adherence(rows: list[dict], items: dict, title: str) -> list[str]:
    conds = [c for c in ("closed_book", "text", "kv", "wrong_doc", "closed_tail") if any(c in r for r in rows)]
    out = [f"**{title}**: share of answers matching the document (CF gold) / the prior (memorized speaker) / other",
           "", "| condition | doc-adherence | prior-adherence | other / unparsed | n |", "|---|---|---|---|---|"]
    for c in conds:
        d, p, o = [], [], []
        for r in rows:
            if c not in r:
                continue
            it = items[r["key"]]
            lab = extract_label(r[c]["gen"], "".join(ch["label"] for ch in it["choices"]))
            d.append(float(lab == it["correct_label"]))
            p.append(float(lab == it["meta"]["prior_label"]))
            o.append(float(lab not in (it["correct_label"], it["meta"]["prior_label"])))
        (dm, ds), (pm, ps), (om, os_) = mse(d), mse(p), mse(o)
        out.append(f"| {c} | {dm:.3f}±{ds:.3f} | {pm:.3f}±{ps:.3f} | {om:.3f}±{os_:.3f} | {len(d)} |")
    return out + [""]


def aggregate(ev: dict, split_items: list[dict], name: str) -> list[str]:
    items = {it["key"]: it for it in split_items}
    rows = [r for r in ev["items"] if r["key"] in items]
    md = [f"### {name}", ""]
    if name.startswith("contractnli"):
        md += table(rows, items, "all test docs", baseline="hyp_prior_label")
        clean = [r for r in rows if not items[r["key"]]["meta"].get("near_dup_train")]
        md += table(clean, items, "clean docs (near_dup_train False)", baseline="hyp_prior_label")
    elif name.startswith("shakespeare"):
        md += table(rows, items, "all items")
        for kind in ("speaker", "act"):
            sub = [r for r in rows if items[r["key"]]["meta"]["kind"] == kind]
            if sub:
                md += table(sub, items, f"{kind} items")
        if name.startswith("shakespeare_cf"):
            sp = [r for r in rows if items[r["key"]]["meta"].get("conflict")]
            if sp:
                md += adherence(sp, items, "CF speaker (conflict) items")
    else:
        md += table(rows, items, "all items")
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--md_out", default=None)
    a = ap.parse_args()
    ev = json.load(open(a.eval))
    split_items = json.load(open(Path(a.split_dir) / a.split / "items.json"))
    name = Path(a.split_dir.rstrip("/")).name.removesuffix("_split")
    md = "\n".join(aggregate(ev, split_items, name))
    print(md)
    if a.md_out:
        Path(a.md_out).write_text(md + "\n")


if __name__ == "__main__":
    main()
