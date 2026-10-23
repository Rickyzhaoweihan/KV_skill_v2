"""F1: long-document read-sharpening pilot table, recomputed from the raw per-item jsonl.

Control = <dir>/eval_<split>.ckpt.jsonl (registration + judge at read multiplier 1.0); variants = the eval-only reruns
on the SAME registered adapters (<dir>/eval_<split>_<tag>.ckpt.jsonl). Rows: kv@0 (unregistered reference),
kv@200, wrong_doc@200, text (control), and variant − control paired on the same items. With --pairs, CLIPPER pair
accuracy (both claims of a pair right) as well.

    python scripts/f1_long_table.py --dir outputs/F1_long_scf --split_dir data/shakespeare_cf_split --tags logn16k,tau08,tau067
"""
import argparse
import json
import re
from pathlib import Path


def letter(g, valid):
    m = re.findall(r"<answer>(.*?)</answer>", g or "", re.S | re.I)
    s = (m[-1] if m else (g or "")).strip().upper().strip("()[]* .):")
    if len(s) == 1 and s in valid:
        return s
    h = [x for x in re.findall(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", m[-1] if m else (g or "")) if x in valid]
    return h[0] if h else None


def ms(v):
    n = len(v)
    mu = sum(v) / n
    return mu, (sum((x - mu) ** 2 for x in v) / max(n - 1, 1)) ** .5 / n ** .5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--tags", default="logn16k,tau08,tau067")
    ap.add_argument("--pairs", action="store_true")
    a = ap.parse_args()
    items = {}
    for sp in ("train", "val", "test"):
        p = Path(a.split_dir) / sp / "items.json"
        if p.exists():
            items.update({x["key"]: x for x in json.load(open(p))})

    def sc(key, cell):
        it = items[key]
        valid = "".join(c["label"] for c in it["choices"])
        gold = set((it.get("meta") or {}).get("accept_labels") or [it["correct_label"]])
        return float(letter(cell["gen"], valid) in gold)

    def load(tag):
        f = Path(a.dir) / (f"eval_{a.split}.ckpt.jsonl" if not tag else f"eval_{a.split}_{tag}.ckpt.jsonl")
        return {r["key"]: r for r in map(json.loads, open(f))} if f.exists() else None

    ctl = load("")
    V = {t: load(t) for t in a.tags.split(",")}
    V = {t: v for t, v in V.items() if v}
    keys = sorted(set(ctl).intersection(*[set(v) for v in V.values()]) if V else set(ctl))
    print(f"# {a.dir}: n = {len(keys)} items (intersection of control and every variant), scorer recomputed\n")
    T = {k: sc(k, ctl[k]["text"]) for k in keys}
    print(f"text {sum(T.values()) / len(T):.3f}; closed_tail "
          f"{sum(sc(k, ctl[k]['closed_tail']) for k in keys) / len(keys):.3f}\n")
    print("| read | kv@0 | kv@200 | wrong@200 | kv@200 − text | kv@200 − wrong | Δ vs control @0 | Δ vs control @200 |")
    print("|---|---|---|---|---|---|---|---|")
    base = {b: {k: sc(k, ctl[k][f"kv@{b}"]) for k in keys} for b in (0, 200)}
    for name, R in [("control ×1.0", ctl)] + [(t, v) for t, v in V.items()]:
        kv = {b: {k: sc(k, R[k][f"kv@{b}"]) for k in keys} for b in (0, 200)}
        w = {k: sc(k, R[k]["wrong_doc@200"]) for k in keys}
        dt = ms([kv[200][k] - T[k] for k in keys])
        dw = ms([kv[200][k] - w[k] for k in keys])
        d0 = ms([kv[0][k] - base[0][k] for k in keys])
        d2 = ms([kv[200][k] - base[200][k] for k in keys])
        print(f"| {name} | {sum(kv[0].values()) / len(keys):.3f} | {sum(kv[200].values()) / len(keys):.3f} | "
              f"{sum(w.values()) / len(keys):.3f} | {dt[0]:+.3f}±{dt[1]:.3f} | {dw[0]:+.3f}±{dw[1]:.3f} | "
              f"{d0[0]:+.3f}±{d0[1]:.3f} | {d2[0]:+.3f}±{d2[1]:.3f} |")
        if a.pairs:
            by = {}
            for k in keys:
                by.setdefault(items[k]["meta"]["pair_id"], []).append(k)
            prs = [v for v in by.values() if len(v) == 2]
            pa = lambda f: sum(all(f(k) for k in v) for v in prs) / max(1, len(prs))   # noqa: E731
            print(f"|   pair acc ({len(prs)} pairs) | {pa(lambda k: kv[0][k] == 1):.3f} | {pa(lambda k: kv[200][k] == 1):.3f} | "
                  f"{pa(lambda k: w[k] == 1):.3f} | text {pa(lambda k: T[k] == 1):.3f} | | | |")


if __name__ == "__main__":
    main()
