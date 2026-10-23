"""F1: ContractNLI pilot table, recomputed from the raw per-item jsonl (own scorer: MCQ letter == gold).

Per arm (outputs/F1_nli_{ctrl,prior,bal}): kv at every budget and at es<=200 (per-doc early stop on held-out
self-question loss, from the arm's eval json), wrong_doc, text, closed_tail, kv-text / kv-wrong (paired se),
per-gold-label accuracy, predicted-label histogram, and each arm minus ctrl (paired, same items).

    python scripts/f1_nli_table.py [--arms ctrl,prior,bal] [--root outputs/F1_nli_]
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

GL = {"A": "E", "B": "C", "C": "N"}


def letter(g):
    m = re.findall(r"<answer>(.*?)</answer>", g or "", re.S | re.I)
    s = (m[-1] if m else (g or "")).strip().upper().strip("()[]* .):")
    if len(s) == 1 and s in "ABC":
        return s
    h = re.findall(r"(?<![A-Za-z])([ABC])(?![A-Za-z])", (m[-1] if m else g or ""))
    return h[0] if h else None


def ms(v):
    n = len(v)
    mu = sum(v) / n
    return mu, (sum((x - mu) ** 2 for x in v) / max(n - 1, 1)) ** .5 / n ** .5


def load(root, arm, split="val"):
    E = json.load(open(f"{root}{arm}/eval_{split}.json"))
    es = E.get("es_choice") or {}
    rows = [json.loads(x) for x in open(f"{root}{arm}/eval_{split}.ckpt.jsonl")]
    return rows, es


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="ctrl,prior,bal")
    ap.add_argument("--root", default="outputs/F1_nli_")
    ap.add_argument("--split_dir", default="data/contractnli_split")
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    gold = {x["key"]: x["correct_label"] for x in json.load(open(Path(a.split_dir) / a.split / "items.json"))}
    arms = [x for x in a.arms.split(",") if Path(f"{a.root}{x}/eval_{a.split}.json").exists()]
    R = {}
    for arm in arms:
        rows, es = load(a.root, arm, a.split)
        out = {}
        for r in rows:
            g = gold[r["key"]]
            d = {"gold": g, "doc": r["doc_id"]}
            for c in ("text", "closed_tail"):
                if c in r:
                    d[c] = letter(r[c]["gen"])
            for k in r:
                if k.startswith("kv@") or k.startswith("wrong_doc@"):
                    d[k] = letter(r[k]["gen"])
            bk = es.get(r["doc_id"], {}).get("200", 200)
            bw = es.get(r["wrong"], {}).get("200", 200) if r.get("wrong") else 200
            d["kv_es"], d["wrong_es"] = d.get(f"kv@{bk}"), d.get(f"wrong_doc@{bw}")
            out[r["key"]] = d
        R[arm] = out
    keys = sorted(set.intersection(*[set(v) for v in R.values()]))
    print(f"# F1 ContractNLI pilot ({a.split}, first 20 docs), n = {len(keys)} common items; scorer recomputed from raw generations\n")
    print("| arm | cond | acc | − text | − wrong_es | acc E / C / N | predicted A/B/C |")
    print("|---|---|---|---|---|---|---|")
    ngold = Counter(R[arms[0]][k]["gold"] for k in keys)
    print(f"| (gold) | | | | | n {ngold['A']} / {ngold['B']} / {ngold['C']} | |")
    for arm in arms:
        D = R[arm]
        conds = ["closed_tail", "text"] + sorted([c for c in D[keys[0]] if c.startswith("kv@")], key=lambda s: int(s[3:])) + ["kv_es", "wrong_es"]
        for c in conds:
            if c not in D[keys[0]]:
                continue
            ok = [float(D[k][c] == D[k]["gold"]) for k in keys]
            t = [float(D[k]["text"] == D[k]["gold"]) for k in keys]
            w = [float(D[k]["wrong_es"] == D[k]["gold"]) for k in keys]
            dt, dw = ms([x - y for x, y in zip(ok, t)]), ms([x - y for x, y in zip(ok, w)])
            per = {L: sum(D[k][c] == L for k in keys if D[k]["gold"] == L) / max(1, ngold[L]) for L in "ABC"}
            pred = Counter(D[k][c] for k in keys)
            print(f"| {arm} | {c} | {sum(ok) / len(ok):.3f} | {dt[0]:+.3f}±{dt[1]:.3f} | {dw[0]:+.3f}±{dw[1]:.3f} | "
                  f"{per['A']:.2f} / {per['B']:.2f} / {per['C']:.2f} | {pred['A']}/{pred['B']}/{pred['C']} (none {pred[None]}) |")
    if "ctrl" in R:
        print("\n**Arm − ctrl (paired, same items)**\n\n| arm | cond | Δ acc | Δ on E | Δ on C | Δ on N |\n|---|---|---|---|---|---|")
        for arm in arms:
            if arm == "ctrl":
                continue
            for c in ("kv@0", "kv@100", "kv@200", "kv_es"):
                d = [float(R[arm][k][c] == R[arm][k]["gold"]) - float(R["ctrl"][k][c] == R["ctrl"][k]["gold"]) for k in keys]
                mu, se = ms(d)
                sub = {L: ms([x for x, k in zip(d, keys) if R["ctrl"][k]["gold"] == L]) for L in "ABC"}
                print(f"| {arm} | {c} | {mu:+.3f}±{se:.3f} | " + " | ".join(f"{sub[L][0]:+.3f}±{sub[L][1]:.3f}" for L in "ABC") + " |")


if __name__ == "__main__":
    main()
