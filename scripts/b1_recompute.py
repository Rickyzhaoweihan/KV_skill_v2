"""B1 independent recompute (PAPERFORM verification step 3). Deliberately does NOT import xenc or d22_summary.

Re-scores every generation from the raw per-item jsonl with its own scorer (MCQ letter / SQuAD token F1), re-applies
the per-document early-stop choice from eval_*.json, and checks against (a) the stored per-item scores, (b) the
D22_table.json means, (c) D22_C1/summary.json (4B text).

    python scripts/b1_recompute.py
"""
import json
import re
import string
from collections import Counter
from pathlib import Path

RUNS = {  # name -> (eval json, split dir, C1 file, D22_table key)
    "LongHealth": ("outputs/D20_4b_lh_f0/eval_test.json", "data/longhealth_split", "longhealth", "LongHealth (f0)"),
    "QuALITY": ("outputs/D20_4b_quality/eval_test_full.json", "data/quality_split", "quality", "QuALITY"),
    "ContractNLI": ("outputs/D22_contractnli/eval_test.json", "data/contractnli_split", "contractnli", "ContractNLI"),
    "Shakespeare-CF": ("outputs/D22_shakespeare_cf/eval_test.json", "data/shakespeare_cf_split", "shakespeare-cf", "Shakespeare-CF"),
    "Qasper": ("outputs/D22_qasper/eval_test.json", "data/qasper_split", "qasper", "Qasper"),
    "MuSiQue": ("outputs/D22_musique/eval_test.json", "data/musique_split", "musique", "MuSiQue"),
    "LastingBench": ("outputs/D22_lastingbench/eval_test.json", "data/lastingbench_split", "lastingbench", "LastingBench"),
    "LooGLE": ("outputs/D22_loogle/eval_test.json", "data/loogle_split", "loogle", "LooGLE"),
    "LooGLE-long": ("outputs/D22_loogle_long/eval_test.json", "data/loogle_split", "loogle-long", "LooGLE-long (≥48k)"),
    "CLIPPER": ("outputs/D22_clipper/eval_test.json", "data/clipper_split", "clipper", "CLIPPER (T/F)"),
}
C1KEY = {"LongHealth": "LongHealth (f0)", "LooGLE-long": "LooGLE-long (≥48k)", "CLIPPER": "CLIPPER (T/F)"}


def norm(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def answer_text(g):
    m = re.findall(r"<answer>(.*?)</answer>", g or "", re.S | re.I)
    if m:
        return m[-1].strip()
    ls = [x.strip() for x in (g or "").strip().splitlines() if x.strip()]
    return ls[-1] if ls else (g or "").strip()


def letter(g, valid):
    m = re.findall(r"<answer>(.*?)</answer>", g or "", re.S | re.I)
    if m:
        inner = m[-1].strip()
        core = inner.upper().strip("()[]* .):")
        if len(core) == 1 and core in valid:
            return core
        for h in re.findall(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", inner):
            if h in valid:
                return h
        mm = re.match(r"^\(?([a-e])\)?[.):]?(\s|$)", inner)
        if mm and mm.group(1).upper() in valid:
            return mm.group(1).upper()
    hs = [h for h in re.findall(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", g or "") if h in valid]
    return hs[-1] if hs else None


def f1(p, golds):
    pt = norm(p).split()
    if not pt:
        return float(any(not norm(x).split() for x in golds))
    best = 0.0
    for x in golds:
        gt = norm(x).split()
        c = sum((Counter(pt) & Counter(gt)).values())
        if gt and c:
            pr, rc = c / len(pt), c / len(gt)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best


def score(it, g):
    if it.get("choices"):
        valid = "".join(c["label"] for c in it["choices"])
        gold = set((it.get("meta") or {}).get("accept_labels") or [it["correct_label"]])
        return float(letter(g, valid) in gold)
    return f1(answer_text(g), it["answers"])


def mse(xs):
    n = len(xs)
    m = sum(xs) / n
    return m, (sum((x - m) ** 2 for x in xs) / (n - 1)) ** .5 / n ** .5


def main():
    T = json.load(open("outputs/D22_table.json"))
    C1S = json.load(open("outputs/D22_C1/summary.json"))
    print("| benchmark | n | text | kv_reg | wrong | kv−text ± se | kv−wrong ± se | 4B text | Δ vs D22_table (text/kv/wrong) | Δ vs C1 summary | stored≠rescored items |")
    print("|" + "---|" * 11)
    for name, (ev, sd, c1, tk) in RUNS.items():
        items = {}
        for sp in ("train", "val", "test"):
            p = Path(sd) / sp / "items.json"
            if p.exists():
                for d in json.load(open(p)):
                    items[d["key"]] = d
        E = json.load(open(ev))
        es = E.get("es_choice") or {}
        bud = sorted(int(b) for b in (E["args"].get("eval_budgets") or E["args"]["budgets"]).split(","))
        Bk = str(max(b for b in bud if b <= 200))
        S = {"text": [], "kv": [], "wrong": []}
        mism = 0
        for r in E["items"]:
            it = items[r["key"]]
            bk = es[r["doc_id"]][Bk] if r["doc_id"] in es else int(Bk)
            bw = es[r["wrong"]][Bk] if r["wrong"] in es else int(Bk)
            for c, cell in (("text", r["text"]), ("kv", r[f"kv@{bk}"]), ("wrong", r[f"wrong_doc@{bw}"])):
                s = score(it, cell["gen"])
                mism += abs(s - cell["score"]) > 1e-9
                S[c].append(s)
        c1 = [json.loads(x) for x in open(f"outputs/D22_C1/{c1}.jsonl")]
        assert {x["key"] for x in c1} == {r["key"] for r in E["items"]}, name
        t4 = [score(items[x["key"]], x["gen"]) for x in c1]
        mism += sum(abs(score(items[x["key"]], x["gen"]) - x["score"]) > 1e-9 for x in c1)
        t = T[tk]["all"]
        dk = mse([a - b for a, b in zip(S["kv"], S["text"])])
        dw = mse([a - b for a, b in zip(S["kv"], S["wrong"])])
        m = {k: sum(v) / len(v) for k, v in S.items()}
        c1m = sum(t4) / len(t4)
        print(f"| {name} | {len(S['kv'])} | {m['text']:.4f} | {m['kv']:.4f} | {m['wrong']:.4f} | {dk[0]:+.3f}±{dk[1]:.3f} | "
              f"{dw[0]:+.3f}±{dw[1]:.3f} | {c1m:.4f} | {m['text'] - t['text']:+.4f}/{m['kv'] - t['kv_reg']:+.4f}/"
              f"{m['wrong'] - t['wrong_doc']:+.4f} | {c1m - C1S[C1KEY.get(name, name)]['text_enc']:+.4f} | {mism} |")


if __name__ == "__main__":
    main()
