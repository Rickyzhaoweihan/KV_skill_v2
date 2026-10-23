#!/usr/bin/env python
"""ROUTE: per-document table of realized registration gain + label-free deployment signals (CPU only).

Reads existing register_d20/d22 outputs: <run>/eval_*.json (per-item kv@b scores) and <run>/adapters/*.pt
(per-doc held-out self-question loss per budget, training history, teacher targets, timings).
Writes one JSON row per (run, doc) to outputs/ROUTE/docs.jsonl.

No test label enters any SIGNAL column: signals come only from the adapter files (self-questions, teacher, encoder)
and docs.json (length). Test scores are only used for the realized-gain columns (kv0, kv_reg, ...).

usage: python scripts/route_extract.py [--runs outputs/D13L1f_qasper ...] [--out outputs/ROUTE/docs.jsonl]
"""
import argparse
import glob
import json
import math
import os
import sys

BASES = {
    "outputs/D13_lever1_full/d13_best.pt": "L1f_9b",
    "outputs/D13_lever1_pilot/d13_best.pt": "L1p_9b",
    "outputs/D13_4b_reg_w10/d13_best.pt": "old_9b",
    "outputs/D12p_lever_full_s2/d13_best.pt": "L1f_27b",
    "outputs/D12p_4b_s2/d13_best.pt": "old_27b",
}
# runs whose registration / read is a modified recipe (read sharpening, quantised operator, other recipes) are
# excluded by default so every row uses the D22 recipe on its base.
EXCLUDE = ("E4_", "D21_", "F1_long_", "F1_nli_prior", "F1_nli_bal", "D17_", "D25_reg_fecf_m2", "D26_", "D20_dev",
           "D22_L3", "D20_4b_cnli_enc", "D20_4b_scf_enc", "D22_smoke", "D23_")


def bench_name(run, split_dir):
    b = os.path.basename(split_dir.rstrip("/")).replace("_split", "")
    r = os.path.basename(run)
    if "loogle_long" in r:
        b = "loogle_long"
    if "lh_f0" in r or b == "longhealth":
        b = "longhealth"
    return b


def eval_file(run):
    for n in ("eval_test_full.json", "eval_test.json", "eval_val.json"):
        p = os.path.join(run, n)
        if os.path.exists(p):
            return p
    return None


def load_selfq(path, cache={}):
    if path not in cache:
        m = {}
        try:
            for line in open(path):
                x = json.loads(line)
                m[x["doc_id"]] = x["items"]
        except Exception:
            pass
        cache[path] = m
    return cache[path]


def teacher_agreement(A, did, targets):
    """Label-free target quality: agreement of the 9B text teacher's registration targets with the self-question
    WRITER's own answer (the writer read the document as text). MCQ/NLI: same option text; span: token F1.
    Returns (agree_mean, n_matched, per-template dict)."""
    sys.path.insert(0, ".")
    from register_d20 import shuffle_choices   # same deterministic option shuffle as registration
    from xenc.doc_tasks import extract_label, f1_score
    sq = load_selfq(A.get("selfq") or "")
    if did not in sq or not targets:
        return None, 0, {}
    items = shuffle_choices([dict(x) for x in sq[did]][: A.get("max_items") or 10 ** 9], A.get("shuffle") or 0,
                            bool(A.get("shuffle_nli")))
    orig = {x["key"]: x for x in sq[did]}
    j, res = 0, []
    for t in targets:
        while j < len(items) and items[j]["question"][:200] != t["q"]:
            j += 1
        if j >= len(items):
            break
        it = items[j]
        j += 1
        tmpl = (it.get("meta") or {}).get("template")
        ans = t["t"].split("<answer>")[-1].split("</answer>")[0]
        if it.get("choices"):
            base = orig.get(_base_key(it["key"])) or it
            want_txt = None
            if base.get("correct_label") and base.get("choices"):
                want_txt = next((c["text"] for c in base["choices"] if c["label"] == base["correct_label"]), None)
            want_txt = want_txt or (base.get("answers") or [None])[0]
            want = next((c["label"] for c in it["choices"] if c["text"] == want_txt), None)
            got = extract_label(t["t"], "".join(c["label"] for c in it["choices"]))
            if want is None:
                continue
            res.append((tmpl, float(got == want)))
        else:
            gold = [g for g in (it.get("answers") or []) if g]
            if gold:
                res.append((tmpl, f1_score(ans, gold)))
    if not res:
        return None, 0, {}
    per = {}
    for k, v in res:
        per.setdefault(k, []).append(v)
    return sum(v for _, v in res) / len(res), len(res), {k: round(sum(v) / len(v), 3) for k, v in per.items()}


def _base_key(k):
    import re
    return re.sub(r"_p\d+$", "", str(k))


def mean(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(x) / len(x) if x else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--out", default="outputs/ROUTE/docs.jsonl")
    ap.add_argument("--all_bases", type=int, default=0, help="1 = keep runs whose interface is not in BASES")
    a = ap.parse_args()
    import torch

    runs = a.runs or sorted({os.path.dirname(p) for p in glob.glob("outputs/*/eval_*.json")})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    tok_cache = {}
    n_rows = 0
    with open(a.out, "w") as fo:
        for run in runs:
            if any(os.path.basename(run).startswith(e) for e in EXCLUDE):
                continue
            ef = eval_file(run)
            if ef is None or not os.path.isdir(os.path.join(run, "adapters")):
                continue
            try:
                E = json.load(open(ef))
            except Exception:
                continue
            if not isinstance(E, dict) or "items" not in E or not E.get("es_choice"):
                continue
            A = E["args"]
            base = BASES.get(A.get("interface"))
            if base is None and not a.all_bases:
                continue
            split_dir = A["split_dir"]
            bench = bench_name(run, split_dir)
            if split_dir not in tok_cache:
                try:
                    tok_cache[split_dir] = json.load(open(os.path.join(split_dir, "docs.json")))
                except Exception:
                    tok_cache[split_dir] = {}
            dix = tok_cache[split_dir]
            eb = sorted({int(b) for b in (A.get("eval_budgets") or A["budgets"]).split(",")})
            esc = E["es_choice"]
            gpu = (E.get("cost") or {}).get("gpu")
            # per-doc item aggregation
            per = {}
            for r in E["items"]:
                d = r["doc_id"]
                if d not in esc:
                    continue
                p = per.setdefault(d, {"n": 0, **{f"kv{b}": [] for b in eb}, "kv_es": [], "wrong0": [],
                                       "text": [], "tail": []})
                p["n"] += 1
                for b in eb:
                    if f"kv@{b}" in r:
                        p[f"kv{b}"].append(r[f"kv@{b}"]["score"])
                bk = int(esc[d][str(max(eb))]) if str(max(eb)) in esc[d] else max(eb)
                if f"kv@{bk}" in r:
                    p["kv_es"].append(r[f"kv@{bk}"]["score"])
                if f"wrong_doc@{eb[0]}" in r:
                    p["wrong0"].append(r[f"wrong_doc@{eb[0]}"]["score"])
                for c, k in (("text", "text"), ("closed_tail", "tail")):
                    if c in r:
                        p[k].append(r[c]["score"])
            for d, p in per.items():
                f = os.path.join(run, "adapters", d + ".pt")
                if not os.path.exists(f):
                    continue
                ad = torch.load(f, map_location="cpu", weights_only=False, mmap=True)
                ho = {int(k): float(v) for k, v in (ad.get("ho_loss") or {}).items()}
                hist = ad.get("hist") or []
                tg = ad.get("targets") or []
                conf = [t["conf"] for t in tg if t.get("conf") is not None]
                kinds = {}
                for t in tg:
                    kinds[t.get("kind")] = kinds.get(t.get("kind"), 0) + 1
                h5 = hist[:5]
                hl = hist[-10:]
                n_tok = (dix.get(d) or {}).get("n_tokens")
                tq, tq_n, tq_per = (None, 0, {})
                if not (A.get("enc_only") or A.get("teacher") == "none"):
                    tq, tq_n, tq_per = teacher_agreement(A, d, tg)
                row = {
                    "run": run, "bench": bench, "base": base, "reader": A.get("reader"), "gpu": gpu,
                    "split": A.get("split"), "doc_id": d,
                    "recipe": "rf" if (A.get("enc_only") or A.get("teacher") == "none") else "tt",
                    "n_items": p["n"], "max_tokens": A.get("max_tokens"),
                    "es200": int(esc[d][str(max(eb))]),
                    # realized test scores (NOT signals)
                    **{f"kv{b}": mean(p[f"kv{b}"]) for b in eb},
                    "kv_es": mean(p["kv_es"]), "wrong0": mean(p["wrong0"]), "text": mean(p["text"]),
                    "tail": mean(p["tail"]),
                    # label-free signals
                    "n_tok": n_tok, "n_op": (min(n_tok, A["max_tokens"]) if n_tok and A.get("max_tokens") else n_tok),
                    "ho": ho, "n_holdout": ad.get("n_holdout"), "n_reg_items": ad.get("n_items"),
                    "kl1": hist[0]["kl"] if hist else None, "ce1": hist[0]["ce"] if hist else None,
                    "dist1": hist[0]["dist"] if hist else None,
                    "kl5": mean([h["kl"] for h in h5]), "ce5": mean([h["ce"] for h in h5]),
                    "dist5": mean([h["dist"] for h in h5]),
                    "kl_last": mean([h["kl"] for h in hl]), "dist_last": mean([h["dist"] for h in hl]),
                    "conf_mean": mean(conf), "conf_lo": (sum(c < 0.5 for c in conf) / len(conf)) if conf else None,
                    "kinds": kinds, "tq": tq, "tq_n": tq_n, "tq_per": tq_per, "t_teacher": ad.get("t_teacher_s"),
                    "t_reg": {int(k): v for k, v in (ad.get("t_reg_s") or {}).items()},
                }
                fo.write(json.dumps(row) + "\n")
                n_rows += 1
            print(f"{run}: bench {bench} base {base} docs {len(per)} gpu {gpu}", flush=True)
    print(f"wrote {n_rows} rows -> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
