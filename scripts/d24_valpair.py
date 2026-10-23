"""D24: paired val comparison from history.json files written with --save_val_items 1.

    python scripts/d24_valpair.py outputs/D24_h5p outputs/D24_c5p [--step 300]
Prints, per val point of run A: kv(A) − kv_Hswap(A) (document specificity) and kv(A) − kv(B) (H vs control) at B's
same step, paired over item keys, overall and per task, with se = sd/√n.
"""
import argparse
import json
import os


def paired(xs):
    n = len(xs)
    if not n:
        return float("nan"), float("nan"), 0
    mu = sum(xs) / n
    sd = (sum((x - mu) ** 2 for x in xs) / max(n - 1, 1)) ** 0.5
    return mu, sd / n ** 0.5, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    a = ap.parse_args()
    A = {e["step"]: e for e in json.load(open(os.path.join(a.a, "history.json")))["evals"] if "items" in e}
    B = {e["step"]: e for e in json.load(open(os.path.join(a.b, "history.json")))["evals"] if "items" in e} if a.b else {}
    for st, e in sorted(A.items()):
        ia = {x["key"]: x for x in e["items"]}
        out = [f"step {st}: n={len(ia)} kv {sum(x['kv'] for x in ia.values()) / len(ia):.3f}"]
        sw = [x["kv"] - x["kv_Hswap"] for x in ia.values() if "kv_Hswap" in x]
        mu, se, n = paired(sw)
        out.append(f"own−swap {mu:+.3f}±{se:.3f} (n={n})")
        if st in B:
            ib = {x["key"]: x for x in B[st]["items"]}
            ks = sorted(set(ia) & set(ib))
            mu, se, n = paired([ia[k]["kv"] - ib[k]["kv"] for k in ks])
            out.append(f"A−B {mu:+.3f}±{se:.3f} (n={n})")
            for t in sorted({k.split("_")[0] for k in ks}):
                kt = [k for k in ks if k.split("_")[0] == t]
                m2, s2, n2 = paired([ia[k]["kv"] - ib[k]["kv"] for k in kt])
                m3, s3, _ = paired([ia[k]["kv"] - ia[k]["kv_Hswap"] for k in kt if "kv_Hswap" in ia[k]])
                out.append(f"[{t} n={n2}: A−B {m2:+.3f}±{s2:.3f}, own−swap {m3:+.3f}±{s3:.3f}]")
        print(" | ".join(out))


if __name__ == "__main__":
    main()
