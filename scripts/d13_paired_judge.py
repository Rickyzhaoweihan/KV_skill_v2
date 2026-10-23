"""Paired comparison of two registered-judge eval files (register_d20.py / register_d22.py eval_test*.json) on the
same items: kv@0 (unregistered base), kv@200, kv_es<=200 (each run's own early-stop choice), wrong_doc and text.
Prints per-condition means and the paired difference B - A (+- sd/sqrt(n)), and whether the text column is identical
item by item (same items + same GPU type).

    python scripts/d13_paired_judge.py outputs/D22_qasper/eval_test.json outputs/D13L1p_qasper/eval_test.json
"""
import json
import sys


def es_score(J, r, cap="200"):
    b = J.get("es_choice", {}).get(r["doc_id"], {}).get(cap)
    return None if b is None else r[f"kv@{b}"]["score"]


def wrong_es(J, r, cap="200"):
    b = J.get("es_choice", {}).get(r["wrong"], {}).get(cap) if r.get("wrong") else None
    return None if b is None or f"wrong_doc@{b}" not in r else r[f"wrong_doc@{b}"]["score"]


def stats(x):
    n = len(x)
    mu = sum(x) / n
    se = (sum((v - mu) ** 2 for v in x) / max(n - 1, 1)) ** 0.5 / n ** 0.5
    return mu, se


def main(a_path, b_path):
    A, B = json.load(open(a_path)), json.load(open(b_path))
    ra, rb = {r["key"]: r for r in A["items"]}, {r["key"]: r for r in B["items"]}
    ks = sorted(set(ra) & set(rb))
    print(f"A = {a_path}\nB = {b_path}\npaired items {len(ks)} (A {len(ra)}, B {len(rb)})")
    same_text = all(ra[k]["text"]["score"] == rb[k]["text"]["score"] for k in ks if "text" in ra[k] and "text" in rb[k])
    print(f"text column identical item by item: {same_text}")
    conds = {
        "text": lambda J, r: r["text"]["score"] if "text" in r else None,
        "kv@0 (base, unregistered)": lambda J, r: r["kv@0"]["score"],
        "kv@200": lambda J, r: r["kv@200"]["score"] if "kv@200" in r else None,
        "kv es<=200 (kv_reg)": es_score,
        "wrong_doc es<=200": wrong_es,
    }
    for name, f in conds.items():
        pa = [f(A, ra[k]) for k in ks]
        pb = [f(B, rb[k]) for k in ks]
        pairs = [(x, y) for x, y in zip(pa, pb) if x is not None and y is not None]
        if not pairs:
            continue
        ma, _ = stats([x for x, _ in pairs])
        mb, _ = stats([y for _, y in pairs])
        d, se = stats([y - x for x, y in pairs])
        print(f"  {name:28s} A {ma:.4f}  B {mb:.4f}  B-A {d:+.4f} ± {se:.4f}  (n={len(pairs)})")
    def text_of(k):                       # a run judged without the text cond reuses the other run's (same GPU type)
        r = ra[k] if "text" in ra[k] else rb[k]
        return r["text"]["score"] if "text" in r else None
    for tag, J, R in (("A", A, ra), ("B", B, rb)):
        kt = [(es_score(J, R[k]), text_of(k)) for k in ks if es_score(J, R[k]) is not None and text_of(k) is not None]
        if not kt:
            continue
        d, se = stats([x - y for x, y in kt])
        print(f"  {tag}: kv_reg - text {d:+.4f} ± {se:.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
