"""D21 (b): pair a compressed-interface REGISTERED judge (scripts/register_d20.py eval json) with the full-interface
registered judge on the same items (D20's eval json; also the source of text / closed_tail, which do not depend on the
interface). Prints a markdown row per budget: kv, wrong_doc, text, closed_tail, kv−wrong, kv−tail, kv−text,
kv − full_kv (all paired, se = sd/√n), and agreement of the kv answers with the full interface's.

    python scripts/d21_pair.py --comp outputs/D21_reg_lh_int8svdM/eval_test.json --ref outputs/D20_4b_lh_f0/eval_test.json \
        --label int8+svd128_M --budgets 0,200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    mu = sum(d) / len(d)
    sd = (sum((x - mu) ** 2 for x in d) / max(len(d) - 1, 1)) ** 0.5
    return mu, sd / len(d) ** 0.5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comp", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--budgets", default="0,200")
    a = ap.parse_args(argv)
    from xenc.doc_tasks import extract_answer, extract_label, normalize_answer
    C = {r["key"]: r for r in json.load(open(a.comp))["items"]}
    Rf = {r["key"]: r for r in json.load(open(a.ref))["items"]}
    keys = [k for k in C if k in Rf]
    for b in [int(x) for x in a.budgets.split(",")]:
        kv = [C[k][f"kv@{b}"]["score"] for k in keys]
        wr = [C[k][f"wrong_doc@{b}"]["score"] for k in keys]
        tx = [Rf[k]["text"]["score"] for k in keys]
        ct = [Rf[k]["closed_tail"]["score"] for k in keys]
        fk = [Rf[k][f"kv@{b}"]["score"] for k in keys]
        ag = []
        for k in keys:
            g1, g2 = C[k][f"kv@{b}"]["gen"], Rf[k][f"kv@{b}"]["gen"]
            l1, l2 = extract_label(g1, "ABCDE"), extract_label(g2, "ABCDE")
            ag.append((l1 == l2) if (l1 or l2) else normalize_answer(extract_answer(g1)) == normalize_answer(extract_answer(g2)))
        m = lambda x: sum(x) / len(x)   # noqa: E731
        f = lambda p: f"{p[0]:+.3f} ± {p[1]:.3f}"   # noqa: E731
        print(f"| {a.label} | {b} | {len(keys)} | {m(kv):.3f} | {m(wr):.3f} | {m(tx):.3f} | {m(ct):.3f} | "
              f"{f(paired(kv, wr))} | {f(paired(kv, ct))} | {f(paired(kv, tx))} | {m(fk):.3f} | {f(paired(kv, fk))} | "
              f"{m(ag):.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
