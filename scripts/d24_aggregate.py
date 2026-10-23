"""D24: paired table of eval_d24.py runs (+ D20 registered rows matched by item key).

    python scripts/d24_aggregate.py outputs/D24_h1_q outputs/D24_h1_lh ... [--d20q outputs/D20_4b_quality/eval_test_n300.json]
"""

from __future__ import annotations

import argparse
import json
import os


def paired(rows, a, b):
    pr = [r[a]["score"] - r[b]["score"] for r in rows if a in r and b in r]
    if not pr:
        return None
    mu = sum(pr) / len(pr)
    sd = (sum((p - mu) ** 2 for p in pr) / max(len(pr) - 1, 1)) ** 0.5
    return mu, sd / len(pr) ** 0.5, sum(p > 0 for p in pr), sum(p < 0 for p in pr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--d20q", default="outputs/D20_4b_quality/eval_test_n300.json")
    ap.add_argument("--d20lh", default="outputs/D20_4b_lh_f0/eval_test.json")
    ap.add_argument("--d20_budget", default="200")
    a = ap.parse_args()
    d20 = {}
    for p in (a.d20q, a.d20lh):
        if p and os.path.exists(p):
            for r in json.load(open(p))["items"]:
                if f"kv@{a.d20_budget}" in r:
                    d20[r["key"]] = r[f"kv@{a.d20_budget}"]
    f = lambda x: "—" if x is None else f"{x[0]:+.3f}±{x[1]:.3f} ({x[2]}/{x[3]})"   # noqa: E731
    print("| run | n | closed_tail | text | kv_base | kv_reg@200 (D20) | **kv_H** | kv_Hswap | wrong_doc | kv_H−text | "
          "kv_H−wrong | kv_H−base | kv_H−reg | kv_H−Hswap | cost/doc 4B prefill + H (9B text prefill) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in a.dirs:
        p = os.path.join(d, "eval_test.json")
        if not os.path.exists(p):
            continue
        J = json.load(open(p))
        rows = J["items"]
        for r in rows:
            if "kv_reg" not in r and r["key"] in d20:
                r["kv_reg"] = d20[r["key"]]
        m = lambda c: (f"{sum(r[c]['score'] for r in rows if c in r) / max(sum(c in r for r in rows), 1):.3f}"   # noqa: E731
                       if any(c in r for r in rows) else "—")
        c = J.get("cost", {})
        cost = (f"{c.get('enc_prefill_s', 0):.3f}s + {c.get('h_forward_s', 0) * 1e3:.1f}ms "
                f"({c.get('reader_text_prefill_s', 0):.3f}s), {c.get('n_doc_tokens', 0):.0f} tok") if c else "—"
        print(f"| {os.path.basename(d)} (H step {J.get('h_extra', {}).get('step')}) | {len(rows)} | {m('closed_tail')} | "
              f"{m('text')} | {m('kv_base')} | {m('kv_reg')} | **{m('kv')}** | {m('kv_Hswap')} | {m('wrong_doc')} | "
              f"{f(paired(rows, 'kv', 'text'))} | {f(paired(rows, 'kv', 'wrong_doc'))} | {f(paired(rows, 'kv', 'kv_base'))} | "
              f"{f(paired(rows, 'kv', 'kv_reg'))} | {f(paired(rows, 'kv', 'kv_Hswap'))} | {cost} |")


if __name__ == "__main__":
    main()
