"""D12: NQ-Swap doc- vs prior-adherence from an eval_d13/eval_d12p eval_test.json (greedy span answers).

In data/d25_nqswap_split the released `answers` are the PRIOR (original) answers (meta.answers_are_prior), so the
judge's `score` = prior-adherence (lower = more document-following). Here, per condition:
  prior_adh = token-F1(answer, prior) >= .5
  doc_adh   = not prior_adh AND the normalised answer occurs in the asked passage (meta.span of the doc text)
Paired kv − cond differences (± se) for doc_adh.

    python scripts/d12_nqswap_adh.py outputs/D12p_9b_deep_v2_nqswap [outputs/D12p_9b_d13r_nqswap ...]
"""
import json
import re
import string
import sys


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(a, b):
    a, b = norm(a).split(), norm(b).split()
    common = sum(min(a.count(w), b.count(w)) for w in set(a))
    if not a or not b or not common:
        return 0.0
    p, r = common / len(a), common / len(b)
    return 2 * p * r / (p + r)


def ans(g):
    m = re.search(r"<answer>(.*?)(</answer>|$)", g, re.S)
    return (m.group(1) if m else g).strip()


def main():
    items = {i["key"]: i for i in json.load(open("data/d25_nqswap_split/test/items.json"))}
    sys.path.insert(0, ".")
    from xenc.doc_tasks import load_docs
    docs = load_docs("data/d25_nqswap_split")
    for path in sys.argv[1:]:
        ev = json.load(open(path + "/eval_test.json"))["items"]
        res = {}
        for r in ev:
            it = items[r["key"]]
            text = docs[it["doc_id"]]
            s0, s1 = it["meta"]["span"]
            passage = norm(text[s0:s1])
            prior = it["meta"]["prior_answers"]
            for c in ("closed_tail", "text", "kv", "wrong_doc"):
                if c not in r:
                    continue
                a = ans(r[c]["gen"])
                pa = max(f1(a, p) for p in prior) >= 0.5
                da = (not pa) and bool(norm(a)) and norm(a) in passage
                res.setdefault(c, []).append((r["key"], int(da), int(pa)))
        print(f"== {path} (n={len(ev)})")
        for c, v in res.items():
            print(f"  {c:12s} doc_adh {sum(x[1] for x in v) / len(v):.3f}  prior_adh {sum(x[2] for x in v) / len(v):.3f}")
        if "kv" in res:
            kv = {k: d for k, d, _ in res["kv"]}
            for c in ("text", "wrong_doc", "closed_tail"):
                if c in res:
                    p = [kv[k] - d for k, d, _ in res[c]]
                    m = sum(p) / len(p)
                    se = (sum((x - m) ** 2 for x in p) / (len(p) - 1)) ** .5 / len(p) ** .5
                    print(f"  doc_adh kv − {c}: {m:+.3f} ± {se:.3f}")


if __name__ == "__main__":
    main()
