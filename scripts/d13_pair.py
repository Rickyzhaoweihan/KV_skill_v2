"""D13: paired differences between conditions stored in different eval_d13 / eval_encside json files (same items).

    python scripts/d13_pair.py outputs/D13_s2w10_lh/eval_test.json:kv outputs/D13_text_lh/eval_test.json:text
-> mean A, mean B, A − B ± paired se, wins/losses, n (items matched by key).
"""

from __future__ import annotations

import json
import sys


def load(spec):
    path, cond = spec.rsplit(":", 1)
    d = json.load(open(path))
    return {r["key"]: r[cond]["score"] for r in d["items"] if cond in r}


def pair(a: dict, b: dict) -> dict:
    keys = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in keys]
    n = len(d)
    mu = sum(d) / max(n, 1)
    sd = (sum((x - mu) ** 2 for x in d) / max(n - 1, 1)) ** 0.5
    return {"n": n, "a": sum(a[k] for k in keys) / max(n, 1), "b": sum(b[k] for k in keys) / max(n, 1),
            "diff": mu, "se": sd / max(n, 1) ** 0.5, "wins": sum(x > 0 for x in d), "losses": sum(x < 0 for x in d)}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    r = pair(load(argv[0]), load(argv[1]))
    print(f"n={r['n']}  A {r['a']:.4f}  B {r['b']:.4f}  A−B {r['diff']:+.4f} ± {r['se']:.4f} ({r['wins']}/{r['losses']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
