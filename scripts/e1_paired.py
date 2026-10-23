"""E1: paired comparison of two register_d13 runs on the SAME val items (needs --save_items 1).
usage: python scripts/e1_paired.py outputs/E1_ctrl outputs/E1_lr3x [--last 3]
Per-item kv score is averaged over the last k val checks of each run (k >= 2 damps single-eval noise); the report is the
mean paired difference (run B - run A) with its SE and a >2-SE flag. Also gives each run's step-0 kv on the same items."""
import argparse, json, math

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--last", type=int, default=3)
ap.add_argument("--steps", default="", help="comma list of val steps to average (e.g. 250,500) when the runs have different eval_every")
ap.add_argument("--allow_step0_diff", type=int, default=0, help="E2: max number of items whose step-0 kv may differ between "
                "the runs (numerical round-off, e.g. a zero-init operator translator); the step-0 difference is reported")
x = ap.parse_args()


def load(d):
    ev = json.load(open(f"{d}/history.json"))["evals"]
    assert all("items" in e for e in ev), f"{d}: run with --save_items 1"
    zero = {r[0]: r[1] for r in ev[0]["items"]}
    last = [e for e in ev if str(e['step']) in x.steps.split(',')] if x.steps else ev[-x.last:]
    assert last, f'{d}: no evals at steps {x.steps}'
    avg = {}
    for e in last:
        for r in e["items"]:
            avg.setdefault(r[0], []).append(r[1])
    return zero, {k: sum(v) / len(v) for k, v in avg.items()}, [e["step"] for e in last]


(z1, m1, s1), (z2, m2, s2) = load(x.a), load(x.b)
keys = sorted(set(m1) & set(m2))
assert len(keys) == len(m1) == len(m2), "runs were not evaluated on the same items"
_d0 = [k for k in keys if abs(z1[k] - z2[k]) >= 1e-9]
if _d0 and len(_d0) <= x.allow_step0_diff:
    _m = sum(z2[k] - z1[k] for k in keys) / len(keys)
    print(f"NOTE: step-0 kv differs on {len(_d0)}/{len(keys)} items (allowed <= {x.allow_step0_diff}): mean B-A at step 0 "
          f"{_m:+.4f}; items {_d0}")
else:
    assert not _d0 or print(f"WARNING: step-0 scores differ between runs on {len(_d0)} items (different init/val?)")


def stat(d):
    n = len(d); mu = sum(d) / n
    se = math.sqrt(sum((v - mu) ** 2 for v in d) / (n - 1) / n) if n > 1 else float("nan")
    return mu, se


for name, d in (("step0 kv (A) ", [z1[k] for k in keys]), (f"A last{x.last}", [m1[k] for k in keys]),
                (f"B last{x.last}", [m2[k] for k in keys])):
    mu, se = stat(d); print(f"{name}: {mu:.4f} +- {se:.4f}  (n={len(keys)})")
mu, se = stat([m2[k] - m1[k] for k in keys])
print(f"B - A (paired, steps {s2} vs {s1}): {mu:+.4f} +- {se:.4f}  -> {'>2 SE' if abs(mu) > 2 * se else 'within 2 SE'}")
mu, se = stat([m1[k] - z1[k] for k in keys]); print(f"A last{x.last} - A step0: {mu:+.4f} +- {se:.4f}")
mu, se = stat([m2[k] - z2[k] for k in keys]); print(f"B last{x.last} - B step0: {mu:+.4f} +- {se:.4f}")
