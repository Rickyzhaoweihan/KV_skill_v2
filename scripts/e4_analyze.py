"""E4: summarize scripts/e4_hopdiag.py output (CPU). DIAGNOSTIC ONLY (step-0 base, val docs).

    python scripts/e4_analyze.py outputs/E4_diag            -> prints markdown, writes outputs/E4_diag/analysis.md
Tables:
  1. read variants: mean score per task, paired diff vs base ± se (sd/√n).
  2. hop attention per layer: mass on hop-1 / last-hop supporting paragraph (mean over heads, max over heads),
     enrichment = mass / uniform share, for student (D13 read), teacher (encoder's own attention), text (9B reading
     the document), on the question row (q) and the gold-answer rows (ans). Hops whose paragraph lies in the reader's
     64-token tail (empty range) are dropped.
  3. top-paragraph hit rate: the paragraph with the highest mass (summed over heads) is hop-1 / last hop.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict


def mean_se(xs):
    n = len(xs)
    if not n:
        return float("nan"), float("nan")
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(n - 1, 1))
    return m, sd / math.sqrt(n)


def main(d: str) -> int:
    rows = [json.loads(line) for line in open(os.path.join(d, "items.jsonl"))]
    out = [f"# E4 diagnostic ({d}; step-0 base, val docs, n={len(rows)}) — DIAGNOSTIC ONLY", ""]
    # 1. variants
    for task in ("musique", "lastingbench", "quality"):
        R = [r for r in rows if r["task"] == task]
        if not R:
            continue
        vs = sorted({v for r in R for v in r["gens"]}, key=lambda v: (v != "base", v))
        out += [f"## Read variants: {task} (n={len(R)})", "", "| variant | n | score | − base ± se | gen tok |", "|---|---|---|---|---|"]
        for v in vs:
            P = [r for r in R if v in r["gens"] and "base" in r["gens"]]
            m, _ = mean_se([r["gens"][v]["score"] for r in P])
            dm, ds = mean_se([r["gens"][v]["score"] - r["gens"]["base"]["score"] for r in P])
            g, _ = mean_se([r["gens"][v]["gen_tokens"] for r in P])
            out.append(f"| {v} | {len(P)} | {m:.3f} | {dm:+.3f} ± {ds:.3f} | {g:.1f} |")
        out.append("")
        hops = defaultdict(list)
        for r in R:
            if r.get("hop_type"):
                hops[r["hop_type"][:4]].append(r)
        if hops:
            out += ["by hop count (score): " + "; ".join(
                f"{h} n={len(x)}: " + ", ".join(f"{v} {mean_se([r['gens'][v]['score'] for r in x if v in r['gens']])[0]:.2f}"
                                             for v in ("base", "text", "oracle_sup", "closed_tail") if v in x[0]["gens"])
                for h, x in sorted(hops.items())), ""]
    # 2./3. attention
    A = [r for r in rows if r.get("attn")]
    if A:
        out += [f"## Hop attention (MuSiQue, n={len(A)}): mass on the hop's supporting paragraph",
                "mean-over-heads / max-over-heads (enrichment = mean / uniform share); top-hit = the max-mass paragraph "
                "(summed over heads) is that hop's paragraph", ""]
        for rs in ("q", "ans"):
            out += [f"### rows: {rs}", "", "| cond | layer | hop1 mean | hop1 max | hop1 enr | last mean | last max | last enr | "
                    "sup total | top-hit hop1 | top-hit last |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for cond in ("student", "teacher", "text"):
                layers = sorted({int(li) for r in A for li in r["attn"][cond]})
                for li in layers:
                    acc = defaultdict(list)
                    for r in A:
                        mm = r["attn"][cond].get(str(li), {}).get(rs)
                        if mm is None:
                            continue
                        H = len(mm)
                        P = len(mm[0])
                        tot = [sum(mm[h][p] for h in range(H)) for p in range(P)]
                        top = max(range(P), key=lambda p: tot[p])
                        hi = r["hop_idx"]
                        for name, k in (("h1", hi[0]), ("hl", hi[-1])):
                            if r["para_len"][k] <= 0:
                                continue
                            col = [mm[h][k] for h in range(H)]
                            mean = sum(col) / H
                            acc[name + "_mean"].append(mean)
                            acc[name + "_max"].append(max(col))
                            acc[name + "_enr"].append(mean / (r["para_len"][k] / r["N"]))
                            acc[name + "_top"].append(float(top == k))
                        acc["sup"].append(sum(sum(mm[h][k] for h in range(H)) / H for k in set(hi) if r["para_len"][k] > 0))
                    f = lambda k: f"{mean_se(acc[k])[0]:.3f}" if acc[k] else "—"  # noqa: E731
                    out.append(f"| {cond} | {li} | {f('h1_mean')} | {f('h1_max')} | {f('h1_enr')} | {f('hl_mean')} | "
                               f"{f('hl_max')} | {f('hl_enr')} | {f('sup')} | {f('h1_top')} | {f('hl_top')} |")
            out.append("")
    txt = "\n".join(out)
    open(os.path.join(d, "analysis.md"), "w").write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "outputs/E4_diag"))
