"""OOD report: raw-score table per OOD set (reader Qwen3.5-9B, encoder 4B), recomputed from the per-item eval files of
outputs/OOD_L1f_<set>/eval_test.json (lever-1 full; has text / closed_tail) and outputs/OOD_old_<set>/eval_test.json
(old base D13_4b_reg_w10). kv_reg = each run's own per-doc early-stop choice over budgets <= 200 (es_choice).
Paired SE = sd/sqrt(n) over the items both runs kept. Sub-tasks (e.g. InfiniteBench En.MC / En.QA) are split by the
item's `task` field.

    python -I scripts/ood_report.py [--sets ...] [--md outputs/OOD_tables.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from d13_paired_judge import es_score, wrong_es, stats  # noqa: E402

SETS = "ood_cuad,ood_infbench,ood_lbv2,d25_nqswap,d25_fecf,d25_cqacf"


def ms(x):
    return "n/a" if not x else f"{sum(x) / len(x):.3f}"


def pse(pairs):
    if not pairs:
        return "n/a"
    d, se = stats([b - a for a, b in pairs])
    return f"{d:+.3f} ± {se:.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default=SETS)
    ap.add_argument("--md", default="outputs/OOD_tables.md")
    a = ap.parse_args(argv)
    out = ["| set (metric) | n | closed_tail | **text** | kv@0 old | kv_reg old | **kv@0 lever-1** | kv_reg lever-1 | "
           "wrong_doc (L1, es) | reg−kv@0 old | reg−kv@0 L1 | L1−old kv@0 | L1−old kv_reg | GPU |",
           "|" + "---|" * 14]
    for s in a.sets.split(","):
        pl, po = REPO / f"outputs/OOD_L1f_{s}/eval_test.json", REPO / f"outputs/OOD_old_{s}/eval_test.json"
        if not pl.exists():
            out.append(f"| {s} | (pending) |" + " |" * 12)
            continue
        L = json.load(open(pl))
        O = json.load(open(po)) if po.exists() else None
        S = json.load(open(REPO / f"outputs/OOD_sets/{s}.json"))
        from register_doc import load_split_items
        task = {it.key: it.task for it in load_split_items(str(REPO / S["split_dir"]), "test")}
        rl = {r["key"]: r for r in L["items"]}
        ro = {r["key"]: r for r in O["items"]} if O else {}
        groups = {"all": sorted(rl)}
        sub = sorted({task[k] for k in rl})
        if len(sub) > 1:
            for t in sub:
                groups[t] = sorted(k for k in rl if task[k] == t)
        gpu = "/".join(sorted(set(L["cost"].get("gpu", [])) | (set(O["cost"].get("gpu", [])) if O else set())))
        for g, ks in groups.items():
            ks = [k for k in ks if not ro or k in ro]
            col = lambda f: [f(k) for k in ks if f(k) is not None]  # noqa: E731
            ct = col(lambda k: rl[k]["closed_tail"]["score"] if "closed_tail" in rl[k] else None)
            tx = col(lambda k: rl[k]["text"]["score"] if "text" in rl[k] else None)
            l0 = col(lambda k: rl[k]["kv@0"]["score"])
            lr = col(lambda k: es_score(L, rl[k]))
            wr = col(lambda k: wrong_es(L, rl[k]))
            o0 = col(lambda k: ro[k]["kv@0"]["score"]) if O else []
            orr = col(lambda k: es_score(O, ro[k])) if O else []
            def pair(fa, fb):
                return [(fa(k), fb(k)) for k in ks if fa(k) is not None and fb(k) is not None]
            reg_l = pair(lambda k: rl[k]["kv@0"]["score"], lambda k: es_score(L, rl[k]))
            reg_o = pair(lambda k: ro[k]["kv@0"]["score"], lambda k: es_score(O, ro[k])) if O else []
            d0 = pair(lambda k: ro[k]["kv@0"]["score"], lambda k: rl[k]["kv@0"]["score"]) if O else []
            dr = pair(lambda k: es_score(O, ro[k]), lambda k: es_score(L, rl[k])) if O else []
            tk = {task[k] for k in ks}
            metric = "(acc)" if tk <= {"ood_lbv2", "ood_ibmc", "d25_fecf", "d25_cqacf"} else \
                "(F1)" if not tk & {"ood_lbv2", "ood_ibmc", "d25_fecf", "d25_cqacf"} else "(acc/F1)"
            name = f"{s}" if g == "all" else f"  └ {g}"
            out.append(f"| {name} {metric} | {len(ks)} | {ms(ct)} | **{ms(tx)}** | {ms(o0)} | {ms(orr)} | **{ms(l0)}** | "
                       f"{ms(lr)} | {ms(wr)} | {pse(reg_o)} | {pse(reg_l)} | {pse(d0)} | {pse(dr)} | {gpu} |")
        if L.get("oom_dropped") or (O and O.get("oom_dropped")):
            out.append(f"| ↳ {s} OOM-dropped: L1 {L.get('oom_dropped')} old {O.get('oom_dropped') if O else ''} |" + " |" * 13)
    txt = "\n".join(out)
    print(txt)
    Path(REPO / a.md).write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
