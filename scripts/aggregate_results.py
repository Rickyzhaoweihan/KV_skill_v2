"""Collect every eval_*.json, ladder json, history.json and probe_read.json under outputs/ into one
results table (outputs/RESULTS.json + a markdown summary on stdout)."""
from __future__ import annotations
import glob, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONDS = ["closed_book", "text", "kv", "wrong_doc", "null", "oracle_chunk"]


def main() -> int:
    out = {"runs": {}, "ladders": {}, "probes": {}}
    for f in sorted(glob.glob(str(ROOT / "outputs" / "*" / "eval_*.json")) + glob.glob(str(ROOT / "outputs" / "*" / "eval_rl" / "eval_*.json")) + glob.glob(str(ROOT / "outputs" / "*" / "eval_init" / "eval_*.json"))):
        run = Path(f).parent.name if Path(f).parent.name not in ("eval_rl", "eval_init") else Path(f).parent.parent.name + "_" + Path(f).parent.name[5:]
        if run.endswith("_tiny") or "smoke" in run:
            continue
        d = json.load(open(f)); s = d.get("summary") or {}
        args = {}
        try:
            args = json.load(open(Path(f).parent / "args.json"))
        except Exception:
            pass
        hist = None
        try:
            h = json.load(open(Path(f).parent / "history.json"))
            evs = [e for e in h.get("evals", h.get("history", [])) if isinstance(e, dict) and "val" in str(e)]
            hist = {"n_steps": h.get("steps") or len(h.get("steps_log", [])), "best_val": h.get("best_val")}
        except Exception:
            pass
        row = {"split": Path(f).stem.replace("eval_", ""), "reader": args.get("reader"), "encoder": args.get("encoder"),
               "center": bool(args.get("center")), "steps": args.get("steps"), "max_docs": args.get("max_docs"),
               "q_split": args.get("q_split"), "test_fold": args.get("test_fold"), "history": hist}
        for c in CONDS:
            if c in s and isinstance(s[c], dict):
                row[c] = {"score": s[c].get("score"), "n": s[c].get("n"), "gen": s[c].get("mean_gen_tokens")}
        g = s.get("kv_minus_wrong_doc")
        if isinstance(g, dict):
            gap = g.get("gap", g.get("mean", g.get("diff")))
            if gap is None:
                kv_s, wd_s = (s.get("kv") or {}).get("score"), (s.get("wrong_doc") or {}).get("score")
                gap = None if kv_s is None or wd_s is None else kv_s - wd_s
            row["kv_minus_wrong"] = gap; row["paired_n"] = g.get("n_paired") or g.get("n") or (s.get("kv") or {}).get("n")
        elif isinstance(g, (int, float)):
            row["kv_minus_wrong"] = g; row["paired_n"] = (s.get("kv") or {}).get("n")
        else:
            kv_s, wd_s = (s.get("kv") or {}).get("score"), (s.get("wrong_doc") or {}).get("score")
            row["kv_minus_wrong"] = None if kv_s is None or wd_s is None else kv_s - wd_s
            row["paired_n"] = (s.get("kv") or {}).get("n")
        out["runs"][f"{run}/{row['split']}"] = row
    # D7 native-state ceilings (scripts/eval_state.py): training-free prefix-cache conditions
    out["state_ceiling"] = {}
    for f in sorted(glob.glob(str(ROOT / "outputs" / "*" / "eval_state.json"))):
        run = Path(f).parent.name
        d = json.load(open(f)); s = d.get("summary") or {}; a = d.get("args") or {}
        row = {"reader": a.get("reader"), "encoder": a.get("encoder"), "split_dir": a.get("split_dir"),
               "fold": a.get("fold"), "n": d.get("n"), "harvest_check": d.get("harvest_check"),
               "bytes": d.get("mean_state_bytes"), "agree_all_text": s.get("agree_all_text")}
        for c, v in s.items():
            if isinstance(v, dict) and "score" in v:
                row[c] = {"score": v["score"], "n": v.get("n"), "gen": v.get("gen_tokens")}
            elif isinstance(v, dict) and "gap" in v:
                row[c] = {"gap": v["gap"], "se": v.get("se")}
        out["state_ceiling"][run] = row
        out["runs"].pop(f"{run}/state", None)
    # D8 native-KV compression sweeps (eval_prune.json)
    out["compression"] = {}
    for f in sorted(glob.glob(str(ROOT / "outputs" / "*" / "eval_prune.json"))):
        run = Path(f).parent.name
        d = json.load(open(f)); s = d.get("summary") or {}; g = d.get("gaps") or {}
        rows = {}
        for c, v in s.items():
            if not isinstance(v, dict) or "score" not in v or c.startswith("wrong"):
                continue
            w = s.get("wrong_" + c) or s.get("wrong_" + c.replace("kv_only", "kv")) or {}
            wk = "wrong_" + c if ("wrong_" + c) in s else ("wrong_kv" if c == "kv_only" else "wrong_" + c)
            w = s.get(wk) or {}
            gg = g.get(f"{c}-{wk}") or {}
            rows[c] = {"score": v["score"], "n": v.get("n"), "bytes": v.get("bytes"),
                       "wrong": w.get("score"), "gap": gg.get("gap"), "se": gg.get("se")}
        out["compression"][run] = {"n": d.get("n"), "full_bytes": d.get("mean_full_bytes"), "rows": rows,
                                   "args": {k: d.get("args", {}).get(k) for k in ("reader", "split_dir", "fold")}}
    for f in sorted(glob.glob(str(ROOT / "outputs" / "ladder*" / "*.json"))):
        if "tiny" in f:
            continue
        d = json.load(open(f)); out["ladders"][f"{Path(f).parent.name}/{Path(f).stem}"] = d.get("readers", d)
    for f in sorted(glob.glob(str(ROOT / "outputs" / "*" / "probe_read.json"))):
        out["probes"][Path(f).parent.name] = json.load(open(f)).get("summary")
    json.dump(out, open(ROOT / "outputs" / "RESULTS.json", "w"), indent=1)
    # markdown
    print("| run | split | enc | center | n | closed | text | kv | wrong | null | oracle | kv−wrong |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, r in out["runs"].items():
        run, split = k.split("/")
        def g(c):
            v = r.get(c); return "—" if not v or v.get("score") is None else f"{v['score']:.3f}"
        n = (r.get("kv") or {}).get("n", "")
        gap = "—" if r.get("kv_minus_wrong") is None else f"{r['kv_minus_wrong']:+.3f}"
        print(f"| {run} | {split} | {Path(str(r.get('encoder'))).name} | {'y' if r.get('center') else 'n'} | {n} | {g('closed_book')} | {g('text')} | {g('kv')} | {g('wrong_doc')} | {g('null')} | {g('oracle_chunk')} | {gap} |")
    print()
    for k, v in out["ladders"].items():
        print(f"ladder {k}: " + "; ".join(f"{m}: cb {x.get('closed_book',{}).get('score',float('nan')):.3f} text {x.get('text',{}).get('score',float('nan')):.3f}" for m, x in v.items() if isinstance(x, dict)))
    for k, v in out["probes"].items():
        print(f"probe {k}: {json.dumps(v)[:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
