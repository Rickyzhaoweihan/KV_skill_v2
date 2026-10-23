"""P6 go/no-go (EVAL-ONLY, no training): can the reader READ a document token that is injected in its NATIVE attention
value form (xenc/p6_native.py) at the right place?

Base: a plain D13 interface (default lever-1 full). Items: the P3 diagnostic's span val items (gold occurs whole-word in
the operator; scripts/p3_diag.py helpers). At every answer row the injection puts a one-hot p on the column holding the
NEXT gold token (P3's char-offset rule) and adds g · o_proj(W_V(LN(h_src(id))) ⊙ σ(gate)) to the attention output of
the chosen reader layers; nothing is trained (g is swept). Per condition:
  greedy F1 / EM                      (vs the base on the same items; go if oracle injection raises F1 clearly)
  teacher-forced gold acc / log p     (content rows of "<answer>gold</answer>": argmax == gold, mean log p(gold))
  forced copy rate (random token)     inject a RANDOM other document token instead: how often argmax == that token
                                      (is the injected token readable at all, independent of the question?)
  ‖injection‖ / ‖attention output‖    calibration per layer

    python scripts/p6_oracle.py --init outputs/D13_lever1_full/d13_best.pt --out outputs/P6_oracle
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401

DEFAULT_CONDS = ("emb@19:1,emb@19:4,emb@19+23+27+31:1,emb@19+23+27+31:4,"
                 "ctxfree@19:1,ctxfree@19:4,ctxfree@19+23+27+31:1,ctxfree@19+23+27+31:4")


def parse_conds(spec: str) -> list:
    """'emb@19+23:4,...' -> [(name, source, [layers], g)]."""
    out = []
    for c in [x for x in spec.split(",") if x]:
        src, rest = c.split("@")
        lay, g = rest.split(":")
        out.append((c, src, [int(x) for x in lay.split("+")], float(g)))
    return out


def stat(x):
    n = len(x)
    if not n:
        return float("nan"), float("nan")
    m = sum(x) / n
    return m, (math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1) / n) if n > 1 else float("nan"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--init", default="outputs/D13_lever1_full/d13_best.pt")
    ap.add_argument("--conds", default=DEFAULT_CONDS)
    ap.add_argument("--val", default="quality:5,musique:70,lastingbench:50,longhealth:0,qasper:90,contractnli:2,"
                    "shakespeare_cf:1,loogle:0,clipper:1")
    ap.add_argument("--tasks", default="qasper,musique,lastingbench,loogle")
    ap.add_argument("--exclude_sets", default=",".join(sorted(glob.glob("outputs/D22_sets/*.json"))))
    ap.add_argument("--exclude_docs", default="outputs/D13_lever1_leakage.json")
    ap.add_argument("--doc_len", default="loogle:32000:96000,clipper:32000:96000")
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=100000)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enc_device", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    from fit_d13 import D13Rt, build, select_items
    from p3_diag import col_covering, occurrences, oracle_columns, tok_match
    from register_d13 import doc_len_filter, excluded_docs
    from register_doc import SYSTEM_ANSWER, _get, log
    from register_state import _stop_ids
    from xenc.backbone import decode, text_ids
    from xenc.doc_tasks import exact_match, extract_answer, f1_score
    from xenc.e3_retrieval import answer_content_rows
    from xenc.p6_native import NativeInjector
    from xenc.state_skill import doc_prefix_text
    t0 = time.time()
    rng = random.Random(a.seed)
    reader, enc, tok, itf, lmap = build(a, init=a.init)
    rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens, lru=2)
    R = rt.R
    conds = parse_conds(a.conds)
    all_layers = sorted({l for _, _, ls, _ in conds for l in ls})
    inj = {s: NativeInjector(reader, all_layers, s) for s in sorted({c[1] for c in conds})}
    val, docs = select_items(a.val, "val", {1}, 4, 1)
    drop = excluded_docs(a.exclude_sets)
    X = json.load(open(a.exclude_docs))
    drop |= set(sum(X.values(), [])) if isinstance(X, dict) else set(X)
    keep = doc_len_filter(a.doc_len)
    tasks = set(a.tasks.split(","))
    val = [i for i in val if _get(i, "doc_id") not in drop and keep(_get(i, "doc_id"))
           and _get(i, "doc_id").split("_")[0] in tasks and not _get(i, "choices")]
    assert not ({_get(i, "doc_id") for i in val} & drop), "excluded doc in items"
    rt.docs = docs
    stops = _stop_ids(tok)
    log(f"P6 oracle: init {a.init} ({type(itf).__name__}) | conds {[c[0] for c in conds]} | layers {all_layers} | "
        f"items {len(val)} | GPU {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    def set_cond(c):
        for s, P in inj.items():
            for l in all_layers:
                P.g[l] = (c[3] if (c is not None and c[1] == s and l in c[2]) else 0.0)
            P.cols, P.rows = None, None

    @torch.no_grad()
    def gen(did, it, c, ctx, op, n_op):
        rows = rt.rows(did, it)
        set_cond(c)
        P = inj[c[1]] if c else None
        R.load(op, n_op)
        R.mode = "student"
        new = []
        try:
            out = reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=True, logits_to_keep=1)
            for _ in range(a.budget):
                nxt = int(out.logits[0, -1].argmax())
                new.append(nxt)
                txt = tok.decode(new, skip_special_tokens=True)
                if nxt in stops or "</answer>" in txt[-14:]:
                    break
                if P is not None:
                    P.cols = None
                    if "<answer>" in txt:
                        L = len(txt.split("<answer>", 1)[1])
                        P.cols = oracle_columns(ctx["ends"], ctx["spans"], L, n_op) or None
                out = reader(input_ids=torch.tensor([[nxt]], device=rt.dev), past_key_values=out.past_key_values,
                             use_cache=True, logits_to_keep=1)
        finally:
            R.mode, R.past = "off", {}
            set_cond(None)
        g = decode(tok, a.reader, new)
        return g.split("</answer>", 1)[0] + "</answer>" if "</answer>" in g else g

    @torch.no_grad()
    def tf(did, it, c, ctx, op, n_op, ans, mode):
        """mode 'gold': inject the next gold token at each content row; 'rand': a random other document token.
        -> per content row (argmax==gold, log p gold, argmax==injected)."""
        target = f"<answer>{ans}</answer>"
        enc_t = tok(target, add_special_tokens=False, return_offsets_mapping=True)
        tt, loc = answer_content_rows(tok, target)
        m = len(tt)
        rows = rt.rows(did, it, tt)
        base0 = len(rows) - (m + 1)
        a0 = len("<answer>")
        plan, inj_tok = {}, {}
        for i in loc:
            L = max(0, enc_t["offset_mapping"][i][0] - a0)
            cs = oracle_columns(ctx["ends"], ctx["spans"], L, n_op)
            if mode == "rand" and cs:
                gid = set(int(op["ids"][x]) for x in cs)
                for _ in range(50):
                    j = rng.randrange(ctx["S"], n_op)
                    if int(op["ids"][j]) not in gid:
                        cs = [j]
                        break
            if cs:
                plan[base0 + i] = cs
                inj_tok[i] = tok.convert_ids_to_tokens(int(op["ids"][cs[0]]))
        set_cond(c)
        if c is not None:
            inj[c[1]].rows = plan
        R.load(op, n_op)
        R.mode = "student"
        try:
            lg = reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=False,
                        logits_to_keep=m + 1).logits[0].float()
        finally:
            R.mode, R.past = "off", {}
            set_cond(None)
        lp = torch.log_softmax(lg, -1)
        res = []
        for i in loc:
            am = int(lp[i].argmax())
            r_ = {"gold": am == int(tt[i]), "lp": float(lp[i, int(tt[i])])}
            if i in inj_tok:
                r_["copy"] = tok_match(inj_tok[i], tok.convert_ids_to_tokens(am), i == loc[0]) or \
                    tok_match(tok.convert_ids_to_tokens(am), inj_tok[i], False)
            res.append(r_)
        return res

    items_out, n_skip = [], Counter()
    for it in val:
        if a.max_items and len(items_out) >= a.max_items:
            break
        did = _get(it, "doc_id")
        doc = docs[did]
        ptxt = doc_prefix_text(tok, a.reader, doc, SYSTEM_ANSWER)
        enc_ = tok(ptxt, add_special_tokens=False, return_offsets_mapping=True)
        full = list(enc_["input_ids"])
        if full != rt.pre(did):
            n_skip["prefix_mismatch"] += 1
            continue
        N = len(full) - a.tail
        ends = [e for _, e in enc_["offset_mapping"]]
        ans, spans = occurrences(ptxt, doc, _get(it, "answers"), ends[N - 1])
        if not spans:
            n_skip["gold_not_in_operator"] += 1
            continue
        S = len(tok(doc_prefix_text(tok, a.reader, "", SYSTEM_ANSWER), add_special_tokens=False)["input_ids"])
        ctx = {"ends": ends, "spans": spans, "S": S}
        rec = {"key": str(_get(it, "key")), "task": did.split("_")[0], "n_op": N, "answer_used": ans}
        try:
            op, n_op, _ = rt.operator(did)
            for P in inj.values():
                P.ids = op["ids"]
            gold_cols = sorted({c for s, e in spans for c in range(col_covering(ends, s), min(col_covering(ends, e - 1) + 1, N))})
            rand_cols = [rng.randrange(S, N) for _ in range(64)]
            for P in inj.values():                  # ctxfree runs the reader: precompute with every patch off
                P.values([int(op["ids"][c]) for c in gold_cols + rand_cols])
            answers = list(_get(it, "answers"))
            for c in [None] + conds:
                name = "base" if c is None else c[0]
                g = gen(did, it, c, ctx, op, n_op)
                pred = extract_answer(g)
                rec[name] = {"gen": g, "f1": f1_score(pred, answers), "em": exact_match(pred, answers)}
                rec[name]["tf_gold"] = tf(did, it, c, ctx, op, n_op, ans, "gold")
                if c is not None:
                    rec[name]["tf_rand"] = tf(did, it, c, ctx, op, n_op, ans, "rand")
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            rt._ops.clear()
            n_skip["oom"] += 1
            continue
        finally:
            for P in inj.values():
                P.ids = None
        items_out.append(rec)
        if len(items_out) % 20 == 0:
            log(f"  {len(items_out)} items ({time.time() - t0:.0f}s) | skips {dict(n_skip)}")
    with open(os.path.join(a.out, "items.jsonl"), "w") as f:
        for r in items_out:
            f.write(json.dumps(r) + "\n")
    # calibration pass: injection norm ratio per condition on a few items (gold injection, teacher-forced)
    ratios = {}
    summ = {"init": a.init, "n": len(items_out), "skips": dict(n_skip),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "conds": {}}
    for name in ["base"] + [c[0] for c in conds]:
        rs = [r for r in items_out if name in r]
        d = {"f1": stat([r[name]["f1"] for r in rs]), "em": stat([r[name]["em"] for r in rs])}
        if name != "base":
            d["f1_minus_base"] = stat([r[name]["f1"] - r["base"]["f1"] for r in rs])
            d["em_minus_base"] = stat([r[name]["em"] - r["base"]["em"] for r in rs])
            rr = [x for r in rs for x in r[name]["tf_rand"] if "copy" in x]
            d["rand_copy_rate"] = stat([float(x["copy"]) for x in rr])[0]
        gg = [x for r in rs for x in r[name]["tf_gold"]]
        d["tf_gold_acc"] = stat([float(x["gold"]) for x in gg])[0]
        d["tf_gold_lp"] = stat([x["lp"] for x in gg])[0]
        d["by_task_f1"] = {t: stat([r[name]["f1"] for r in rs if r["task"] == t])[0] for t in sorted({r["task"] for r in rs})}
        summ["conds"][name] = d
    for P in inj.values():
        P.ratios = {}
    for r in items_out[:20]:
        it = next(i for i in val if str(_get(i, "key")) == r["key"])
        did = _get(it, "doc_id")
        doc = docs[did]
        ptxt = doc_prefix_text(tok, a.reader, doc, SYSTEM_ANSWER)
        enc_ = tok(ptxt, add_special_tokens=False, return_offsets_mapping=True)
        ends = [e for _, e in enc_["offset_mapping"]]
        N = len(enc_["input_ids"]) - a.tail
        ans, spans = occurrences(ptxt, doc, _get(it, "answers"), ends[N - 1])
        S = len(tok(doc_prefix_text(tok, a.reader, "", SYSTEM_ANSWER), add_special_tokens=False)["input_ids"])
        op, n_op, _ = rt.operator(did)
        for c in conds:
            for P in inj.values():
                P.ids, P.ratios = op["ids"], {}
            tf(did, it, c, {"ends": ends, "spans": spans, "S": S}, op, n_op, ans, "gold")
            for l, v in inj[c[1]].ratios.items():
                ratios.setdefault(c[0], {}).setdefault(l, []).extend(v)
    summ["inj_ratio"] = {k: {str(l): sum(v) / len(v) for l, v in d.items()} for k, d in ratios.items()}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    L = [f"init `{a.init}` | n {summ['n']} | skips {summ['skips']} | {summ['gpu']}", "",
         "| condition | greedy F1 | Δ F1 vs base | EM | Δ EM | TF gold acc | TF mean log p(gold) | forced copy rate (random token) | ‖inj‖/‖attn‖ |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, d in summ["conds"].items():
        fmt = lambda t: f"{t[0]:+.3f} ± {t[1]:.3f}" if name != "base" else "—"   # noqa: E731
        rat = summ["inj_ratio"].get(name, {})
        L.append(f"| {name} | {d['f1'][0]:.3f} | {fmt(d.get('f1_minus_base', (0, 0)))} | {d['em'][0]:.3f} | "
                 f"{fmt(d.get('em_minus_base', (0, 0)))} | {d['tf_gold_acc']:.3f} | {d['tf_gold_lp']:.3f} | "
                 f"{d.get('rand_copy_rate', float('nan')):.3f} | "
                 + " ".join(f"L{l} {v:.2f}" for l, v in rat.items()) + " |")
    L += ["", "per-task greedy F1: " + "; ".join(f"{n}: " + ", ".join(f"{t} {v:.3f}" for t, v in d["by_task_f1"].items())
                                                for n, d in summ["conds"].items())]
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(L))
    log("\n".join(L))
    log(f"done ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
