#!/usr/bin/env python
"""ROUTE probe: a cheap, label-free per-document metric for "should this new document be registered?".

For each document, K self-questions (written by the self-question writer from the document; NO test question,
NO test label) are answered by
  kv    : the UNREGISTERED read (kv@0, the shared base; reader never sees the document text),
  tail  : the reader with only the 64-token tail and no read (closed_tail: how much is answerable without the doc),
  text  : (analysis only, optional) the reader with the document text in context,
and each answer is scored against the WRITER's own answer (MCQ: same option; span: token F1).
Deployment metric = probe_kv (and probe_kv − probe_tail): a handful of short kv decodes + the self-question writer.

The backbone / interface / layer map / max_tokens / self-question file are read from an existing registration run
(--from_run), so the probe sees exactly the operator that run registered. --interface overrides the base.
Reuses scripts/register_d20.py (D20 runtime, unmodified) for every model call.

usage (GPU):
  python scripts/route_probe.py --from_run outputs/D13L1f_qasper outputs/D13L1f_loogle ... \
      --out outputs/ROUTE_probe/L1f --k 12 --conds kv,tail,text --text_max_tok 40000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import zlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def log(*x):
    print(time.strftime("%H:%M:%S"), *x, flush=True)


def run_args(run):
    for n in ("eval_test_full.json", "eval_test.json", "eval_val.json"):
        p = os.path.join(run, n)
        if os.path.exists(p):
            E = json.load(open(p))
            return E["args"], sorted({r["doc_id"] for r in E["items"]})
    raise FileNotFoundError(f"no eval_*.json in {run}")


def register_d20_namespace(argv):
    """register_d20's own argparse defaults (+ its post-processing), without running its main."""
    import argparse as _ap
    import register_d20

    class _Got(Exception):
        pass

    orig = _ap.ArgumentParser.parse_args

    def grab(self, args=None, namespace=None):
        raise _Got(orig(self, args, namespace))

    _ap.ArgumentParser.parse_args = grab
    try:
        register_d20.main(argv)
    except _Got as e:
        return e.args[0]
    finally:
        _ap.ArgumentParser.parse_args = orig
    raise RuntimeError("register_d20.main did not parse its arguments")


def probe_items(selfq_items, did, k, holdout=0.2, max_items=48):
    """K unshuffled self-questions: the registration's held-out group first (same split rule), then the rest."""
    its = [dict(x) for x in selfq_items][:max_items]
    base = lambda x: x["key"].rsplit("_p", 1)[0] if "_p" in x["key"] else x["key"]   # noqa: E731
    keys = sorted({base(x) for x in its})
    random.Random(zlib.crc32(did.encode())).shuffle(keys)
    hk = set(keys[: max(1, int(round(holdout * len(keys))))])
    H = [x for x in its if base(x) in hk]
    rest = [x for x in its if base(x) not in hk]
    out = []
    for x in H + rest:
        if x.get("choices") and not x.get("correct_label"):
            continue                              # no writer label to score against
        if not x.get("choices") and not any(a for a in (x.get("answers") or [])):
            continue
        out.append(x)
        if len(out) >= k:
            break
    return out, len(H)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_run", nargs="+", required=True)
    ap.add_argument("--interface", default=None, help="override the base interface of the runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--conds", default="kv,tail,text")
    ap.add_argument("--text_max_tok", type=int, default=40000, help="skip the (analysis-only) text cond above this")
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--docs_json", default=None, help="optional JSON list of doc ids to probe (default: judged docs)")
    ap.add_argument("--selfq", default=None, help="override the runs' self-question file")
    ap.add_argument("--split_dir", default=None, help="override (OOD sets)")
    a = ap.parse_args(argv)
    import torch
    from register_d20 import D20, load_selfq
    from register_state import greedy
    from xenc.backbone import build_chat
    from xenc.d20_reg import assert_no_test_questions
    from xenc.doc_tasks import ctx_prompt, load_docs, score
    from d20_selfq import task_questions
    from register_doc import SYSTEM_ANSWER

    os.makedirs(a.out, exist_ok=True)
    conds = a.conds.split(",")
    runs = [(r, *run_args(r)) for r in a.from_run]
    A0 = runs[0][1]
    key_args = ("reader", "encoder", "B", "layers", "runtime", "tail")
    for r, A, _ in runs[1:]:
        for k in key_args:
            assert A.get(k) == A0.get(k), f"{r}: {k} {A.get(k)} != {A0.get(k)} (one backbone per process)"
        if a.interface is None:
            assert A.get("interface") == A0.get("interface"), f"{r}: different interface; pass --interface"
    S = None
    out_path = os.path.join(a.out, "probe.jsonl")
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            try:
                x = json.loads(line)
                done.add((x["run"], x["doc_id"]))
            except Exception:
                pass
    fo = open(out_path, "a")
    for run, A, docs_judged in runs:
        split_dir = a.split_dir or A["split_dir"]
        selfq_path = a.selfq or A["selfq"]
        ns = register_d20_namespace(["--split_dir", split_dir, "--selfq", selfq_path, "--out", a.out])
        for k, v in A.items():
            if hasattr(ns, k):
                setattr(ns, k, v)
        ns.out, ns.split_dir, ns.selfq = a.out, split_dir, selfq_path
        if a.interface:
            ns.interface = a.interface
        if ns.teacher == "none" or ns.enc_only:       # same post-processing as register_d20.main
            ns.enc_only, ns.teacher, ns.kl, ns.ce = 1, "none", 0.0, 0.0
        if S is None:
            torch.manual_seed(ns.seed)
            S = D20(ns)
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            log(f"ROUTE probe: {ns.encoder} -> {ns.reader} | interface {ns.interface} | B {ns.B} | layers {ns.layers} "
                f"| tail {ns.tail} | gpu {gpu}")
        S.a = ns
        S.rt.max_tokens = int(ns.max_tokens)
        S.rt._pre, S.rt._ops = {}, type(S.rt._ops)()
        S.rt.docs = load_docs(split_dir)
        S.itf.read_tau, S.itf.read_logn_ref = float(getattr(ns, "read_tau", 1.0)), int(getattr(ns, "read_logn_ref", 0))
        selfq = load_selfq(selfq_path)
        tq = task_questions(split_dir)
        dlist = json.load(open(a.docs_json)) if a.docs_json else docs_judged
        if a.max_docs:
            dlist = dlist[: a.max_docs]
        log(f"{run}: {len(dlist)} docs | split {split_dir} | selfq {selfq_path} | max_tokens {ns.max_tokens} | conds {conds}")
        S.ad.reset()
        snap0 = S.ad.snapshot()                    # step 0 = the shared base, unregistered
        for j, did in enumerate(dlist):
            if (run, did) in done:
                continue
            if did not in selfq:
                log(f"  {did}: no self-questions, skipped")
                continue
            P, n_h = probe_items(selfq[did], did, a.k, ns.holdout or 0.2, ns.max_items)
            if not P:
                continue
            assert_no_test_questions(P, tq)        # LEAKAGE GUARD: no probe equals / contains a task question
            n_tok = len(S.rt.pre(did))
            t0 = time.time()
            S.rt.operator(did)
            torch.cuda.synchronize()
            t_op = time.time() - t0
            recs, tm = [], {c: 0.0 for c in conds}
            try:
                for it in P:
                    rec = {"key": it["key"], "template": (it.get("meta") or {}).get("template")}
                    if "kv" in conds:
                        t1 = time.time()
                        g, n = S.gen(did, it, snap0, a.budget)
                        tm["kv"] += time.time() - t1
                        rec["kv"] = {"gen": g, "score": float(score(it, g)), "n": n}
                    if "tail" in conds:
                        t1 = time.time()
                        g, n = S.rt.gen(did, it, a.budget, read=False)
                        tm["tail"] += time.time() - t1
                        rec["tail"] = {"gen": g, "score": float(score(it, g)), "n": n}
                    if "text" in conds and n_tok <= a.text_max_tok:
                        t1 = time.time()
                        g, n = greedy(S.reader, S.tok, ns.reader,
                                      build_chat(S.tok, ns.reader, SYSTEM_ANSWER, ctx_prompt(it, S.rt.docs[did])), a.budget)
                        tm["text"] += time.time() - t1
                        rec["text"] = {"gen": g, "score": float(score(it, g)), "n": n}
                    recs.append(rec)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                log(f"  {did}: OOM after {len(recs)} probes; kept what finished")
            m = lambda c: (sum(r[c]["score"] for r in recs if c in r) / sum(c in r for r in recs)  # noqa: E731
                           if any(c in r for r in recs) else None)
            row = {"run": run, "doc_id": did, "interface": ns.interface, "n_probe": len(recs), "n_holdout_first": n_h,
                   "n_tok": n_tok, "probe_kv": m("kv"), "probe_tail": m("tail"), "probe_text": m("text"),
                   "t_op_s": round(t_op, 2), "t_s": {c: round(v, 2) for c, v in tm.items()},
                   "gpu": torch.cuda.get_device_name(0), "items": recs}
            fo.write(json.dumps(row) + "\n")
            fo.flush()
            log(f"  [{j + 1}/{len(dlist)}] {did} n_tok {n_tok}: kv {row['probe_kv']} tail {row['probe_tail']} "
                f"text {row['probe_text']} | op {t_op:.1f}s kv {tm.get('kv', 0):.1f}s text {tm.get('text', 0):.1f}s")
    log("done")


if __name__ == "__main__":
    sys.exit(main())
