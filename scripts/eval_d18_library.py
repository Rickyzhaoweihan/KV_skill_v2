"""D18 (2/2): KV-Skill as retrieval memory — a library of N document operators loaded at once; the question is about
one of them. The paper-form read routes over all operators in ONE reader pass (xenc/d18_library.py: routing softmax
over documents × slot softmax, on the D13 read); the encoder-side read and text RAG need a retriever first.

Reader rows (every condition except text): the doc-free chat header ("...<|im_start|>user\\n## Document\\n") + the
question suffix (or a context variant's conversation, --variant). Operators: the encoder's post-RoPE K/V of the FULL
document prefix at the paired encoder layers (no tail is held back: the reader sees no document tokens).
Conditions per item:
  closed            header + question, no read (the do-no-harm floor of this setting)
  paper_N           library read over N operators (gold + N−1 distractors), N ∈ --Ns; route mass on the gold document
                    at the last prompt row (mean over layers, heads) and top-1 routing accuracy
  encside_oracle    encoder-side read of the gold document (encoder: document + question; question rows injected)
  encside_bm25_N    encoder-side read of BM25's top-1 document among the N (re-uses encside_oracle when it is gold)
  text_oracle       the gold document's text in context;  text_bm25_N: BM25 top-1 document's text (RAG baseline)
Cost: per question the encoder-side read without a retriever needs N encoder passes over N encoder caches; we time
one pass (t_enc_q) and report N·t_enc_q next to the measured paper-form time per question at each N, and the bytes
each form must keep resident per document (paper: paired-layer K/V only; encoder-side: the whole encoder cache).

Libraries (deterministic): eval documents E = the first --eval_docs documents (sorted id order after a seeded
shuffle) of the judged split/fold; distractor pool = the other documents of --pool_splits (sorted, seeded shuffle).
For N ≤ |E|: E is cut into consecutive groups of N (N = 1: the gold document alone); for N > |E|: E ∪ the first
N − |E| pool documents. Every item of an eval document is judged.

    python scripts/eval_d18_library.py --interface outputs/D13_maps/d13_mlp.pt --split_dir data/quality_split \
        --eval_docs 10 --Ns 1,5,20,50 --out outputs/D18_lib_q_d13mlp
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402


def libraries(eval_docs: list, pool: list, Ns: list) -> dict:
    """-> {N: {doc_id_of_E: [library doc ids, gold included]}} (deterministic, see module doc)."""
    out = {}
    for N in Ns:
        m = {}
        if N <= len(eval_docs):
            for i in range(0, len(eval_docs), N):
                grp = eval_docs[i:i + N]
                if len(grp) < N:                          # ragged last group: pad from the pool
                    grp = grp + [d for d in pool if d not in grp][: N - len(grp)]
                for d in grp:
                    if d in eval_docs:
                        m[d] = list(grp)
        else:
            extra = [d for d in pool if d not in eval_docs][: N - len(eval_docs)]
            if len(extra) < N - len(eval_docs):
                raise ValueError(f"pool too small for N={N}")
            for d in eval_docs:
                m[d] = list(eval_docs) + extra
        out[N] = m
    return out


def bm25_query(it, ci=None, mode: str = "final") -> str:
    """Retriever query: the final user message (deployment) or the whole conversation; orig = question + options."""
    if ci is None or not ci.get("turns"):
        q = str(_get(it, "question"))
        ch = _get(it, "choices") or []
        return q + " " + " ".join(c["text"] for c in ch)
    from xenc.d18_context import strip_template
    if mode == "final":
        return strip_template(ci["final"])
    return strip_template(" ".join(u for u, _ in ci["turns"]) + " " + ci["final"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--B", default="outputs/D10_maps/encB8_9b_2b_post.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--interface", default=None)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None, help="LongHealth: eval documents = this fold's documents")
    ap.add_argument("--pool_splits", default="test,val")
    ap.add_argument("--pool_folds", default="", help="LongHealth: restrict distractor documents to these folds "
                    "(e.g. 0,1 = documents never used to train the D18 router; default all folds)")
    ap.add_argument("--eval_docs", type=int, default=10)
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--Ns", default="1,5,20,50")
    ap.add_argument("--beta", type=float, default=1.0, help="routing temperature on the content log-mass")
    ap.add_argument("--top_k", type=int, default=0, help="hard top-k routing (0 = soft)")
    ap.add_argument("--variant", default="orig", help="context variant of xenc/d18_context.py")
    ap.add_argument("--bm25_query", default="final", choices=["final", "conv"])
    ap.add_argument("--conds", default="closed,paper,paper2,encside_oracle,encside_bm25,text_oracle,text_bm25",
                    help="paper2 = two reader passes: question-level routing (xenc.d18_library.question_route, primary "
                         "aggregator lse_norm) then the single routed operator's read; no encoder at inference")
    ap.add_argument("--router", default=None, help="D18 router (scripts/train_d18router.py): adds paper2r_N (router "
                    "top-1 from a stock reader pass, then the single routed read) and paperr_N (library read with the "
                    "router's softmax as fixed document weights)")
    ap.add_argument("--adapters", default=None, help="D20 registration dir: every operator READ loads its document's "
                    "registered adapter (unregistered = hard error); multi-document mixed reads (paper_N>1, paper2's "
                    "soft first pass, paperr) are disabled — routing is by the router on the base interface")
    ap.add_argument("--adapter_budget", type=int, default=0)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--tail", type=int, default=64, help="(D13 runtime; the library itself holds back no tail)")
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from eval_d18_context import Runtime, ids_of, paired
    from xenc.d18_context import build_context_items, common_suffix_len, segments
    from xenc.d18_library import BM25, ROUTE_AGGS, D18LibReader, question_route, route_stats
    from xenc.doc_tasks import ctx_prompt, load_docs, score
    from xenc.state_skill import doc_prefix_text
    conds = a.conds.split(",")
    Ns = [int(x) for x in a.Ns.split(",")]
    os.makedirs(a.out, exist_ok=True)
    rt = Runtime(a)
    tok, hub = rt.tok, a.reader
    LR = D18LibReader(rt.reader, rt.itf)
    rt.P.grab("lib")
    LR.beta, LR.top_k = a.beta, a.top_k
    router = cap = None
    if a.router:
        from xenc.d18_router import D18Router, HiddenCap
        router = D18Router.load(a.router, rt.itf)[0].to(rt.dev).eval()
        cap = HiddenCap(rt.reader, rt.itf.layers)
    docs = load_docs(a.split_dir)
    rt.rt.docs = docs
    items = load_split_items(a.split_dir, a.split)
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
    rng = random.Random(a.seed)
    E = sorted({_get(i, "doc_id") for i in items})
    rng.shuffle(E)
    E = sorted(E[: a.eval_docs])
    pool_ids = set()
    for sp in a.pool_splits.split(","):
        pool_ids |= {_get(i, "doc_id") for i in load_split_items(a.split_dir, sp)}
    if a.fold is not None:                                       # LongHealth: all documents of the other folds
        pool_ids |= set(docs)
    if a.pool_folds:
        keep = {int(x) for x in a.pool_folds.split(",")}
        fold_of = {}
        for sp in ("train", "val", "test"):
            for i in load_split_items(a.split_dir, sp):
                f = (_get(i, "meta") or {}).get("fold")
                if f is not None:
                    fold_of[_get(i, "doc_id")] = int(f)
        pool_ids = {d for d in pool_ids if fold_of.get(d) in keep}
    pool = sorted(pool_ids - set(E))
    random.Random(a.seed + 1).shuffle(pool)
    libs = libraries(E, pool, Ns)
    items = [i for i in items if _get(i, "doc_id") in set(E)]
    cis = build_context_items(items, docs, a.variant)
    cis.sort(key=lambda c: c["doc_id"])
    if a.max_items and len(cis) > a.max_items:
        cis = cis[: a.max_items]
    all_lib_docs = sorted({d for m in libs.values() for g in m.values() for d in g})
    bm = BM25({d: docs[d] for d in all_lib_docs})
    log(f"D18 library eval: {a.encoder} -> {a.reader} | layers {rt.lmap} | interface {a.interface} | E={len(E)} docs, "
        f"{len(cis)} items ({a.variant}) | Ns {Ns} | {len(all_lib_docs)} operators | beta {a.beta} top_k {a.top_k}")

    # ---------------- operators: full document prefix, paired encoder layers (kept on the reader device)
    Tc = rt.rt.Tc
    header = ids_of(tok, doc_prefix_text(tok, hub, "", SYSTEM_ANSWER))
    ops, skip, t_h = {}, len(header), time.time()
    for d in all_lib_docs:
        ids = rt.rt.pre(d)
        skip = min(skip, next((k for k in range(min(len(ids), len(header))) if ids[k] != header[k]),
                              min(len(ids), len(header))))
        Tc.rec, Tc.kv_only = {}, True
        with torch.no_grad():
            rt.enc(input_ids=torch.tensor([ids], device=rt.rt.edev), use_cache=False, logits_to_keep=1)
        ops[d] = {e: (r["k"].to(rt.dev), r["v"].to(rt.dev)) for e, r in Tc.rec.items()}
        Tc.rec, Tc.kv_only = None, False
    t_harvest = (time.time() - t_h) / max(len(all_lib_docs), 1)
    op_bytes = {d: sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in o.values())
                for d, o in ops.items()}
    log(f"harvested {len(ops)} operators ({t_harvest:.2f}s/doc, mean {sum(op_bytes.values()) / len(op_bytes) / 2**20:.1f}"
        f" MiB/doc), shared header skip={skip}")

    def lib_ops(lib):
        return {e: [ops[d][e] for d in lib] for e in rt.lmap.values()}

    @torch.no_grad()
    def paper_lib(lib, rows, gold, qrows=None):
        from register_state import _stop_ids
        from xenc.backbone import decode
        rt.P.use("lib")
        if rt.ad is not None:
            assert len(lib) == 1, "registered judge: only single-operator reads (each with its own adapter)"
            tf = rt.use_doc(lib[0])
            LR.load({e: [tf({e: ops[lib[0]][e]})[e]] for e in rt.lmap.values()}, skip)
        else:
            LR.load(lib_ops(lib), skip)
        LR.mode = "lib"
        t = time.time()
        try:
            out = rt.reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=True, logits_to_keep=1)
            rs = route_stats(LR.route, lib.index(gold))
            qr = question_route(LR.logit_rows, qrows) if qrows is not None else {}
            stops, new = _stop_ids(tok), []
            for _ in range(a.budget):
                nxt = int(out.logits[0, -1].argmax())
                new.append(nxt)
                if nxt in stops or "</answer>" in tok.decode(new[-12:], skip_special_tokens=True):
                    break
                out = rt.reader(input_ids=torch.tensor([[nxt]], device=rt.dev), past_key_values=out.past_key_values,
                                use_cache=True, logits_to_keep=1)
        finally:
            LR.mode = "off"
            LR.reset()
        txt = decode(tok, hub, new)
        if "</answer>" in txt:
            txt = txt.split("</answer>", 1)[0] + "</answer>"
        return txt, len(new), rs, time.time() - t, qr

    enc_cache: dict = {}

    def enc_doc_cache(d):
        if d not in enc_cache:
            enc_cache.clear()
            with torch.no_grad():
                out = rt.enc(input_ids=torch.tensor([rt.rt.pre(d)], device=rt.rt.edev), use_cache=True,
                             logits_to_keep=1)
            enc_cache[d] = out.past_key_values
        return enc_cache[d]

    def encside(d, conv, header_rows):
        t = time.time()
        g, n = rt.gen_encside(enc_doc_cache(d), conv, [(header_rows, None), (conv, (0, len(conv)))], a.budget)
        return g, n, time.time() - t

    def enc_q_time(d, conv):
        import copy as _c
        past = _c.deepcopy(enc_doc_cache(d))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t = time.time()
        rt.enc_run(conv, past)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        return time.time() - t

    rows, t0 = [], time.time()
    ck = open(os.path.join(a.out, f"lib_{a.split}.ckpt.jsonl"), "w")
    for ci in cis:
        did, it = ci["doc_id"], ci["item"]
        seg = segments(tok, hub, ci, SYSTEM_ANSWER)
        conv = ids_of(tok, seg["conv_str"])
        rows_ids = header + conv
        rec = {"key": ci["key"], "doc_id": did}
        single: dict = {}                                     # routed doc -> single-operator read (paper2 cache)
        q = bm25_query(it, ci, a.bm25_query)
        sc = lambda g, n: {"gen": g, "score": float(score(it, g)), "gen_tokens": n}   # noqa: E731
        if "closed" in conds:
            rt.P.use("d13")
            rec["closed"] = sc(*rt.greedy(rows_ids, a.budget))
        if "encside_oracle" in conds or "encside_bm25" in conds:
            g, n, tt = encside(did, conv, header)
            rec["encside_oracle"] = {**sc(g, n), "sec": tt}
            rec["t_enc_q"] = enc_q_time(did, conv)
        if "text_oracle" in conds or "text_bm25" in conds:
            rt.P.use("d13")
            rec["text_oracle"] = sc(*rt.text_cached(did, conv, a.budget))
        if router is not None:
            rt.P.use("lib")
            LR.mode = "off"
            t_r = time.time()
            cap.h, cap.on = {}, True
            try:
                with torch.no_grad():
                    rt.reader(input_ids=torch.tensor([rows_ids], device=rt.dev), use_cache=False, logits_to_keep=1)
            finally:
                cap.on = False
            rxs = {li: h[0, len(header):].float() for li, h in cap.h.items()}
            rt.base_itf()                                        # router scores with the BASE interface
            rpos = torch.arange(len(header), len(rows_ids), device=rt.dev)
            t_r = time.time() - t_r
        for N in Ns:
            lib = libs[N][did]
            if router is not None and N > 1:
                rt.base_itf()
                with torch.no_grad():
                    t_s = time.time()
                    sco = router(rxs, rpos, {e: [ops[d][e][0] for d in lib] for e in rt.lmap.values()}, skip)
                    pr = torch.softmax(sco.float(), 0)
                    t_s = time.time() - t_s + t_r
                gi = lib.index(did)
                ch = lib[int(sco.argmax())]
                rinfo = {"top1": float(ch == did), "gold_rank": int((sco > sco[gi]).sum()),
                         "gold_mass": float(pr[gi]), "chance": 1.0 / len(lib)}
                if ch not in single:
                    g1, n1, _, t1, _ = paper_lib([ch], rows_ids, ch)
                    single[ch] = (sc(g1, n1), t1)
                rec[f"paper2r_{N}"] = {**single[ch][0], **rinfo, "sec": t_s + single[ch][1]}
            if router is not None and N > 1 and rt.ad is None:
                LR.wfix = pr
                try:
                    g, n, _, tt, _ = paper_lib(lib, rows_ids, did)
                finally:
                    LR.wfix = None
                rec[f"paperr_{N}"] = {**sc(g, n), **rinfo, "sec": t_s + tt}
            if "paper" in conds and a.interface and (rt.ad is None or len(lib) == 1):
                g, n, rs, tt, qr = paper_lib(lib, rows_ids, did, slice(len(header), len(rows_ids)))
                rec[f"paper_{N}"] = {**sc(g, n), **rs, "sec": tt}
                if "paper2" in conds and N > 1:
                    gi = lib.index(did)
                    for agg in ROUTE_AGGS:
                        sco = qr[agg]
                        ch = lib[int(sco.argmax())]
                        rk = sorted(range(len(lib)), key=lambda j: -float(sco[j])).index(gi)
                        if ch not in single:
                            g1, n1, _, t1, _ = paper_lib([ch], rows_ids, ch)
                            single[ch] = (sc(g1, n1), t1)
                        rec[f"paper2{'' if agg == 'lse_norm' else '_' + agg}_{N}"] = {
                            **single[ch][0], "top1": float(ch == did), "gold_rank": rk, "chance": 1.0 / len(lib),
                            "sec": tt + single[ch][1]}
            top = bm.top1(q, set(lib))
            rec[f"bm25_{N}"] = {"doc": top, "hit": float(top == did)}
            if "encside_bm25" in conds:
                if top == did:
                    rec[f"encside_bm25_{N}"] = dict(rec["encside_oracle"])
                else:
                    g, n, tt = encside(top, conv, header)
                    rec[f"encside_bm25_{N}"] = {**sc(g, n), "sec": tt}
            if "text_bm25" in conds:
                if top == did:
                    rec[f"text_bm25_{N}"] = dict(rec["text_oracle"])
                else:
                    rt.P.use("d13")
                    rec[f"text_bm25_{N}"] = sc(*rt.greedy(ids_of(tok, _render_text(tok, hub, ci, docs[top])),
                                                         a.budget))
        rows.append(rec)
        ck.write(json.dumps(rec) + "\n")
        ck.flush()
        if len(rows) % 10 == 0:
            s = _summ(rows, Ns)
            log(f"  [{len(rows)}/{len(cis)}] " + " ".join(f"{k}={v['score']:.3f}" for k, v in s.items()
                                                          if isinstance(v, dict) and "score" in v)
                + f" | {(time.time() - t0):.0f}s")
    summ = _summ(rows, Ns)
    summ["cost"] = {"harvest_sec_per_doc": t_harvest, "op_MiB_per_doc": sum(op_bytes.values()) / len(op_bytes) / 2**20,
                    "enc_cache_MiB_per_doc": _cache_mib(enc_cache), "skip": skip}
    tq = [r["t_enc_q"] for r in rows if "t_enc_q" in r]
    if tq:
        mq = sum(tq) / len(tq)
        summ["cost"]["t_enc_q"] = mq
        summ["cost"]["encside_no_retriever_sec"] = {str(N): N * mq for N in Ns}
    for N in Ns:
        p = paired(rows, f"paper_{N}", "closed")
        if p:
            summ[f"paper_{N}-closed"] = p
        for a2, b2 in [(f"{p2}_{N}", b) for p2 in ("paper2", "paper2r", "paperr")
                       for b in ("closed", "text_oracle", f"text_bm25_{N}", f"encside_bm25_{N}")]:
            p = paired(rows, a2, b2)
            if p:
                summ[f"{a2}-{b2}"] = p
    json.dump({"summary": summ, "args": vars(a), "libraries": {str(N): libs[N] for N in Ns}, "items": rows},
              open(os.path.join(a.out, f"lib_{a.split}.json"), "w"), indent=1)
    log(f"==== eval_d18_library [{a.split}] n={len(rows)} -> {a.out}")
    for k, v in summ.items():
        log(f"  {k}: {json.dumps(v) if not isinstance(v, dict) else {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}}")
    return 0


def _render_text(tok, hub, ci, doc_text):
    from xenc.d18_context import render
    return render(tok, hub, ci, SYSTEM_ANSWER, doc_text)


def _cache_mib(cache_dict) -> float:
    tot = 0
    for c in cache_dict.values():
        for t in _tensors(c):
            tot += t.numel() * t.element_size()
    return tot / 2**20


def _tensors(obj, depth=0):
    if depth > 4:
        return
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _tensors(x, depth + 1)
    elif hasattr(obj, "__dict__"):
        for x in vars(obj).values():
            yield from _tensors(x, depth + 1)


def _summ(rows, Ns) -> dict:
    keys = ["closed", "encside_oracle", "text_oracle"] + [f"{p}_{N}" for N in Ns for p in (
        "paper", "paper2", "paper2_lse", "paper2_max", "paper2r", "paperr", "encside_bm25", "text_bm25")]
    s = {}
    for k in keys:
        v = [r[k] for r in rows if k in r]
        if not v:
            continue
        s[k] = {"score": sum(x["score"] for x in v) / len(v), "n": len(v),
                "mean_gen_tokens": sum(x["gen_tokens"] for x in v) / len(v)}
        if k.startswith(("paper2", "paperr")):
            s[k]["route_top1"] = sum(x["top1"] for x in v) / len(v)
            s[k]["mean_gold_rank"] = sum(x["gold_rank"] for x in v) / len(v)
            s[k]["chance"] = v[0]["chance"]
            s[k]["sec"] = sum(x["sec"] for x in v) / len(v)
            if "gold_mass" in v[0]:
                s[k]["gold_mass"] = sum(x["gold_mass"] for x in v) / len(v)
        if k.startswith("paper_"):
            s[k]["gold_mass"] = sum(x["gold_mass"] for x in v) / len(v)
            s[k]["route_top1"] = sum(x["top1"] for x in v) / len(v)
            s[k]["chance"] = v[0]["chance"]
            s[k]["sec"] = sum(x["sec"] for x in v) / len(v)
    for N in Ns:
        v = [r[f"bm25_{N}"]["hit"] for r in rows if f"bm25_{N}" in r]
        if v:
            s[f"bm25_hit_{N}"] = {"score": sum(v) / len(v), "n": len(v)}
    return s


if __name__ == "__main__":
    sys.exit(main())
