"""E3 diagnostic: does the read retrieve the gold span? (held-out VAL documents only; no training, no test docs)

For each val item whose gold answer occurs in the operator's document columns, at the rows that predict the gold
answer tokens (teacher-forced on the gold answer), compare where three attention distributions put their mass over
the document-body columns [S, N) (head-mixture of the full softmax, xenc/e3_retrieval.mixture_logp):
  student   the base interface's read (D13 e2e mode: the student's own read flows downstream = inference condition)
  enc4b     the 4B encoder's own pass over [document ; tail + question + answer] (the stage-1 / dist teacher)
  text9b    the 9B reader reading the document as text (every full-attention layer of the 9B)
Metrics: hit@1 (argmax column inside the gold span, also ±3 tokens), gold mass, MRR, uniform mass; best single head;
MuSiQue: argmax inside a supporting paragraph. Also the base's greedy kv answer F1 (diagnostic only; UNREGISTERED, not
a result), split by whether the student's read hits the gold span.

    python scripts/e3_retrieval_diag.py --tasks qasper:60,musique:60,lastingbench:40 --out outputs/E3_diag
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, _get, log  # noqa: E402


def head_hits(logits, rows, S, N, mask):
    """per-head hit@1 at the given rows -> [H] float (mean over rows)."""
    lg = logits.reshape(-1, logits.shape[-2], logits.shape[-1])[:, rows, S:N]
    top = lg.argmax(-1)                                    # [H, R]
    return mask[top].float().mean(-1).cpu()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--interface", default="outputs/D13_4b_reg_w10/d13_best.pt")
    ap.add_argument("--tasks", default="qasper:60,musique:60,lastingbench:40")
    ap.add_argument("--split", default="val")
    ap.add_argument("--q_per_doc", type=int, default=2)
    ap.add_argument("--text_layers", default="3,7,11,15,19,23,27,31")
    ap.add_argument("--exclude_sets", default="")
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--no_gen", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    from fit_d13 import D13Rt, build, select_items
    from register_d13 import excluded_docs
    from xenc.backbone import build_chat, text_ids
    from xenc.d13_read import teacher_logits
    from xenc.doc_tasks import ctx_prompt, gold_text, score
    from xenc.e3_retrieval import QKCap, answer_content_rows, body_start, chars_to_mask, dilate, gold_mask, mixture_logp, retrieval_metrics
    from xenc.state_skill import doc_prefix_text
    reader, enc, tok, itf, lmap = build(a, init=a.interface)
    rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens)
    tl = [int(x) for x in a.text_layers.split(",")]
    cap = QKCap(reader, tl)
    items, docs = select_items(a.tasks, a.split, None, a.q_per_doc, a.seed)
    drop = excluded_docs(a.exclude_sets)
    items = [i for i in items if _get(i, "doc_id") not in drop and not _get(i, "choices")]
    rt.docs = docs
    S = body_start(tok, a.reader, SYSTEM_ANSWER)
    n_open, n_close = len(text_ids(tok, "<answer>")), len(text_ids(tok, "</answer>"))
    log(f"E3 diag: interface {a.interface} | encoder {a.encoder} | B {a.B} | layer map {lmap} | text layers {tl} | "
        f"{len(items)} items ({a.tasks}, split {a.split}) | excluded docs {len(drop)} | body start S={S}")
    out_rows, t0 = [], time.time()
    skip = {"truncated": 0, "no_gold": 0, "misaligned": 0, "fmt": 0}
    R = rt.R
    for n, it in enumerate(items):
        did = _get(it, "doc_id")
        doc = docs[did]
        ptxt = doc_prefix_text(tok, a.reader, doc, SYSTEM_ANSWER)
        full = tok(ptxt, add_special_tokens=False)["input_ids"]
        pre = rt.pre(did)
        if len(pre) != len(full):
            skip["truncated"] += 1
            continue
        N = len(pre) - a.tail
        mask = gold_mask(tok, ptxt, doc, _get(it, "answers"), N, S)
        if mask is None:
            skip["no_gold"] += 1
            continue
        tt = text_ids(tok, gold_text(it))
        m = len(tt)
        tt_chk, loc = answer_content_rows(tok, gold_text(it))       # window rows predicting the answer content
        if tt_chk != tt or not loc:
            skip["fmt"] += 1
            continue
        tids = build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, doc)) + tt
        if len(tids) > a.max_tokens or tids[:N] != pre[:N]:
            skip["misaligned"] += 1
            continue
        rows = rt.rows(did, it, tt)
        T = len(rows)
        srows = [T - (m + 1) + i for i in loc]
        masks = {"span": mask, "span_w3": dilate(mask, 3)}
        meta = _get(it, "meta") or {}
        if meta.get("para_spans") and meta.get("supporting_idx") is not None:
            d0 = ptxt.rfind(doc)
            sp = [(d0 + meta["para_spans"][k][0], d0 + meta["para_spans"][k][1]) for k in meta["supporting_idx"]]
            off = tok(ptxt, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            pm = chars_to_mask(off, sp, N, S)
            if pm is not None:
                masks["passage"] = pm
        dev = rt.dev
        masks = {k: v.to(dev) for k, v in masks.items()}
        rec_out = {"key": _get(it, "key"), "doc_id": did, "task": did.split("_")[0], "N": N, "n_gold": int(mask.sum()),
                   "src": {}}
        with torch.no_grad():
            op, n_op, _ = rt.operator(did)
            assert n_op == N
            # student (on-policy e2e read)
            R.load(op, n_op)
            R.mode, R.need_logits, R.rec = "e2e", True, {}
            try:
                reader(input_ids=torch.tensor([rows], device=dev), use_cache=False, logits_to_keep=1)
                stu = {li: R.rec[li]["logits"] for li in lmap}
            finally:
                R.mode, R.rec, R.need_logits = "off", {}, False
            teach = rt.teacher(did, rows)
            encl = {li: teacher_logits(teach[e]["q"], op[e][0], teach[e]["k"], itf.att[str(li)].scaling)
                    for li, e in lmap.items()}
            cap.keep, cap.rec = m + 1, {}
            try:
                reader(input_ids=torch.tensor([tids], device=dev), use_cache=False, logits_to_keep=1)
                txt = {e: cap.logits(e) for e in tl}
            finally:
                cap.keep = None
            srcs = [(f"student@{li}", stu[li], srows) for li in lmap] + \
                   [(f"enc4b@{li}", encl[li], srows) for li in lmap] + [(f"text9b@{e}", txt[e], loc) for e in tl]
            tops = {}
            for name, lg, rr in srcs:
                lp = mixture_logp(lg, rr, S, N)
                full_p = torch.softmax(lg.reshape(-1, lg.shape[-2], lg.shape[-1])[:, rr, :].float(), -1).mean(0)
                d = {"doc_mass": float(full_p[:, S:N].sum(-1).mean())}
                for mk, mv in masks.items():
                    r = retrieval_metrics(lp, mv)
                    d[mk] = {k: r[k] for k in ("hit1", "mass", "mrr", "uniform")}
                d["first_hit_w3"] = float(masks["span_w3"][lp[0].argmax()])
                d["head_hit_w3"] = head_hits(lg, rr, S, N, masks["span_w3"]).tolist()
                tops[name] = lp.argmax(-1).cpu()
                rec_out["src"][name] = d
            for li in lmap:   # agreement of the student's top column with each teacher's (±3 tokens)
                rec_out["src"][f"student@{li}"]["agree_enc_w3"] = float(
                    ((tops[f"student@{li}"] - tops[f"enc4b@{li}"]).abs() <= 3).float().mean())
                rec_out["src"][f"student@{li}"]["agree_text_w3"] = float(
                    ((tops[f"student@{li}"] - tops[f"text9b@{li}"]).abs() <= 3).float().mean()) if li in tl else None
            del stu, encl, txt, teach
        if not a.no_gen:
            g, ng = rt.gen(did, it, 32)
            rec_out["kv_base_f1"] = float(score(it, g))
            rec_out["kv_base_gen"] = g
        out_rows.append(rec_out)
        if (n + 1) % 20 == 0:
            log(f"  {n + 1}/{len(items)} items, kept {len(out_rows)}, skipped {skip} ({time.time() - t0:.0f}s)")
            with open(os.path.join(a.out, "items.jsonl"), "w") as f:
                for r in out_rows:
                    f.write(json.dumps(r) + "\n")
    with open(os.path.join(a.out, "items.jsonl"), "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    summ = summarize(out_rows, list(lmap), tl)
    summ["skipped"], summ["args"] = skip, vars(a)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    log(summ["table"])
    log(f"done: {len(out_rows)} items, skipped {skip} ({time.time() - t0:.0f}s)")
    return 0


def summarize(rows, layers, tl) -> dict:
    import statistics as st
    out, lines = {}, []
    tasks = sorted({r["task"] for r in rows})
    names = [f"student@{li}" for li in layers] + [f"enc4b@{li}" for li in layers] + [f"text9b@{e}" for e in tl]
    lines.append("| task | source | n | hit@1 span | hit@1 ±3 | first-row hit ±3 | gold mass | uniform | MRR | best-head hit ±3 "
                 "| passage hit | doc mass | agree enc / text ±3 |")
    lines.append("|" + "---|" * 13)
    for t in tasks:
        R = [r for r in rows if r["task"] == t]
        out[t] = {}
        for nm in names:
            D = [r["src"][nm] for r in R if nm in r["src"]]
            if not D:
                continue
            mean = lambda xs: sum(xs) / max(len(xs), 1)   # noqa: E731
            H = torch.tensor([d["head_hit_w3"] for d in D]).mean(0)
            pas = [d["passage"]["hit1"] for d in D if "passage" in d]
            o = {"n": len(D), "hit1": mean([d["span"]["hit1"] for d in D]), "hit1_w3": mean([d["span_w3"]["hit1"] for d in D]),
                 "first_w3": mean([d["first_hit_w3"] for d in D]), "mass": mean([d["span"]["mass"] for d in D]),
                 "uniform": mean([d["span"]["uniform"] for d in D]), "mrr": mean([d["span"]["mrr"] for d in D]),
                 "best_head_w3": float(H.max()), "best_head": int(H.argmax()), "doc_mass": mean([d["doc_mass"] for d in D]),
                 "passage_hit": mean(pas) if pas else None,
                 "agree_enc": mean([d["agree_enc_w3"] for d in D]) if "agree_enc_w3" in D[0] else None,
                 "agree_text": mean([d["agree_text_w3"] for d in D if d.get("agree_text_w3") is not None])
                 if D[0].get("agree_text_w3") is not None else None}
            out[t][nm] = o
            f = lambda x: "—" if x is None else f"{x:.3f}"   # noqa: E731
            lines.append(f"| {t} | {nm} | {o['n']} | {o['hit1']:.3f} | {o['hit1_w3']:.3f} | {o['first_w3']:.3f} | "
                         f"{o['mass']:.3f} | {o['uniform']:.3f} | {o['mrr']:.3f} | {o['best_head_w3']:.3f} (h{o['best_head']}) | "
                         f"{f(o['passage_hit'])} | {o['doc_mass']:.3f} | {f(o['agree_enc'])} / {f(o['agree_text'])} |")
        if R and "kv_base_f1" in R[0]:
            li_mid = layers[len(layers) // 2]
            hit = [r["kv_base_f1"] for r in R if r["src"][f"student@{li_mid}"]["span_w3"]["hit1"] >= 0.5]
            miss = [r["kv_base_f1"] for r in R if r["src"][f"student@{li_mid}"]["span_w3"]["hit1"] < 0.5]
            out[t]["kv_base_f1"] = {"all": st.mean([r["kv_base_f1"] for r in R]),
                                    f"student@{li_mid}_hit": (st.mean(hit) if hit else None, len(hit)),
                                    f"student@{li_mid}_miss": (st.mean(miss) if miss else None, len(miss))}
            lines.append(f"| {t} | kv_base F1 (UNREGISTERED, diag) | {len(R)} | all {out[t]['kv_base_f1']['all']:.3f}; "
                         f"student@{li_mid} hit≥.5: {out[t]['kv_base_f1'][f'student@{li_mid}_hit']} / miss: "
                         f"{out[t]['kv_base_f1'][f'student@{li_mid}_miss']} |" + " |" * 9)
    out["table"] = "\n".join(lines)
    return out


if __name__ == "__main__":
    sys.exit(main())
