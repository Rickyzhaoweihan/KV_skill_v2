"""D6 probe 1 — content presence / native-space addressability of a harvested document operator.

For held-out LongHealth items (fold K) whose gold answer string occurs in the document, build the
question's query IN THE ENCODER'S OWN SPACE: run the ENCODER on q_prompt(item) alone (raw text, same
tokenization as the document prefill), take the residual at each depth rho (last token = `last`; mean
over the question-text tokens = `qmean`), L2 it, center it with the encoder's center file exactly like
the slots (L2(x - mu)), and score cosine against the document's centered slots.

Reported per (depth, query variant), averaged over items:
  gold best-rank (1-based, and as a fraction of T), top-1 / top-10 / top-100 gold hit vs analytic chance
  ret_mcq     : argmax over present choices of max-slot cosine == gold choice (vs chance 1/#present)
  ans         : POSITIVE CONTROL query = encoder states of the gold answer text alone (mean over its tokens)
  shuf_*      : same gold hits with ANOTHER item's question (same doc) — question-specificity control
  doc_disc    : mean top-10 cosine over own doc > over the wrong doc (eval_doc rule: sorted order +7)
  chunk_top1  : summary key (L2 mean of centered chunk slots) retrieval, gold chunk top-1 vs 1/C
  chunk_own   : top-1 chunk over own+wrong pooled chunks is from the own doc

    python scripts/diag_content.py --encoder Qwen/Qwen3.5-2B --center outputs/centers/center_Qwen3.5-2B.pt \
        --split_dir data/longhealth_split --fold 0 --out outputs/D6_content/content_Qwen3.5-2B.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from diag_common import (center_rows, choice_masks, chunk_summaries, doc_cache, doc_offsets, doc_states,  # noqa: E402
                         gold_mask, load_center, load_items, topk_chance, wrong_doc_id)
import xenc  # noqa: E402
from xenc import DEPTHS  # noqa: E402


def best_rank(sims: torch.Tensor, mask: torch.Tensor) -> int:
    """1-based rank of the highest-scoring masked slot."""
    top = sims[mask].max()
    return int((sims > top).sum()) + 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--center", default=None)
    ap.add_argument("--split_dir", default="data/longhealth_split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cache_dir", default=xenc.DEFAULT_CACHE)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_items", type=int, default=0, help="plumbing runs: first N fold items only")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from xenc.backbone import load_encoder, short_name
    from xenc.depth_skill import derive_by_depth
    from xenc.doc_tasks import load_docs, q_prompt

    depths = [float(r) for r in DEPTHS]
    enc_short = short_name(a.encoder)
    docs = load_docs(a.split_dir)
    items = load_items(a.split_dir, a.split, a.fold)
    if a.max_items:
        items = items[: a.max_items]
    doc_order = sorted(docs)
    center = load_center(a.center)
    dev = a.device if torch.cuda.is_available() else "cpu"
    if center:
        center = {k: v.to(dev) for k, v in center.items()}
    model, tok = load_encoder(a.encoder, device_map=dev)

    def query_states(it) -> dict:
        """{variant: {rho: [d] centered}} from the encoder run on q_prompt alone."""
        txt = q_prompt(it)
        d = derive_by_depth(model, tok, txt, depths, chunk_tokens=10 ** 9, doc_id="q", encoder=enc_short)
        nq = len(tok(str(it["question"]).strip(), add_special_tokens=False)["input_ids"])
        out = {"last": {}, "qmean": {}, "ans": {}}
        gold = [c["text"] for c in it.get("choices") or [] if c["label"] == it.get("correct_label")] or list(it.get("answers") or [""])
        da = derive_by_depth(model, tok, " " + str(gold[0]).strip(), depths, chunk_tokens=10 ** 9, doc_id="a", encoder=enc_short)
        for rho in depths:
            h = d["chunks"][0]["K"][rho].to(dev).float()          # [Tq,d] L2 rows
            out["last"][rho] = center_rows(h[-1], center, rho)
            out["qmean"][rho] = center_rows(h[: max(1, nq)].mean(0), center, rho)
            # POSITIVE CONTROL (answer-aware, not a legitimate query): encoder run on the gold answer text alone
            out["ans"][rho] = center_rows(da["chunks"][0]["K"][rho].to(dev).float().mean(0), center, rho)
        return out

    dcache: dict = {}

    def get_doc(did):
        if did not in dcache:
            c = doc_cache(docs, did, enc_short, a.cache_dir, depths)
            if c is None:
                raise SystemExit(f"no {enc_short} cache for {did}")
            st, starts, cid = doc_states(c, center, depths, device=dev)
            T = next(iter(st.values())).shape[0]
            offs = doc_offsets(tok, docs[did])
            assert len(offs) == T, (did, len(offs), T)
            dcache[did] = {"st": st, "cid": cid.to(dev), "S": chunk_summaries(st, cid.to(dev)), "offs": offs, "T": T}
        return dcache[did]

    # queries for all fold items (used for the shuffled-question control too)
    qs = {it["key"]: query_states(it) for it in items}
    by_doc: dict = {}
    for it in items:
        by_doc.setdefault(it["doc_id"], []).append(it)

    recs = []
    for it in items:
        did = it["doc_id"]
        D = get_doc(did)
        T = D["T"]
        g = gold_mask(it, docs[did], D["offs"], T).to(dev)
        if not bool(g.any()):
            continue
        cms = {k: m.to(dev) for k, m in choice_masks(it, docs[did], D["offs"], T).items()}
        present = [k for k, m in cms.items() if bool(m.any())]
        wid = wrong_doc_id(did, doc_order)
        W = get_doc(wid)
        sib = by_doc[did]
        other = sib[(sib.index(it) + 1) % len(sib)]               # another question of the SAME doc
        gchunks = sorted(set(D["cid"][g].tolist()))
        ng = int(g.sum())
        rec = {"key": it["key"], "doc_id": did, "wrong": wid, "T": T, "n_gold": ng, "n_present": len(present),
               "gold_chunks": gchunks, "C": int(D["cid"].max()) + 1, "m": {}}
        for var in ("last", "qmean", "ans"):
            for rho in depths:
                q = qs[it["key"]][var][rho]
                s = D["st"][rho] @ q                               # [T]
                r = best_rank(s, g)
                sw = W["st"][rho] @ q
                qo = qs[other["key"]][var][rho]
                so = D["st"][rho] @ qo
                ro = best_rank(so, g)
                ret = None
                if it.get("correct_label") in present and len(present) >= 2:
                    best = max(present, key=lambda k: float(s[cms[k]].max()))
                    ret = float(best == it["correct_label"])
                cs = D["S"][rho] @ q                               # [C]
                crank = int((cs > cs[gchunks].max()).sum()) + 1
                pooled_own = float(cs.max() >= (W["S"][rho] @ q).max())
                rec["m"][f"{var}/{rho}"] = {
                    "rank": r, "rank_frac": r / T, "top1": float(r <= 1), "top10": float(r <= 10),
                    "top100": float(r <= 100),
                    "shuf_rank_frac": ro / T, "shuf_top10": float(ro <= 10), "shuf_top100": float(ro <= 100),
                    "ret_mcq": ret, "ret_mcq_chance": (1.0 / len(present)) if ret is not None else None,
                    "doc_disc": float(s.topk(10).values.mean() > sw.topk(10).values.mean()),
                    "own_top10_cos": float(s.topk(10).values.mean()), "wrong_top10_cos": float(sw.topk(10).values.mean()),
                    "gold_mean_cos": float(s[g].mean()), "all_mean_cos": float(s.mean()),
                    "chunk_rank": crank, "chunk_top1": float(crank <= 1), "chunk_own": pooled_own,
                }
        rec["chance"] = {"top1": topk_chance(T, ng, 1), "top10": topk_chance(T, ng, 10),
                         "top100": topk_chance(T, ng, 100), "chunk_top1": len(gchunks) / rec["C"],
                         "rank_frac": 1.0 / (ng + 1)}
        recs.append(rec)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    summ = {"n_items": len(recs), "n_fold_items": len(items), "encoder": a.encoder, "center": a.center,
            "chance": {k: mean([r["chance"][k] for r in recs]) for k in recs[0]["chance"]} if recs else {}}
    keys = list(recs[0]["m"]) if recs else []
    for k in keys:
        rows = [r["m"][k] for r in recs]
        summ[k] = {f: mean([x[f] for x in rows]) for f in rows[0]}
        summ[k]["median_rank_frac"] = sorted(x["rank_frac"] for x in rows)[len(rows) // 2]
        summ[k]["n_ret_mcq"] = sum(1 for x in rows if x["ret_mcq"] is not None)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "records": recs, "args": vars(a)}, open(a.out, "w"), indent=1)
    print(f"== {a.encoder} fold {a.fold}: n={len(recs)}/{len(items)} items with gold string; chance "
          + " ".join(f"{k}={v:.4f}" for k, v in summ["chance"].items()))
    print(f"{'query/rho':12s} {'top1':>6s} {'top10':>6s} {'top100':>6s} {'rankfr':>7s} {'med':>6s} {'shuf10':>6s} "
          f"{'shuf100':>7s} {'shufrf':>6s} {'retMCQ':>6s} {'chMCQ':>6s} {'docdis':>6s} {'chk1':>5s} {'chkown':>6s} {'gold-all cos':>12s}")
    for k in keys:
        v = summ[k]
        print(f"{k:12s} {v['top1']:6.3f} {v['top10']:6.3f} {v['top100']:6.3f} {v['rank_frac']:7.3f} {v['median_rank_frac']:6.3f} "
              f"{v['shuf_top10']:6.3f} {v['shuf_top100']:7.3f} {v['shuf_rank_frac']:6.3f} {v['ret_mcq']:6.3f} {v['ret_mcq_chance']:6.3f} "
              f"{v['doc_disc']:6.3f} {v['chunk_top1']:5.3f} {v['chunk_own']:6.3f} {v['gold_mean_cos'] - v['all_mean_cos']:12.4f}")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
