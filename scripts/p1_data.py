"""P1 (data scale): the label-free distillation corpus at TRAIN time (register_d13.py --p1_corpus).

load_p1(): reads <corpus>/docs.json + teacher/shard*.jsonl + the top-k chunks written by scripts/p1_teacher.py.
topk_kl(): KL(text teacher || student) on the answer rows from the stored top-k teacher log-probs plus one "rest"
bucket (the teacher's mass outside its top-k): sum_K p_t (log p_t - log q_s) + p_t(rest) (log p_t(rest) - log q_s(rest)).
With k = V it equals xenc.losses.kl_teacher_student exactly (tests/test_p1.py); with k < V it is the exact KL of the
two distributions coarse-grained to {top-k, rest}.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re

import torch


def norm_md5(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", text.lower()).strip().encode()).hexdigest()


def topk_kl(ids: torch.Tensor, lp: torch.Tensor, rest: torch.Tensor, s_logits: torch.Tensor) -> torch.Tensor:
    """ids/lp [m, k] teacher top-k ids / log-probs, rest [m] teacher log-mass outside top-k, s_logits [m, V]."""
    ls = torch.log_softmax(s_logits.float(), -1)
    ids = ids.to(ls.device).long()
    lpt = lp.to(ls.device).float()
    rt = rest.to(ls.device).float()
    ls_k = ls.gather(-1, ids)
    out_k = torch.ones_like(ls, dtype=torch.bool).scatter(-1, ids, False)
    rs = torch.logsumexp(ls.masked_fill(~out_k, float("-inf")), -1)          # student log-mass outside the teacher top-k
    kl = (lpt.exp() * (lpt - ls_k)).sum(-1)
    pr = rt.exp()
    has_rest = out_k.any(-1)
    kl = kl + torch.where(has_rest & (pr > 0), pr * (rt - rs), torch.zeros_like(kl))
    return kl.mean()


def load_p1(corpus: str, groups: str = "", min_conf: float = 0.0, max_docs: int = 0, seed: int = 0, log=print,
            span_only: bool = False, min_tokens: int = 0):
    """-> (items [dict with doc_id, target, agree, conf, ...], tensors {key: {ids, lp, rest, tgt}}, docs {id: meta})."""
    docs = json.load(open(os.path.join(corpus, "docs.json")))
    gset = {g for g in groups.split(",") if g}
    items, keep_chunks = [], set()
    for fn in sorted(glob.glob(os.path.join(corpus, "teacher", "shard*.jsonl"))):
        for line in open(fn):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not os.path.exists(os.path.join(corpus, "teacher", r["chunk"])):
                continue
            did = r["doc_id"]
            if did not in docs or (gset and docs[did].get("group") not in gset):
                continue
            if min_tokens and int(docs[did].get("n_tokens", 0)) < min_tokens:
                continue
            for it in r["items"]:
                if it["conf"] < min_conf or (span_only and it.get("choices")):
                    continue
                items.append({**it, "doc_id": did, "_chunk": r["chunk"]})
                keep_chunks.add(r["chunk"])
    if max_docs:
        import random
        ds = sorted({i["doc_id"] for i in items})
        random.Random(seed).shuffle(ds)
        ds = set(ds[:max_docs])
        items = [i for i in items if i["doc_id"] in ds]
        keep_chunks = {i["_chunk"] for i in items}
    T = {}
    need = {i["key"] for i in items}
    for c in sorted(keep_chunks):
        d = torch.load(os.path.join(corpus, "teacher", c), map_location="cpu")
        T.update({k: v for k, v in d.items() if k in need})
    items = [i for i in items if i["key"] in T]
    log(f"P1 corpus {corpus}: {len(items)} items / {len({i['doc_id'] for i in items})} docs from {len(keep_chunks)} chunks "
        f"(groups {sorted(gset) or 'all'}, min_conf {min_conf}, span_only {span_only}, min_tokens {min_tokens}); agree {sum(i['agree'] for i in items)}")
    return items, T, docs


def weighted_subset(items: list, weights: dict, top_frac: float = 1.0) -> tuple[list, list]:
    """P1-hard: keep items that have a weight (the base's KL to the text teacher, scripts/p1_score.py), then the
    top `top_frac` of them by weight -> (items, weights) for sampling proportional to weight."""
    w = [(float(weights[i["key"]]), i) for i in items if i["key"] in weights and float(weights[i["key"]]) > 0]
    w.sort(key=lambda x: -x[0])
    w = w[: max(1, int(round(len(w) * top_frac)))] if w else []
    return [i for _, i in w], [x for x, _ in w]
