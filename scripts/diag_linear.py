"""D6 probe 3 — linear probes on harvested (centered) document slots. No model forward needed.

(a) document identity: 64-token windows of a document's centered slots, mean-pooled + L2, labelled by
    document (20 LongHealth patients). Train on the first 70% of each document's windows, test on the
    last 30% (contiguous hold-out). Multinomial logistic regression (torch, full batch) and
    nearest-centroid; also nearest-centroid on RAW (uncentered) slots. Chance = 1/#docs.
(b) slot-level content: ridge maps from a centered slot to the (L2) input embedding of the token AT
    that slot (`cur`) and of the NEXT token (`next`), fit on slots of documents outside the held-out
    fold. Evaluated on held-out-fold documents: full-vocabulary top-1/top-10 token identity on random
    slots, and a 5-way MCQ readout — at the first token of each gold-answer occurrence, does the decoded
    embedding pick the gold choice's first token among the 5 choices' first tokens? (chance 0.2), with
    the same readout at distractor-choice occurrences as a control (picks THAT distractor).

    python scripts/diag_linear.py --encoder Qwen/Qwen3.5-2B --center outputs/centers/center_Qwen3.5-2B.pt \
        --out outputs/D6_content/linear_Qwen3.5-2B.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from diag_common import (doc_cache, doc_offsets, doc_states, first_token_positions, l2, load_center,  # noqa: E402
                         load_embeddings, load_items)
import xenc  # noqa: E402
from xenc import DEPTHS  # noqa: E402


def logreg(Xtr, ytr, Xte, yte, n_cls, steps=300, wd=1e-3, lr=0.05):
    W = torch.zeros(Xtr.shape[1], n_cls, device=Xtr.device, requires_grad=True)
    b = torch.zeros(n_cls, device=Xtr.device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    s = 10.0                                    # inputs are unit vectors; scale logits
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(s * Xtr @ W + b, ytr) + wd * (W ** 2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(((s * Xte @ W + b).argmax(-1) == yte).float().mean())


def centroid(Xtr, ytr, Xte, yte, n_cls):
    C = torch.stack([l2(Xtr[ytr == c].mean(0)) for c in range(n_cls)])
    return float(((Xte @ C.T).argmax(-1) == yte).float().mean())


def ridge(X, Y, lam):
    d = X.shape[1]
    A = X.T @ X + lam * torch.eye(d, device=X.device)
    return torch.linalg.solve(A, X.T @ Y)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--center", default=None)
    ap.add_argument("--split_dir", default="data/longhealth_split")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--cache_dir", default=xenc.DEFAULT_CACHE)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--n_ridge", type=int, default=40000, help="training slots for the token probes")
    ap.add_argument("--max_docs", type=int, default=0, help="plumbing: first N docs only")
    ap.add_argument("--skip_tokens", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    from xenc.backbone import short_name
    from xenc.doc_tasks import load_docs

    dev = a.device if torch.cuda.is_available() else "cpu"
    depths = [float(r) for r in DEPTHS]
    enc_short = short_name(a.encoder)
    docs = load_docs(a.split_dir)
    doc_ids = sorted(docs)[: a.max_docs] if a.max_docs else sorted(docs)
    items = load_items(a.split_dir, "test", a.fold)
    fold_docs = sorted({i["doc_id"] for i in items} & set(doc_ids))
    tok = AutoTokenizer.from_pretrained(a.encoder)
    center = load_center(a.center)
    rng = random.Random(a.seed)

    res = {"encoder": a.encoder, "center": a.center, "n_docs": len(doc_ids), "doc_label": {}, "token": {}}
    # ------------------------------------------------------------------ (a) document identity
    feats = {rho: {"tr": [], "te": [], "rtr": [], "rte": []} for rho in depths}
    labels = {"tr": [], "te": []}
    per_doc_states = {}
    for di, did in enumerate(doc_ids):
        c = doc_cache(docs, did, enc_short, a.cache_dir, depths)
        st, _, _ = doc_states(c, center, depths)
        raw, _, _ = doc_states(c, None, depths)
        T = next(iter(st.values())).shape[0]
        nw = T // a.window
        cut = int(0.7 * nw)
        for rho in depths:
            w = st[rho][: nw * a.window].reshape(nw, a.window, -1).mean(1)
            wr = raw[rho][: nw * a.window].reshape(nw, a.window, -1).mean(1)
            feats[rho]["tr"].append(l2(w[:cut])); feats[rho]["te"].append(l2(w[cut:]))
            feats[rho]["rtr"].append(l2(wr[:cut])); feats[rho]["rte"].append(l2(wr[cut:]))
        labels["tr"] += [di] * cut
        labels["te"] += [di] * (nw - cut)
        if not a.skip_tokens:
            per_doc_states[did] = {rho: st[rho].half() for rho in depths}
        del raw
    ytr = torch.tensor(labels["tr"], device=dev)
    yte = torch.tensor(labels["te"], device=dev)
    for rho in depths:
        f = {k: torch.cat(v).to(dev) for k, v in feats[rho].items()}
        res["doc_label"][str(rho)] = {
            "logreg_centered": logreg(f["tr"], ytr, f["te"], yte, len(doc_ids)),
            "centroid_centered": centroid(f["tr"], ytr, f["te"], yte, len(doc_ids)),
            "centroid_raw": centroid(f["rtr"], ytr, f["rte"], yte, len(doc_ids)),
            "n_train": len(labels["tr"]), "n_test": len(labels["te"]), "chance": 1.0 / len(doc_ids)}
        print(f"[doc-label] rho={rho}: " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                                   for k, v in res["doc_label"][str(rho)].items()), flush=True)
    del feats
    if a.skip_tokens:
        json.dump(res, open(a.out, "w"), indent=1)
        return 0

    # ------------------------------------------------------------------ (b) token identity / MCQ readout
    E = l2(load_embeddings(a.encoder)).to(dev)                     # [V, d_e]
    ids = {did: torch.tensor(tok(docs[did], add_special_tokens=False)["input_ids"]) for did in doc_ids}
    train_docs = [d for d in doc_ids if d not in fold_docs]
    pool = [(d, t) for d in train_docs for t in range(len(ids[d]) - 1)]
    rng.shuffle(pool)
    pool = pool[: a.n_ridge]
    nval = len(pool) // 10
    for rho in depths:
        X = torch.stack([per_doc_states[d][rho][t] for d, t in pool]).float().to(dev)
        out = {}
        for tgt, off in (("cur", 0), ("next", 1)):
            Y = E[torch.tensor([int(ids[d][t + off]) for d, t in pool], device=dev)]
            best = None
            for lam in (1e-1, 1.0, 10.0, 100.0):
                Wm = ridge(X[nval:], Y[nval:], lam)
                cos = float((l2(X[:nval] @ Wm) * Y[:nval]).sum(-1).mean())
                if best is None or cos > best[0]:
                    best = (cos, lam)
            Wm = ridge(X, Y, best[1])
            # full-vocab identity on random held-out-fold slots
            test = [(d, t) for d in fold_docs for t in range(len(ids[d]) - 1)]
            rng.shuffle(test)
            test = test[:2000]
            Xt = torch.stack([per_doc_states[d][rho][t] for d, t in test]).float().to(dev)
            gt = torch.tensor([int(ids[d][t + off]) for d, t in test], device=dev)
            sc = l2(Xt @ Wm) @ E.T
            top10 = sc.topk(10, -1).indices
            r = {"lambda": best[1], "val_cos": best[0], "vocab_top1": float((top10[:, 0] == gt).float().mean()),
                 "vocab_top10": float((top10 == gt[:, None]).any(-1).float().mean())}
            # 5-way MCQ readout at answer occurrences
            gold_hits, dis_hits = [], []
            for it in items:
                did = it["doc_id"]
                if did not in fold_docs or not it.get("choices"):
                    continue
                offs = doc_offsets(tok, docs[did])
                T = len(ids[did])
                cand = []
                for ch in it["choices"]:
                    vs = {tok(" " + ch["text"].strip(), add_special_tokens=False)["input_ids"][0],
                          tok(ch["text"].strip(), add_special_tokens=False)["input_ids"][0]}
                    cand.append(sorted(vs))
                for ci, ch in enumerate(it["choices"]):
                    pos = first_token_positions(docs[did], ch["text"], offs, T)
                    pos = [p - off for p in pos if 0 <= p - off < T]   # next: read the slot BEFORE the occurrence
                    if not pos:
                        continue
                    pv = l2(per_doc_states[did][rho][torch.tensor(pos)].float().to(dev) @ Wm)   # [P, d_e]
                    s5 = torch.stack([(pv @ E[torch.tensor(c, device=dev)].T).max(-1).values for c in cand], -1)  # [P,5]
                    hit = (s5.argmax(-1) == ci).float().mean().item()
                    (gold_hits if ch["label"] == it.get("correct_label") else dis_hits).append(hit)
            r["mcq_gold_acc"] = sum(gold_hits) / len(gold_hits) if gold_hits else None
            r["mcq_gold_n"] = len(gold_hits)
            r["mcq_distractor_acc"] = sum(dis_hits) / len(dis_hits) if dis_hits else None
            r["mcq_distractor_n"] = len(dis_hits)
            r["mcq_chance"] = 0.2
            out[tgt] = r
            print(f"[token] rho={rho} {tgt}: " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                                          for k, v in r.items()), flush=True)
            del Y
        res["token"][str(rho)] = out
        del X
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
