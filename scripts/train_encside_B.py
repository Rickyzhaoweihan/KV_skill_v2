"""D12 stage 3: train the encoder-side write-back B end to end through the frozen reader with the paper's
registration loss, initialised from a closed-form (ridge) B.

Student = reader on its own 64-token document tail + question + gold answer (teacher-forced), with the replaced
attention outputs y_ℓ = B_ℓ(encoder read-outs) at the paired layers (xenc/encside_ctx.TrainCtxEncSide).
Teacher = the same reader with the whole document in context (text condition).
    loss = kl·KL(p_T ‖ p_S)[answer rows] + ce·CE(gold answer) + reg·mean_ℓ ‖B_ℓ − B0_ℓ‖²/‖B0_ℓ‖²
Only B (low-rank delta or full), a per-layer bias and per-layer log-gain train; reader and encoder are frozen.

Two stages (both run by default):
  cache: one pass with reader + encoder: for every train / val item store the encoder read-outs (all encoder
         full-attention layers, post o_proj) at the student rows (+2 rows of causal history), the student token
         ids, and the teacher's top-k answer-row log-probs. Val items also store the wrong-document variant.
  train: reader only (no encoder in memory). Val (held-out DOCUMENTS) every --eval_every steps: teacher-forced
         answer accuracy (all answer tokens argmax-correct == greedy-correct), KL, CE for kv / wrong_doc /
         closed_tail; select on mean kv accuracy over val tasks. Exports the best B as an encside_B2 blob.

Train docs: QuALITY/MuSiQue/LastingBench train split, LongHealth train items of folds --lh_train_folds (never
fold 0). Val: QuALITY val split (other documents) + LongHealth fold --lh_val_fold.
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
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402


# ============================================================================ items
def select_items(a):
    """-> (train [(task, item)], val [(task, item, doc_order)])."""
    from xenc.doc_tasks import load_docs
    docs, train, val = {}, [], []
    rng = random.Random(a.seed)
    for spec in a.train_splits.split(","):
        name, nd = spec.split(":")
        sd = f"data/{name}_split"
        docs.update(load_docs(sd))
        items = load_split_items(sd, "train")
        if name == "longhealth":
            folds = {int(x) for x in a.lh_train_folds.split(",")}
            assert 0 not in folds, "LongHealth fold 0 is the judged test fold"
            items = [i for i in items if (_get(i, "meta") or {}).get("fold") in folds]
        by_doc = {}
        for it in items:
            by_doc.setdefault(_get(it, "doc_id"), []).append(it)
        dids = sorted(by_doc)[: int(nd)]
        sel = [it for d in dids for it in by_doc[d][: a.max_q_per_doc]]
        train += [(name, it) for it in sel]
        log(f"  train {name}: {len(dids)} docs, {len(sel)} items")
    for spec in a.val_splits.split(","):
        name, n = spec.split(":")
        sd = f"data/{name}_split"
        docs.update(load_docs(sd))
        items = load_split_items(sd, "val")
        if name == "longhealth":
            items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.lh_val_fold]
        order = sorted({_get(i, "doc_id") for i in items})
        tr_docs = {_get(it, "doc_id") for _, it in train}
        items = [i for i in items if _get(i, "doc_id") not in tr_docs]
        sel = random.Random(a.seed + 1).sample(items, min(int(n), len(items)))
        val += [(name, it, order) for it in sel]
        log(f"  val {name}: {len(sel)} items from {len({_get(i, 'doc_id') for i in sel})} held-out docs")
    rng.shuffle(train)
    return docs, train, val


# ============================================================================ cache
@torch.no_grad()
def cache_one(a, reader, enc, tok, cap, docs, it, did):
    from eval_encside import full_ids
    from xenc.backbone import input_device, text_ids
    from xenc.doc_tasks import gold_text
    pre, suf = full_ids(tok, a.reader, docs, it, did)
    pre = pre[-(a.max_tokens - len(suf)):]
    tgt = text_ids(tok, gold_text(it))
    ids = pre + suf + tgt
    rows = a.tail + len(suf) + len(tgt)
    cap.start()
    enc(input_ids=torch.tensor([ids], device=input_device(enc)), use_cache=False, logits_to_keep=1)
    o = cap.get()
    feats = {e: v[0, -(rows + 2):].half().cpu() for e, v in o.items()}
    return ids, rows, tgt, feats


@torch.no_grad()
def teacher_topk(reader, ids, m, k):
    from xenc.backbone import input_device
    lo = reader(input_ids=torch.tensor([ids], device=input_device(reader)), use_cache=False,
                logits_to_keep=m + 1).logits[0, -(m + 1):].float()
    lp = F.log_softmax(lo, -1)
    v, i = lp.topk(k, dim=-1)
    return v.half().cpu(), i.int().cpu()


def run_cache(a):
    from eval_doc import wrong_doc_id
    from xenc.backbone import load_encoder, load_reader
    from xenc.encside_read import EncCapture
    from xenc.xattn_read import attn_dims
    docs, train, val = select_items(a)
    reader, tok = load_reader(a.reader, device_map=a.device)
    enc = load_encoder(a.encoder, device_map=a.device)[0]
    cap = EncCapture(enc, attn_dims(enc)["full"], "post")
    os.makedirs(a.cache_dir, exist_ok=True)
    t0 = time.time()

    def item_rec(task, it, did):
        ids, rows, tgt, feats = cache_one(a, reader, enc, tok, cap, docs, it, did)
        return {"task": task, "key": _get(it, "key"), "doc_id": did, "sids": ids[-rows:], "m": len(tgt),
                "feats": feats}, ids

    shard, k = [], 0
    for n, (task, it) in enumerate(train):
        did = _get(it, "doc_id")
        rec, ids = item_rec(task, it, did)
        if len(ids) > a.max_teacher_tokens:
            continue
        rec["t_lp"], rec["t_idx"] = teacher_topk(reader, ids, rec["m"], a.topk)
        shard.append(rec)
        if len(shard) == a.shard_size:
            torch.save(shard, os.path.join(a.cache_dir, f"train_{k:03d}.pt"))
            shard, k = [], k + 1
        if (n + 1) % 100 == 0:
            log(f"  cache train {n + 1}/{len(train)} ({(time.time() - t0) / (n + 1):.2f}s/item)")
    if shard:
        torch.save(shard, os.path.join(a.cache_dir, f"train_{k:03d}.pt"))
    vrecs = []
    for task, it, order in val:
        did = _get(it, "doc_id")
        rec, ids = item_rec(task, it, did)
        rec["t_lp"], rec["t_idx"] = teacher_topk(reader, ids, rec["m"], a.topk)
        w = wrong_doc_id(did, order)
        wr, _ = item_rec(task, it, w)
        rec["wrong"] = {"sids": wr["sids"], "feats": wr["feats"], "doc_id": w}
        vrecs.append(rec)
    torch.save(vrecs, os.path.join(a.cache_dir, "val.pt"))
    log(f"cache done: {len(train)} train items, {len(vrecs)} val items in {time.time() - t0:.0f}s -> {a.cache_dir}")
    cap.detach()
    del reader, enc
    torch.cuda.empty_cache()


# ============================================================================ train
def student_logits(model, R, itf, rec, sids, feats, m, dev, grad: bool, inject: bool = True):
    if inject:
        f = itf.build({e: v[None].to(dev) for e, v in feats.items()}, reset=True)
        rows = len(sids)
        R.cur = {li: x[:, -rows:] for li, x in f.items()}
    else:
        R.cur = None
    with torch.set_grad_enabled(grad):
        lo = model(input_ids=torch.tensor([sids], device=dev), use_cache=False, logits_to_keep=m + 1).logits[0]
    R.cur = None
    return lo


def losses(lo, rec, sids, m):
    s = F.log_softmax(lo[:-1].float(), -1)
    tl = rec["t_lp"][:-1].to(lo.device).float()
    ti = rec["t_idx"][:-1].to(lo.device).long()
    pt = tl.exp()
    kl = (pt * (tl - s.gather(-1, ti))).sum(-1).mean()
    tgt = torch.tensor(sids[-m:], device=lo.device)
    ce = F.nll_loss(s, tgt)
    acc = float((s.argmax(-1) == tgt).all())
    return kl, ce, acc


@torch.no_grad()
def run_val(model, R, itf, vrecs, dev):
    out = {}
    for rec in vrecs:
        t = out.setdefault(rec["task"], {k: [] for k in ("kv", "wrong", "tail", "kl_kv", "ce_kv")})
        m = rec["m"]
        lo = student_logits(model, R, itf, rec, rec["sids"], rec["feats"], m, dev, False)
        kl, ce, acc = losses(lo, rec, rec["sids"], m)
        t["kv"].append(acc)
        t["kl_kv"].append(float(kl))
        t["ce_kv"].append(float(ce))
        w = rec["wrong"]
        lo = student_logits(model, R, itf, rec, w["sids"], w["feats"], m, dev, False)
        t["wrong"].append(losses(lo, rec, w["sids"], m)[2])
        lo = student_logits(model, R, itf, rec, rec["sids"], None, m, dev, False, inject=False)
        t["tail"].append(losses(lo, rec, rec["sids"], m)[2])
    res = {task: {k: sum(v) / len(v) for k, v in d.items()} | {"n": len(d["kv"])} for task, d in out.items()}
    res["mean_kv"] = sum(r["kv"] for r in res.values() if isinstance(r, dict)) / len(out)
    res["mean_gap"] = sum(r["kv"] - r["wrong"] for k, r in res.items() if isinstance(r, dict)) / len(out)
    return res


def run_train(a):
    from xenc.backbone import input_device, load_reader
    from xenc.encside_ctx import TrainCtxEncSide, from_tokenwise
    from xenc.encside_read import EncSideReader
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    train = [r for p in sorted(glob.glob(os.path.join(a.cache_dir, "train_*.pt")))
             for r in torch.load(p, weights_only=False)]
    vrecs = torch.load(os.path.join(a.cache_dir, "val.pt"), weights_only=False)
    log(f"train items {len(train)} ({len({r['doc_id'] for r in train})} docs) | val items {len(vrecs)}")
    model, tok = load_reader(a.reader, device_map=a.device)
    dev = input_device(model)
    blob = {}
    for k, mp in enumerate(a.init.split(",")):
        b = torch.load(mp, map_location="cpu", weights_only=False)
        if b.get("format") != "xenc.encside_B2.v1":
            b = from_tokenwise(b)
        blob = b if k == 0 else {**blob, "layers": {**b["layers"], **blob["layers"]}}
    layers = [int(x) for x in a.layers.split(",")]
    itf = TrainCtxEncSide(blob, layers, rank=a.rank, seed=a.seed).to(dev)
    R = EncSideReader(model, itf)
    groups = itf.param_groups(a.lr, a.lr_gain)
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(a.warmup, 1)) * (0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
                                                             * (1 - a.min_lr_frac) + a.min_lr_frac))
    log(f"init {a.init} | layers {layers} | rank {a.rank} | trainable {sum(p.numel() for p in params)} | "
        f"kl {a.kl} ce {a.ce} reg {a.reg} lr {a.lr}")
    hist = {"steps": [], "evals": []}
    best, best_path = -1.0, os.path.join(a.out, "B_best.pt")

    def evaluate(step):
        nonlocal best
        r = run_val(model, R, itf, vrecs, dev)
        r["step"] = step
        with torch.no_grad():
            r["reg"] = float(itf.reg())
        r["gain"] = [round(float(x), 4) for x in itf.lg.exp()]
        hist["evals"].append(r)
        tag = ""
        if r["mean_kv"] > best:
            best = r["mean_kv"]
            torch.save({**itf.export(), "d12_train": {"step": step, "val": r, "args": vars(a)}}, best_path)
            tag = " *best*"
        log(f"  [val @ {step}] " + " | ".join(
            f"{t}: kv {d['kv']:.3f} wrong {d['wrong']:.3f} tail {d['tail']:.3f} KL {d['kl_kv']:.3f} CE {d['ce_kv']:.3f}"
            for t, d in r.items() if isinstance(d, dict)) + f" | mean kv {r['mean_kv']:.3f} gap {r['mean_gap']:+.3f}"
            f" | drift {r['reg']:.4f}{tag}")
        with open(os.path.join(a.out, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)

    evaluate(0)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        acc = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "acc": 0.0}
        for _ in range(a.batch):
            rec = train[rng.randrange(len(train))]
            lo = student_logits(model, R, itf, rec, rec["sids"], rec["feats"], rec["m"], dev, True)
            kl, ce, ac = losses(lo, rec, rec["sids"], rec["m"])
            loss = a.kl * kl + a.ce * ce
            (loss / a.batch).backward()
            for k_, v in (("loss", loss), ("kl", kl), ("ce", ce)):
                acc[k_] += float(v.detach()) / a.batch
            acc["acc"] += ac / a.batch
        rg = itf.reg()
        (a.reg * rg).backward()
        gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step()
        sched.step()
        rec_ = {**acc, "reg": float(rg), "grad_norm": float(gn), "step": step}
        hist["steps"].append(rec_)
        if not math.isfinite(rec_["loss"]):
            log(f"ERROR: non-finite loss at step {step}")
            break
        if step % 25 == 0 or step <= 3:
            w = hist["steps"][-25:]
            log(f"  [step {step}/{a.steps}] loss {sum(x['loss'] for x in w) / len(w):.4f} "
                f"kl {sum(x['kl'] for x in w) / len(w):.4f} ce {sum(x['ce'] for x in w) / len(w):.4f} "
                f"acc {sum(x['acc'] for x in w) / len(w):.3f} drift {rec_['reg']:.4f} |g| {rec_['grad_norm']:.3f} "
                f"({(time.time() - t0) / step:.2f}s/step)")
        if step % a.eval_every == 0 or step == a.steps:
            evaluate(step)
    torch.save({**itf.export(), "d12_train": {"step": a.steps, "args": vars(a)}}, os.path.join(a.out, "B_last.pt"))
    log(f"done; best val mean kv {best:.4f} -> {best_path}")
    R.detach()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", default="cache,train")
    ap.add_argument("--reader", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--train_splits", default="quality:120,musique:60,lastingbench:60,longhealth:12")
    ap.add_argument("--lh_train_folds", default="2,3,4")
    ap.add_argument("--max_q_per_doc", type=int, default=20)
    ap.add_argument("--val_splits", default="quality:48,longhealth:48")
    ap.add_argument("--lh_val_fold", type=int, default=1)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--max_teacher_tokens", type=int, default=16000)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--shard_size", type=int, default=250)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--init", default="", help="closed-form B (encside_B2 or D10 token-wise blob; comma list merges)")
    ap.add_argument("--layers", default="3,7,11,15,19,23,27,31")
    ap.add_argument("--rank", type=int, default=128, help="low-rank delta on B; 0 = train the full matrix")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr_gain", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--ce", type=float, default=0.5)
    ap.add_argument("--reg", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    st = a.stage.split(",")
    if "cache" in st:
        run_cache(a)
    if "train" in st:
        if not a.out or not a.init:
            ap.error("--out and --init are required for the train stage")
        run_train(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
