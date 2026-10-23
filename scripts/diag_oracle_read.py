"""D6 probe 2 — oracle-addressing upper bound: can the registered read branch USE the operator's content
when addressing is forced?  LibraryInjector.force_attn replaces the computed (route x slot) attention by a
given [C,N] distribution at every position; everything else (B, gate, gains) is the registered interface.

Conditions per held-out item (items whose gold answer string occurs in the document):
  closed_book    no library
  kv             own Library, learned attention                     (the judge's kv)
  wrong_doc      wrong doc's Library, learned attention             (the judge's wrong_doc)
  oracle_chunk   own Library restricted to the gold chunk(s), learned attention
  gold           forced uniform attention over the gold-answer slots
  gold_x4        same, branch gain x4 (gain_scale)
  gold_match     forced gold read rescaled per token to the learned read's norm
  random         forced uniform over the same number of random valid slots
  random_x4      same, gain x4
  distractor     forced uniform over the slots of one WRONG choice string present in the document
Metrics: generation score (greedy, stop on </answer>), P(gold label) and P(distractor label) from the
next-token distribution after "<answer>" (5-way renormalized), mean ||delta||/||h|| on the prompt.

Stage 2 (--register_steps N > 0): warm-start from the interface and register N steps with attention FORCED
onto the gold slots during training (train split, items outside --fold with a gold string; loss =
kl·KL(teacher‖student) + ce·CE + mag·||Δ||/||h||, as register_doc). A receives no gradient (forced
attention), so this trains B/gate/gains "under correct addressing"; then re-evaluate on the held-out fold
(and on --n_train_eval training items).

    python scripts/diag_oracle_read.py --reader Qwen/Qwen3.5-9B --split_dir data/longhealth_split --fold 0 \
        --interface outputs/p1_longhealth_9b_LL_f0_center/interface_best.pt --out_dir outputs/D6_oracle \
        --register_steps 100
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

from diag_common import choice_masks, doc_offsets, gold_mask, load_items, wrong_doc_id  # noqa: E402
import xenc  # noqa: E402
from xenc import DEPTHS  # noqa: E402

ALL = ("closed_book", "kv", "wrong_doc", "oracle_chunk", "gold", "gold_x4", "gold_match", "random", "random_x4",
       "distractor")


def to_cn(mask_t: torch.Tensor, lib) -> torch.Tensor:
    """[T] token mask -> [C, N_max] slot mask of the Library (chunk_meta start / n_tokens)."""
    C, N = next(iter(lib.mask.values())).shape
    M = torch.zeros(C, N, dtype=torch.bool)
    for c, meta in enumerate(lib.chunk_meta):
        st, n = int(meta["start"]), int(meta["n_tokens"])
        seg = mask_t[st: st + n]
        M[c, : seg.shape[0]] = seg
    return M


def uniform(M: torch.Tensor) -> torch.Tensor:
    return M.float() / M.float().sum().clamp_min(1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--interface", required=True)
    ap.add_argument("--split_dir", default="data/longhealth_split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="0 = all held-out items with a gold string")
    ap.add_argument("--conds", default=",".join(ALL))
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--center", default=None, help="default: the interface's registration center")
    ap.add_argument("--register_steps", type=int, default=0)
    ap.add_argument("--reg_lr", type=float, default=2e-4)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--ce", type=float, default=0.5)
    ap.add_argument("--mag", type=float, default=0.3)
    ap.add_argument("--n_train_eval", type=int, default=40)
    ap.add_argument("--max_teacher_tokens", type=int, default=32000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true", help="CPU plumbing on the TinyLM fixture (register_doc.tiny_setup)")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    from register_doc import DocRuntime, build_injector, system_prompt, tiny_setup
    from xenc.backbone import (build_chat, eval_mode, generate, input_device, load_reader, short_name, text_ids,
                               train_mode, with_target)
    from xenc.doc_tasks import ctx_prompt, extract_label, gold_text, load_docs, q_prompt, score
    from xenc.interface import load_interface, save_interface
    from xenc.losses import answer_ce, kl_teacher_student, magnitude_penalty, student_logits, teacher_logits

    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    conds = [c for c in ALL if c in a.conds.split(",")]
    blob = torch.load(a.interface, map_location="cpu", weights_only=False)
    extra = blob.get("extra") or {}
    targs = extra.get("args") or {}
    depths = extra.get("depths") or DEPTHS
    encoder = targs.get("encoder") or "same"
    encoder_same = encoder == "same"
    chunk_tokens = int(extra.get("chunk_tokens") or targs.get("chunk_tokens") or 512)

    class _H:
        read_temp = targs.get("read_temp", 12.0)
        route_temp = targs.get("route_temp", 10.0)
        top_k = targs.get("top_k", 0)
        linear_read = bool(targs.get("linear_read", False))

    if a.tiny:
        torch.set_num_threads(1)
        tz = tiny_setup(a.out_dir, encoder_same=encoder_same, depths=depths)
        model, tok, hub, docs = tz["reader"], tz["tok"], tz["hub"], tz["docs"]
        cache_dir, enc_short, chunk_tokens = tz["cache_dir"], tz["enc_short"], tz["chunk_tokens"]
        test = [i.to_dict() if hasattr(i, "to_dict") else i for i in tz["test"]]
        train = list(test)
        a.budget = min(a.budget, 4)
    else:
        docs = load_docs(a.split_dir)
        test = load_items(a.split_dir, a.split, a.fold)
        train = [i for i in load_items(a.split_dir, "train", None) if (i.get("meta") or {}).get("fold") != a.fold]
        cache_dir = a.cache_dir or targs.get("cache_dir") or xenc.DEFAULT_CACHE
        enc_short = short_name(a.reader) if encoder_same else short_name(encoder)
        hub = a.reader
        model, tok = load_reader(a.reader, device_map=a.device, grad_ckpt=a.register_steps > 0)
    rt = DocRuntime(model, tok, hub, docs, cache_dir, enc_short, chunk_tokens, depths, encoder_same=encoder_same)
    center_path = a.center or (None if a.tiny else extra.get("center"))
    if center_path:
        rt.center = {float(k): v for k, v in torch.load(center_path, map_location="cpu", weights_only=True)["center"].items()}
    dev = input_device(model)
    doc_order = sorted(docs)
    d_enc = rt.d_encoder(doc_order)
    itf, selector, _ = load_interface(a.interface, rt.d_reader, d_enc, rt.layers)
    itf = itf.to(dev)
    inj = build_injector(model, itf, _H, None)
    rt.injector = inj
    sysp = system_prompt(extra.get("style", "answer"))

    # ---------------------------------------------------------------- per-item slot masks
    offs_cache: dict = {}

    def masks(it, lib):
        did = it["doc_id"]
        if a.tiny:   # whitespace tokenizer: token i == word i
            words = docs[did].split()
            T = len(words)
            gt = torch.tensor([w.lower() in {str(x).lower() for x in it.get("answers") or []} for w in words])
            cm = {c["label"]: torch.tensor([w == c["text"] for w in words]) for c in it.get("choices") or []}
        else:
            if did not in offs_cache:
                offs_cache[did] = doc_offsets(tok, docs[did])
            offs = offs_cache[did]
            T = sum(int(m["n_tokens"]) for m in lib.chunk_meta)
            assert len(offs) == T, (did, len(offs), T)
            gt = gold_mask(it, docs[did], offs, T)
            cm = choice_masks(it, docs[did], offs, T)
        G = to_cn(gt, lib)
        valid = next(iter(lib.mask.values())).cpu()
        g = torch.Generator().manual_seed(zlib.crc32(it["key"].encode()))
        idx = valid.flatten().nonzero().flatten()
        pick = idx[torch.randperm(idx.numel(), generator=g)[: max(1, int(G.sum()))]]
        R = torch.zeros(valid.numel(), dtype=torch.bool)
        R[pick] = True
        D, dlab = None, None
        wrong = [k for k, m in cm.items() if k != it.get("correct_label") and bool(m.any())]
        if wrong:
            dlab = wrong[zlib.crc32(it["key"].encode()) % len(wrong)]
            D = to_cn(cm[dlab], lib)
        return G, R.view_as(valid), D, dlab

    # label tokens after "<answer>"
    # "<answer>X" may merge ">X" into one token: use the common prefix of the 5 encodings, each of which
    # must then end in exactly one label-specific token.
    pre, lab_tok = [], {}
    if not a.tiny:
        enc = {L: text_ids(tok, "<answer>" + L) for L in "ABCDE"}
        n = 0
        while all(len(v) > n for v in enc.values()) and len({v[n] for v in enc.values()}) == 1:
            n += 1
        pre = enc["A"][:n]
        for L, v in enc.items():
            assert len(v) == n + 1, ("label tokenization", L, v, pre)
            lab_tok[L] = v[-1]
        assert len(set(lab_tok.values())) == 5, lab_tok

    def label_probs(ids):
        if a.tiny:
            return {}
        x = torch.tensor([ids + pre], device=dev)
        with torch.no_grad():
            lg = model(input_ids=x, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        p = torch.softmax(lg[[lab_tok[L] for L in "ABCDE"]], -1)
        return {L: float(p[i]) for i, L in enumerate("ABCDE")}

    def run_items(items, conds, tag):
        rows = []
        t0 = time.time()
        for k, it in enumerate(items):
            did = it["doc_id"]
            lib = rt.library(did)
            if lib is None:
                continue
            G, R, D, dlab = masks(it, lib)
            if not bool(G.any()):
                continue
            rec = {"key": it["key"], "doc_id": did, "n_gold_slots": int(G.sum()), "dis_label": dlab}
            ids = build_chat(tok, hub, sysp, q_prompt(it))
            for c in conds:
                L, fa, gs, nm = None, None, 1.0, False
                if c == "kv":
                    L = lib
                elif c == "wrong_doc":
                    L = rt.library(wrong_doc_id(did, doc_order))
                elif c == "oracle_chunk":
                    gch = sorted(set(G.any(-1).nonzero().flatten().tolist()))
                    L = lib.subset(gch)
                elif c.startswith("gold"):
                    L, fa = lib, uniform(G)
                    gs = 4.0 if c.endswith("x4") else 1.0
                    nm = c == "gold_match"
                elif c.startswith("random"):
                    L, fa = lib, uniform(R)
                    gs = 4.0 if c.endswith("x4") else 1.0
                elif c == "distractor":
                    if D is None:
                        continue
                    L, fa = lib, uniform(D)
                inj.force_attn, inj.force_norm_match, inj.gain_scale = fa, nm, gs
                if L is not None:
                    inj.load_library(L)
                try:
                    inj.mag_terms.clear()
                    p = label_probs(ids)
                    mag = float(torch.stack([m.float().cpu() for m in inj.mag_terms]).mean()) if inj.mag_terms else 0.0
                    g, ng = generate(model, tok, hub, ids, a.budget, return_n=True)
                finally:
                    inj.unload()
                    inj.force_attn, inj.force_norm_match, inj.gain_scale = None, False, 1.0
                valid = "".join(ch["label"] for ch in it.get("choices") or []) or "ABCDE"
                lab = extract_label(g, valid) if it.get("choices") else None
                rec[c] = {"score": float(score(it, g)), "gen": g, "gen_tokens": ng, "label": lab, "mag": mag,
                          "p_gold": p.get(it.get("correct_label")), "p_dis": p.get(dlab) if dlab else None,
                          "picked_dis": float(lab == dlab) if dlab else None}
            rows.append(rec)
            if (k + 1) % 10 == 0:
                print(f"  [{tag} {len(rows)}] " + " ".join(f"{c}={_m(rows, c, 'score'):.3f}" for c in conds)
                      + f" | {(time.time() - t0) / len(rows):.1f}s/item", flush=True)
        return rows

    def summarize(rows, conds):
        s = {"n_items": len(rows)}
        for c in conds:
            s[c] = {f: _m(rows, c, f) for f in ("score", "p_gold", "p_dis", "picked_dis", "mag", "gen_tokens")}
            s[c]["n"] = sum(1 for r in rows if c in r)
        for x, y in (("gold", "random"), ("gold", "closed_book"), ("gold", "kv"), ("kv", "wrong_doc"),
                     ("gold_x4", "random_x4"), ("gold_match", "kv"), ("oracle_chunk", "kv"), ("kv", "closed_book")):
            if x in conds and y in conds:
                for f in ("score", "p_gold"):
                    d = [r[x][f] - r[y][f] for r in rows if x in r and y in r and r[x][f] is not None and r[y][f] is not None]
                    if d:
                        mu = sum(d) / len(d)
                        sd = math.sqrt(sum((v - mu) ** 2 for v in d) / max(1, len(d) - 1))
                        s[f"{x}-{y}/{f}"] = {"mean": mu, "se": sd / math.sqrt(len(d)), "n": len(d)}
        return s

    def show(title, s, conds):
        print(f"==== {title}: n_items={s['n_items']}")
        for c in conds:
            v = s[c]
            print(f"  {c:13s} n={v['n']:3d} score={_f(v['score'])} p_gold={_f(v['p_gold'])} p_dis={_f(v['p_dis'])} "
                  f"picked_dis={_f(v['picked_dis'])} mag={_f(v['mag'])} gen={_f(v['gen_tokens'])}")
        for k, v in s.items():
            if "/" in k:
                print(f"  {k:28s} {v['mean']:+.4f} ± {v['se']:.4f} (n={v['n']})")

    rng = random.Random(a.seed)
    items = list(test)
    if a.n:
        items = items[: a.n]
    rows = run_items(items, conds, "registered")
    s1 = summarize(rows, conds)
    show("stage 1: registered interface (" + a.interface + ")", s1, conds)
    result = {"args": vars(a), "stage1": {"summary": s1, "items": rows}}
    json.dump(result, open(os.path.join(a.out_dir, "oracle_read.json"), "w"), indent=1)

    if a.register_steps > 0:
        pool = []
        for it in train:
            lib = rt.library(it["doc_id"])
            if lib is not None and bool(masks(it, lib)[0].any()):
                pool.append(it)
        print(f"stage 2: forced-gold registration, {a.register_steps} steps, pool {len(pool)} train items", flush=True)
        params = [p for p in itf.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.reg_lr, weight_decay=1e-4)
        hist = []
        t0 = time.time()
        for step in range(1, a.register_steps + 1):
            it = pool[rng.randrange(len(pool))]
            did = it["doc_id"]
            target = gold_text(it)
            tids, m, _ = with_target(tok, hub, sysp, ctx_prompt(it, docs[did]), target, device=dev)
            if tids.shape[1] > a.max_teacher_tokens:
                continue
            lib = rt.library(did)
            G = masks(it, lib)[0]
            inj.unload()
            eval_mode(model)
            t_log = teacher_logits(model, tids, m)
            sids, m2, _ = with_target(tok, hub, sysp, q_prompt(it), target, device=dev)
            inj.load_library(lib)
            inj.force_attn = uniform(G)
            inj.mag_terms.clear()
            inj.route_terms.clear()
            train_mode(model)
            opt.zero_grad(set_to_none=True)
            s_log = student_logits(model, sids, m)
            kl = kl_teacher_student(t_log[:-1], s_log[:-1])
            ce = answer_ce(s_log, sids[0, -m:])
            mag = magnitude_penalty(inj).to(kl.device)
            loss = a.kl * kl + a.ce * ce + a.mag * mag
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            inj.unload()
            inj.force_attn = None
            inj.mag_terms.clear()
            inj.route_terms.clear()
            eval_mode(model)
            hist.append({"step": step, "loss": float(loss.detach()), "kl": float(kl.detach()), "ce": float(ce.detach()), "mag": float(mag.detach()),
                         "grad_norm": float(gn)})
            if step % 10 == 0 or step <= 3:
                print(f"  [forced-gold step {step}] loss {float(loss):.4f} kl {float(kl):.4f} ce {float(ce):.4f} "
                      f"mag {float(mag):.4f} |g| {float(gn):.3f} ({(time.time() - t0) / step:.2f}s/step)", flush=True)
        save_interface(itf, os.path.join(a.out_dir, "interface_forcedgold.pt"), None,
                       extra={**extra, "d6_forced_gold_steps": a.register_steps, "d6_init": a.interface})
        conds2 = [c for c in conds if c not in ("gold_x4", "random_x4")]
        rows2 = run_items(items, conds2, "forced-gold-registered")
        s2 = summarize(rows2, conds2)
        show("stage 2: forced-gold registered interface, held-out fold", s2, conds2)
        tr = [i for i in pool]
        random.Random(a.seed + 1).shuffle(tr)
        rows3 = run_items(tr[: a.n_train_eval], conds2, "train-items")
        s3 = summarize(rows3, conds2)
        show("stage 2: forced-gold registered interface, TRAINING items (fit check)", s3, conds2)
        result["stage2"] = {"history": hist, "heldout": {"summary": s2, "items": rows2},
                            "train_items": {"summary": s3, "items": rows3}}
        json.dump(result, open(os.path.join(a.out_dir, "oracle_read.json"), "w"), indent=1)
    if torch.cuda.is_available():
        print(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print("->", os.path.join(a.out_dir, "oracle_read.json"))
    return 0


def _m(rows, c, f):
    v = [r[c][f] for r in rows if c in r and r[c].get(f) is not None]
    return sum(v) / len(v) if v else None


def _f(x):
    return "  n/a " if x is None else f"{x:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
