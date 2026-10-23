"""D8 probe (called by `scripts/probe_read.py --read topk`): does question-dependent selection find gold slots?

Per item (own document's TopKLibrary), ONE reader forward on the eval prompt (system prompt + q_prompt) with the
top-k injector loaded:
  selection : the bare reader's queries at the selection layer vs the encoder's attention keys. For every
              (query ∈ last | all_max | all_mean) × (head pool ∈ max | mean) and k ∈ --topk_ks: gold-hit (top-k
              contains ≥1 gold slot), chance of that (hypergeometric), #gold selected, best gold rank fraction;
              plus a shuffled-question control (another question of the same document scores the slots).
  read      : per injection layer, the read attention's mass on gold slots after selection (trained and fresh
              interface), its chance level under a uniform read over the selected set, entropy, top-1.
Gold slots = xenc.gold_slots.item_gold_slot_mask (answer strings + the correct MCQ choice text).
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import xenc  # noqa: E402
from xenc import DEPTHS  # noqa: E402

VARIANTS = [(q, p) for q in ("last", "all_max", "all_mean") for p in ("max", "mean")]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def probe_topk(a) -> int:
    from register_doc import DocRuntime, _get, load_split_items, system_prompt
    from xenc.backbone import build_chat, load_reader, short_name
    from xenc.doc_tasks import load_docs, q_prompt
    from xenc.gold_slots import answer_spans, item_gold_slot_mask, spans_to_slot_mask
    from xenc.interface import load_interface, make_interface
    from xenc.topk_read import build_topk

    extra = {}
    if a.interface != "fresh":
        extra = torch.load(a.interface, map_location="cpu", weights_only=False).get("extra") or {}
    targs = extra.get("args") or {}
    depths = extra.get("depths") or DEPTHS
    encoder = targs.get("encoder") or a.encoder or "same"
    encoder_same = encoder == "same"
    enc_short = short_name(a.reader) if encoder_same else short_name(encoder)
    cache_dir = a.cache_dir or targs.get("cache_dir") or xenc.DEFAULT_CACHE
    chunk_tokens = int(extra.get("chunk_tokens") or targs.get("chunk_tokens") or 512)
    center_path = a.center or extra.get("center")
    style = extra.get("style", "answer")
    ks = [int(x) for x in str(a.topk_ks).split(",") if x.strip()]

    docs = load_docs(a.split_dir)
    items = load_split_items(a.split_dir, a.split)
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
    if a.q_split != "all":
        import hashlib
        want = 0 if a.q_split == "train" else 1
        items = [i for i in items if int(hashlib.sha1(str(_get(i, "key")).encode()).hexdigest(), 16) % 2 == want]
    if a.max_docs:
        keep = set(sorted({_get(i, "doc_id") for i in items})[: a.max_docs])
        items = [i for i in items if _get(i, "doc_id") in keep]
    random.Random(a.seed).shuffle(items)

    model, tok = load_reader(a.reader, device_map=a.device, grad_ckpt=False)
    rt = DocRuntime(model, tok, a.reader, docs, cache_dir, enc_short, chunk_tokens, depths, encoder_same=encoder_same)
    if center_path:
        rt.center = {float(k): v for k, v in torch.load(center_path, map_location="cpu", weights_only=True)["center"].items()}
    items = [i for i in items if rt.has_cache(_get(i, "doc_id"))][: a.n]
    d_enc = rt.d_encoder([_get(i, "doc_id") for i in items])
    dev = next(model.parameters()).device
    itfs = []
    if a.interface != "fresh":
        itf, _, _ = load_interface(a.interface, rt.d_reader, d_enc, rt.layers)
        itfs.append(("trained", itf.to(dev)))
    itfs.append(("fresh", make_interface(rt.d_reader, d_enc, rt.layers).to(dev)))

    class _H:
        read_temp = float(targs.get("read_temp", 12.0))
        route_temp = float(targs.get("route_temp", 10.0))
        top_k = 0
        linear_read = bool(targs.get("linear_read", False))
        topk = int(a.topk)
        select_layer_rho = float(a.select_layer_rho)
        sel_harvest, sel_pool, sel_query, sel_kmap = a.sel_harvest, a.sel_pool, a.sel_query, a.sel_kmap

    sysp = system_prompt(style)

    def hit_at(sc, g, k):
        top = sc.topk(min(k, int(torch.isfinite(sc).sum()))).indices
        return float(bool(g[: sc.numel()][top].any()))
    sel_rows, read_rows = [], []
    by_doc_scores: dict = {}            # doc -> [(key, {variant: pooled scores cpu})]
    gold_of: dict = {}
    for which, itf in itfs:
        inj = build_topk(model, rt, _H, itf)
        inj.record_attn = True
        inj.tk.keep_heads = which == itfs[0][0]
        print(f"[{which}] selection layer {inj.tk.sel_layer}, k={inj.tk.k}, {_H.sel_harvest} keys", flush=True)
        for n_, it in enumerate(items):
            did, key = _get(it, "doc_id"), _get(it, "key")
            lib = rt.library(did)
            if lib is None:
                continue
            gm = item_gold_slot_mask(it, docs[did], tok, chunk_tokens, lib)
            inj.load_library(lib)
            inj.last_attn = {}
            ids = build_chat(tok, a.reader, sysp, q_prompt(it))
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=1)
            tk = inj.tk
            rec = {"key": key, "doc_id": did, "interface": which, "n_gold": int(gm.sum()) if gm is not None else 0,
                   "n_valid": tk.n_valid, "layers": {}}
            if tk.keep_heads:
                pooled = {}
                for (qv, pv) in VARIANTS:
                    pooled[f"{qv}/{pv}"] = tk.pool_scores(tk.last_head_scores, qv, pv)[0].float().cpu()
                by_doc_scores.setdefault(did, []).append((key, pooled))
                if gm is not None:
                    gold_of[key] = gm.reshape(-1)
                    srec = {"key": key, "doc_id": did, "n_gold": int(gm.sum()), "n_valid": tk.n_valid}
                    saved = tk.last_scores
                    for name, sc in pooled.items():
                        tk.last_scores = sc.unsqueeze(0)
                        srec[name] = {str(k): v for k, v in tk.gold_hit(gm.reshape(-1), ks=ks).items()}
                    tk.last_scores = saved
                    # distractor control: per WRONG choice, does top-k contain a slot of that choice's text?
                    lab = _get(it, "correct_label")
                    dms = []
                    for ch in (_get(it, "choices") or []):
                        if ch.get("label") == lab:
                            continue
                        dm = spans_to_slot_mask(docs[did], answer_spans(docs[did], [ch["text"]]), tok, chunk_tokens, lib)
                        if dm is not None:
                            dms.append(dm.reshape(-1) & ~gm.reshape(-1))
                    if dms:
                        srec["distractor"] = {name: {str(k): _mean([hit_at(sc, dm, k) for dm in dms]) for k in ks}
                                              for name, sc in pooled.items()}
                        srec["n_distractors"] = len(dms)
                    sel_rows.append(srec)
                tk.last_head_scores = None
            if gm is not None:
                gf = gm.reshape(-1)
                sel_gold = int(gf[tk.last_idx[0].cpu()].sum())
                for li, A in inj.last_attn.items():
                    p = A.reshape(-1)
                    p = p / p.sum().clamp_min(1e-9)
                    nz = p[p > 0]
                    rec["layers"][str(li)] = {"gold_mass": float(p[gf[: p.numel()]].sum()),
                                              "gold_chance": sel_gold / max(tk.last_idx.shape[1], 1),
                                              "entropy": float(-(nz * nz.log()).sum()),
                                              "entropy_max": math.log(tk.last_idx.shape[1]),
                                              "top1": float(p.max())}
                read_rows.append(rec)
            inj.unload()
            if (n_ + 1) % 10 == 0:
                print(f"  [{which}] {n_ + 1}/{len(items)}", flush=True)
        inj.detach()

    # shuffled-question control: another question of the same doc scores the slots
    shuf = {}
    for did, lst in by_doc_scores.items():
        for i, (key, _) in enumerate(lst):
            if key not in gold_of or len(lst) < 2:
                continue
            shuf[key] = lst[(i + 1) % len(lst)][1]

    summ = {"selection": {}, "read": {}, "n_items": len(items), "n_gold_items": len(sel_rows),
            "mean_n_gold": _mean([r["n_gold"] for r in sel_rows]), "mean_n_valid": _mean([r["n_valid"] for r in sel_rows])}
    for (qv, pv) in VARIANTS:
        name = f"{qv}/{pv}"
        d = {"best_rank_frac": _mean([r[name].get("best_rank_frac") for r in sel_rows])}
        for k in ks:
            rows = [r[name][str(k)] for r in sel_rows if str(k) in r[name]]
            sk = [hit_at(shuf[r["key"]][name], gold_of[r["key"]], k) for r in sel_rows if r["key"] in shuf]
            d[f"k{k}"] = {"hit": _mean([x["hit"] for x in rows]), "chance_any": _mean([x["chance_any"] for x in rows]),
                          "n_gold_sel": _mean([x["n_gold_sel"] for x in rows]),
                          "chance_n_gold_sel": _mean([x["chance_frac"] for x in rows]),
                          "shuffled_q_hit": _mean(sk), "n_shuffled": len(sk),
                          "distractor_hit": _mean([r["distractor"][name][str(k)] for r in sel_rows if "distractor" in r]),
                          "gold_hit_on_distractor_items": _mean([r[name][str(k)]["hit"] for r in sel_rows
                                                                 if "distractor" in r and str(k) in r[name]]),
                          "n_distractor_items": sum(1 for r in sel_rows if "distractor" in r)}
        summ["selection"][name] = d
    for which, _ in itfs:
        rs = [r for r in read_rows if r["interface"] == which]
        for li in sorted({li for r in rs for li in r["layers"]}, key=int):
            xs = [r["layers"][li] for r in rs if li in r["layers"]]
            summ["read"][f"{which}/L{li}"] = {k: _mean([x[k] for x in xs]) for k in
                                              ("gold_mass", "gold_chance", "entropy", "entropy_max", "top1")}
            summ["read"][f"{which}/L{li}"]["n"] = len(xs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "selection": sel_rows, "read": read_rows, "args": vars(a)}, open(a.out, "w"), indent=1)
    print(f"items {summ['n_items']}, with gold slots {summ['n_gold_items']}, mean gold slots {summ['mean_n_gold']}, "
          f"mean valid slots {summ['mean_n_valid']}")
    f = lambda x: "nan" if x is None else f"{x:.3f}"
    for name, d in summ["selection"].items():
        print(f"select {name:14s} best_rank_frac {f(d['best_rank_frac'])} | " + " | ".join(
            f"k{k}: hit {f(d[f'k{k}']['hit'])} (chance {f(d[f'k{k}']['chance_any'])}, shuffled-q {f(d[f'k{k}']['shuffled_q_hit'])}, "
            f"distractor {f(d[f'k{k}']['distractor_hit'])} vs gold {f(d[f'k{k}']['gold_hit_on_distractor_items'])}) "
            f"#gold {f(d[f'k{k}']['n_gold_sel'])} vs {f(d[f'k{k}']['chance_n_gold_sel'])}" for k in ks))
    for name, d in summ["read"].items():
        print(f"read {name}: gold mass {f(d['gold_mass'])} (uniform-over-selected {f(d['gold_chance'])}) "
              f"H {f(d['entropy'])}/{f(d['entropy_max'])} top1 {f(d['top1'])} n={d['n']}")
    print("->", a.out)
    return 0
