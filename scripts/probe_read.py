"""Read-attention probe: does the registered interface put slot-attention mass on the operator slots
that contain the gold answer text?  For each item: build the item's Library (and the wrong doc's),
run ONE student forward on q_prompt with attention recording, and report, per injection layer:
entropy of the (route x slot) attention at the last prompt position, mass on gold slots, and the
chance level (fraction of gold slots).  Also runs the same with a FRESH (untrained) interface.

    python scripts/probe_read.py --reader Qwen/Qwen3.5-9B --split_dir data/longhealth_split --split test \
        --interface outputs/<run>/interface_best.pt --n 40 --out outputs/<run>/probe_read.json [--fold 0]
"""
from __future__ import annotations
import argparse, json, math, os, random, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import xenc  # noqa: F401
import torch

from xenc import DEPTHS


from xenc.gold_slots import gold_slot_mask  # noqa: E402,F401  (moved to xenc/gold_slots.py by D2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True); ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test"); ap.add_argument("--interface", required=True)
    ap.add_argument("--n", type=int, default=40); ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--q_split", choices=["all", "train", "heldout"], default="all")
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--cache_dir", default=None); ap.add_argument("--center", default=None)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    # D7 native-attention read: --read native_attn [--interface fresh --encoder same|hub --depths .. --kv_map ..]
    ap.add_argument("--read", choices=["library", "native_attn", "topk"], default="library")
    from xenc.topk_read import add_topk_args   # D8: --read topk (selection gold-hit + read gold mass; scripts/probe_topk.py)
    add_topk_args(ap)
    ap.add_argument("--topk_ks", default="32,64,128", help="D8: k values for the selection gold-hit table")
    ap.add_argument("--encoder", default="same", help="D7 fresh native interface: encoder hub or 'same'")
    ap.add_argument("--depths", default=None, help="D7 fresh native interface: cache depths")
    ap.add_argument("--native_layers", default=None)
    ap.add_argument("--kv_map", default=None, help="D7 fresh S/L native interface: fit_kv_map.py output")
    ap.add_argument("--null_init", type=float, default=-30.0)
    ap.add_argument("--rope", default="none,doc", help="D7: comma list of native read modes (none = position-free, doc = RoPE-consistent)")
    ap.add_argument("--teacher_ceiling", type=int, default=1, choices=[0, 1], help="D7: also measure the in-context teacher attention")
    ap.add_argument("--sel_split", default="test"); ap.add_argument("--sel_n", type=int, default=20,
                    help="D7: items (outside --fold) used to pick the gold-attending heads from the teacher")
    ap.add_argument("--sel_k", type=int, default=2, help="D7: heads selected per layer")
    a = ap.parse_args(argv)
    if a.read == "native_attn":
        return probe_native(a)
    if a.read == "topk":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from probe_topk import probe_topk
        return probe_topk(a)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from register_doc import DocRuntime, _get, load_split_items  # noqa: E402
    from xenc.backbone import load_reader, build_chat, short_name
    from xenc.doc_tasks import load_docs, q_prompt
    from xenc.injector import LibraryInjector
    from xenc.interface import load_interface, make_interface
    from xenc.library import Library

    saved = torch.load(a.interface, map_location="cpu", weights_only=False)
    extra = saved.get("extra") or {}; targs = extra.get("args") or {}
    depths = extra.get("depths") or DEPTHS
    encoder = targs.get("encoder") or "same"; encoder_same = encoder == "same"
    enc_short = short_name(a.reader) if encoder_same else short_name(encoder)
    cache_dir = a.cache_dir or targs.get("cache_dir") or xenc.DEFAULT_CACHE
    chunk_tokens = int(extra.get("chunk_tokens") or targs.get("chunk_tokens") or 512)
    center_path = a.center or extra.get("center")

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
    random.Random(a.seed).shuffle(items); items = items[: a.n]

    model, tok = load_reader(a.reader, device_map=a.device, grad_ckpt=False)
    rt = DocRuntime(model, tok, a.reader, docs, cache_dir, enc_short, chunk_tokens, depths, encoder_same=encoder_same)
    rt.harvest = extra.get("harvest") or targs.get("harvest") or "resid"
    if center_path:
        rt.center = {float(k): v for k, v in torch.load(center_path, map_location="cpu", weights_only=True)["center"].items()}
    d_enc = rt.d_encoder([_get(i, "doc_id") for i in items])
    itf, selector, _ = load_interface(a.interface, rt.d_reader, d_enc, rt.layers)
    fresh = make_interface(rt.d_reader, d_enc, rt.layers)
    dev = next(model.parameters()).device
    itf.to(dev); fresh.to(dev)
    inj_kw = dict(read_softmax=bool(targs.get("read_softmax", True)), read_temp=float(targs.get("read_temp", 12.0)),
                  route_temp=float(targs.get("route_temp", 10.0)), top_k=int(targs.get("top_k", 0)))
    doc_order = sorted({_get(i, "doc_id") for i in load_split_items(a.split_dir, a.split)})
    results = []
    for which, interface in (("trained", itf), ("fresh", fresh)):
        inj = LibraryInjector(model, interface, **inj_kw); inj.record_attn = True
        for it in items:
            did = _get(it, "doc_id"); lib = rt.library(did)
            if lib is None:
                continue
            wrong_id = doc_order[(doc_order.index(did) + 7) % len(doc_order)]
            lib_w = rt.library(wrong_id)
            rec = {"key": _get(it, "key"), "doc_id": did, "interface": which, "layers": {}}
            for tag, L in (("own", lib), ("wrong", lib_w)):
                if L is None:
                    continue
                inj.load_library(L.to(dev)); inj.last_attn = {}
                ids = build_chat(tok, a.reader, None, q_prompt(it))
                with torch.no_grad():
                    model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=1)
                gm = gold_slot_mask(docs[did], _get(it, "answers"), tok, chunk_tokens, L) if tag == "own" else None
                for li, A in inj.last_attn.items():
                    p = A.flatten(); p = p / p.sum().clamp_min(1e-9)
                    H = float(-(p * (p + 1e-12).log()).sum()); Hmax = math.log(max(1, int((p > 0).sum())))
                    d = rec["layers"].setdefault(str(li), {})
                    d[f"{tag}_entropy"] = H; d[f"{tag}_entropy_max"] = Hmax; d[f"{tag}_top1"] = float(p.max())
                    d[f"{tag}_route_top1"] = float(A.sum(-1).max() / A.sum().clamp_min(1e-9))
                    if gm is not None:
                        gmf = gm.flatten()[: p.numel()]
                        d["gold_mass"] = float(p[gmf].sum()); d["gold_chance"] = float(gmf.float().mean())
                        d["gold_n_slots"] = int(gmf.sum())
                inj.unload()
            results.append(rec)
        inj.detach()
    # summary
    summ = {}
    for which in ("trained", "fresh"):
        rs = [r for r in results if r["interface"] == which]
        for li in sorted({li for r in rs for li in r["layers"]}, key=int):
            rows = [r["layers"][li] for r in rs if li in r["layers"]]
            g = [x for x in rows if "gold_mass" in x]
            summ[f"{which}/L{li}"] = {
                "n": len(rows), "own_entropy": sum(x["own_entropy"] for x in rows) / len(rows),
                "own_entropy_max": sum(x["own_entropy_max"] for x in rows) / len(rows),
                "own_top1": sum(x["own_top1"] for x in rows) / len(rows),
                "own_route_top1": sum(x["own_route_top1"] for x in rows) / len(rows),
                "gold_mass": (sum(x["gold_mass"] for x in g) / len(g)) if g else None,
                "gold_chance": (sum(x["gold_chance"] for x in g) / len(g)) if g else None,
                "gold_n": len(g),
            }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "records": results, "args": vars(a)}, open(a.out, "w"), indent=1)
    for k, v in summ.items():
        print(f"{k}: n={v['n']} entropy {v['own_entropy']:.2f}/{v['own_entropy_max']:.2f} top1 {v['own_top1']:.3f} route_top1 {v['own_route_top1']:.3f} "
              f"gold_mass {v['gold_mass'] if v['gold_mass'] is None else round(v['gold_mass'],4)} chance {v['gold_chance'] if v['gold_chance'] is None else round(v['gold_chance'],4)} (n_gold={v['gold_n']})")
    print("->", a.out)
    return 0


def _gold_token_flags(tok, text: str, spans) -> torch.Tensor:
    """bool [T] over tok(text) tokens overlapping any char span (offset mapping)."""
    from xenc.gold_slots import _token_offsets
    offs = _token_offsets(tok, text)
    f = torch.zeros(len(offs), dtype=torch.bool)
    for s, e in spans:
        for i, (x, y) in enumerate(offs):
            if y > s and x < e:
                f[i] = True
    return f


def _head_stats(P: torch.Tensor, gold: torch.Tensor) -> dict:
    """P [H, S] distributions over S doc slots (rows renormalized), gold bool [S] -> per-head + head-mean stats."""
    P = P / P.sum(-1, keepdim=True).clamp_min(1e-12)
    gm = P[:, gold].sum(-1)                                           # [H]
    top10 = P.topk(min(10, P.shape[1]), -1).indices
    hit = gold[top10].any(-1).float()                                 # [H]
    Pm = P.mean(0)
    H = float(-(Pm * (Pm + 1e-12).log()).sum())
    return {"gold_mass_heads": gm.tolist(), "top10_heads": hit.tolist(), "gold_mass": float(Pm[gold].sum()),
            "top10": float(bool(gold[Pm.topk(min(10, Pm.numel())).indices].any())), "entropy": H,
            "top1": float(Pm.max()), "chance": float(gold.float().mean())}


def probe_native(a) -> int:
    """D7: (i) TEACHER ceiling: real in-context attention of the last prompt token (document in the prompt) over
    the document tokens at each full-attention layer; (ii) the native-attention read (reader's own queries over
    the encoder's raw K/V, fresh or registered interface) for each --rope mode (none = position-free, doc =
    RoPE-consistent: slot i at position i, query at N_doc + p). Per layer: gold mass (renormalized over the
    document), top-10 gold hit, entropy, per-head gold mass; plus the read restricted to the heads selected on
    a DISJOINT selection set by teacher gold mass (--sel_n items of --sel_split, other folds)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from register_doc import DocRuntime, _get, load_split_items, system_prompt  # noqa: E402
    from xenc.backbone import build_chat, load_reader, short_name
    from xenc.doc_tasks import ctx_prompt, load_docs, q_prompt
    from xenc.gold_slots import answer_spans
    from xenc.native_attn import (NativeAttnInjector, NativeAttnInterface, NativeSpec, attn_geometry,
                                  load_native_interface, native_layers_arg, teacher_attention)
    from deltaskill.backbone_utils import chat_template_kwargs, decoder_layers
    if a.interface != "fresh":
        itf, extra = load_native_interface(a.interface)
        targs = extra.get("args") or {}
        depths = extra.get("depths") or DEPTHS
        encoder = targs.get("encoder") or "same"
        layers = native_layers_arg(targs.get("native_layers"))
        chunk_tokens = int(extra.get("chunk_tokens") or 512)
    else:
        itf, encoder = None, a.encoder
        depths = sorted(float(x) for x in a.depths.split(",")) if a.depths else DEPTHS
        layers = native_layers_arg(a.native_layers)
        chunk_tokens = 512
    encoder_same = encoder == "same"
    enc_short = short_name(a.reader) if encoder_same else short_name(encoder)
    docs = load_docs(a.split_dir)
    model, tok = load_reader(a.reader, device_map=a.device, grad_ckpt=False)
    rt = DocRuntime(model, tok, a.reader, docs, a.cache_dir or xenc.DEFAULT_CACHE, enc_short, chunk_tokens, depths,
                    encoder_same=encoder_same)
    rt.harvest = "attn_raw"
    rt.native = NativeSpec(model, depths, layers)
    L = rt.native.layers

    def pick(split, fold, exclude_fold, n, seed):
        its = load_split_items(a.split_dir, split)
        if fold is not None:
            its = [i for i in its if (_get(i, "meta") or {}).get("fold") == fold]
        if exclude_fold is not None:
            its = [i for i in its if (_get(i, "meta") or {}).get("fold") != exclude_fold]
        if a.max_docs:
            keep = set(sorted({_get(i, "doc_id") for i in its})[: a.max_docs])
            its = [i for i in its if _get(i, "doc_id") in keep]
        its = [i for i in its if rt.has_cache(_get(i, "doc_id")) and answer_spans(docs[_get(i, "doc_id")], _get(i, "answers"))]
        random.Random(seed).shuffle(its)
        return its[:n]

    items = pick(a.split, a.fold, None, a.n, a.seed)
    sel_items = pick(a.sel_split, None, a.fold, a.sel_n, a.seed + 1) if a.sel_n else []
    sysp = system_prompt("answer")
    hub = a.reader
    dev = next(model.parameters()).device

    def teacher(it):
        did = _get(it, "doc_id")
        doc = docs[did]
        msgs = [{"role": "system", "content": sysp}, {"role": "user", "content": ctx_prompt(it, doc)}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **chat_template_kwargs(hub))
        ids = tok(text, add_special_tokens=False)["input_ids"]
        d0 = text.find(doc)
        doc_flag = _gold_token_flags(tok, text, [(d0, d0 + len(doc))])
        gold = _gold_token_flags(tok, text, [(s + d0, e + d0) for s, e in answer_spans(doc, _get(it, "answers"))])
        A = teacher_attention(model, torch.tensor([ids], device=dev), L)
        out = {}
        for li in L:
            P = A[li][:, : doc_flag.numel()]
            st = _head_stats(P[:, doc_flag], gold[doc_flag])
            st["doc_mass"] = float(P[:, doc_flag].sum(-1).mean())
            out[li] = st
        return out

    # ---- teacher on the selection set -> fixed head choice per layer
    head_sel = {}
    if sel_items:
        acc = {li: [] for li in L}
        for it in sel_items:
            for li, st in teacher(it).items():
                acc[li].append(torch.tensor(st["gold_mass_heads"]) / max(st["chance"], 1e-9))
        for li in L:
            ratio = torch.stack(acc[li]).mean(0)                     # mean gold mass / chance per head
            head_sel[li] = [int(h) for h in ratio.topk(a.sel_k).indices]
        print("head selection (top-%d teacher gold/chance on %d disjoint items): %s" % (a.sel_k, len(sel_items), head_sel), flush=True)

    # ---- interface / injector
    d_enc = rt.d_encoder([_get(i, "doc_id") for i in items])
    if itf is None:
        geo = attn_geometry(decoder_layers(model)[L[0]].self_attn)
        d_dst = geo["n_kv"] * geo["hd"]
        itf = NativeAttnInterface(L, d_enc, d_dst, maps=bool(a.kv_map) or d_enc != d_dst, null_init=a.null_init)
        if a.kv_map:
            itf.load_maps(a.kv_map, rt.native.rho_of)
    itf.to(dev)
    inj = NativeAttnInjector(model, itf)
    inj.record_attn = True
    modes = [m.strip() for m in a.rope.split(",") if m.strip()]
    rows = []
    for k_, it in enumerate(items):
        did = _get(it, "doc_id")
        rec = {"key": _get(it, "key"), "doc_id": did}
        if a.teacher_ceiling:
            rec["teacher"] = {str(li): v for li, v in teacher(it).items()}
        lib = rt.library(did)
        gm = gold_slot_mask(docs[did], _get(it, "answers"), tok, chunk_tokens, lib)
        ids = build_chat(tok, hub, sysp, q_prompt(it))
        for mode in modes:
            inj.rope_mode = mode
            inj.load_library(lib)
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=1)
            r = {}
            for li in L:
                idx = inj.last_idx[li]
                gold = gm.flatten()[idx]
                st = _head_stats(inj.last_attn_heads[li], gold)
                st["null"] = inj.last_null[li]
                if li in head_sel:
                    sub = _head_stats(inj.last_attn_heads[li][head_sel[li]], gold)
                    st["sel_gold_mass"], st["sel_top10_any"] = sub["gold_mass"], float(max(sub["top10_heads"]))
                    if a.teacher_ceiling:
                        tg = torch.tensor(rec["teacher"][str(li)]["gold_mass_heads"])[head_sel[li]]
                        rec["teacher"][str(li)]["sel_gold_mass"] = float(tg.mean())
                r[str(li)] = st
            rec[f"read_{mode}"] = r
            inj.unload()
        rows.append(rec)
        if (k_ + 1) % 10 == 0:
            print(f"  [{k_ + 1}/{len(items)}]", flush=True)
    inj.detach()

    # ---- summary
    def agg(key, li):
        rs = [r[key][str(li)] for r in rows if key in r]
        if not rs:
            return None
        o = {}
        for f in ("gold_mass", "top10", "entropy", "top1", "chance", "doc_mass", "sel_gold_mass", "sel_top10_any", "null"):
            v = [x[f] for x in rs if f in x]
            if v:
                o[f] = sum(v) / len(v)
        gh = torch.tensor([x["gold_mass_heads"] for x in rs]).mean(0)
        th = torch.tensor([x["top10_heads"] for x in rs]).mean(0)
        o["best_head"] = int(gh.argmax()); o["best_head_gold_mass"] = float(gh.max())
        o["best_head_top10"] = float(th.max()); o["gold_mass_heads"] = [round(float(x), 4) for x in gh]
        return o

    keys = (["teacher"] if a.teacher_ceiling else []) + [f"read_{m}" for m in modes]
    summ = {f"L{li}": {k: agg(k, li) for k in keys} for li in L}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "head_sel": {str(k): v for k, v in head_sel.items()}, "n": len(rows),
               "records": rows, "args": vars(a)}, open(a.out, "w"), indent=1)
    f = lambda x: "  -  " if x is None else f"{x:.4f}"
    print(f"n={len(rows)} items with a gold answer span; per layer: gold mass (doc-renorm, head-mean) | top10 | best head "
          f"(mass, top10) | selected heads mass | entropy | chance")
    for li in L:
        for k in keys:
            o = summ[f"L{li}"][k]
            print(f"L{li:<3d} {k:10s} gold {f(o['gold_mass'])} top10 {f(o['top10'])} best h{o['best_head']} "
                  f"{f(o['best_head_gold_mass'])}/{f(o['best_head_top10'])} sel {f(o.get('sel_gold_mass'))} "
                  f"H {f(o['entropy'])} chance {f(o['chance'])}" + (f" docmass {f(o.get('doc_mass'))}" if k == "teacher" else ""))
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
