"""D9 probe: does the registered read reproduce the text-in-context teacher on HELD-OUT items?

For each item: one teacher forward (reader, document in prompt, SYSTEM_ANSWER) recording, per injection layer,
the attention of the last prompt position over the document mapped to operator slots (xenc/distill.py) and
the residual after the layer; one no-operator forward (h^0); then student forwards (question only) with the
OWN document's Library and with a WRONG document's Library. Per layer it reports
  attention : KL(teacher ‖ student), top-20 slot overlap, student mass on the teacher's top-20 slots
              (vs the teacher's own mass there), entropies, gold-slot mass of teacher and student (when a gold
              answer string occurs in the document), chance = gold-slot fraction;
  residual  : gap ratio ‖n(h^S) − n(h^T)‖² / ‖n(h^0) − n(h^T)‖² at the last prompt position for the own and
              the wrong document (own < wrong ⇒ the write carries document-specific content).
Both the trained interface and a FRESH one are probed.

    python scripts/probe_distill.py --reader Qwen/Qwen3.5-9B --split_dir data/quality_split --split test \
        --interface outputs/D9_q_LL/interface_best.pt --n 40 --out outputs/D9_q_LL/probe_distill.json [--fold 0]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))
import torch  # noqa: E402

import xenc  # noqa: E402
from xenc import DEPTHS  # noqa: E402


def _ent(p):
    return float(-(p * p.clamp_min(1e-12).log()).sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--interface", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_teacher_tokens", type=int, default=32000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from register_doc import SYSTEM_ANSWER, DocRuntime, _get, load_split_items
    from xenc.backbone import build_chat, load_reader, short_name
    from xenc.distill import Distiller, topk_overlap
    from xenc.doc_tasks import ctx_prompt, load_docs, q_prompt
    from xenc.gold_slots import item_gold_slot_mask
    from xenc.injector import LibraryInjector
    from xenc.interface import load_interface, make_interface
    from deltaskill.derive import rms_norm

    saved = torch.load(a.interface, map_location="cpu", weights_only=False)
    extra = saved.get("extra") or {}
    targs = extra.get("args") or {}
    depths = extra.get("depths") or DEPTHS
    encoder = targs.get("encoder") or "same"
    encoder_same = encoder == "same"
    enc_short = short_name(a.reader) if encoder_same else short_name(encoder)
    cache_dir = a.cache_dir or targs.get("cache_dir") or xenc.DEFAULT_CACHE
    chunk_tokens = int(extra.get("chunk_tokens") or targs.get("chunk_tokens") or 512)
    center_path = extra.get("center")

    docs = load_docs(a.split_dir)
    all_items = load_split_items(a.split_dir, a.split)
    items = list(all_items)
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
    random.Random(a.seed).shuffle(items)
    model, tok = load_reader(a.reader, device_map=a.device, grad_ckpt=False)
    rt = DocRuntime(model, tok, a.reader, docs, cache_dir, enc_short, chunk_tokens, depths, encoder_same=encoder_same)
    rt.harvest = extra.get("harvest") or targs.get("harvest") or "resid"
    if center_path:
        rt.center = {float(k): v for k, v in torch.load(center_path, map_location="cpu", weights_only=True)["center"].items()}
    items = [i for i in items if rt.has_cache(_get(i, "doc_id"))][: a.n]
    d_enc = rt.d_encoder([_get(i, "doc_id") for i in items])
    itf, _, _ = load_interface(a.interface, rt.d_reader, d_enc, rt.layers)
    fresh = make_interface(rt.d_reader, d_enc, rt.layers)
    dev = next(model.parameters()).device
    itf.to(dev)
    fresh.to(dev)
    inj_kw = dict(read_softmax=not bool(targs.get("linear_read", False)), read_temp=float(targs.get("read_temp", 12.0)),
                  route_temp=float(targs.get("route_temp", 10.0)), top_k=int(targs.get("top_k", 0)))
    doc_order = sorted({_get(i, "doc_id") for i in all_items if rt.has_cache(_get(i, "doc_id"))})
    dist = Distiller(model, tok, a.reader, rt.layers, attn_w=1.0, hid_w=1.0)
    print(f"teacher attention windows {dist.windows}; {len(items)} items", flush=True)

    # ---- teacher pass (once per item, shared by both interfaces)
    teach = {}
    for it in items:
        did = _get(it, "doc_id")
        lib = rt.library(did)
        user = ctx_prompt(it, docs[did])
        tids = build_chat(tok, a.reader, SYSTEM_ANSWER, user)
        if len(tids) > a.max_teacher_tokens:
            continue
        dist.prepare(_get(it, "key"), SYSTEM_ANSWER, user, docs[did], lib, chunk_tokens, len(tids))
        with torch.no_grad(), dist.teacher(len(tids), 0):
            model(input_ids=torch.tensor([tids], device=dev), use_cache=False, logits_to_keep=1)
        sids = torch.tensor([build_chat(tok, a.reader, SYSTEM_ANSWER, q_prompt(it))], device=dev)
        dist.hid_ratio = True
        dist.baseline(sids, 0)
        gm = item_gold_slot_mask(it, docs[did], tok, chunk_tokens, lib)
        teach[_get(it, "key")] = {"t_attn": {k: v.cpu() for k, v in dist.t_attn.items()},
                                  "t_hid": {k: v[-1].cpu() for k, v in dist.t_hid.items()},
                                  "b_hid": {k: v[-1].cpu() for k, v in dist.b_hid.items()},
                                  "gold": gm, "sids": sids.cpu()}
        dist.clear()

    records = []
    for which, interface in (("trained", itf), ("fresh", fresh)):
        inj = LibraryInjector(model, interface, **inj_kw)
        inj.record_attn = True
        for it in items:
            key = _get(it, "key")
            if key not in teach:
                continue
            T = teach[key]
            did = _get(it, "doc_id")
            wid = doc_order[(doc_order.index(did) + 7) % len(doc_order)]
            rec = {"key": key, "doc_id": did, "interface": which, "layers": {}}
            for tag, d_ in (("own", did), ("wrong", wid)):
                L = rt.library(d_)
                if L is None:
                    continue
                inj.load_library(L)
                inj.last_attn = {}
                dist.s_out = {}
                with torch.no_grad(), dist.student():
                    model(input_ids=T["sids"].to(dev), use_cache=False, logits_to_keep=1)
                for li in rt.layers:
                    d = rec["layers"].setdefault(str(li), {})
                    hS = rms_norm(dist.s_out[li][0, -1].float())
                    hT = rms_norm(T["t_hid"][li].to(hS.device).float())
                    h0 = rms_norm(T["b_hid"][li].to(hS.device).float())
                    d[f"{tag}_hid_ratio"] = float((hS - hT).pow(2).sum() / (h0 - hT).pow(2).sum().clamp_min(1e-6))
                    if tag != "own" or li not in inj.last_attn:
                        continue
                    s = inj.last_attn[li].reshape(-1).float()
                    s = s / s.sum().clamp_min(1e-12)
                    t = T["t_attn"].get(li)
                    d["s_entropy"] = _ent(s)
                    d["entropy_max"] = math.log(max(1, int((s > 0).sum())))
                    if t is not None:
                        t = t.float()
                        d["t_entropy"] = _ent(t)
                        d["kl"] = float((t * (t.clamp_min(1e-12).log() - s.clamp_min(1e-12).log())).sum())
                        d["top20_overlap"] = topk_overlap(t, s, 20)
                        top = t.topk(20).indices
                        d["s_mass_t_top20"] = float(s[top].sum())
                        d["t_mass_t_top20"] = float(t[top].sum())
                    g = T["gold"]
                    if g is not None:
                        gf = g.reshape(-1)[: s.numel()]
                        d["gold_mass"] = float(s[gf].sum())
                        d["gold_chance"] = float(gf.float().mean())
                        if t is not None:
                            d["t_gold_mass"] = float(t[gf].sum())
                inj.unload()
            records.append(rec)
        inj.detach()

    summ = {}
    for which in ("trained", "fresh"):
        rs = [r for r in records if r["interface"] == which]
        for li in rt.layers:
            rows = [r["layers"][str(li)] for r in rs if str(li) in r["layers"]]
            keys = sorted({k for x in rows for k in x})
            s = {"n": len(rows)}
            for k in keys:
                v = [x[k] for x in rows if k in x]
                s[k] = sum(v) / len(v)
                if k.startswith("gold") or k == "t_gold_mass":
                    s[k + "_n"] = len(v)
            pr = [(x["own_hid_ratio"], x["wrong_hid_ratio"]) for x in rows if "wrong_hid_ratio" in x]
            s["own_closer_than_wrong"] = sum(o < w for o, w in pr) / max(len(pr), 1)
            summ[f"{which}/L{li}"] = s
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"summary": summ, "records": records, "args": vars(a), "windows": dist.windows}, open(a.out, "w"), indent=1)
    for k, v in summ.items():
        f = lambda x: "-" if x is None else f"{x:.4f}"
        print(f"{k}: n={v['n']} KL {f(v.get('kl'))} top20 {f(v.get('top20_overlap'))} s@t20 {f(v.get('s_mass_t_top20'))}"
              f"/{f(v.get('t_mass_t_top20'))} H_s {f(v.get('s_entropy'))} H_t {f(v.get('t_entropy'))} max {f(v.get('entropy_max'))} "
              f"gold s {f(v.get('gold_mass'))} t {f(v.get('t_gold_mass'))} chance {f(v.get('gold_chance'))} (n={v.get('gold_mass_n', 0)}) "
              f"| hid ratio own {f(v.get('own_hid_ratio'))} wrong {f(v.get('wrong_hid_ratio'))} own<wrong {v['own_closer_than_wrong']:.2f}")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
