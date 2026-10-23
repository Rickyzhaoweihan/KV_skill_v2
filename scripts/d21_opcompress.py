"""D21 (operator): interface-fidelity screen of the compressed per-document OPERATOR under the paper-form D13 read.

The operator = the 4B encoder's post-RoPE K/V at the paired encoder layers over the document prefix (minus the
64-token tail), stored bf16 [Hkv,N,hd] per layer. Variants ('+'-joined), applied to every val document's operator:
    kv8 | kv4                per (kv head, channel) symmetric over tokens (D14's fake_quant), fp16 scales
    kv4g<G>                  same, one scale per group of G tokens
    acc<p> | rand<p>         keep p% of the columns per kv head and layer: acc = attention each column receives in the
                             encoder's OWN document prefill (D8/D14 rule; static, query-independent) | random
    full_rerun               unchanged (noise floor of regenerating)
The interface (outputs/D13_4b_reg_w10/d13_best.pt) is unchanged. Columns keep their RoPE positions; the reader rows
stay at position N. Fewer layers = no operator at a read layer = the interface layer-drop rows of d21_compress.py.
Metrics as in d21_compress.py (KL, read-out cos, agreement, val acc on the base's held-out VAL docs) + bytes per doc
vs the 9B reader's own prefix cache of the same document.

    python scripts/d21_opcompress.py --out outputs/D21_op
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import _get, log  # noqa: E402

DEFAULT = ("full,full_rerun,kv8,kv4,kv4g128,kv4g64,acc50,acc25,acc10,rand25,acc50+kv8,acc50+kv4g64,acc25+kv8,"
           "acc25+kv4g64,acc10+kv4g64")


def parse(spec):
    out = {"bits": 16, "group": 0, "score": None, "frac": 1.0}
    for t in [x for x in spec.split("+") if x and not x.startswith("full")]:
        m = re.fullmatch(r"kv(\d+)(?:g(\d+))?", t)
        if m:
            out["bits"], out["group"] = int(m.group(1)), int(m.group(2) or 0)
            continue
        m = re.fullmatch(r"(acc|rand)(\d+)", t)
        if m:
            out["score"], out["frac"] = m.group(1), int(m.group(2)) / 100.0
            continue
        raise ValueError(t)
    return out


def n_scales(Hkv, N, hd, group):
    return Hkv * hd * (-(-N // group) if group else 1)


@torch.no_grad()
def transform(op: dict, imp: dict, spec: str, seed: int = 0) -> tuple[dict, int]:
    """op {e: (K, V) [Hkv,N,hd]} -> (compressed op, stored bytes)."""
    from xenc.cache_compress import fake_quant
    p = parse(spec)
    out, nb = {}, 0
    for e, (K, V) in op.items():
        Hkv, N, hd = K.shape
        if p["score"]:
            b = max(1, int(round(p["frac"] * N)))
            if p["score"] == "acc":
                sc = imp[e]
            else:
                g = torch.Generator().manual_seed(seed * 1000 + e)
                sc = torch.rand(Hkv, N, generator=g).to(K.device)
            idx = sc.topk(b, -1).indices.sort(-1).values
            gi = idx.unsqueeze(-1).expand(-1, -1, hd)
            K, V = K.gather(1, gi), V.gather(1, gi)
            N = b
        if p["bits"] < 16:
            K = fake_quant(K, p["bits"], dim=1, group=p["group"])
            V = fake_quant(V, p["bits"], dim=1, group=p["group"])
            nb += 2 * (Hkv * N * hd * p["bits"] // 8 + 2 * n_scales(Hkv, N, hd, p["group"]))
        else:
            nb += 2 * Hkv * N * hd * 2
        if p["score"]:
            nb += Hkv * N * 2                     # column index (int16) per kept column; positions are implied
        out[e] = (K, V)
    return out, nb


def cache_bytes(cache) -> tuple[int, int]:
    """(all tensors, full-attention K/V only) bytes of an HF cache."""
    tot, kv = 0, 0
    for L in getattr(cache, "layers", []):
        for k, v in vars(L).items():
            if torch.is_tensor(v):
                tot += v.numel() * v.element_size()
                if k in ("keys", "values"):
                    kv += v.numel() * v.element_size()
    return tot, kv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--interface", default="outputs/D13_4b_reg_w10/d13_best.pt")
    ap.add_argument("--val", default="quality:10,musique:20,lastingbench:20,longhealth:0")
    ap.add_argument("--val_fold", type=int, default=1)
    ap.add_argument("--val_q_per_doc", type=int, default=0)
    ap.add_argument("--val_seed", type=int, default=1)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--variants", default=DEFAULT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    from d21_compress import answer_key, summarize_variant
    from fit_d13 import D13Rt, build, select_items
    from xenc.doc_tasks import score
    from deltaskill.backbone_utils import decoder_layers
    from xenc.kv_prune import received_attention
    variants = [v for v in a.variants.split(",") if v]
    if a.smoke:
        a.val = ",".join(f"{s.split(':')[0]}:1" for s in a.val.split(","))
        a.val_q_per_doc, variants = 2, variants[:4]
    if variants[0] != "full":
        variants = ["full"] + [v for v in variants if v != "full"]
    for v in variants:
        parse(v)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    reader, enc, tok, itf, lmap = build(a, init=a.interface)
    rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens, lru=1)
    layers = sorted(lmap)
    val, docs = select_items(a.val, "val", {a.val_fold}, a.val_q_per_doc, a.val_seed)
    sub, _ = select_items(a.val, "val", {a.val_fold}, 4, a.val_seed)
    subkeys = {_get(i, "key") for i in sub}
    rt.docs = docs
    by_doc: OrderedDict = OrderedDict()
    for it in sorted(val, key=lambda i: _get(i, "doc_id")):
        by_doc.setdefault(_get(it, "doc_id"), []).append(it)
    log(f"D21 operator screen: {a.interface} | val {len(val)} items / {len(by_doc)} docs | {len(variants)} variants")
    cap: dict = {"on": None}
    orig_read = itf.read

    def rec_read(li, *args, **kw):
        res = orig_read(li, *args, **kw)
        if cap["on"] is not None:
            cap["on"][li] = res["y"].detach()
        return res
    itf.read = rec_read
    R, Tc = rt.R, rt.Tc

    @torch.no_grad()
    def harvest(did):
        """Operator + acc importance from ONE encoder prefill of the document prefix (no question)."""
        ids = rt.pre(did)[: -a.tail]
        Tc.rec, Tc.kv_only = {}, False
        enc(input_ids=torch.tensor([ids], device=rt.edev), use_cache=False, logits_to_keep=1)
        rec, Tc.rec = Tc.rec, None
        op, imp = {}, {}
        for e, r in rec.items():
            op[e] = (r["k"].to(rt.dev), r["v"].to(rt.dev))
            sc = decoder_layers(enc)[e].self_attn.scaling
            imp[e] = received_attention(r["q"].unsqueeze(0), r["k"].unsqueeze(0), sc).to(rt.dev)
        del rec
        return op, imp, len(ids)

    @torch.no_grad()
    def gen(op, n_op, did, it):
        from register_state import _stop_ids
        from xenc.backbone import decode
        R.load(op, n_op)
        R.mode = "student"
        try:
            out = reader(input_ids=torch.tensor([rt.rows(did, it)], device=rt.dev), use_cache=True, logits_to_keep=1)
            stops, new = _stop_ids(tok), []
            for _ in range(a.budget):
                nxt = int(out.logits[0, -1].argmax())
                new.append(nxt)
                if nxt in stops or "</answer>" in tok.decode(new[-12:], skip_special_tokens=True):
                    break
                out = reader(input_ids=torch.tensor([[nxt]], device=rt.dev), past_key_values=out.past_key_values,
                             use_cache=True, logits_to_keep=1)
        finally:
            R.mode, R.past = "off", {}
        txt = decode(tok, a.reader, new)
        return (txt.split("</answer>", 1)[0] + "</answer>" if "</answer>" in txt else txt), new

    @torch.no_grad()
    def tf(op, n_op, did, it, ids):
        R.load(op, n_op)
        R.mode, cap["on"] = "student", {}
        try:
            out = reader(input_ids=torch.tensor([rt.rows(did, it, ids)], device=rt.dev), use_cache=False,
                         logits_to_keep=len(ids) + 1)
            lp, ys = torch.log_softmax(out.logits[0, :-1].float(), -1), cap["on"]
        finally:
            R.mode, R.past, cap["on"] = "off", {}, None
        return lp, ys

    rows_v = {v: [] for v in variants}
    sizes = {v: [] for v in variants}
    rc_all, rc_kv, n_tok = [], [], []
    t0 = time.time()
    for di, (did, its) in enumerate(by_doc.items()):
        op, imp, N = harvest(did)
        with torch.no_grad():
            o = reader(input_ids=torch.tensor([rt.pre(did)], device=rt.dev), use_cache=True, logits_to_keep=1)
            tb, kb = cache_bytes(o.past_key_values)
            del o
        rc_all.append(tb)
        rc_kv.append(kb)
        n_tok.append(N)
        refs = []
        for it in its:
            g, ids = gen(op, N, did, it)
            lp, ys = tf(op, N, did, it, ids)
            refs.append({"g": g, "ids": ids, "lp": lp, "ys": ys, "score": float(score(it, g)), "ak": answer_key(it, g)})
        for v in variants:
            opv, nb = transform(op, imp, v)
            sizes[v].append(nb)
            for it, ref in zip(its, refs):
                if v == "full":
                    g, lpv, ysv = ref["g"], ref["lp"], ref["ys"]
                else:
                    g, _ = gen(opv, N, did, it)
                    lpv, ysv = tf(opv, N, did, it, ref["ids"])
                kl_t = (ref["lp"].exp() * (ref["lp"] - lpv)).sum(-1)
                cos = {str(li): float(torch.nn.functional.cosine_similarity(ysv[li], ref["ys"][li], dim=-1).mean())
                       for li in layers}
                rows_v[v].append({"key": _get(it, "key"), "doc_id": did, "task": did.split("_")[0],
                                  "sub96": _get(it, "key") in subkeys, "gen": g, "score": float(score(it, g)),
                                  "full_score": ref["score"], "agree": answer_key(it, g) == ref["ak"],
                                  "kl": float(kl_t.mean()), "kl_first": float(kl_t[0]), "cos": cos})
            del opv
        del refs, op, imp
        torch.cuda.empty_cache()
        log(f"  [{di + 1}/{len(by_doc)} docs, {len(rows_v['full'])} items, N {N}] {time.time() - t0:.0f}s | " + " ".join(
            f"{v}: agr {sum(r['agree'] for r in rows_v[v]) / len(rows_v[v]):.3f}" for v in variants[1:5]))
        if (di + 1) % 5 == 0 or di + 1 == len(by_doc):
            summ = {}
            for v in variants:
                s = summarize_variant(rows_v[v], layers)
                s["MiB_per_doc"] = sum(sizes[v]) / len(sizes[v]) / 2 ** 20
                s["x_vs_reader_cache"] = sum(rc_all) / max(sum(sizes[v]), 1)
                s["x_vs_reader_kv"] = sum(rc_kv) / max(sum(sizes[v]), 1)
                summ[v] = s
            ref = {"reader_cache_MiB_per_doc": sum(rc_all) / len(rc_all) / 2 ** 20,
                   "reader_fullattn_kv_MiB_per_doc": sum(rc_kv) / len(rc_kv) / 2 ** 20,
                   "mean_operator_tokens": sum(n_tok) / len(n_tok)}
            json.dump({"args": vars(a), "n_docs_done": di + 1, "reference": ref, "summary": summ},
                      open(os.path.join(a.out, "op_fidelity.json.tmp"), "w"), indent=1)
            os.replace(os.path.join(a.out, "op_fidelity.json.tmp"), os.path.join(a.out, "op_fidelity.json"))
    with open(os.path.join(a.out, "items.jsonl"), "w") as f:
        for v in variants:
            for r in rows_v[v]:
                f.write(json.dumps({"variant": v, **r}) + "\n")
    L = ["| operator | MiB/doc | × smaller than 9B prefix cache (all / full-attn KV) | KL | agreement | val acc | Δ ± se |",
         "|---|---|---|---|---|---|---|"]
    for v, s in summ.items():
        L.append(f"| `{v}` | {s['MiB_per_doc']:.2f} | {s['x_vs_reader_cache']:.1f} / {s['x_vs_reader_kv']:.1f} | "
                 f"{s['kl']:.4f} | {s['agree']:.3f} | {s['acc']:.3f} | {s['d_acc']:+.3f} ± {s['d_acc_se']:.3f} |")
    md = "\n".join(L)
    open(os.path.join(a.out, "op_fidelity.md"), "w").write(f"## D21 operator screen\n\nreference {ref}\n\n{md}\n")
    log("\n" + md)
    log(f"done ({time.time() - t0:.0f}s) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
