"""D10: fit the cross-model query map A by the ADDRESSING objective on prefill statistics (no labels, no questions).

For documents of several domains both models read the same `text` prompt (identical tokenizers). At reader full-
attention layer ℓ (paired encoder layer ℓ'), reader head h, and sampled document positions t:
    teacher  p9 = softmax(RoPE_t(q9_h,t) · K9_{g(h)}[≤t] / √d)          the reader's own causal attention
    student  p2 = softmax(RoPE_t(A_ℓ,h q9_t) · K2_{e(h)}[≤t] / √d)       the cross-model read of the 2B keys
    loss     KL(p9 ‖ p2), Adam on A (scale of A = free temperature), init from ridge maps (--init)
A modes: head (A_h: q9_h -> 256) or mix (A_h: concat_h q9 (16·256) -> 256). Afterwards B is ridge-fitted per head on
the branch's read output o = p2·V2_{e(h)} -> the reader's own attention output o9_h at the same rows.
Reports held-out-doc JS(p9 ‖ p2) per layer: ridge init vs fitted. Saves maps (attn_A / attn_B) for --init.

    python scripts/fit_attn_addressing.py --init outputs/D10_maps/maps_9b_2b_xdomain.pt --a_mode head \
        --out outputs/D10_maps/maps_9b_2b_addr_head.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from fit_state_maps import FORMAT, Gram, ridge  # noqa: E402
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402


def rope_rows(rot, x, pos, dev):
    """x [H,R,hd] rotated at positions pos [R] with the reader's rotary embedding (partial RoPE aware)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    p = pos.view(1, 1, -1).expand(3, 1, -1).to(dev)
    cos, sin = rot(x.unsqueeze(0), p)
    out, _ = apply_rotary_pos_emb(x.unsqueeze(0), x[:1].unsqueeze(0), cos.float(), sin.float())
    return out[0]


def scores(qrot, K, pos, scale):
    """qrot [H,R,hd], K [H,T,hd] -> masked logits [H,R,T] (cols > pos masked)."""
    s = torch.einsum("hrd,htd->hrt", qrot, K) * scale
    T = K.shape[1]
    mask = torch.arange(T, device=s.device).view(1, 1, T) > pos.view(1, -1, 1).to(s.device)
    return s.masked_fill(mask, float("-inf"))


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (a.clamp_min(1e-12).log() - b.clamp_min(1e-12).log())).sum(-1)   # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--splits", default="quality:40,longhealth:12,musique:30,lastingbench:30")
    ap.add_argument("--exclude_folds", default="0,1")
    ap.add_argument("--layers", default="3,7,11,15")
    ap.add_argument("--max_tokens", type=int, default=6000)
    ap.add_argument("--rows", type=int, default=256, help="sampled query positions per document")
    ap.add_argument("--min_pos", type=int, default=64)
    ap.add_argument("--heldout_frac", type=float, default=0.1)
    ap.add_argument("--init", default=None, help="ridge maps with attn_A / attn_B (head mode)")
    ap.add_argument("--a_mode", choices=["head", "mix"], default="head")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from xenc.backbone import build_chat, input_device, load_encoder, load_reader
    from xenc.doc_tasks import ctx_prompt, load_docs
    from xenc.xattn_read import AttnReader, attn_dims, full_depth_map, make_attn_interface
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    reader, tok = load_reader(a.reader, device_map=a.device)
    enc = load_encoder(a.encoder, device_map=a.device)[0]
    ra, ea = attn_dims(reader), attn_dims(enc)
    keep = {int(x) for x in a.layers.split(",")}
    amap = {k: v for k, v in full_depth_map(ra, ea).items() if k in keep}
    Ar = AttnReader(reader, make_attn_interface(reader, ra, layers=sorted(amap)), record_mag=False)
    Ae = AttnReader(enc, make_attn_interface(enc, ea, layers=sorted(set(amap.values()))), record_mag=False)
    dev = input_device(reader)
    H, hd = ra["n_q"], ra["hd"]
    g9 = [h // (ra["n_q"] // ra["n_kv"]) for h in range(H)]                     # reader kv head
    e2 = [g * ea["n_kv"] // ra["n_kv"] for g in g9]                              # encoder kv head
    scale = hd ** -0.5
    rot = Ar.rot

    # ---------------- cache prefill statistics
    excl = {int(x) for x in a.exclude_folds.split(",") if x != ""}
    data = {"fit": [], "held": []}
    for spec in a.splits.split(","):
        name, nd = spec.split(":")
        sd = f"data/{name}_split"
        docs = load_docs(sd)
        items = [i for i in load_split_items(sd, "train") if (_get(i, "meta") or {}).get("fold") not in excl]
        first = {}
        for it in items:
            first.setdefault(_get(it, "doc_id"), it)
        dids = sorted(first)[: int(nd)]
        n_held = max(1, int(round(len(dids) * a.heldout_frac)))
        for k, did in enumerate(dids):
            ids = build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(first[did], docs[did]))[: a.max_tokens]
            T = len(ids)
            if T <= a.min_pos + 8:
                continue
            x = torch.tensor([ids])
            Ar.capture, Ar.capture_kv, Ae.capture_kv = {}, {}, {}
            with torch.no_grad():
                reader(input_ids=x.to(dev), use_cache=False, logits_to_keep=1)
                enc(input_ids=x.to(input_device(enc)), use_cache=False, logits_to_keep=1)
            pos = torch.tensor(sorted(rng.sample(range(a.min_pos, T), min(a.rows, T - a.min_pos))))
            rec = {"doc": did, "domain": name, "pos": pos, "layers": {}}
            for li, e in amap.items():
                q_pre, o9 = Ar.capture[li]
                K9, _ = Ar.capture_kv[li]
                K2, V2 = Ae.capture_kv[e]
                pi = pos.to(q_pre.device)
                rec["layers"][li] = {"q": q_pre[0][:, pi].half().cpu(), "o9": o9[0][:, pi].half().cpu(),
                                     "K9": K9[0].half().cpu(), "K2": K2[0].half().cpu(), "V2": V2[0].half().cpu()}
            Ar.capture = Ar.capture_kv = Ae.capture_kv = None
            data["held" if k >= len(dids) - n_held else "fit"].append(rec)
        log(f"  cached {name}: {len(dids)} docs")
    torch.cuda.empty_cache()
    log(f"cached {len(data['fit'])} fit / {len(data['held'])} held-out documents, {a.rows} rows each")

    # ---------------- parameters
    init = torch.load(a.init, map_location="cpu", weights_only=False) if a.init else {}
    A = {}
    for li in amap:
        A0 = init.get("attn_A", {}).get(str(li))
        A0 = A0.float() if A0 is not None else torch.eye(hd).repeat(H, 1, 1)
        if a.a_mode == "mix":
            W = torch.zeros(H, hd, H * hd)
            for h in range(H):
                W[h, :, h * hd:(h + 1) * hd] = A0[h]
            A0 = W
        A[li] = torch.nn.Parameter(A0.to(dev))

    def student_logits(li, rec_l, pos):
        q = rec_l["q"].to(dev).float()                                       # [H,R,hd]
        if a.a_mode == "head":
            qa = torch.einsum("hrd,hed->hre", q, A[li])
        else:
            qa = torch.einsum("rf,hef->hre", q.transpose(0, 1).reshape(q.shape[1], -1), A[li])
        qr = rope_rows(rot, qa, pos, dev)
        K2 = rec_l["K2"].to(dev).float()[e2]                                 # [H,T,hd]
        return scores(qr, K2, pos, scale)

    def teacher_probs(rec_l, pos):
        q = rope_rows(rot, rec_l["q"].to(dev).float(), pos, dev)
        K9 = rec_l["K9"].to(dev).float()[g9]
        return torch.softmax(scores(q, K9, pos, scale), -1)

    @torch.no_grad()
    def held_js():
        out = {}
        for li in amap:
            v = []
            for rec in data["held"]:
                rl = rec["layers"][li]
                p = teacher_probs(rl, rec["pos"])
                qd = torch.softmax(student_logits(li, rl, rec["pos"]), -1)
                v.append(float(js(p, qd).mean()))
            out[li] = sum(v) / max(len(v), 1)
        return out

    js0 = held_js()
    log("held-out JS (init): " + " ".join(f"L{li}:{v:.4f}" for li, v in js0.items()))
    opt = torch.optim.Adam([A[li] for li in amap], lr=a.lr)
    hist = []
    for ep in range(a.epochs):
        order = list(range(len(data["fit"])))
        rng.shuffle(order)
        tot, n = 0.0, 0
        for k in order:
            rec = data["fit"][k]
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for li in amap:
                rl = rec["layers"][li]
                with torch.no_grad():
                    p = teacher_probs(rl, rec["pos"])
                ls = torch.nan_to_num(torch.log_softmax(student_logits(li, rl, rec["pos"]), -1), neginf=0.0)
                kl = (p * (p.clamp_min(1e-12).log() - ls)).sum(-1).mean()
                loss = loss + kl
            (loss / len(amap)).backward()
            opt.step()
            tot += float(loss) / len(amap)
            n += 1
        jsx = held_js()
        hist.append({"epoch": ep + 1, "train_kl": tot / max(n, 1), "held_js": jsx})
        log(f"  epoch {ep + 1}: train KL {tot / max(n, 1):.4f} | held-out JS " +
            " ".join(f"L{li}:{v:.4f}" for li, v in jsx.items()))

    # ---------------- B: ridge on the branch's read output -> reader's own attention output
    AB = {}
    with torch.no_grad():
        for li in amap:
            G = [Gram(hd, hd, dev) for _ in range(H)]
            for rec in data["fit"]:
                rl = rec["layers"][li]
                p2 = torch.softmax(student_logits(li, rl, rec["pos"]), -1)
                V2 = rl["V2"].to(dev).float()[e2]
                ob = torch.einsum("hrt,htd->hrd", p2, V2)
                o9 = rl["o9"].to(dev).float()
                for h in range(H):
                    G[h].add(ob[h], o9[h])
            AB[str(li)] = torch.stack([ridge(G[h], a.lam).t().float().cpu() for h in range(H)])
            r2 = sum(G[h].r2(ridge(G[h], a.lam)) for h in range(H)) / H
            log(f"  B L{li}: fit R² (branch read -> reader output) {r2:.3f}")
    jsf = held_js()
    rep = {"held_js_init": js0, "held_js_final": jsf, "history": hist, "a_mode": a.a_mode,
           "n_fit": len(data["fit"]), "n_held": len(data["held"])}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"format": FORMAT, "A": {}, "B": {}, "layer_map": {}, "attn_A": {str(li): A[li].detach().float().cpu()
                for li in amap}, "attn_B": AB, "attn_map": amap, "attn_a_mode": a.a_mode, "report": rep,
                "args": vars(a)}, a.out)
    with open(os.path.splitext(a.out)[0] + "_report.json", "w") as f:
        json.dump(rep, f, indent=1)
    log(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
