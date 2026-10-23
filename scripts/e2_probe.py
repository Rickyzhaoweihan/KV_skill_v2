"""E2 diagnostic: which ENCODER layers should feed which READER read layers (label-free except the logit lens).

Both models read the same TRAIN documents as text (ctx_prompt + gold answer, teacher-forced). Per item we keep the
rows the read acts on (the last --rows positions: question + answer) plus a few random document rows, and record

  targets  reader attention-module output a_ℓ (post o_proj, d_r) at reader full-attention layers ℓ
  sources  encoder attention-module output o_e (post o_proj, d_e) at EVERY encoder full-attention layer e
           ms(e, e2): layer e's attention pattern reading layer e2's own values through e2's o_proj (shared
           addressing, deeper values; axis-2 candidate)
           deep-value reads dv(e, e'): encoder layer e's own attention pattern at the row, but values from a DEEPER
           residual stream: v' = v_proj_e(input_layernorm_e(h_{e'})) with h_{e'} = output of encoder layer e'
           (the axis-3 candidate; at doc positions h_{e'} is question-independent, i.e. harvestable once)
  lens     logit lens of every encoder layer output h_ℓ at the answer rows: mean log p(gold answer-content token)
           and top-1 accuracy (how decodable the answer is per encoder depth)
  span     for extractive items whose gold string occurs in the document: encoder attention mass on the gold span
           from the answer-start row per full layer, as a lift over uniform (which layers retrieve the evidence)

Then ridge (closed form, docs split fit/val/held; lambda picked on val, R² reported on HELD docs, question/answer rows
only) from source sets to each reader target: every single o_e (B-style map quality; the base pairs ℓ <- e = ℓ),
every pair / triple containing the base layer (axis 2: multi-layer operator), and o_e + dv(e, e') (axis 3).
A sanity check recomputes o_e from the captured q/k/v/gate and must match the module output.

    python scripts/e2_probe.py --out outputs/E2_probe
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
import types
from collections import OrderedDict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402


# ----------------------------------------------------------------------------- capture
class EncCap:
    """Patch encoder full-attention modules: record per layer k, v (post-RoPE / raw, [Hkv,T,hd]) for all positions,
    q (post-RoPE) and gate at `rows`, and the module output at `rows`. Decoder-layer outputs (residual h_ℓ) via hooks."""

    def __init__(self, enc, full, keep_h: set):
        from deltaskill.backbone_utils import decoder_layers
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        self.L = decoder_layers(enc)
        self.full, self.keep_h = list(full), set(keep_h)
        self.rows = None
        self.rec, self.h = {}, {}
        self._orig, self._hooks = {}, []
        cap = self
        for e in self.full:
            m = self.L[e].self_attn
            self._orig[e] = m.forward

            def fwd(self_, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, _e=e, **kw):
                shp = hidden_states.shape[:-1]
                hs = (*shp, -1, self_.head_dim)
                q, g = torch.chunk(self_.q_proj(hidden_states).view(*shp, -1, self_.head_dim * 2), 2, dim=-1)
                g = g.reshape(*shp, -1)
                q = self_.q_norm(q.reshape(hs)).transpose(1, 2)
                k = self_.k_norm(self_.k_proj(hidden_states).view(hs)).transpose(1, 2)
                v = self_.v_proj(hidden_states).view(hs).transpose(1, 2)
                cos, sin = position_embeddings
                qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
                out = cap._orig[_e](hidden_states, position_embeddings, attention_mask, past_key_values=past_key_values, **kw)
                r = cap.rows
                cap.rec[_e] = {"q": qr[0][:, r], "k": kr[0], "v": v[0], "g": g[0][r], "out": out[0][0][r]}
                return out
            m.forward = types.MethodType(fwd, m)
        for li in range(len(self.L)):
            self._hooks.append(self.L[li].register_forward_hook(self._hook(li)))

    def _hook(self, li):
        def h(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            self.h[li] = x[0] if li in self.keep_h else x[0][self.rows_lens]
        return h

    def start(self, rows, rows_lens):
        self.rows, self.rows_lens, self.rec, self.h = rows, rows_lens, {}, {}


def attend_rows(att, q, K, V, g, rows):
    """q [Hq,R,hd] at absolute positions `rows`; K,V [Hkv,T,hd]; causal. -> (o_proj output [R,d], p [Hkv,G,R,T])."""
    Hq, R, hd = q.shape
    Hkv = K.shape[0]
    G = Hq // Hkv
    s = torch.einsum("kgrd,ktd->kgrt", q.float().view(Hkv, G, R, hd), K.float()) * att.scaling
    T = K.shape[1]
    mask = torch.arange(T, device=q.device).view(1, T) > rows.view(R, 1)
    p = torch.softmax(s.masked_fill(mask.view(1, 1, R, T), float("-inf")), -1)
    o = torch.einsum("kgrt,ktd->kgrd", p, V.float()).reshape(Hq, R, hd).transpose(0, 1).reshape(R, Hq * hd)
    o = att.o_proj((o * torch.sigmoid(g.float())).to(att.o_proj.weight.dtype))
    return o.float(), p


# ----------------------------------------------------------------------------- ridge
def ridge_r2(Xs, Y, split, lams=(1e-3, 1e-2, 1e-1, 1.0), ret_map=False):
    """Xs list of [N,d_i] (fp16, GPU), Y [N,D]; split dict of index tensors fit/val/held (held, val = q/a rows).
    -> (held R², val R², lambda). R² = 1 - SSE/SST pooled over output dims (around the fit mean)."""
    X = torch.cat([x.float() for x in Xs], 1)
    f, v, h = split["fit"], split["val"], split["held"]
    mx, my = X[f].mean(0), Y[f].float().mean(0)
    Xf, Yf = X[f] - mx, Y[f].float() - my
    G = Xf.t() @ Xf
    C = Xf.t() @ Yf
    sc = float(G.diagonal().mean())
    I = torch.eye(G.shape[0], device=G.device)

    def r2(W, idx):
        P = (X[idx] - mx) @ W + my
        Yi = Y[idx].float()
        return 1.0 - float(((P - Yi) ** 2).sum()) / float(((Yi - Yi.mean(0)) ** 2).sum())
    best = None
    for lm in lams:
        W = torch.linalg.solve(G + lm * sc * I, C)
        rv = r2(W, v)
        if best is None or rv > best[1]:
            best = (W, rv, lm)
    W, rv, lm = best
    if ret_map:                 # y ≈ x @ W + b  ->  {"W": [D_out, D_in] (as B), "b": [D_out]}
        return r2(W, h), rv, lm, {"W": W.t().contiguous().cpu(), "b": (my - mx @ W).cpu()}
    return r2(W, h), rv, lm


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--train", default="quality:40,qasper:40,musique:25,lastingbench:25,longhealth:0")
    ap.add_argument("--q_per_doc", type=int, default=2)
    ap.add_argument("--lh_folds", default="2,3,4")
    ap.add_argument("--read_layers", default="3,7,11,15,19", help="reader layers whose base pair is ℓ <- e = ℓ")
    ap.add_argument("--rows", type=int, default=128, help="last rows (question + answer) kept per item")
    ap.add_argument("--doc_rows", type=int, default=32, help="extra random document rows per item (fit only)")
    ap.add_argument("--max_tokens", type=int, default=12000)
    ap.add_argument("--exclude_sets", default=",".join(f"outputs/D22_sets/{t}.json" for t in (
        "qasper", "contractnli", "shakespeare_cf", "loogle", "loogle_long", "clipper", "musique", "lastingbench")))
    ap.add_argument("--exclude_docs", default="outputs/D13_lever1_leakage.json")
    ap.add_argument("--save_B2", type=int, default=1, help="1: save the ridge maps of every ms(ℓ, e2) source alone "
                    "(<out>/e2_B2.pt; init of xenc/e2_read.py's B2)")
    ap.add_argument("--tiny", action="store_true", help="CPU smoke with random tiny models (code path only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    from fit_d13 import TASK_SPLIT
    from register_d13 import excluded_docs
    from xenc.backbone import build_chat, input_device, load_encoder, load_reader, text_ids
    from xenc.doc_tasks import ctx_prompt, gold_text, load_docs
    from xenc.state_skill import full_layers
    from deltaskill.backbone_utils import decoder_layers
    os.makedirs(a.out, exist_ok=True)
    if a.tiny:                 # CPU smoke: random tiny Qwen3.5 models over the real Qwen tokenizer's vocabulary
        sys.path.insert(0, str(_ROOT / "tests"))
        from _tiny import tiny_qwen35
        from xenc.doc_tasks import load_qwen_tokenizer
        tok = load_qwen_tokenizer(a.reader)
        V = len(tok)
        reader, enc = tiny_qwen35(vocab=V, d=64, seed=0), tiny_qwen35(vocab=V, d=48, seed=1)
    else:
        reader, tok = load_reader(a.reader, device_map=a.device)
        enc = load_encoder(a.encoder, device_map=a.device)[0]
    rfull, efull = full_layers(reader), full_layers(enc)
    nE = len(decoder_layers(enc))
    read_layers = [int(x) for x in a.read_layers.split(",")]
    assert set(read_layers) <= set(efull) & set(rfull), (read_layers, rfull, efull)
    deep = {e: sorted({x for x in (e + 4, e + 8, e + 12, efull[-1]) if x in efull and x > e}) for e in read_layers}
    keep_h = sorted({x for v in deep.values() for x in v})
    log(f"E2 probe: {a.encoder} -> {a.reader} | reader full {rfull} | encoder full {efull} | base pairs "
        f"{ {l: l for l in read_layers} } | deep-value sources {deep}")

    # ---- items (train split only; judged docs and near-duplicates dropped)
    drop = excluded_docs(a.exclude_sets)
    if a.exclude_docs:
        X = json.load(open(a.exclude_docs))
        drop |= set(sum(X.values(), [])) if isinstance(X, dict) else set(X)
    lh_folds = {int(x) for x in a.lh_folds.split(",")}
    items, docs, doc_split = [], {}, {}
    for spec in a.train.split(","):
        name, nd = spec.split(":")
        sd = TASK_SPLIT[name]
        dd = load_docs(sd)
        its = load_split_items(sd, "train")
        if name == "longhealth":
            its = [i for i in its if (_get(i, "meta") or {}).get("fold") in lh_folds]
        by = OrderedDict()
        for it in its:
            if _get(it, "doc_id") not in drop:
                by.setdefault(_get(it, "doc_id"), []).append(it)
        dids = sorted(by)
        rng.shuffle(dids)
        dids = dids[: int(nd)] if int(nd) else dids
        nh, nv = max(1, round(0.2 * len(dids))), max(1, round(0.1 * len(dids)))
        for k, d in enumerate(dids):
            doc_split[d] = "held" if k < nh else ("val" if k < nh + nv else "fit")
            docs[d] = dd[d]
            qs = by[d][:]
            rng.shuffle(qs)
            items += qs[: a.q_per_doc]
        log(f"  {name}: {len(dids)} docs (held {nh}, val {nv}), excluded {sum(1 for x in drop if x.startswith(name))}")
    assert not (set(docs) & drop)
    log(f"  {len(items)} items over {len(docs)} docs")

    # ---- capture
    rcap = {}
    rL = decoder_layers(reader)
    hooks = []
    for li in rfull:
        def mk(li=li):
            def h(mod, inp, out):
                rcap[li] = (out[0] if isinstance(out, tuple) else out)[0]
            return h
        hooks.append(rL[li].self_attn.register_forward_hook(mk()))
    ecap = EncCap(enc, efull, keep_h)
    eL = decoder_layers(enc)
    norm, head = enc.model.norm, enc.get_output_embeddings()
    pre_ids, post_ids = text_ids(tok, "<answer>"), text_ids(tok, "</answer>")
    S = {"src": {}, "tgt": {}, "qa": [], "split": [], "task": []}
    lens = {"lp": torch.zeros(nE), "top1": torch.zeros(nE), "n": 0}
    span = {e: [] for e in efull}
    sanity = []
    dev, edev = input_device(reader), input_device(enc)
    t0 = time.time()
    n_ok = 0
    for n, it in enumerate(items):
        did = _get(it, "doc_id")
        gold = gold_text(it)
        tt = text_ids(tok, gold)
        ids = build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, docs[did])) + tt
        if len(ids) > a.max_tokens:
            continue
        T, m = len(ids), len(tt)
        R = min(a.rows, T - 1)
        qa = list(range(T - R, T))
        extra = sorted(rng.sample(range(0, T - R), min(a.doc_rows, T - R)))
        rows = torch.tensor(extra + qa, device=edev)
        # answer-content tokens: gold = <answer> content </answer>
        ok_lens = tt[: len(pre_ids)] == pre_ids and tt[len(tt) - len(post_ids):] == post_ids
        c0, c1 = T - m + len(pre_ids), T - len(post_ids)          # content token positions [c0, c1)
        lens_pos = torch.arange(c0 - 1, c1 - 1, device=edev) if ok_lens and c1 > c0 else None
        ecap.start(rows, lens_pos if lens_pos is not None else rows[-1:])
        with torch.no_grad():
            reader(input_ids=torch.tensor([ids], device=dev), use_cache=False, logits_to_keep=1)
            enc(input_ids=torch.tensor([ids], device=edev), use_cache=False, logits_to_keep=1)
            # logit lens
            if lens_pos is not None:
                gold_c = torch.tensor(ids[c0:c1], device=edev)
                for li in range(nE):
                    hl = ecap.h[li] if li not in ecap.keep_h else ecap.h[li][lens_pos]
                    lp = torch.log_softmax(head(norm(hl)).float(), -1)
                    lens["lp"][li] += float(lp.gather(1, gold_c.view(-1, 1)).mean())
                    lens["top1"][li] += float(lp[0].argmax() == gold_c[0])
                lens["n"] += 1
            # sources
            src = {}
            for e in efull:
                r = ecap.rec[e]
                att = eL[e].self_attn
                if n_ok < 8:                       # sanity: recompute the module output from captured q/k/v/gate
                    o_re, _ = attend_rows(att, r["q"], r["k"], r["v"], r["g"], rows)
                    sanity.append(float((o_re - r["out"].float()).norm() / r["out"].float().norm()))
                src[f"o{e}"] = r["out"].half().cpu()
                if e in deep:
                    for e2 in deep[e]:
                        v2 = att.v_proj(eL[e].input_layernorm(ecap.h[e2])).view(T, -1, att.head_dim).transpose(0, 1)
                        o2, p = attend_rows(att, r["q"], r["k"], v2, r["g"], rows)
                        src[f"dv{e}_{e2}"] = o2.half().cpu()
                        # ms(e, e2): layer e's attention pattern reads layer e2's own values through e2's o_proj
                        att2, r2 = eL[e2].self_attn, ecap.rec[e2]
                        o3, _ = attend_rows(att2, r["q"], r["k"], r2["v"], r["g"], rows)
                        src[f"ms{e}_{e2}"] = o3.half().cpu()
            # span retrieval: gold string inside the document part (extractive items)
            ans = _get(it, "answers") if not _get(it, "choices") else None
            if ans and lens_pos is not None:
                doc_end = T - R
                for form in (" " + str(ans[0]).strip(), str(ans[0]).strip()):
                    sp = text_ids(tok, form)
                    if len(sp) < 1:
                        continue
                    hit = next((s for s in range(0, doc_end - len(sp)) if ids[s:s + len(sp)] == sp), None)
                    if hit is not None:
                        rowk = int(lens_pos[0])          # the row predicting the first content token
                        for e in efull:
                            r = ecap.rec[e]
                            att = eL[e].self_attn
                            j = int((rows == rowk).nonzero()[0]) if (rows == rowk).any() else None
                            if j is None:
                                break
                            _, p = attend_rows(att, r["q"][:, j:j + 1], r["k"], r["v"], r["g"][j:j + 1], rows[j:j + 1])
                            mass = float(p[..., hit:hit + len(sp)].sum(-1).mean())
                            span[e].append(mass / (len(sp) / (rowk + 1)))
                        break
        for k, v in src.items():
            S["src"].setdefault(k, []).append(v)
        for li in rfull:
            S["tgt"].setdefault(li, []).append(rcap[li][rows.to(dev)].half().cpu())
        nr = len(rows)
        S["qa"].append(torch.tensor([0] * len(extra) + [1] * len(qa), dtype=torch.bool))
        S["split"] += [doc_split[did]] * nr
        S["task"] += [did.split("_")[0]] * nr
        n_ok += 1
        if n_ok % 20 == 0:
            log(f"  {n_ok}/{len(items)} items ({time.time() - t0:.0f}s)")
    for h in hooks:
        h.remove()
    log(f"captured {n_ok} items; sanity rel err (recomputed o_e vs module) max {max(sanity):.2e}")
    assert max(sanity) < 2e-2, "attention re-implementation does not match the encoder module"
    del reader, enc, ecap
    torch.cuda.empty_cache()

    # ---- ridge
    gd = torch.device(a.device)
    src = {k: torch.cat(v).to(gd) for k, v in S["src"].items()}
    tgt = {li: torch.cat(v).to(gd) for li, v in S["tgt"].items()}
    qa = torch.cat(S["qa"]).to(gd)
    sp = S["split"]
    idx = {s: torch.tensor([i for i, x in enumerate(sp) if x == s], device=gd) for s in ("fit", "val", "held")}
    split = {"fit": idx["fit"], "val": idx["val"][qa[idx["val"]]], "held": idx["held"][qa[idx["held"]]]}
    log(f"rows: fit {len(split['fit'])}, val(q/a) {len(split['val'])}, held(q/a) {len(split['held'])}")
    res = {"single": {}, "pair": {}, "triple": {}, "deep": {}, "lens": {}, "span": {}, "sanity_max": max(sanity),
           "n_items": n_ok, "args": vars(a)}
    for li in rfull:
        Y = tgt[li]
        res["single"][li] = {e: ridge_r2([src[f"o{e}"]], Y, split)[0] for e in efull}
        log(f"  reader L{li}: single " + " ".join(f"E{e} {r:.3f}" for e, r in res["single"][li].items()))
    B2s = {}
    for li in read_layers:
        Y = tgt[li]
        res["pair"][li] = {f"{li}+{e}": ridge_r2([src[f"o{li}"], src[f"o{e}"]], Y, split)[0] for e in efull if e != li}
        res["triple"][li] = {f"{li}+{e1}+{e2}": ridge_r2([src[f"o{li}"], src[f"o{e1}"], src[f"o{e2}"]], Y, split)[0]
                             for e1, e2 in itertools.combinations([e for e in efull if e != li], 2)}
        res["deep"][li] = {}
        for e2 in deep[li]:
            res["deep"][li][f"dv{li}_{e2}"] = ridge_r2([src[f"dv{li}_{e2}"]], Y, split)[0]
            res["deep"][li][f"o{li}+dv{li}_{e2}"] = ridge_r2([src[f"o{li}"], src[f"dv{li}_{e2}"]], Y, split)[0]
            res["deep"][li][f"o{li}+ms{li}_{e2}"] = ridge_r2([src[f"o{li}"], src[f"ms{li}_{e2}"]], Y, split)[0]
        for e2 in deep[li]:
            if a.save_B2:
                r_, _, lm_, mp = ridge_r2([src[f"ms{li}_{e2}"]], Y, split, ret_map=True)
                res["deep"][li][f"ms{li}_{e2}"] = r_
                B2s[f"ms{li}_{e2}"] = {**mp, "lam": lm_, "r2_held": r_}
        bp = max(res["pair"][li].items(), key=lambda x: x[1])
        bt = max(res["triple"][li].items(), key=lambda x: x[1])
        log(f"  reader L{li}: base E{li} {res['single'][li][li]:.3f} | best pair {bp[0]} {bp[1]:.3f} | best triple "
            f"{bt[0]} {bt[1]:.3f} | deep " + " ".join(f"{k} {v:.3f}" for k, v in res["deep"][li].items()))
    if lens["n"]:
        res["lens"] = {li: {"logp": float(lens["lp"][li] / lens["n"]), "top1": float(lens["top1"][li] / lens["n"])}
                       for li in range(nE)}
        res["lens_n"] = lens["n"]
        log("  logit lens (enc layer: mean log p gold / top1): " + " ".join(
            f"{li}:{v['logp']:.2f}/{v['top1']:.2f}" for li, v in res["lens"].items()))
    res["span"] = {e: (sum(v) / len(v) if v else None) for e, v in span.items()}
    res["span_n"] = len(span[efull[0]])
    log(f"  gold-span attention lift (n={res['span_n']}): " + " ".join(
        f"E{e} {v:.1f}" for e, v in res["span"].items() if v is not None))
    if B2s:
        torch.save(B2s, os.path.join(a.out, "e2_B2.pt"))
    with open(os.path.join(a.out, "e2_probe.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    log(f"saved {a.out}/e2_probe.json ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
