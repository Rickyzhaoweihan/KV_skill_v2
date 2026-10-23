"""D10 (fallback): judge ENCODER-SIDE addressing — the encoder reads document + question natively, its attention
outputs at the question tokens are carried into the reader by B (xenc/encside_read.py). Reader-native tail.

Conditions (greedy, budget, SYSTEM_ANSWER):
  closed_book  chat(q_prompt)                    closed_tail  doc tail (last --tail tokens) + question suffix
  text         chat(ctx_prompt)
  kv           reader: own-doc tail + question suffix; injection from the encoder run over the own document + question
  wrong_doc    same with ANOTHER document (its tail in the reader prompt, its text in the encoder)
--fid: read-out fidelity at the reader prompt rows: relative error and cosine between the injected B·o2 and the
reader's own in-context attention output o9 (teacher = text prompt), per layer (level pre or post).

    python scripts/eval_encside.py --encoder same --identity --split_dir data/longhealth_split --fold 0 --n 0
    python scripts/eval_encside.py --encoder Qwen/Qwen3.5-2B --maps outputs/D10_maps/encB_9b_2b_pre.pt ...
"""

from __future__ import annotations

import argparse
import json
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

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402
from register_state import _stop_ids, greedy  # noqa: E402

CONDS = ("closed_book", "closed_tail", "text", "kv", "wrong_doc")


def layer_pairs(reader, enc, layers):
    from xenc.xattn_read import attn_dims, full_depth_map
    m = full_depth_map(attn_dims(reader), attn_dims(enc))
    return {k: v for k, v in m.items() if k in set(layers)}


def full_ids(tok, hub, docs, it, did):
    from xenc.state_skill import doc_prefix_text, question_suffix_ids
    pre = [int(x) for x in tok(doc_prefix_text(tok, hub, docs[did], SYSTEM_ANSWER),
                               add_special_tokens=False)["input_ids"]]
    return pre, question_suffix_ids(tok, hub, it, SYSTEM_ANSWER)


@torch.no_grad()
def enc_outputs(enc, ids, R, cap, same: bool, past=None):
    """Run the encoder on ids (optionally continuing `past`); -> ({enc_layer: [1,T,d]}, cache)."""
    from xenc.backbone import input_device
    x = torch.tensor([ids], device=input_device(enc))
    if same:
        R.enabled, R.capture = False, {}
        out = enc(input_ids=x, past_key_values=past, use_cache=True, logits_to_keep=1)
        o = dict(R.capture)
        R.capture, R.enabled = None, True
    else:
        cap.start()
        out = enc(input_ids=x, past_key_values=past, use_cache=True, logits_to_keep=1)
        o = cap.get()
    return o, out.past_key_values


@torch.no_grad()
def gen_encside(model, tok, hub, enc, R, cap, same, lmap, enc_ids, n_s, budget, stop="</answer>"):
    """Reader reads enc_ids[-n_s:] with injections from the encoder run over enc_ids; greedy decode, the
    encoder stepping along on every generated token."""
    from xenc.backbone import decode, input_device
    dev = input_device(model)
    o2, ecache = enc_outputs(enc, enc_ids, R, cap, same)
    if hasattr(R.itf, "build"):      # D12 context B: multi-layer / causal-window features
        R.cur = {li: f[:, -n_s:] for li, f in R.itf.build(o2, reset=True).items()}
    else:
        R.cur = {li: o2[e][:, -n_s:] for li, e in lmap.items()}
    out = model(input_ids=torch.tensor([enc_ids[-n_s:]], device=dev), use_cache=True, logits_to_keep=1)
    stops, new = _stop_ids(tok), []
    for _ in range(budget):
        nxt = int(out.logits[0, -1].argmax())
        new.append(nxt)
        if nxt in stops or stop in tok.decode(new[-12:], skip_special_tokens=True):
            break
        o2, ecache = enc_outputs(enc, [nxt], R, cap, same, past=ecache)
        R.cur = R.itf.build(o2) if hasattr(R.itf, "build") else {li: o2[e] for li, e in lmap.items()}
        out = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=out.past_key_values,
                    use_cache=True, logits_to_keep=1)
    R.cur = None
    txt = decode(tok, hub, new)
    if stop in txt:
        txt = txt.split(stop, 1)[0] + stop
    return txt, len(new)


@torch.no_grad()
def fidelity(model, enc, R, cap, same, lmap, enc_ids, n_s, teacher_ids):
    """Per layer: rel. error and cosine of B·o2 vs the reader's in-context own output at the last n_s rows."""
    from xenc.backbone import input_device
    hooks = []
    if R.itf.level == "post":      # D12: teacher = the attention module output (post o_proj)
        o9 = {}
        for li, m in R.mods.items():
            hooks.append(m.register_forward_hook(
                lambda mod, i, o, li=li: o9.__setitem__(li, (o[0] if isinstance(o, tuple) else o).detach())))
        R.enabled = False
    else:
        R.enabled, R.capture = False, {}
    model(input_ids=torch.tensor([teacher_ids], device=input_device(model)), use_cache=False, logits_to_keep=1)
    for h in hooks:
        h.remove()
    if not hooks:
        o9 = dict(R.capture)
    R.capture, R.enabled = None, True
    o2, _ = enc_outputs(enc, enc_ids, R, cap, same)
    feats = R.itf.build(o2, reset=True) if hasattr(R.itf, "build") else None   # D12 context B
    res = {}
    for li, e in lmap.items():
        if feats is not None:
            y = R.itf.map(li, feats[li][0, -n_s:].float().to(o9[li].device))
        else:
            y = R.itf.map(li, o2[e][0, -n_s:].float().to(o9[li].device))
        t = o9[li][0, -n_s:].float()
        res[str(li)] = {"rel_err": float(((y - t).norm(dim=-1) / t.norm(dim=-1).clamp_min(1e-6)).mean()),
                        "cos": float(torch.nn.functional.cosine_similarity(y, t, dim=-1).mean()),
                        "r2": float(1 - ((y - t) ** 2).sum() / ((t - t.mean(0)) ** 2).sum().clamp_min(1e-6))}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--layers", default="3,7,11,15")
    ap.add_argument("--level", choices=["pre", "post"], default="pre")
    ap.add_argument("--mode", choices=["replace", "add", "mix"], default="replace")
    ap.add_argument("--gate", type=float, default=1.0)
    ap.add_argument("--mix_w", type=float, default=0.5, help="mode mix: initial weight of the injected read")
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--maps", default=None, help="fit_encside_B.py output (enc_B per reader layer)")
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--fid", action="store_true")
    ap.add_argument("--add_layers", default="", help="these reader layers use mode add with --gate (others: --mode)")
    ap.add_argument("--no_mlp", action="store_true", help="ignore an MLP correction stored in --maps")
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--rgain", type=float, default=1.0, help="D12: scalar gain on the replaced output")
    ap.add_argument("--rgain_layers", default="", help="D12: layers that get --rgain (default: all replaced)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from eval_doc import wrong_doc_id
    from xenc.backbone import build_chat, load_encoder, load_reader
    from xenc.doc_tasks import ctx_prompt, load_docs, q_prompt, score
    from xenc.encside_read import EncCapture, EncSideInterface, EncSideReader
    from xenc.xattn_read import attn_dims
    conds = [c for c in CONDS if c in a.conds.split(",")]
    os.makedirs(a.out, exist_ok=True)
    model, tok = load_reader(a.reader, device_map=a.device)
    same = a.encoder == "same"
    enc = model if same else load_encoder(a.encoder, device_map=a.device)[0]
    lmap = layer_pairs(model, enc, [int(x) for x in a.layers.split(",")])
    ra, ea = attn_dims(model), attn_dims(enc)
    from xenc.state_skill import text_config
    d_in = ea["n_q"] * ea["hd"] if a.level == "pre" else int(text_config(enc).hidden_size)
    d_out = ra["n_q"] * ra["hd"] if a.level == "pre" else int(text_config(model).hidden_size)
    import math
    blob = {}
    for k, mp in enumerate(a.maps.split(",") if a.maps else []):    # D12: comma list = merge per-layer B
        b = torch.load(mp, map_location="cpu", weights_only=False)
        if k == 0:
            blob = b
        else:
            assert b.get("format") == blob.get("format") and b.get("level") == blob.get("level"), "cannot merge maps"
            key = "layers" if "layers" in b else "enc_B"
            blob[key] = {**blob[key], **{li: v for li, v in b[key].items() if li not in blob[key]}}
            if "layer_pairs" in b:
                blob["layer_pairs"] = {**b["layer_pairs"], **blob.get("layer_pairs", {})}
    use_mlp = bool(blob.get("mlp")) and not a.no_mlp
    ctx = blob.get("format") == "xenc.encside_B2.v1"
    if ctx:
        from xenc.encside_ctx import CtxEncSideInterface
        assert blob.get("level") == a.level, "maps level mismatch"
        itf = CtxEncSideInterface(blob, sorted(lmap), a.mode, gate_init=a.gate,
                                  mix_init=math.log(a.mix_w / max(1 - a.mix_w, 1e-6)))
        assert itf.layer_map == lmap, f"context B layer pairing {itf.layer_map} != depth map {lmap}"
    else:
        itf = EncSideInterface(lmap, d_in, d_out, a.mode, a.level, identity=a.identity or a.maps is None,
                               gate_init=a.gate, mix_init=math.log(a.mix_w / max(1 - a.mix_w, 1e-6)),
                               mlp_hidden=int(blob.get("mlp_hidden", 0)) if use_mlp else 0)
    if a.rgain != 1.0:
        itf.rgain = {int(li): a.rgain for li in (a.rgain_layers.split(",") if a.rgain_layers else itf.layers)}
    if a.add_layers:
        itf.layer_mode = {int(x): "add" for x in a.add_layers.split(",")}
    if use_mlp:
        for li in itf.layers:
            itf.M[str(li)].load_state_dict({k.replace("", ""): v for k, v in blob["mlp"][str(li)].items()})
    if a.maps and not ctx:
        assert blob.get("level", "pre") == a.level, "maps level mismatch"
        with torch.no_grad():
            for li in itf.layers:
                itf.B[str(li)].copy_(blob["enc_B"][str(li)].float())
    itf = itf.to(next(model.parameters()).device)
    R = EncSideReader(model, itf)
    cap = None if same else EncCapture(enc, getattr(itf, "enc_layers", sorted(set(lmap.values()))), a.level)
    if same and a.level == "post":
        raise SystemExit("L/L with --level post: use pre (the reader's own capture)")
    docs = load_docs(a.split_dir)
    items = load_split_items(a.split_dir, a.split)
    doc_order = sorted({_get(i, "doc_id") for i in items})
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
    if a.n and len(items) > a.n:
        items = random.Random(a.seed).sample(items, a.n)
    log(f"encoder-side read: {a.encoder} -> {a.reader} | layers {lmap} | level {a.level} mode {a.mode} | "
        f"B {'identity' if itf.identity else a.maps} | tail {a.tail} | {len(items)} items")
    rows, t0 = [], time.time()
    ck = open(os.path.join(a.out, f"eval_{a.split}.ckpt.jsonl"), "w")
    for it in items:
        did = _get(it, "doc_id")
        w = wrong_doc_id(did, doc_order)
        rec = {"key": _get(it, "key"), "doc_id": did, "wrong": w}
        for c in conds:
            if c == "closed_book":
                g, n = greedy(model, tok, a.reader, build_chat(tok, a.reader, SYSTEM_ANSWER, q_prompt(it)), a.budget)
            elif c == "text":
                g, n = greedy(model, tok, a.reader, build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, docs[did])), a.budget)
            else:
                d = w if c == "wrong_doc" else did
                pre, suf = full_ids(tok, a.reader, docs, it, d)
                pre = pre[-(a.max_tokens - len(suf)):]
                n_s = a.tail + len(suf)
                if c == "closed_tail":
                    g, n = greedy(model, tok, a.reader, (pre + suf)[-n_s:], a.budget)
                else:
                    g, n = gen_encside(model, tok, a.reader, enc, R, cap, same, lmap, pre + suf, n_s, a.budget)
                    if a.fid and c == "kv":
                        rec["fid"] = fidelity(model, enc, R, cap, same, lmap, pre + suf, n_s, pre + suf)
            rec[c] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
        rows.append(rec)
        ck.write(json.dumps(rec) + "\n")
        ck.flush()
        if len(rows) % 10 == 0:
            log(f"  [{len(rows)}/{len(items)}] " + " ".join(f"{c}={_m(rows, c):.3f}" for c in conds)
                + f" | {(time.time() - t0) / len(rows):.1f}s/item")
    summ = {c: {"score": _m(rows, c), "n": sum(1 for r in rows if c in r),
                "mean_gen_tokens": sum(r[c]["gen_tokens"] for r in rows if c in r) / max(len(rows), 1)} for c in conds}
    if "kv" in summ and "wrong_doc" in summ:
        pair = [r["kv"]["score"] - r["wrong_doc"]["score"] for r in rows]
        mu = sum(pair) / len(pair)
        sd = (sum((p - mu) ** 2 for p in pair) / max(len(pair) - 1, 1)) ** 0.5
        summ["kv_minus_wrong_doc"] = {"paired_gap": mu, "se": sd / len(pair) ** 0.5, "n_paired": len(pair),
                                      "wins": sum(p > 0 for p in pair), "losses": sum(p < 0 for p in pair)}
    if a.fid:
        f = [r["fid"] for r in rows if "fid" in r]
        summ["fidelity"] = {li: {k: sum(x[li][k] for x in f) / len(f) for k in ("rel_err", "cos", "r2") if k in f[0][li]}
                            for li in f[0]} if f else {}
    json.dump({"summary": summ, "args": vars(a), "items": rows}, open(os.path.join(a.out, f"eval_{a.split}.json"), "w"), indent=1)
    log(f"==== eval_encside [{a.split}] n={len(rows)} -> {a.out}")
    for c in conds:
        log(f"  {c:12s} {summ[c]['score']:.4f}  n={summ[c]['n']}  gen {summ[c]['mean_gen_tokens']:.1f}")
    if "kv_minus_wrong_doc" in summ:
        v = summ["kv_minus_wrong_doc"]
        log(f"  kv - wrong_doc: {v['paired_gap']:+.4f} ± {v['se']:.4f} (n={v['n_paired']}, {v['wins']}/{v['losses']})")
    if a.fid and summ.get("fidelity"):
        log("  fidelity " + " ".join(f"L{li}: rel {d['rel_err']:.3f} cos {d['cos']:.3f} r2 {d.get('r2', float('nan')):.3f}"
                                     for li, d in summ["fidelity"].items()))
    return 0


def _m(rows, c):
    v = [r[c]["score"] for r in rows if c in r]
    return sum(v) / len(v) if v else float("nan")


if __name__ == "__main__":
    sys.exit(main())
