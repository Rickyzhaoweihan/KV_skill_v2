"""D14: accuracy vs bytes of the SHIPPED encoder operator in the cross-model encoder-side read.

Setting = D10's passing read (scripts/eval_encside.py): encoder (2B) memory of the document -> the encoder processes
[doc tail; question] over it -> its post-o_proj attention outputs, mapped by ridge B, replace the reader's (9B) at
reader layers --layers; the reader reads only the tail (--tail tokens) + question.
The shipped operator = the encoder's memory of the document prefix WITHOUT the tail (full-attention K/V + DeltaNet
recurrent + conv states); the tail is plain text both sides have. With no compression this is exactly D10's read
(prefix caching). Each condition compresses the operator (xenc/cache_compress.py grammar: acc25, rand10,
late:11+15+19+23_acc10, q8_s8, q4g64_s4, conv, nostate, sc0.5, ...). wrong_doc = another document's operator
compressed the same way (and its tail in both prompts).

Per condition: kv, wrong_doc, gen tokens, kv − wrong_doc ± se (paired), kv − closed_tail ± se (paired), and the MiB
shipped per document: `needed` = layers ≤ the deepest encoder layer the read uses (deeper layers cannot affect the
captured outputs), `all` = the whole encoder cache.

    python scripts/eval_encside_prune.py --maps outputs/D10_maps/encB8_9b_2b_post.pt --layers 3,7,11,15,19 \
        --split_dir data/longhealth_split --fold 0 --n 0 --conds full,acc25,rand25 --out outputs/D14_x
"""

from __future__ import annotations

import argparse
import contextlib
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

MIB = 2 ** 20


def layer_pairs(reader, enc, layers):
    from xenc.xattn_read import attn_dims, full_depth_map
    m = full_depth_map(attn_dims(reader), attn_dims(enc))
    return {k: v for k, v in m.items() if k in set(layers)}


@torch.no_grad()
def gen_from_operator(model, tok, hub, enc, R, cap, lmap, cache, q_ids, budget, ctx, stop="</answer>"):
    """Encoder continues `cache` over q_ids (tail + question) and steps along every generated token (under ctx,
    e.g. a ColumnBudget mask); the reader reads q_ids with the mapped encoder outputs replacing its own."""
    from xenc.backbone import decode, input_device
    dev, edev = input_device(model), input_device(enc)
    with ctx:
        cap.start()
        enc(input_ids=torch.tensor([q_ids], device=edev), past_key_values=cache, use_cache=True, logits_to_keep=1)
        o2 = cap.get()
        R.cur = {li: o2[e] for li, e in lmap.items()}
        out = model(input_ids=torch.tensor([q_ids], device=dev), use_cache=True, logits_to_keep=1)
        stops, new = _stop_ids(tok), []
        for _ in range(budget):
            nxt = int(out.logits[0, -1].argmax())
            new.append(nxt)
            if nxt in stops or stop in tok.decode(new[-12:], skip_special_tokens=True):
                break
            cap.start()
            enc(input_ids=torch.tensor([[nxt]], device=edev), past_key_values=cache, use_cache=True, logits_to_keep=1)
            o2 = cap.get()
            R.cur = {li: o2[e] for li, e in lmap.items()}
            out = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=out.past_key_values,
                        use_cache=True, logits_to_keep=1)
    R.cur = None
    txt = decode(tok, hub, new)
    if stop in txt:
        txt = txt.split(stop, 1)[0] + stop
    return txt, len(new)


def _paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    mu = sum(d) / len(d)
    sd = (sum((p - mu) ** 2 for p in d) / max(len(d) - 1, 1)) ** 0.5
    return mu, sd / len(d) ** 0.5, sum(p > 0 for p in d), sum(p < 0 for p in d)


def summarize(rows, conds):
    out = {}
    if rows and "closed_tail" in rows[0]:
        out["closed_tail"] = {"score": sum(r["closed_tail"]["score"] for r in rows) / len(rows),
                              "mean_gen_tokens": sum(r["closed_tail"]["gen_tokens"] for r in rows) / len(rows)}
    full = [r["conds"]["full"]["kv"]["score"] for r in rows if "full" in r["conds"]]
    for c in conds:
        rr = [r for r in rows if c in r["conds"]]
        if not rr:
            continue
        kv = [r["conds"][c]["kv"]["score"] for r in rr]
        wr = [r["conds"][c]["wrong_doc"]["score"] for r in rr]
        d = {"n": len(rr), "kv": sum(kv) / len(kv), "wrong_doc": sum(wr) / len(wr),
             "gen_kv": sum(r["conds"][c]["kv"]["gen_tokens"] for r in rr) / len(rr),
             "gen_wrong": sum(r["conds"][c]["wrong_doc"]["gen_tokens"] for r in rr) / len(rr)}
        g = _paired(kv, wr)
        d["gap"], d["gap_se"], d["wins"], d["losses"] = g
        if "closed_tail" in rr[0]:
            t = _paired(kv, [r["closed_tail"]["score"] for r in rr])
            d["minus_tail"], d["minus_tail_se"] = t[0], t[1]
        if len(full) == len(rr) and c != "full":
            f = _paired(kv, full)
            d["minus_full"], d["minus_full_se"] = f[0], f[1]
        for k in ("needed", "all"):
            for part in ("kv", "recurrent", "conv", "total"):
                d[f"mib_{k}_{part}"] = sum(r["conds"][c]["bytes"][k][part] for r in rr) / len(rr) / MIB
        out[c] = d
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--maps", required=True)
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--conds", default="full,acc25,rand25")
    ap.add_argument("--no_closed_tail", action="store_true")
    ap.add_argument("--check", action="store_true", help="first item: prefix-cache read == D10 single-prefill read")
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from eval_doc import wrong_doc_id
    from xenc.backbone import load_encoder, load_reader
    from xenc.cache_compress import compress_memory, parse_spec, shipped_bytes
    from xenc.doc_tasks import load_docs, score
    from xenc.encside_read import EncCapture, EncSideInterface, EncSideReader
    from xenc.kv_prune import ColumnBudget, harvest_with_importance
    from xenc.native_state import build_prefix_cache
    from xenc.state_skill import doc_prefix_text, question_suffix_ids, text_config
    conds = [c for c in a.conds.split(",") if c]
    specs = {c: parse_spec(c) for c in conds}
    os.makedirs(a.out, exist_ok=True)
    model, tok = load_reader(a.reader, device_map=a.device)
    enc = load_encoder(a.encoder, device_map=a.device)[0]
    lmap = layer_pairs(model, enc, [int(x) for x in a.layers.split(",")])
    needed_upto = max(lmap.values())
    blob = torch.load(a.maps, map_location="cpu", weights_only=False)
    assert blob.get("level", "pre") == "post", "D14 uses post-level maps"
    itf = EncSideInterface(lmap, int(text_config(enc).hidden_size), int(text_config(model).hidden_size),
                           "replace", "post")
    with torch.no_grad():
        for li in itf.layers:
            itf.B[str(li)].copy_(blob["enc_B"][str(li)].float())
    itf = itf.to(next(model.parameters()).device)
    R = EncSideReader(model, itf)
    cap = EncCapture(enc, sorted(set(lmap.values())), "post")
    docs = load_docs(a.split_dir)
    items = load_split_items(a.split_dir, a.split)
    doc_order = sorted({_get(i, "doc_id") for i in items})
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
    if a.n and len(items) > a.n:
        items = random.Random(a.seed).sample(items, a.n)
    log(f"D14 encoder-side operator compression: {a.encoder} -> {a.reader} | layers {lmap} (needed enc layers ≤ "
        f"{needed_upto}) | tail {a.tail} | {len(items)} items | {len(conds)} conds: {conds}")

    mem: dict = {}                                          # doc_id -> (st on CPU, imp on CPU, prefix ids, tail ids)

    def get_mem(did):
        if did not in mem:
            pre = [int(x) for x in tok(doc_prefix_text(tok, a.reader, docs[did], SYSTEM_ANSWER),
                                       add_special_tokens=False)["input_ids"]]
            if len(pre) > a.max_tokens - 512:
                log(f"  doc {did}: {len(pre)} tokens truncated to {a.max_tokens - 512}")
                pre = pre[-(a.max_tokens - 512):]
            body, tail = pre[:-a.tail], pre[-a.tail:]
            st, imp = harvest_with_importance(enc, tok, ids=torch.tensor([body]))
            st = {**st, "kv": {li: (K.cpu(), V.cpu()) for li, (K, V) in st["kv"].items()},
                  "recurrent": {li: s.cpu() for li, s in st["recurrent"].items()},
                  "conv": {li: s.cpu() for li, s in st["conv"].items()}}
            mem[did] = (st, {li: v.cpu() for li, v in imp.items()}, pre, tail)
            torch.cuda.empty_cache()
        return mem[did]

    def on_dev(st, imp):
        dev = next(enc.parameters()).device
        return ({**st, "kv": {li: (K.to(dev), V.to(dev)) for li, (K, V) in st["kv"].items()},
                 "recurrent": {li: s.to(dev) for li, s in st["recurrent"].items()},
                 "conv": {li: s.to(dev) for li, s in st["conv"].items()}},
                {li: v.to(dev) for li, v in imp.items()})

    def run_cond(c, did, it):
        sp = specs[c]
        st, imp, pre, tail = get_mem(did)
        st, imp = on_dev(st, imp)
        st_c, info = compress_memory(enc, st, imp, sp, seed=doc_order.index(did) if did in doc_order else 0)
        cache = build_prefix_cache(enc, st_c, keep_kv=True, keep_state=sp.keep_state, keep_conv=sp.keep_conv,
                                   state_scale=sp.state_scale)
        ctx = ColumnBudget(enc, info["col_budget"], info["b_max"]) if info["masked"] else contextlib.nullcontext()
        q_ids = tail + question_suffix_ids(tok, a.reader, it, SYSTEM_ANSWER)
        g, n = gen_from_operator(model, tok, a.reader, enc, R, cap, lmap, cache, q_ids, a.budget, ctx)
        nb = {"needed": shipped_bytes(st, info, sp, needed_upto), "all": shipped_bytes(st, info, sp, None)}
        del st, st_c, cache
        return {"gen": g, "score": float(score(it, g)), "gen_tokens": n}, nb, info

    ck_path = os.path.join(a.out, f"eval_{a.split}.ckpt.jsonl")
    rows, done = [], set()
    if a.resume and os.path.exists(ck_path):
        for line in open(ck_path):
            r = json.loads(line)
            rows.append(r)
            done.add(r["key"])
        log(f"  resumed {len(rows)} items")
    ck = open(ck_path, "a" if a.resume else "w")
    t0 = time.time()
    n_new = 0
    for it in items:
        key = _get(it, "key")
        if key in done:
            continue
        did = _get(it, "doc_id")
        w = wrong_doc_id(did, doc_order)
        rec = {"key": key, "doc_id": did, "wrong": w, "conds": {}}
        if a.check and n_new == 0:
            rec["check"] = check_equivalence(model, enc, R, cap, lmap, get_mem(did), tok, a, it, on_dev)
            log(f"  check (prefix cache vs single prefill, encoder outputs at q rows): {rec['check']}")
        if not a.no_closed_tail:
            st, imp, pre, tail = get_mem(did)
            ids = (pre + question_suffix_ids(tok, a.reader, it, SYSTEM_ANSWER))
            n_s = a.tail + len(ids) - len(pre)
            g, n = greedy(model, tok, a.reader, ids[-n_s:], a.budget)
            rec["closed_tail"] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
        for c in conds:
            kv, nb, info = run_cond(c, did, it)
            wr, _, _ = run_cond(c, w, it)
            rec["conds"][c] = {"kv": kv, "wrong_doc": wr, "bytes": nb,
                               "b_max": info["b_max"], "n_doc": info["n_doc_full"]}
        rows.append(rec)
        n_new += 1
        ck.write(json.dumps(rec) + "\n")
        ck.flush()
        if len(rows) % 10 == 0:
            s = summarize(rows, conds)
            log(f"  [{len(rows)}/{len(items)}] tail={s.get('closed_tail', {}).get('score', float('nan')):.3f} "
                + " ".join(f"{c}={s[c]['kv']:.3f}/{s[c]['wrong_doc']:.3f}" for c in conds)
                + f" | {(time.time() - t0) / max(n_new, 1):.1f}s/item")
    summ = summarize(rows, conds)
    json.dump({"summary": summ, "args": vars(a), "specs": {c: specs[c].to_dict() for c in conds}, "items": rows},
              open(os.path.join(a.out, f"eval_{a.split}.json"), "w"), indent=1)
    log(f"==== D14 eval_encside_prune [{a.split}] n={len(rows)} -> {a.out}")
    if "closed_tail" in summ:
        log(f"  closed_tail {summ['closed_tail']['score']:.4f} gen {summ['closed_tail']['mean_gen_tokens']:.1f}")
    for c in conds:
        d = summ[c]
        log(f"  {c:28s} kv {d['kv']:.4f} wrong {d['wrong_doc']:.4f} gap {d['gap']:+.4f}±{d['gap_se']:.4f} "
            f"({d['wins']}/{d['losses']}) -tail {d.get('minus_tail', float('nan')):+.4f} "
            f"-full {d.get('minus_full', 0.0):+.4f} gen {d['gen_kv']:.1f}/{d['gen_wrong']:.1f} "
            f"MiB needed {d['mib_needed_total']:.2f} all {d['mib_all_total']:.2f}")
    return 0


@torch.no_grad()
def check_equivalence(model, enc, R, cap, lmap, m, tok, a, it, on_dev):
    """Encoder outputs at the q rows: one prefill of [doc; question] (D10) vs prefix cache + continuation (D14)."""
    from xenc.backbone import input_device
    from xenc.native_state import build_prefix_cache
    from xenc.state_skill import question_suffix_ids
    st, imp, pre, tail = m
    st, _ = on_dev(st, imp)
    suf = question_suffix_ids(tok, a.reader, it, SYSTEM_ANSWER)
    q = tail + suf
    dev = input_device(enc)
    cap.start()
    enc(input_ids=torch.tensor([pre + suf], device=dev), use_cache=True, logits_to_keep=1)
    ref = cap.get()
    cap.start()
    enc(input_ids=torch.tensor([q], device=dev), past_key_values=build_prefix_cache(enc, st), use_cache=True,
        logits_to_keep=1)
    got = cap.get()
    res = {}
    for e in sorted(set(lmap.values())):
        x, y = ref[e][0, -len(q):].float(), got[e][0].float()
        res[str(e)] = {"cos": float(torch.nn.functional.cosine_similarity(x, y, dim=-1).mean()),
                       "rel": float(((x - y).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6)).mean())}
    return res


if __name__ == "__main__":
    sys.exit(main())
