"""D16 judge: the generalised reader-query read (xenc/d16_read.py) of the encoder's fixed operator, training-free at
test. Same conditions / items / seed as scripts/eval_d13.py (closed_book, closed_tail, text, kv, wrong_doc): the reader
prompt is its own 64-token doc tail + question; kv / wrong_doc replace the attention output at the read layers by the
D16 read of the own / another document's operator (the encoder only prefilled the document, never sees the question).

    python scripts/eval_d16.py --interface outputs/D16_reg_x/d16_best.pt --split_dir data/longhealth_split --fold 0 --n 0 \
        --out outputs/D16_x_lh
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

CONDS = ("closed_book", "closed_tail", "text", "kv", "wrong_doc")


def enc_layers16(lmap: dict, cfg: dict) -> list:
    """Encoder layers whose operator must be harvested: the paired layers plus every kcat layer (lists)."""
    out = set(int(v) for v in lmap.values())
    for v in (cfg.get("kcat") or {}).values():
        out |= {int(x) for x in (v if isinstance(v, (list, tuple)) else [v])}
    return sorted(out)


def build16(a, path):
    """reader, encoder, tok, D16 interface (from `path`), Rt16 runtime with a D16Reader bound to the interface."""
    from fit_d16 import Rt16
    from xenc.backbone import load_encoder, load_reader
    from xenc.d13_read import load_B
    from xenc.d16_read import D16Reader, load_d16
    reader, tok = load_reader(a.reader, device_map=a.device)
    enc = load_encoder(a.encoder, device_map=a.device)[0]
    blob = torch.load(path, map_location="cpu", weights_only=False)
    lmap = {int(k): int(v) for k, v in blob["config"]["layer_map"].items()}
    B = load_B(a.B, sorted(lmap))
    itf, extra = load_d16(path, enc, B)
    from xenc.backbone import input_device
    itf = itf.to(input_device(reader))
    elayers = enc_layers16(lmap, itf.cfg)
    rt = Rt16(reader, enc, tok, a.reader, lmap, sorted(elayers), a.tail, a.max_tokens,
              x_layers=sorted(set(lmap.values())) if itf.cfg["keymap"] else ())
    rt.R = D16Reader(reader, itf)
    return reader, enc, tok, itf, lmap, rt, extra


@torch.no_grad()
def gen16(rt, did, it, budget: int = 32, stop: str = "</answer>", read: bool = True):
    from register_state import _stop_ids
    from xenc.backbone import decode
    rows = rt.rows(did, it)
    R = rt.R
    if read:
        op, n_op, _ = rt.operator(did)
        R.load(op, n_op)
        R.mode = "student"
    try:
        out = rt.reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=True, logits_to_keep=1)
        stops, new = _stop_ids(rt.tok), []
        for _ in range(budget):
            nxt = int(out.logits[0, -1].argmax())
            new.append(nxt)
            if nxt in stops or stop in rt.tok.decode(new[-12:], skip_special_tokens=True):
                break
            out = rt.reader(input_ids=torch.tensor([[nxt]], device=rt.dev), past_key_values=out.past_key_values,
                            use_cache=True, logits_to_keep=1)
    finally:
        R.mode, R.past = "off", {}
    txt = decode(rt.tok, rt.hub, new)
    if stop in txt:
        txt = txt.split(stop, 1)[0] + stop
    return txt, len(new)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--B", default="outputs/D10_maps/encB8_9b_2b_post.pt")
    ap.add_argument("--interface", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--conds", default="closed_tail,text,kv,wrong_doc")
    ap.add_argument("--tau", type=float, default=None, help="fixed read temperature at test (sweep)")
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from eval_d13 import summarize
    from eval_doc import wrong_doc_id
    from register_state import greedy
    from xenc.backbone import build_chat
    from xenc.doc_tasks import ctx_prompt, load_docs, q_prompt, score
    conds = [c for c in CONDS if c in a.conds.split(",")]
    os.makedirs(a.out, exist_ok=True)
    reader, enc, tok, itf, lmap, rt, _ = build16(a, a.interface)
    rt.R.tau, rt.R.topk = a.tau, a.topk
    docs = load_docs(a.split_dir)
    rt.docs = docs
    items = load_split_items(a.split_dir, a.split)
    doc_order = sorted({_get(i, "doc_id") for i in items})
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
    if a.n and len(items) > a.n:
        items = random.Random(a.seed).sample(items, a.n)
    log(f"D16 eval: {a.encoder} -> {a.reader} | layers {lmap} | cfg {itf.cfg} | interface {a.interface} | gates "
        f"{[round(float(x), 3) for x in itf.g]} | tau {a.tau} topk {a.topk} | tail {a.tail} | {len(items)} items")
    rows, t0 = [], time.time()
    ck = open(os.path.join(a.out, f"eval_{a.split}.ckpt.jsonl"), "w")
    order = sorted(range(len(items)), key=lambda k: _get(items[k], "doc_id"))
    res = {}
    for k in order:
        it = items[k]
        did = _get(it, "doc_id")
        w = wrong_doc_id(did, doc_order)
        rec = {"key": _get(it, "key"), "doc_id": did, "wrong": w}
        for c in conds:
            if c == "closed_book":
                g, n = greedy(reader, tok, a.reader, build_chat(tok, a.reader, SYSTEM_ANSWER, q_prompt(it)), a.budget)
            elif c == "text":
                g, n = greedy(reader, tok, a.reader, build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, docs[did])), a.budget)
            elif c == "closed_tail":
                g, n = gen16(rt, did, it, a.budget, read=False)
            else:
                g, n = gen16(rt, w if c == "wrong_doc" else did, it, a.budget)
            rec[c] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
        res[k] = rec
        rows.append(rec)
        ck.write(json.dumps(rec) + "\n")
        ck.flush()
        if len(rows) % 20 == 0:
            s = summarize(rows, conds)
            log(f"  [{len(rows)}/{len(items)}] " + " ".join(f"{c}={s[c]['score']:.3f}" for c in conds)
                + f" | {(time.time() - t0) / len(rows):.1f}s/item")
    rows = [res[k] for k in range(len(items))]
    summ = summarize(rows, conds)
    if "kv" in conds and "text" in conds:                 # text-parity judge: paired kv − text
        pair = [r["kv"]["score"] - r["text"]["score"] for r in rows]
        mu = sum(pair) / len(pair)
        sd = (sum((p - mu) ** 2 for p in pair) / max(len(pair) - 1, 1)) ** 0.5
        summ["kv_minus_text"] = {"paired_gap": mu, "se": sd / len(pair) ** 0.5, "n_paired": len(pair),
                                 "wins": sum(p > 0 for p in pair), "losses": sum(p < 0 for p in pair)}
    json.dump({"summary": summ, "args": vars(a), "items": rows}, open(os.path.join(a.out, f"eval_{a.split}.json"), "w"), indent=1)
    log(f"==== eval_d16 [{a.split}] n={len(rows)} -> {a.out}")
    for c in conds:
        log(f"  {c:12s} {summ[c]['score']:.4f}  n={summ[c]['n']}  gen {summ[c]['mean_gen_tokens']:.1f}")
    for o in ("wrong_doc", "closed_tail", "text"):
        if f"kv_minus_{o}" in summ:
            v = summ[f"kv_minus_{o}"]
            log(f"  kv - {o}: {v['paired_gap']:+.4f} ± {v['se']:.4f} (n={v['n_paired']}, {v['wins']}/{v['losses']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
