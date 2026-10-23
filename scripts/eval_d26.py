"""D26 judge: the reader-query read of the Qwen encoder's operator inside a reader of ANOTHER family, training-free at
test (base read = step 0 of registration). Conditions as scripts/eval_d13.py (greedy, budget, SYSTEM_ANSWER, same
items / seed / wrong-doc pairing):
  closed_book  chat(q_prompt)              closed_tail  [BOS] + doc tail + question suffix (reader template)
  text         chat(ctx_prompt)            kv           closed_tail prompt + the read of the own document's operator
  wrong_doc    another document's tail + operator
All chat prompts use the reader's own template (xenc.d26_read.chat_ids: gpt-oss forced final channel).
--interface none: only the read-free conditions (closed_book, closed_tail, text) — no B / interface needed.

    python scripts/eval_d26.py --reader glm4 --B outputs/D26_maps/B_glm4.pt --interface outputs/D26_glm4_reg/d13_best.pt \
        --split_dir data/longhealth_split --fold 0 --n 0 --out outputs/D26_glm4_lh
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

import torch  # noqa: E402,F401

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402

CONDS = ("closed_book", "closed_tail", "text", "kv", "wrong_doc")


def main(argv=None) -> int:
    from fit_d26 import add_common
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common(ap)
    ap.add_argument("--interface", required=True, help="d13-format file (fit_d26.py / register_d26.py) or 'none'")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--conds", default="closed_book,closed_tail,text,kv,wrong_doc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wrong_map", default=None)
    ap.add_argument("--resume", type=int, default=1, help="reuse finished conditions from eval_<split>.ckpt.jsonl")
    ap.add_argument("--reuse", default="", help="eval dir whose READ-FREE conditions (closed_book, closed_tail, text) "
                    "are reused by item key (same reader, items, budget, tail)")
    a = ap.parse_args(argv)
    from eval_d13 import summarize
    from eval_doc import wrong_doc_id
    from fit_d26 import D26Rt, build
    from xenc.d26_read import chat_ids, greedy_ids, load_reader_d26, resolve_hub
    from xenc.doc_tasks import ctx_prompt, load_docs, q_prompt, score
    conds = [c for c in CONDS if c in a.conds.split(",")]
    os.makedirs(a.out, exist_ok=True)
    no_read = a.interface == "none"
    if no_read:
        conds = [c for c in conds if c not in ("kv", "wrong_doc")]
        a.reader = resolve_hub(a.reader)
        reader, tok_r = load_reader_d26(a.reader, device=a.device)
        from xenc.backbone import load_encoder
        enc, tok_e = load_encoder(a.encoder, device_map=a.device)       # closed_tail cut = the encoder's tail
        import types as _t
        itf = _t.SimpleNamespace(layers=[], layer_map={})
        lmap = {}
    else:
        reader, enc, tok_r, tok_e, itf, lmap = build(a, init=a.interface)
    rt = D26Rt(reader, enc, tok_r, tok_e, a.reader, a.encoder, itf, a.tail, a.max_tokens, bos=bool(a.bos))
    docs = load_docs(a.split_dir)
    rt.docs = docs
    items = load_split_items(a.split_dir, a.split)
    doc_order = sorted({_get(i, "doc_id") for i in items})
    wmap = json.load(open(a.wrong_map)) if a.wrong_map else {}
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
    if a.n and len(items) > a.n:
        items = random.Random(a.seed).sample(items, a.n)
    gates = [round(float(x), 3) for x in itf.g] if not no_read else []
    log(f"D26 eval: {a.encoder} -> {a.reader} | layers {lmap} | interface {a.interface} | gates {gates} | tail "
        f"{a.tail} | bos {rt.bos!r} | {len(items)} items | conds {conds}")
    ck_path = os.path.join(a.out, f"eval_{a.split}.ckpt.jsonl")
    done: dict = {}

    def _merge(path, keep):
        n = 0
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            d = done.setdefault(r["key"], {k: r[k] for k in ("key", "doc_id", "wrong") if k in r})
            for c in keep:
                if c in r and c not in d:
                    d[c] = r[c]
                    n += 1
        return n
    if a.reuse:
        rp = os.path.join(a.reuse, f"eval_{a.split}.ckpt.jsonl")
        log(f"  reused {_merge(rp, ('closed_book', 'closed_tail', 'text'))} read-free results from {rp}")
    if a.resume and os.path.exists(ck_path):
        log(f"  resumed {_merge(ck_path, CONDS)} results from {ck_path}")
    ck = open(ck_path, "a")
    rows, t0 = [], time.time()
    order = sorted(range(len(items)), key=lambda k: _get(items[k], "doc_id"))
    res = {}
    for k in order:
        it = items[k]
        did = _get(it, "doc_id")
        w = wmap[did] if did in wmap else wrong_doc_id(did, doc_order)
        rec = {"key": _get(it, "key"), "doc_id": did, "wrong": w}
        prev = done.get(_get(it, "key"), {})
        for c in conds:
            if c in prev:
                rec[c] = prev[c]
                continue
            if c == "closed_book":
                g, n = greedy_ids(reader, tok_r, a.reader, chat_ids(tok_r, a.reader, SYSTEM_ANSWER, q_prompt(it)),
                                  a.budget, stops=rt.stops)
            elif c == "text":
                g, n = greedy_ids(reader, tok_r, a.reader,
                                  chat_ids(tok_r, a.reader, SYSTEM_ANSWER, ctx_prompt(it, docs[did])), a.budget,
                                  stops=rt.stops)
            elif c == "closed_tail":
                g, n = rt.gen(did, it, a.budget, read=False)
            else:
                g, n = rt.gen(w if c == "wrong_doc" else did, it, a.budget)
            rec[c] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
        res[k] = rec
        rows.append(rec)
        if not all(c in prev for c in conds):
            ck.write(json.dumps(rec) + "\n")
            ck.flush()
        if len(rows) % 20 == 0:
            s = summarize(rows, conds)
            log(f"  [{len(rows)}/{len(items)}] " + " ".join(f"{c}={s[c]['score']:.3f}" for c in conds)
                + f" | {(time.time() - t0) / len(rows):.1f}s/item")
    rows = [res[k] for k in range(len(items))]
    summ = summarize(rows, conds)
    json.dump({"summary": summ, "args": vars(a), "items": rows}, open(os.path.join(a.out, f"eval_{a.split}.json"), "w"),
              indent=1)
    log(f"==== eval_d26 [{a.split}] n={len(rows)} -> {a.out}")
    for c in conds:
        log(f"  {c:12s} {summ[c]['score']:.4f}  n={summ[c]['n']}  gen {summ[c]['mean_gen_tokens']:.1f}")
    for o in ("wrong_doc", "closed_tail", "text"):
        if f"kv_minus_{o}" in summ:
            v = summ[f"kv_minus_{o}"]
            log(f"  kv - {o}: {v['paired_gap']:+.4f} ± {v['se']:.4f} (n={v['n_paired']}, {v['wins']}/{v['losses']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
