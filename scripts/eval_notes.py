"""NOTES probe: unregistered kv@0 of a D13 base with and without encoder-written NOTE columns (xenc/notes.py).

Same items / prompts / scoring / greedy budget as scripts/eval_d13.py (its CONDS kv and wrong_doc are reproduced
exactly: note = None is the plain D13Rt). Extra conditions, all with the reader reading ONLY encoder K/V:
  n_<type>_<L>    own document + its own notes (type, first L generated tokens; L = 0: the note prompt only)
  wn_<type>_<L>   own document + the WRONG document's notes (same type / L), harvested in the own document's context
                  (wrong map = the set's wrong map, as wrong_doc)
Notes are generated question-independently by the encoder (greedy, all types of one document in one batch, up to
--max_new tokens) and stored under $XENC_CACHE/notes (xenc/notes.NoteStore), so later runs / training reuse them.
--cost_docs K: for the first K documents, time (fresh, synchronized) the encoder operator prefill, a batch-1 note
generation of --cost_type for --cost_len tokens, and the READER's text prefill of the full text prompt.

    python scripts/eval_notes.py --interface outputs/DS_AB_full/d13_best.pt --encoder Qwen/Qwen3.5-4B \
        --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --d22_set outputs/MCQ_sets/mcq_hallu.json ... --out outputs/NT_x
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


def cond_list(a) -> list[tuple[str, dict | None, bool]]:
    """-> [(name, note cfg or None, wrong_doc?)]."""
    out = []
    for c in a.base_conds.split(","):
        if c:
            assert c in ("kv", "wrong_doc"), c
            out.append((c, None, c == "wrong_doc"))
    types = [t for t in a.note_types.split(",") if t]
    for t in types:
        for L in [int(x) for x in a.note_lens.split(",") if x]:
            out.append((f"n_{t}_{L}", {"type": t, "L": L, "src": None}, False))
    if a.prompt_only:
        out.append((f"n_{types[0]}_0", {"type": types[0], "L": 0, "src": None}, False))
    for t in [t for t in a.wrong_note_types.split(",") if t]:
        out.append((f"wn_{t}_{a.wrong_note_len}", {"type": t, "L": int(a.wrong_note_len), "src": "WRONG"}, False))
    return out


def paired(rows, c, o):
    pair = [r[c]["score"] - r[o]["score"] for r in rows if c in r and o in r]
    if not pair:
        return None
    mu = sum(pair) / len(pair)
    sd = (sum((p - mu) ** 2 for p in pair) / max(len(pair) - 1, 1)) ** 0.5
    return {"paired_gap": mu, "se": sd / len(pair) ** 0.5, "n_paired": len(pair),
            "wins": sum(p > 0 for p in pair), "losses": sum(p < 0 for p in pair)}


def summarize(rows, names) -> dict:
    s = {}
    for c in names:
        sc = [r[c]["score"] for r in rows if c in r]
        s[c] = {"score": sum(sc) / max(len(sc), 1), "n": len(sc),
                "mean_gen_tokens": sum(r[c]["gen_tokens"] for r in rows if c in r) / max(len(sc), 1)}
        if c != "kv":
            s[c]["minus_kv"] = paired(rows, c, "kv")
    return s


@torch.no_grad()
def cost_probe(rt, reader, tok, a, did, it):
    """Fresh synchronized timings for one document (encoder prefill, batch-1 notes, reader text prefill)."""
    from xenc.backbone import build_chat
    from xenc.doc_tasks import ctx_prompt
    from xenc.notes import generate_notes

    def sync():
        for d in {rt.dev, rt.edev}:
            if d.type == "cuda":
                torch.cuda.synchronize(d)
    ids = rt.pre(did)[: -rt.tail]
    sync()
    t0 = time.time()
    out = rt.enc(input_ids=torch.tensor([ids], device=rt.edev), use_cache=True, logits_to_keep=1)
    sync()
    t1 = time.time()
    g = generate_notes(rt.enc, out.past_key_values, [rt.suffix[a.cost_type]], a.cost_len, {-1}, rt.pad_id)  # no stop
    sync()
    t2 = time.time()
    del out
    tids = build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, rt.docs[did]))
    sync()
    t3 = time.time()
    reader(input_ids=torch.tensor([tids], device=rt.dev), use_cache=True, logits_to_keep=1)
    sync()
    t4 = time.time()
    return {"doc": did, "n_enc_tokens": len(ids), "n_text_tokens": len(tids), "enc_prefill_s": t1 - t0,
            "notes_b1_s": t2 - t1, "notes_b1_tokens": len(g[0]["gen"]), "reader_text_prefill_s": t4 - t3}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--interface", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enc_device", default="")
    ap.add_argument("--share_encoder", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wrong_map", default=None)
    ap.add_argument("--d22_set", default="")
    ap.add_argument("--base_conds", default="kv,wrong_doc")
    ap.add_argument("--note_types", default="digest,facts,entities,reason,selfqa")
    ap.add_argument("--note_lens", default="64,128,256,512")
    ap.add_argument("--prompt_only", type=int, default=1, help="1: add n_<first type>_0 (note prompt columns only)")
    ap.add_argument("--wrong_note_types", default="digest,facts,entities,reason,selfqa")
    ap.add_argument("--wrong_note_len", type=int, default=256)
    ap.add_argument("--max_new", type=int, default=512, help="generated note budget (stored; L slices it)")
    ap.add_argument("--notes_dir", default="", help="NoteStore base (default $XENC_CACHE/notes)")
    ap.add_argument("--cost_docs", type=int, default=0)
    ap.add_argument("--cost_type", default="facts")
    ap.add_argument("--cost_len", type=int, default=256)
    ap.add_argument("--oom_skip", type=int, default=0, help="as eval_d13: record the OOM item, exit 75, resume on rerun")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from eval_doc import wrong_doc_id
    from fit_d13 import D13Rt, build
    from xenc.doc_tasks import load_docs, score
    from xenc.notes import build_note_rt
    os.makedirs(a.out, exist_ok=True)
    conds = cond_list(a)
    names = [c[0] for c in conds]
    types = sorted({c[1]["type"] for c in conds if c[1]}, key=lambda t: a.note_types.split(",").index(t)
                   if t in a.note_types.split(",") else 99)
    assert max([c[1]["L"] for c in conds if c[1]] + [0]) <= a.max_new
    reader, enc, tok, itf, lmap = build(a, init=a.interface)
    rt = build_note_rt(D13Rt, reader, enc, tok, a.reader, a.encoder, itf, a.tail, a.max_tokens, SYSTEM_ANSWER, types,
                       a.max_new, store_base=a.notes_dir or None)
    docs = load_docs(a.split_dir)
    rt.docs = docs
    items = load_split_items(a.split_dir, a.split)
    _sp = json.load(open(a.d22_set)).get("splits") if a.d22_set else None
    if isinstance(_sp, list) and _sp:
        items = [i for s_ in _sp for i in load_split_items(a.split_dir, s_)]
    doc_order = sorted({_get(i, "doc_id") for i in items})
    wmap = json.load(open(a.wrong_map)) if a.wrong_map else {}
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
    if a.d22_set:
        assert not a.n, "--d22_set fixes the sample: pass --n 0"
        keys = set(json.load(open(a.d22_set))["item_keys"])
        items = [i for i in items if _get(i, "key") in keys]
        assert len(items) == len(keys), (len(items), len(keys))
    if a.n and len(items) > a.n:
        items = random.Random(a.seed).sample(items, a.n)
    log(f"NOTES eval: {a.encoder} -> {a.reader} | layers {lmap} | interface {a.interface} | tail {a.tail} | "
        f"{len(items)} items | store {rt.store.root} | types {types} | conds {names}")
    for t in types:
        log(f"  note prompt [{t}] ({len(rt.suffix[t])} tokens): {tok.decode(rt.suffix[t])!r}")
    ck_path = os.path.join(a.out, f"eval_{a.split}.ckpt.jsonl")
    oom_path = os.path.join(a.out, "oom_skipped.json")
    done, oom_keys = {}, set()
    if a.oom_skip:
        if os.path.exists(ck_path):
            for line in open(ck_path):
                if line.strip():
                    r_ = json.loads(line)
                    done[r_["key"]] = r_
        if os.path.exists(oom_path):
            oom_keys = {x["key"] for x in json.load(open(oom_path))}
        if done or oom_keys:
            log(f"  --oom_skip resume: {len(done)} done, {len(oom_keys)} OOM-skipped")
    oom_skipped: list = json.load(open(oom_path)) if a.oom_skip and os.path.exists(oom_path) else []
    ck = open(ck_path, "a" if a.oom_skip else "w")
    cost_path = os.path.join(a.out, "cost.jsonl")
    n_cost = sum(1 for _ in open(cost_path)) if os.path.exists(cost_path) else 0
    order = sorted(range(len(items)), key=lambda k: _get(items[k], "doc_id"))
    res, rows, t0, seen = {}, [], time.time(), set()
    for k in order:
        it = items[k]
        did = _get(it, "doc_id")
        if _get(it, "key") in done:
            res[k] = done[_get(it, "key")]
            rows.append(res[k])
            continue
        if _get(it, "key") in oom_keys:
            continue
        w = wmap[did] if did in wmap else wrong_doc_id(did, doc_order)
        rec = {"key": _get(it, "key"), "doc_id": did, "wrong": w}
        c = None
        try:
            if a.cost_docs and n_cost < a.cost_docs and did not in seen:
                cr = cost_probe(rt, reader, tok, a, did, it)
                with open(cost_path, "a") as f:
                    f.write(json.dumps(cr) + "\n")
                n_cost += 1
                log(f"  [cost] {cr}")
            seen.add(did)
            for c, cfg, wrong in conds:
                if cfg is not None and cfg["src"] == "WRONG":
                    cfg = dict(cfg, src=w)
                rt.note = cfg
                g, n = rt.gen(w if wrong else did, it, a.budget)
                rec[c] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
            rt.note = None
        except torch.OutOfMemoryError:
            rt.note = None
            if not a.oom_skip:
                raise
            oom_skipped.append({"key": rec["key"], "doc_id": did, "cond": c})
            json.dump(oom_skipped, open(oom_path, "w"), indent=1)
            ck.close()
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception as ex:                  # noqa: BLE001
                log(f"  (synchronize after OOM raised {type(ex).__name__})")
            log(f"  OOM on {rec['key']} ({did}, cond {c}); EXIT 75 to resume (--oom_skip)")
            sys.exit(75)
        res[k] = rec
        rows.append(rec)
        ck.write(json.dumps(rec) + "\n")
        ck.flush()
        if rt.note_log:
            with open(os.path.join(a.out, "note_gen.jsonl"), "a") as f:
                for x in rt.note_log:
                    f.write(json.dumps(x) + "\n")
            rt.note_log.clear()
        if len(rows) % 20 == 0:
            s = summarize(rows, names)
            log(f"  [{len(rows)}/{len(items)}] " + " ".join(f"{c}={s[c]['score']:.3f}" for c in names)
                + f" | {(time.time() - t0) / len(rows):.1f}s/item")
    rows = [res[k] for k in range(len(items)) if k in res]
    summ = summarize(rows, names)
    json.dump({"summary": summ, "args": vars(a), "conds": names, "items": rows},
              open(os.path.join(a.out, f"eval_{a.split}.json"), "w"), indent=1)
    log(f"==== eval_notes [{a.split}] n={len(rows)} -> {a.out}")
    for c in names:
        v = summ[c].get("minus_kv")
        log(f"  {c:20s} {summ[c]['score']:.4f}  n={summ[c]['n']}  gen {summ[c]['mean_gen_tokens']:.1f}"
            + (f"  | − kv {v['paired_gap']:+.4f} ± {v['se']:.4f} ({v['wins']}/{v['losses']})" if v else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
