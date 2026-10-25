"""CAD judge (agent CAD, 2026-10-23): context-aware decoding of the UNREGISTERED kv read (kv@0), inference only.

    logits_cad = (1 + a)·logits_kv − a·logits_ref        ref = closed_tail (same reader prompt, no operator)
                                                               or closed_book (chat(q_prompt), no document)
Formula borrowed from Hub Memory, hm_A_cad.py:32-43 (there: option-letter logp only). Mechanics in xenc/cad.py.

Per item (same item selection / wrong-doc rule as eval_d13 (--arch d13, 9B reader) or eval_d12p (--arch d12p, 27B)):
  plain generations, the exact eval_d13 / eval_d12p code paths (reproduce the kv@0 rows):
     kv, closed_tail, wrong_doc  = rt.gen(...);  closed_book = register_state.greedy (HF generate)
  text      the reader reading the document (manual greedy loop on a Stream; reference only)
  kv_cad_<ct|cb>_<a>     per-step CAD generation, kv vs closed_tail (ct) / closed_book (cb), for a in --cad_alphas
                         (a = 0 is run WITH the reference advancing: the interleaving check, must equal kv)
  text_cad_cb_<a>        text vs closed_book (reference row, --text 1)
  fc (MCQ items): option-letter log-softmax after "<answer>" for kv, closed_tail, closed_book, text, wrong_doc,
     wrong_tail (closed_tail of the wrong doc); CAD over letters is then a free CPU recompute (scripts/cad_table.py).
  timing: operator build, every stream prefill, plain kv gen, every CAD decode loop (latency overhead).
The reader never sees the document text in any kv / kv_cad condition. a is selected on VALIDATION splits by
scripts/cad_table.py; this script only records.

    python scripts/eval_cad.py --arch d13 --interface outputs/DS_AB_full/d13_best.pt --encoder Qwen/Qwen3.5-4B \
        --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --split_dir data/d25_fecf_split --split test --n 0 \
        --enc_device cuda:0 --device cuda:1 --out outputs/CAD_9b_fecf
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import string
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

ANS_PREFIX = "<answer>"
FC_AS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0)        # hm_A_cad.py:19 grid (log summary only)


# ----------------------------------------------------------------------------- NQ-Swap doc adherence
def _norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(a, b):
    a, b = _norm(a).split(), _norm(b).split()
    common = sum(min(a.count(w), b.count(w)) for w in set(a))
    if not a or not b or not common:
        return 0.0
    p, r = common / len(a), common / len(b)
    return 2 * p * r / (p + r)


def nqswap_adh(it, gen: str, doc_text: str) -> dict:
    """scripts/d12_nqswap_adh.py rule: prior_adh = F1(answer, prior) >= .5; doc_adh = not prior_adh and the
    normalised answer occurs in the asked passage (meta.span)."""
    m = re.search(r"<answer>(.*?)(</answer>|$)", gen or "", re.S)
    a = (m.group(1) if m else (gen or "")).strip()
    meta = _get(it, "meta") or {}
    s0, s1 = meta["span"]
    pa = max(_f1(a, p) for p in meta["prior_answers"]) >= 0.5
    da = (not pa) and bool(_norm(a)) and _norm(a) in _norm(doc_text[s0:s1])
    return {"doc_adh": float(da), "prior_adh": float(pa)}


def _ms(xs):
    n = len(xs)
    if not n:
        return float("nan"), float("nan")
    m = sum(xs) / n
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / max(n - 1, 1)) / math.sqrt(n))


def fc_correct(rec, main, ref, alpha) -> float | None:
    from xenc.cad import cad_pick
    fc = rec.get("fc")
    if not fc or main not in fc["lp"] or (alpha and ref not in fc["lp"]):
        return None
    i = cad_pick(fc["lp"][main], fc["lp"].get(ref, fc["lp"][main]), alpha)
    return float(fc["labels"][i] in fc["gold"])


def summarize(rows) -> dict:
    gconds = sorted({c for r in rows for c in r.get("gen", {})})
    out = {"gen": {}, "fc": {}}
    for c in gconds:
        xs = [r["gen"][c]["score"] for r in rows if c in r["gen"]]
        out["gen"][c] = {"score": _ms(xs)[0], "n": len(xs)}
        da = [r["gen"][c]["doc_adh"] for r in rows if c in r["gen"] and "doc_adh" in r["gen"][c]]
        if da:
            out["gen"][c]["doc_adh"] = _ms(da)[0]
        if c != "kv" and "kv" in gconds:
            p = [r["gen"][c]["score"] - r["gen"]["kv"]["score"] for r in rows if c in r["gen"] and "kv" in r["gen"]]
            out["gen"][c]["minus_kv"] = _ms(p)
    for main, ref in (("kv", "closed_tail"), ("kv", "closed_book"), ("text", "closed_book"),
                      ("wrong_doc", "wrong_tail")):
        cur = {}
        for a_ in FC_AS:
            xs = [x for x in (fc_correct(r, main, ref, a_) for r in rows) if x is not None]
            if xs:
                cur[str(a_)] = _ms(xs)[0]
        if cur:
            out["fc"][f"{main}|{ref}"] = cur
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arch", default="d13", choices=["d13", "d12p"], help="d13 = fit_d13 runtime (9B reader, live "
                    "encoder); d12p = fit_d12p runtime (27B reader, cached operators via --enc_cache)")
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--interface", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d22_set", default="")
    ap.add_argument("--wrong_map", default="")
    ap.add_argument("--variant", default="", help="keep only items with meta.variant == this (e.g. 'orig' for the "
                    "ConflictQA ORIGINAL bundles of data/d25_cqa_split)")
    ap.add_argument("--doc_first", type=int, default=0, help="keep only the first K sorted doc ids (the base's own "
                    "checkpoint-selection val docs, e.g. quality:20); applied before --n")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enc_device", default="")
    ap.add_argument("--share_encoder", type=int, default=0)
    ap.add_argument("--reader_mem", default=None)
    ap.add_argument("--enc_cache", default=None)
    ap.add_argument("--itf_device", default=None)
    ap.add_argument("--plain", default="kv,closed_tail,closed_book,wrong_doc", help="plain generation conds")
    ap.add_argument("--cad_alphas", default="0,0.25,0.5,1,2", help="per-step CAD alphas for kv ('' = no CAD gen)")
    ap.add_argument("--cad_refs", default="closed_tail,closed_book")
    ap.add_argument("--cad_plaus", type=float, default=0.0, help="adaptive plausibility cut (0 = off: pure CAD)")
    ap.add_argument("--text", type=int, default=1, help="1: text stream (text gen row, text FC, text+CAD reference)")
    ap.add_argument("--text_alphas", default="0.5,1,2", help="text+CAD (ref closed_book) generation alphas")
    ap.add_argument("--fc", type=int, default=1, help="1: forced-choice letter log-probs on MCQ items")
    ap.add_argument("--oom_skip", type=int, default=0, help="as eval_d13: record the OOM item, exit 75, resume")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from eval_doc import wrong_doc_id
    from register_state import _stop_ids, greedy
    from xenc.backbone import build_chat
    from xenc.cad import Stream, cad_decode, fc_letter_logp
    from xenc.doc_tasks import ctx_prompt, load_docs, q_prompt, score
    os.makedirs(a.out, exist_ok=True)
    if a.arch == "d13":
        from fit_d13 import D13Rt, build
        reader, enc, tok, itf, lmap = build(a, init=a.interface)
        rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens)
    else:
        from fit_d12p import build, make_rt
        a.enc_device = a.enc_device or None
        reader, enc, tok, itf, lmap = build(a, init=a.interface)
        rt = make_rt(a, reader, enc, tok, itf)
    alphas = [float(x) for x in a.cad_alphas.split(",") if x != ""]
    refs = [x for x in a.cad_refs.split(",") if x]
    assert set(refs) <= {"closed_tail", "closed_book"}, refs
    t_alphas = [float(x) for x in a.text_alphas.split(",") if x != ""] if a.text else []
    plain = [c for c in ("kv", "closed_tail", "closed_book", "wrong_doc") if c in a.plain.split(",")]
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
    if a.variant:
        items = [i for i in items if (_get(i, "meta") or {}).get("variant") == a.variant]
    if a.doc_first:
        keep = set(sorted({_get(i, "doc_id") for i in items})[: a.doc_first])
        items = [i for i in items if _get(i, "doc_id") in keep]
    if a.d22_set:
        keys = set(json.load(open(a.d22_set))["item_keys"])
        items = [i for i in items if _get(i, "key") in keys]
        assert len(items) == len(keys), (len(items), len(keys))
    elif a.n and len(items) > a.n:
        items = random.Random(a.seed).sample(items, a.n)
    stops = _stop_ids(tok)
    ansp = [int(x) for x in tok(ANS_PREFIX, add_special_tokens=False)["input_ids"]]
    letter = {L: int(tok(L, add_special_tokens=False)["input_ids"][0]) for L in "ABCDEFGHIJ"}
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    log(f"CAD eval ({a.arch}): {a.encoder} -> {a.reader} on {gpu} | layers {lmap} | interface {a.interface} | gates "
        f"{[round(float(x.detach()), 3) for x in itf.g]} | tail {a.tail} | {len(items)} items ({len({_get(i, 'doc_id') for i in items})} "
        f"docs) | plain {plain} | CAD a {alphas} refs {refs} plaus {a.cad_plaus} | text {a.text} {t_alphas} | fc {a.fc}")
    sync = (lambda: torch.cuda.synchronize()) if torch.cuda.is_available() else (lambda: None)   # noqa: E731

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
            log(f"  --oom_skip resume: {len(done)} items done, {len(oom_keys)} OOM items skipped")
    ck = open(ck_path, "a" if a.oom_skip else "w")
    oom_skipped: list = json.load(open(oom_path)) if a.oom_skip and os.path.exists(oom_path) else []
    order = sorted(range(len(items)), key=lambda k: _get(items[k], "doc_id"))
    res, rows, t0, op_t = {}, [], time.time(), {}

    def timed(f):
        sync()
        t = time.time()
        v = f()
        sync()
        return v, time.time() - t

    def operator(d):
        if d not in rt._ops:
            (o, n, _), dt = timed(lambda: rt.operator(d))
            op_t.setdefault(d, dt)
        return rt.operator(d)

    for k in order:
        it = items[k]
        did = _get(it, "doc_id")
        key = _get(it, "key")
        if key in done:
            res[k] = done[key]
            rows.append(res[k])
            continue
        if key in oom_keys:
            continue
        w = wmap[did] if did in wmap else wrong_doc_id(did, doc_order)
        rec = {"key": key, "doc_id": did, "wrong": w, "gen": {}, "t": {}}
        nq = bool((_get(it, "meta") or {}).get("answers_are_prior"))
        stage = "op"

        def put(c, g, n, extra=None):
            r_ = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
            if nq:
                r_.update(nqswap_adh(it, g, docs[did]))
            if extra:
                r_.update(extra)
            rec["gen"][c] = r_
        try:
            op, n_op, _ = operator(did)
            rec["t"]["op_build"] = op_t.get(did, 0.0)
            # ---- plain generations: the exact eval_d13 / eval_d12p code paths
            for c in plain:
                stage = c
                if c == "closed_book":
                    (g, n), dt = timed(lambda: greedy(reader, tok, a.reader,
                                                      build_chat(tok, a.reader, SYSTEM_ANSWER, q_prompt(it)), a.budget))
                elif c == "closed_tail":
                    (g, n), dt = timed(lambda: rt.gen(did, it, a.budget, read=False))
                elif c == "wrong_doc":
                    operator(w)
                    (g, n), dt = timed(lambda: rt.gen(w, it, a.budget))
                else:
                    operator(did)
                    (g, n), dt = timed(lambda: rt.gen(did, it, a.budget))
                put(c, g, n, {"t": dt})
            # ---- streams (prefilled once, forked per decode)
            stage = "streams"
            op, n_op, _ = operator(did)
            S = {"kv": Stream(rt, rt.rows(did, it), op, n_op, "kv"), "closed_tail": Stream(rt, rt.rows(did, it)),
                 "closed_book": Stream(rt, build_chat(tok, a.reader, SYSTEM_ANSWER, q_prompt(it)))}
            if a.text:
                S["text"] = Stream(rt, build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, docs[did])))
            for nm, s in S.items():
                _, rec["t"][f"pre_{nm}"] = timed(s.prefill)
            # ---- per-step CAD generation (kv main)
            stage = "cad"
            for ref in refs:
                rs = "ct" if ref == "closed_tail" else "cb"
                for al in alphas:
                    if al == 0 and ref != "closed_tail":
                        continue                          # one interleaving check is enough
                    r_ = cad_decode(S["kv"], S[ref], al, a.budget, tok, a.reader, stops, plaus=a.cad_plaus,
                                    force_ref=(al == 0))
                    put(f"kv_cad_{rs}_{al:g}", r_["gen"], r_["gen_tokens"], {"t_loop": r_["t_loop"]})
            if a.text:
                stage = "text"
                r_ = cad_decode(S["text"], None, 0.0, a.budget, tok, a.reader, stops)
                put("text", r_["gen"], r_["gen_tokens"], {"t_loop": r_["t_loop"]})
                for al in t_alphas:
                    r_ = cad_decode(S["text"], S["closed_book"], al, a.budget, tok, a.reader, stops,
                                    plaus=a.cad_plaus)
                    put(f"text_cad_cb_{al:g}", r_["gen"], r_["gen_tokens"], {"t_loop": r_["t_loop"]})
            # ---- forced choice (MCQ items)
            if a.fc and _get(it, "choices"):
                stage = "fc"
                labs = [str(c["label"]) for c in _get(it, "choices")]
                lids = [letter[x] for x in labs]
                meta = _get(it, "meta") or {}
                fc = {"labels": labs, "gold": list(meta.get("accept_labels") or [_get(it, "correct_label")]), "lp": {},
                      "t": {}}
                for nm, s in S.items():
                    (lp, dt) = timed(lambda: fc_letter_logp(s, ansp, lids))
                    fc["lp"][nm], fc["t"][nm] = lp, dt
                opw, nw, _ = operator(w)
                for nm, s in (("wrong_doc", Stream(rt, rt.rows(w, it), opw, nw)), ("wrong_tail", Stream(rt, rt.rows(w, it)))):
                    s.prefill()
                    fc["lp"][nm] = fc_letter_logp(s, ansp, lids)
                rec["fc"] = fc
            del S
        except torch.OutOfMemoryError:
            if not a.oom_skip:
                raise
            oom_skipped.append({"key": key, "doc_id": did, "cond": stage})
            json.dump(oom_skipped, open(oom_path, "w"), indent=1)
            ck.close()
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception as ex:                  # noqa: BLE001
                log(f"  (synchronize after OOM raised {type(ex).__name__})")
            log(f"  OOM on {key} ({did}, stage {stage}); EXIT 75 to resume in a fresh process (--oom_skip)")
            sys.exit(75)
        # interleaving check: kv_cad_ct_0 (reference advanced, a = 0) must equal the plain kv decode
        if "kv" in rec["gen"] and "kv_cad_ct_0" in rec["gen"]:
            rec["cad0_match"] = rec["gen"]["kv_cad_ct_0"]["gen"] == rec["gen"]["kv"]["gen"]
        res[k] = rec
        rows.append(rec)
        ck.write(json.dumps(rec) + "\n")
        ck.flush()
        if len(rows) % 20 == 0:
            s = summarize(rows)
            g = s["gen"]
            log(f"  [{len(rows)}/{len(items)}] " + " ".join(f"{c}={v['score']:.3f}" for c, v in g.items()
                                                            if c in ("kv", "text", "closed_tail") or c.startswith("kv_cad"))
                + " | fc kv|ct " + " ".join(f"{x}:{y:.3f}" for x, y in s["fc"].get("kv|closed_tail", {}).items())
                + f" | {(time.time() - t0) / len(rows):.1f}s/item")
    rows = [res[k] for k in range(len(items)) if k in res]
    summ = summarize(rows)
    mm = [r["cad0_match"] for r in rows if "cad0_match" in r]
    summ["cad0_match"] = {"n": len(mm), "match": sum(mm)}
    json.dump({"summary": summ, "args": vars(a), "gpu": gpu, "items": rows},
              open(os.path.join(a.out, f"eval_{a.split}.json"), "w"), indent=1)
    log(f"==== eval_cad [{a.split}] n={len(rows)} -> {a.out} | a=0 interleaving check: {sum(mm)}/{len(mm)} identical")
    for c, v in summ["gen"].items():
        x = f"  {c:18s} {v['score']:.4f}  n={v['n']}"
        if "doc_adh" in v:
            x += f"  doc_adh {v['doc_adh']:.4f}"
        if "minus_kv" in v:
            x += f"  (−kv {v['minus_kv'][0]:+.4f} ± {v['minus_kv'][1]:.4f})"
        log(x)
    for c, v in summ["fc"].items():
        log(f"  FC {c:24s} " + " ".join(f"a={x}:{y:.3f}" for x, y in v.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
