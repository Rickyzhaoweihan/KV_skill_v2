"""Text-in-context size ladder (agent [D]): closed_book + text across reader sizes.

For each reader in --readers (loaded one at a time, freed before the next), score the same n items
of --split under `closed_book` (question only) and `text` (document in the prompt), greedy at the
same --budget. Results are MERGED into outputs/ladder/<task>.json keyed by reader short name, so
readers can be run in separate jobs:

    {"task", "split", "n", "budget", "readers": {short: {cond: {score, mean_gen_tokens,
     mean_prompt_tokens, n}, "items": [{key, cond: score, ...}]}}}

    python scripts/ladder.py --readers Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B,Qwen/Qwen3.5-27B \
        --split_dir data/quality_split --conds closed_book,text --n 200
    python scripts/ladder.py --tiny
"""

from __future__ import annotations

import argparse
import gc
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

from register_doc import _get, load_split_items, log, system_prompt, task_name, tiny_setup  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--readers", default="Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B,Qwen/Qwen3.5-27B")
    ap.add_argument("--split_dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--conds", default="closed_book,text")
    ap.add_argument("--n", type=int, default=200, help="0 = whole split")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--style", choices=["answer", "reason"], default="answer")
    ap.add_argument("--max_ctx_chars", type=int, default=None)
    ap.add_argument("--max_prompt_tokens", type=int, default=0, help="skip text items longer than this (0=no limit)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_mem", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="outputs/ladder")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    if not a.tiny and not a.split_dir:
        ap.error("--split_dir is required unless --tiny")
    return a


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main(argv=None) -> int:
    a = parse_args(argv)
    from xenc.backbone import build_chat, generate, load_reader, short_name
    from xenc.doc_tasks import ctx_prompt, q_prompt, score

    conds = [c.strip() for c in a.conds.split(",") if c.strip()]
    bad = [c for c in conds if c not in ("closed_book", "text")]
    if bad:
        raise SystemExit(f"ladder supports closed_book,text only (got {bad})")
    task = task_name(a.split_dir, a.tiny)
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"{task}.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    res.update(task=task, split=a.split, budget=a.budget, style=a.style)
    res.setdefault("readers", {})

    if a.tiny:
        torch.set_num_threads(1)
        tz = tiny_setup(a.out_dir)
        docs, items = tz["docs"], tz["test"]
        readers = ["tiny-a", "tiny-b"]
        a.budget = min(a.budget, 4)
    else:
        from xenc.doc_tasks import load_docs
        docs = load_docs(a.split_dir)
        items = load_split_items(a.split_dir, a.split)
        readers = [r.strip() for r in a.readers.split(",") if r.strip()]
    if a.n and len(items) > a.n:
        items = random.Random(a.seed).sample(items, a.n)
    res["n"] = len(items)
    sysp = system_prompt(a.style)

    for ri, hub in enumerate(readers):
        if a.tiny:
            sys.path.insert(0, str(_ROOT / "tests"))
            from _tiny import TinyLM
            model = TinyLM(d=32 + 16 * ri, n_layers=10, seed=ri).eval().requires_grad_(False)
            tok = tz["tok"]
            name = hub
        else:
            model, tok = load_reader(hub, device_map=a.device, max_mem=a.max_mem)
            name = short_name(hub)
        t0 = time.time()
        recs = []
        for k, it in enumerate(items):
            did = _get(it, "doc_id")
            rec = {"key": _get(it, "key")}
            for c in conds:
                if c == "text" and did not in docs:
                    continue
                user = ctx_prompt(it, docs[did], a.max_ctx_chars) if c == "text" else q_prompt(it)
                ids = build_chat(tok, "tiny" if a.tiny else hub, sysp, user)
                if a.max_prompt_tokens and len(ids) > a.max_prompt_tokens:
                    rec[c] = {"skip": "too long", "prompt_tokens": len(ids)}
                    continue
                g, ng = generate(model, tok, "tiny" if a.tiny else hub, ids, a.budget, return_n=True)
                rec[c] = {"score": float(score(it, g)), "gen_tokens": ng, "prompt_tokens": len(ids),
                          "gen": g[:400]}
            recs.append(rec)
            if (k + 1) % 25 == 0:
                log(f"  [{name} {k + 1}/{len(items)}] " + " ".join(
                    f"{c}={_mean([r[c]['score'] for r in recs if 'score' in r.get(c, {})]):.3f}" for c in conds)
                    + f" | {(time.time() - t0) / (k + 1):.1f}s/item")
        entry = {"hub": hub, "items": recs}
        for c in conds:
            ok = [r[c] for r in recs if "score" in r.get(c, {})]
            entry[c] = {"score": _mean([x["score"] for x in ok]), "mean_gen_tokens": _mean([x["gen_tokens"] for x in ok]),
                        "mean_prompt_tokens": _mean([x["prompt_tokens"] for x in ok]), "n": len(ok)}
        res["readers"][name] = entry
        with open(path + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(path + ".tmp", path)
        log(f"{name}: " + " ".join(f"{c}={entry[c]['score']:.4f} (n={entry[c]['n']})" for c in conds)
            + f" | {time.time() - t0:.0f}s")
        if torch.cuda.is_available():
            log(f"  max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    log(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
