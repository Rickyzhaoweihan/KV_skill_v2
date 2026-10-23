"""D22 / BEAT_TEXT C1: the ENCODER model's own text accuracy (default Qwen3.5-4B reads the document as text), on
exactly the items D22 judged, with the same prompt (SYSTEM_ANSWER + ctx_prompt, full document), greedy decoding,
the same 32-token budget and the same scorer (xenc.doc_tasks.score: MCQ accuracy / span token-F1).
This is the encoder-information reference: kv through a 4B operator is not expected to exceed what the 4B itself
extracts from the text unless the reader adds reasoning on top.
One document-prefix prefill shared by every question of the document (exact: greedy continuation from the cache).

    python scripts/d22_c1_text4b.py --model Qwen/Qwen3.5-4B --out outputs/D22_C1   # resumable per benchmark
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--benches", default="", help="comma list of d22_summary.RUNS names (default all)")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--out", default="outputs/D22_C1")
    a = ap.parse_args(argv)
    from d22_summary import RUNS
    from register_doc import SYSTEM_ANSWER
    from register_state import greedy
    from xenc.backbone import build_chat, input_device, load_reader
    from xenc.doc_tasks import ctx_prompt, load_doc_split, load_docs, score
    os.makedirs(a.out, exist_ok=True)
    model, tok = load_reader(a.model, device_map="cuda")
    dev = input_device(model)
    names = [n for n in (a.benches.split(",") if a.benches else RUNS) if n]
    summ = {}
    for name in names:
        path, sd, fmt, _ = RUNS[name]
        if not Path(path).exists():
            continue
        tag = name.split(" ")[0].lower()
        fout = Path(a.out) / f"{tag}.jsonl"
        done = {json.loads(x)["key"]: json.loads(x) for x in open(fout)} if fout.exists() else {}
        E = json.load(open(path))
        keys = [r["key"] for r in E["items"]]
        meta = {it.key: it for sp in ("train", "val", "test") if (Path(sd) / sp / "items.json").exists()
                for it in load_doc_split(sd, sp)}
        docs = load_docs(sd)
        by = {}
        for k in keys:
            if k not in done:
                by.setdefault(meta[k].doc_id, []).append(meta[k])
        f = open(fout, "a")
        t0, nd = time.time(), 0
        for did, its in sorted(by.items()):
            seqs = [build_chat(tok, a.model, SYSTEM_ANSWER, ctx_prompt(it, docs[did])) for it in its]
            P = min(len(s) for s in seqs) - 1
            for s in seqs[1:]:
                P = min(P, next((i for i, (x, y) in enumerate(zip(seqs[0], s)) if x != y), P))
            with torch.no_grad():
                out = model(input_ids=torch.tensor([seqs[0][:P]], device=dev), use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
                del out
                for it, s in zip(its, seqs):
                    g, n = greedy(model, tok, a.model, s[P:], a.budget, cache=copy.deepcopy(cache))
                    rec = {"key": it.key, "doc_id": did, "gen": g, "score": float(score(it, g)), "gen_tokens": n}
                    f.write(json.dumps(rec) + "\n")
                    done[it.key] = rec
                del cache
            torch.cuda.empty_cache()
            nd += 1
            if nd % 10 == 0:
                print(f"  {name}: {nd}/{len(by)} docs ({(time.time() - t0) / nd:.1f}s/doc)", flush=True)
        f.close()
        sc = [done[k]["score"] for k in keys]
        summ[name] = {"n": len(sc), "text_enc": sum(sc) / len(sc), "format": fmt}
        print(f"{name}: {a.model} text = {summ[name]['text_enc']:.3f} (n={len(sc)})", flush=True)
    json.dump(summ, open(Path(a.out) / "summary.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
