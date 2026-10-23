"""D15 step 2: label-free questions for every pool document, written by the 9B READER itself (HF, no vLLM in the
env). One shared document prefill per doc; the cache is then fanned out to --streams sampled continuations that
differ only in a one-digit "focus on part k of S" hint (equal token length -> no padding). Each stream writes
2 short-answer questions + 1 four-option multiple-choice question; parsed, de-duplicated, at most --keep per doc.
Answers are stored but stage 1 (fit_d15.py) never uses them.

Output: data/D15_selfq/selfq/shard<i>.jsonl, one line per document {"doc_id", "items": [DocItem dicts], "n_raw"}.
Resumable (documents already present in any shard file are skipped).

    python scripts/d15_selfq.py --pool data/D15_selfq --shard 0 --nshard 2 --device cuda:0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

SYSTEM = "You write precise reading-comprehension questions about a document."
TASK = ("Write 3 questions that can only be answered by reading this document. Split the document into {S} equal "
        "parts and take all 3 questions from part {k}. Each question asks about a different specific detail (a name, "
        "number, date, place, event, reason or relation). Questions 1 and 2 are short-answer; question 3 is "
        "multiple-choice with four options, exactly one correct. Use exactly this format and nothing else:\n"
        "Q1: <question>\nA1: <short answer>\nQ2: <question>\nA2: <short answer>\nQ3: <question>\n"
        "A) <option>\nB) <option>\nC) <option>\nD) <option>\nA3: <letter>")


def parse(txt: str) -> list[dict]:
    out = []
    for n in (1, 2):
        m = re.search(rf"Q{n}:\s*(.+?)\s*\n\s*A{n}:\s*(.+?)\s*(?:\n|$)", txt)
        if m:
            out.append({"question": m.group(1).strip(), "answers": [m.group(2).strip()], "choices": None,
                        "correct_label": None})
    m = re.search(r"Q3:\s*(.+?)\s*\n\s*A\)\s*(.+?)\s*\n\s*B\)\s*(.+?)\s*\n\s*C\)\s*(.+?)\s*\n\s*D\)\s*(.+?)\s*\n\s*A3:\s*\(?([ABCD])",
                  txt, re.S)
    if m:
        ch = [{"label": L, "text": m.group(i + 2).strip()} for i, L in enumerate("ABCD")]
        lab = m.group(6)
        out.append({"question": m.group(1).strip(), "answers": [ch["ABCD".index(lab)]["text"]], "choices": ch,
                    "correct_label": lab})
    return [q for q in out if 12 <= len(q["question"]) <= 400 and all(len(a) < 300 for a in q["answers"])]


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


@torch.no_grad()
def gen_doc(model, tok, hub, doc, streams, S, max_new, temp, top_p, max_doc_tokens, dev):
    from xenc.backbone import build_chat
    from xenc.doc_tasks import ctx_prompt  # noqa: F401  (same "## Document" framing as the reader prompts)
    ids_doc = tok(doc, add_special_tokens=False)["input_ids"]
    if len(ids_doc) > max_doc_tokens:              # the operator covers the document END (D13Rt keeps the last tokens)
        doc = tok.decode(ids_doc[-max_doc_tokens:])
    parts = [k % S + 1 for k in range(streams)]
    seqs = [build_chat(tok, hub, SYSTEM, f"## Document\n{doc}\n\n## Task\n{TASK.format(S=S, k=k)}") for k in parts]
    P = 0
    while all(len(s) > P for s in seqs) and len({s[P] for s in seqs}) == 1:
        P += 1
    rest = [s[P:] for s in seqs]
    assert len({len(r) for r in rest}) == 1, "stream suffixes differ in length"
    out = model(input_ids=torch.tensor([seqs[0][:P]], device=dev), use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    cache.reorder_cache(torch.zeros(streams, dtype=torch.long, device=dev))
    out = model(input_ids=torch.tensor(rest, device=dev), past_key_values=cache, use_cache=True, logits_to_keep=1)
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    gen = [[] for _ in range(streams)]
    done = torch.zeros(streams, dtype=torch.bool, device=dev)
    pad = tok.convert_tokens_to_ids("<|im_end|>")
    for _ in range(max_new):
        lg = out.logits[:, -1].float() / temp
        p = torch.softmax(lg, -1)
        sp, si = p.sort(-1, descending=True)
        cut = sp.cumsum(-1) - sp > top_p
        sp = sp.masked_fill(cut, 0.0)
        nxt = si.gather(-1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1)).squeeze(-1)
        nxt = torch.where(done, torch.full_like(nxt, pad), nxt)
        for b, t in enumerate(nxt.tolist()):
            if not done[b]:
                if t in eos:
                    done[b] = True
                else:
                    gen[b].append(t)
        if bool(done.all()):
            break
        out = model(input_ids=nxt.view(-1, 1), past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
    return [tok.decode(g, skip_special_tokens=True) for g in gen]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/D15_selfq")
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--streams", type=int, default=6)
    ap.add_argument("--parts", type=int, default=6)
    ap.add_argument("--keep", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_doc_tokens", type=int, default=14900)
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--max_hours", type=float, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from xenc.backbone import load_reader
    from xenc.doc_tasks import load_docs_index, _doc_path
    torch.manual_seed(a.seed + a.shard)
    pool = Path(a.pool)
    idx = load_docs_index(str(pool))
    order = json.load(open(pool / "order.json"))
    od = pool / "selfq"
    od.mkdir(exist_ok=True)
    done = set()
    for fn in glob.glob(str(od / "shard*.jsonl")):
        for line in open(fn):
            try:
                done.add(json.loads(line)["doc_id"])
            except Exception:
                pass
    todo = [d for i, d in enumerate(order) if i % a.nshard == a.shard and d not in done]
    if a.max_docs:
        todo = todo[: a.max_docs]
    model, tok = load_reader(a.reader, device_map=a.device)
    from xenc.backbone import input_device
    dev = input_device(model)
    f = open(od / f"shard{a.shard}.jsonl", "a")
    t0, nq = time.time(), 0
    print(f"selfq shard {a.shard}/{a.nshard}: {len(todo)} docs to do ({len(done)} done)", flush=True)
    for i, did in enumerate(todo):
        if a.max_hours and time.time() - t0 > a.max_hours * 3600:
            print("time budget reached", flush=True)
            break
        text = _doc_path(pool, idx[did]["path"]).read_text()
        try:
            outs = gen_doc(model, tok, a.reader, text, a.streams, a.parts, a.max_new, a.temp, a.top_p,
                           a.max_doc_tokens, dev)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM on {did} ({idx[did].get('n_tokens')} tok); skipped", flush=True)
            continue
        items, seen = [], set()
        for o in outs:
            for q in parse(o):
                k = norm(q["question"])
                if k in seen:
                    continue
                seen.add(k)
                items.append(q)
        mcq = [q for q in items if q["choices"]]
        sa = [q for q in items if not q["choices"]]
        items = (mcq[: a.keep // 3] + sa)[: a.keep] if len(items) > a.keep else items
        items = [{"key": f"{did}_s{j}", "doc_id": did, **q, "task": "d15selfq", "hop": None,
                  "meta": {"selfq": True, "src": idx[did].get("src")}} for j, q in enumerate(items)]
        f.write(json.dumps({"doc_id": did, "items": items, "n_raw": len(outs),
                            "raw0": outs[0][:600] if not items else ""}) + "\n")
        f.flush()
        nq += len(items)
        if i < 3 or (i + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  [{i + 1}/{len(todo)}] {did} ({idx[did].get('n_tokens')} tok): {len(items)} q | total {nq} q | "
                  f"{el / (i + 1):.1f}s/doc", flush=True)
            if i < 3:
                print("    e.g. " + " || ".join(q["question"] for q in items[:4]), flush=True)
    print(f"done shard {a.shard}: {nq} questions ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
