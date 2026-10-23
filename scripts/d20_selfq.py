"""D20 step 1: label-free self-questions for the documents that will be REGISTERED (test + dev documents), written
by the 9B reader from the DOCUMENT TEXT ONLY (same machinery as scripts/d15_selfq.py: one shared document prefill,
fanned out to S streams that differ only in a one-digit "part k of S" hint).

Templates (each run as its own batch of S streams):
  mcq  3 four-option multiple-choice questions on part k (detail / reason / relation / implication)
  mix  2 short-answer + 1 multiple-choice (the D15 template)
  nli  3 statements (one supported, one contradicted, one not mentioned) -> 3-option "what does the document say"
The generator's own answers are stored but NEVER used: registration targets are the text teacher's (9B reading the
document) greedy answers and distributions.

Leakage: the generator never sees any dataset question. As a second guard, any generated question that coincides with
(or contains) a judged question string of the task is dropped (xenc.d20_reg.drop_test_matches), and
scripts/register_d20.py asserts the same before registration.

    python scripts/d20_selfq.py --split_dir data/quality_split --docs test:0,val:10 --out outputs/D20_selfq/quality.jsonl
"""
from __future__ import annotations

import argparse
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
TEMPLATES = {
    "mcq": ("Write 3 multiple-choice questions that can only be answered by reading this document. Split the document "
            "into {S} equal parts and take all 3 questions from part {k}. The questions test different things: a "
            "specific detail, the reason for or consequence of an event, and what a person thinks, wants or implies. "
            "Each has four options with plausible wrong options and exactly one correct option. Use exactly this "
            "format and nothing else:\nQ1: <question>\nA) <option>\nB) <option>\nC) <option>\nD) <option>\n"
            "A1: <letter>\nQ2: <question>\nA) <option>\nB) <option>\nC) <option>\nD) <option>\nA2: <letter>\n"
            "Q3: <question>\nA) <option>\nB) <option>\nC) <option>\nD) <option>\nA3: <letter>"),
    "mix": ("Write 3 questions that can only be answered by reading this document. Split the document into {S} equal "
            "parts and take all 3 questions from part {k}. Each question asks about a different specific detail (a "
            "name, number, date, place, event, reason or relation). Questions 1 and 2 are short-answer; question 3 is "
            "multiple-choice with four options, exactly one correct. Use exactly this format and nothing else:\n"
            "Q1: <question>\nA1: <short answer>\nQ2: <question>\nA2: <short answer>\nQ3: <question>\n"
            "A) <option>\nB) <option>\nC) <option>\nD) <option>\nA3: <letter>"),
    "global": ("Write 3 multiple-choice questions about the WHOLE document that need information from several "
               "places to answer: why a character acts as they do, how the situation changes from beginning to end, "
               "what a key event reveals, or what the text implies overall (variant {k} of {S}: vary the focus). "
               "Each has four options with plausible wrong options and exactly one correct option. Use exactly this "
               "format and nothing else:\nQ1: <question>\nA) <option>\nB) <option>\nC) <option>\nD) <option>\n"
               "A1: <letter>\nQ2: <question>\nA) <option>\nB) <option>\nC) <option>\nD) <option>\nA2: <letter>\n"
               "Q3: <question>\nA) <option>\nB) <option>\nC) <option>\nD) <option>\nA3: <letter>"),
    "nli": ("Write 3 one-sentence statements about part {k} of this document (split it into {S} equal parts). "
            "Statement 1 is SUPPORTED by the document. Statement 2 CONTRADICTS the document (a plausible but false "
            "variant of something it says). Statement 3 is plausible but NOT MENTIONED in the document. Use exactly "
            "this format and nothing else:\nS1: <statement>\nS2: <statement>\nS3: <statement>"),
}
NLI_Q = "Consider this statement about the document:\n\"{s}\"\nIs the statement supported by the document?"
NLI_CHOICES = [{"label": "A", "text": "Supported: the document says or implies it"},
               {"label": "B", "text": "Contradicted: the document says otherwise"},
               {"label": "C", "text": "Not mentioned: the document does not cover it"}]


def _ok(q):
    return 12 <= len(q["question"]) <= 600 and all(len(a) < 300 for a in q["answers"])


def parse(tmpl: str, txt: str) -> list[dict]:
    out = []
    if tmpl == "mix":
        from d15_selfq import parse as p15
        return [dict(q) for q in p15(txt)]
    if tmpl in ("mcq", "global"):
        for n in (1, 2, 3):
            m = re.search(rf"Q{n}:\s*(.+?)\s*\n\s*A\)\s*(.+?)\s*\n\s*B\)\s*(.+?)\s*\n\s*C\)\s*(.+?)\s*\n\s*D\)\s*(.+?)"
                          rf"\s*\n\s*A{n}:\s*\(?([ABCD])", txt, re.S)
            if m:
                ch = [{"label": L, "text": m.group(i + 2).strip()} for i, L in enumerate("ABCD")]
                out.append({"question": m.group(1).strip(), "answers": [ch["ABCD".index(m.group(6))]["text"]],
                            "choices": ch, "correct_label": m.group(6)})
    elif tmpl == "nli":
        for n, lab in ((1, "A"), (2, "B"), (3, "C")):
            m = re.search(rf"S{n}:\s*(.+?)\s*(?:\n|$)", txt)
            if m and len(m.group(1)) > 15:
                s = m.group(1).strip().replace('"', "'")
                out.append({"question": NLI_Q.format(s=s), "answers": [NLI_CHOICES["ABC".index(lab)]["text"]],
                            "choices": [dict(c) for c in NLI_CHOICES], "correct_label": lab})
    return [q for q in out if _ok(q)]


@torch.no_grad()
def gen_streams(model, tok, hub, doc, tmpl, S, max_new, temp, top_p, max_doc_tokens, dev):
    """One prefix prefill shared by S streams (template `tmpl`, part k = 1..S) -> S sampled texts."""
    from xenc.backbone import build_chat
    ids_doc = tok(doc, add_special_tokens=False)["input_ids"]
    if len(ids_doc) > max_doc_tokens:               # the operator covers the document END (D13Rt keeps the last tokens)
        doc = tok.decode(ids_doc[-max_doc_tokens:])
    seqs = [build_chat(tok, hub, SYSTEM, f"## Document\n{doc}\n\n## Task\n{TEMPLATES[tmpl].format(S=S, k=k)}")
            for k in range(1, S + 1)]
    P = 0
    while all(len(s) > P for s in seqs) and len({s[P] for s in seqs}) == 1:
        P += 1
    rest = [s[P:] for s in seqs]
    assert len({len(r) for r in rest}) == 1, "stream suffixes differ in length"
    out = model(input_ids=torch.tensor([seqs[0][:P]], device=dev), use_cache=True, logits_to_keep=1)
    cache = out.past_key_values
    cache.reorder_cache(torch.zeros(S, dtype=torch.long, device=dev))
    out = model(input_ids=torch.tensor(rest, device=dev), past_key_values=cache, use_cache=True, logits_to_keep=1)
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    gen = [[] for _ in range(S)]
    done = torch.zeros(S, dtype=torch.bool, device=dev)
    pad = tok.convert_tokens_to_ids("<|im_end|>")
    for _ in range(max_new):
        p = torch.softmax(out.logits[:, -1].float() / temp, -1)
        sp, si = p.sort(-1, descending=True)
        sp = sp.masked_fill(sp.cumsum(-1) - sp > top_p, 0.0)
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
    del cache, out
    return [tok.decode(g, skip_special_tokens=True) for g in gen]


def task_questions(split_dir: str) -> list[str]:
    """Every question string of the task (train, val, test item files): the leakage reference."""
    from register_doc import load_split_items, _get
    qs = []
    for s in ("train", "val", "test"):
        try:
            qs += [str(_get(i, "question")) for i in load_split_items(split_dir, s)]
        except Exception:
            pass
    return qs


def doc_selection(split_dir: str, spec: str) -> list[str]:
    """spec "test:0,val:10" -> sorted doc ids (0 = all docs of that split's items, else the first K sorted)."""
    from register_doc import load_split_items, _get
    out = []
    for s in spec.split(","):
        name, k = s.split(":")
        d = sorted({_get(i, "doc_id") for i in load_split_items(split_dir, name)})
        out += d[: int(k)] if int(k) else d
    return sorted(set(out))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--docs", default="test:0", help="which documents: split:K (K first sorted docs, 0 = all)")
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--templates", default="mcq,mix,nli")
    ap.add_argument("--parts", type=int, default=6)
    ap.add_argument("--keep", type=int, default=48)
    ap.add_argument("--max_new", type=int, default=380)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_doc_tokens", type=int, default=14900)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from xenc.backbone import input_device, load_reader
    from xenc.d20_reg import drop_test_matches, norm_q
    from xenc.doc_tasks import load_docs
    torch.manual_seed(a.seed + a.shard)
    docs = load_docs(a.split_dir)
    sel = doc_selection(a.split_dir, a.docs)
    sel = [d for i, d in enumerate(sel) if i % a.nshard == a.shard]
    tq = task_questions(a.split_dir)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["doc_id"])
            except Exception:
                pass
    todo = [d for d in sel if d not in done]
    if a.max_docs:
        todo = todo[: a.max_docs]
    model, tok = load_reader(a.reader, device_map=a.device)
    dev = input_device(model)
    f = open(a.out, "a")
    t0, nq, ndrop = time.time(), 0, 0
    print(f"D20 selfq {a.split_dir} [{a.docs}] shard {a.shard}/{a.nshard}: {len(todo)} docs to do ({len(done)} done); "
          f"templates {a.templates}; leakage reference {len(tq)} task questions", flush=True)
    for i, did in enumerate(todo):
        td = time.time()
        items, seen = [], set()
        for tmpl in a.templates.split(","):
            try:
                outs = gen_streams(model, tok, a.reader, docs[did], tmpl, a.parts, a.max_new, a.temp, a.top_p,
                                   a.max_doc_tokens, dev)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  OOM on {did} [{tmpl}]; skipped", flush=True)
                continue
            for o in outs:
                for q in parse(tmpl, o):
                    k = norm_q(q["question"])
                    if k in seen:
                        continue
                    seen.add(k)
                    q["meta"] = {"d20selfq": True, "template": tmpl}
                    items.append(q)
        items, nd = drop_test_matches(items, tq)
        ndrop += nd
        if len(items) > a.keep:                 # keep a balanced mix across templates
            by = {}
            for q in items:
                by.setdefault(q["meta"]["template"], []).append(q)
            kept, j = [], 0
            while len(kept) < a.keep:
                for t in list(by):
                    if j < len(by[t]) and len(kept) < a.keep:
                        kept.append(by[t][j])
                j += 1
            items = kept
        items = [{"key": f"{did}_d20s{j}", "doc_id": did, **q, "task": "d20selfq", "hop": None} for j, q in enumerate(items)]
        f.write(json.dumps({"doc_id": did, "items": items, "n_dropped_test_match": nd,
                            "seconds": round(time.time() - td, 1)}) + "\n")
        f.flush()
        nq += len(items)
        if i < 3 or (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(todo)}] {did}: {len(items)} q (dropped {nd}) | total {nq} | "
                  f"{(time.time() - t0) / (i + 1):.1f}s/doc", flush=True)
            if i < 2:
                for q in items[:6]:
                    print(f"    e.g. [{q['meta']['template']}] {q['question'][:160]!r}", flush=True)
    print(f"done: {nq} questions, {ndrop} dropped as test matches ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
