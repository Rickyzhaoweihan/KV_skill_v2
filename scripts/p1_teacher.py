"""P1 (data scale), GPU step: per corpus document, ONE 9B prefill of the text-teacher prefix, then
  (a) --qgen: diverse questions written by the 9B itself from that same cached prefix (5 sampled streams, one per type:
      span / choice (4-way MCQ) / claim (NLI 3-way or true/false) / chain (two-part, multi-hop) / bool (yes/no MCQ));
  (b) the 9B TEXT TEACHER's top-k log-probs on the answer rows of every question, teacher-forced on its target
      (gold / native / generator answer), from the same cache, in right-padded batches (causal => exact rows).
The rows are exactly those register_d13.py's KL(text || kv) uses: kept_logits(reader, tids, m+1)[:-1] with
tids = build_chat(SYSTEM_ANSWER, ctx_prompt(item, doc)) + text_ids(target). Items whose tids exceed --max_tokens are
skipped (training would skip them too).

Output (--corpus/teacher/): shard<i>.jsonl (one line per doc: items with target, m, conf = geometric-mean teacher prob
of the target, agree = target is the teacher's greedy path, t_top = teacher argmax at the first answer row) and
shard<i>_c<j>.pt chunks {key: {"ids": int32[m,k], "lp": fp16[m,k], "rest": fp32[m] (log of the mass outside the
top-k), "tgt": int32[m]}}. Resumable: docs present in shard<i>.jsonl are skipped (chunks are saved before lines).

--verify N: for the first N items also run the plain full forward (exactly the training code's teacher pass) and log
max |delta logp| over the stored top-k and the top-k-vs-full KL, so the precompute is checked against training.

    python scripts/p1_teacher.py --corpus /scratch/.../P1_corpus --shard 0 --nshard 4 --qgen 1 --verify 20
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

QTYPES = ["span", "choice", "claim", "chain", "bool"]
QGEN = ("Write questions that can only be answered by reading this document. Split the document into {S} equal parts "
        "and take every question from part {k}. The question types are:\n"
        "- span: 3 short-answer questions about specific details (a name, number, date, place, event, reason or "
        "relation); each answer is a short phrase from the document. Format:\nQ1: <question>\nA1: <answer>\n"
        "Q2: <question>\nA2: <answer>\nQ3: <question>\nA3: <answer>\n"
        "- choice: 2 multiple-choice questions with four plausible options and exactly one correct option. Format:\n"
        "Q1: <question>\nA) <option>\nB) <option>\nC) <option>\nD) <option>\nA1: <letter>\n(then Q2 the same way)\n"
        "- claim: 3 statements on one topic: one the document supports, one it contradicts (change one detail), and "
        "one it does not mention. Format:\nS1: <statement>\nL1: <supported, contradicted or not mentioned>\n"
        "S2: <statement>\nL2: <label>\nS3: <statement>\nL3: <label>\n"
        "- chain: 2 questions that each need two facts from different parts of the document combined; short answers. "
        "Format:\nQ1: <question>\nA1: <answer>\nQ2: <question>\nA2: <answer>\n"
        "- bool: 3 yes/no questions with a mix of yes and no answers. Format:\nQ1: <question>\nA1: <yes or no>\n"
        "Q2: <question>\nA2: <yes or no>\nQ3: <question>\nA3: <yes or no>\n"
        "Write type {T} only, in exactly its format and nothing else.")
NLI_CH = [{"label": "A", "text": "Entailed: the document states or implies this"},
          {"label": "B", "text": "Contradicted: the document states the opposite"},
          {"label": "C", "text": "Not mentioned: the document does not mention this"}]
TF_CH = [{"label": "A", "text": "True"}, {"label": "B", "text": "False"}]
YN_CH = [{"label": "A", "text": "Yes"}, {"label": "B", "text": "No"}]


def _ok_q(q: str) -> bool:
    return 12 <= len(q) <= 400


def parse_qgen(txt: str, qtype: str, rng: random.Random) -> list[dict]:
    txt = re.sub(r"</?answer>", "", txt)
    out = []
    if qtype in ("span", "chain"):
        for n in (1, 2, 3):
            m = re.search(rf"Q{n}:\s*(.+?)\s*\n\s*A{n}:\s*(.+?)\s*(?:\n|$)", txt)
            if m and _ok_q(m.group(1)) and 0 < len(m.group(2).split()) <= 12:
                out.append({"question": m.group(1).strip(), "choices": None, "answers": [m.group(2).strip()],
                            "correct_label": None})
    elif qtype == "choice":
        for n in (1, 2):
            m = re.search(rf"Q{n}:\s*(.+?)\s*\n\s*A\)\s*(.+?)\s*\n\s*B\)\s*(.+?)\s*\n\s*C\)\s*(.+?)\s*\n\s*D\)\s*(.+?)\s*\n"
                          rf"\s*A{n}:\s*\(?([ABCD])", txt, re.S)
            if m and _ok_q(m.group(1)):
                opts = [m.group(i + 2).strip() for i in range(4)]
                if len(set(o.lower() for o in opts)) == 4 and all(0 < len(o) < 300 for o in opts):
                    lab = m.group(6)
                    out.append({"question": m.group(1).strip(), "choices": [{"label": L, "text": o} for L, o in zip("ABCD", opts)],
                                "answers": [opts["ABCD".index(lab)]], "correct_label": lab})
    elif qtype == "claim":
        for n in (1, 2, 3):
            m = re.search(rf"S{n}:\s*(.+?)\s*\n\s*L{n}:\s*(supported|contradicted|not mentioned)", txt, re.I)
            if not m or not _ok_q(m.group(1)):
                continue
            s, lab = m.group(1).strip(), m.group(2).lower()
            if lab != "not mentioned" and rng.random() < 0.5:     # CLIPPER-style true/false
                out.append({"question": f"Is the following statement about the document true or false?\n\nStatement: {s}",
                            "choices": TF_CH, "answers": ["True" if lab == "supported" else "False"],
                            "correct_label": "A" if lab == "supported" else "B"})
            else:                                                 # ContractNLI-style 3-way
                L = {"supported": "A", "contradicted": "B", "not mentioned": "C"}[lab]
                out.append({"question": f"Consider the following statement about this document:\n\"{s}\"\n"
                                        "What does the document say about the statement?",
                            "choices": NLI_CH, "answers": [NLI_CH["ABC".index(L)]["text"]], "correct_label": L})
    elif qtype == "bool":
        for n in (1, 2, 3):
            m = re.search(rf"Q{n}:\s*(.+?)\s*\n\s*A{n}:\s*(yes|no)\b", txt, re.I)
            if m and _ok_q(m.group(1)):
                yes = m.group(2).lower() == "yes"
                out.append({"question": m.group(1).strip(), "choices": YN_CH, "answers": ["Yes" if yes else "No"],
                            "correct_label": "A" if yes else "B"})
    return out


def lcp(a: list, b: list) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def expand(cache, n: int, dev):
    c = copy.deepcopy(cache)
    if n > 1:
        c.reorder_cache(torch.zeros(n, dtype=torch.long, device=dev))
    return c


@torch.no_grad()
def qgen_streams(model, tok, cache, P: int, seqs: list[list[int]], max_new: int, temp: float, top_p: float, dev):
    rest = [s[P:] for s in seqs]
    assert len({len(r) for r in rest}) == 1, "qgen stream suffixes differ in length"
    c = expand(cache, len(seqs), dev)
    out = model(input_ids=torch.tensor(rest, device=dev), past_key_values=c, use_cache=True, logits_to_keep=1)
    eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    pad = tok.convert_tokens_to_ids("<|im_end|>")
    gen = [[] for _ in seqs]
    done = torch.zeros(len(seqs), dtype=torch.bool, device=dev)
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
    del c, out
    return [tok.decode(g, skip_special_tokens=True) for g in gen]


@torch.no_grad()
def teacher_rows(model, cache, P: int, seqs: list[list[int]], ms: list[int], k: int, pad: int, dev):
    """Right-padded batch continuing from the shared prefix cache -> per seq dict(ids, lp, rest, tgt, conf, agree, top1)."""
    rest = [s[P:] for s in seqs]
    L = max(len(r) for r in rest)
    x = torch.tensor([r + [pad] * (L - len(r)) for r in rest], device=dev)
    c = expand(cache, len(seqs), dev)
    lg = model(input_ids=x, past_key_values=c, use_cache=True, logits_to_keep=L).logits     # [B, L, V]
    del c
    out = []
    for b, (r, m) in enumerate(zip(rest, ms)):
        Ls = len(r)
        rows = lg[b, Ls - m - 1: Ls - 1].float().log_softmax(-1)        # m rows predicting the m target tokens
        tgt = torch.tensor(r[Ls - m:], device=dev)
        lp_t = rows.gather(-1, tgt.view(-1, 1)).squeeze(-1)
        v, i = rows.topk(k, -1)
        rest_mass = torch.log1p(-v.exp().sum(-1).clamp(max=1 - 1e-6))
        out.append({"ids": i.int().cpu(), "lp": v.half().cpu(), "rest": rest_mass.cpu(), "tgt": tgt.int().cpu(),
                    "conf": float(lp_t.mean().exp()), "agree": bool((rows.argmax(-1) == tgt).all()),
                    "top1": int(rows[0].argmax())})
    del lg
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--qgen", type=int, default=1, help="1: generate questions for docs with < --qgen_min questions")
    ap.add_argument("--qgen_min", type=int, default=6)
    ap.add_argument("--max_q", type=int, default=16, help="max existing questions per doc (seeded sample)")
    ap.add_argument("--parts", type=int, default=5)
    ap.add_argument("--max_new", type=int, default=280)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=100000)
    ap.add_argument("--cache_budget_gb", type=float, default=10.0, help="memory for the batch's prefix-cache copies")
    ap.add_argument("--max_batch", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=50, help="docs per saved chunk")
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--max_hours", type=float, default=0)
    ap.add_argument("--order", default="interleave", choices=["interleave", "existing_first"])
    ap.add_argument("--verify", type=int, default=0)
    ap.add_argument("--verify_gate", type=float, default=1.0, help="abort the shard when, after --verify items, the stored "
                    "rows differ from the training-code teacher pass by > this many nats (or top-1 agreement < 0.8)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from register_doc import SYSTEM_ANSWER
    from xenc.backbone import build_chat, input_device, load_reader, text_ids
    from xenc.doc_tasks import ctx_prompt, gold_text
    from xenc.state_skill import doc_prefix_text
    rng = random.Random(a.seed * 1000 + a.shard)
    torch.manual_seed(a.seed * 1000 + a.shard)
    C = Path(a.corpus)
    docs = json.load(open(C / "docs.json"))
    qs = {}
    for line in open(C / "questions.jsonl"):
        r = json.loads(line)
        qs[r["doc_id"]] = r["items"]
    od = C / "teacher"
    od.mkdir(exist_ok=True)
    jf = od / f"shard{a.shard}.jsonl"
    done = set()
    if jf.exists():
        for line in open(jf):
            try:
                r = json.loads(line)
                if (od / r["chunk"]).exists():
                    done.add(r["doc_id"])
            except Exception:
                pass
    order = sorted(docs)
    random.Random(1234).shuffle(order)
    if a.order == "existing_first":
        order = [d for d in order if len(qs.get(d, [])) >= a.qgen_min] + [d for d in order if len(qs.get(d, [])) < a.qgen_min]
    todo = [d for i, d in enumerate(order) if i % a.nshard == a.shard and d not in done]
    if a.max_docs:
        todo = todo[: a.max_docs]
    model, tok = load_reader(a.reader, device_map=a.device)
    dev = input_device(model)
    pad = tok.convert_tokens_to_ids("<|im_end|>")
    gpu = torch.cuda.get_device_name(dev) if torch.cuda.is_available() else "cpu"
    print(f"P1 teacher shard {a.shard}/{a.nshard} on {gpu}: {len(todo)} docs to do ({len(done)} done) | qgen {a.qgen} "
          f"(min {a.qgen_min}) | top-k {a.topk} | max_tokens {a.max_tokens} | corpus {C}", flush=True)
    tw = [tok(f"type {t} from", add_special_tokens=False)["input_ids"] for t in QTYPES]
    assert len({len(x) for x in tw}) == 1, f"qgen type words differ in token length: {tw}"
    buf_lines, buf_t, nchunk = [], {}, len(list(od.glob(f"shard{a.shard}_c*.pt")))
    stats = {"docs": 0, "items": 0, "skipped_long": 0, "qgen_docs": 0, "qgen_items": 0, "oom": 0, "verify": []}
    t0 = time.time()

    def flush():
        nonlocal buf_lines, buf_t, nchunk
        if not buf_lines:
            return
        name = f"shard{a.shard}_c{nchunk:04d}.pt"
        torch.save(buf_t, od / name)
        with open(jf, "a") as f:
            for ln in buf_lines:
                f.write(json.dumps({**ln, "chunk": name}) + "\n")
        nchunk += 1
        buf_lines, buf_t = [], {}

    for di, did in enumerate(todo):
        if a.max_hours and time.time() - t0 > a.max_hours * 3600:
            print("time budget reached", flush=True)
            break
        meta = docs[did]
        text = Path(meta["path"]).read_text()
        items = list(qs.get(did, []))
        if len(items) > a.max_q:
            items = random.Random(int(hashlib.md5(did.encode()).hexdigest()[:8], 16) + a.seed).sample(items, a.max_q)
        try:
            pre = [int(x) for x in tok(doc_prefix_text(tok, a.reader, text, SYSTEM_ANSWER), add_special_tokens=False)["input_ids"]]
            if len(pre) + 300 > a.max_tokens:
                stats["skipped_long"] += 1
                buf_lines.append({"doc_id": did, "src": meta["src"], "items": [], "skip": "long", "n_pre": len(pre)})
                continue
            P = len(pre) - 4
            out = model(input_ids=torch.tensor([pre[:P]], device=dev), use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
            del out
            per_tok = sum(t.numel() * t.element_size() for t in cache_tensors(cache)) / max(P, 1)
            B = max(1, min(a.max_batch, int(a.cache_budget_gb * 2**30 / max(per_tok * P, 1))))
            gen_info = None
            if a.qgen and len(items) < a.qgen_min:
                ks = list(range(1, a.parts + 1))
                rng.shuffle(ks)
                seqs = [build_chat(tok, a.reader, SYSTEM_ANSWER,
                                   f"## Document\n{text}\n\n## Question\n{QGEN.format(S=a.parts, k=ks[j % len(ks)], T=t)}")
                        for j, t in enumerate(QTYPES)]
                assert all(lcp(s, pre) >= P for s in seqs), "qgen prompt does not share the doc prefix"
                outs = []
                for j0 in range(0, len(seqs), B):
                    outs += qgen_streams(model, tok, cache, P, seqs[j0:j0 + B], a.max_new, a.temp, a.top_p, dev)
                seen = {re.sub(r"[^a-z0-9]+", " ", i["question"].lower()) for i in items}
                n0 = len(items)
                for t, o in zip(QTYPES, outs):
                    for j, q in enumerate(parse_qgen(o, t, rng)):
                        kq = re.sub(r"[^a-z0-9]+", " ", q["question"].lower())
                        if kq in seen:
                            continue
                        seen.add(kq)
                        items.append({"key": f"{did}_p1{t}{j}", "doc_id": did, **q, "task": "p1q", "hop": None,
                                      "meta": {"qtype": t, "src": meta["src"]}, "origin": "p1qgen"})
                stats["qgen_docs"] += 1
                stats["qgen_items"] += len(items) - n0
                gen_info = {"n_new": len(items) - n0, "raw0": outs[0][:300] if len(items) == n0 else ""}
            recs = []
            for it in items:
                target = gold_text(it)
                tt = text_ids(tok, target)
                tids = build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, text)) + tt
                if len(tids) > a.max_tokens:
                    stats["skipped_long"] += 1
                    continue
                assert tids[:P] == pre[:P], f"item prompt does not share the doc prefix ({did})"
                recs.append((it, target, tids, len(tt)))
            outl = []
            for j0 in range(0, len(recs), B):
                bt = recs[j0:j0 + B]
                rows = teacher_rows(model, cache, P, [r[2] for r in bt], [r[3] for r in bt], a.topk, pad, dev)
                for (it, target, tids, m), R in zip(bt, rows):
                    buf_t[it["key"]] = {k: R[k] for k in ("ids", "lp", "rest", "tgt")}
                    outl.append({**{k: it.get(k) for k in ("key", "question", "choices", "answers", "correct_label", "task",
                                                          "meta", "origin")},
                                 "target": target, "m": m, "n_tids": len(tids), "conf": round(R["conf"], 4),
                                 "agree": R["agree"], "t_top": tok.decode([R["top1"]])})
                    if len(stats["verify"]) < a.verify:
                        stats["verify"].append(verify(model, tids, m, R, dev))
                        v = stats["verify"][-1]
                        print(f"  [verify {len(stats['verify'])}] {it['key']} m={m} n={len(tids)}: max|dlogp| top-k "
                              f"{v['max_dlp']:.4f}, |d log rest| {v['rest_dl']:.4f}, max rest mass {v['rest']:.2e}, "
                              f"top1 match {v['top1_match']}", flush=True)
                        if len(stats["verify"]) == a.verify:
                            vv_ = stats["verify"]
                            mx, t1 = max(x["max_dlp"] for x in vv_), sum(x["top1_match"] for x in vv_) / len(vv_)
                            print(f"  [verify gate] {len(vv_)} items: max |dlogp| {mx:.4f}, top-1 agreement {t1:.3f}", flush=True)
                            if mx > a.verify_gate or t1 < 0.8:
                                raise SystemExit(f"VERIFY GATE FAILED: max |dlogp| {mx:.4f} top-1 {t1:.3f}")
            del cache
            buf_lines.append({"doc_id": did, "src": meta["src"], "group": meta.get("group"), "items": outl, "P": P,
                              "B": B, "gpu": gpu, **({"qgen": gen_info} if gen_info else {})})
            stats["docs"] += 1
            stats["items"] += len(outl)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            stats["oom"] += 1
            print(f"  OOM on {did} ({meta.get('n_tokens')} tok); skipped", flush=True)
            continue
        if len(buf_lines) >= a.chunk:
            flush()
        if di < 3 or (di + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  [{di + 1}/{len(todo)}] {did} ({meta.get('n_tokens')} tok, {meta['src']}): {len(outl)} items | total "
                  f"{stats['items']} items / {stats['docs']} docs, qgen {stats['qgen_items']} on {stats['qgen_docs']} docs | "
                  f"{el / (di + 1):.1f}s/doc", flush=True)
            if di < 3:
                print("    e.g. " + " || ".join(f"[{o['origin']}] {o['question'][:90]} -> {o['target']} (conf {o['conf']}, "
                                               f"agree {o['agree']})" for o in outl[:6]), flush=True)
    flush()
    vv = stats.pop("verify")
    if vv:
        stats["verify_max_dlp"] = max(v["max_dlp"] for v in vv)
        stats["verify_max_rest_mass"] = max(v["rest"] for v in vv)
        stats["verify_top1_match"] = sum(v["top1_match"] for v in vv) / len(vv)
    print(f"done shard {a.shard}: {json.dumps(stats)} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def cache_tensors(cache):
    out = []
    for name in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
        v = getattr(cache, name, None)
        if isinstance(v, (list, tuple)):
            out += [t for t in v if isinstance(t, torch.Tensor)]
    if not out and hasattr(cache, "layers"):
        for L in cache.layers:
            for name in ("keys", "values", "conv_states", "recurrent_states"):
                t = getattr(L, name, None)
                if isinstance(t, torch.Tensor):
                    out.append(t)
    return out


@torch.no_grad()
def verify(model, tids, m, R, dev) -> dict:
    """The training code's teacher pass (one full forward, no cache) vs the stored top-k rows."""
    full = model(input_ids=torch.tensor([tids], device=dev), logits_to_keep=m + 1).logits[0, -(m + 1):-1].float().log_softmax(-1)
    ids = R["ids"].long().to(dev)
    st = R["lp"].float().to(dev)
    dlp = (full.gather(-1, ids) - st).abs().max().item()
    # KL(full teacher || stored top-k + rest bucket) -- the information lost by storing top-k
    rest_full = torch.log1p(-full.gather(-1, ids).exp().sum(-1).clamp(max=1 - 1e-6))
    return {"max_dlp": dlp, "rest_dl": float((rest_full - R["rest"].to(dev)).abs().max()), "rest": float(R["rest"].exp().max()),
            "top1_match": int(int(full[0].argmax()) == R["top1"])}


if __name__ == "__main__":
    sys.exit(main())
