"""E4 diagnostic: where does the D13 read go on MULTI-HOP questions, and which read change could help?

DIAGNOSTIC ONLY (shared base, step 0, no registration, VAL documents). Not a kv result; nothing here is reported as one.

Part 1 (MuSiQue val; per-hop supporting paragraphs are labelled in meta.decomposition):
    per reader read layer, the attention mass the read puts on each hop's supporting paragraph, for
      student  = the D13 reader-query read (the read that runs at test time),
      teacher  = the 4B encoder's own attention over the same operator columns (encoder-side; reference only),
      text     = the 9B reader's own attention at ALL its full-attention layers when it reads the document as text,
    on two row sets: the last question row ("q": predicts the first answer token) and the teacher-forced gold-answer
    rows ("ans"). Per head and per paragraph, so hop-1 vs hop-2 retrieval can be told apart per layer.
Part 2 (MuSiQue + LastingBench val; greedy, budget 32): training-free read variants at step 0
      base | topK (keep the K highest operator columns per head and row) | tempX (logits × X) |
      oracle_sup (only header + supporting paragraphs visible) | oracle_hop1 | mask_sup (supporting paragraphs hidden)
    plus text and closed_tail. The oracle / mask rows use gold paragraph labels: they bound what perfect addressing
    could give (diagnostic upper / lower bounds), never a method.

    python scripts/e4_hopdiag.py --out outputs/E4_diag --n_musique 100 --n_lasting 60
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
import xenc.d13_read as D13  # noqa: E402
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402


# ----------------------------------------------------------------------------- read variants (xenc/e4_read.py)
from xenc import e4_read  # noqa: E402
from xenc.e4_read import Variant  # noqa: E402

e4_read.install()
_ORIG_ATTEND = e4_read._ORIG_ATTEND
attend_variant = e4_read.attend_variant


def parse_variant(name: str, ranges: dict | None, N: int) -> bool:
    """Set Variant for `name` (base | tempX[@l+l] | topK[@l+l] | oracle_sup | oracle_hop1 | mask_sup); False if the
    variant needs paragraph labels that this item lacks."""
    if name not in ("oracle_sup", "oracle_hop1", "mask_sup"):
        e4_read.set_from(name)
        return True
    Variant.reset()
    if not ranges:
        return False
    m = torch.zeros(N, dtype=torch.bool)
    if name in ("oracle_sup", "oracle_hop1"):
        m[: ranges["header_end"]] = True
        hops = ranges["hops"] if name == "oracle_sup" else ranges["hops"][:1]
        for a, b in hops:
            m[a:b] = True
    else:
        m[:] = True
        for a, b in ranges["hops"]:
            m[a:b] = False
    Variant.colmask = m
    return True


# ----------------------------------------------------------------------------- paragraphs -> operator columns
def para_ranges(tok, prefix_text: str, pre_ids: list, doc_text: str, spans: list, N: int) -> dict:
    """Char spans of the document's paragraphs -> token ranges [a, b) in the operator's columns (clipped to N)."""
    head = "## Document\n"
    s0 = prefix_text.find(head + doc_text)
    assert s0 >= 0, "document text not found in the prefix"
    s0 += len(head)
    enc = tok(prefix_text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(x) for x in enc["input_ids"]]
    assert ids == list(pre_ids), "prefix re-tokenization differs from the runtime's prefix (truncated document?)"
    starts = [o[0] for o in enc["offset_mapping"]]
    import bisect

    def tok_at(c):                     # first token whose start char >= c
        return bisect.bisect_left(starts, c)
    out = []
    for a, b in spans:
        ta, tb = tok_at(s0 + a), tok_at(s0 + b)
        out.append((min(ta, N), min(tb, N)))
    return {"paras": out, "header_end": min(tok_at(s0), N)}


def para_mass(p_op: torch.Tensor, paras: list) -> torch.Tensor:
    """p_op [H, R, N] (rows already selected) -> [H, P] mean-over-rows mass per paragraph."""
    cs = torch.nn.functional.pad(p_op.float().cumsum(-1), (1, 0))
    m = torch.stack([cs[..., b] - cs[..., a] for a, b in paras], -1)      # [H, R, P]
    return m.mean(1)


def text_attn_probs(q: torch.Tensor, k: torch.Tensor, rows: list, scale: float) -> torch.Tensor:
    """The reader's own causal attention probabilities at the given absolute rows: q [Hq,T,hd], k [Hkv,T,hd]
    (post-RoPE, as EncTeacher records them) -> [Hq, len(rows), T]."""
    Hq, Hkv = q.shape[0], k.shape[0]
    idx = torch.tensor(list(rows), device=q.device)
    qs = q[:, idx].view(Hkv, Hq // Hkv, len(rows), -1)
    s = torch.einsum("kgtd,kcd->kgtc", qs, k) * scale
    col = torch.arange(k.shape[1], device=q.device).view(1, -1)
    s = s.masked_fill((col > idx.view(-1, 1)).view(1, 1, len(rows), -1), float("-inf"))
    return torch.softmax(s, -1).reshape(Hq, len(rows), -1)


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--interface", default="outputs/D13_4b_reg_w10/d13_best.pt")
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_musique", type=int, default=100)
    ap.add_argument("--n_lasting", type=int, default=60)
    ap.add_argument("--n_quality", type=int, default=0, help="QuALITY val items (5 per doc; MCQ harm check)")
    ap.add_argument("--quality_skip_docs", type=int, default=0)
    ap.add_argument("--variants", default="base,top16,top64,top256,temp1.5,temp2,oracle_sup,oracle_hop1,mask_sup")
    ap.add_argument("--conds", default="text,closed_tail")
    ap.add_argument("--attn", type=int, default=1, help="part 1 (attention masses) on MuSiQue")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    from fit_d13 import D13Rt, build
    from register_state import greedy
    from xenc.backbone import build_chat, text_ids
    from xenc.d13_read import EncTeacher, teacher_logits
    from xenc.doc_tasks import ctx_prompt, gold_text, load_docs, score
    from xenc.state_skill import doc_prefix_text, full_layers

    a.share_encoder = 0
    reader, enc, tok, itf, lmap = build(a, init=a.interface)
    if itf.delta_rank:
        itf.merge_delta()
    itf.eval()
    rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens)
    rfull = full_layers(reader)
    from deltaskill.backbone_utils import decoder_layers
    RL = decoder_layers(reader)
    TcR = EncTeacher(reader, rfull)          # records the reader's own q/k in the text pass only (rec=None otherwise)
    log(f"E4 diag: init {a.interface} | B {a.B} | layer map {lmap} | gates {[round(float(x), 3) for x in itf.g]} | "
        f"reader full layers {rfull} | split {a.split}")

    rng = random.Random(a.seed)
    items = []
    mus = load_split_items("data/musique_split", a.split)
    rng.shuffle(mus)
    items += [("musique", it) for it in mus[: a.n_musique]]
    las = load_split_items("data/lastingbench_split", a.split)
    by: dict = {}
    for it in las:
        by.setdefault(_get(it, "doc_id"), []).append(it)
    sel = []
    for d in sorted(by):
        if len(sel) >= a.n_lasting:
            break
        sel += by[d][:3]
    items += [("lastingbench", it) for it in sel[: a.n_lasting]]
    if a.n_quality:
        qa = load_split_items("data/quality_split", a.split)
        byq: dict = {}
        for it in qa:
            byq.setdefault(_get(it, "doc_id"), []).append(it)
        selq = []
        for d in sorted(byq)[a.quality_skip_docs:]:
            if len(selq) >= a.n_quality:
                break
            selq += byq[d][:5]
        items += [("quality", it) for it in selq[: a.n_quality]]
    docs = {**load_docs("data/musique_split"), **load_docs("data/lastingbench_split"), **load_docs("data/quality_split")}
    rt.docs = docs
    log(f"variant patch installed: {D13._attend is attend_variant} | items: quality {sum(t == 'quality' for t, _ in items)} | musique {sum(t == 'musique' for t, _ in items)} (docs {len({_get(i, 'doc_id') for t, i in items if t == 'musique'})}) "
        f"lastingbench {sum(t == 'lastingbench' for t, _ in items)} (docs {len({_get(i, 'doc_id') for t, i in items if t == 'lastingbench'})})")
    variants = [v for v in a.variants.split(",") if v]
    conds = [c for c in a.conds.split(",") if c]
    fo = open(os.path.join(a.out, "items.jsonl"), "w")
    t0 = time.time()
    for j, (task, it) in enumerate(items):
        did = _get(it, "doc_id")
        rec = {"task": task, "key": _get(it, "key"), "doc_id": did, "hop_type": (_get(it, "meta") or {}).get("hop_type")}
        op, N, _ = rt.operator(did)
        ranges = None
        meta = _get(it, "meta") or {}
        if task == "musique" and meta.get("decomposition"):
            pr = para_ranges(tok, doc_prefix_text(tok, a.reader, docs[did], SYSTEM_ANSWER), rt.pre(did), docs[did],
                             meta["para_spans"], N)
            hops = [pr["paras"][s["paragraph_support_idx"]] for s in meta["decomposition"]]
            ranges = {"hops": hops, "header_end": pr["header_end"], "paras": pr["paras"],
                      "hop_idx": [s["paragraph_support_idx"] for s in meta["decomposition"]]}
            rec["N"], rec["hop_idx"], rec["hop_tok"] = N, ranges["hop_idx"], [b - a_ for a_, b in hops]
            rec["para_len"] = [b - a_ for a_, b in pr["paras"]]
        # ---------------- part 1: attention masses
        if a.attn and ranges is not None:
            Variant.reset()
            tt = text_ids(tok, gold_text(it))
            rows = rt.rows(did, it, tt)
            rq = a.tail + len(rt.suf(it)) - 1
            rsets = {"q": [rq], "ans": list(range(rq + 1, rq + len(tt)))}
            P = ranges["paras"]
            att = {"student": {}, "teacher": {}, "text": {}}
            with torch.no_grad():
                R = rt.R
                R.load(op, N)
                R.mode, R.need_logits, R.rec = "e2e", True, {}
                try:
                    reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=False, logits_to_keep=1)
                    for li in itf.layers:
                        p = torch.softmax(R.rec[li]["logits"].float(), -1)
                        p = p.reshape(-1, p.shape[2], p.shape[3])[..., :N]
                        att["student"][li] = {k: para_mass(p[:, r], P).tolist() for k, r in rsets.items()}
                finally:
                    R.mode, R.rec, R.need_logits = "off", {}, False
                trec = rt.teacher(did, rows)
                for li, e in itf.layer_map.items():
                    s = teacher_logits(trec[e]["q"], op[e][0], trec[e]["k"], itf.att[str(li)].scaling)
                    p = torch.softmax(s.float(), -1)
                    p = p.reshape(-1, p.shape[2], p.shape[3])[..., :N]
                    att["teacher"][li] = {k: para_mass(p[:, r], P).tolist() for k, r in rsets.items()}
                del trec
                pre = rt.pre(did)
                full = pre + rt.suf(it) + list(tt)
                off = len(pre) - a.tail
                TcR.rec, TcR.kv_only = {}, False
                try:
                    reader(input_ids=torch.tensor([full], device=rt.dev), use_cache=False, logits_to_keep=1)
                    for li in rfull:
                        q, k = TcR.rec[li]["q"].float(), TcR.rec[li]["k"].float()
                        sc = RL[li].self_attn.scaling
                        for kname, r in rsets.items():
                            p = text_attn_probs(q, k, [off + x for x in r], sc)[..., :N]
                            att["text"].setdefault(li, {})[kname] = para_mass(p, P).tolist()
                finally:
                    TcR.rec = None
            rec["attn"] = {c: {str(li): {k: [[round(x, 5) for x in h] for h in v] for k, v in d.items()}
                               for li, d in dd.items()} for c, dd in att.items()}
        # ---------------- part 2: generation variants
        gens = {}
        for v in variants:
            if not parse_variant(v, ranges, N):
                continue
            try:
                g, n = rt.gen(did, it, a.budget)
            finally:
                Variant.reset()
            gens[v] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
        if "closed_tail" in conds:
            g, n = rt.gen(did, it, a.budget, read=False)
            gens["closed_tail"] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
        if "text" in conds:
            g, n = greedy(reader, tok, a.reader, build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, docs[did])), a.budget)
            gens["text"] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
        rec["gens"] = gens
        fo.write(json.dumps(rec) + "\n")
        fo.flush()
        if (j + 1) % 10 == 0:
            log(f"  [{j + 1}/{len(items)}] {task} " + " ".join(f"{k}={v['score']:.2f}" for k, v in gens.items())
                + f" | {(time.time() - t0) / (j + 1):.1f}s/item")
    fo.close()
    log("E4 diag done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
