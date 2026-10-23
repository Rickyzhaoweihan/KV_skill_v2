"""D25 step-1 diagnostics in TEXT mode and CLOSED BOOK (no operator, no training): D0 adherence, D2 logit lens,
D3 context-attention profile, D6 per-layer conflict direction. One reader per run.

Per item (MCQ: forced-choice letter after the "<answer>" prefix; span: greedy continuation of that prefix):
  closed_book  chat(q_prompt) + "<answer>"            -> letter logits, lens [n_layers, n_letters] (recall depth)
  text         document prefix prefilled ONCE per doc (cache deep-copied per question), then the question suffix
               + "<answer>"                            -> letter logits, lens (commit depth), and at every full-attention
               layer the attention mass of the question rows over {head (system/template), doc, evid (the asked
               passage / the quoted line), self (question rows)}: last row and mean over rows.
  D6           paired items (ConflictQA CF vs ORIGINAL bundle; Shakespeare-CF vs the original play, same question):
               last-position residual of every layer saved to h_<split>.pt.

Sets: comma list of split_dir[:n]; Shakespeare-CF gets synthetic ORIGINAL-play items (variant "orig") via --scf_orig.

  python scripts/d25_diag.py --reader Qwen/Qwen3.5-9B --split test \
      --sets data/shakespeare_cf_split,data/d25_cqa_split,data/d25_fecf_split,data/d25_nqswap_split \
      --out outputs/D25_diag_9b
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, log  # noqa: E402

ANS_PREFIX = "<answer>"


def load_items(split_dir, split, n, seed=0):
    items = json.load(open(os.path.join(split_dir, split, "items.json")))
    if n and len(items) > n:
        items = random.Random(seed).sample(items, n)
    return items


def scf_orig_items(items, orig_dir):
    """Shakespeare-CF conflict items re-asked on the ORIGINAL play (gold = the memorized speaker = prior_label)."""
    from xenc.doc_tasks import load_docs_index
    idx = load_docs_index(orig_dir)
    out = []
    for it in items:
        m = it.get("meta") or {}
        if not m.get("conflict"):
            continue
        od = it["doc_id"].replace("shakespeare_cf_", "shakespeare_", 1)
        if od not in idx:
            continue
        o = copy.deepcopy(it)
        o["key"] = it["key"] + "_orig"
        o["doc_id"] = od
        o["correct_label"] = m["prior_label"]
        o["answers"] = [c["text"] for c in it["choices"] if c["label"] == m["prior_label"]]
        o["meta"] = {**m, "variant": "orig", "conflict": False, "pair_key": it["key"]}
        out.append(o)
    return out


def evid_char_span(it, doc_text):
    """Char span (in the doc) of the evidence: the asked passage (bundles) or the quoted line + 60 chars before it
    (Shakespeare speaker items: the speaker prefix precedes the line)."""
    m = it.get("meta") or {}
    if m.get("span"):
        return tuple(m["span"])
    if m.get("passage"):
        k = int(m["passage"])
        a = doc_text.find(f"Passage {k}:\n")
        b = doc_text.find(f"Passage {k + 1}:\n")
        return (a, b if b > 0 else len(doc_text)) if a >= 0 else None
    q = re.search(r'line beginning: "(.+?)"', it["question"])
    if q:
        frag = q.group(1)[:40]
        a = doc_text.find(frag)
        if a < 0:
            a = re.sub(r"\s+", " ", doc_text).find(frag)       # approximate; whitespace differences
        if a >= 0:
            return (max(0, a - 60), a + len(q.group(1)))
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--sets", required=True, help="comma list of split_dir[:n]")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scf_orig", default="data/shakespeare_split")
    ap.add_argument("--d6", default="d25_cqa,shakespeare_cf", help="set names whose text-mode residuals are saved")
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reader_mem", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from register_state import _stop_ids
    from xenc.backbone import build_chat, decode, input_device, load_reader
    from xenc.d25_probe import AttnTap, LayerTap, lens_logits, segment_mass
    from xenc.doc_tasks import load_docs, q_prompt
    from xenc.state_skill import doc_prefix_text, full_layers, question_suffix_ids
    os.makedirs(a.out, exist_ok=True)
    reader, tok = load_reader(a.reader, device_map=a.device, max_mem=a.reader_mem)
    dev = input_device(reader)
    FL = full_layers(reader)
    tap, atap = LayerTap(reader), AttnTap(reader, FL)
    stops = _stop_ids(tok)
    ansp = [int(x) for x in tok(ANS_PREFIX, add_special_tokens=False)["input_ids"]]
    letter_ids = {}
    for L in "ABCDE":
        ids = tok(L, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, (L, ids)
        letter_ids[L] = int(ids[0])
    log(f"D25 diag: {a.reader} | layers {len(tap.layers)} full {FL} | ans prefix {ansp} | letters {letter_ids}")

    def fwd(ids, cache=None):
        return reader(input_ids=torch.tensor([ids], device=dev), past_key_values=cache, use_cache=True,
                      logits_to_keep=1)

    def cont(out, budget):
        tap.on, atap.on = False, False
        new = []
        try:
            for _ in range(budget):
                nxt = int(out.logits[0, -1].argmax())
                new.append(nxt)
                if nxt in stops or "</answer>" in tok.decode(new[-12:], skip_special_tokens=True):
                    break
                out = fwd([nxt], out.past_key_values)
        finally:
            tap.on, atap.on = True, True
        return ANS_PREFIX + decode(tok, a.reader, new)

    def readout(out, it, labels):
        """letter logits + lens of the latest forward; span items: greedy continuation."""
        r = {}
        H = tap.stack()
        if labels:
            ids = [letter_ids[x] for x in labels]
            r["logits"] = [round(float(v), 4) for v in out.logits[0, -1, ids].float()]
            r["pred"] = labels[int(torch.tensor(r["logits"]).argmax())]
            r["lens"] = [[round(float(v), 3) for v in row] for row in lens_logits(reader, H, ids)]
        else:
            r["gen"] = cont(out, a.budget)
        return r, H

    for spec in a.sets.split(","):
        sd, _, n = spec.partition(":")
        name = Path(sd.rstrip("/")).name.removesuffix("_split")
        items = load_items(sd, a.split, int(n or 0), a.seed)
        docs = load_docs(sd)
        if name == "shakespeare_cf" and a.scf_orig:
            orig = scf_orig_items(items, a.scf_orig)
            docs.update({k: v for k, v in load_docs(a.scf_orig).items() if k in {o["doc_id"] for o in orig}})
            items = items + orig
        save_h = name in a.d6.split(",")
        log(f"== {name} [{a.split}] {len(items)} items, {len({i['doc_id'] for i in items})} docs, save_h={save_h}")
        by_doc = defaultdict(list)
        for it in items:
            by_doc[it["doc_id"]].append(it)
        rows, Hs = [], {}
        ck = open(os.path.join(a.out, f"{name}_{a.split}.ckpt.jsonl"), "w")
        t0 = time.time()
        with torch.no_grad():
            for did in sorted(by_doc):
                text = docs[did]
                pre_text = doc_prefix_text(tok, a.reader, text, SYSTEM_ANSWER)
                enc = tok(pre_text, add_special_tokens=False, return_offsets_mapping=True)
                pre, offs = [int(x) for x in enc["input_ids"]], [tuple(o) for o in enc["offset_mapping"]]
                dstart = pre_text.find(text[:200])
                assert dstart >= 0, "document text not found in the rendered prefix"
                d0 = next(i for i, (s, e) in enumerate(offs) if s >= dstart)
                Np = len(pre)
                atap.reset_k()
                atap.want_q, tap.on = False, False
                out_pre = fwd(pre)
                snap = atap.snapshot()
                tap.on = True
                for it in by_doc[did]:
                    labels = [c["label"] for c in it["choices"]] if it.get("choices") else None
                    rec = {"key": it["key"], "doc_id": did, "set": name, "gold": it.get("correct_label"),
                           "meta": {k: v for k, v in (it.get("meta") or {}).items() if k not in ("sigma", "span")},
                           "n_doc_tokens": Np - d0}
                    # closed book
                    atap.on = False
                    cb = build_chat(tok, a.reader, SYSTEM_ANSWER, q_prompt(it)) + ansp
                    o = fwd(cb)
                    rec["closed_book"], _ = readout(o, it, labels)
                    atap.on = True
                    # text (prefix cache copy)
                    atap.restore(snap)
                    atap.want_q = True
                    suf = question_suffix_ids(tok, a.reader, it, SYSTEM_ANSWER) + ansp
                    o = fwd(suf, copy.deepcopy(out_pre.past_key_values))
                    atap.want_q = False
                    ev = evid_char_span(it, text)
                    segs = {"head": (0, d0), "doc": (d0, Np), "self": (Np, Np + len(suf))}
                    if ev:
                        a0, b0 = dstart + ev[0], dstart + ev[1]
                        m = torch.tensor([s < b0 and e > a0 for (s, e) in offs] + [False] * len(suf))
                        segs["evid"] = m
                        rec["evid_tokens"] = int(m.sum())
                    att = {}
                    for li in FL:
                        p = atap.probs(li)
                        ms = segment_mass(p, segs)
                        att[li] = {k: [round(float(v[-1]), 5), round(float(v.mean()), 5)] for k, v in ms.items()}
                        del p
                    rec["attn"] = att
                    rec["text"], H = readout(o, it, labels)
                    if save_h:
                        Hs[it["key"]] = H.half().cpu()
                    rows.append(rec)
                    ck.write(json.dumps(rec) + "\n")
                    ck.flush()
                del out_pre, snap
                atap.reset_k()
                torch.cuda.empty_cache()
                log(f"  {did}: {len(by_doc[did])} items ({Np} tokens) | {len(rows)}/{len(items)} "
                    f"| {(time.time() - t0) / max(len(rows), 1):.2f}s/item")
        json.dump({"args": vars(a), "rows": rows}, open(os.path.join(a.out, f"{name}_{a.split}.json"), "w"))
        if save_h:
            torch.save(Hs, os.path.join(a.out, f"h_{name}_{a.split}.pt"))
        log(f"== {name}: done {len(rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
