"""H2 / KV-RAG: hierarchical chunked-operator read for super-long documents, judged through D20/D22 per-document
registration (scripts/register_d20.py, unmodified; this file patches in a question-dependent operator).

Operator of a question (static per-chunk tensors; the encoder never sees the question):
  layout    the D13 prefix of the document (chat header + "## Document\\n" + text; the last `tail` tokens are the
            reader's own rows, exactly as the flat read). Header = the tokens before the document body (S tokens).
            Body = token-aligned chunks of C tokens (scripts/h2_retrieval.py char spans, mapped onto prefix tokens).
  --src chunk  each chunk is encoded SEPARATELY by the 4B encoder as header + chunk body (positions 0..S+L-1, i.e.
            a short document inside the base's <=16k training regime); post-RoPE K/V at the paired encoder layers.
  --src flat   ablation: the flat D22 harvest of the whole prefix; chunk columns are sliced out of it.
  select    top-k chunks of the question by the chosen retriever (h2_retrieval rankings, or the D18 latent router
            via --phase route), laid out in DOCUMENT order after the shared header, keys RE-ROTATED (exact RoPE
            rebase R(d)R(p) = R(p+d)) to contiguous positions S, S+L1, ... -> n_op = S + sum L <= ~16k, the reader
            rows sit at pos0 = n_op exactly as in D13 training.
  read      the base D13 paper-form read (outputs/D13_4b_reg_w10) over those columns; per-document registration
            (D22 recipe) with each self-question reading ITS OWN retrieved chunks; distillation teacher = the
            encoder over header + the selected chunks' text (contiguous) continuing over the rows.
Judge: kv (KV-RAG, registered; kv@0 = the same read unregistered), wrong_doc (another doc's KV-RAG read: the
question retrieves inside the WRONG document's chunks, with that doc's registered adapter), closed_tail.
text / closed_book / flat kv come from the D22 run on the same items (scripts/h2_summary.py merges them);
--phase textrag adds text-RAG (the same retrieved chunks as TEXT in the 9B's context).

    python scripts/register_h2.py --d22_set outputs/D22_sets/clipper.json --retrieval outputs/H2_retrieval/clipper.json \
        --C 4096 --k 3 --retriever hybrid --src chunk <register_d20 flags> --out outputs/H2_clipper_c4k3
"""
from __future__ import annotations

import bisect
import copy
import json
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import fit_d13  # noqa: E402
import register_d20  # noqa: E402
from register_doc import SYSTEM_ANSWER, _get, log  # noqa: E402

CFG: dict = {}
LAST: dict = {}
_BaseRt = fit_d13.D13Rt


def base_key(k: str) -> str:
    """Self-question option-permutation copies (register_d20.shuffle_choices: <key>_p<j>) share the base query."""
    return re.sub(r"_p\d+$", "", k)


def load_rank(path: str, C: int, retriever: str) -> dict:
    R = json.load(open(path))
    rk = R["rank"][str(C)]
    if retriever not in rk:
        extra = path.replace(".json", "_router.json")
        assert os.path.exists(extra), f"retriever {retriever} not in {path} and no {extra}"
        rk = json.load(open(extra))["rank"][str(C)]
    return {"chunks": R["chunks"][str(C)], "rank": rk[retriever], "tail": R["tail"]}


def rebase(itf, K: torch.Tensor, delta: int) -> torch.Tensor:
    """Post-RoPE keys [Hkv,L,hd] at positions p -> positions p + delta (exact: RoPE is a rotation per frequency)."""
    if delta == 0:
        return K
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    x = K.float().unsqueeze(0)
    pos = torch.full((3, 1, 1), int(delta), device=K.device, dtype=torch.long)
    cos, sin = itf.rot(x, pos)
    k2, _ = apply_rotary_pos_emb(x, x, cos.float(), sin.float())
    return k2[0].to(K.dtype)


class KVRAGRt(_BaseRt):
    def __init__(self, reader, enc, tok, hub, itf, tail: int = 64, max_tokens: int = 16000, lru: int = 3):
        super().__init__(reader, enc, tok, hub, itf, tail, max_tokens, lru)
        c = CFG
        self.C, self.k, self.src, self.retriever = int(c["C"]), int(c["k"]), c["src"], c["retriever"]
        self.RR = load_rank(c["retrieval"], self.C, self.retriever)
        assert int(self.RR["tail"]) == int(tail), "retrieval chunks were cut with another tail"
        assert float(getattr(itf.rot, "attention_scaling", 1.0)) == 1.0, "RoPE rebase assumes unscaled rotary"
        self.cur = None
        self._lay: dict = {}
        self._store: OrderedDict = OrderedDict()
        self._sel_ops: OrderedDict = OrderedDict()
        self.sel_log: dict = {}
        self.hdr_diff = 0.0
        LAST["rt"] = self

    # ------------------------------------------------------------------ selection context
    def rows(self, did, it, target_ids=()):
        self.cur = it
        return super().rows(did, it, target_ids)

    def layout(self, did) -> dict:
        if did in self._lay:
            return self._lay[did]
        from xenc.state_skill import doc_prefix_text
        doc = self.docs[did]
        ps = doc_prefix_text(self.tok, self.hub, doc, SYSTEM_ANSWER)
        assert ps.endswith(doc), "document prefix does not end with the document text"
        d0 = len(ps) - len(doc)
        enc = self.tok(ps, add_special_tokens=False, return_offsets_mapping=True)
        ids = [int(x) for x in enc["input_ids"]]
        assert ids == self.pre(did), f"{did}: prefix truncated by max_tokens or tokenization mismatch"
        offs = [int(o[0]) for o in enc["offset_mapping"]]
        S = bisect.bisect_left(offs, d0)
        end = len(ids) - self.tail
        spans = self.RR["chunks"][did]
        b = [S] + [max(S, bisect.bisect_left(offs, d0 + int(cs))) for cs, _ in spans[1:]]
        ranges = [(b[j], b[j + 1] if j + 1 < len(b) else end) for j in range(len(b))]
        assert all(t1 > t0 for t0, t1 in ranges), f"{did}: empty chunk"
        self._lay[did] = {"S": S, "end": end, "ranges": ranges, "ids": ids}
        return self._lay[did]

    @torch.no_grad()
    def store(self, did) -> dict:
        """Per-document chunk store: header K/V + per-chunk body K/V at their encoded positions (pos0)."""
        if did in self._store:
            self._store.move_to_end(did)
            return self._store[did]
        L = self.layout(did)
        ids, S = L["ids"], L["S"]
        Tc = self.Tc
        hdr, bodies, pos0 = None, [], []
        sdev = self.edev        # chunk store on the ENCODER device (2-GPU runs keep the reader GPU for the 9B)
        t0 = time.time()
        if self.src == "chunk":
            for (a, b) in L["ranges"]:
                Tc.rec, Tc.kv_only = {}, True
                self.enc(input_ids=torch.tensor([ids[:S] + ids[a:b]], device=self.edev), use_cache=False,
                         logits_to_keep=1)
                rec = Tc.rec
                h = {e: (r["k"][:, :S], r["v"][:, :S]) for e, r in rec.items()}
                if hdr is None:
                    hdr = {e: (k.to(sdev).clone(), v.to(sdev).clone()) for e, (k, v) in h.items()}
                else:
                    self.hdr_diff = max(self.hdr_diff, max(float((h[e][0].to(sdev).float() - hdr[e][0].float())
                                                                 .abs().max()) for e in h))
                bodies.append({e: (r["k"][:, S:].to(sdev).contiguous(), r["v"][:, S:].to(sdev).contiguous())
                               for e, r in rec.items()})
                pos0.append(S)
        else:                                   # flat ablation: slice the D22 harvest
            Tc.rec, Tc.kv_only = {}, True
            self.enc(input_ids=torch.tensor([ids[: L["end"]]], device=self.edev), use_cache=False, logits_to_keep=1)
            rec = Tc.rec
            hdr = {e: (r["k"][:, :S].to(sdev), r["v"][:, :S].to(sdev)) for e, r in rec.items()}
            for (a, b) in L["ranges"]:
                bodies.append({e: (r["k"][:, a:b].to(sdev), r["v"][:, a:b].to(sdev)) for e, r in rec.items()})
                pos0.append(a)
        Tc.rec, Tc.kv_only = None, False
        self._store[did] = {"hdr": hdr, "bodies": bodies, "pos0": pos0, "S": S}
        while len(self._store) > 2:
            self._store.popitem(last=False)
            self._sel_ops.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        log(f"    [H2] stored {did}: {len(bodies)} chunks of C={self.C} ({self.src}), header S={S}, "
            f"{time.time() - t0:.1f}s, header max|dK| across chunks {self.hdr_diff:.2e}")
        return self._store[did]

    def select(self, did) -> list[int]:
        assert self.cur is not None, "operator() before rows(): no question context"
        qid = f"{did}|{base_key(_get(self.cur, 'key'))}"
        assert qid in self.RR["rank"], f"no {self.retriever} ranking for {qid}"
        sel = sorted(int(c) for c in self.RR["rank"][qid][: self.k])
        self.sel_log[qid] = sel
        return sel

    @torch.no_grad()
    def operator(self, did):
        sel = self.select(did)
        key = (did, tuple(sel))
        if key in self._sel_ops:
            self._sel_ops.move_to_end(key)
            return self._sel_ops[key]
        st = self.store(did)
        op = {}
        for e in st["hdr"]:
            Ks, Vs, pos = [st["hdr"][e][0].to(self.dev)], [st["hdr"][e][1].to(self.dev)], st["S"]
            for j in sel:
                Kb, Vb = st["bodies"][j][e]
                Ks.append(rebase(self.itf, Kb.to(self.dev), pos - st["pos0"][j]))
                Vs.append(Vb.to(self.dev))
                pos += Kb.shape[1]
            op[e] = (torch.cat(Ks, 1), torch.cat(Vs, 1))
        n_op = next(iter(op.values()))[0].shape[1]
        self._sel_ops[key] = (op, n_op, sel)
        while len(self._sel_ops) > max(self.lru, 3):
            self._sel_ops.popitem(last=False)
        return self._sel_ops[key]

    @torch.no_grad()
    def teacher(self, did, rows) -> dict:
        """Encoder over header + the selected chunks' tokens (contiguous, positions 0..n_op-1), then the rows."""
        _, n_op, sel = self.operator(did)
        L = self.layout(did)
        ids = L["ids"][: L["S"]] + [t for j in sel for t in L["ids"][L["ranges"][j][0]: L["ranges"][j][1]]]
        assert len(ids) == n_op, (len(ids), n_op)
        self.Tc.rec = None
        out = self.enc(input_ids=torch.tensor([ids], device=self.edev), use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        del out
        self.Tc.rec, self.Tc.kv_only = {}, False
        self.enc(input_ids=torch.tensor([rows], device=self.edev), past_key_values=cache, use_cache=True,
                 logits_to_keep=1)
        rec = {e: {k: v.to(self.dev) for k, v in r.items()} for e, r in self.Tc.rec.items()}
        self.Tc.rec = None
        del cache
        return rec


class H2D20(register_d20.D20):
    def loss(self, did, x):
        self.rt.cur = x["it"]
        return super().loss(did, x)

    @torch.no_grad()
    def ho_loss(self, did, H) -> float:
        """register_d20.D20.ho_loss with the operator re-selected for every held-out self-question."""
        from register_state import kept_logits
        from xenc.losses import answer_ce, kl_teacher_student
        rt, R = self.rt, self.rt.R
        tot = 0.0
        assert not self.a.enc_only
        for x in H:
            rt.cur = x["it"]
            op, n_op, _ = rt.operator(did)
            m = len(x["tt"])
            R.load(self.ad.transform_op(op), n_op)
            R.mode, R.need_logits, R.rec = "e2e", False, {}
            try:
                s_log = kept_logits(self.reader, torch.tensor([x["rows"]]), m + 1)
            finally:
                R.mode, R.rec = "off", {}
            tot += float(kl_teacher_student(x["t_log"].float(), s_log[:-1])) + 0.5 * float(answer_ce(s_log, x["tt"]))
        return tot / max(len(H), 1)


# ============================================================================ phases of this file
def set_items(set_path: str, smoke: int = 0):
    S = json.load(open(set_path))
    keys = set(S["item_keys"])
    from register_doc import load_split_items
    splits = S.get("splits") or ["test"]
    items = [it for s in splits for it in load_split_items(S["split_dir"], s) if _get(it, "key") in keys]
    assert len(items) == len(keys), f"{len(items)} of {len(keys)} set items found"
    if smoke:
        idx = json.load(open(Path(S["split_dir"]) / "docs.json"))
        d0 = max(sorted({_get(i, "doc_id") for i in items}), key=lambda d: idx[d].get("n_tokens", 0))
        items = [i for i in items if _get(i, "doc_id") == d0][:smoke]
    return S, items


def patch_judge(set_path, smoke):
    def judge_items(a):
        assert not a.n and not a.max_docs and a.fold is None, "the set fixes the sample: leave --n/--max_docs/--fold unset"
        _, items = set_items(set_path, smoke)
        return items, sorted({_get(it, "doc_id") for it in items})
    register_d20.judge_items = judge_items


def phase_textrag(a, rest):
    """text-RAG: the 9B reads the SAME retrieved chunks (document order) + the doc tail as text."""
    from register_state import greedy
    from xenc.backbone import build_chat, load_reader
    from xenc.doc_tasks import ctx_prompt, load_docs, score
    S, items = set_items(a.d22_set, a.smoke)
    docs = load_docs(S["split_dir"])
    RR = load_rank(a.retrieval, a.C, a.retriever)
    reader, tok = load_reader(a.reader, device_map=a.device)
    out_rows, t0 = [], time.time()
    path = os.path.join(a.out, f"eval_textrag_{a.retriever}_C{a.C}_k{a.k}.jsonl")
    done = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            done[r["key"]] = r
    f = open(path, "a")
    for j, it in enumerate(sorted(items, key=lambda i: _get(i, "doc_id"))):
        did, key = _get(it, "doc_id"), _get(it, "key")
        if key in done:
            out_rows.append(done[key])
            continue
        doc = docs[did]
        sel = sorted(int(c) for c in RR["rank"][f"{did}|{key}"][: a.k])
        spans = RR["chunks"][did]
        tail_txt = tok.decode(tok(doc, add_special_tokens=False)["input_ids"][-a.tail:])
        ctx = "\n\n[...]\n\n".join(doc[spans[c][0]: spans[c][1]].strip() for c in sel) + "\n\n[...]\n\n" + tail_txt
        g, n = greedy(reader, tok, a.reader, build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, ctx)), a.budget)
        r = {"key": key, "doc_id": did, "sel": sel, "text_rag": {"gen": g, "score": float(score(it, g)), "gen_tokens": n},
             "ctx_tokens": len(tok(ctx, add_special_tokens=False)["input_ids"])}
        f.write(json.dumps(r) + "\n")
        f.flush()
        out_rows.append(r)
        if (j + 1) % 50 == 0:
            log(f"  textrag [{j + 1}/{len(items)}] mean {sum(x['text_rag']['score'] for x in out_rows) / len(out_rows):.3f}"
                f" | {(time.time() - t0) / max(len(out_rows), 1):.1f}s/item")
    log(f"text_rag ({a.retriever} C={a.C} k={a.k}) n={len(out_rows)}: "
        f"{sum(x['text_rag']['score'] for x in out_rows) / max(len(out_rows), 1):.4f}, mean ctx tokens "
        f"{sum(x['ctx_tokens'] for x in out_rows) / max(len(out_rows), 1):.0f}")
    return 0


@torch.no_grad()
def phase_route(a, rest):
    """Latent router (D18B4b_router, trained for cross-DOCUMENT routing) scoring the chunks of a document: every query
    of the retrieval file -> ranking. Writes <retrieval>_router.json with rank[C]['router'] and 'hybrid_r' (RRF of
    router, bm25, dense)."""
    import argparse
    from h2_retrieval import rrf
    from xenc.d18_router import D18Router, HiddenCap
    from xenc.doc_tasks import load_docs
    from xenc.state_skill import doc_prefix_text
    S, items = set_items(a.d22_set, a.smoke)
    R = json.load(open(a.retrieval))
    b = argparse.Namespace(reader=a.reader, encoder=a.encoder, B=a.B, layers=a.layers, device=a.device,
                           enc_device=a.enc_device or "")
    reader, enc, tok, itf, lmap = fit_d13.build(b, init=a.interface)
    if itf.delta_rank:
        itf.merge_delta()
    CFG.update(C=a.C, k=a.k, src=a.src, retriever="bm25", retrieval=a.retrieval)
    rt = KVRAGRt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens)
    rt.docs = load_docs(S["split_dir"])
    router = D18Router.load(a.router, itf)[0].to(rt.dev).eval()
    cap = HiddenCap(reader, itf.layers)
    header = [int(x) for x in tok(doc_prefix_text(tok, a.reader, "", SYSTEM_ANSWER), add_special_tokens=False)["input_ids"]]
    qtext = {}
    from register_doc import load_split_items
    sel_items = {}
    for s in (S.get("splits") or ["test"]):
        for it in load_split_items(S["split_dir"], s):
            sel_items[_get(it, "key")] = it
    for line in (open(a.selfq) if a.selfq and os.path.exists(a.selfq) else []):
        for it in json.loads(line)["items"]:
            sel_items[it["key"]] = it
    qids = list(R["rank"][str(a.C)]["bm25"])
    if a.smoke:
        dd = {_get(i, "doc_id") for i in items}
        qids = [q for q in qids if q.split("|")[0] in dd][: 4 * a.smoke]
    by_doc: dict = {}
    for q in qids:
        by_doc.setdefault(q.split("|")[0], []).append(q)
    out = {}
    t0 = time.time()
    for did, qs in sorted(by_doc.items()):
        st = rt.store(did)
        Kc = {e: [torch.cat([st["hdr"][e][0].to(rt.dev),
                             rebase(itf, st["bodies"][j][e][0].to(rt.dev), st["S"] - st["pos0"][j])], 1)
                  for j in range(len(st["bodies"]))] for e in st["hdr"]}
        skip = min(st["S"], len(header))
        for q in qs:
            it = sel_items[q.split("|", 1)[1]]
            rows = header + rt.suf(it)
            cap.h, cap.on = {}, True
            try:
                reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=False, logits_to_keep=1)
            finally:
                cap.on = False
            rxs = {li: h[0, len(header):].float() for li, h in cap.h.items()}
            rpos = torch.arange(len(header), len(rows), device=rt.dev)
            sco = router(rxs, rpos, Kc, skip).float()
            out[q] = sorted(range(len(sco)), key=lambda c: -float(sco[c]))[:32]
        log(f"  routed {did}: {len(qs)} queries ({time.time() - t0:.0f}s)")
    rk = R["rank"][str(a.C)]
    hyb = {q: rrf(out[q], rk["bm25"][q], rk["dense"][q])[:32] for q in out if q in rk.get("dense", {})}
    path = a.retrieval.replace(".json", "_router.json")
    prev = json.load(open(path)) if os.path.exists(path) else {"rank": {}}
    prev["rank"][str(a.C)] = {"router": out, "hybrid_r": hyb}
    prev["router"] = a.router
    json.dump(prev, open(path, "w"))
    log(f"router rankings -> {path} ({len(out)} queries, C={a.C})")
    return 0


@torch.no_grad()
def phase_sanity(a, rest):
    """(1) RoPE rebase == encoding at shifted positions; (2) chunk-source header identical across chunks;
    (3) the selected operator's length / positions; (4) one KV-RAG read differs from the flat read."""
    import argparse
    from xenc.doc_tasks import load_docs
    S, items = set_items(a.d22_set, 1)
    b = argparse.Namespace(reader=a.reader, encoder=a.encoder, B=a.B, layers=a.layers, device=a.device,
                           enc_device=a.enc_device or "")
    reader, enc, tok, itf, lmap = fit_d13.build(b, init=a.interface)
    CFG.update(C=a.C, k=a.k, src="chunk", retriever=a.retriever, retrieval=a.retrieval)
    rt = KVRAGRt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens)
    rt.docs = load_docs(S["split_dir"])
    did = _get(items[0], "doc_id")
    L = rt.layout(did)
    ids, Ssz = L["ids"], L["S"]
    a0, b0 = L["ranges"][3]
    delta = 5000
    Tc = rt.Tc
    Tc.rec, Tc.kv_only = {}, True
    enc(input_ids=torch.tensor([ids[:Ssz] + ids[a0:b0]], device=rt.edev), use_cache=False, logits_to_keep=1)
    ref = {e: r["k"][:, Ssz:].to(rt.dev) for e, r in Tc.rec.items()}
    pos = list(range(Ssz)) + list(range(Ssz + delta, Ssz + delta + (b0 - a0)))
    Tc.rec = {}
    enc(input_ids=torch.tensor([ids[:Ssz] + ids[a0:b0]], device=rt.edev),
        position_ids=torch.tensor([pos], device=rt.edev).view(1, 1, -1).expand(3, 1, -1), use_cache=False,
        logits_to_keep=1)   # (3,1,T): mrope rows only, so the causal mask is built from cache positions
    sh = {e: r["k"][:, Ssz:].to(rt.dev) for e, r in Tc.rec.items()}
    Tc.rec, Tc.kv_only = None, False
    for e in ref:
        rb = rebase(itf, ref[e], delta).float()
        err = float((rb - sh[e].float()).abs().max())
        rel = float((rb - sh[e].float()).norm() / sh[e].float().norm())
        base = float((ref[e].float() - sh[e].float()).norm() / sh[e].float().norm())
        log(f"[sanity] enc layer {e}: rebase vs shifted-position encoding max|d| {err:.3e} rel {rel:.2e} "
            f"(un-rebased rel {base:.2e})")
        assert rel < 2e-2, "RoPE rebase does not match encoding at shifted positions"
    rt.cur = items[0]
    op, n_op, sel = rt.operator(did)
    log(f"[sanity] {did}: {len(L['ranges'])} chunks, S={Ssz}, selected {sel}, n_op {n_op}, header diff "
        f"{rt.hdr_diff:.2e}")
    assert n_op == Ssz + sum(L["ranges"][j][1] - L["ranges"][j][0] for j in sel)
    log("[sanity] PASS")
    return 0


def main(argv=None) -> int:
    import argparse
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--d22_set", required=True)
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--C", type=int, default=4096)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--retriever", default="hybrid")
    ap.add_argument("--src", default="chunk", choices=("chunk", "flat"))
    ap.add_argument("--router", default="outputs/D18B4b_router/router_best.pt")
    ap.add_argument("--smoke", type=int, default=0, help="N>0: the longest document only, its first N items")
    ap.add_argument("--h2_phase", default="", help="textrag | route | sanity (else: register_d20 phases)")
    a, rest = ap.parse_known_args(argv)
    # flags shared with register_d20 that the own phases need
    sp = argparse.ArgumentParser(add_help=False)
    for k_, d_ in (("--reader", "Qwen/Qwen3.5-9B"), ("--encoder", "Qwen/Qwen3.5-4B"),
                   ("--B", "outputs/D11_maps/encB5_9b_4b_post_xf01.pt"), ("--layers", "3,7,11,15,19"),
                   ("--interface", "outputs/D13_4b_reg_w10/d13_best.pt"), ("--out", None), ("--device", "cuda"),
                   ("--enc_device", None), ("--selfq", None)):
        sp.add_argument(k_, default=d_)
    for k_, d_ in (("--tail", 64), ("--max_tokens", 128000), ("--budget", 32)):
        sp.add_argument(k_, type=int, default=d_)
    b, _ = sp.parse_known_args(rest)
    for k_, v_ in vars(b).items():
        setattr(a, k_, v_)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    if a.h2_phase:
        return {"textrag": phase_textrag, "route": phase_route, "sanity": phase_sanity}[a.h2_phase](a, rest)
    CFG.update(C=a.C, k=a.k, src=a.src, retriever=a.retriever, retrieval=a.retrieval)
    log(f"H2 KV-RAG: set {a.d22_set} | retrieval {a.retrieval} | C {a.C} k {a.k} retriever {a.retriever} src {a.src}"
        f" | smoke {a.smoke}")
    patch_judge(a.d22_set, a.smoke)
    fit_d13.D13Rt = KVRAGRt
    register_d20.D20 = H2D20
    try:
        rc = register_d20.main(rest)
    finally:
        fit_d13.D13Rt = _BaseRt
    if torch.cuda.is_available():
        log(f"H2: peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
            f"({torch.cuda.get_device_name(0)})")
    rt = LAST.get("rt")
    if rt is not None and a.out:
        ph = "eval" if "eval" in (rest[rest.index("--phase") + 1] if "--phase" in rest else "register,eval") else "reg"
        p = os.path.join(a.out, f"selections_{ph}.json")
        old = json.load(open(p)) if os.path.exists(p) else {}
        old.setdefault("sel", {}).update(rt.sel_log)
        old["meta"] = {"C": a.C, "k": a.k, "retriever": a.retriever, "src": a.src, "header_diff": rt.hdr_diff}
        json.dump(old, open(p, "w"))
        log(f"H2: {len(rt.sel_log)} selections -> {p}; chunk-source header max|dK| {rt.hdr_diff:.2e}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
