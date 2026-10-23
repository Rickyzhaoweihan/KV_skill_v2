"""D13 stage 1: fit the reader-query map A of the paper's read (xenc/d13_read.py) to the ENCODER's OWN question-pass
attention — label-free (questions are inputs; no answers are used).

For every training document (QuALITY / MuSiQue / LastingBench train, LongHealth folds ∉ {0, --val_fold}):
  operator   encoder prefill of the document prefix minus the last `tail` tokens -> post-RoPE K/V at the paired
             encoder layers (fixed harvest)
  rows       the last `tail` document tokens + the question suffix (the reader's own prompt)
  teacher    the encoder continuing over those rows: per-head attention over [operator ; own rows] and its post-o_proj
             output o_t (B o_t is exactly what the passing encoder-side read injects)
  student    the reader's own layer input at the paired reader layer -> A -> the encoder's frozen q/k/v/gate -> read of
             the operator -> B (D10's encoder-side post B, fixed)
1) ridge: A by least squares from the reader's layer input to the encoder's layer input at the same rows (closed form;
   rows teacher-forced: upstream reader layers receive the encoder-side read). This is the ridge baseline.
2) A trained by KL(p_teacher ‖ p_student) + λ·relMSE(B o_s, B o_t); later epochs partly on-policy (upstream layers
   receive the student's own read). Variants: --variants lin,mlp (affine A; affine + residual 2-layer MLP).
Selection on HELD-OUT documents: on-policy read-out cosine (reported with attention JS, operator-mass, top-5).

    python scripts/fit_d13.py --out outputs/D13_maps --B outputs/D10_maps/encB8_9b_2b_post.pt

Shared runtime (D13Rt: operator / teacher passes, greedy decode with the read) is imported by register_d13.py and
eval_d13.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402

TASK_SPLIT = {"quality": "data/quality_split", "musique": "data/musique_split",
              "lastingbench": "data/lastingbench_split", "longhealth": "data/longhealth_split",
              "loogle": "data/loogle_split", "clipper": "data/clipper_split",   # long-operator pools (D13 long)
              "qasper": "data/qasper_split", "contractnli": "data/contractnli_split",
              "shakespeare_cf": "data/shakespeare_cf_split"}                     # BEAT_TEXT lever 1


# ============================================================================ data
def select_items(spec: str, split: str, folds: set | None = None, q_per_doc: int = 0, seed: int = 0):
    """spec "quality:150,musique:150,longhealth:0" (0 = all docs) -> (items, docs {doc_id: text}).
    LongHealth items are restricted to `folds` (document-level folds)."""
    from xenc.doc_tasks import load_docs
    rng = random.Random(seed)
    items, docs = [], {}
    for s in spec.split(","):
        if not s:
            continue
        name, nd = s.split(":")
        sd = TASK_SPLIT[name]
        dd = load_docs(sd)
        its = load_split_items(sd, split)
        if name == "longhealth" and folds is not None:
            its = [i for i in its if (_get(i, "meta") or {}).get("fold") in folds]
        by = OrderedDict()
        for it in its:
            by.setdefault(_get(it, "doc_id"), []).append(it)
        dids = sorted(by)
        if int(nd):
            dids = dids[: int(nd)]
        for d in dids:
            qs = by[d]
            if q_per_doc and len(qs) > q_per_doc:
                qs = rng.sample(qs, q_per_doc)
            items += qs
            docs[d] = dd[d]
    return items, docs


# ============================================================================ runtime
class D13Rt:
    def __init__(self, reader, enc, tok, hub, itf, tail: int = 64, max_tokens: int = 16000, lru: int = 3):
        from xenc.d13_read import D13Reader, EncTeacher
        self.reader, self.enc, self.tok, self.hub, self.itf = reader, enc, tok, hub, itf
        self.tail, self.max_tokens = int(tail), int(max_tokens)
        enc_layers = itf.enc_layers() if hasattr(itf, "enc_layers") else sorted(set(itf.layer_map.values()))
        self.Tc = EncTeacher(enc, enc_layers)
        self.R = itf.make_reader(reader) if hasattr(itf, "make_reader") else D13Reader(reader, itf)   # E2: E2Reader
        self.docs: dict = {}
        self._pre: dict = {}
        self._ops: OrderedDict = OrderedDict()
        self.lru = int(lru)
        from xenc.backbone import input_device
        self.dev, self.edev = input_device(reader), input_device(enc)

    def pre(self, did) -> list[int]:
        if did not in self._pre:
            from xenc.state_skill import doc_prefix_text
            ids = [int(x) for x in self.tok(doc_prefix_text(self.tok, self.hub, self.docs[did], SYSTEM_ANSWER),
                                            add_special_tokens=False)["input_ids"]]
            self._pre[did] = ids[-(self.max_tokens - 1024):]
        return self._pre[did]

    def suf(self, it) -> list[int]:
        from xenc.state_skill import question_suffix_ids
        return question_suffix_ids(self.tok, self.hub, it, SYSTEM_ANSWER)

    def rows(self, did, it, target_ids=()) -> list[int]:
        return self.pre(did)[-self.tail:] + self.suf(it) + list(target_ids)

    @torch.no_grad()
    def operator(self, did):
        """-> (op {enc_layer: (K, V) [Hkv,N,hd]}, N, encoder cache after the operator prefix)."""
        if did in self._ops:
            self._ops.move_to_end(did)
            return self._ops[did]
        ids = self.pre(did)[: -self.tail] if self.tail else self.pre(did)
        self.Tc.rec, self.Tc.kv_only = {}, True
        hk = self.itf.harvest_hooks(self.enc) if hasattr(self.itf, "harvest_hooks") else None   # E2 deep values
        out = self.enc(input_ids=torch.tensor([ids], device=self.edev), use_cache=True, logits_to_keep=1)
        op = {e: (r["k"].to(self.dev), r["v"].to(self.dev)) for e, r in self.Tc.rec.items()}
        if hk is not None:
            op.update(hk.finish(self.dev))
        if getattr(self.itf, "needs_ids", False):              # P3 token channel: column c's token id (= ids[c])
            op["ids"] = torch.tensor(ids, dtype=torch.long, device=self.dev)
            assert op["ids"].shape[0] == next(iter(self.Tc.rec.values()))["k"].shape[1], "ids / operator misaligned"
        if getattr(self.itf, "needs_seg", False):              # ADDR SEG: paragraph / section ids per column
            from xenc.addr_seg import seg_for_doc
            op["seg"] = seg_for_doc(self.tok, self.hub, self.docs[did], SYSTEM_ANSWER, self.pre(did), len(ids),
                                    int(self.itf.seg.get("block", 128)))
        self.Tc.rec, self.Tc.kv_only = None, False
        self._ops[did] = (op, len(ids), out.past_key_values)
        while len(self._ops) > self.lru:
            self._ops.popitem(last=False)
        return self._ops[did]

    @torch.no_grad()
    def teacher(self, did, rows) -> dict:
        """Encoder continuing over `rows` from the operator prefix: {enc_layer: {x, q, k, v, out}}."""
        _, _, cache = self.operator(did)
        self.Tc.rec, self.Tc.kv_only = {}, False
        self.enc(input_ids=torch.tensor([rows], device=self.edev), past_key_values=copy.deepcopy(cache),
                 use_cache=True, logits_to_keep=1)
        rec = {e: {k: v.to(self.dev) for k, v in r.items()} for e, r in self.Tc.rec.items()}
        self.Tc.rec = None
        return rec

    def attn_terms(self, rec, op, n_op, y_t: dict, metrics=False) -> dict:
        """Per reader layer d13_terms between R.rec (student) and the encoder teacher."""
        from xenc.d13_read import d13_terms, teacher_logits
        out = {}
        for li, e in self.itf.layer_map.items():
            r = self.R.rec.get(li)
            if r is None or r.get("logits") is None:
                continue
            lt = teacher_logits(rec[e]["q"], op[e][0], rec[e]["k"], self.itf.att[str(li)].scaling)
            if int(getattr(self, "kl_chunk", 0) or 0) > 0 and not metrics:   # F1 --dist_chunk (default off)
                from xenc.d13_read import d13_terms_chunked
                out[li] = d13_terms_chunked(r["logits"], lt, r["y"], y_t[li], n_op, int(self.kl_chunk))
            elif (not getattr(self, "ckpt", False) or metrics or not torch.is_grad_enabled()
                    or not r["logits"].requires_grad):
                out[li] = d13_terms(r["logits"], lt, r["y"], y_t[li], n_op, metrics)
            else:                      # opt-in (ported from D12p): recompute softmax/KL intermediates in backward
                from torch.utils.checkpoint import checkpoint
                yt = y_t[li]

                def _f(ls, ys, lt=lt, yt=yt):
                    d = d13_terms(ls, lt, ys, yt, n_op, False)
                    return d["kl"], d["rel"]
                kl, rel = checkpoint(_f, r["logits"], r["y"], use_reentrant=False)
                out[li] = {"kl": kl, "rel": rel}
            del lt
        return out

    def y_teacher(self, rec) -> dict:
        with torch.no_grad():
            return {li: self.itf.Bmap(li, rec[e]["out"].float()).detach() for li, e in self.itf.layer_map.items()}

    def s1_pass(self, did, it, onpolicy=False, grad=True, metrics=False, y_ref: dict | None = None):
        """One stage-1 pass over one question -> per-layer terms (kl, rel [, diagnostics])."""
        op, n_op, _ = self.operator(did)
        rows = self.rows(did, it)
        rec = self.teacher(did, rows)
        R = self.R
        R.load(op, n_op)
        yt = self.y_teacher(rec)
        R.mode, R.onpolicy, R.grad, R.inject, R.rec = "train", bool(onpolicy), bool(grad), yt, {}
        try:
            with torch.no_grad():
                self.reader(input_ids=torch.tensor([rows], device=self.dev), use_cache=False, logits_to_keep=1)
            return self.attn_terms(rec, op, n_op, y_ref or yt, metrics)
        finally:
            R.mode, R.rec, R.inject = "off", {}, {}

    @torch.no_grad()
    def gen(self, did, it, budget: int = 32, stop: str = "</answer>", read: bool = True):
        """Greedy decode from the reader prompt (doc tail + question) with the D13 read of did's operator
        (read=False: closed_tail)."""
        from register_state import _stop_ids
        from xenc.backbone import decode
        rows = self.rows(did, it)
        R = self.R
        if read:
            op, n_op, _ = self.operator(did)
            R.load(op, n_op)
            R.mode = "student"
        try:
            out = self.reader(input_ids=torch.tensor([rows], device=self.dev), use_cache=True, logits_to_keep=1)
            stops, new = _stop_ids(self.tok), []
            for _ in range(budget):
                nxt = int(out.logits[0, -1].argmax())
                new.append(nxt)
                if nxt in stops or stop in self.tok.decode(new[-12:], skip_special_tokens=True):
                    break
                out = self.reader(input_ids=torch.tensor([[nxt]], device=self.dev),
                                  past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
        finally:
            R.mode, R.past = "off", {}
        txt = decode(self.tok, self.hub, new)
        if stop in txt:
            txt = txt.split(stop, 1)[0] + stop
        return txt, len(new)


def extend_itf(itf, enc, lmap_full: dict, B_full, src: int):
    """Lever 2a (ported from D12p extend_itf): a D13 interface over lmap_full. Existing layers keep A / MLP / B / bias /
    gate; each NEW layer copies A, MLP and write-back bias from layer `src`, takes its own encoder-side B from B_full
    (if given, else src's B), and starts at gate 0 -- so the extended read equals the init's exactly. The new layers
    are recorded in itf.new_layers (register_d13.py --lr_g_new)."""
    from xenc.d13_read import D13Interface
    assert itf.delta_rank == 0
    if itf.identity_B:
        Bd = None
    else:
        Bd = {li: (itf.B[str(li)].detach().cpu().clone() if li in itf.layer_map
                   else (B_full[li] if B_full is not None and li in B_full else itf.B[str(src)].detach().cpu().clone()))
              for li in lmap_full}
    new = D13Interface(enc, lmap_full, itf.d_r, itf.d_e, B=Bd, mlp_hidden=itf.mlp_hidden)
    with torch.no_grad():
        for li in lmap_full:
            s_ = li if li in itf.layer_map else src
            new.A[str(li)].load_state_dict(itf.A[str(s_)].state_dict())
            if itf.mlp_hidden:
                new.M[str(li)].load_state_dict(itf.M[str(s_)].state_dict())
            if not itf.identity_B:
                new.Bb[str(li)].copy_(itf.Bb[str(s_)].detach().cpu())
            new.g[new._gi[li]] = float(itf.g[itf._gi[li]]) if li in itf.layer_map else 0.0
        if getattr(itf, "att_trained", False):       # CAP: keep trained read projections on the existing layers
            new.att_trained = True
            for li in itf.layer_map:
                new.att[str(li)].load_state_dict(itf.att[str(li)].state_dict())
    if getattr(itf, "mlp_B", 0):                      # CAP: keep the write-back MLP; new layers start at zero output
        new.add_mlp_B(itf.mlp_B)
        with torch.no_grad():
            for li in itf.layer_map:
                new.MB[str(li)].load_state_dict(itf.MB[str(li)].state_dict())
    new.new_layers = sorted(set(lmap_full) - set(itf.layer_map))
    return new


def widen_mlp(itf, hidden: int, seed: int = 0):
    """Lever 2b: widen every residual query MLP to `hidden` units. The old units are copied; the new units get a random
    input projection and ZERO output weights, so the widened read equals the old one exactly at step 0."""
    import torch.nn as nn
    g = torch.Generator().manual_seed(seed)
    old = itf.mlp_hidden
    itf.mlp_hidden_init = old
    for k, m in itf.M.items():
        n = nn.Sequential(nn.Linear(itf.d_r, hidden), nn.GELU(), nn.Linear(hidden, itf.d_e)).to(m[0].weight.device)
        with torch.no_grad():
            n[0].weight.copy_((torch.randn(hidden, itf.d_r, generator=g) / itf.d_r ** 0.5).to(n[0].weight))
            n[0].bias.zero_()
            n[0].weight[:old].copy_(m[0].weight)
            n[0].bias[:old].copy_(m[0].bias)
            n[2].weight.zero_()
            n[2].weight[:, :old].copy_(m[2].weight)
            n[2].bias.copy_(m[2].bias)
        itf.M[k] = n
    itf.mlp_hidden = int(hidden)


def build(a, init: str | None = None, mlp_hidden: int = 0):
    """Load reader + encoder, D13 interface (identity A, or `init` file), B from --B."""
    from xenc.backbone import load_encoder, load_reader
    from xenc.d13_read import D13Interface, load_B, load_d13
    from xenc.state_skill import text_config
    from xenc.xattn_read import attn_dims, full_depth_map
    reader, tok = load_reader(a.reader, device_map=a.device)
    if getattr(a, "share_encoder", 0):
        # C2: the reader's own weights serve as the encoder (one instance). The encoder passes run with the read
        # patch in mode "off" and EncTeacher chained under it (xenc/d13_read.py), so the operator is the plain
        # prefill K/V (tests/test_d13.py::test_shared_encoder_instance_matches_separate_models).
        assert a.encoder == a.reader, "--share_encoder needs --encoder == --reader"
        enc = reader
    else:
        enc = load_encoder(a.encoder, device_map=getattr(a, "enc_device", "") or a.device)[0]
    keep = {int(x) for x in a.layers.split(",")}
    lmap = {k: v for k, v in full_depth_map(attn_dims(reader), attn_dims(enc)).items() if k in keep}
    B = load_B(a.B, sorted(lmap))
    if init:
        itf, _ = load_d13(init, enc, B)
        new = sorted(set(lmap) - set(itf.layer_map))
        if new:                                    # lever 2a: extend the init to deeper reader layers (gate 0)
            itf = extend_itf(itf, enc, lmap, B, int(getattr(a, "extend_src", 0) or max(itf.layer_map)))
        if int(getattr(a, "mlp_widen", 0) or 0) > itf.mlp_hidden:
            widen_mlp(itf, int(a.mlp_widen))
    else:
        itf = D13Interface(enc, lmap, int(text_config(reader).hidden_size), int(text_config(enc).hidden_size),
                           B=B, mlp_hidden=mlp_hidden)
    e2_dv, e2_ms, e2_rs = getattr(a, "e2_dv", ""), getattr(a, "e2_ms", ""), getattr(a, "e2_rs", "")
    e2_tr = int(getattr(a, "e2_tr", 0) or 0)
    if e2_dv or e2_ms or e2_rs or e2_tr:                    # E2 (xenc/e2_read.py): zero-init deep-value / extra-source mixing
        from xenc.e2_read import load_B2, parse_pairs, upgrade
        dv, ms, rs = parse_pairs(e2_dv), parse_pairs(e2_ms), parse_pairs(e2_rs)
        assert not hasattr(itf, "dv"), "--e2_* on an init that already is an E2 interface"
        itf = upgrade(itf, enc, dv, ms, load_B2(getattr(a, "e2_B2", ""), ms), rs, load_B2(getattr(a, "e2_B2", ""), rs),
                      bool(e2_tr))
    tok_layers = str(getattr(a, "tok_layers", "") or "")
    if tok_layers and not hasattr(itf, "tok_layers"):          # P3 token channel (xenc/p3_tok.py): zero-init U
        from xenc.p3_tok import parse_tok, upgrade_tok
        tcfg = parse_tok(tok_layers, int(getattr(a, "tok_rank", 256)), getattr(a, "tok_heads", "kv"),
                         getattr(a, "tok_shifts", "-1,0,1"), int(getattr(a, "tok_topk", 0)),
                         int(getattr(a, "tok_topk_norm", 0) or 0), float(getattr(a, "tok_temp", 1.0) or 1.0),
                         str(getattr(a, "tok_src", "emb") or "emb"), str(getattr(a, "tok_adapter", "lin") or "lin"),
                         int(getattr(a, "tok_mlp_hidden", 0) or 0), str(getattr(a, "tok_inj_layers", "") or ""),
                         float(getattr(a, "tok_floor", 0.0) or 0.0))
        from xenc.backbone import input_device as _idev
        Etab = reader.get_output_embeddings().weight if tcfg["src"] == "unemb" else reader.get_input_embeddings().weight
        itf = upgrade_tok(itf.to(_idev(reader)), enc, tcfg, Etab)
    elif tok_layers:
        assert sorted(int(x) for x in tok_layers.split(",")) == itf.tok_layers, "init already has another token channel"
    if hasattr(itf, "tok_src") and itf.tok_src == "unemb" and itf.tok_maps is None:   # P6: first-token variant maps
        from xenc.p4_ptr import vocab_maps
        _ns, _gm, _gt, _ans = vocab_maps(tok, reader.get_output_embeddings().weight.shape[0])
        itf.set_tok_maps(_ns, _gm, _gt)
    l1_on = int(getattr(a, "l1_logn", 0) or 0) or int(getattr(a, "l1_chunk", 0) or 0) or str(getattr(a, "l1_sel", "") or "")
    if l1_on and not hasattr(itf, "l1"):                       # L1 length-aware read forms (xenc/l1_read.py): σ = c = 0
        from xenc.l1_read import parse_l1, upgrade_l1
        itf = upgrade_l1(itf, enc, parse_l1(a.l1_logn, a.l1_nref, a.l1_chunk, a.l1_sel, a.l1_k, a.l1_block, a.l1_nblk))
    elif l1_on:
        from xenc.l1_read import parse_l1
        assert parse_l1(a.l1_logn, a.l1_nref, a.l1_chunk, a.l1_sel, a.l1_k, a.l1_block, a.l1_nblk) == itf.l1, \
            "init already has another L1 read form"
    seg_on = int(getattr(a, "seg", 0) or 0)
    if seg_on and not hasattr(itf, "seg"):                     # ADDR SEG structure-aware keys / values: W = 0
        from xenc.addr_seg import parse_seg, upgrade_seg
        itf = upgrade_seg(itf, enc, parse_seg(seg_on, getattr(a, "seg_block", 128), getattr(a, "seg_parts", "kP,kS,vP,vS")))
    elif seg_on:
        from xenc.addr_seg import parse_seg
        assert parse_seg(seg_on, getattr(a, "seg_block", 128), getattr(a, "seg_parts", "kP,kS,vP,vS")) == itf.seg, \
            "init already has another SEG config"
    from xenc.r2_read import parse_r2
    r2cfg = parse_r2(getattr(a, "r2_temp", 0) or 0, getattr(a, "r2_hop", 0) or 0, getattr(a, "r2_tgate", 0) or 0,
                     getattr(a, "r2_ctx", 0) or 0, getattr(a, "r2_mq", 0) or 0, getattr(a, "r2_mq_rank", 16) or 16,
                     getattr(a, "r2_mq_init", 0.1), getattr(a, "seed", 0) or 0)
    if r2cfg and not hasattr(itf, "r2"):                       # R2 query-side read forms (xenc/r2_read.py): zero-init mixing
        from xenc.r2_read import set_temp_init, upgrade_r2
        itf = upgrade_r2(itf, enc, r2cfg)
        if float(getattr(a, "r2_temp_init", 1.0) or 1.0) != 1.0:      # exploration: start the learned temp at x mult
            set_temp_init(itf, float(a.r2_temp_init))
    elif r2cfg:
        assert {k: v for k, v in r2cfg.items() if k != "seed"} == {k: v for k, v in itf.r2.items() if k != "seed"}, \
            "init already has another R2 read form"
    ptr_layers = str(getattr(a, "ptr_layers", "") or "")
    if ptr_layers or hasattr(itf, "ptr_layers"):              # P4 pointer / copy head (xenc/p4_ptr.py): gate a = 0
        from xenc.p4_ptr import parse_ptr, template_ids, upgrade_ptr
        pre_ids = template_ids(tok, a.reader, SYSTEM_ANSWER)
        if ptr_layers and not hasattr(itf, "ptr_layers"):
            itf = upgrade_ptr(itf, enc, parse_ptr(ptr_layers, getattr(a, "ptr_shifts", "0,1"),
                                                  int(getattr(a, "ptr_chain", 1)), pre_ids,
                                                  int(getattr(a, "ptr_topk", 0) or 0), int(getattr(a, "ptr_nbr", 0) or 0),
                                                  float(getattr(a, "ptr_kappa0", 1.0) or 1.0),
                                                  int(getattr(a, "ptr_nosp", 0) or 0)))
        elif ptr_layers:
            assert sorted(int(x) for x in ptr_layers.split(",")) == itf.ptr_layers, "init already has another pointer"
        assert itf.ptr_prefix == pre_ids, "pointer: stored chat-template prefix differs from this tokenizer / system"
        if itf.ptr_nosp:                                       # first-answer-token space / '>'-merge variants
            from xenc.p4_ptr import vocab_maps
            itf.set_vocab_maps(*vocab_maps(tok, reader.get_output_embeddings().weight.shape[0]))
    if hasattr(itf, "set_nosp") and getattr(itf, "p5", None) and itf.p5_nosp.numel() == 0:
        # P5 checkpoint loaded for judging (register_d20 / d22): rebuild the non-persistent space tables from the
        # tokenizer exactly as scripts/p5_train.py does, and check the stored template / tag ids against this tokenizer
        from xenc.p4_ptr import template_ids as _tids
        from xenc.p5_copy import space_maps
        nosp_t, gtm_t, gt_id, ans_id = space_maps(tok, int(reader.get_output_embeddings().weight.shape[0]))
        assert itf.p5["prefix"] == _tids(tok, a.reader, SYSTEM_ANSWER), "P5: stored chat-template prefix differs"
        assert itf.p5["gt_id"] == gt_id and itf.p5.get("ans_id", ans_id) == ans_id, "P5: stored tag ids differ"
        itf.set_nosp(nosp_t, gtm_t)
    from xenc.backbone import input_device
    itf = itf.to(input_device(reader))
    return reader, enc, tok, itf, lmap


def mean_terms(rows: list[dict]) -> dict:
    """[{li: {metric: v}}] -> {"layers": {li: {metric: mean}}, metric: mean over layers}."""
    lay: dict = {}
    for r in rows:
        for li, d in r.items():
            for k, v in d.items():
                lay.setdefault(li, {}).setdefault(k, []).append(float(v))
    L = {li: {k: sum(v) / len(v) for k, v in d.items()} for li, d in lay.items()}
    out = {"layers": {str(li): d for li, d in L.items()}}
    if L:
        for k in next(iter(L.values())):
            out[k] = sum(d[k] for d in L.values()) / len(L)
    return out


def fmt(m: dict) -> str:
    keys = ("js", "js_doc", "mass_t", "mass_s", "top5", "cos")
    s = " ".join(f"{k} {m[k]:.3f}" for k in keys if k in m)
    per = " | ".join(f"L{li} js {d['js_doc']:.3f} cos {d['cos']:.3f}" for li, d in m.get("layers", {}).items()
                     if "js_doc" in d)
    return s + (f" || {per}" if per else "")


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--B", default="outputs/D10_maps/encB8_9b_2b_post.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--train", default="quality:150,musique:150,lastingbench:150,longhealth:0")
    ap.add_argument("--val", default="quality:6,musique:12,lastingbench:12,longhealth:0")
    ap.add_argument("--val_fold", type=int, default=1, help="LongHealth fold used as held-out documents")
    ap.add_argument("--test_fold", type=int, default=0)
    ap.add_argument("--q_per_doc", type=int, default=8)
    ap.add_argument("--val_q_per_doc", type=int, default=4)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--lam", type=float, default=1e-3, help="ridge λ (relative to trace)")
    ap.add_argument("--variants", default="lin,mlp")
    ap.add_argument("--mlp_hidden", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr_mlp", type=float, default=1e-4)
    ap.add_argument("--rel_w", type=float, default=1.0)
    ap.add_argument("--onpolicy_p", type=float, default=0.5, help="on-policy prob. from epoch 2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--share_encoder", type=int, default=0, help="1: C2, --encoder == --reader, one shared instance")
    ap.add_argument("--enc_device", default="", help="put the encoder (and its operator prefill) on this device, e.g. "
                    "cuda:0 with --device cuda:1 (2-GPU long-operator runs); default = --device")
    ap.add_argument("--smoke", action="store_true", help="2 docs per task, 1 epoch")
    a = ap.parse_args(argv)
    if a.smoke:
        a.train = ",".join(f"{s.split(':')[0]}:2" for s in a.train.split(","))
        a.val = ",".join(f"{s.split(':')[0]}:1" for s in a.val.split(","))
        a.epochs, a.q_per_doc, a.val_q_per_doc = 1, 2, 2
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    from fit_state_maps import Gram, ridge
    reader, enc, tok, itf, lmap = build(a)
    rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens)
    tr_folds = {f for f in range(5)} - {a.test_fold, a.val_fold}
    train, d1 = select_items(a.train, "train", tr_folds, a.q_per_doc, a.seed)
    val, d2 = select_items(a.val, "val", {a.val_fold}, a.val_q_per_doc, a.seed + 1)
    rt.docs = {**d1, **d2}
    assert not ({_get(i, "doc_id") for i in train} & {_get(i, "doc_id") for i in val}), "val docs overlap train"
    log(f"D13 stage 1: {a.encoder} -> {a.reader} | layers {lmap} | train {len(train)} q / "
        f"{len({_get(i, 'doc_id') for i in train})} docs | held-out {len(val)} q / {len({_get(i, 'doc_id') for i in val})} docs")
    by_doc: OrderedDict = OrderedDict()
    for it in train:
        by_doc.setdefault(_get(it, "doc_id"), []).append(it)
    t0 = time.time()

    # ---------------- 1) ridge
    d_r, d_e = itf.d_r, itf.d_e
    G = {s: {li: Gram(d_r + 1, d_e, rt.dev) for li in lmap} for s in ("fit", "held")}
    for split, groups in (("fit", by_doc), ("held", _group(val))):
        for did, qs in groups.items():
            for it in qs:
                rows = rt.rows(did, it)
                op, n_op, _ = rt.operator(did)
                rec = rt.teacher(did, rows)
                R = rt.R
                R.load(op, n_op)
                R.mode, R.inject, R.xcap = "teacher", rt.y_teacher(rec), {}
                with torch.no_grad():
                    reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=False, logits_to_keep=1)
                R.mode = "off"
                for li, e in lmap.items():
                    X = R.xcap[li]
                    G[split][li].add(torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1), rec[e]["x"].float())
    r2 = {}
    with torch.no_grad():
        for li in lmap:
            W = ridge(G["fit"][li], a.lam)                        # [d_r+1, d_e]
            r2[str(li)] = {"fit": G["fit"][li].r2(W), "held": G["held"][li].r2(W)}
            itf.A[str(li)].weight.copy_(W[:-1].t().float())
            itf.A[str(li)].bias.copy_(W[-1].float())
    del G
    torch.cuda.empty_cache()
    log(f"ridge A (reader layer input -> encoder layer input) R² fit/held: " +
        " ".join(f"L{li} {v['fit']:.3f}/{v['held']:.3f}" for li, v in r2.items()) + f" ({time.time() - t0:.0f}s)")
    from xenc.d13_read import save_d13

    def held(onpolicy=True):
        rows = []
        for did, qs in _group(val).items():
            for it in qs:
                rows.append(rt.s1_pass(did, it, onpolicy=onpolicy, grad=False, metrics=True))
        return mean_terms(rows)

    h_tf, h_op = held(False), held(True)
    log(f"[ridge] held-out teacher-forced: {fmt(h_tf)}")
    log(f"[ridge] held-out on-policy:      {fmt(h_op)}")
    report = {"args": vars(a), "lmap": lmap, "ridge_r2": r2, "ridge_held": {"tf": h_tf, "op": h_op}, "variants": {}}
    save_d13(itf, os.path.join(a.out, "d13_ridge.pt"), {"stage": "ridge", "held": h_op})
    ridge_sd = {k: v.detach().clone() for k, v in itf.state_dict().items() if k.startswith("A.")}

    # ---------------- 2) attention distillation
    for var in [v for v in a.variants.split(",") if v]:
        mh = a.mlp_hidden if var == "mlp" else 0
        from xenc.d13_read import D13Interface
        if mh != itf.mlp_hidden:
            Bd = None if itf.identity_B else {li: itf.B[str(li)].detach().cpu() for li in lmap}
            itf2 = D13Interface(enc, lmap, d_r, d_e, B=Bd, mlp_hidden=mh).to(rt.dev)
            rt.itf = rt.R.itf = itf2
            itf = itf2
        itf.load_state_dict(ridge_sd, strict=False)
        if mh:
            for li in lmap:
                torch.nn.init.zeros_(itf.M[str(li)][2].weight)
                torch.nn.init.zeros_(itf.M[str(li)][2].bias)
        groups = [{"params": list(itf.A.parameters()), "lr": a.lr}]
        if mh:
            groups.append({"params": list(itf.M.parameters()), "lr": a.lr_mlp})
        itf.trainable()
        opt = torch.optim.AdamW(groups, weight_decay=0.0)
        params = [p for g in groups for p in g["params"]]
        best, hist = -1e9, []
        bpath = os.path.join(a.out, f"d13_{var}.pt")
        for ep in range(1, a.epochs + 1):
            docs_ep = list(by_doc)
            rng.shuffle(docs_ep)
            p_on = a.onpolicy_p if ep >= 2 else 0.0
            acc, n, tl = 0, 0, {"kl": 0.0, "rel": 0.0}
            opt.zero_grad(set_to_none=True)
            for did in docs_ep:
                for it in by_doc[did]:
                    terms = rt.s1_pass(did, it, onpolicy=rng.random() < p_on)
                    if not terms:
                        continue
                    kl = torch.stack([t["kl"] for t in terms.values()]).mean()
                    rel = torch.stack([t["rel"] for t in terms.values()]).mean()
                    loss = kl + a.rel_w * rel
                    if not math.isfinite(float(loss)):
                        log(f"  non-finite loss on {did}; skipped")
                        continue
                    (loss / a.accum).backward()
                    tl["kl"] += float(kl)
                    tl["rel"] += float(rel)
                    n += 1
                    acc += 1
                    if acc == a.accum:
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        acc = 0
                    if n % 200 == 0:
                        log(f"  [{var} ep {ep}] {n} q: kl {tl['kl'] / n:.4f} rel {tl['rel'] / n:.4f} "
                            f"({time.time() - t0:.0f}s)")
            if acc:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            h = held(True)
            hist.append({"epoch": ep, "train_kl": tl["kl"] / max(n, 1), "train_rel": tl["rel"] / max(n, 1), "held": h})
            tag = ""
            if h["cos"] > best:
                best = h["cos"]
                save_d13(itf, bpath, {"stage": f"attn-distill {var}", "epoch": ep, "held": h})
                tag = " *best*"
            log(f"[{var} ep {ep}] train kl {tl['kl'] / max(n, 1):.4f} rel {tl['rel'] / max(n, 1):.4f} | held-out "
                f"on-policy: {fmt(h)}{tag} ({time.time() - t0:.0f}s)")
        report["variants"][var] = {"history": hist, "best_cos": best, "path": bpath}
        with open(os.path.join(a.out, "report.json"), "w") as f:
            json.dump(report, f, indent=1)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    if torch.cuda.is_available():
        log(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    log(f"done ({time.time() - t0:.0f}s) -> {a.out}")
    return 0


def _group(items) -> OrderedDict:
    g: OrderedDict = OrderedDict()
    for it in items:
        g.setdefault(_get(it, "doc_id"), []).append(it)
    return g


if __name__ == "__main__":
    sys.exit(main())
