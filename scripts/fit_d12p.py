"""D12p (copy of D13 for the 27B reader). D13 stage 1: fit the reader-query map A of the paper's read (xenc/d13_read.py) to the ENCODER's OWN question-pass
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
              # D12p deep: D25 conflict sets (pool key = doc-id prefix)
              "d25cqa": "data/d25_cqa_split", "d25fecf": "data/d25_fecf_split", "d25nqs": "data/d25_nqswap_split",
              # D12p lever 1 (as fit_d13.TASK_SPLIT): all benchmark train splits
              "loogle": "data/loogle_split", "clipper": "data/clipper_split", "qasper": "data/qasper_split",
              "contractnli": "data/contractnli_split", "shakespeare_cf": "data/shakespeare_cf_split"}


def add_extra_splits(spec: str):
    """D12b (as register_d13 --extra_splits): 'name=data/<name>_split,...' train-only splits, registered in this
    module's and fit_d13's TASK_SPLIT (the latter is what register_d13.doc_len_filter reads). No-op for ''."""
    import fit_d13 as _f13
    for s in [x for x in (spec or "").split(",") if x]:
        n, d = s.split("=")
        for T in (TASK_SPLIT, _f13.TASK_SPLIT):
            assert n not in T or T[n] == d, f"--extra_splits would redefine {n}"
            T[n] = d
    if spec:
        log(f"  extra_splits: {spec}")


def lever_filter(a, *item_lists):
    """D12p lever 1: drop items of excluded (judged / near-duplicate) documents and of documents outside --doc_len, with
    register_d13's exact helpers (excluded_docs, doc_len_filter). -> (filtered lists, excluded doc set). No-op if unset."""
    es, ed, dl = getattr(a, "exclude_sets", ""), getattr(a, "exclude_docs", ""), getattr(a, "doc_len", "")
    if not (es or ed or dl):
        return list(item_lists), set()
    from register_d13 import doc_len_filter, excluded_docs
    drop = excluded_docs(es)
    if ed:
        X = json.load(open(ed))
        drop |= set(sum(X.values(), [])) if isinstance(X, dict) else set(X)
    keep_len = doc_len_filter(dl)
    out = [[i for i in L if _get(i, "doc_id") not in drop and keep_len(_get(i, "doc_id"))] for L in item_lists]
    log(f"  lever filter: excluded {len(drop)} docs, doc_len '{dl}': items " +
        " ".join(f"{len(x)}->{len(y)}" for x, y in zip(item_lists, out)))
    for L in out:
        assert not ({_get(i, "doc_id") for i in L} & drop), "an excluded (judged) document reached train/val"
    return out, drop


def extra_val_items(spec: str, q_per_doc: int, seed: int, doc_len: str = ""):
    """D12b: extra held-out val docs, spec "loogle:val:8,narrativeqa:train:-8" (task:split:k; k>0 = the first k sorted
    docs of that split, k<0 = the LAST |k| sorted docs, after the --doc_len window). A train split may serve val only
    with k<0 (the last docs), and the caller must assert they are not training docs. -> (items, {doc_id: text})."""
    from xenc.doc_tasks import load_docs
    if not spec:
        return [], {}
    keep_len = (lambda d: True)
    if doc_len:
        from register_d13 import doc_len_filter
        keep_len = doc_len_filter(doc_len)
    rng = random.Random(seed)
    items, docs = [], {}
    for s in [x for x in spec.split(",") if x]:
        name, split, k = s.split(":")
        k = int(k)
        assert split != "train" or k < 0, f"val_extra from a TRAIN split must take the last docs (k<0): {s}"
        sd = TASK_SPLIT[name]
        dd = load_docs(sd)
        by = OrderedDict()
        for it in load_split_items(sd, split):
            by.setdefault(_get(it, "doc_id"), []).append(it)
        dids = [d for d in sorted(by) if keep_len(d)]
        dids = dids[:k] if k > 0 else dids[k:]
        for d in dids:
            qs = by[d]
            if q_per_doc and len(qs) > q_per_doc:
                qs = rng.sample(qs, q_per_doc)
            items += qs
            docs[d] = dd[d]
    return items, docs


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
        from xenc.d12p_read import D13Reader, EncTeacher
        self.reader, self.enc, self.tok, self.hub, self.itf = reader, enc, tok, hub, itf
        self.tail, self.max_tokens = int(tail), int(max_tokens)
        self.Tc = EncTeacher(enc, sorted(set(itf.layer_map.values())))
        self.R = D13Reader(reader, itf)
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
        out = self.enc(input_ids=torch.tensor([ids], device=self.edev), use_cache=True, logits_to_keep=1)
        op = {e: (r["k"].to(self.dev), r["v"].to(self.dev)) for e, r in self.Tc.rec.items()}
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
        from xenc.d12p_read import d13_terms, teacher_logits
        out = {}
        for li, e in self.itf.layer_map.items():
            r = self.R.rec.get(li)
            if r is None or r.get("logits") is None:
                continue
            lt = teacher_logits(rec[e]["q"], op[e][0], rec[e]["k"], self.itf.att[str(li)].scaling)
            if metrics or not torch.is_grad_enabled() or not r["logits"].requires_grad:
                out[li] = d13_terms(r["logits"], lt, r["y"], y_t[li], n_op, metrics)
            else:                      # D12p: recompute the [Hkv,G,T,C] softmax/KL intermediates in backward (memory)
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


# ============================================================================ D12p: two-phase (cached encoder) runtime
def row_key(did: str, rows) -> str:
    import hashlib
    return hashlib.sha1((str(did) + "|" + ",".join(str(int(x)) for x in rows)).encode()).hexdigest()


OP_ALIAS: dict = {}          # D12b: doc-id prefix aliases, e.g. {"nqamc": "narrativeqa"} (same document file)


def set_op_alias(spec: str):
    """D12b --op_alias 'nqamc=narrativeqa,looglemc=loogle': a derived split whose docs ARE another split's docs (same
    file) reads that split's cached operator (CachedRt asserts the prefix length matches)."""
    for s_ in [x for x in (spec or "").split(",") if x]:
        a_, b_ = s_.split("=")
        OP_ALIAS[a_] = b_
    if spec:
        log(f"  op_alias: {OP_ALIAS}")


def op_candidates(did: str) -> list:
    did = str(did)
    out = [did]
    pre = did.split("_")[0]
    if pre in OP_ALIAS:
        out.append(OP_ALIAS[pre] + did[len(pre):])
    return out


def find_op(dirs, did: str):
    """-> path of the cached operator of did (or of its alias), or None."""
    for c in dirs:
        for d in op_candidates(did):
            q = op_path(c, d)
            if os.path.exists(q):
                return q
    return None


def op_path(root: str, did: str) -> str:
    return os.path.join(root, "op", str(did).replace("/", "_") + ".pt")


def rec_path(root: str, did: str, rows) -> str:
    return os.path.join(root, "rec", row_key(did, rows) + ".pt")


class CachedRt(D13Rt):
    """D13Rt without the encoder in memory: the operator (K/V at the paired encoder layers) and the encoder's
    question-pass teacher records come from the phase-A cache written by scripts/cache_d12p.py (same rows:
    identical pre/suf/rows code). Reader-side behaviour (s1_pass, gen, attn_terms) is inherited unchanged."""

    def __init__(self, reader, tok, hub, itf, cache_dir: str, tail: int = 64, max_tokens: int = 16000, lru: int = 3):
        from xenc.backbone import input_device
        from xenc.d12p_read import D13Reader
        self.reader, self.enc, self.tok, self.hub, self.itf = reader, None, tok, hub, itf
        self.tail, self.max_tokens = int(tail), int(max_tokens)
        self.R = itf.make_reader(reader) if hasattr(itf, "make_reader") else D13Reader(reader, itf)   # P3: TokReader
        self.docs, self._pre, self._ops, self.lru = {}, {}, OrderedDict(), int(lru)
        self.dev = self.edev = itf.g.device                   # operator / teacher tensors live with the interface
        self.cache_dirs = [d for d in str(cache_dir).split(",") if d]   # D12p lever: several caches, searched in order
        self.cache_dir = self.cache_dirs[0]

    @torch.no_grad()
    def operator(self, did):
        if did in self._ops:
            self._ops.move_to_end(did)
            return self._ops[did]
        p = find_op(self.cache_dirs, did)                    # D12b: alias-aware (default: op_path only)
        if p is None:
            raise FileNotFoundError(f"D12p cache: no operator for {did} in {self.cache_dirs}")
        b = torch.load(p, map_location="cpu", weights_only=False)
        op = {int(e): (k.to(self.dev), v.to(self.dev)) for e, (k, v) in b["kv"].items()}
        if p != op_path(os.path.dirname(os.path.dirname(p)), did):   # D12b alias: the prefix must match exactly
            ids = self.pre(did)[: -self.tail] if self.tail else self.pre(did)
            assert len(ids) == int(b["n"]), f"op_alias: {did} prefix {len(ids)} != aliased operator N {b['n']} ({p})"
            if b.get("ids") is not None:
                assert list(b["ids"]) == list(ids), f"op_alias: token ids of {did} differ from the aliased operator"
        if getattr(self.itf, "needs_ids", False):     # P3 token channel: column token ids (stored, or re-derived)
            ids = self.pre(did)[: -self.tail] if self.tail else self.pre(did)
            if b.get("ids") is not None:
                assert list(b["ids"]) == list(ids), f"token ids: cached operator ids differ from the tokenization of {did}"
            assert len(ids) == int(b["n"]) == next(iter(op.values()))[0].shape[1], \
                f"token ids: re-tokenized prefix ({len(ids)}) does not match the cached operator ({b['n']}) for {did}"
            op["ids"] = torch.tensor(ids, dtype=torch.long, device=self.dev)
        self._ops[did] = (op, int(b["n"]), None)
        while len(self._ops) > self.lru:
            self._ops.popitem(last=False)
        return self._ops[did]

    @torch.no_grad()
    def teacher(self, did, rows) -> dict:
        """Records are keyed by the question-only rows and hold rows + gold answer; `rows` may be either (or any
        prefix that still contains the question), so the cut is searched back over up to 256 trailing tokens."""
        k = self.find(did, rows)
        if k is not None:
            rows = [int(x) for x in rows]
            b = torch.load(os.path.join(self._index[k], "rec", k + ".pt"), map_location="cpu", weights_only=False)
            T = len(rows)
            assert b["rows"][:T] == rows, f"D12p cache: rows mismatch for {did}"
            out = {}
            for e, r in b["rec"].items():
                d = {}
                for f, v in r.items():
                    d[f] = (v[:, :T] if f in ("q", "k", "v") else v[:T]).to(self.dev)
                out[int(e)] = d
            return out
        raise FileNotFoundError(f"D12p cache: no teacher record for {did} / {len(rows)} rows")

    def find(self, did, rows):
        """-> record key covering `rows` (question rows, optionally + answer), or None. Index: key -> cache dir (the
        first dir that holds it)."""
        if not hasattr(self, "_index"):
            self._index = {}
            for c in reversed(self.cache_dirs):
                self._index.update({f[:-3]: c for f in os.listdir(os.path.join(c, "rec")) if f.endswith(".pt")})
        rows = [int(x) for x in rows]
        for m in range(0, min(257, len(rows))):
            k = row_key(did, rows[: len(rows) - m])
            if k in self._index:
                return k
        return None


def make_rt(a, reader, enc, tok, itf):
    if getattr(a, "enc_cache", None):
        return CachedRt(reader, tok, a.reader, itf, a.enc_cache, a.tail, a.max_tokens)
    return D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens)


def build(a, init: str | None = None, mlp_hidden: int = 0):
    """Load reader + encoder, D13 interface (identity A, or `init` file), B from --B."""
    from xenc.backbone import load_encoder, load_reader
    from xenc.d12p_read import D13Interface, load_B, load_d13
    from xenc.state_skill import text_config
    from xenc.xattn_read import attn_dims, full_depth_map
    if getattr(a, "reader_mem", None):                             # D12p: per-GPU budgets, e.g. "27GiB,27GiB"
        import xenc.backbone as bb
        mm = {i: m for i, m in enumerate(a.reader_mem.split(","))}
        orig = bb._device_kwargs
        bb._device_kwargs = lambda dm, mx: {"device_map": "auto", "max_memory": mm}
        try:
            reader, tok = load_reader(a.reader, device_map="auto")
        finally:
            bb._device_kwargs = orig
    else:
        reader, tok = load_reader(a.reader, device_map=a.device)
    enc = load_encoder(a.encoder, device_map=getattr(a, "enc_device", None) or a.device)[0]   # D12p: 2-GPU split
    keep = {int(x) for x in a.layers.split(",")}
    lmap = {k: v for k, v in full_depth_map(attn_dims(reader), attn_dims(enc)).items() if k in keep}
    B, Bb = load_B_any(a.B, sorted(lmap), lmap, missing_ok=True)
    if init:
        fl = {int(k): int(v) for k, v in torch.load(init, map_location="cpu", weights_only=False)["config"]["layer_map"].items()}
        src = max(B) if B else None
        itf, _ = load_d13(init, enc, {li: B.get(li, B[src]) for li in fl})   # file B overrides
        new = sorted(set(lmap) - set(fl))
        if new:                                                    # D12p deep: extend the interface to more layers
            itf = extend_itf(itf, enc, lmap, int(getattr(a, "extend_src", 0) or max(fl)))
            log(f"extended interface {sorted(fl)} -> {sorted(lmap)}: new layers {new} copy L"
                f"{int(getattr(a, 'extend_src', 0) or max(fl))} (A, MLP, B, bias), gate 0 | layer_map {lmap}")
        lmap = dict(itf.layer_map)                                 # the interface's layers are authoritative
    else:
        itf = D13Interface(enc, lmap, int(text_config(reader).hidden_size), int(text_config(enc).hidden_size),
                           B=B, mlp_hidden=mlp_hidden)
        with torch.no_grad():
            for li, b in Bb.items():
                itf.Bb[str(li)].copy_(b)
    from xenc.backbone import input_device
    itf = itf.to(getattr(a, "itf_device", None) or input_device(reader))    # D12p: e.g. the last reader GPU
    return reader, enc, tok, itf, lmap


def peak_mem(tag: str = ""):
    """D12p: peak allocated / reserved memory on every visible GPU."""
    if torch.cuda.is_available():
        log(f"PEAK MEM {tag}: " + " | ".join(
            f"cuda:{i} alloc {torch.cuda.max_memory_allocated(i) / 2**30:.1f} GiB reserved "
            f"{torch.cuda.max_memory_reserved(i) / 2**30:.1f} GiB / {torch.cuda.get_device_properties(i).total_memory / 2**30:.1f}"
            for i in range(torch.cuda.device_count())))


def extend_itf(itf, enc, lmap_full: dict, src: int):
    """D12p deep: a D13 interface over lmap_full. Existing layers keep everything; each new layer starts as a copy of
    layer `src` (A, MLP, B, write-back bias) with gate 0, so the extended read equals the old one exactly."""
    from xenc.d12p_read import D13Interface
    assert itf.delta_rank == 0
    Bd = {li: itf.B[str(li if li in itf.layer_map else src)].detach().cpu().clone() for li in lmap_full}
    new = D13Interface(enc, lmap_full, itf.d_r, itf.d_e, B=Bd, mlp_hidden=itf.mlp_hidden)
    with torch.no_grad():
        for li in lmap_full:
            s_ = li if li in itf.layer_map else src
            new.A[str(li)].load_state_dict(itf.A[str(s_)].state_dict())
            if itf.mlp_hidden:
                new.M[str(li)].load_state_dict(itf.M[str(s_)].state_dict())
            new.Bb[str(li)].copy_(itf.Bb[str(s_)].detach().cpu())
            new.g[new._gi[li]] = float(itf.g[itf._gi[li]]) if li in itf.layer_map else 0.0
    return new


def load_B_any(path: str, layers, lmap: dict, missing_ok: bool = False):
    """D12p: D10 token-wise blob (load_B, zero bias) or a D12 "xenc.encside_B2.v1" single-source blob
    (y = W(x − μx) + μy  ->  B = W, Bb = μy − W μx). -> ({li: B [d_r, d_e]}, {li: Bb [d_r]})."""
    from xenc.d12p_read import load_B
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if missing_ok:                                                 # D12p deep: layers the blob does not cover are skipped
        have = blob["enc_B"] if blob.get("format") != "xenc.encside_B2.v1" else blob["layers"]
        layers = [li for li in layers if str(li) in have]
    if blob.get("format") != "xenc.encside_B2.v1":
        return load_B(path, layers), {}
    B, Bb = {}, {}
    for li in layers:
        L = blob["layers"][str(li)]
        assert [list(x) for x in L["src"]] == [[lmap[li], 0]], f"L{li}: D13 read needs a single matched source"
        W = L["W"].float()
        B[li], Bb[li] = W, L["my"].float() - W @ L["mx"].float()
    return B, Bb


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
    ap.add_argument("--enc_device", default=None, help="D12p: encoder device (e.g. cuda:1); default = --device")
    ap.add_argument("--reader_mem", default=None, help="D12p: shard the reader, per-GPU budgets '27GiB,27GiB'")
    ap.add_argument("--enc_cache", default=None, help="D12p phase B: cache_d12p.py dir (load the encoder with "
                    "--enc_device cpu; it never runs)")
    ap.add_argument("--itf_device", default=None, help="D12p: device of the interface + cached tensors")
    ap.add_argument("--init", default="", help="D12p deep: start from this d13 file (extended to --layers if needed)")
    ap.add_argument("--train_layers", default="", help="D12p deep: layers stage 1 fits (ridge + distillation); "
                    "default all")
    ap.add_argument("--extend_src", type=int, default=0, help="D12p deep: layer copied into new layers (default "
                    "the init's deepest)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="2 docs per task, 1 epoch")
    a = ap.parse_args(argv)
    if a.smoke:
        a.train = ",".join(f"{s.split(':')[0]}:2" for s in a.train.split(","))
        a.val = ",".join(f"{s.split(':')[0]}:1" for s in a.val.split(","))
        a.epochs, a.q_per_doc, a.val_q_per_doc = 1, 2, 2
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    from fit_state_maps import ridge
    CpuGram = _cpu_gram()
    reader, enc, tok, itf, lmap = build(a, init=a.init or None)
    rt = make_rt(a, reader, enc, tok, itf)
    tl_set = {int(x) for x in a.train_layers.split(",") if x} or set(lmap)    # D12p deep: layers stage 1 fits
    if a.init:
        rt.R.gate_tf = True                                        # teacher forcing honours the init's gates
        log(f"stage 1 from {a.init}: fitting layers {sorted(tl_set)} (others frozen); gate-aware teacher forcing")
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
    G = {s: {li: CpuGram(d_r + 1, d_e, "cpu") for li in lmap if li in tl_set} for s in ("fit", "held")}   # D12p: fp64 on CPU
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
                    if li not in tl_set:
                        continue
                    X = R.xcap[li]
                    G[split][li].add(torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1), rec[e]["x"].float())
    r2 = {}
    for s_ in G.values():
        for g_ in s_.values():
            g_.flush()
    with torch.no_grad():
        for li in sorted(tl_set):
            W = ridge(G["fit"][li], a.lam)                        # [d_r+1, d_e]
            r2[str(li)] = {"fit": G["fit"][li].r2(W), "held": G["held"][li].r2(W)}
            itf.A[str(li)].weight.copy_(W[:-1].t().float())
            itf.A[str(li)].bias.copy_(W[-1].float())
    del G
    torch.cuda.empty_cache()
    peak_mem("stage 1 ridge")
    log(f"ridge A (reader layer input -> encoder layer input) R² fit/held: " +
        " ".join(f"L{li} {v['fit']:.3f}/{v['held']:.3f}" for li, v in r2.items()) + f" ({time.time() - t0:.0f}s)")
    from xenc.d12p_read import save_d13

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
        from xenc.d12p_read import D13Interface
        if mh != itf.mlp_hidden:
            Bd = {li: itf.B[str(li)].detach().cpu() for li in lmap}
            itf2 = D13Interface(enc, lmap, d_r, d_e, B=Bd, mlp_hidden=mh).to(rt.dev)
            with torch.no_grad():                                  # D12p: keep the write-back bias
                for li in lmap:
                    itf2.Bb[str(li)].copy_(itf.Bb[str(li)].detach())
            rt.itf = rt.R.itf = itf2
            itf = itf2
        itf.load_state_dict(ridge_sd, strict=False)
        if mh:
            for li in sorted(tl_set):
                torch.nn.init.zeros_(itf.M[str(li)][2].weight)
                torch.nn.init.zeros_(itf.M[str(li)][2].bias)
        itf.trainable()
        for li in lmap:                                            # D12p deep: only the fitted layers train
            for p_ in list(itf.A[str(li)].parameters()) + (list(itf.M[str(li)].parameters()) if mh else []):
                p_.requires_grad_(li in tl_set)
        groups = [{"params": [p_ for li in sorted(tl_set) for p_ in itf.A[str(li)].parameters()], "lr": a.lr}]
        if mh:
            groups.append({"params": [p_ for li in sorted(tl_set) for p_ in itf.M[str(li)].parameters()], "lr": a.lr_mlp})
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
                        if not getattr(opt, "_d12p_mem", False):
                            opt._d12p_mem = True
                            peak_mem(f"stage 1 [{var}] first step")
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
    peak_mem()
    log(f"done ({time.time() - t0:.0f}s) -> {a.out}")
    return 0


def _cpu_gram():
    from fit_state_maps import Gram

    class CpuGram(Gram):
        """Gram with fp64 accumulators on the CPU; products computed in fp32 on the rows' device."""
        def add(self, X, Y, flush_rows: int = 8192):
            """Rows are buffered on their device and folded in every `flush_rows` rows (one GPU->CPU copy of the
            d×d products per flush instead of per question); call flush() before reading the statistics."""
            if not hasattr(self, "_buf"):
                self._buf, self._nb = [], 0
            self._buf.append((X.float(), Y.float()))
            self._nb += X.shape[0]
            if self._nb >= flush_rows:
                self.flush()

        def flush(self):
            if not getattr(self, "_buf", None):
                return
            X = torch.cat([x for x, _ in self._buf])
            Y = torch.cat([y for _, y in self._buf])
            self._buf, self._nb = [], 0
            self.xx += (X.t() @ X).double().cpu()
            self.xy += (X.t() @ Y).double().cpu()
            self.yy += (Y.double() * Y.double()).sum().cpu()
            self.ys += Y.double().sum(0).cpu()
            self.n += X.shape[0]
    return CpuGram


def _group(items) -> OrderedDict:
    g: OrderedDict = OrderedDict()
    for it in items:
        g.setdefault(_get(it, "doc_id"), []).append(it)
    return g


if __name__ == "__main__":
    sys.exit(main())
