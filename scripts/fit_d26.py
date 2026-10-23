"""D26 stage 1: D13's label-free stage 1 (ridge A, then attention distillation of A to the encoder's own question-pass
attention) for a reader from ANOTHER model family (gpt-oss-20b, GLM-4-9B-0414, DeepSeek-V2-Lite-Chat), encoder
Qwen3.5-4B. Everything is scripts/fit_d13.py except the cross-tokenizer pieces (xenc/d26_read.py):

  operator   identical to D13: encoder prefill of the document prefix minus its last `tail` ENCODER tokens
  rows       encoder: those `tail` tokens + the encoder's question suffix (exactly D13's teacher rows)
             reader : [reader BOS header] + the reader's tokenization of the same tail TEXT + the reader's own question
                      suffix (its chat template; gpt-oss: forced final channel)
  alignment  reader row t <- the last encoder row that has consumed no more characters (xenc.d26_read.coords/align)
  losses     KL over [operator columns ; bucketed own-row mass] + relMSE(B o_s, B o_t), at aligned rows
  ridge A    reader layer input (valid rows) -> encoder layer input at the aligned rows

    python scripts/fit_d26.py --reader glm4 --B outputs/D26_maps/B_glm4.pt --variants mlp --out outputs/D26_glm4_maps

D26Rt (operator / teacher passes, aligned row pairs, greedy decode with the read) is shared with register_d26.py and
eval_d26.py.
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
from register_doc import SYSTEM_ANSWER, _get, log  # noqa: E402


# ============================================================================ runtime
class D26Rt:
    def __init__(self, reader, enc, tok_r, tok_e, hub_r, hub_e, itf, tail: int = 64, max_tokens: int = 16000,
                 lru: int = 3, bos: bool = True):
        from xenc.backbone import input_device
        from xenc.d13_read import EncTeacher
        from xenc.d26_read import D26Reader, encode, reader_bos, stop_ids
        self.reader, self.enc, self.tok_r, self.tok_e = reader, enc, tok_r, tok_e
        self.hub_r, self.hub_e, self.itf = hub_r, hub_e, itf
        self.tail, self.max_tokens, self.lru = int(tail), int(max_tokens), int(lru)
        self.Tc = EncTeacher(enc, sorted(set(itf.layer_map.values())))
        self.R = D26Reader(reader, itf)
        self.bos = reader_bos(tok_r, hub_r) if bos else ""
        self.bos_ids, self.bos_offs = encode(tok_r, self.bos) if self.bos else ([], [])
        self.R.keep0 = len(self.bos_ids)
        self.same_tok = tok_r.get_vocab() == tok_e.get_vocab()
        self.stops = stop_ids(tok_r, reader)
        self.docs: dict = {}
        self._pre: dict = {}
        self._suf: dict = {}
        self._ops: OrderedDict = OrderedDict()
        self.ckpt = False
        self.dev, self.edev = input_device(reader), input_device(enc)

    # ---------------------------------------------------------------- rows
    def pre(self, did):
        """Encoder prefix (ids, offsets, string, cut): D13Rt.pre plus offsets; cut = first char of the tail."""
        if did not in self._pre:
            from xenc.d26_read import encode
            from xenc.state_skill import doc_prefix_text
            S = doc_prefix_text(self.tok_e, self.hub_e, self.docs[did], SYSTEM_ANSWER)
            ids, offs = encode(self.tok_e, S)
            k = max(0, len(ids) - (self.max_tokens - 1024))
            ids, offs = ids[k:], offs[k:]
            cut = offs[-self.tail][0] if self.tail else len(S)
            self._pre[did] = (ids, offs, S, cut)
        return self._pre[did]

    def tail_text(self, did) -> str:
        _, _, S, cut = self.pre(did)
        return S[cut:]

    def suf(self, which: str, it):
        from xenc.d26_read import encode, suffix_str
        key = (which, _get(it, "key") or _get(it, "question"))
        if key not in self._suf:
            tok, hub = (self.tok_r, self.hub_r) if which == "r" else (self.tok_e, self.hub_e)
            s, qc = suffix_str(tok, hub, it, SYSTEM_ANSWER)
            ids, offs = encode(tok, s)
            self._suf[key] = (s, qc, ids, offs)
        return self._suf[key]

    def pair(self, did, it, target: str | None = None) -> dict:
        """Aligned row sequences. -> {r_ids, e_ids, idx [T_r], valid [T_r], m_r, m_e (target tokens)}."""
        from xenc.d26_read import align, coords, encode
        ids, offs, S, cut = self.pre(did)
        tail = S[cut:]
        sh = lambda o, d: [(a + d, b + d) for a, b in o]   # noqa: E731
        se, qc, se_ids, se_offs = self.suf("e", it)
        e_ids = ids[-self.tail:] + se_ids
        e_off = sh(offs[-self.tail:], -cut) + sh(se_offs, len(tail))
        Se = tail + se
        sr, qc_r, sr_ids, sr_offs = self.suf("r", it)
        assert qc_r == qc
        if self.same_tok:
            rt_ids, rt_offs = ids[-self.tail:], sh(offs[-self.tail:], -cut)
        else:
            rt_ids, rt_offs = encode(self.tok_r, tail)
        nb = len(self.bos)
        r_ids = self.bos_ids + rt_ids + sr_ids
        r_off = list(self.bos_offs) + sh(rt_offs, nb) + sh(sr_offs, nb + len(tail))
        Sr = self.bos + tail + sr
        contents = [tail + qc]
        m_r = m_e = 0
        if target:
            te, te_o = encode(self.tok_e, target)
            tr, tr_o = encode(self.tok_r, target)
            e_ids, e_off, m_e = e_ids + te, e_off + sh(te_o, len(Se)), len(te)
            r_ids, r_off, m_r = r_ids + tr, r_off + sh(tr_o, len(Sr)), len(tr)
            Se, Sr = Se + target, Sr + target
            contents.append(target)
        idx, valid = align(coords(Sr, r_off, contents), coords(Se, e_off, contents))
        return {"r_ids": r_ids, "e_ids": e_ids, "idx": idx, "valid": valid, "m_r": m_r, "m_e": m_e}

    def rows(self, did, it) -> list[int]:
        return self.pair(did, it)["r_ids"]

    # ---------------------------------------------------------------- encoder passes
    @torch.no_grad()
    def operator(self, did):
        if did in self._ops:
            self._ops.move_to_end(did)
            return self._ops[did]
        ids = self.pre(did)[0]
        ids = ids[: -self.tail] if self.tail else ids
        self.Tc.rec, self.Tc.kv_only = {}, True
        out = self.enc(input_ids=torch.tensor([ids], device=self.edev), use_cache=True, logits_to_keep=1)
        op = {e: (r["k"].to(self.dev), r["v"].to(self.dev)) for e, r in self.Tc.rec.items()}
        self.Tc.rec, self.Tc.kv_only = None, False
        self._ops[did] = (op, len(ids), out.past_key_values)
        while len(self._ops) > self.lru:
            self._ops.popitem(last=False)
        return self._ops[did]

    @torch.no_grad()
    def teacher(self, did, e_ids) -> dict:
        _, _, cache = self.operator(did)
        self.Tc.rec, self.Tc.kv_only = {}, False
        self.enc(input_ids=torch.tensor([e_ids], device=self.edev), past_key_values=copy.deepcopy(cache),
                 use_cache=True, logits_to_keep=1)
        rec = {e: {k: v.to(self.dev) for k, v in r.items()} for e, r in self.Tc.rec.items()}
        self.Tc.rec = None
        return rec

    def y_teacher(self, rec) -> dict:
        with torch.no_grad():
            return {li: self.itf.Bmap(li, rec[e]["out"].float()).detach() for li, e in self.itf.layer_map.items()}

    def attn_terms(self, rec, op, n_op, y_t: dict, P: dict, metrics=False) -> dict:
        from xenc.d13_read import teacher_logits
        from xenc.d26_read import d26_terms
        out = {}
        for li, e in self.itf.layer_map.items():
            r = self.R.rec.get(li)
            if r is None or r.get("logits") is None:
                continue
            lt = teacher_logits(rec[e]["q"], op[e][0], rec[e]["k"], self.itf.att[str(li)].scaling)
            idx, valid = P["idx"], P["valid"]
            if (not self.ckpt or metrics or not torch.is_grad_enabled() or not r["logits"].requires_grad):
                out[li] = d26_terms(r["logits"], lt, r["y"], y_t[li], idx, valid, n_op, metrics)
            else:
                from torch.utils.checkpoint import checkpoint
                yt = y_t[li]

                def _f(ls, ys, lt=lt, yt=yt):
                    d = d26_terms(ls, lt, ys, yt, idx, valid, n_op, False)
                    return d["kl"], d["rel"]
                kl, rel = checkpoint(_f, r["logits"], r["y"], use_reentrant=False)
                out[li] = {"kl": kl, "rel": rel}
            del lt
        return out

    def s1_pass(self, did, it, onpolicy=False, grad=True, metrics=False, y_ref: dict | None = None):
        op, n_op, _ = self.operator(did)
        P = self.pair(did, it)
        rec = self.teacher(did, P["e_ids"])
        R = self.R
        R.load(op, n_op)
        yt = self.y_teacher(rec)
        R.mode, R.onpolicy, R.grad, R.rec = "train", bool(onpolicy), bool(grad), {}
        R.inject = {li: yt[li][P["idx"].to(yt[li].device)] for li in yt}
        R.valid = P["valid"]
        try:
            with torch.no_grad():
                self.reader(input_ids=torch.tensor([P["r_ids"]], device=self.dev), use_cache=False, logits_to_keep=1)
            return self.attn_terms(rec, op, n_op, y_ref or yt, P, metrics)
        finally:
            R.mode, R.rec, R.inject, R.valid = "off", {}, {}, None

    @torch.no_grad()
    def gen(self, did, it, budget: int = 32, stop: str = "</answer>", read: bool = True, rows_did=None):
        """Greedy decode from the reader prompt ([BOS] + doc tail + question) with the read of did's operator
        (read=False: closed_tail). rows_did: document whose tail is in the prompt (default did)."""
        from xenc.backbone import decode
        rows = self.rows(rows_did or did, it)
        R = self.R
        if read:
            op, n_op, _ = self.operator(did)
            R.load(op, n_op)
            R.mode = "student"
        try:
            out = self.reader(input_ids=torch.tensor([rows], device=self.dev), use_cache=True, logits_to_keep=1)
            new = []
            for _ in range(budget):
                nxt = int(out.logits[0, -1].argmax())
                new.append(nxt)
                if nxt in self.stops or stop in self.tok_r.decode(new[-12:], skip_special_tokens=True):
                    break
                out = self.reader(input_ids=torch.tensor([[nxt]], device=self.dev),
                                  past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
        finally:
            R.mode, R.past = "off", {}
        txt = decode(self.tok_r, self.hub_r, new)
        if stop in txt:
            txt = txt.split(stop, 1)[0] + stop
        return txt, len(new)


def parse_layers(spec: str, reader, enc) -> dict:
    """"auto" (pick_read_layers over D13's encoder layers 3,7,11,15,19) or "r:e,r:e,..."."""
    from xenc.d26_read import pick_read_layers, reader_dims
    if spec == "auto":
        return pick_read_layers(reader_dims(reader), reader_dims(enc))
    return {int(a): int(b) for a, b in (x.split(":") for x in spec.split(",") if x)}


def build(a, init: str | None = None, mlp_hidden: int = 0):
    """Reader (any family) + Qwen encoder + D13Interface (identity A or `init`) + B from --B."""
    from xenc.backbone import input_device, load_encoder
    from xenc.d13_read import D13Interface, load_B, load_d13
    from xenc.d26_read import load_reader_d26, resolve_hub
    from xenc.state_skill import text_config
    a.reader = resolve_hub(a.reader)
    reader, tok_r = load_reader_d26(a.reader, device=a.device)
    enc, tok_e = load_encoder(a.encoder, device_map=a.device)
    lmap = parse_layers(a.layers, reader, enc)
    assert a.B, "--B is required"
    B = load_B(a.B, sorted(lmap))
    if init:
        itf, _ = load_d13(init, enc, B)
        assert itf.layer_map == lmap, (itf.layer_map, lmap)
    else:
        itf = D13Interface(enc, lmap, int(text_config(reader).hidden_size), int(text_config(enc).hidden_size),
                           B=B, mlp_hidden=mlp_hidden)
    itf = itf.to(input_device(reader))
    return reader, enc, tok_r, tok_e, itf, lmap


def make_rt(a, reader, enc, tok_r, tok_e, itf) -> D26Rt:
    return D26Rt(reader, enc, tok_r, tok_e, a.reader, a.encoder, itf, a.tail, a.max_tokens, bos=bool(a.bos))


def add_common(ap):
    ap.add_argument("--reader", default="glm4", help="hub id or key of xenc.d26_read.READERS")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="", help="span-aligned ridge B (scripts/fit_d26_B.py); required unless eval --interface none")
    ap.add_argument("--layers", default="auto")
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--bos", type=int, default=1, help="prepend the reader's BOS header to the read prompt")
    ap.add_argument("--device", default="cuda")


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common(ap)
    ap.add_argument("--train", default="quality:150,musique:150,lastingbench:150,longhealth:0")
    ap.add_argument("--val", default="quality:6,musique:12,lastingbench:12,longhealth:0")
    ap.add_argument("--val_fold", type=int, default=1)
    ap.add_argument("--test_fold", type=int, default=0)
    ap.add_argument("--q_per_doc", type=int, default=8)
    ap.add_argument("--val_q_per_doc", type=int, default=4)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--variants", default="mlp")
    ap.add_argument("--mlp_hidden", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr_mlp", type=float, default=1e-4)
    ap.add_argument("--rel_w", type=float, default=1.0)
    ap.add_argument("--onpolicy_p", type=float, default=0.5)
    ap.add_argument("--init_from", default="", help="optional d13/d26 file whose A (+MLP) initialises A instead of "
                    "ridge (transfer test; needs the same d_r)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.train = ",".join(f"{s.split(':')[0]}:2" for s in a.train.split(","))
        a.val = ",".join(f"{s.split(':')[0]}:1" for s in a.val.split(","))
        a.epochs, a.q_per_doc, a.val_q_per_doc = 1, 2, 2
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    from fit_d13 import _group, fmt, mean_terms, select_items
    from fit_state_maps import Gram, ridge
    from xenc.d13_read import D13Interface, save_d13
    reader, enc, tok_r, tok_e, itf, lmap = build(a)
    rt = make_rt(a, reader, enc, tok_r, tok_e, itf)
    tr_folds = set(range(5)) - {a.test_fold, a.val_fold}
    train, d1 = select_items(a.train, "train", tr_folds, a.q_per_doc, a.seed)
    val, d2 = select_items(a.val, "val", {a.val_fold}, a.val_q_per_doc, a.seed + 1)
    rt.docs = {**d1, **d2}
    assert not ({_get(i, "doc_id") for i in train} & {_get(i, "doc_id") for i in val}), "val docs overlap train"
    log(f"D26 stage 1: {a.encoder} -> {a.reader} | layers {lmap} | bos {rt.bos!r} same_tok {rt.same_tok} | train "
        f"{len(train)} q / {len({_get(i, 'doc_id') for i in train})} docs | held-out {len(val)} q / "
        f"{len({_get(i, 'doc_id') for i in val})} docs")
    it0 = train[0]
    P0 = rt.pair(_get(it0, "doc_id"), it0)
    log(f"  example rows: reader {len(P0['r_ids'])} / encoder {len(P0['e_ids'])} tokens, valid {int(P0['valid'].sum())}"
        f" | reader tail {rt.tok_r.decode(P0['r_ids'][:12])!r} … {rt.tok_r.decode(P0['r_ids'][-10:])!r}")
    by_doc = _group(train)
    t0 = time.time()
    d_r, d_e = itf.d_r, itf.d_e
    # ---------------- 1) ridge A (reader layer input -> encoder layer input at aligned rows)
    G = {s: {li: Gram(d_r + 1, d_e, rt.dev) for li in lmap} for s in ("fit", "held")}
    for split, groups in (("fit", by_doc), ("held", _group(val))):
        for did, qs in groups.items():
            for it in qs:
                P = rt.pair(did, it)
                op, n_op, _ = rt.operator(did)
                rec = rt.teacher(did, P["e_ids"])
                R = rt.R
                R.load(op, n_op)
                yt = rt.y_teacher(rec)
                R.mode, R.xcap, R.valid = "teacher", {}, P["valid"]
                R.inject = {li: yt[li][P["idx"].to(yt[li].device)] for li in yt}
                with torch.no_grad():
                    reader(input_ids=torch.tensor([P["r_ids"]], device=rt.dev), use_cache=False, logits_to_keep=1)
                R.mode, R.valid = "off", None
                v = P["valid"].to(rt.dev)
                ii = P["idx"].to(rt.dev)[v]
                for li, e in lmap.items():
                    X = R.xcap[li][v]
                    G[split][li].add(torch.cat([X, torch.ones(X.shape[0], 1, device=X.device)], 1),
                                     rec[e]["x"].float()[ii])
    r2 = {}
    with torch.no_grad():
        for li in lmap:
            W = ridge(G["fit"][li], a.lam)
            r2[str(li)] = {"fit": G["fit"][li].r2(W), "held": G["held"][li].r2(W)}
            itf.A[str(li)].weight.copy_(W[:-1].t().float())
            itf.A[str(li)].bias.copy_(W[-1].float())
    del G
    torch.cuda.empty_cache()
    log("ridge A (reader layer input -> encoder layer input, aligned rows) R² fit/held: " +
        " ".join(f"L{li} {v['fit']:.3f}/{v['held']:.3f}" for li, v in r2.items()) + f" ({time.time() - t0:.0f}s)")

    def held(onpolicy=True):
        return mean_terms([rt.s1_pass(did, it, onpolicy=onpolicy, grad=False, metrics=True)
                           for did, qs in _group(val).items() for it in qs])

    h_tf, h_op = held(False), held(True)
    log(f"[ridge] held-out teacher-forced: {fmt(h_tf)}")
    log(f"[ridge] held-out on-policy:      {fmt(h_op)}")
    report = {"args": vars(a), "lmap": lmap, "ridge_r2": r2, "ridge_held": {"tf": h_tf, "op": h_op}, "variants": {}}
    save_d13(itf, os.path.join(a.out, "d13_ridge.pt"), {"stage": "d26 ridge", "held": h_op, "reader": a.reader})
    ridge_sd = {k: v.detach().clone() for k, v in itf.state_dict().items() if k.startswith("A.")}
    # ---------------- 2) attention distillation
    for var in [v for v in a.variants.split(",") if v]:
        mh = a.mlp_hidden if var == "mlp" else 0
        if mh != itf.mlp_hidden:
            Bd = {li: itf.B[str(li)].detach().cpu() for li in lmap}
            itf = D13Interface(enc, lmap, d_r, d_e, B=Bd, mlp_hidden=mh).to(rt.dev)
            rt.itf = rt.R.itf = itf
        itf.load_state_dict(ridge_sd, strict=False)
        if mh:
            for li in lmap:
                torch.nn.init.zeros_(itf.M[str(li)][2].weight)
                torch.nn.init.zeros_(itf.M[str(li)][2].bias)
        if a.init_from:
            blob = torch.load(a.init_from, map_location="cpu", weights_only=False)["state_dict"]
            sd = {k: v for k, v in blob.items() if k.startswith(("A.", "M."))}
            miss = itf.load_state_dict(sd, strict=False)
            log(f"  A/M initialised from {a.init_from} ({len(sd)} tensors; unexpected {miss.unexpected_keys})")
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
                save_d13(itf, bpath, {"stage": f"d26 attn-distill {var}", "epoch": ep, "held": h, "reader": a.reader})
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


if __name__ == "__main__":
    sys.exit(main())
