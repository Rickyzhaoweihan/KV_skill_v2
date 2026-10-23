"""Evaluate a registered interface on document QA under matched-budget conditions (agent [D]).

Conditions (greedy, same --budget, stop on </answer>):
  closed_book   question only, no library                       -> the floor
  text          document in the prompt (ctx_prompt), no library  -> the expensive baseline
  kv            question only + the item's own Library
  wrong_doc     question only + the Library of ANOTHER doc of the same task (doc order +7)
  null          question only + interface on, content-free operator (--null_mode random: unit
                Gaussian rows shaped like the own Library; zero: all-zero operator ⇒ Δ = 0)
  oracle_chunk  question only + own Library restricted to the gold chunk(s)
                (meta.supporting_idx paragraphs or the chunk holding the first gold answer;
                items with no known gold chunk are skipped for this condition)

Writes <out>/eval_<split>.json: {"summary": {cond: {score, mean_gen_tokens, mean_prompt_tokens, n},
"kv_minus_wrong_doc": {gap, paired_gap, n_paired}},
"items": [per-item records]}; resumable via <out>/eval_<split>.ckpt.jsonl.

    python scripts/eval_doc.py --reader Qwen/Qwen3.5-9B --split_dir data/quality_split --split test \
        --interface outputs/<run>/interface_best.pt --n 300 --budget 64
    python scripts/eval_doc.py --tiny --n 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import zlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402
from register_doc import (DocRuntime, _get, build_injector, load_split_items, log,  # noqa: E402
                          system_prompt, tiny_setup)
from xenc import DEPTHS  # noqa: E402

ALL_CONDS = ("closed_book", "text", "kv", "wrong_doc", "null", "oracle_chunk")
LIB_CONDS = ("kv", "wrong_doc", "null", "oracle_chunk")


# ============================================================================ condition builders
def wrong_doc_id(doc_id: str, doc_order: list[str], shift: int = 7) -> str | None:
    """Deterministic other document: doc_order[(i + shift) % n], stepping on if it lands on doc_id."""
    n = len(doc_order)
    if n < 2 or doc_id not in doc_order:
        return None if n < 2 else doc_order[shift % n]
    i = doc_order.index(doc_id)
    for s in range(shift, shift + n):
        j = doc_order[(i + s) % n]
        if j != doc_id:
            return j
    return None


def null_library(lib, mode: str = "random", seed: int = 0):
    """Content-free operator with the own Library's shape (same chunks/slots/mask)."""
    import copy
    new = copy.copy(lib)
    new.K, new.mask = {}, {}
    new.V = {} if getattr(lib, "V", None) is not None else None   # untied (attn) library: values too
    g = torch.Generator().manual_seed(seed)
    for li, K in lib.K.items():
        if mode == "zero":
            R = torch.zeros_like(K)
        else:
            R = torch.randn(K.shape, generator=g).to(K.device, K.dtype)
            R = R / (R.norm(dim=-1, keepdim=True) + 1e-6)
        M = lib.mask[li]
        new.K[li] = R * M.unsqueeze(-1).to(R.dtype)
        new.mask[li] = M.clone()
        if new.V is not None:
            Rv = torch.zeros_like(K) if mode == "zero" else torch.randn(K.shape, generator=g).to(K.device, K.dtype)
            Rv = Rv / (Rv.norm(dim=-1, keepdim=True) + 1e-6)
            new.V[li] = Rv * M.unsqueeze(-1).to(Rv.dtype)
    new._summaries()
    return new


def build_condition_library(cond: str, item, rt: DocRuntime, doc_order: list[str], null_mode="random"):
    """-> (Library | None, info dict). (None, {"skip": reason}) means the item is skipped for cond."""
    did = _get(item, "doc_id")
    if cond in ("closed_book", "text"):
        return None, {}
    if cond == "wrong_doc":
        w = wrong_doc_id(did, doc_order)
        lib = rt.library(w) if w else None
        return (lib, {"wrong_doc": w}) if lib is not None else (None, {"skip": "no wrong doc"})
    lib = rt.library(did)
    if lib is None:
        return None, {"skip": "no cache"}
    if cond == "kv":
        return lib, {}
    if cond == "null":
        if hasattr(lib, "null_like"):    # D7 native library: norm-matched random K/V rows
            return lib.null_like(null_mode, seed=zlib.crc32(did.encode())), {"null_mode": null_mode}
        return null_library(lib, null_mode, seed=zlib.crc32(did.encode())), {"null_mode": null_mode}
    if cond == "oracle_chunk":
        idx = rt.gold_chunks(item)
        if not idx:
            return None, {"skip": "gold chunk unknown"}
        return lib.subset(idx), {"gold_chunks": idx}
    raise ValueError(f"unknown condition {cond}")


class OperatorRuntime(DocRuntime):
    """D4: per-document compact operators (<operator_dir>/<doc_id>.pt, xenc.doc_operator format) in place
    of the harvested chunk caches. Operators are final (never re-centered); oracle_chunk is skipped."""

    def __init__(self, *args, operator_dir: str, **kw):
        super().__init__(*args, **kw)
        self.operator_dir = operator_dir

    def cache_path(self, doc_id: str) -> str:
        from xenc.doc_operator import operator_path
        return operator_path(self.operator_dir, doc_id)

    def has_cache(self, doc_id: str) -> bool:
        return os.path.exists(self.cache_path(doc_id))

    def get_cache(self, doc_id: str):
        from xenc.doc_operator import load_operator
        if doc_id not in self._lru:
            d = load_operator(self.operator_dir, doc_id)
            if d is None:
                return None
            self._lru[doc_id] = d
        return self._lru[doc_id]

    def library(self, doc_id: str):
        from xenc.library import Library
        d = self.get_cache(doc_id)
        return None if d is None else Library.from_doc_cache(d, self.model, self.depths, device="cpu", center=None)

    def gold_chunks(self, item, cache=None):
        return []


# ============================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default=None, help="default: from the interface's extra (or 'same')")
    ap.add_argument("--split_dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--interface", default=None, help="interface_best.pt (required for library conditions)")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--chunk_tokens", type=int, default=None)
    ap.add_argument("--conds", default=",".join(ALL_CONDS))
    ap.add_argument("--n", type=int, default=300, help="0 = whole split")
    ap.add_argument("--fold", type=int, default=None, help="keep only items with meta.fold == K (held-out fold)")
    ap.add_argument("--center", default=None, help="center file ({rho: mu}); default = the one saved at registration")
    ap.add_argument("--q_split", choices=["all", "train", "heldout"], default="all",
                    help="within-document question split by sha1(key) parity (train=even, heldout=odd); use --split train --q_split heldout to evaluate unseen questions on the SAME documents used for registration")
    ap.add_argument("--max_docs", type=int, default=0,
                    help="restrict items to the first N doc_ids (sorted) of the split, as encode_docs.py "
                         "--max_docs / --splits does (smoke runs); 0 = all")
    ap.add_argument("--no_resume", action="store_true",
                    help="ignore and overwrite <out>/eval_<split>.ckpt.jsonl (default: resume from it)")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1, help="only 1 is implemented")
    ap.add_argument("--style", choices=["auto", "answer", "reason"], default="auto")
    ap.add_argument("--null_mode", choices=["random", "zero"], default="random",
                    help="null condition operator (default: random = unit-Gaussian rows with the own "
                         "Library's chunk/slot/mask shape, seeded by crc32(doc_id); zero = all-zero "
                         "operator, i.e. delta == 0, a sanity check that should equal closed_book)")
    ap.add_argument("--max_ctx_chars", type=int, default=None)
    ap.add_argument("--exclude_fold", type=int, default=None, help="D4: drop items whose meta.fold == K")
    ap.add_argument("--operator_dir", default=None,
                    help="D4: read per-document compact operators <dir>/<doc_id>.pt (scripts/rl_doc_operator.py) "
                         "instead of the harvested chunk caches; no centering, oracle_chunk skipped")
    ap.add_argument("--read_temp", type=float, default=None)
    ap.add_argument("--route_temp", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--read", choices=["library", "topk"], default=None,
                    help="D8: topk = question-dependent top-k slot selection (default: the interface's training --read)")
    from xenc.topk_read import add_topk_args   # D8 flags; unset ones fall back to the interface's training args
    add_topk_args(ap)
    ap.set_defaults(topk=None, select_layer_rho=None, sel_harvest=None, sel_pool=None, sel_query=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_mem", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: the interface's directory")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    if not a.tiny and not a.split_dir:
        ap.error("--split_dir is required unless --tiny")
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    from xenc.backbone import build_chat, generate, input_device, load_reader, short_name
    from xenc.doc_tasks import ctx_prompt, q_prompt, score
    from xenc.interface import load_interface, make_interface

    conds = [c.strip() for c in a.conds.split(",") if c.strip()]
    bad = [c for c in conds if c not in ALL_CONDS]
    if bad:
        raise SystemExit(f"unknown conditions {bad}; choose from {ALL_CONDS}")
    conds = [c for c in ALL_CONDS if c in conds]     # fixed canonical order, whatever the CLI order
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    if a.batch != 1:
        log(f"WARN: --batch {a.batch} not implemented; using 1")
    out = a.out or (os.path.dirname(os.path.abspath(a.interface)) if a.interface else
                    ("outputs/eval_tiny" if a.tiny else "outputs/eval_doc"))
    os.makedirs(out, exist_ok=True)

    extra = {}
    if a.interface:
        blob = torch.load(a.interface, map_location="cpu", weights_only=False)
        extra = blob.get("extra") or {}
    targs = extra.get("args") or {}
    depths = extra.get("depths") or (xenc.CAND if targs.get("selector") else DEPTHS)
    encoder = a.encoder or targs.get("encoder") or "same"
    encoder_same = encoder == "same"
    style = a.style if a.style != "auto" else extra.get("style", "answer")

    class _H:   # injector hyper-params: CLI > training args > defaults
        read_temp = a.read_temp if a.read_temp is not None else targs.get("read_temp", 12.0)
        route_temp = a.route_temp if a.route_temp is not None else targs.get("route_temp", 10.0)
        top_k = a.top_k if a.top_k is not None else targs.get("top_k", 0)
        linear_read = bool(targs.get("linear_read", False))

    if a.tiny:
        torch.set_num_threads(1)
        tz = tiny_setup(out, encoder_same=bool(targs.get("encoder") == "same"), depths=depths)
        model, tok, hub, docs = tz["reader"], tz["tok"], tz["hub"], tz["docs"]
        items = tz["test"]
        cache_dir, enc_short, chunk_tokens = tz["cache_dir"], tz["enc_short"], tz["chunk_tokens"]
        encoder_same = targs.get("encoder") == "same"
        a.budget = min(a.budget, 4)
    else:
        from xenc.doc_tasks import load_docs
        docs = load_docs(a.split_dir)
        items = load_split_items(a.split_dir, a.split)
        cache_dir = a.cache_dir or targs.get("cache_dir") or extra.get("cache_dir") or xenc.DEFAULT_CACHE
        enc_short = short_name(a.reader) if encoder_same else short_name(encoder)
        chunk_tokens = a.chunk_tokens or extra.get("chunk_tokens") or 512
        hub = a.reader
        model, tok = load_reader(a.reader, device_map=a.device, max_mem=a.max_mem, grad_ckpt=False)

    if a.operator_dir:
        rt = OperatorRuntime(model, tok, hub, docs, cache_dir, enc_short, chunk_tokens, depths,
                             encoder_same=False, operator_dir=a.operator_dir)
    else:
        rt = DocRuntime(model, tok, hub, docs, cache_dir, enc_short, chunk_tokens, depths,
                        encoder_same=encoder_same)
    rt.harvest = extra.get("harvest") or targs.get("harvest") or "resid"
    native = targs.get("read") == "native_attn"     # D7
    if native:
        from xenc.native_attn import NativeSpec, native_layers_arg
        rt.native = NativeSpec(model, depths, native_layers_arg(targs.get("native_layers")))
    center_path = a.center or (extra.get("center") if not a.tiny else None)
    if a.operator_dir:
        center_path = None    # operators are stored final
    if center_path:
        rt.center = {float(k): v for k, v in torch.load(center_path, map_location="cpu", weights_only=True)["center"].items()}
        print(f"centering slots with {center_path}", flush=True)
    by_task: dict[str, list[str]] = {}
    for i in items:   # doc order per task, over the WHOLE split (before sub-sampling)
        did = _get(i, "doc_id")
        if rt.has_cache(did):
            by_task.setdefault(_get(i, "task") or "", set()).add(did)
    by_task = {t: sorted(v) for t, v in by_task.items()}
    doc_order = sorted({d for v in by_task.values() for d in v})
    if a.q_split != "all":
        import hashlib as _hl
        want = 0 if a.q_split == "train" else 1
        items = [i for i in items if int(_hl.sha1(str(_get(i, "key")).encode()).hexdigest(), 16) % 2 == want]
        print(f"q_split={a.q_split}: {len(items)} items kept", flush=True)
    if getattr(a, "exclude_fold", None) is not None:   # D4: drop items of fold K (e.g. interface-registration docs)
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") != a.exclude_fold]
        print(f"exclude_fold={a.exclude_fold}: {len(items)} items kept", flush=True)
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
        print(f"fold={a.fold}: {len(items)} items kept", flush=True)
    if a.max_docs:
        keep = set(sorted({_get(i, "doc_id") for i in items})[: a.max_docs])
        items = [i for i in items if _get(i, "doc_id") in keep]
    if a.operator_dir:    # D4: only documents that have an operator (all conditions on the same items)
        items = [i for i in items if rt.has_cache(_get(i, "doc_id"))]
        print(f"operator_dir: {len(items)} items on docs with an operator", flush=True)
    if a.n and len(items) > a.n:
        items = random.Random(a.seed).sample(items, a.n)

    inj = None
    if any(c in LIB_CONDS for c in conds):
        dev = input_device(model)
        d_enc = rt.d_encoder(doc_order) or rt.d_encoder([_get(i, "doc_id") for i in items])
        if d_enc is None:
            raise SystemExit("no encoder cache found for any doc; run scripts/encode_docs.py first")
        if native:      # D7: native-attention read (reader's own q/o projections)
            from xenc.native_attn import NativeAttnInjector, load_native_interface
            itf, _ = load_native_interface(a.interface)
            itf, selector = itf.to(dev), None
        elif a.interface:
            itf, selector, _ = load_interface(a.interface, rt.d_reader, d_enc, rt.layers)
        elif a.tiny:
            itf, selector = make_interface(rt.d_reader, d_enc, rt.layers), None
        else:
            raise SystemExit("--interface is required for library conditions")
        itf = itf.to(dev)
        selector = selector.to(dev) if selector is not None else None
        inj = NativeAttnInjector(model, itf) if native else build_injector(model, itf, _H, selector)
        if (a.read or targs.get("read")) == "topk" and not native:     # D8
            from xenc.topk_read import build_topk
            for k_, dflt in (("topk", 64), ("select_layer_rho", 0.25), ("sel_harvest", "attn"), ("sel_pool", "max"),
                             ("sel_query", "last"), ("sel_kmap", None)):
                setattr(_H, k_, getattr(a, k_) if getattr(a, k_) is not None else targs.get(k_, dflt))
            inj.detach()
            inj = build_topk(model, rt, _H, itf)
            log(f"D8 topk read: k={_H.topk}, selection layer {inj.tk.sel_layer} ({_H.sel_harvest}, pool {_H.sel_pool}, "
                f"query {_H.sel_query})")
        rt.injector = inj      # disabled while an L/L reader lazily encodes a document

    ckpt_path = os.path.join(out, f"eval_{a.split}.ckpt.jsonl")
    rows = []
    if os.path.exists(ckpt_path) and not a.no_resume:
        done = {}
        for ln in open(ckpt_path):
            if ln.strip():
                r = json.loads(ln)
                if all(c in r for c in conds):      # a row missing a requested condition is redone
                    done[r["key"]] = r
        rows = [done[_get(i, "key")] for i in items if _get(i, "key") in done]
        log(f"resume: {len(rows)} items from {ckpt_path}")
    done_keys = {r["key"] for r in rows}
    ck = open(ckpt_path, "w" if a.no_resume else "a")
    sysp = system_prompt(style)
    t0 = time.time()
    for k, it in enumerate(items):
        if _get(it, "key") in done_keys:
            continue
        did = _get(it, "doc_id")
        rec = {"key": _get(it, "key"), "doc_id": did}
        for c in conds:
            if c == "text" and did not in docs:
                rec[c] = {"skip": "no doc"}
                continue
            lib, info = build_condition_library(c, it, rt, by_task.get(_get(it, "task") or "", doc_order), a.null_mode) if c in LIB_CONDS else (None, {})
            if "skip" in info:
                rec[c] = info
                continue
            user = ctx_prompt(it, docs[did], a.max_ctx_chars) if c == "text" else q_prompt(it)
            ids = build_chat(tok, hub, sysp, user)
            if lib is not None:
                inj.load_library(lib)
            try:
                g, ng = generate(model, tok, hub, ids, a.budget, return_n=True)
            finally:
                if inj is not None:
                    inj.unload()
            rec[c] = {"gen": g, "score": float(score(it, g)), "gen_tokens": ng, "prompt_tokens": len(ids), **info}
        rows.append(rec)
        ck.write(json.dumps(rec) + "\n")
        ck.flush()
        if (k + 1) % 20 == 0 or a.tiny:
            log(f"  [{len(rows)}/{len(items)}] " + " ".join(
                f"{c}={_mean(rows, c, 'score'):.3f}" for c in conds) + f" | {(time.time() - t0) / max(len(rows), 1):.1f}s/item")
    ck.close()

    summary = {c: {"score": _mean(rows, c, "score"), "mean_gen_tokens": _mean(rows, c, "gen_tokens"),
                   "mean_prompt_tokens": _mean(rows, c, "prompt_tokens"),
                   "n": sum(1 for r in rows if "score" in r.get(c, {}))} for c in conds}
    if "kv" in summary and "wrong_doc" in summary:
        summary["identity_kv_minus_wrong"] = summary["kv"]["score"] - summary["wrong_doc"]["score"]
        pair = [r["kv"]["score"] - r["wrong_doc"]["score"] for r in rows
                if "score" in r.get("kv", {}) and "score" in r.get("wrong_doc", {})]
        summary["kv_minus_wrong_doc"] = {"gap": summary["identity_kv_minus_wrong"],
                                         "paired_gap": sum(pair) / len(pair) if pair else float("nan"),
                                         "n_paired": len(pair)}
    res = {"summary": summary, "split": a.split, "n": len(rows), "budget": a.budget, "style": style,
           "reader": hub, "encoder": enc_short, "interface": a.interface, "conds": conds, "items": rows}
    path = os.path.join(out, f"eval_{a.split}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    log(f"==== eval [{a.split}] n={len(rows)} budget={a.budget} -> {path}")
    for c in conds:
        s = summary[c]
        log(f"  {c:13s} {s['score']:.4f}  n={s['n']:4d}  gen {s['mean_gen_tokens']:.1f}  prompt {s['mean_prompt_tokens']:.0f}")
    if "kv_minus_wrong_doc" in summary:
        g = summary["kv_minus_wrong_doc"]
        log(f"  kv - wrong_doc  {g['gap']:+.4f}  (paired {g['paired_gap']:+.4f}, n={g['n_paired']})")
    if torch.cuda.is_available():
        log(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    return 0


def _mean(rows, c, k) -> float:
    v = [r[c][k] for r in rows if k in r.get(c, {})]
    return sum(v) / len(v) if v else float("nan")


if __name__ == "__main__":
    sys.exit(main())
