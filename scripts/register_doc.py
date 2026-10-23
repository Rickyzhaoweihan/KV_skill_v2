"""Register a reader interface (and optional LayerSelector) to read per-document Libraries (agent [D]).

Port of delta-skill/scripts/longhealth_kv.py onto xenc: teacher = READER with the document in the
prompt (`ctx_prompt`), student = READER on the question alone (`q_prompt`) with the item's Library
(built from the ENCODER's disk cache, never re-derived here unless `--encoder same`) loaded through
`LibraryInjector`. Only the interface (+ selector) train; the backbone is frozen.

    loss = kl·KL(p_T‖p_S) + ce·CE(target) + mag·mean(||Δ||/||h||) + route_aux·(−log w[gold chunks])

`--target gold`: target = `<answer>X</answer>` (KL + CE over its m tokens).
`--target trace`: target = the teacher's own greedy trace (doc in context, reasoning system prompt),
generated once per train item and cached in `<out>/traces.json`; KL over the whole trace, CE on the
answer span only. A trace whose answer scores 0 is handled by `--wrong_trace kl_only|drop|gold`.

Outputs in --out: interface_best.pt (val kv score every --eval_every), interface_last.pt,
history.json, args.json (+ traces.json). Prints torch.cuda.max_memory_allocated() at the end.

    python scripts/register_doc.py --reader Qwen/Qwen3.5-9B --encoder Qwen/Qwen3.5-2B \
        --split_dir data/quality_split --out outputs/q_9b_2b --steps 300
    python scripts/register_doc.py --tiny --steps 3 --out /tmp/x      # CPU plumbing, TinyLM
    python scripts/register_doc.py --dry_run --reader ... --split_dir ...   # cache coverage only

Helpers here (runtime, prompts, gold-chunk mapping, tiny fixtures) are imported by eval_doc.py and
ladder.py.
"""

from __future__ import annotations

import argparse
import contextlib
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

import xenc  # noqa: E402
from xenc import CAND, DEPTHS  # noqa: E402

SYSTEM_ANSWER = ("You are a careful question answering assistant. Provide your final answer inside "
                 "<answer>...</answer> tags.")
SYSTEM_REASON = ("You are a careful question answering assistant. Think step by step briefly, then give "
                 "your final answer inside <answer>...</answer> tags.")


def system_prompt(style: str = "answer") -> str:
    return SYSTEM_REASON if style == "reason" else SYSTEM_ANSWER


def _get(item, k, default=None):
    if isinstance(item, dict):
        return item.get(k, default)
    return getattr(item, k, default)


def log(msg: str) -> None:
    print(msg, flush=True)


# ============================================================================ tiny fixtures
TINY_CHUNK = 16


def tiny_setup(workdir: str, encoder_same: bool = False, depths=None, chunk_tokens: int = TINY_CHUNK,
               reader_d: int = 32, seed: int = 0):
    """TinyLM reader (+ separate TinyLM encoder unless encoder_same), whitespace fake tokenizer,
    3 synthetic docs of 3 paragraphs ([P{i}] markers, one paragraph == one chunk) and 4 items.
    Pre-encodes the docs into <workdir>/tiny_cache when the encoder is separate.
    -> dict(reader, encoder, tok, hub, enc_short, docs, train, val, test, cache_dir, chunk_tokens)"""
    sys.path.insert(0, str(_ROOT / "tests"))
    from _tiny import TinyLM
    from xenc.depth_skill import TinyTokenizer, cache_key, derive_by_depth, save_doc_cache
    from xenc.doc_tasks import DocItem

    depths = sorted(depths or DEPTHS)
    reader = TinyLM(d=reader_d, n_layers=10, seed=seed).eval().requires_grad_(False)
    encoder = reader if encoder_same else TinyLM(d=24, n_layers=10, seed=seed + 1).eval().requires_grad_(False)
    tok = TinyTokenizer(vocab=50)
    rng = random.Random(seed)
    words = [f"w{i}" for i in range(120)]
    docs, items = {}, []
    for d in range(3):
        paras = []
        for p in range(3):
            body = [rng.choice(words) for _ in range(chunk_tokens - 1)]
            paras.append(f"[P{p}] " + " ".join(body))
        did = f"tinydoc_{d}"
        docs[did] = "\n".join(paras)
        ans = paras[1].split()[3]
        items.append(DocItem(key=f"tiny_{d}", doc_id=did, question=f"What is word three of paragraph one in {did}?",
                             choices=None, answers=[ans], correct_label=None, task="tiny", hop=None,
                             meta={"supporting_idx": [1]}))
    items.append(DocItem(key="tiny_mcq", doc_id="tinydoc_2", question="Which word opens paragraph two?",
                         choices=[{"label": "A", "text": "w1"}, {"label": "B", "text": "w2"}],
                         answers=["w2"], correct_label="B", task="tiny", hop=None,
                         meta={"supporting_idx": [2, 0]}))
    cache_dir = os.path.join(workdir, "tiny_cache")
    os.makedirs(cache_dir, exist_ok=True)
    enc_short = "tiny-reader" if encoder_same else "tiny-enc"
    if not encoder_same:
        for did, text in docs.items():
            path = os.path.join(cache_dir, cache_key(text, enc_short, chunk_tokens, depths) + ".pt")
            if not os.path.exists(path):
                save_doc_cache(derive_by_depth(encoder, tok, text, depths, chunk_tokens, doc_id=did,
                                               encoder=enc_short), path)
    return {"reader": reader, "encoder": encoder, "tok": tok, "hub": "tiny", "enc_short": enc_short,
            "docs": docs, "train": items, "val": items, "test": items, "cache_dir": cache_dir,
            "chunk_tokens": chunk_tokens}


# ============================================================================ runtime
class DocRuntime:
    """Reader + doc-cache access (LRU), Library construction, gold-chunk mapping."""

    def __init__(self, model, tok, hub: str, docs: dict, cache_dir: str, enc_short: str,
                 chunk_tokens: int, depths, encoder_same: bool = False, max_doc_tokens: int = 0,
                 lru: int = 50):
        from xenc.depth_skill import rho_layer_map
        self.model, self.tok, self.hub, self.docs = model, tok, hub, docs
        self.cache_dir, self.enc_short, self.chunk_tokens = cache_dir, enc_short, int(chunk_tokens)
        self.depths = sorted(float(r) for r in depths)
        self.encoder_same, self.max_doc_tokens = encoder_same, max_doc_tokens
        rmap = rho_layer_map(model, self.depths)
        self.layers = [rmap[r] for r in self.depths]
        c = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        self.d_reader = int(c.hidden_size)
        self._lru: OrderedDict = OrderedDict()
        self._lru_cap = lru
        self._missing_warned: set = set()
        self.injector = None   # set by the caller; disabled while the reader encodes itself (L/L)
        self.center = None     # optional {rho: mu} from scripts/compute_center.py (set by the caller)
        self.harvest = "resid"  # "resid" | "attn" (D3: encoder attention K/V, untied); set by the caller

    def cache_path(self, doc_id: str) -> str:
        from xenc.depth_skill import cache_key
        return os.path.join(self.cache_dir, cache_key(self.docs[doc_id], self.enc_short,
                                                      self.chunk_tokens, self.depths,
                                                      harvest=getattr(self, "harvest", "resid")) + ".pt")

    def has_cache(self, doc_id: str) -> bool:
        return doc_id in self.docs and (self.encoder_same or os.path.exists(self.cache_path(doc_id)))

    def get_cache(self, doc_id: str) -> dict | None:
        if doc_id in self._lru:
            self._lru.move_to_end(doc_id)
            return self._lru[doc_id]
        if doc_id not in self.docs:
            return None
        from xenc.depth_skill import derive_by_depth, load_doc_cache, save_doc_cache
        path = self.cache_path(doc_id)
        if os.path.exists(path):
            d = load_doc_cache(path)
        elif self.encoder_same:   # L/L: encode with the reader itself, then persist
            inj = self.injector
            was = inj.enabled if inj is not None else None
            if inj is not None:
                inj.enabled = False   # never encode through a loaded library / trained branch
            try:
                d = derive_by_depth(self.model, self.tok, self.docs[doc_id], self.depths, self.chunk_tokens,
                                    self.max_doc_tokens, doc_id=doc_id, encoder=self.hub,
                                    harvest=getattr(self, "harvest", "resid"))
            finally:
                if inj is not None:
                    inj.enabled = was
            save_doc_cache(d, path)
        else:
            if doc_id not in self._missing_warned:
                log(f"WARN: no encoder cache for {doc_id} ({os.path.basename(path)}); skipping its items")
                self._missing_warned.add(doc_id)
            return None
        self._lru[doc_id] = d
        while len(self._lru) > self._lru_cap:
            self._lru.popitem(last=False)
        return d

    def library(self, doc_id: str):
        from xenc.library import Library
        d = self.get_cache(doc_id)
        if d is None:
            return None
        if getattr(self, "native", None) is not None:    # D7: raw attention K/V for the native-attention read
            return self.native.library(d)
        if getattr(self, "topk", None) is not None:      # D8: residual Library + slot-aligned attention keys
            return self.topk.library(self, doc_id, d)
        return Library.from_doc_cache(d, self.model, self.depths, device="cpu", center=self.center)

    def d_encoder(self, doc_ids) -> int | None:
        for did in doc_ids:
            d = self.get_cache(did)
            if d is not None:
                return int(d["d_s"])
        return None

    # ---- gold chunk mapping
    def tok_offset(self, doc_text: str, char_idx: int) -> int:
        from xenc.backbone import text_ids
        return len(text_ids(self.tok, doc_text[:char_idx])) if char_idx > 0 else 0

    def gold_chunks(self, item, cache: dict | None = None) -> list[int]:
        doc_id = _get(item, "doc_id")
        cache = cache if cache is not None else self.get_cache(doc_id)
        if cache is None:
            return []
        return gold_chunk_indices(item, self.docs.get(doc_id, ""), cache, self.tok_offset)


def gold_chunk_indices(item, doc_text: str, cache: dict, tok_offset) -> list[int]:
    """Gold chunk(s) of an item's Library.

    1. `meta.supporting_idx` (MuSiQue paragraphs): the chunk holding each `[P{i}]` marker; if the
       document has no such markers, the indices are taken as chunk indices directly.
    2. else the chunk containing the first occurrence of the first gold answer string (≥3 chars,
       case-insensitive).
    Returns sorted unique indices ([] = unknown). `tok_offset(doc_text, char) -> token index`."""
    chunks = cache["chunks"]
    C = len(chunks)
    starts = [int(c.get("start", 0)) for c in chunks]

    def chunk_of_tok(t: int) -> int:
        k = 0
        for i, s in enumerate(starts):
            if s <= t:
                k = i
        return k

    meta = _get(item, "meta") or {}
    sup = meta.get("supporting_idx")
    if sup:
        out = []
        has_markers = "[P" in doc_text
        for p in sup:
            k = doc_text.find(f"[P{int(p)}]")
            if k >= 0:
                out.append(chunk_of_tok(tok_offset(doc_text, k)))
            elif not has_markers and 0 <= int(p) < C:
                out.append(int(p))
        return sorted(set(out))
    answers = list(_get(item, "answers") or [])
    if _get(item, "choices") and _get(item, "correct_label"):
        lab = _get(item, "correct_label")
        answers += [c["text"] for c in _get(item, "choices") if c.get("label") == lab]
    low = doc_text.lower()
    for a in answers:
        a = str(a).strip()
        if len(a) < 3:
            continue
        k = low.find(a.lower())
        if k >= 0:
            return [chunk_of_tok(tok_offset(doc_text, k))]
    return []


# ============================================================================ build helpers
def build_injector(model, itf, args, selector=None):
    from xenc.injector import LibraryInjector
    return LibraryInjector(model, itf, read_softmax=not getattr(args, "linear_read", False),
                           read_temp=args.read_temp, route_temp=args.route_temp, top_k=args.top_k,
                           selector=selector)


def load_split_items(split_dir: str, split: str):
    from xenc.doc_tasks import load_doc_split
    return load_doc_split(split_dir, split)


def task_name(split_dir: str | None, tiny: bool = False) -> str:
    if tiny or not split_dir:
        return "tiny"
    n = Path(split_dir.rstrip("/")).name
    return n[:-6] if n.endswith("_split") else n


@torch.no_grad()
def eval_kv(rt: DocRuntime, inj, items, budget: int, style: str = "answer") -> dict:
    """Mean kv-condition score (own Library) + mean generated tokens over `items`."""
    from xenc.backbone import build_chat, generate
    from xenc.doc_tasks import q_prompt, score
    tot, ntok, n = 0.0, 0, 0
    for it in items:
        lib = rt.library(_get(it, "doc_id"))
        if lib is None:
            continue
        ids = build_chat(rt.tok, rt.hub, system_prompt(style), q_prompt(it))
        inj.load_library(lib)
        try:
            g, k = generate(rt.model, rt.tok, rt.hub, ids, budget, return_n=True)
        finally:
            inj.unload()
        tot += score(it, g)
        ntok += k
        n += 1
    return {"score": tot / max(n, 1), "gen_tokens": ntok / max(n, 1), "n": n}


# ============================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B", help="hub id, or 'same' (encode with the reader)")
    ap.add_argument("--split_dir", default=None)
    ap.add_argument("--cache_dir", default=None, help=f"default $XENC_CACHE or {xenc.DEFAULT_CACHE}")
    ap.add_argument("--out", default=None)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--selector_lr", type=float, default=None)
    ap.add_argument("--lr_A", type=float, default=None, help="separate lr for the query map itf.A (D5)")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--target", choices=["gold", "trace"], default="gold")
    ap.add_argument("--wrong_trace", choices=["kl_only", "drop", "gold"], default="kl_only")
    ap.add_argument("--trace_budget", type=int, default=512)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--ce", type=float, default=0.5)
    ap.add_argument("--mag", type=float, default=0.3)
    ap.add_argument("--route_aux", type=float, default=0.1)
    ap.add_argument("--slot_aux", type=float, default=0.0,
                    help="D2: weight of the slot-level retrieval aux −log(read-attention mass on gold slots) at the "
                         "last prompt position, mean over layers (0 = off; items without a gold slot get no term)")
    ap.add_argument("--attn_distill", type=float, default=0.0,
                    help="D9: weight of KL(teacher doc-attention over slots || student route x slot read attention) at "
                         "the last prompt position (teacher = doc-in-context reader, full-attn layers per window; 0 = off)")
    ap.add_argument("--hid_distill", type=float, default=0.0,
                    help="D9: weight of the RMS-normalised residual distillation after each injection layer at the last "
                         "m+1 positions (0 = off)")
    ap.add_argument("--hid_ratio", type=int, default=1, choices=[0, 1],
                    help="D9: 1 = divide by the no-operator gap ||n(h0)-n(hT)||^2 (detached); 0 = raw mean sq. error / d")
    ap.add_argument("--hid_layer_w", default=None, help="D9: comma list of per-injection-layer weights (default equal)")
    ap.add_argument("--chunk_tokens", type=int, default=512)
    ap.add_argument("--top_k", type=int, default=0, help="routing top-k over chunks (0 = dense)")
    ap.add_argument("--read_temp", type=float, default=12.0)
    ap.add_argument("--route_temp", type=float, default=10.0)
    ap.add_argument("--linear_read", action="store_true", help="linear within-chunk read (default softmax)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--selector", type=int, default=0, choices=[0, 1],
                    help="1 = LayerSelector over CAND depths (encoder cache must be derived at CAND)")
    ap.add_argument("--sel_k", type=int, default=2)
    ap.add_argument("--max_docs", type=int, default=0, help="restrict train items to the first N docs (0=all)")
    ap.add_argument("--doc_offset", type=int, default=0,
                    help="D4: skip the first K sorted train docs before --max_docs (register on docs disjoint from others)")
    ap.add_argument("--only_fold", type=int, default=None,
                    help="D4: keep ONLY items with meta.fold == K in train and val (inverse of --test_fold)")
    ap.add_argument("--read", choices=["library", "native_attn", "topk"], default="library",
                    help="D7: native_attn = the reader's own q_proj/q_norm (A) and o_proj (B) read the encoder's raw "
                         "attention K/V per head (needs --harvest attn_raw); trains only per-layer null/gate/gain/temp "
                         "(+ the KV maps with --train_maps 1)")
    from xenc.topk_read import add_topk_args   # D8: --topk --select_layer_rho --sel_harvest --sel_pool --sel_query --sel_kmap
    add_topk_args(ap)
    ap.add_argument("--native_layers", default=None, help="D7: comma list of reader attention layers to inject (default all)")
    ap.add_argument("--null_init", type=float, default=-30.0, help="D7: initial null-slot logit (-30 = off)")
    ap.add_argument("--gain_init", type=float, default=0.0, help="D7: initial log-gain (0 => gamma = 1)")
    ap.add_argument("--out_gate", type=int, default=1, choices=[0, 1], help="D7: apply the module's gated-attention output gate")
    ap.add_argument("--kv_map", default=None, help="D7: encoder->reader KV maps from scripts/fit_kv_map.py (S/L)")
    ap.add_argument("--train_maps", type=int, default=0, choices=[0, 1], help="D7: fine-tune the KV maps")
    ap.add_argument("--map_lr", type=float, default=None, help="D7: lr of the KV maps (default --lr)")
    ap.add_argument("--depths", default=None, help="comma list of encoder cache depths (default xenc.DEPTHS / CAND)")
    ap.add_argument("--harvest", choices=["resid", "attn", "attn_raw"], default="resid",
                    help="encoder operator source: residual stream (tied K=V) or attention keys/values (D3, untied)")
    ap.add_argument("--center", default=None,
                    help="path to a center file from scripts/compute_center.py ({rho: mu}); slots become L2(slot - mu)")
    ap.add_argument("--q_split", choices=["all", "train", "heldout"], default="all",
                    help="within-document question split by sha1(key) parity: train=even, heldout=odd; 'train' here keeps even-parity questions for registration so eval_doc --q_split heldout tests the SAME docs on unseen questions")
    ap.add_argument("--test_fold", type=int, default=None,
                    help="hold out items whose meta.fold == K from train AND val (LongHealth-style CV)")
    ap.add_argument("--max_teacher_tokens", type=int, default=32000)
    ap.add_argument("--max_doc_tokens", type=int, default=0, help="--encoder same: truncate doc for encoding")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1, help="items per optimizer step (grad accumulation)")
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--n_val", type=int, default=32)
    ap.add_argument("--budget", type=int, default=256, help="generation budget for val eval")
    ap.add_argument("--device", default="cuda", help="cuda | cuda:N | auto | cuda:0+cuda:1")
    ap.add_argument("--max_mem", default=None, help="per-GPU budget for sharded loads, e.g. 40GiB")
    ap.add_argument("--grad_ckpt", type=int, default=1, choices=[0, 1])
    ap.add_argument("--init", default=None, help="warm-start interface_*.pt")
    ap.add_argument("--init_A", default=None, help="init the query map A from scripts/fit_query_map.py output (ridge-fit reader->encoder)")
    ap.add_argument("--lru", type=int, default=50)
    ap.add_argument("--tiny", action="store_true", help="TinyLM + fake tokenizer, CPU, 3 synthetic docs")
    ap.add_argument("--dry_run", action="store_true", help="report data/cache coverage and exit (no model)")
    a = ap.parse_args(argv)
    if not a.tiny and not a.split_dir:
        ap.error("--split_dir is required unless --tiny")
    if a.out is None:
        a.out = "outputs/register_tiny" if a.tiny else f"outputs/reg_{task_name(a.split_dir)}_" \
            f"{a.reader.split('/')[-1]}_{a.encoder.split('/')[-1]}_{a.target}_s{a.seed}"
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    from xenc.backbone import (build_chat, eval_mode, generate, input_device, load_reader, short_name,
                               text_ids, train_mode, with_target)
    from xenc.doc_tasks import ctx_prompt, gold_text, q_prompt, score
    from xenc.interface import interface_param_groups, load_interface, make_interface, save_interface
    from xenc.losses import (answer_ce, answer_span_mask, kl_teacher_student, magnitude_penalty,
                             route_aux_loss, slot_aux_loss, student_logits, teacher_logits, trace_target)
    from xenc.gold_slots import item_gold_slot_mask
    from xenc.selector import LayerSelector

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "args.json"), "w") as f:   # written at start; enriched below
        json.dump(vars(a), f, indent=1)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    depths = CAND if a.selector else DEPTHS
    if getattr(a, "depths", None):
        depths = sorted(float(x) for x in a.depths.split(","))
    encoder_same = a.encoder == "same"

    # ---------------- data + model
    if a.tiny:
        torch.set_num_threads(1)
        tz = tiny_setup(a.out, encoder_same=encoder_same, depths=depths)
        model, tok, hub, docs = tz["reader"], tz["tok"], tz["hub"], tz["docs"]
        train, val = tz["train"], tz["val"]
        cache_dir, enc_short, chunk_tokens = tz["cache_dir"], tz["enc_short"], tz["chunk_tokens"]
        a.eval_every = min(a.eval_every, 2)
        a.budget = min(a.budget, 4)
        a.trace_budget = min(a.trace_budget, 6)
    else:
        from xenc.doc_tasks import load_docs
        docs = load_docs(a.split_dir)
        train = load_split_items(a.split_dir, "train")
        val = load_split_items(a.split_dir, "val")
        cache_dir = a.cache_dir or xenc.DEFAULT_CACHE
        enc_short = short_name(a.reader) if encoder_same else short_name(a.encoder)
        chunk_tokens = a.chunk_tokens
        hub = a.reader
        if a.dry_run:
            from xenc.depth_skill import cache_key
            for name, its in (("train", train), ("val", val)):
                ids = sorted({_get(i, "doc_id") for i in its})
                have = sum(os.path.exists(os.path.join(cache_dir, cache_key(docs[d], enc_short, chunk_tokens,
                                                                            depths, harvest=a.harvest) + ".pt"))
                           for d in ids if d in docs)
                log(f"[dry_run] {name}: {len(its)} items / {len(ids)} docs; cached for {enc_short} "
                    f"(c{chunk_tokens}, depths {depths}): {have}/{len(ids)} in {cache_dir}")
            return 0
        model, tok = load_reader(a.reader, device_map=a.device, max_mem=a.max_mem, grad_ckpt=bool(a.grad_ckpt))

    rt = DocRuntime(model, tok, hub, docs, cache_dir, enc_short, chunk_tokens, depths,
                    encoder_same=encoder_same, max_doc_tokens=a.max_doc_tokens, lru=a.lru)
    rt.harvest = a.harvest
    if a.center:
        rt.center = {float(k): v for k, v in torch.load(a.center, map_location="cpu", weights_only=True)["center"].items()}
        log(f"centering slots with {a.center} (depths {sorted(rt.center)})")

    if a.q_split != "all":
        import hashlib as _hl
        _par = lambda i: int(_hl.sha1(str(_get(i, "key")).encode()).hexdigest(), 16) % 2
        want = 0 if a.q_split == "train" else 1
        train = [i for i in train if _par(i) == want]
        log(f"q_split={a.q_split}: train {len(train)} items after within-doc question split")
    if a.test_fold is not None:
        _nf = lambda i: (_get(i, "meta") or {}).get("fold")
        train = [i for i in train if _nf(i) != a.test_fold]
        val = [i for i in val if _nf(i) != a.test_fold]
        log(f"test_fold={a.test_fold}: train {len(train)} items, val {len(val)} items after holdout")
    if getattr(a, "only_fold", None) is not None:   # D4: register ONLY on fold-K docs (train AND val)
        _of = lambda i: (_get(i, "meta") or {}).get("fold")
        train = [i for i in train if _of(i) == a.only_fold]
        val = [i for i in val if _of(i) == a.only_fold]
        log(f"only_fold={a.only_fold}: train {len(train)} items, val {len(val)} items")
    doc_order = sorted({_get(i, "doc_id") for i in train})
    if getattr(a, "doc_offset", 0):                 # D4: skip the first K docs (disjoint from RL docs)
        doc_order = doc_order[a.doc_offset:]
    if a.max_docs:
        doc_order = doc_order[: a.max_docs]
    keep = set(doc_order)
    pool = [i for i in train if _get(i, "doc_id") in keep and rt.has_cache(_get(i, "doc_id"))]
    if not pool:
        log("ERROR: no train item has an encoder cache; run scripts/encode_docs.py first")
        return 2
    val_items = [i for i in val if rt.has_cache(_get(i, "doc_id"))]
    rng_val = random.Random(a.seed + 1)
    val_items = rng_val.sample(val_items, min(a.n_val, len(val_items))) if val_items else []
    d_enc = rt.d_encoder([_get(i, "doc_id") for i in pool])
    log(f"reader {hub} (d={rt.d_reader}, layers {rt.layers}) | encoder {enc_short} (d_s={d_enc}) | "
        f"train pool {len(pool)} items / {len({_get(i, 'doc_id') for i in pool})} docs | val {len(val_items)}")

    # ---------------- interface / selector / injector
    dev = input_device(model)
    selector = LayerSelector(rt.d_reader, rt.layers, k=a.sel_k).to(dev) if a.selector else None
    if a.init and getattr(a, "read", "library") != "native_attn":   # (D7: native init is loaded by build_native)
        itf, sel0, _ = load_interface(a.init, rt.d_reader, d_enc, rt.layers)
        if selector is not None and sel0 is not None:
            selector = sel0.to(dev)
    else:
        itf = make_interface(rt.d_reader, d_enc, rt.layers, rank=a.rank)
    if a.init_A:
        qm = torch.load(a.init_A, map_location="cpu", weights_only=True)
        Wq = qm["A_weight"]
        if a.rank:
            raise SystemExit("--init_A requires a full-rank interface (rank 0)")
        if tuple(itf.A.weight.shape) != tuple(Wq.shape):
            raise SystemExit(f"--init_A shape {tuple(Wq.shape)} != A {tuple(itf.A.weight.shape)}")
        with torch.no_grad():
            itf.A.weight.copy_(Wq.to(itf.A.weight.dtype))
        log(f"query map A initialised from {a.init_A} ({qm.get('n_pairs')} pairs)")
    itf = itf.to(dev)
    inj = build_injector(model, itf, a, selector)
    rt.injector = inj
    opt = torch.optim.AdamW(interface_param_groups(itf, a.lr, selector, a.selector_lr, lr_A=a.lr_A), weight_decay=a.weight_decay)
    train_params = [p for g in opt.param_groups for p in g["params"]]
    if a.read == "topk":          # D8: question-dependent top-k selection, then the standard read over k slots
        from xenc.topk_read import build_topk
        inj.detach()
        inj = build_topk(model, rt, a, itf)
        rt.injector = inj
        log(f"topk read: k={a.topk}, selection layer {inj.tk.sel_layer} (rho {a.select_layer_rho}, {a.sel_harvest} keys, "
            f"pool {a.sel_pool}, query {a.sel_query})")
    if a.read == "native_attn":   # D7: replace the from-scratch interface by the reader's own attention
        from xenc.native_attn import build_native, native_param_groups
        inj.detach()
        selector = None
        itf, inj, save_interface = build_native(model, rt, a, d_enc, dev)
        rt.injector = inj
        opt = torch.optim.AdamW(native_param_groups(itf, a.lr, a.map_lr), weight_decay=0.0)
        train_params = [p for g in opt.param_groups for p in g["params"]]
        log(f"native_attn read: layers {itf.inject_layers}, d_src {itf.d_src} -> d_dst {itf.d_dst}, maps "
            f"{itf.has_maps} (train {bool(a.train_maps)}), trainable {sum(p.numel() for p in train_params)}")

    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump({**vars(a), "depths": depths, "layers": rt.layers, "d_reader": rt.d_reader, "d_encoder": d_enc,
                   "enc_short": enc_short, "cache_dir": cache_dir, "chunk_tokens": chunk_tokens}, f, indent=1)

    style = "reason" if a.target == "trace" else "answer"
    sysp = system_prompt(style)

    # ---------------- trace cache
    trace_path = os.path.join(a.out, "traces.json")
    traces = json.load(open(trace_path)) if (a.target == "trace" and os.path.exists(trace_path)) else {}

    def get_trace(it) -> dict:
        k = _get(it, "key")
        if k not in traces:
            inj.unload()
            ids = build_chat(tok, hub, sysp, ctx_prompt(it, docs[_get(it, "doc_id")]))
            g = generate(model, tok, hub, ids, a.trace_budget)
            tr = trace_target(g)
            traces[k] = {"trace": tr, "score": float(score(it, tr))}
            with open(trace_path + ".tmp", "w") as f:
                json.dump(traces, f)
            os.replace(trace_path + ".tmp", trace_path)
        return traces[k]

    def make_target(it):
        """-> (target_text, ce_mask_or_None, ce_weight) or None to skip."""
        if a.target == "gold":
            return gold_text(it), None, a.ce
        tr = get_trace(it)
        text, ok = tr["trace"], tr["score"] > 0
        if not text:
            return None
        if not ok:
            if a.wrong_trace == "drop":
                return None
            if a.wrong_trace == "gold":
                k = text.rfind("<answer>")
                text = (text[:k] if k >= 0 else text + "\n") + gold_text(it)
                ok = True
        n = len(text_ids(tok, text))
        return text, answer_span_mask(tok, text, n), (a.ce if ok else 0.0)

    # ---------------- loop
    history = {"steps": [], "evals": []}
    best = -1.0
    best_path = os.path.join(a.out, "interface_best.pt")
    t0 = time.time()
    skipped = 0
    slot_masks: dict = {}                          # D2: item key -> [C,N] gold-slot mask or None
    slot_stats = {"with_gold": 0, "no_gold": 0}    # D2: per-draw counts (items lacking a gold slot get no aux)
    dist = None                                    # D9: dense distillation (xenc/distill.py); None when both flags 0
    if a.attn_distill or a.hid_distill:
        from xenc.distill import Distiller
        if a.attn_distill and getattr(a, "read", "library") != "library":
            raise SystemExit("--attn_distill needs the Library read (LibraryInjector.attn_terms)")
        dist = Distiller(model, tok, hub, itf.inject_layers, a.attn_distill, a.hid_distill, hid_ratio=bool(a.hid_ratio),
                         hid_layer_w=[float(x) for x in a.hid_layer_w.split(",")] if a.hid_layer_w else None)
        log(f"D9 distill: attn {a.attn_distill} hid {a.hid_distill} (ratio {a.hid_ratio}); teacher attention windows "
            f"{dist.windows}")

    def run_eval(step):
        nonlocal best
        if not val_items:
            return
        eval_mode(model)
        r = eval_kv(rt, inj, val_items, a.budget, style)
        r["step"] = step
        history["evals"].append(r)
        tag = ""
        if r["score"] > best:
            best = r["score"]
            save_interface(itf, best_path, selector,
                           extra={"args": vars(a), "step": step, "val": r, "depths": depths, "center": a.center, "init_A": a.init_A, "harvest": a.harvest,
                                  "enc_short": enc_short, "chunk_tokens": chunk_tokens, "style": style})
            tag = " *best*"
        log(f"  [eval @ {step}] val kv score {r['score']:.4f} (n={r['n']}, gen {r['gen_tokens']:.1f} tok){tag}")

    def dump_history():
        history["skipped"] = skipped
        if a.slot_aux:
            history["slot_aux_items"] = {**slot_stats, "unique_no_gold": sum(v is None for v in slot_masks.values()),
                                         "unique_with_gold": sum(v is not None for v in slot_masks.values())}
        if dist is not None:
            history["distill"] = dict(dist.stats)
        history["best_val"] = best
        hp = os.path.join(a.out, "history.json")
        with open(hp + ".tmp", "w") as f:
            json.dump(history, f, indent=1)
        os.replace(hp + ".tmp", hp)

    try:
        for step in range(1, a.steps + 1):
            opt.zero_grad(set_to_none=True)
            acc = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "mag": 0.0, "aux": 0.0, "slot": 0.0, "n": 0}
            if dist is not None:
                acc.update(attn=0.0, hid=0.0, hid_raw=0.0)
            for _ in range(a.batch):
                it = pool[rng.randrange(len(pool))]
                did = _get(it, "doc_id")
                tg = make_target(it)
                if tg is None:
                    skipped += 1
                    continue
                target, ce_mask, ce_w = tg
                tids, m, _ = with_target(tok, hub, sysp, ctx_prompt(it, docs[did]), target, device=dev)
                if tids.shape[1] > a.max_teacher_tokens:
                    skipped += 1
                    continue
                lib = rt.library(did)
                if lib is None:
                    skipped += 1
                    continue
                gold = rt.gold_chunks(it)
                inj.unload()
                eval_mode(model)
                if dist is not None:   # D9: the same teacher forward also yields attention + residual targets
                    dist.prepare(_get(it, "key"), sysp, ctx_prompt(it, docs[did]), docs[did], lib, chunk_tokens,
                                 tids.shape[1] - m)
                    with dist.teacher(tids.shape[1], m):
                        t_log = teacher_logits(model, tids, m)
                else:
                    t_log = teacher_logits(model, tids, m)                      # [m+1, V], no grad
                sids, m2, _ = with_target(tok, hub, sysp, q_prompt(it), target, device=dev)
                assert m2 == m
                if dist is not None:
                    dist.baseline(sids, m)                                 # D9: h^0 (no operator loaded yet)
                inj.load_library(lib)
                inj.mag_terms.clear()
                inj.route_terms.clear()
                gmask = None
                if a.slot_aux:
                    k_ = _get(it, "key")
                    if k_ not in slot_masks:
                        slot_masks[k_] = item_gold_slot_mask(it, docs[did], tok, chunk_tokens, lib)
                    gmask = slot_masks[k_]
                    slot_stats["with_gold" if gmask is not None else "no_gold"] += 1
                    inj.attn_terms.clear()
                    inj.training_attn = gmask is not None
                    inj.attn_pos = -(m + 1)                                # last prompt position
                if dist is not None and a.attn_distill:                   # D9: keep route x slot attention (grad)
                    if not inj.training_attn:
                        inj.attn_terms.clear()
                    inj.training_attn = True
                    inj.attn_pos = -(m + 1)
                if a.read == "topk":
                    inj.sel_pos = -(m + 1)                                  # D8: select at the last prompt position
                train_mode(model)
                with (dist.student() if dist is not None else contextlib.nullcontext()):
                    s_log = student_logits(model, sids, m)                      # [m+1, V]
                if a.read == "topk":
                    inj.sel_pos = -1
                kl = kl_teacher_student(t_log[:-1], s_log[:-1])
                tgt = sids[0, -m:]
                ce = answer_ce(s_log, tgt, ce_mask)
                mag = magnitude_penalty(inj).to(kl.device)
                aux = route_aux_loss(inj, gold) if a.route_aux else None
                loss = a.kl * kl + ce_w * ce + a.mag * mag
                if aux is not None:
                    loss = loss + a.route_aux * aux.to(loss.device)
                slot = slot_aux_loss(inj, gmask) if gmask is not None else None
                if slot is not None:
                    loss = loss + a.slot_aux * slot.to(loss.device)
                dterms = dist.losses(inj) if dist is not None else {}
                if dterms.get("attn") is not None:
                    loss = loss + a.attn_distill * dterms["attn"].to(loss.device)
                if dterms.get("hid") is not None:
                    loss = loss + a.hid_distill * dterms["hid"].to(loss.device)
                (loss / a.batch).backward()
                if dist is not None:
                    dist.clear()
                inj.training_attn = False
                inj.attn_terms.clear()
                inj.unload()
                inj.mag_terms.clear()
                inj.route_terms.clear()
                del t_log, s_log
                acc["loss"] += float(loss.detach())
                acc["kl"] += float(kl.detach())
                acc["ce"] += float(ce.detach())
                acc["mag"] += float(mag.detach())
                acc["aux"] += float(aux.detach()) if aux is not None else 0.0
                acc["slot"] += float(slot.detach()) if slot is not None else 0.0
                for k_ in ("attn", "hid"):
                    if dterms.get(k_) is not None:
                        acc[k_] += float(dterms[k_].detach())
                if dterms.get("hid_raw") is not None:
                    acc["hid_raw"] += float(dterms["hid_raw"])
                acc["n"] += 1
            eval_mode(model)
            if acc["n"] > 0:   # (a step whose items were all skipped still reaches the eval below)
                gn = torch.nn.utils.clip_grad_norm_(train_params, 1.0)
                opt.step()
                rec = {k: (v / acc["n"] if k != "n" else v) for k, v in acc.items()}
                rec.update(step=step, grad_norm=float(gn), t=round(time.time() - t0, 2))
                history["steps"].append(rec)
                if not all(math.isfinite(rec[k]) for k in ("loss", "kl", "ce")):
                    log(f"ERROR: non-finite loss at step {step}: {rec}")
                    break
                if step % 10 == 0 or step <= 3 or a.tiny:
                    log(f"  [step {step}/{a.steps}] loss {rec['loss']:.4f} kl {rec['kl']:.4f} ce {rec['ce']:.4f} "
                        f"mag {rec['mag']:.4f} aux {rec['aux']:.4f} slot {rec['slot']:.4f} |g| {rec['grad_norm']:.3f} "
                        f"({(time.time() - t0) / step:.2f}s/step)")
                    if dist is not None:
                        log(f"    D9 attn {rec['attn']:.4f} hid {rec['hid']:.4f} hid_raw {rec['hid_raw']:.5f}")
            if step % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if step % a.eval_every == 0 or step == a.steps:
                run_eval(step)
                dump_history()
    except BaseException as e:   # KeyboardInterrupt / OOM / anything: keep the partial history
        history["error"] = f"{type(e).__name__}: {e}"
        log(f"ERROR: {history['error']} -- writing partial history.json")
        raise
    finally:
        dump_history()

    if not os.path.exists(best_path):   # no val items: best == last
        save_interface(itf, best_path, selector, extra={"args": vars(a), "step": a.steps, "val": None, "center": a.center, "init_A": a.init_A, "harvest": a.harvest,
                                                         "depths": depths, "enc_short": enc_short,
                                                         "chunk_tokens": chunk_tokens, "style": style})
    save_interface(itf, os.path.join(a.out, "interface_last.pt"), selector,
                   extra={"args": vars(a), "step": a.steps, "depths": depths, "enc_short": enc_short, "center": a.center, "init_A": a.init_A, "harvest": a.harvest,
                          "chunk_tokens": chunk_tokens, "style": style})
    dump_history()
    if torch.cuda.is_available():
        log(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if a.slot_aux:
        log(f"slot_aux items: {history.get('slot_aux_items')}")
    log(f"done: {len(history['steps'])} steps, skipped {skipped}, best val {best:.4f} -> {best_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
