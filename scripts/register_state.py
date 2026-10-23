"""D10: register a cross-model DeltaNet-STATE read interface (A_ℓ, B_ℓ, g_ℓ) on a frozen reader.

The document KV-Skill = the encoder's Gated-DeltaNet recurrent states after one frozen prefill of the document
(xenc/state_skill.py, cached under <state_root>/<encoder_short>/<doc_id>.pt). The reader reads them through a
branch inside each of its own DeltaNet layers (xenc/state_interface.py). Only A, B, g train.

Objective = register_doc.py's: teacher = reader with the document in context (ctx_prompt), student = reader on
q_prompt with the item's states loaded; loss = kl·KL(p_T‖p_S) + ce·CE(gold) + mag·mean(‖y‖/‖core‖).
Checkpoint selection on val: --select gap (kv − wrong_doc, default) or kv.

    python scripts/register_state.py --reader Qwen/Qwen3.5-9B --encoder Qwen/Qwen3.5-2B \
        --split_dir data/quality_split --out outputs/D10_q_SL --steps 300 --max_docs 40 [--init maps.pt]
    python scripts/register_state.py --tiny --steps 3 --out /tmp/x

Helpers here (StateRuntime, greedy decode with a prefix cache, tiny fixtures) are imported by eval_state_xm.py
and fit_state_maps.py.
"""

from __future__ import annotations

import argparse
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

TINY_HUB = "Qwen/Qwen3.5-2B"     # tiny mode: real tokenizer/chat template, random tiny Qwen3.5 models


# ============================================================================ tiny fixtures
def tiny_state_setup(seed: int = 0):
    """Tiny random Qwen3.5 reader (8 layers) + a different tiny encoder (6 layers, 4 value heads), the real
    Qwen3.5 tokenizer (offline cache), the tiny docs/items of register_doc.tiny_setup."""
    import tempfile
    from transformers import AutoTokenizer
    sys.path.insert(0, str(_ROOT / "tests"))
    from _tiny import tiny_qwen35
    from register_doc import tiny_setup
    from test_state_interface import _tiny_encoder
    tok = AutoTokenizer.from_pretrained(TINY_HUB)
    tz = tiny_setup(tempfile.mkdtemp(prefix="d10tiny_"), encoder_same=True)
    reader = tiny_qwen35(vocab=len(tok), seed=seed)
    enc = _tiny_encoder(seed + 1)
    enc.resize_token_embeddings(len(tok))
    enc.eval().requires_grad_(False)
    return {"reader": reader, "encoder": enc, "tok": tok, "hub": TINY_HUB, "docs": tz["docs"],
            "train": tz["train"], "val": tz["val"], "test": tz["test"]}


# ============================================================================ runtime
class StateRuntime:
    """Per-document encoder states (disk cache under <state_root>/<enc_short>/, LRU in memory). Missing
    states are harvested lazily when an encoder model is attached (L/L: the reader itself, with the branch
    disabled while it encodes)."""

    def __init__(self, reader, tok, hub, docs: dict, enc_short: str, state_root: str, encoder=None,
                 enc_tok=None, enc_hub: str | None = None, system: str = SYSTEM_ANSWER, max_doc_tokens: int = 0,
                 lru: int = 64, persist: bool = True):
        self.model, self.tok, self.hub, self.docs = reader, tok, hub, docs
        self.enc_short, self.state_root = enc_short, state_root
        self.encoder, self.enc_tok, self.enc_hub = encoder, enc_tok or tok, enc_hub or hub
        self.system, self.max_doc_tokens = system, max_doc_tokens
        self.persist = persist
        self.reader_patch = None            # StateReader; disabled while an L/L reader encodes
        self._lru: OrderedDict = OrderedDict()
        self._cap = lru
        self._warned: set = set()
        self.kv = False                     # default harvest variant: True = with full-attention K/V

    def path(self, doc_id: str, kv: bool = False) -> str:
        from xenc.state_skill import state_path
        return state_path(self.state_root, self.enc_short, doc_id, kv=kv)

    def has(self, doc_id: str, kv: bool | None = None) -> bool:
        kv = self.kv if kv is None else kv
        if doc_id not in self.docs:
            return False
        return self.encoder is not None or os.path.exists(self.path(doc_id, kv)) or (
            not kv and os.path.exists(self.path(doc_id, True)))

    def states(self, doc_id: str, kv: bool | None = None) -> dict | None:
        kv = self.kv if kv is None else kv
        from xenc.state_skill import harvest_states, load_states, save_states
        key = (doc_id, kv)
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]
        if doc_id not in self.docs:
            return None
        d = load_states(self.path(doc_id, kv))
        if d is None and not kv:
            d = load_states(self.path(doc_id, True))
        if d is None:
            if self.encoder is None:
                if doc_id not in self._warned:
                    log(f"WARN: no states for {doc_id} under {self.state_root}/{self.enc_short}; skipping")
                    self._warned.add(doc_id)
                return None
            R = self.reader_patch
            was = R.enabled if R is not None else None
            if R is not None:
                R.enabled = False
            try:
                d = harvest_states(self.encoder, self.enc_tok, self.docs[doc_id], hub=self.enc_hub,
                                   system=self.system, keep_kv=kv, max_tokens=self.max_doc_tokens)
            finally:
                if R is not None:
                    R.enabled = was
            d["doc_id"] = doc_id
            if self.persist:
                save_states(d, self.path(doc_id, kv))
        self._lru[key] = d
        while len(self._lru) > self._cap:
            self._lru.popitem(last=False)
        return d


# ============================================================================ decoding
def _stop_ids(tok) -> set:
    s = set()
    for t in ("<|im_end|>", "<|endoftext|>"):
        try:
            i = tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0:
                s.add(i)
        except Exception:
            pass
    if tok.eos_token_id is not None:
        s.add(int(tok.eos_token_id))
    return s


@torch.no_grad()
def greedy(model, tok, hub, ids, budget: int, cache=None, stop: str = "</answer>"):
    """Greedy decode -> (text, n_generated). Without a cache: xenc.backbone.generate (HF generate). With a
    prefix cache (DynamicCache holding the document): manual loop continuing from it."""
    from xenc.backbone import decode, generate, input_device
    if cache is None:
        return generate(model, tok, hub, ids, budget, return_n=True)
    dev = input_device(model)
    x = torch.tensor([list(ids)], dtype=torch.long, device=dev)
    stops = _stop_ids(tok)
    out = model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
    new = []
    for _ in range(budget):
        nxt = int(out.logits[0, -1].argmax())
        new.append(nxt)
        if nxt in stops or stop in tok.decode(new[-12:], skip_special_tokens=True):
            break
        out = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=out.past_key_values,
                    use_cache=True, logits_to_keep=1)
    txt = decode(tok, hub, new)
    if stop and stop in txt:
        txt = txt.split(stop, 1)[0] + stop
    return txt, len(new)


def kept_logits(model, ids: torch.Tensor, keep: int, cache=None) -> torch.Tensor:
    """[keep, V] last logits; with a prefix cache when given (use_cache then required by HF)."""
    from xenc.backbone import input_device
    out = model(input_ids=ids.to(input_device(model)), logits_to_keep=keep, use_cache=cache is not None,
                past_key_values=cache)
    return out.logits[0, -keep:]


# ============================================================================ interface construction
def enc_dims(S, rt, doc_id) -> tuple[dict, dict]:
    """(DeltaNet state dims, attention dims) of the encoder: from the model if loaded, else from a harvest."""
    from xenc.state_skill import state_dims
    from xenc.xattn_read import attn_dims
    if S.get("encoder") is not None:
        return state_dims(S["encoder"]), attn_dims(S["encoder"])
    st = rt.states(doc_id)
    return dict(st["dims"]), dict(st.get("adims") or {})


def build_reader(a, reader, sd: dict, ad: dict, dev, record_mag: bool = True):
    """D10Read: loaded from --init (a D10 interface file) or built fresh (+ ridge maps from --init)."""
    from xenc.d10_read import FORMAT, build_d10, load_d10
    if a.init and torch.load(a.init, map_location="cpu", weights_only=False).get("format") == FORMAT:
        R, _ = load_d10(a.init, reader, dev, record_mag)
        log(f"D10 interface loaded from {a.init} (branches {R.branches})")
        return R
    br = {b.strip() for b in a.branches.split(",") if b.strip()}
    layers = [int(x) for x in a.layers.split(",")] if a.layers else None
    R = build_d10(reader, br, sd, ad,
                  state_kw={"a_mode": a.a_mode, "b_mode": a.b_mode, "gate_init": a.gate_init,
                            "decay": bool(a.decay), "layers": layers},
                  attn_kw={"a_mode": a.attn_a_mode, "b_mode": a.attn_b_mode, "bias_init": a.attn_bias_init,
                           "layers": [int(x) for x in a.attn_layers.split(",")] if a.attn_layers else None},
                  maps=a.init, device=dev, record_mag=record_mag)
    if a.init:
        log(f"maps initialised from ridge fit {a.init}")
    return R


def add_branch_args(ap):
    ap.add_argument("--branches", default="state", help="state, attn or attn,state")
    ap.add_argument("--a_mode", choices=["full", "head", "none"], default="full", help="state branch A")
    ap.add_argument("--b_mode", choices=["full", "head", "none"], default="full", help="state branch B")
    ap.add_argument("--gate_init", type=float, default=0.1, help="state branch gate g")
    ap.add_argument("--decay", type=int, default=1, help="state branch: reader cumulative decay on the read")
    ap.add_argument("--layers", default="", help="state branch: reader linear layers (default all)")
    ap.add_argument("--attn_a_mode", choices=["head", "mix", "none"], default="head")
    ap.add_argument("--attn_b_mode", choices=["head", "none"], default="head")
    ap.add_argument("--attn_bias_init", type=float, default=0.0, help="attn branch log-mass bias b")
    ap.add_argument("--attn_layers", default="", help="attn branch: reader full-attention layers (default all)")
    ap.add_argument("--tail", type=int, default=0,
                    help="the reader natively reads the last N document tokens (prompt = doc tail + question suffix); "
                         "the attn branch then covers only the first n_doc − N document positions")


# ============================================================================ eval helper
@torch.no_grad()
def eval_gap(rt: StateRuntime, R, items, budget: int, doc_order: list[str]) -> dict:
    """Val: kv (own states) and wrong_doc (states of doc_order[i+7]) greedy scores."""
    from eval_doc import wrong_doc_id
    from xenc.backbone import build_chat
    from xenc.doc_tasks import q_prompt, score
    tot = {"kv": 0.0, "wrong_doc": 0.0}
    ntok, n = 0, 0
    for it in items:
        did = _get(it, "doc_id")
        st, w = rt.states(did), wrong_doc_id(did, doc_order)
        sw = rt.states(w) if w else None
        if st is None or sw is None:
            continue
        tail = getattr(getattr(R, "ar", None), "tail", 0) or 0
        for c, s, dd in (("kv", st, did), ("wrong_doc", sw, w)):
            ids = student_prompt(rt, it, dd, tail)
            R.load(s)
            try:
                g, k = greedy(rt.model, rt.tok, rt.hub, ids, budget)
            finally:
                R.unload()
            tot[c] += score(it, g)
            if c == "kv":
                ntok += k
        n += 1
    r = {c: v / max(n, 1) for c, v in tot.items()}
    r.update(gap=r["kv"] - r["wrong_doc"], gen_tokens=ntok / max(n, 1), n=n)
    return r


@torch.no_grad()
def js_item(a, R, rt, model, tok, hub, docs, it, did, sysp) -> dict:
    """Teacher pass (text prompt) keeps attention over the document columns for the shared prompt suffix;
    student pass (q_prompt + own encoder K/V through the branch) keeps the branch's rows."""
    from xenc.backbone import build_chat
    del a
    from xenc.doc_tasks import ctx_prompt, q_prompt
    from xenc.xattn_read import attn_js
    st = rt.states(did)
    t_ids = build_chat(tok, hub, sysp, ctx_prompt(it, docs[did]))
    s_ids = student_prompt(rt, it, did, getattr(R.ar, "tail", 0))
    K = 0
    while K < min(len(t_ids), len(s_ids), 256) and t_ids[-1 - K] == s_ids[-1 - K]:
        K += 1
    K = max(K, 1)
    ar = R.ar
    R.unload()
    ar.tcap, ar.n_doc_t, ar.teacher_probs = K, int(st["n_doc"]) - int(getattr(ar, "tail", 0) or 0), {}
    kept_logits(model, torch.tensor([t_ids]), 1)
    ar.tcap = 0
    R.load(st)
    ar.keep_rows, ar.doc_rows = K, []
    kept_logits(model, torch.tensor([s_ids]), 1)
    out = attn_js(ar)
    R.unload()
    ar.keep_rows, ar.doc_rows, ar.teacher_probs = 0, [], {}
    for d in out.values():
        d["rows"] = K
    return {str(k): v for k, v in out.items()}


_PREFIX_IDS: dict = {}


def prefix_ids(rt, did: str) -> list[int]:
    """Token ids of the harvested document prefix (system + user header + document), cached per doc."""
    if did not in _PREFIX_IDS:
        from xenc.state_skill import doc_prefix_text
        _PREFIX_IDS[did] = [int(x) for x in rt.tok(doc_prefix_text(rt.tok, rt.hub, rt.docs[did], rt.system),
                                                   add_special_tokens=False)["input_ids"]]
    return _PREFIX_IDS[did]


def student_prompt(rt, it, did: str, tail: int) -> list[int]:
    """tail = 0: chat(q_prompt) (the branch carries the whole document). tail > 0: the last `tail` tokens of the
    document prefix, read natively by the reader, followed by the `text` prompt's question suffix — i.e. the text
    prompt with all but the last `tail` document tokens moved into the external branch."""
    from xenc.backbone import build_chat
    from xenc.doc_tasks import q_prompt
    from xenc.state_skill import question_suffix_ids
    if not tail:
        return build_chat(rt.tok, rt.hub, rt.system, q_prompt(it))
    return prefix_ids(rt, did)[-tail:] + question_suffix_ids(rt.tok, rt.hub, it, rt.system)


def set_tail(R, tail: int):
    if R is not None and getattr(R, "ar", None) is not None:
        R.ar.tail = int(tail)


def suffix_rows(t_ids, s_ids, cap: int = 512) -> int:
    """Number of trailing tokens shared by the teacher (text) and student (q_prompt) sequences."""
    K = 0
    while K < min(len(t_ids), len(s_ids), cap) and t_ids[-1 - K] == s_ids[-1 - K]:
        K += 1
    return max(K, 1)


@torch.no_grad()
def eval_js(rt, R, items, model, tok, hub, docs) -> dict:
    """Mean addressing JS over val items (held-out docs), per layer and overall (lower = closer to native)."""
    acc, n = {}, 0
    for it in items:
        did = _get(it, "doc_id")
        if rt.states(did) is None:
            continue
        r = js_item(None, R, rt, model, tok, hub, docs, it, did, rt.system)
        for li, d in r.items():
            acc.setdefault(li, []).append(d["js"])
        n += 1
    per = {li: sum(v) / len(v) for li, v in acc.items()}
    return {"js": sum(per.values()) / max(len(per), 1), "js_layers": per, "n_js": n}


# ============================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B", help="hub id or 'same' (L/L)")
    ap.add_argument("--split_dir", default=None)
    ap.add_argument("--state_root", default=None, help="default $XENC_CACHE/states")
    ap.add_argument("--out", default="outputs/D10_reg")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_g", type=float, default=1e-2, help="lr of the per-layer gate scalars")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--ce", type=float, default=0.5)
    ap.add_argument("--mag", type=float, default=0.0)
    ap.add_argument("--attn_distill", type=float, default=0.0,
                    help="weight of the dense attention distillation (teacher's attention over document positions "
                         "at the attn-branch layers, answer rows) — needs --branches attn")
    ap.add_argument("--distill_mass", type=float, default=1.0, help="weight of the doc-mass MSE inside attn_distill")
    ap.add_argument("--distill_rows", choices=["answer", "suffix"], default="answer",
                    help="answer: the m+1 answer-predicting rows; suffix: every row of the shared question+answer suffix")
    ap.add_argument("--n_val_gen", type=int, default=0, help="--select js: also run kv/wrong_doc generation on this many val items")
    add_branch_args(ap)
    ap.add_argument("--init", default=None, help="ridge maps (fit_state_maps.py) or a D10 interface .pt")
    ap.add_argument("--train_maps", type=int, default=1, help="0: only the gates g train")
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--test_fold", type=int, default=None)
    ap.add_argument("--val_fold", type=int, default=None,
                    help="items with meta.fold == K are removed from train; val keeps only fold-less items and fold K")
    ap.add_argument("--split_dirs", default="", help="comma list of split dirs pooled for train/val (multi-domain)")
    ap.add_argument("--n_val", type=int, default=60)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--select", choices=["gap", "kv", "js"], default="gap")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--max_teacher_tokens", type=int, default=16000)
    ap.add_argument("--max_doc_tokens", type=int, default=0)
    ap.add_argument("--harvest", type=int, default=1, help="lazily harvest missing states (loads the encoder)")
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_mem", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    if not a.tiny and not (a.split_dir or a.split_dirs):
        ap.error("--split_dir (or --split_dirs) is required unless --tiny")
    return a


def setup(a, need_train: bool = True):
    """-> dict(model, tok, hub, docs, train, val, test?, rt, enc_short, enc_dims, encoder)."""
    from xenc.backbone import load_encoder, load_reader, short_name
    from xenc.state_skill import STATE_ROOT, state_dims
    state_root = a.state_root or STATE_ROOT
    if a.tiny:
        torch.set_num_threads(2)
        tz = tiny_state_setup(a.seed)
        same = a.encoder == "same"
        enc = tz["reader"] if same else tz["encoder"]
        import tempfile
        rt = StateRuntime(tz["reader"], tz["tok"], tz["hub"], tz["docs"], "tiny-same" if same else "tiny-enc",
                          tempfile.mkdtemp(prefix="d10st_"), encoder=enc)
        return {"model": tz["reader"], "tok": tz["tok"], "hub": tz["hub"], "docs": tz["docs"], "train": tz["train"],
                "val": tz["val"], "test": tz["test"], "rt": rt, "enc_dims": state_dims(enc), "encoder": enc,
                "enc_short": rt.enc_short}
    from xenc.doc_tasks import load_docs
    sdirs = [x for x in (getattr(a, "split_dirs", "") or "").split(",") if x] or [a.split_dir]
    docs = {}
    for sd_ in sdirs:
        docs.update(load_docs(sd_))
    model, tok = load_reader(a.reader, device_map=a.device, max_mem=a.max_mem, grad_ckpt=bool(getattr(a, "grad_ckpt", 0)))
    same = a.encoder == "same"
    enc_short = short_name(a.reader if same else a.encoder)
    enc, etok = (model, tok) if same else (None, None)
    if not same and getattr(a, "harvest", 1):
        enc, etok = load_encoder(a.encoder, device_map=a.device)
    rt = StateRuntime(model, tok, a.reader, docs, enc_short, state_root, encoder=enc, enc_tok=etok,
                      enc_hub=a.reader if same else a.encoder, max_doc_tokens=getattr(a, "max_doc_tokens", 0))
    enc_dims = state_dims(enc) if enc is not None else None
    return {"model": model, "tok": tok, "hub": a.reader, "docs": docs,
            "train": [i for sd_ in sdirs for i in load_split_items(sd_, "train")] if need_train else [],
            "val": [i for sd_ in sdirs for i in load_split_items(sd_, "val")], "rt": rt, "enc_dims": enc_dims, "encoder": enc,
            "enc_short": enc_short}


def main(argv=None) -> int:
    a = parse_args(argv)
    from xenc.backbone import eval_mode, input_device, train_mode, with_target
    from xenc.doc_tasks import ctx_prompt, gold_text, q_prompt
    from xenc.losses import answer_ce, kl_teacher_student

    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    S = setup(a)
    model, tok, hub, docs, rt = S["model"], S["tok"], S["hub"], S["docs"], S["rt"]
    train, val = S["train"], S["val"]
    if a.test_fold is not None:
        nf = lambda i: (_get(i, "meta") or {}).get("fold")   # noqa: E731
        train = [i for i in train if nf(i) != a.test_fold]
        val = [i for i in val if nf(i) != a.test_fold]
        log(f"test_fold={a.test_fold}: train {len(train)} items, val {len(val)} items")
    if a.val_fold is not None:
        nf = lambda i: (_get(i, "meta") or {}).get("fold")   # noqa: E731
        train = [i for i in train if nf(i) != a.val_fold]
        val = [i for i in val if nf(i) is None or nf(i) == a.val_fold]
        log(f"val_fold={a.val_fold}: train {len(train)} items, val {len(val)} items")
    doc_order = sorted({_get(i, "doc_id") for i in train})
    if a.max_docs:   # per task: the first N doc ids of each task
        per = {}
        for d in doc_order:
            per.setdefault(d.split("_")[0], []).append(d)
        doc_order = sorted(x for v in per.values() for x in v[: a.max_docs])
    keep = set(doc_order)
    pool = [i for i in train if _get(i, "doc_id") in keep and rt.has(_get(i, "doc_id"))]
    val_docs = sorted({_get(i, "doc_id") for i in val if rt.has(_get(i, "doc_id"))})
    val_items = [i for i in val if _get(i, "doc_id") in set(val_docs)]
    by_task: dict = {}
    for i in val_items:                       # stratified by task (multi-domain val)
        by_task.setdefault(_get(i, "task") or "", []).append(i)
    rv = random.Random(a.seed + 1)
    per = max(1, a.n_val // max(len(by_task), 1))
    val_items = [x for t in sorted(by_task) for x in rv.sample(by_task[t], min(per, len(by_task[t])))]
    if not pool:
        log("ERROR: empty train pool (no states and no encoder)")
        return 2
    dev = input_device(model)
    rt.kv = "attn" in a.branches
    sd, ad = enc_dims(S, rt, _get(pool[0], "doc_id"))
    R = build_reader(a, model, sd, ad, dev)
    rt.reader_patch = R
    set_tail(R, a.tail)
    rt.kv = R.needs_kv
    groups = R.param_groups(a.lr, a.lr_g, bool(a.train_maps))
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    params = [p for g in groups for p in g["params"]]
    lmaps = {"state": R.state_itf.layer_map if R.state_itf is not None else None,
             "attn": R.attn_itf.layer_map if R.attn_itf is not None else None}
    log(f"reader {hub} | encoder {S['enc_short']} | branches {R.branches} | layer maps {lmaps} | trainable "
        f"{sum(p.numel() for p in params)} | pool {len(pool)} items / {len(keep)} docs | val {len(val_items)} "
        f"items / {len(val_docs)} docs")
    cfg = {**vars(a), "layer_map": lmaps, "enc_short": S["enc_short"], "state_root": rt.state_root}
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    history = {"steps": [], "evals": []}
    best = -1e9
    best_path = os.path.join(a.out, "interface_best.pt")
    t0 = time.time()
    skipped = 0

    def run_eval(step):
        nonlocal best
        if not val_items:
            return
        eval_mode(model)
        if a.select == "js":
            r = eval_js(rt, R, val_items, model, tok, hub, docs)
            if a.n_val_gen:
                r.update(eval_gap(rt, R, val_items[: a.n_val_gen], a.budget, val_docs))
            r.setdefault("kv", float("nan")); r.setdefault("wrong_doc", float("nan")); r.setdefault("gap", float("nan"))
            r.setdefault("n", r["n_js"]); r.setdefault("gen_tokens", float("nan"))
        else:
            r = eval_gap(rt, R, val_items, a.budget, val_docs)
        r["step"] = step
        r.update(R.gates())
        history["evals"].append(r)
        crit = {"gap": r["gap"], "kv": r["kv"], "js": -r.get("js", 0.0)}[a.select]
        tag = ""
        if crit > best:
            best = crit
            R.save(best_path, extra={"args": cfg, "step": step, "val": r})
            tag = " *best*"
        if "js" in r:
            log(f"  [eval @ {step}] val JS {r['js']:.4f} " + " ".join(f"L{k}:{v:.3f}" for k, v in r["js_layers"].items()))
        log(f"  [eval @ {step}] val kv {r['kv']:.4f} wrong {r['wrong_doc']:.4f} gap {r['gap']:+.4f} "
            f"(n={r['n']}, gen {r['gen_tokens']:.1f}){tag}")

    def dump():
        history["skipped"], history["best"] = skipped, best
        with open(os.path.join(a.out, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

    run_eval(0)
    try:
        for step in range(1, a.steps + 1):
            opt.zero_grad(set_to_none=True)
            acc = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "mag": 0.0, "dist": 0.0, "n": 0}
            for _ in range(a.batch):
                it = pool[rng.randrange(len(pool))]
                did = _get(it, "doc_id")
                st = rt.states(did)
                if st is None:
                    skipped += 1
                    continue
                target = gold_text(it)
                tids, m, _ = with_target(tok, hub, rt.system, ctx_prompt(it, docs[did]), target, device=dev)
                if tids.shape[1] > a.max_teacher_tokens:
                    skipped += 1
                    continue
                R.unload()
                eval_mode(model)
                dist_on = bool(a.attn_distill) and R.ar is not None
                rows = m + 1
                if dist_on and a.distill_rows == "suffix":
                    if a.tail:
                        from xenc.backbone import text_ids as _tids
                        _s = student_prompt(rt, it, did, a.tail) + _tids(tok, target)
                    else:
                        _s = with_target(tok, hub, rt.system, q_prompt(it), target)[0][0].tolist()
                    rows = suffix_rows(tids[0].tolist(), _s)
                if dist_on:
                    R.ar.tcap, R.ar.n_doc_t, R.ar.teacher_probs = rows, int(st["n_doc"]) - a.tail, {}
                with torch.no_grad():
                    t_log = kept_logits(model, tids, m + 1).detach()
                if dist_on:
                    R.ar.tcap = 0
                    R.ar.keep_rows, R.ar.doc_rows = rows, []
                if a.tail:
                    from xenc.backbone import text_ids as _tids
                    _t = _tids(tok, target)
                    sids = torch.tensor([student_prompt(rt, it, did, a.tail) + _t], device=dev)
                else:
                    sids, m2, _ = with_target(tok, hub, rt.system, q_prompt(it), target, device=dev)
                R.load(st)
                R.clear_mag()
                train_mode(model)
                with torch.enable_grad():
                    s_log = kept_logits(model, sids, m + 1)
                    kl = kl_teacher_student(t_log[:-1], s_log[:-1])
                    ce = answer_ce(s_log, sids[0, -m:])
                    mag = torch.stack(R.mag_terms).mean() if R.mag_terms else torch.zeros((), device=kl.device)
                    loss = a.kl * kl + a.ce * ce + a.mag * mag.to(kl.device)
                    dl = None
                    if dist_on:
                        from xenc.xattn_read import attn_distill_loss
                        dl = attn_distill_loss(R.ar, a.distill_mass)
                        if dl is not None:
                            loss = loss + a.attn_distill * dl.to(loss.device)
                    (loss / a.batch).backward()
                R.unload()
                R.clear_mag()
                if dist_on:
                    R.ar.keep_rows, R.ar.doc_rows, R.ar.teacher_probs = 0, [], {}
                    acc["dist"] += float(dl.detach()) if dl is not None else 0.0
                eval_mode(model)
                for k, v in (("loss", loss), ("kl", kl), ("ce", ce), ("mag", mag)):
                    acc[k] += float(v.detach())
                acc["n"] += 1
            if acc["n"]:
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                rec = {k: (v / acc["n"] if k != "n" else v) for k, v in acc.items()}
                rec.update(step=step, grad_norm=float(gn), t=round(time.time() - t0, 1))
                history["steps"].append(rec)
                if not all(math.isfinite(rec[k]) for k in ("loss", "kl", "ce")):
                    log(f"ERROR: non-finite loss at step {step}: {rec}")
                    break
                if step % 10 == 0 or step <= 3 or a.tiny:
                    log(f"  [step {step}/{a.steps}] loss {rec['loss']:.4f} kl {rec['kl']:.4f} ce {rec['ce']:.4f} "
                        f"mag {rec['mag']:.4f} dist {rec['dist']:.4f} |g| {rec['grad_norm']:.3f} gates {R.gates()} "
                        f"({(time.time() - t0) / step:.2f}s/step)")
            if step % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if step % a.eval_every == 0 or step == a.steps:
                run_eval(step)
                dump()
    finally:
        dump()
    if not os.path.exists(best_path):
        R.save(best_path, extra={"args": cfg, "step": a.steps, "val": None})
    R.save(os.path.join(a.out, "interface_last.pt"), extra={"args": cfg, "step": a.steps})
    if torch.cuda.is_available():
        log(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    log(f"done: {len(history['steps'])} steps, skipped {skipped}, best val ({a.select}) {best:.4f} -> {best_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
