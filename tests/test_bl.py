"""BL baselines (agent BASE): xenc/textbudget.py (S2), xenc/lingua.py (S4), xenc/kvcomp.py (S5). CPU only.

S2  budget enforcement (content tokens <= K, re-tokenized context close to K), N <= K passes the document through,
    retrieval query never contains the gold answer / label, deterministic selection in document order,
    BM25 ranks the planted chunk first, parse_cond grammar.
S4  LLMLingua-2 output <= K reader tokens (real microsoft/llmlingua-2 model, CPU; skipped if absent);
    LongLLMLingua hybrid-safe get_ppl == an uncached full forward for the llmlingua call pattern; snapshot path ==
    recompute path end to end (identical compressed text) on a tiny REAL Qwen3_5ForCausalLM; output <= K.
S5  prompt split: prefix-cache continuation == one-pass greedy over the full prompt (exact tokens);
    MaskHook with all positions kept == no hook; dropped (layer, head, position) entries have no effect, kept ones
    do; SnapKV / KVzip scores == brute force from the eager attention weights; budget accounting.
Run: python tests/test_bl.py   (or pytest)
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS))
sys.path.insert(0, str(_TESTS.parent))
sys.path.insert(0, str(_TESTS.parent / "scripts"))

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402

torch.set_num_threads(4)
HUB = "Qwen/Qwen3.5-9B"
_TOK = None


def qtok():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(HUB)
    return _TOK


def _doc(n_par=60, plant=None, at=37):
    pars = []
    for i in range(n_par):
        s = f"Paragraph {i}. The committee reviewed item {i * 7} and noted routine progress on the budget."
        if plant is not None and i == at:
            s = plant
        pars.append(s)
    return "\n".join(pars)


# =========================================================================================== S2
def test_textbudget_budget_and_passthrough():
    from xenc.textbudget import DocIndex, build_context
    tok = qtok()
    text = _doc(200)
    idx = DocIndex(tok, text, chunk=512)
    assert idx.N > 4096
    for m in ("trunc_head", "trunc_ht", "bm25"):
        for K in (1024, 2048):
            ctx, info = build_context(m, K, idx, "committee budget item 700")
            assert info["full"] == 0 and info["content_tokens"] <= K
            n = len(tok(ctx, add_special_tokens=False)["input_ids"])
            assert abs(n - info["content_tokens"]) <= 8 * max(1, K // 512) + 4, (m, K, n, info)
    ctx, info = build_context("bm25", 10 ** 6, idx, "x")
    assert info["full"] == 1 and ctx == text


def test_retrieval_query_and_ranking():
    from xenc.textbudget import DocIndex, query_text
    tok = qtok()
    plant = "The zebra-striped lighthouse keeper Ignatius Quarrell repaired the brass foghorn in 1887."
    text = _doc(200, plant=plant, at=150)
    idx = DocIndex(tok, text, chunk=512)
    it = {"question": "Who repaired the brass foghorn of the lighthouse?", "answers": ["SECRETGOLD"],
          "choices": [{"label": "A", "text": "Ignatius Quarrell the keeper"}, {"label": "B", "text": "a passing sailor of the fleet"}],
          "correct_label": "A", "meta": {"evidence": "SECRETEVIDENCE"}}
    q = query_text(it)
    assert "SECRETGOLD" not in q and "SECRETEVIDENCE" not in q
    assert "Ignatius" in q and "sailor" in q                     # every option, not only the gold one
    sel = idx.select("bm25", q, 2)
    gold_chunk = next(j for j, t in enumerate(idx.chunk_text) if "Quarrell" in t)
    assert gold_chunk in sel and sel == sorted(sel)
    assert idx.select("bm25", q, 2) == sel                       # deterministic


def test_parse_cond():
    from xenc.textbudget import parse_cond
    assert parse_cond("bm25@2048") == ("bm25", 2048, None)
    assert parse_cond("lingua2@r4") == ("lingua2", None, 0.25)
    assert parse_cond("trunc_ht@4096") == ("trunc_ht", 4096, None)


# =========================================================================================== S4
def test_lingua2_budget():
    from huggingface_hub import try_to_load_from_cache
    from xenc.lingua import L2_NAME, Lingua2
    if not isinstance(try_to_load_from_cache(L2_NAME, "config.json"), str):
        print("  (skip: llmlingua-2 model not cached)")
        return
    tok = qtok()
    l2 = Lingua2(tok, device="cpu")
    text = _doc(60)
    n0 = len(tok(text, add_special_tokens=False)["input_ids"])
    for K in (200, 400):
        out, info = l2.compress(text, target_token=K)
        n = len(tok(out, add_special_tokens=False)["input_ids"])
        assert n <= K and info["content_tokens"] == n and n > 0.5 * K, (K, n, info)
    out, info = l2.compress(text, rate=0.25)
    assert info["target"] == round(0.25 * n0) and info["content_tokens"] <= info["target"]
    out, info = l2.compress("short text.", target_token=100)
    assert info["full"] == 1 and out == "short text."


def test_prefix_length_seam():
    """llmlingua's get_prefix_length asserts a decode round trip that fails when the Qwen BPE merges the
    "\n\n" after the question with a context starting with a newline (job 27978260, LongHealth); the override
    counts prefix tokens by character offsets."""
    from xenc.lingua import _ensure_importable, make_hybrid_safe
    _ensure_importable()
    from llmlingua import PromptCompressor
    tok = qtok()
    orig = PromptCompressor.__new__(PromptCompressor)
    orig.tokenizer = tok
    H = make_hybrid_safe(PromptCompressor)
    h = H.__new__(H)
    h.tokenizer = tok
    prefix, text = "What dose?\n\n", "\nDoe was admitted to the hospital in 2015 with a long history of complaints."
    try:
        orig.get_prefix_length(prefix, text)
        raised = False
    except AssertionError:
        raised = True
    assert raised                                            # the upstream bug is reproduced
    n = h.get_prefix_length(prefix, text)
    full = tok(prefix + text, add_special_tokens=False)["input_ids"]
    assert prefix.startswith(tok.decode(full[:n])) and n >= len(tok("What dose?", add_special_tokens=False)["input_ids"])
    ok = "Doe was admitted to the hospital."                 # no merge at the seam: same answer as upstream
    assert h.get_prefix_length(prefix, ok) == orig.get_prefix_length(prefix, ok)


def _tiny_lm(vocab):
    m = tiny_qwen35(n_layers=4, d=32, n_heads=4, n_kv=2, head_dim=8, vocab=vocab, seed=1)
    return m


def test_hybrid_safe_get_ppl_equals_full_forward():
    from xenc.lingua import _fake_kv, make_hybrid_safe, _ensure_importable
    _ensure_importable()
    from llmlingua import PromptCompressor
    m = _tiny_lm(100)
    cls = make_hybrid_safe(PromptCompressor)
    pc = cls.__new__(cls)
    pc.model, pc.device, pc.max_position_embeddings = m, torch.device("cpu"), 1 << 30
    pc._snaps, pc.stats, pc.resume_after = [], {"fresh": 0, "hit": 0, "recompute": 0}, 5

    def ref(ids, past, end):
        with torch.no_grad():
            lg = m(input_ids=ids[:, :end]).logits[0].float()
        return torch.nn.functional.cross_entropy(lg[past:end - 1], ids[0, past + 1:end], reduction="none")

    ids = torch.randint(3, 100, (1, 40), generator=torch.Generator().manual_seed(0))
    am = torch.ones_like(ids)
    # llmlingua's pattern: call (past=0, end=15) -> resume at 10; compress inside [10,15) -> call (past=10, end=..)
    l1, kv = pc.get_ppl("", "token", ids, am, None, True, 15)
    assert torch.allclose(l1, ref(ids, 0, 15), atol=1e-5)
    kv = [[k[:, :, :10], v[:, :, :10]] for k, v in kv]
    ids2 = torch.cat([ids[:, :12], ids[:, 14:]], 1)          # two tokens removed inside the current window
    l2, kv2 = pc.get_ppl("", "token", ids2, torch.ones_like(ids2), kv, True, 20)
    assert pc.stats["hit"] == 1 and torch.allclose(l2, ref(ids2, 10, 20), atol=1e-5)
    # a past that matches no snapshot is recomputed from scratch, same numbers
    kv3 = _fake_kv(7, "cpu")
    l3 = pc.get_ppl("", "token", ids2, torch.ones_like(ids2), kv3, False, 18)
    assert pc.stats["recompute"] == 1 and torch.allclose(l3, ref(ids2, 7, 18), atol=1e-5)


def test_longlingua_snapshot_equals_recompute():
    from xenc.lingua import LongLingua
    tok = qtok()
    m = _tiny_lm(len(tok))
    text = _doc(40)
    q = "What did the committee note about item 77?"
    a = LongLingua(m, tok, tok, chunk=128, iterative_size=50)
    out_a, info_a = a.compress(text, q, target_token=150)
    b = LongLingua(m, tok, tok, chunk=128, iterative_size=50)
    b.pc.resume_after = 10 ** 9                               # never snapshot: every resumed call recomputes
    out_b, info_b = b.compress(text, q, target_token=150)
    assert out_a == out_b, (out_a[:200], out_b[:200])
    assert info_a["ppl_calls"]["hit"] > 0 and info_b["ppl_calls"]["hit"] == 0
    assert info_a["content_tokens"] <= 150


# =========================================================================================== S5
def _setup_s5(n_doc_par=6):
    from xenc.doc_tasks import ctx_prompt
    from xenc.kvcomp import split_prompt
    tok = qtok()
    m = tiny_qwen35(n_layers=4, d=32, n_heads=4, n_kv=2, head_dim=8, vocab=len(tok), seed=2)
    doc = _doc(n_doc_par)
    items = [{"question": "What did the committee review?"}, {"question": "Which item number appears in paragraph 3?"}]
    sys_ = "You are a careful question answering assistant. Provide your final answer inside <answer>...</answer> tags."
    out = []
    for it in items:
        user = ctx_prompt(it, doc)
        ids, ds, P = split_prompt(tok, HUB, sys_, user, doc, len("## Document\n"))
        out.append((ids, ds, P))
    return tok, m, doc, out


def _greedy_cached(m, cache, suffix, n=6):
    x = torch.tensor([suffix])
    toks = []
    with torch.no_grad():
        o = m(input_ids=x, past_key_values=cache, use_cache=True)
        for _ in range(n):
            t = int(o.logits[0, -1].argmax())
            toks.append(t)
            o = m(input_ids=torch.tensor([[t]]), past_key_values=o.past_key_values, use_cache=True)
    return toks


def _greedy_full(m, ids, n=6):
    cur = list(ids)
    toks = []
    with torch.no_grad():
        for _ in range(n):
            t = int(m(input_ids=torch.tensor([cur])).logits[0, -1].argmax())
            toks.append(t)
            cur.append(t)
    return toks


def test_s5_prefix_split_exact():
    from xenc.kvcomp import prefill
    tok, m, doc, sp = _setup_s5()
    (ids0, ds0, P0), (ids1, ds1, P1) = sp
    assert P0 == P1 and ds0 == ds1 and ids0[:P0] == ids1[:P1]          # question-agnostic prefix
    assert tok.decode(ids0[ds0:P0]).strip() in doc and P0 - ds0 > 0.9 * len(tok(doc)["input_ids"])
    cache, _ = prefill(m, ids0[:P0], window=4)
    for ids in (ids0, ids1):
        assert _greedy_cached(m, copy.deepcopy(cache), ids[P0:]) == _greedy_full(m, ids)


def test_s5_mask_hook_semantics():
    from xenc.kvcomp import MaskHook, full_layers, keep_masks, prefill
    tok, m, doc, sp = _setup_s5()
    ids, ds, P = sp[0]
    cache, _ = prefill(m, ids[:P], window=4)
    L = full_layers(m)
    n_kv = cache.layers[L[0]].keys.shape[1]

    def logits(c, hook=None):
        x = torch.tensor([ids[P:]])
        with torch.no_grad():
            if hook is None:
                return m(input_ids=x, past_key_values=c, use_cache=True).logits
            with hook:
                return m(input_ids=x, past_key_values=c, use_cache=True).logits

    ref = logits(copy.deepcopy(cache))
    allm, _ = keep_masks("streaming", 1.0, P, ds, n_kv, L)
    assert torch.allclose(logits(copy.deepcopy(cache), MaskHook(m, allm, P)), ref, atol=1e-5)
    masks, info = keep_masks("streaming", 0.25, P, ds, n_kv, L)
    li, h = L[0], 1
    drop = int((~masks[li][h]).nonzero()[0])
    keep = int(masks[li][h, ds:].nonzero()[0]) + ds
    base = logits(copy.deepcopy(cache), MaskHook(m, masks, P))
    for pos, should_change in ((drop, False), (keep, True)):
        c = copy.deepcopy(cache)
        c.layers[li].keys[0, h, pos] += 3.0
        c.layers[li].values[0, h, pos] += 3.0
        out = logits(c, MaskHook(m, masks, P))
        assert (not torch.allclose(out, base, atol=1e-6)) == should_change, (pos, should_change)


def _record_eager():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    rec, orig = {}, M.eager_attention_forward

    def wrapped(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        o, w = orig(module, query, key, value, attention_mask, scaling, dropout=dropout, **kw)
        rec[module.layer_idx] = w.detach()
        return o, w
    M.eager_attention_forward = wrapped
    return rec, (lambda: setattr(M, "eager_attention_forward", orig))


def test_s5_scores_equal_brute_force():
    from xenc.kvcomp import full_layers, gen_prompt_suffix, keep_masks, kvzip_scores, prefill, snapkv_scores
    tok, m, doc, sp = _setup_s5(n_doc_par=4)
    ids, ds, P = sp[0]
    L = full_layers(m)
    rec, undo = _record_eager()
    try:
        cache, win = prefill(m, ids[:P], window=4)
        W = {li: rec[li][0] for li in L}                       # [n_q,P,P] prefill weights
        snap = snapkv_scores(m, cache, win, P, window=4, kernel=3)
        for li in L:
            n_q = W[li].shape[0]
            s = W[li][:, P - 4:, :].float().sum(1)             # [n_q,P]
            s = torch.nn.functional.avg_pool1d(s.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
            g = n_q // snap[li].shape[0]
            assert torch.allclose(snap[li], s.reshape(-1, g, P).sum(1), atol=1e-4)
        c0 = copy.deepcopy(cache)
        kz = kvzip_scores(m, tok, HUB, cache, ids, ds, P, chunk=16)
        # cache restored to the post-prefill state
        for li in range(len(cache.layers)):
            for k, v in vars(cache.layers[li]).items():
                if torch.is_tensor(v):
                    assert torch.equal(v, getattr(c0.layers[li], k)), (li, k)
        pre = tok("\n\nRepeat the previous context exactly." + gen_prompt_suffix(tok, HUB), add_special_tokens=False)["input_ids"]
        doc_ids = ids[ds:P]
        bf = {li: None for li in L}
        for s in range(0, len(doc_ids), 16):
            c = copy.deepcopy(c0)
            with torch.no_grad():
                m(input_ids=torch.tensor([pre + doc_ids[s:s + 16]]), past_key_values=c, use_cache=True)
            for li in L:
                w = rec[li][0][:, :, :P].float()                 # [n_q, m, P]
                g = w.shape[0] // kz[li].shape[0]
                mx = w.amax(1).reshape(-1, g, P).amax(1)
                bf[li] = mx if bf[li] is None else torch.maximum(bf[li], mx)
        for li in L:
            assert torch.allclose(kz[li], bf[li], atol=1e-5)
    finally:
        undo()
    n_kv = snap[L[0]].shape[0]
    N = P - ds
    for meth, sc in (("snapkv", snap), ("kvzip", kz), ("streaming", None)):
        masks, info = keep_masks(meth, 0.25, P, ds, n_kv, L, sc, window=4)
        tot = sum(int(mm[:, ds:].sum()) for mm in masks.values())
        assert tot == info["keep_n"] * n_kv * len(L), (meth, tot, info)
        assert all(bool(mm[:, :ds].all()) for mm in masks.values())
        if meth != "kvzip":
            assert all(int(mm[h, ds:].sum()) == info["keep_n"] for mm in masks.values() for h in range(n_kv))
        if meth == "snapkv":
            assert all(bool(mm[:, P - 4:].all()) for mm in masks.values())
    assert info["keep_n"] == round(0.25 * N)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f()
        print(f"ok {f.__name__}")
    print(f"test_bl: {len(fns)} passed")
