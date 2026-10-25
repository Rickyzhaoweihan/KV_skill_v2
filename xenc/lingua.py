"""BL / S4: LLMLingua-2 (question-agnostic) and LongLLMLingua (question-aware, Qwen3.5-4B as the scoring LM) as
document compressors for the same reader (agent BASE, 2026-10-23).

Both use the OFFICIAL `llmlingua` package (0.2.2, installed with --no-deps). Two environment adaptations, both
behaviour-preserving for what we measure:

1. Token counting. llmlingua counts budgets with tiktoken's gpt-3.5 BPE (`self.oai_tokenizer`), which needs a network
   download. We set `oai_tokenizer` to the READER's Qwen3.5 tokenizer, so every target_token / rate is in the tokens the
   reader actually reads. `tiktoken` / `nltk` are only imported at llmlingua's module top; when they are not installed,
   placeholder modules are registered so the import succeeds (nltk.sent_tokenize is used only by the sentence-level
   filter, which is off; the placeholder raises if it is ever called; tiktoken is never called).

2. Hybrid models (LongLLMLingua only). llmlingua's iterative token-level compression reuses a KV cache that it SLICES
   (`k[..., :s, :]`), which is impossible for Qwen3.5's Gated-DeltaNet recurrent state. `HybridSafeCompressor.get_ppl`
   keeps llmlingua's exact contract (same per-token losses for the same positions) but never slices a cache: it
   returns a length-only stand-in "past_key_values" ([[zeros(1,1,L,1)]*2]) that llmlingua slices as before, and
   computes the losses from a real cache SNAPSHOT of the exact prefix input_ids[:, :L] (taken during the previous
   call at the position llmlingua will resume from; looked up by length + token equality) or, if no snapshot
   matches, by recomputing the prefix from scratch. Either way the logits are those of a forward over
   input_ids[:, :end] — what the original code computes on a pure-attention model. tests/test_bl.py checks this
   equivalence against a full uncached forward.

    l2 = Lingua2(reader_tok, device="cuda")            # microsoft/llmlingua-2-xlm-roberta-large-meetingbank
    txt, info = l2.compress(doc_text, target_token=2048)          # or rate=0.25
    ll = LongLingua(model_4b, tok_4b, reader_tok)
    txt, info = ll.compress(doc_text, question, target_token=2048)
"""

from __future__ import annotations

import copy
import sys
import types

import torch

L2_NAME = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
L2_FORCE = ["\n", "?"]      # llmlingua-2 README example; adding "." / "," makes forced tokens overrun the budget


class QwenCounter:
    """Stand-in for llmlingua's tiktoken encoder: encode() -> the reader tokenizer's ids."""

    def __init__(self, tok):
        self.tok = tok

    def encode(self, text, **kw):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids, **kw):
        return self.tok.decode(ids)


def _ensure_importable():
    for name in ("tiktoken", "nltk"):
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            m = types.ModuleType(name)
            if name == "tiktoken":
                def _no(*a, **k):
                    raise RuntimeError("tiktoken placeholder: token counting is done with the reader tokenizer")
                m.encoding_for_model = _no
                m.get_encoding = _no
            else:
                def _no_sent(*a, **k):
                    raise RuntimeError("nltk placeholder: the sentence-level filter is not used")
                m.sent_tokenize = _no_sent
            m.__bl_placeholder__ = True
            sys.modules[name] = m


def _bare_compressor(cls, oai_tok):
    """PromptCompressor without __init__ (which would call tiktoken and load a model)."""
    pc = cls.__new__(cls)
    pc.model_name, pc.use_llmlingua2, pc.retrieval_model, pc.retrieval_model_name = None, False, None, None
    pc.open_api_config, pc.cache_bos_num, pc.prefix_bos_num = {}, 10, 100
    pc.oai_tokenizer = QwenCounter(oai_tok)
    pc.context_idxs = []
    return pc


def trim_to(tok, text: str, K: int) -> tuple[str, int, bool]:
    """Hard cap at K reader tokens (keeps the beginning). -> (text, n_tokens, trimmed?)"""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= K:
        return text, len(ids), False
    return tok.decode(ids[:K]), K, True


class Lingua2:
    def __init__(self, reader_tok, device: str = "cuda", name: str = L2_NAME):
        _ensure_importable()
        from llmlingua import PromptCompressor
        self.tok = reader_tok
        pc = _bare_compressor(PromptCompressor, reader_tok)
        pc.model_name, pc.use_llmlingua2 = name, True
        pc.load_model(name, device, {})
        pc.init_llmlingua2()
        self.pc = pc

    def compress(self, text: str, target_token: int | None = None, rate: float | None = None) -> tuple[str, dict]:
        n0 = len(self.tok(text, add_special_tokens=False)["input_ids"])
        K = target_token if target_token is not None else max(1, int(round(n0 * rate)))
        if n0 <= K:
            return text, {"full": 1, "doc_tokens": n0, "content_tokens": n0, "target": K}
        r = self.pc.compress_prompt_llmlingua2([text], rate=min(1.0, K / n0), target_token=K, force_tokens=L2_FORCE)
        out, n, trimmed = trim_to(self.tok, r["compressed_prompt"], K)
        return out, {"full": 0, "doc_tokens": n0, "content_tokens": n, "target": K, "hard_trim": int(trimmed)}


class _Len(list):
    pass


def _fake_kv(L: int, device):
    t = torch.zeros(1, 1, int(L), 1, device=device)
    return _Len([[t, t]])


def make_hybrid_safe(cls):
    """Subclass of llmlingua.PromptCompressor whose get_ppl never slices a model cache (see module doc)."""

    class HybridSafeCompressor(cls):
        _snaps: list
        resume_after: int = 200                    # = iterative_size: next call resumes at end - iterative_size
        stats: dict

        def _snapshot_for(self, ids: torch.Tensor, L: int):
            for pref, cache in reversed(self._snaps):
                if pref.shape[1] == L and torch.equal(pref, ids[:, :L]):
                    return copy.deepcopy(cache)
            return None

        @torch.no_grad()
        def _forward_hidden(self, ids: torch.Tensor, past: int, end: int):
            """Hidden states of positions [past, end) of a forward over ids[:, :end]; snapshots the cache at
            end - resume_after (if past < that < end)."""
            m = self.model
            cache = self._snapshot_for(ids, past) if past > 0 else None
            if past > 0 and cache is None:
                self.stats["recompute"] += 1
                cache = None
                lo = 0
            else:
                self.stats["hit" if past > 0 else "fresh"] += 1
                lo = past
            base = m.model if hasattr(m, "model") else m
            hs = []
            cut = end - self.resume_after
            bounds = [lo] + ([cut] if lo < cut < end else []) + [end]
            for s, e in zip(bounds[:-1], bounds[1:]):
                o = base(input_ids=ids[:, s:e], past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                hs.append(o.last_hidden_state)
                if e == cut:
                    self._snaps.append((ids[:, :cut].clone(), copy.deepcopy(cache)))
                    self._snaps = self._snaps[-4:]
            h = torch.cat(hs, 1)
            return h[:, past - lo:]

        def get_prefix_length(self, prefix: str, text: str):
            """Number of tokens of `prefix` at the start of tokenize(prefix + text), found by character offsets.
            (llmlingua's version decodes token prefixes and asserts a string round trip, which fails when the
            Qwen BPE merges across the prefix / text seam.) A token straddling the seam counts as text."""
            enc = self.tokenizer(prefix + text[:100], add_special_tokens=False, return_offsets_mapping=True)
            n = sum(1 for _, e in enc["offset_mapping"] if e <= len(prefix))
            full = self.tokenizer(prefix + text, add_special_tokens=False)["input_ids"]
            assert enc["input_ids"][:n] == full[:n], "prefix tokenization is not stable"
            return n

        def get_ppl(self, text: str, granularity: str = "sentence", input_ids=None, attention_mask=None,
                    past_key_values=None, return_kv=False, end=None, condition_mode: str = "none",
                    condition_pos_id: int = 0):
            if input_ids is None:
                t = self.tokenizer(text, return_tensors="pt")
                input_ids = t["input_ids"].to(self.device)
                attention_mask = t["attention_mask"].to(self.device)
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            if end is None:
                end = input_ids.shape[1]
            end = min(end, past_length + self.max_position_embeddings)
            assert bool((attention_mask[:, :end] == 1).all()), "padding is not supported (batch 1, no pad)"
            h = self._forward_hidden(input_ids, past_length, end)            # [1, end - past, d]
            head = self.model.lm_head if hasattr(self.model, "lm_head") else self.model.get_output_embeddings()
            labels = input_ids[0, past_length + 1:end]
            losses = []
            for s in range(0, labels.shape[0], 512):
                lg = head(h[0, s:min(s + 512, labels.shape[0])]).float()
                losses.append(torch.nn.functional.cross_entropy(lg, labels[s:s + lg.shape[0]], reduction="none"))
            loss = torch.cat(losses) if losses else torch.zeros(0, device=input_ids.device)
            if condition_mode == "before":
                loss = loss[:condition_pos_id]
            elif condition_mode == "after":
                loss = loss[condition_pos_id:]
            res = loss.mean() if granularity == "sentence" else loss
            return (res, _fake_kv(end, input_ids.device)) if return_kv else res

    return HybridSafeCompressor


class LongLingua:
    """LongLLMLingua (rank_method="longllmlingua", the paper's recommended settings) with a loaded causal LM as the
    scoring model. The document is split into C-token chunks (the same 512-token chunks as the RAG rows); the
    chunks are llmlingua's "contexts" (coarse ranking by question-conditioned perplexity, dynamic ratios, then the
    iterative token-level compression with the contrastive question condition)."""

    def __init__(self, model, tok, reader_tok, chunk: int = 512, iterative_size: int = 200):
        _ensure_importable()
        from llmlingua import PromptCompressor
        cls = make_hybrid_safe(PromptCompressor)
        pc = _bare_compressor(cls, reader_tok)
        tok = copy.copy(tok)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        pc.model, pc.tokenizer = model, tok
        try:
            pc.device = model.get_input_embeddings().weight.device
        except Exception:
            pc.device = next(model.parameters()).device
        pc.max_position_embeddings = 1 << 30           # never trigger llmlingua's cache-eviction branch
        pc._snaps, pc.stats, pc.resume_after = [], {"fresh": 0, "hit": 0, "recompute": 0}, int(iterative_size)
        self.pc, self.tok, self.rtok, self.C, self.iter = pc, tok, reader_tok, int(chunk), int(iterative_size)

    def compress(self, text: str, question: str, target_token: int) -> tuple[str, dict]:
        ids = self.rtok(text, add_special_tokens=False)["input_ids"]
        n0 = len(ids)
        K = int(target_token)
        if n0 <= K:
            return text, {"full": 1, "doc_tokens": n0, "content_tokens": n0, "target": K}
        ctx = [self.rtok.decode(ids[s:s + self.C]) for s in range(0, n0, self.C)]
        self.pc._snaps = []
        st0 = dict(self.pc.stats)
        r = self.pc.compress_prompt(
            ctx, instruction="", question=question, target_token=K, iterative_size=self.iter,
            condition_in_question="after_condition", reorder_context="sort", dynamic_context_compression_ratio=0.3,
            condition_compare=True, context_budget="+100", rank_method="longllmlingua", concate_question=False)
        out, n, trimmed = trim_to(self.rtok, r["compressed_prompt"], K)
        st = {k: self.pc.stats[k] - st0[k] for k in st0}
        return out, {"full": 0, "doc_tokens": n0, "content_tokens": n, "target": K, "hard_trim": int(trimmed),
                     "ppl_calls": st}


__all__ = ["Lingua2", "LongLingua", "make_hybrid_safe", "QwenCounter", "trim_to", "L2_NAME"]
