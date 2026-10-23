"""Gold-slot masks over a document Library (D2; shared by scripts/probe_read.py and register_doc.py).

A Library slot == one encoder token of the document (chunked by `chunk_meta[c]["start"/"n_tokens"]`
over ONE whole-document tokenization, as in `depth_skill.derive_by_depth`). A slot is "gold" when its
token overlaps a gold character span:
  - MuSiQue-style items (`meta.supporting_idx` + `meta.para_spans`): the supporting paragraphs' spans;
  - otherwise every (case-insensitive) occurrence of any answer string (len >= 2) — and, for MCQ items
    whose `answers` is empty, the correct choice text.

    gold_slot_mask(doc_text, answers, tok, chunk_tokens, lib)          # answer strings (probe_read API)
    item_gold_slot_mask(item, doc_text, tok, chunk_tokens, lib)        # dispatch on item meta
Both return a bool [C, N] mask (N = lib's padded slot count) or None when no gold span is known.
"""

from __future__ import annotations

import re

import torch


def _get(item, k, default=None):
    if isinstance(item, dict):
        return item.get(k, default)
    return getattr(item, k, default)


def answer_spans(doc_text: str, answers) -> list[tuple[int, int]]:
    spans = []
    for ans in answers or []:
        ans = str(ans).strip()
        if len(ans) < 2:
            continue
        for m in re.finditer(re.escape(ans), doc_text, flags=re.IGNORECASE):
            spans.append((m.start(), m.end()))
    return spans


def item_gold_spans(item, doc_text: str) -> list[tuple[int, int]]:
    meta = _get(item, "meta") or {}
    sup, ps = meta.get("supporting_idx"), meta.get("para_spans")
    if sup and ps:
        return [(int(ps[int(i)][0]), int(ps[int(i)][1])) for i in sup if 0 <= int(i) < len(ps)]
    answers = list(_get(item, "answers") or [])
    if _get(item, "choices") and _get(item, "correct_label"):
        lab = _get(item, "correct_label")
        answers += [c["text"] for c in _get(item, "choices") if c.get("label") == lab]
    return answer_spans(doc_text, answers)


def _token_offsets(tok, doc_text: str) -> list[tuple[int, int]]:
    try:
        enc = tok(doc_text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        if offs and isinstance(offs[0], (list, tuple)) and offs[0] and isinstance(offs[0][0], (list, tuple)):
            offs = offs[0]
        return [(int(a), int(b)) for a, b in offs]
    except (TypeError, KeyError, NotImplementedError):
        # whitespace fake tokenizer (TinyTokenizer): one token per \S+ word
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", doc_text)]


def spans_to_slot_mask(doc_text: str, spans, tok, chunk_tokens: int, lib) -> torch.Tensor | None:
    if not spans:
        return None
    offs = _token_offsets(tok, doc_text)
    gold_tok = torch.zeros(len(offs), dtype=torch.bool)
    for (s, e) in spans:
        for ti, (a, b) in enumerate(offs):
            if b > s and a < e:
                gold_tok[ti] = True
    if not bool(gold_tok.any()):
        return None
    C, N = next(iter(lib.K.values())).shape[:2]
    M = torch.zeros(C, N, dtype=torch.bool)
    for c, meta in enumerate(lib.chunk_meta):
        st = int(meta.get("start", c * chunk_tokens)); n = int(meta.get("n_tokens", chunk_tokens))
        seg = gold_tok[st: st + min(n, N)]
        M[c, : seg.shape[0]] = seg
    return M if bool(M.any()) else None


def gold_slot_mask(doc_text: str, answers, tok, chunk_tokens: int, lib) -> torch.Tensor | None:
    """[C,N] bool: slots whose token lies inside an occurrence of any answer string."""
    return spans_to_slot_mask(doc_text, answer_spans(doc_text, answers), tok, chunk_tokens, lib)


def item_gold_slot_mask(item, doc_text: str, tok, chunk_tokens: int, lib) -> torch.Tensor | None:
    """[C,N] bool gold-slot mask of an item (supporting paragraphs for MuSiQue, else answer strings)."""
    return spans_to_slot_mask(doc_text, item_gold_spans(item, doc_text), tok, chunk_tokens, lib)


__all__ = ["answer_spans", "item_gold_spans", "spans_to_slot_mask", "gold_slot_mask", "item_gold_slot_mask"]
