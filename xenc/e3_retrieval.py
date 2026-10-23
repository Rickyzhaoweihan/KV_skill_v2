"""E3: retrieval-aligned signals for the D13 reader-query read (xenc/d13_read.py).

Problem (outputs/D22_C3.md): on span tasks the kv answers are well-formed document spans but the WRONG spans, i.e. the
read addresses the wrong operator columns. Operator columns are encoder token positions of the document prefix; the 4B
encoder and the 9B reader share the tokenizer and the prefix, so column c == position c of the reader's text prompt
(checked per item: text_ids[:N] == operator ids).

Everything here works on head-MIXTURE distributions over the document-body columns [S, N) at selected rows:
    p(c) ∝ mean_h softmax_h(logits)[c]        (full softmax over all columns first, so heads that look elsewhere
                                               contribute little; then restricted to the body and renormalised)
S = first document-body column (the system prompt / chat header columns, i.e. the attention sink, are excluded).

Signals (scripts/register_d13.py --ret_w / --ret_src, default off):
  text   KL(p_text9B ‖ p_student) — the 9B reader's own attention at the same layer while it reads the document as text
         (label-free; captured during the teacher pass that already computes the KL(text ‖ kv) logits)
  enc    KL(p_enc4B ‖ p_student) — the 4B encoder's own question-pass attention (the stage-1 teacher), restricted to the
         body and optionally sharpened to its top-k columns (retrieval-focused version of the existing dist term)
  gold   −log Σ_{c ∈ gold span} p_student(c) — gold-answer-span columns vs all other columns (train items only)
Optional --ret_topk k keeps only the teacher's top-k columns (renormalised) per row.
"""

from __future__ import annotations

import math
import types

import torch


# ----------------------------------------------------------------------------- distributions
def mixture_logp(logits: torch.Tensor, rows, S: int, N: int) -> torch.Tensor:
    """logits [..., T, C] (any leading head dims, -inf masked) -> log p [len(rows), N-S]: head-mixture of the full
    softmax over C, restricted to columns [S, N) and renormalised. Differentiable."""
    lg = logits.reshape(-1, logits.shape[-2], logits.shape[-1])[:, rows, :].float()      # [H, R, C]
    lp = torch.log_softmax(lg, -1)[..., S:N]                                              # [H, R, N-S]
    mix = torch.logsumexp(lp, 0) - math.log(lp.shape[0])                                  # [R, N-S]
    return mix - torch.logsumexp(mix, -1, keepdim=True)


def sharpen(logp_t: torch.Tensor, topk: int) -> torch.Tensor:
    """Keep each row's top-k columns of the teacher (renormalised); topk <= 0: unchanged."""
    if topk <= 0 or topk >= logp_t.shape[-1]:
        return logp_t
    v, i = logp_t.topk(topk, -1)
    out = torch.full_like(logp_t, float("-inf"))
    out.scatter_(-1, i, v - torch.logsumexp(v, -1, keepdim=True))
    return out


def kl_rows(logp_t: torch.Tensor, logp_s: torch.Tensor) -> torch.Tensor:
    """mean over rows of KL(p_t ‖ p_s); teacher -inf columns contribute 0."""
    pt = logp_t.exp()
    d = pt * (torch.nan_to_num(logp_t, neginf=0.0) - torch.nan_to_num(logp_s, neginf=-1e4))
    return d.sum(-1).mean()


def span_nll(logp_s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask [N-S] bool (gold columns). -> mean over rows of −log Σ_{c∈mask} p_s(c)."""
    return -(torch.logsumexp(logp_s[:, mask], -1)).mean()


def retrieval_metrics(logp: torch.Tensor, mask: torch.Tensor, window: int = 0) -> dict:
    """logp [R, N-S]; mask [N-S] bool -> hit@1 (argmax within mask dilated by ±window), gold mass, gold rank (min over
    gold columns of the rank, 1 = top), uniform mass |mask|/(N-S)."""
    m = dilate(mask, window) if window else mask
    p = logp.exp()
    top = p.argmax(-1)
    order = p.argsort(-1, descending=True)
    pos = torch.empty_like(order)
    pos.scatter_(-1, order, torch.arange(order.shape[-1], device=order.device).expand_as(order).contiguous())
    best_rank = pos[:, mask].min(-1).values + 1
    return {"hit1": float(m[top].float().mean()), "mass": float(p[:, mask].sum(-1).mean()),
            "rank": float(best_rank.float().median()), "mrr": float((1.0 / best_rank.float()).mean()),
            "uniform": float(mask.float().mean())}


def dilate(mask: torch.Tensor, w: int) -> torch.Tensor:
    if w <= 0:
        return mask
    x = mask.float().view(1, 1, -1)
    return (torch.nn.functional.max_pool1d(x, 2 * w + 1, stride=1, padding=w).view(-1) > 0)


# ----------------------------------------------------------------------------- gold columns
def gold_mask(tok, prefix_text: str, doc_text: str, answers, N: int, S: int, min_chars: int = 2) -> torch.Tensor | None:
    """Bool [N-S] over the body columns: tokens of the document prefix overlapping any (case-insensitive) occurrence of
    any gold answer inside the DOCUMENT part of prefix_text. None when no occurrence lies in columns [S, N)."""
    off = tok(prefix_text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    d0 = prefix_text.rfind(doc_text) if doc_text else -1
    if d0 < 0:
        return None
    low = prefix_text.lower()
    spans = []
    for a in answers or []:
        a = (a or "").strip().lower()
        if len(a) < min_chars:
            continue
        j = low.find(a, d0)
        while j >= 0:
            spans.append((j, j + len(a)))
            j = low.find(a, j + 1)
    return chars_to_mask(off, spans, N, S)


def chars_to_mask(off, spans, N: int, S: int) -> torch.Tensor | None:
    if not spans:
        return None
    m = torch.zeros(N - S, dtype=torch.bool)
    for c in range(S, min(N, len(off))):
        a, b = off[c]
        if b <= a:
            continue
        for s, e in spans:
            if a < e and b > s:
                m[c - S] = True
                break
    return m if bool(m.any()) else None


def body_start(tok, hub: str, system: str) -> int:
    """First document-body column: length of the document prefix with an empty document."""
    from xenc.state_skill import doc_prefix_text
    return len(tok(doc_prefix_text(tok, hub, "", system), add_special_tokens=False)["input_ids"])


def answer_content_rows(tok, target: str, open_tag: str = "<answer>", close_tag: str = "</answer>"):
    """target "<answer>X</answer>" -> (ids, loc): ids = its token ids (as text_ids), loc = indices i of the tokens that
    overlap the content X (char offsets), so the (m+1)-window row i predicts content token ids[i]. Robust to BPE
    merges across the tags (e.g. ">B", ".</"). loc = [] if the target is not tag-wrapped or X is empty."""
    enc = tok(target, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = list(enc["input_ids"]), enc["offset_mapping"]
    if not (target.startswith(open_tag) and target.endswith(close_tag)):
        return ids, []
    a, b = len(open_tag), len(target) - len(close_tag)
    if b <= a:
        return ids, []
    return ids, [i for i, (s, e) in enumerate(off) if e > s and s < b and e > a]


# ----------------------------------------------------------------------------- 9B text-pass capture
class QKCap:
    """Patches the attention modules of `model` at `layers` (chained over any earlier patch, e.g. D13Reader in mode
    "off"). While self.keep is set (int K), a forward records per layer {"q": post-RoPE queries of the LAST K rows
    [Hq,K,hd], "k": post-RoPE keys of all rows [Hkv,L,hd]}; self.keep = None -> pass-through."""

    def __init__(self, model, layers):
        from deltaskill.backbone_utils import decoder_layers
        L = decoder_layers(model)
        self.layers = sorted(set(int(x) for x in layers))
        self.mods = {e: L[e].self_attn for e in self.layers}
        self.keep: int | None = None
        self.rec: dict = {}
        self._orig = {}
        for e, m in self.mods.items():
            self._orig[e] = m.forward
            m.forward = types.MethodType(_cap_forward(self, e), m)

    def detach(self):
        for e, f in self._orig.items():
            self.mods[e].forward = f
        self._orig.clear()

    def logits(self, e: int) -> torch.Tensor:
        """Causal attention logits of the recorded rows: [Hq, K, L] (fp32, -inf above the diagonal)."""
        r = self.rec[e]
        q, k = r["q"].float(), r["k"].float()
        Hq, K, hd = q.shape
        Hkv, L, _ = k.shape
        s = torch.einsum("kgtd,kcd->kgtc", q.view(Hkv, Hq // Hkv, K, hd), k).reshape(Hq, K, L) * r["scale"]
        col = torch.arange(L, device=s.device).view(1, L)
        row = torch.arange(L - K, L, device=s.device).view(K, 1)
        return s.masked_fill((col > row).view(1, K, L), float("-inf"))


def _cap_forward(C: QKCap, e: int):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        if C.keep is not None:
            with torch.no_grad():
                shp = hidden_states.shape[:-1]
                hs = (*shp, -1, self.head_dim)
                k = self.k_norm(self.k_proj(hidden_states).view(hs)).transpose(1, 2)
                K = int(C.keep)
                hq = hidden_states[:, -K:]
                q, _ = torch.chunk(self.q_proj(hq).view(*hq.shape[:-1], -1, self.head_dim * 2), 2, dim=-1)
                q = self.q_norm(q.reshape(*hq.shape[:-1], -1, self.head_dim)).transpose(1, 2)
                cos, sin = position_embeddings
                _, kk = apply_rotary_pos_emb(k, k, cos, sin)
                qq, _ = apply_rotary_pos_emb(q, q, cos[:, -K:], sin[:, -K:])
                C.rec[e] = {"q": qq[0].detach(), "k": kk[0].detach(), "scale": float(self.scaling)}
        return C._orig[e](hidden_states, position_embeddings, attention_mask, past_key_values=past_key_values, **kwargs)

    return forward


__all__ = ["mixture_logp", "sharpen", "kl_rows", "span_nll", "retrieval_metrics", "dilate", "gold_mask",
           "chars_to_mask", "body_start", "QKCap", "answer_content_rows"]
