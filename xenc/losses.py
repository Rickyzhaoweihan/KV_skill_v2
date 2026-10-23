"""Registration losses (agent [D]).

Teacher = reader with the document in context; student = reader on the question alone with the
document's Library loaded. Both forwards keep only the last m+1 logits (`logits_to_keep=m+1`,
`use_cache=False`) — never full-vocab logits at long context — and the student CE is computed by
hand from those kept rows (never via `labels=`).

Row convention: `teacher_logits` / `student_logits` return the m+1 KEPT rows [m+1, V]; rows [:-1]
are the ones that predict the m target tokens. `kl_teacher_student` averages over whatever rows it
is given (callers pass `[:-1]`); `answer_ce` trims an m+1-row input itself.

Fake models (tests/_tiny.TinyLM: no `.generate`): `ids` may be float embeddings [B,T,d] or int ids
(embedded via xenc.backbone.embed_ids); logits come from `model.logits(x)`.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F

from xenc.backbone import embed_ids, input_device, is_fake


def _kept_logits(model, ids: torch.Tensor, keep: int) -> torch.Tensor:
    if is_fake(model):
        x = ids if ids.is_floating_point() else embed_ids(model, ids)
        return model.logits(x.to(model.device))[0, -keep:]
    out = model(input_ids=ids.to(input_device(model)), logits_to_keep=keep, use_cache=False)
    return out.logits[0, -keep:]


def teacher_logits(model, ids_ctx: torch.Tensor, m: int) -> torch.Tensor:
    """[m+1, V] kept logits of the doc-in-context teacher; no grad, detached."""
    with torch.no_grad():
        return _kept_logits(model, ids_ctx, m + 1).detach()


def student_logits(model, ids_q: torch.Tensor, m: int) -> torch.Tensor:
    """[m+1, V] kept logits of the student; grad flows only into the (trainable) injector params."""
    with torch.enable_grad():
        return _kept_logits(model, ids_q, m + 1)


def kl_teacher_student(p_t_logits: torch.Tensor, p_s_logits: torch.Tensor,
                       mask: torch.Tensor | None = None) -> torch.Tensor:
    """KL(p_T || p_S) per row, averaged over rows (optionally a boolean/float row mask)."""
    s = p_s_logits.float()
    t = p_t_logits.to(s.device).float()
    if t.shape != s.shape:
        raise ValueError(f"teacher {tuple(t.shape)} vs student {tuple(s.shape)} logits")
    pt = F.softmax(t, -1)
    kl = (pt * (pt.clamp_min(1e-9).log() - F.log_softmax(s, -1))).sum(-1)
    if mask is None:
        return kl.mean()
    w = mask.to(kl.device, kl.dtype)
    return (kl * w).sum() / w.sum().clamp_min(1.0)


def answer_ce(logits: torch.Tensor, target_ids, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Manual CE of `target_ids` (length m) from kept logits ([m+1,V] or [m,V]); optional span mask."""
    tgt = torch.as_tensor(target_ids, device=logits.device).long().reshape(-1)
    m = tgt.numel()
    if logits.shape[0] == m + 1:
        logits = logits[:-1]
    if logits.shape[0] != m:
        raise ValueError(f"{logits.shape[0]} logit rows for {m} target tokens")
    ce = F.cross_entropy(logits.float(), tgt, reduction="none")
    if mask is None:
        return ce.mean()
    w = mask.to(ce.device, ce.dtype)
    return (ce * w).sum() / w.sum().clamp_min(1.0)


def magnitude_penalty(injector) -> torch.Tensor:
    """Mean of injector.mag_terms (per-layer mean ||delta||/||h|| of the last forward(s))."""
    terms = list(getattr(injector, "mag_terms", []) or [])
    if not terms:
        return torch.zeros(())
    dev = terms[0].device
    return torch.stack([t.to(dev) for t in terms]).mean()


def route_aux_loss(injector, gold_chunk_idx) -> torch.Tensor | None:
    """−log(mean routing weight on the gold chunk(s)), averaged over hooked layers.

    Uses `injector.route_terms` ([(layer, w [B,T,C])] with grad) when present, else
    `injector.last_route` (detached → logging only, no gradient). None if no gold chunk is known."""
    idx = [int(i) for i in (gold_chunk_idx or [])]
    if not idx:
        return None
    terms = [w for _, w in (getattr(injector, "route_terms", None) or [])]
    if not terms and getattr(injector, "last_route", None) is not None:
        terms = [injector.last_route]
    if not terms:
        return None
    dev = terms[0].device
    vals = []
    for w in terms:
        C = w.shape[-1]
        ii = [i for i in idx if 0 <= i < C]
        if not ii:
            continue
        g = w[..., torch.tensor(ii, device=w.device)].sum(-1).mean()
        vals.append(g.to(dev))
    if not vals:
        return None
    return -torch.log(torch.stack(vals).clamp_min(1e-6)).mean()


def slot_aux_loss(injector, gold_mask: torch.Tensor) -> torch.Tensor | None:
    """D2 slot-level retrieval aux: mean over hooked layers of −log(Σ_{gold slots} aw + 1e-6), where aw
    is the (route × slot) read attention [C,N] at `injector.attn_pos` kept WITH grad in
    `injector.attn_terms` (set `injector.training_attn = True` before the student forward).
    None if no term was recorded or the mask is None/empty."""
    terms = list(getattr(injector, "attn_terms", None) or [])
    if gold_mask is None or not terms or not bool(gold_mask.any()):
        return None
    vals = []
    for aw in terms:
        g = gold_mask.to(aw.device)
        C, N = aw.shape[-2:]
        g = g[:C, :N]
        if g.shape != aw.shape[-2:]:
            g = torch.nn.functional.pad(g, (0, N - g.shape[1], 0, C - g.shape[0]))
        vals.append(-torch.log((aw.float() * g).sum() + 1e-6))
    dev = vals[0].device
    return torch.stack([v.to(dev) for v in vals]).mean()


_ANS_CLOSE = "</answer>"


def trace_target(teacher_generation: str) -> str:
    """The teacher's greedy trace up to and including the first </answer> (whitespace-stripped)."""
    t = (teacher_generation or "").strip()
    if _ANS_CLOSE in t:
        t = t.split(_ANS_CLOSE, 1)[0] + _ANS_CLOSE
    return t


def answer_span_mask(tok, target: str, n_tokens: int) -> torch.Tensor:
    """Bool [n_tokens] marking the tokens of the LAST <answer>...</answer> span of `target`
    (approximate on merged token boundaries: span starts at len(tok(prefix))). All-True if absent."""
    from xenc.backbone import text_ids
    k = target.rfind("<answer>")
    mask = torch.zeros(n_tokens, dtype=torch.bool)
    if k < 0:
        return mask | True
    start = min(len(text_ids(tok, target[:k])), max(n_tokens - 1, 0))
    mask[start:] = True
    return mask


def extract_answer(text: str) -> str | None:
    m = re.findall(r"<answer>(.*?)</answer>", text or "", flags=re.S)
    return m[-1].strip() if m else None


def safe_float(x) -> float:
    v = float(x.detach()) if isinstance(x, torch.Tensor) else float(x)
    return v if math.isfinite(v) else float("nan")


__all__ = ["teacher_logits", "student_logits", "kl_teacher_student", "answer_ce", "magnitude_penalty",
           "route_aux_loss", "slot_aux_loss", "trace_target", "answer_span_mask", "extract_answer"]
