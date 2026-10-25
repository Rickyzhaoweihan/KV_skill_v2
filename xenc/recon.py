"""RECON: verbatim reconstruction auxiliary items for base training (agent RECON, 2026-10-23).

Hub Memory trains with a dense doc-LM KL: at every memory position the reader's next-token distribution must match the
text reader's over the document. Our read has no document rows (the reader sees only the 64-token tail + question), so
the same signal is given in ITEM form, through the unchanged register_d13 loss path:

  recite  question = 'Quote the text of the document that comes right after this excerpt: "<cue>"'
          gold     = the true next L document tokens (teacher-forced KL(text || kv) on every one of them + CE)
  verify  question = 'Consider this statement: "<span>" ... does the document state exactly this?'  A. Yes / B. No
          gold     = A for the verbatim span, B for the span with one detail (number / name / word) swapped for another
          one from the SAME document (the 'altered detail' failure of mcq_hallu)

Spans are drawn from the OPERATOR columns of the document body (pre[S_eff : n_op]), never from the visible tail, so the
reader can only answer through the read. Cues are unique in the document (else the continuation is ambiguous).
All randomness comes from the caller's rng (register_d13 keeps it separate from the main item stream).
"""

from __future__ import annotations

import re

RECITE_Q = 'Quote the text of the document that comes right after this excerpt:\n"{cue}"'
VERIFY_Q = ('Consider this statement:\n"{stmt}"\nDoes the document state exactly this, with every detail (names, '
            'numbers and wording) matching the document?')
YES_NO = [{"label": "A", "text": "Yes"}, {"label": "B", "text": "No"}]

_NUM = re.compile(r"(?<![\w.,])\d+(?:,\d{3})*(?:\.\d+)?(?![\w])")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])|\n+")
_CAP = re.compile(r"(?<![\w'])[A-Z][a-z]{2,}(?![\w'])")
_LOW = re.compile(r"(?<![\w'])[a-z]{5,}(?![\w'])")


def body_bounds(n_pre: int, n_full: int, S: int, tail: int) -> tuple[int, int]:
    """[lo, hi) indices of `pre` that are document BODY tokens AND operator columns.
    pre = the last n_pre of the n_full prefix tokens (template [0, S) + body); the operator is pre[:n_pre - tail]."""
    lo = max(0, S - (n_full - n_pre))
    return lo, n_pre - tail


def _snap(tok, pre, s, s_max):
    """Move s forward (<= 8 tokens, <= s_max) to a token that starts a word (leading space / newline)."""
    for j in range(s, min(s + 8, s_max) + 1):
        t = tok.decode([pre[j]])
        if t[:1] in (" ", "\n", "\t"):
            return j
    return s


def make_recite(tok, pre, lo, hi, rng, doc_text, key, cue=(8, 24), cont=(12, 32), tries=8):
    """-> a span item (question = recite cue, answers = [continuation]) or None."""
    for _ in range(tries):
        c, L = rng.randint(*cue), rng.randint(*cont)
        if hi - lo < c + L + 1:
            return None
        s = _snap(tok, pre, rng.randrange(lo, hi - c - L + 1), hi - c - L)
        cue_s = tok.decode(pre[s: s + c]).strip()
        cont_s = tok.decode(pre[s + c: s + c + L]).strip()
        if len(cue_s) < 15 or len(cont_s) < 8 or "<" in cont_s and "answer>" in cont_s:
            continue
        if doc_text is not None and doc_text.count(cue_s) != 1:
            continue
        return {"key": key, "doc_id": None, "question": RECITE_Q.format(cue=cue_s), "choices": None,
                "answers": [cont_s], "correct_label": None, "task": "recon_recite", "hop": None,
                "meta": {"recon": "recite", "s": s, "c": c, "L": L, "cont_end": s + c + L, "e": s + c + L}}
    return None


def alter_detail(stmt: str, doc_text: str, rng):
    """Swap ONE detail of stmt for another of the same kind from the document. -> (altered, (old, new)) or None.
    Preference: numbers, then capitalised words (not the first word), then long lowercase words."""
    for rx in (_NUM, _CAP, _LOW):
        occ = [m for m in rx.finditer(stmt)
               if not (rx is _CAP and (m.start() == 0 or stmt[: m.start()].rstrip()[-1:] in ".!?:\n\"'"))]
        if not occ:
            continue
        in_stmt = {m.group(0) for m in rx.finditer(stmt)}
        pool = sorted({m.group(0) for m in rx.finditer(doc_text)
                       if not (rx is _CAP and (m.start() == 0 or doc_text[max(0, m.start() - 4): m.start()].rstrip()[-1:]
                                               in ".!?:\n\"'"))} - in_stmt)       # names: mid-sentence capitals only
        if rx is _NUM and not pool:                       # perturb the number itself
            m = occ[rng.randrange(len(occ))]
            v = m.group(0)
            digs = [i for i, ch in enumerate(v) if ch.isdigit()]
            i = digs[rng.randrange(len(digs))]
            d = str((int(v[i]) + rng.randint(1, 9)) % 10)
            if i == 0 and d == "0" and len(digs) > 1:
                d = "1" if v[i] != "1" else "2"
            pool = [v[:i] + d + v[i + 1:]]
        if not pool:
            continue
        m = occ[rng.randrange(len(occ))]
        if rx is _NUM:                                    # numbers: prefer the same number of digits
            same = [x for x in pool if sum(ch.isdigit() for ch in x) == sum(ch.isdigit() for ch in m.group(0))]
            pool = same or pool
        new = pool[rng.randrange(len(pool))]
        alt = stmt[: m.start()] + new + stmt[m.end():]
        if alt != stmt and alt not in doc_text:
            return alt, (m.group(0), new)
    return None


def _sentence(win: str, short: str):
    """Sentence-aligned statement from a decoded token window: from the first sentence start in the window's first
    third (else the window start) to the first sentence end at or after len(short) chars; None if < 30 chars."""
    st = 0
    for m in _SENT.finditer(win):
        if m.end() <= len(win) // 3:
            st = m.end()
            break
    body = win[st:]
    end = None
    for m in re.finditer(r"[.!?](?=\s|$)", body):
        if m.end() >= min(len(short), 300):
            end = m.end()
            break
    stmt = (body[:end] if end else body[: len(short)]).strip()
    stmt = stmt.split("\n\n")[0].strip()
    return stmt if 30 <= len(stmt) <= 400 else None


def make_verify(tok, pre, lo, hi, rng, doc_text, key, span=(16, 40), tries=8):
    """-> an MCQ item (A. Yes: the span verbatim / B. No: one detail altered), balanced by rng, or None."""
    for _ in range(tries):
        n = rng.randint(*span)
        if hi - lo < n + 1:
            return None
        s = _snap(tok, pre, rng.randrange(lo, hi - n + 1), hi - n)
        stmt = _sentence(tok.decode(pre[s: min(hi, s + 2 * n)]), tok.decode(pre[s: s + n]))
        if stmt is None or doc_text.count(stmt) != 1:
            continue
        yes = rng.random() < 0.5
        meta = {"recon": "verify", "s": s, "n": n, "e": min(hi, s + 2 * n), "yes": yes}
        if not yes:
            r = alter_detail(stmt, doc_text, rng)
            if r is None:
                continue
            stmt, meta["swap"] = r
        return {"key": key, "doc_id": None, "question": VERIFY_Q.format(stmt=stmt), "choices": [dict(c) for c in YES_NO],
                "answers": ["Yes" if yes else "No"], "correct_label": "A" if yes else "B", "task": "recon_verify",
                "hop": None, "meta": meta}
    return None


def teacher_doc(tok, pre, lo, s, e, win: int, doc_text: str) -> str:
    """Document text the text-mode TEACHER reads: the full document (win <= 0), or a window of `win` body tokens
    around the span [s, e) (the copy target is local; this keeps the teacher pass cheap on 100k-token documents)."""
    if win <= 0:
        return doc_text
    half = max(0, (win - (e - s)) // 2)
    a0 = max(lo, s - half)
    a1 = min(len(pre), max(e + half, a0 + win))
    return tok.decode(pre[a0:a1]).strip()


def ramp_weight(step: int, w: float, ramp: int) -> float:
    """Aux weight at optimizer step `step` (1-based): linear 0 -> w over `ramp` steps (ramp <= 0: w from step 1)."""
    if w <= 0 or step <= 0:
        return 0.0
    return w if ramp <= 0 else w * min(1.0, step / ramp)
