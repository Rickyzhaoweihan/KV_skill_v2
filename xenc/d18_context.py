"""D18 (1/2): context-dependent questions — items whose final question can only be resolved with EARLIER conversation
turns, built deterministically from existing document-QA items (no new labels; the gold answer is unchanged).

Variants (all keep the original gold label and choices):
  orig        no conversation: the original single-turn question (reference, same rows as the D13 judge)
  deferred    turn 1 states the question but withholds the options; the final turn gives ONLY the options and says
              "answer the question from my previous message". The final turn alone does not say what is asked.
  distractor  turn 1 states TWO questions of the same document (the target and another one, order fixed by a hash of
              the key); the final turn gives the target's options and says "answer Question k only".
  referent    a named entity of the question (and choices) is replaced by "X" in the final turn; turn 1 binds X (and,
              when the document offers one, a distractor entity bound to "Y") — "turn 1 establishes who, turn 2 asks
              about X". Items without a detectable entity that occurs in the document are skipped.

A ContextItem renders three ways (see `render`): the full chat (document in the first user message: the `text`
condition), the same chat without the document (`closed_book`), and the segments the read conditions need:
  conv_str    everything after the document prefix (starts with "\\n\\n## Question\\n"), i.e. the reader's own rows
              after the document tail
  final_str   the final user message content + the generation prompt (what the encoder sees in the encoder-side
              "final question only" condition, as a single-turn question right after the document)
"""

from __future__ import annotations

import hashlib
import re

from xenc.doc_tasks import MCQ_INSTRUCTION, SPAN_INSTRUCTION, _get, q_prompt

QSEP = "\n\n## Question\n"
VARIANTS = ("orig", "deferred", "distractor", "referent")

_STOP = {"What", "Why", "How", "Which", "Who", "Whom", "Whose", "When", "Where", "The", "In", "A", "An", "Of", "On",
         "At", "For", "From", "To", "By", "With", "According", "Based", "Given", "During", "After", "Before", "Is",
         "Are", "Was", "Were", "Does", "Did", "Do", "Can", "Could", "Would", "Should", "If", "As", "That", "This",
         "These", "Those", "It", "Its", "He", "She", "They", "His", "Her", "Their", "I", "We", "You", "Not", "NOT",
         "And", "Or", "But", "So", "Then", "There", "Here", "One", "Two", "Three", "Overall", "Throughout", "All",
         "Most", "Least", "Some", "Many", "Mr", "Mrs", "Ms", "Dr", "Earth", "English", "God", "World", "War", "None",
         "Above", "Both", "Answer", "Question", "Document", "Story", "Passage", "Author", "Narrator", "Option"}
_ENT = re.compile(r"\b(?:(?:Mr|Mrs|Ms|Dr|Prof)\.\s)?[A-Z][a-zA-Z'\-]+(?:\s(?:[A-Z][a-zA-Z'\-]+|of|de|van|von))*"
                  r"(?<![\s'])")


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def _opts(item) -> str:
    ch = _get(item, "choices")
    return "\n".join(f"{c['label']}. {c['text']}" for c in ch) if ch else ""


def _instr(item) -> str:
    return MCQ_INSTRUCTION if _get(item, "choices") else SPAN_INSTRUCTION


def entities(text: str, mid_only: bool = False) -> list[str]:
    """Capitalised (multi-word) names in `text`, skipping sentence-initial function words; possessive stripped.
    mid_only: keep only matches that do not start a sentence / line / quotation (proper names, not capitalised
    sentence-initial words)."""
    out = []
    for m in _ENT.finditer(text):
        if mid_only:
            pre = text[max(0, m.start() - 3):m.start()].rstrip(" ")
            if not pre or pre[-1] in ".!?\n\"'“‘:;—-(" or m.start() == 0:
                continue
        e = m.group(0)
        e = re.sub(r"'s$", "", e).strip()
        words = e.split()
        while words and words[0].rstrip(".") in _STOP:
            words = words[1:]
        if not words:
            continue
        e = " ".join(words)
        if e.rstrip(".") in _STOP or len(e) < 3:
            continue
        if words[0] in ("of", "de", "van", "von"):
            continue
        while e.split()[-1] in ("of", "de", "van", "von"):
            e = " ".join(e.split()[:-1])
        if e not in out:
            out.append(e)
    return out


def pick_entity(item, doc_text: str) -> str | None:
    """Longest named entity of the question that occurs verbatim in the document (None if there is none)."""
    q = str(_get(item, "question"))
    cands = [e for e in entities(q) if e in doc_text]
    if not cands:                                  # titled names: "Mrs. Sanders" -> "Sanders" in the document
        cands = [e for e in entities(q) if re.sub(r"^(Mr|Mrs|Ms|Dr|Prof)\.\s", "", e) in doc_text and
                 re.match(r"^(Mr|Mrs|Ms|Dr|Prof)\.\s", e)]
    if not cands:
        return None
    return max(cands, key=lambda e: (len(e), -q.index(e)))


def distractor_entity(entity: str, doc_text: str, others: list) -> str | None:
    """Another named entity of the same document (from its other questions, else the most frequent capitalised
    name of the text), disjoint from `entity`."""
    def ok(e):
        return e and e != entity and e not in entity and entity not in e and e in doc_text
    for it in others:
        for e in entities(str(_get(it, "question"))):
            if ok(e):
                return e
    from collections import Counter
    c = Counter(e for e in _all_mid(doc_text[:20000]) if ok(e) and len(e.split()) <= 3)
    return c.most_common(1)[0][0] if c else None


def _all_mid(text: str) -> list[str]:
    """Every mid-sentence name occurrence (with repeats, for frequency counts)."""
    out = []
    for m in _ENT.finditer(text):
        pre = text[max(0, m.start() - 3):m.start()].rstrip(" ")
        if m.start() == 0 or not pre or pre[-1] in ".!?\n\"'“‘:;—-(":
            continue
        out += entities(m.group(0))
    return out


def _sub(text: str, ent: str, sym: str) -> str:
    return re.sub(re.escape(ent), sym, text)


def make_context_item(item, variant: str, doc_text: str = "", same_doc: list | None = None) -> dict | None:
    """-> {"key", "doc_id", "variant", "item" (scoring item: original gold/choices), "turns": [(user, assistant)],
    "final": final user message text, "meta": {...}} or None if the variant does not apply."""
    key = str(_get(item, "key"))
    q = str(_get(item, "question")).strip()
    same_doc = [o for o in (same_doc or []) if str(_get(o, "key")) != key]
    base = {"key": key, "doc_id": _get(item, "doc_id"), "variant": variant, "item": item, "meta": {}}
    if variant == "orig":
        return {**base, "turns": [], "final": q_prompt(item)}
    if variant == "deferred":
        u1 = f"Here is my question about the document; I will send the answer options next.\n{q}"
        a1 = "Understood. Please send the answer options."
        if _get(item, "choices"):
            fin = (f"Here are the options for the question in my previous message:\n\n{_opts(item)}\n\n"
                   f"Answer the question from my previous message. {_instr(item)}")
        else:
            fin = f"Now answer the question from my previous message. {_instr(item)}"
        return {**base, "turns": [(u1, a1)], "final": fin}
    if variant == "distractor":
        if not same_doc:
            return None
        other = same_doc[_h(key) % len(same_doc)]
        oq = str(_get(other, "question")).strip()
        k = 1 + (_h(key + "#order") % 2)                 # position of the target: 1 or 2
        qs = [q, oq] if k == 1 else [oq, q]
        u1 = ("I have two questions about the document; I will tell you which one to answer.\n"
              f"Question 1: {qs[0]}\nQuestion 2: {qs[1]}")
        a1 = "Got it. Which question should I answer?"
        if _get(item, "choices"):
            fin = f"Answer Question {k} only. Its options are:\n\n{_opts(item)}\n\n{_instr(item)}"
        else:
            fin = f"Answer Question {k} only. {_instr(item)}"
        return {**base, "turns": [(u1, a1)], "final": fin,
                "meta": {"target_pos": k, "distractor_key": str(_get(other, "key"))}}
    if variant == "referent":
        ent = pick_entity(item, doc_text)
        if ent is None:
            return None
        dis = distractor_entity(ent, doc_text, same_doc)
        fq = _sub(q, ent, "X")
        fit = dict(item) if isinstance(item, dict) else item.to_dict()
        if fit.get("choices"):
            fit["choices"] = [{"label": c["label"], "text": _sub(c["text"], ent, "X")} for c in fit["choices"]]
        fit["question"] = fq
        if dis is not None:
            first = _h(key + "#ref") % 2 == 0
            binds = [f"{ent} as X", f"{dis} as Y"] if first else [f"{dis} as Y", f"{ent} as X"]
            u1 = f"In my next message I will refer to {binds[0]} and to {binds[1]}."
        else:
            u1 = f"In my next message I will refer to {ent} as X."
        a1 = "Okay, noted."
        return {**base, "turns": [(u1, a1)], "final": q_prompt(fit),
                "meta": {"entity": ent, "distractor_entity": dis}}
    raise ValueError(f"unknown variant {variant!r}")


def build_context_items(items, docs: dict, variant: str) -> list[dict]:
    """Deterministic: every item for which the variant applies, in input order. `same_doc` = the doc's other items
    (in the order given)."""
    by: dict = {}
    for it in items:
        by.setdefault(_get(it, "doc_id"), []).append(it)
    out = []
    for it in items:
        c = make_context_item(it, variant, docs.get(_get(it, "doc_id"), ""), by[_get(it, "doc_id")])
        if c is not None:
            out.append(c)
    return out


# ----------------------------------------------------------------------------- rendering
def messages(ci: dict, system: str | None, doc_text: str | None) -> list[dict]:
    """Chat messages; the document (if given) opens the FIRST user message as in ctx_prompt."""
    turns = list(ci["turns"]) + [(ci["final"], None)]
    msgs = [{"role": "system", "content": system}] if system else []
    for i, (u, a) in enumerate(turns):
        if i == 0 and doc_text is not None:
            u = f"## Document\n{doc_text}{QSEP}{u}"
        msgs.append({"role": "user", "content": u})
        if a is not None:
            msgs.append({"role": "assistant", "content": a})
    return msgs


def render(tok, hub: str, ci: dict, system: str | None, doc_text: str | None) -> str:
    from deltaskill.backbone_utils import chat_template_kwargs
    return tok.apply_chat_template(messages(ci, system, doc_text), tokenize=False, add_generation_prompt=True,
                                   **chat_template_kwargs(hub))


def segments(tok, hub: str, ci: dict, system: str) -> dict:
    """Strings of the read conditions (document-independent; rendered with a placeholder document):
    conv_str   the chat after the document prefix (starts with QSEP)
    final_str  the final user message content + end-of-turn + generation prompt (the shared token tail of the
               reader's final turn and of the encoder's single-turn "final question only" prompt)
    enc_final  QSEP + final_str (the encoder's own question suffix after the document prefix)"""
    s = render(tok, hub, ci, system, "\x00DOC\x00")
    k = s.find("\x00DOC\x00") + len("\x00DOC\x00")
    assert s[k:k + len(QSEP)] == QSEP, "question separator not found"
    conv = s[k:]
    j = conv.rfind(ci["final"])
    assert j >= 0, "final message not found in the rendered chat"
    return {"conv_str": conv, "final_str": conv[j:], "enc_final": QSEP + conv[j:]}


_TEMPLATE = [r"Here is my question about the document; I will send the answer options next\.",
             r"Here are the options for the question in my previous message:", r"Answer the question from my previous message\.",
             r"Now answer the question from my previous message\.",
             r"I have two questions about the document; I will tell you which one to answer\.", r"Question [12]:",
             r"Answer Question [12] only\.( Its options are:)?", r"In my next message I will refer to", r"\bas [XY]\b",
             r"\band to\b", re.escape(MCQ_INSTRUCTION), re.escape(SPAN_INSTRUCTION), r"^[A-E]\. ", r"\n[A-E]\. "]


def strip_template(text: str) -> str:
    """Remove this module's fixed conversational boilerplate (so a retriever query holds only item content)."""
    for pat in _TEMPLATE:
        text = re.sub(pat, " ", text)
    return " ".join(text.split())


def common_suffix_len(a: list, b: list) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


__all__ = ["VARIANTS", "make_context_item", "build_context_items", "messages", "render", "segments", "entities",
           "pick_entity", "common_suffix_len", "strip_template", "QSEP"]
