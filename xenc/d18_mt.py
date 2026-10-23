"""D18 direction A: multi-turn TRAINING items (deferred / distractor / referent) with paraphrased templates.

The judge (xenc/d18_context.py) renders each variant with ONE fixed wording. Training draws its wording from a small
pool (template 0 = the judge wording, 1–2 = paraphrases), so a gain on the judge cannot come from memorising the judge's
exact phrasing alone. The output dicts have exactly the shape of d18_context.make_context_item (keys, doc_id,
variant, item, turns, final, meta), so `segments` / `render` work unchanged. Only ever call this on TRAIN / VAL
documents (the training script enforces it).
"""

from __future__ import annotations

import random

from xenc.d18_context import _h, _instr, _opts, _sub, distractor_entity, make_context_item, pick_entity
from xenc.doc_tasks import _get, q_prompt

MT_VARIANTS = ("deferred", "distractor", "referent")

_DEF_U1 = ("Here is my question about the document; I will send the answer options next.\n{q}",
           "I want to ask something about the document. The answer options will follow in my next message.\n{q}",
           "Question about the document (options to follow):\n{q}")
_DEF_A1 = ("Understood. Please send the answer options.", "Sure, send the options when ready.",
           "Okay, I'll wait for the options.")
_DEF_FIN_MCQ = ("Here are the options for the question in my previous message:\n\n{o}\n\nAnswer the question from my "
                "previous message. {i}",
                "Options for that question:\n\n{o}\n\nNow answer the question I asked before. {i}",
                "The choices are:\n\n{o}\n\nPlease answer my earlier question. {i}")
_DEF_FIN_SPAN = ("Now answer the question from my previous message. {i}", "Please answer the question I asked before. {i}",
                 "Go ahead and answer my earlier question. {i}")
_DIS_U1 = ("I have two questions about the document; I will tell you which one to answer.\n"
           "Question 1: {a}\nQuestion 2: {b}",
           "Two questions about the document follow; I'll say which one I need.\nQuestion 1: {a}\nQuestion 2: {b}",
           "Consider these two questions about the document; I will pick one afterwards.\n"
           "Question 1: {a}\nQuestion 2: {b}")
_DIS_A1 = ("Got it. Which question should I answer?", "Okay. Which one do you want answered?",
           "Noted both. Which should I answer?")
_DIS_FIN_MCQ = ("Answer Question {k} only. Its options are:\n\n{o}\n\n{i}",
                "Please answer only Question {k}. The options:\n\n{o}\n\n{i}",
                "I need Question {k}. Options:\n\n{o}\n\n{i}")
_DIS_FIN_SPAN = ("Answer Question {k} only. {i}", "Please answer only Question {k}. {i}", "I need Question {k}. {i}")
_REF_U1_2 = ("In my next message I will refer to {a} and to {b}.", "Next, I'll call {a} and {b}.",
             "For my next question: I will use {a}, and {b}.")
_REF_U1_1 = ("In my next message I will refer to {a}.", "Next, I'll call {a}.", "For my next question: I will use {a}.")
_REF_A1 = ("Okay, noted.", "Understood.", "Got it.")


def make_train_item(item, variant: str, doc_text: str, same_doc: list, rng: random.Random,
                    template: int | None = None) -> dict | None:
    """A context item like d18_context.make_context_item, wording drawn from the pool (template=None: random;
    template=0 reproduces the judge wording exactly)."""
    if variant == "orig":
        return make_context_item(item, "orig", doc_text, same_doc)
    t = rng.randrange(3) if template is None else int(template)
    key = str(_get(item, "key"))
    q = str(_get(item, "question")).strip()
    same_doc = [o for o in (same_doc or []) if str(_get(o, "key")) != key]
    base = {"key": key, "doc_id": _get(item, "doc_id"), "variant": variant, "item": item, "meta": {"template": t}}
    mcq = bool(_get(item, "choices"))
    if variant == "deferred":
        fin = (_DEF_FIN_MCQ[t].format(o=_opts(item), i=_instr(item)) if mcq else _DEF_FIN_SPAN[t].format(i=_instr(item)))
        return {**base, "turns": [(_DEF_U1[t].format(q=q), _DEF_A1[t])], "final": fin}
    if variant == "distractor":
        if not same_doc:
            return None
        other = same_doc[rng.randrange(len(same_doc))] if template is None else same_doc[_h(key) % len(same_doc)]
        oq = str(_get(other, "question")).strip()
        k = (1 + rng.randrange(2)) if template is None else 1 + (_h(key + "#order") % 2)
        a_, b_ = (q, oq) if k == 1 else (oq, q)
        fin = (_DIS_FIN_MCQ[t].format(k=k, o=_opts(item), i=_instr(item)) if mcq
               else _DIS_FIN_SPAN[t].format(k=k, i=_instr(item)))
        return {**base, "turns": [(_DIS_U1[t].format(a=a_, b=b_), _DIS_A1[t])], "final": fin,
                "meta": {**base["meta"], "target_pos": k, "distractor_key": str(_get(other, "key"))}}
    if variant == "referent":
        ent = pick_entity(item, doc_text)
        if ent is None:
            return None
        dis = distractor_entity(ent, doc_text, same_doc)
        fit = dict(item) if isinstance(item, dict) else item.to_dict()
        if fit.get("choices"):
            fit["choices"] = [{"label": c["label"], "text": _sub(c["text"], ent, "X")} for c in fit["choices"]]
        fit["question"] = _sub(q, ent, "X")
        if dis is not None:
            first = (rng.randrange(2) == 0) if template is None else (_h(key + "#ref") % 2 == 0)
            bx, by = f"{ent} as X", f"{dis} as Y"
            u1 = _REF_U1_2[t].format(a=bx, b=by) if first else _REF_U1_2[t].format(a=by, b=bx)
        else:
            u1 = _REF_U1_1[t].format(a=f"{ent} as X")
        return {**base, "turns": [(u1, _REF_A1[t])], "final": q_prompt(fit),
                "meta": {**base["meta"], "entity": ent, "distractor_entity": dis}}
    raise ValueError(f"unknown variant {variant!r}")


def applicable(items, docs: dict, variant: str) -> list:
    """Items (of the given list) for which `variant` can be built (deterministic check with template 0)."""
    by: dict = {}
    for it in items:
        by.setdefault(_get(it, "doc_id"), []).append(it)
    rng = random.Random(0)
    return [it for it in items
            if make_train_item(it, variant, docs.get(_get(it, "doc_id"), ""), by[_get(it, "doc_id")], rng, 0) is not None]


__all__ = ["MT_VARIANTS", "make_train_item", "applicable"]
