"""D18 tests: deterministic context-dependent items (xenc/d18_context.py) and their chat segments.
CPU only; the rendering test uses the Qwen3.5 tokenizer if it is in the offline HF cache (skipped otherwise)."""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import xenc  # noqa: E402,F401
from xenc.d18_context import (build_context_items, common_suffix_len, entities, make_context_item,  # noqa: E402
                              messages, pick_entity, segments)
from xenc.doc_tasks import q_prompt  # noqa: E402

DOC = ("Diego met Jan at the airfield near Oostpoort. Later Jan flew to Rotterdam. Diego was sick, and Jan agreed to "
       "help him. The mechanic Pieter Vos repaired the plane.")
CH = [{"label": "A", "text": "Diego is sick."}, {"label": "B", "text": "Jan is Diego's father."}]
ITEMS = [{"key": "k1", "doc_id": "d", "question": "What is the relationship between Diego and Jan?", "choices": CH,
          "answers": ["Diego is sick."], "correct_label": "A", "task": "t", "hop": None, "meta": {}},
         {"key": "k2", "doc_id": "d", "question": "Where did Jan fly later?", "choices": CH, "answers": ["x"],
          "correct_label": "B", "task": "t", "hop": None, "meta": {}}]


def test_variants_content():
    o = make_context_item(ITEMS[0], "orig", DOC, ITEMS)
    assert o["turns"] == [] and o["final"] == q_prompt(ITEMS[0])
    d = make_context_item(ITEMS[0], "deferred", DOC, ITEMS)
    assert ITEMS[0]["question"] in d["turns"][0][0] and ITEMS[0]["question"] not in d["final"]
    assert "A. Diego is sick." in d["final"] and d["item"] is ITEMS[0]
    x = make_context_item(ITEMS[0], "distractor", DOC, ITEMS)
    u = x["turns"][0][0]
    assert ITEMS[0]["question"] in u and ITEMS[1]["question"] in u and ITEMS[0]["question"] not in x["final"]
    k = x["meta"]["target_pos"]
    assert f"Question {k}: {ITEMS[0]['question']}" in u and f"Answer Question {k} only" in x["final"]
    assert x["meta"]["distractor_key"] == "k2"
    assert make_context_item(ITEMS[0], "distractor", DOC, ITEMS[:1]) is None       # no other question
    r = make_context_item(ITEMS[0], "referent", DOC, ITEMS)
    ent = r["meta"]["entity"]
    assert ent in ("Diego", "Jan") and ent not in r["final"].split("\n")[0] and "X" in r["final"]
    assert f"{ent} as X" in r["turns"][0][0]
    assert r["meta"]["distractor_entity"] not in (None, ent)
    assert r["item"]["correct_label"] == "A"                         # gold unchanged


def test_entities_and_determinism():
    assert "Pieter Vos" in entities(DOC) and "The" not in entities("The cat sat.")
    assert entities("What did Mr. John Miller get?") == ["John Miller"]
    assert pick_entity({"question": "Why did Nobody come?"}, DOC) is None
    a = build_context_items(ITEMS, {"d": DOC}, "distractor")
    b = build_context_items(ITEMS, {"d": DOC}, "distractor")
    assert [(c["key"], c["turns"], c["final"]) for c in a] == [(c["key"], c["turns"], c["final"]) for c in b]
    m = messages(make_context_item(ITEMS[0], "deferred", DOC, ITEMS), "SYS", "DOCTEXT")
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user"]
    assert m[1]["content"].startswith("## Document\nDOCTEXT\n\n## Question\n")
    assert common_suffix_len([1, 2, 3, 4], [9, 3, 4]) == 2 and common_suffix_len([1], [2]) == 0


def _tok():
    try:
        from xenc.doc_tasks import load_qwen_tokenizer
        return load_qwen_tokenizer("Qwen/Qwen3.5-9B")
    except Exception:
        return None


def test_segments_real_tokenizer():
    tok = _tok()
    if tok is None:
        print("skip: tokenizer not cached")
        return
    from xenc.state_skill import doc_prefix_text, question_suffix_ids
    sys_ = "You answer."
    o = make_context_item(ITEMS[0], "orig", DOC, ITEMS)
    s = segments(tok, "Qwen/Qwen3.5-9B", o, sys_)
    ids = tok(s["conv_str"], add_special_tokens=False)["input_ids"]
    assert ids == question_suffix_ids(tok, "Qwen/Qwen3.5-9B", ITEMS[0], sys_)
    for var in ("deferred", "distractor", "referent"):
        ci = make_context_item(ITEMS[0], var, DOC, ITEMS)
        s = segments(tok, "Qwen/Qwen3.5-9B", ci, sys_)
        full = tok.apply_chat_template(messages(ci, sys_, DOC), tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        assert full == doc_prefix_text(tok, "Qwen/Qwen3.5-9B", DOC, sys_) + s["conv_str"]
        conv = tok(s["conv_str"], add_special_tokens=False)["input_ids"]
        fin = tok(s["enc_final"], add_special_tokens=False)["input_ids"]
        k = common_suffix_len(fin, conv)
        nf = len(tok(ci["final"], add_special_tokens=False)["input_ids"])
        assert k >= nf, (var, k, nf)                               # the whole final message is token-aligned
        assert s["conv_str"].count("<|im_start|>user") == 1 and "<|im_start|>assistant" in s["final_str"]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f()
            print("ok", n)
    print("ALL D18 CONTEXT TESTS PASSED")
