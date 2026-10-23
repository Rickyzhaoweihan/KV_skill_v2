"""D18 direction A tests: multi-turn training items (xenc/d18_mt.py). CPU only, no models."""

from __future__ import annotations

import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import xenc  # noqa: E402,F401
from xenc.d18_context import make_context_item  # noqa: E402
from xenc.d18_mt import MT_VARIANTS, applicable, make_train_item  # noqa: E402

DOC = ("Diego met Jan at the airfield near Oostpoort. Later Jan flew to Rotterdam. Diego was sick, and Jan agreed to "
       "help him. The mechanic Pieter Vos repaired the plane.")
CH = [{"label": "A", "text": "Diego is sick."}, {"label": "B", "text": "Jan is Diego's father."}]
ITEMS = [{"key": "k1", "doc_id": "d", "question": "What is the relationship between Diego and Jan?", "choices": CH,
          "answers": ["Diego is sick."], "correct_label": "A", "task": "t", "hop": None, "meta": {}},
         {"key": "k2", "doc_id": "d", "question": "Where did Jan fly later?", "choices": CH, "answers": ["x"],
          "correct_label": "B", "task": "t", "hop": None, "meta": {}}]
SPAN = {"key": "s1", "doc_id": "d", "question": "Who repaired the plane for Jan?", "choices": None,
        "answers": ["Pieter Vos"], "correct_label": None, "task": "t", "hop": None, "meta": {}}


def _strip(ci):
    return {k: v for k, v in ci.items() if k != "meta"}, {k: v for k, v in ci["meta"].items() if k != "template"}


def test_template0_is_the_judge_wording():
    for it, same in ((ITEMS[0], ITEMS), (SPAN, [SPAN])):
        for v in MT_VARIANTS:
            j = make_context_item(it, v, DOC, same)
            t = make_train_item(it, v, DOC, same, random.Random(5), template=0)
            if j is None:
                assert t is None, v
                continue
            assert _strip(t) == _strip(j), (v, t, j)
    print("ok test_template0_is_the_judge_wording")


def test_paraphrases_keep_content_and_gold():
    rng = random.Random(0)
    seen = {v: set() for v in MT_VARIANTS}
    for _ in range(60):
        for v in MT_VARIANTS:
            t = make_train_item(ITEMS[0], v, DOC, ITEMS, rng)
            assert t is not None and t["item"] is ITEMS[0]
            seen[v].add(t["turns"][0][0] + t["final"])
            if v == "deferred":
                assert ITEMS[0]["question"] in t["turns"][0][0] and ITEMS[0]["question"] not in t["final"]
                assert "A. Diego is sick." in t["final"]
            if v == "distractor":
                k = t["meta"]["target_pos"]
                assert f"Question {k}: {ITEMS[0]['question']}" in t["turns"][0][0] and f"Question {k}" in t["final"]
            if v == "referent":
                assert "X" in t["final"] and t["meta"]["entity"] not in t["final"]
                assert t["meta"]["entity"] in t["turns"][0][0]
    assert all(len(s) >= 3 for s in seen.values()), {v: len(s) for v, s in seen.items()}
    assert len(applicable(ITEMS + [SPAN], {"d": DOC}, "distractor")) == 3      # every item has a sibling question
    assert applicable([SPAN], {"d": DOC}, "distractor") == []                    # a lone question has none
    print("ok test_paraphrases_keep_content_and_gold")


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f()
    print("ALL D18 MT TESTS PASSED")
