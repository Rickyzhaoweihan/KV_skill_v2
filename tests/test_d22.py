"""D22 tests: span self-question template/parser, file-based doc selection, numeric scorer, Qasper split sanity. CPU."""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))


def test_span_parse_and_install():
    import d20_selfq
    import d22_selfq
    d22_selfq.install()
    assert "span" in d20_selfq.TEMPLATES and "{S}" in d20_selfq.TEMPLATES["span"]
    q = d20_selfq.parse("span", "Q1: How many dialogs were annotated?\nA1: 100 dialogs\nQ2: Who annotated the corpus?\n"
                                "A2: \"two expert annotators\"\nQ3: When?\nA3: 2019")
    assert [x["answers"][0] for x in q] == ["100 dialogs", "two expert annotators"]   # "When?" too short
    assert all(x["choices"] is None for x in q)
    assert len(d20_selfq.parse("mcq", "Q1: Which option is right here?\nA) a\nB) b\nC) c\nD) d\nA1: B")) == 1


def test_numeric_match():
    from d22_summary import num_match

    class It:
        answers = ["3,044"]
    assert num_match(It, "<answer>3044 sentences</answer>") == 1.0
    assert num_match(It, "<answer>3045</answer>") == 0.0


def test_qasper_split_disjoint():
    sd = os.path.join(_ROOT, "data", "qasper_split")
    if not os.path.exists(sd):
        return
    docs = {s: {i["doc_id"] for i in json.load(open(os.path.join(sd, s, "items.json")))} for s in ("train", "val", "test")}
    assert not (docs["train"] & docs["test"]) and not (docs["val"] & docs["test"]) and not (docs["train"] & docs["val"])


if __name__ == "__main__":
    for nm, f in list(globals().items()):
        if nm.startswith("test_") and callable(f):
            f()
            print("ok", nm)
    print("ALL D22 TESTS PASSED")
