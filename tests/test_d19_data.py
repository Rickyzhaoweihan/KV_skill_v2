"""[D19] tests for the D19 dataset converters and their produced splits (CPU, no tokenizer needed).

  python tests/test_d19_data.py     # unit tests + every D19 split present on disk
  pytest tests/test_d19_data.py     # same; data checks print SKIP when a split is absent (XENC_QUICK=1 skips them)
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "scripts" / "data"))

import test_data as TD  # noqa: E402  (shared split checks: _have, _check_common, _items)
import prepare_shakespeare as PS  # noqa: E402
from prepare_loogle import _mcq  # noqa: E402

D19 = ["shakespeare", "shakespeare_cf", "contractnli", "loogle", "clipper"]


# --------------------------------------------------------------------------- unit tests (no data)

def test_derangement_and_rename():
    names = ["HAMLET", "HORATIO", "OPHELIA", "LAERTES", "POLONIUS"]
    for seed in range(20):
        s = PS.derangement(names, random.Random(seed))
        assert sorted(s) == sorted(s.values()) == sorted(names) and all(a != b for a, b in s.items())
    s = {"HAMLET": "HORATIO", "HORATIO": "HAMLET"}
    txt = "HAMLET.\nHoratio, thou art e'en as just a man.\n\nHORATIO.\nO my dear lord, Hamlet's Hamletian.\n"
    out = PS.rename(txt, s)
    assert out == "HORATIO.\nHamlet, thou art e'en as just a man.\n\nHAMLET.\nO my dear lord, Horatio's Hamletian.\n"


def test_parse_speeches():
    play = ("Dramatis\n\nHAMLET, Prince\n\nACT I\n\nSCENE I. A platform.\n\nBARNARDO.\nWho’s there?\n\n"
            "FRANCISCO.\nNay, answer me. [_Aside._] Stand and unfold yourself.\n\nACT II\n\nHAMLET.\nWords, words.\n")
    sp = PS.parse_speeches(play)
    assert [(x["speaker"], x["act"]) for x in sp] == [("BARNARDO", "I"), ("FRANCISCO", "I"), ("HAMLET", "II")]
    assert sp[1]["text"] == "Nay, answer me. Stand and unfold yourself."


def test_loogle_mcq():
    q = "How many people were in Picardo's family when he was twelve?\n1. seven\n2. six\n3. five\n4. four"
    stem, opts, gi = _mcq(q, "3")
    assert stem.endswith("twelve?") and opts == ["seven", "six", "five", "four"] and gi == 2
    assert _mcq(q, "five people") is None and _mcq("Order these:\n1.a\n2.b", "2,1") is None


def test_wrong_map_skips_near_duplicates():
    sys.path.insert(0, str(REPO / "scripts"))
    from d19_wrong_map import wrong_map
    order = ["a", "b", "c", "d"]
    sh = {"a": set(range(100)), "b": set(range(100)), "c": set(range(200, 300)), "d": set(range(400, 500))}
    m = wrong_map(order, sh, shift=1, max_frac=0.15)
    assert m == {"a": "c", "b": "c", "c": "d", "d": "a"}, m      # a -> b would be a near-duplicate: stepped on


def test_aggregate_adherence():
    sys.path.insert(0, str(REPO / "scripts"))
    from d19_eval_aggregate import aggregate
    ch = [{"label": L, "text": t} for L, t in zip("ABCD", ("W", "X", "Y", "Z"))]
    items = [{"key": f"k{i}", "doc_id": "d", "question": "q", "choices": ch, "answers": ["X"], "correct_label": "B",
              "task": "shakespeare_cf", "hop": None,
              "meta": {"kind": "speaker", "conflict": True, "prior_label": "A"}} for i in range(4)]
    gens = {"text": ["<answer>B</answer>"] * 3 + ["<answer>A</answer>"],
            "kv": ["<answer>A</answer>"] * 2 + ["<answer>B</answer>", "<answer>C</answer>"]}
    ev = {"items": [{"key": f"k{i}", "doc_id": "d", **{c: {"gen": g[i], "score": float("B" in g[i]), "gen_tokens": 3}
                                                     for c, g in gens.items()}} for i in range(4)]}
    md = "\n".join(aggregate(ev, items, "shakespeare_cf"))
    assert "| text | 0.750±0.250 | 0.250±0.250 | 0.000±0.000 | 4 |" in md, md
    assert "| kv | 0.250±0.250 | 0.500±0.289 | 0.250±0.250 | 4 |" in md, md


# --------------------------------------------------------------------------- data tests (need prepared splits)

def _disjoint(its):
    docs = {s: {it["doc_id"] for it in v} for s, v in its.items()}
    assert not docs["train"] & docs["val"] and not docs["train"] & docs["test"] and not docs["val"] & docs["test"]
    return docs


def test_d19_common():
    for name in D19:
        if not TD._have(name):
            print(f"SKIP {name} (quick mode or split absent)")
            continue
        its, _ = TD._check_common(name)
        _disjoint(its)


def test_shakespeare_pairing():
    if not (TD._have("shakespeare") and TD._have("shakespeare_cf")):
        return print("SKIP shakespeare pairing")
    o, c = TD._items("shakespeare"), TD._items("shakespeare_cf")
    for s in o:
        assert len(o[s]) == len(c[s]) > 0
        for a, b in zip(o[s], c[s]):
            assert a["choices"] == b["choices"] and a["meta"]["kind"] == b["meta"]["kind"]
            assert a["doc_id"].replace("shakespeare_", "shakespeare_cf_", 1) == b["doc_id"]
            if a["meta"]["kind"] == "speaker":
                assert b["meta"]["conflict"] and b["meta"]["prior_label"] == a["correct_label"] != b["correct_label"]
                assert len(a["choices"]) == 4
            else:
                assert a["correct_label"] == b["correct_label"] and len(a["choices"]) == 5


def test_contractnli_official_split():
    if not TD._have("contractnli"):
        return print("SKIP contractnli")
    its = TD._items("contractnli")
    docs = _disjoint(its)
    assert {s: len(v) for s, v in docs.items()} == {"train": 423, "val": 61, "test": 123}
    for v in its.values():
        assert set(Counter(it["doc_id"] for it in v).values()) == {17}
        assert all(len(it["choices"]) == 3 for it in v)


def test_clipper_pairs():
    if not TD._have("clipper"):
        return print("SKIP clipper")
    for v in TD._items("clipper").values():
        per = {}
        for it in v:
            per.setdefault(it["meta"]["pair_id"], []).append(it["correct_label"])
        assert all(sorted(x) == ["A", "B"] for x in per.values())


def main():
    if "--quick" in sys.argv:
        TD.QUICK = True
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__}: ok")
    print("OK")


if __name__ == "__main__":
    main()
