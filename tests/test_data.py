"""[C] tests for xenc.doc_tasks and the produced data splits.

  python tests/test_data.py            # unit tests + every produced split
  python tests/test_data.py --quick    # unit tests only (no data / tokenizer needed)
  pytest tests/test_data.py            # same as the full run; data checks no-op if splits are absent
                                       # (set XENC_QUICK=1 to force quick mode under pytest)
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from xenc.doc_tasks import (  # noqa: E402
    DocItem, chunk_document, ctx_prompt, exact, gold_text, load_doc_split, load_docs_index, q_prompt,
    score, validate_item,
)

DATA = REPO / "data"
QUICK = os.environ.get("XENC_QUICK") == "1"
NAMES = ["quality", "longhealth", "lastingbench", "musique"]


def _split_dir(name):
    return DATA / f"{name}_split"


def _have(name):
    return not QUICK and all((_split_dir(name) / s / "items.json").exists() for s in ("train", "val", "test"))


def _items(name):
    return {s: json.load(open(_split_dir(name) / s / "items.json")) for s in ("train", "val", "test")}


def _near(x, target, tol=0.10):
    return abs(x - target) <= tol * target


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


# --------------------------------------------------------------------------- unit tests (no data)

MCQ = DocItem(key="k", doc_id="d", question="Who?", answers=["Bob"], correct_label="B", task="quality",
              choices=[{"label": "A", "text": "Al"}, {"label": "B", "text": "Bob"},
                       {"label": "C", "text": "Cy"}, {"label": "D", "text": "Di"}], hop=None, meta={})
SPAN = DocItem(key="s", doc_id="d", question="Where?", answers=["New York City", "NYC"], correct_label=None,
               choices=None, task="musique", hop=2, meta={})


def test_score_mcq():
    assert score(MCQ, "<answer>B</answer>") == 1.0
    assert score(MCQ, "reasoning... <answer> b. </answer>") == 1.0
    assert score(MCQ, "<answer>B) Bob</answer>") == 1.0
    assert score(MCQ, "<answer>A</answer>") == 0.0
    assert score(MCQ, "I think A at first, but the answer is B") == 1.0  # fallback: last standalone letter
    assert score(MCQ, "The answer is C.") == 0.0
    assert score(MCQ, "no idea") == 0.0
    assert score(MCQ, "<answer>A</answer> then <answer>B</answer>") == 1.0  # last tag wins
    assert score(MCQ, "<answer>E</answer>") == 0.0  # E not a valid label for a 4-way item
    dup = DocItem(**{**MCQ.to_dict(), "meta": {"accept_labels": ["B", "D"]}})
    assert score(dup, "<answer>D</answer>") == 1.0 and score(dup, "<answer>A</answer>") == 0.0
    assert score(MCQ.to_dict(), "<answer>B</answer>") == 1.0  # dicts accepted too
    assert gold_text(MCQ) == "<answer>B</answer>"


def test_score_span():
    assert score(SPAN, "<answer>New York City</answer>") == 1.0
    assert score(SPAN, "<answer>the nyc</answer>") == 1.0
    f = score(SPAN, "<answer>New York</answer>")  # P=1, R=2/3 -> 0.8
    assert abs(f - 0.8) < 1e-9, f
    assert score(SPAN, "Some reasoning.\nNew York City") == 1.0  # no tags: last line
    assert score(SPAN, "<answer>Boston</answer>") == 0.0
    assert score(SPAN, "") == 0.0
    assert exact(SPAN, "<answer>New York City.</answer>") == 1.0
    assert exact(SPAN, "<answer>New York</answer>") == 0.0
    assert gold_text(SPAN) == "<answer>New York City</answer>"


def test_prompts():
    q = q_prompt(MCQ)
    assert "Who?" in q and "B. Bob" in q and "<answer>" in q
    c = ctx_prompt(MCQ, "DOCTEXT " * 10, max_chars=7)
    assert c.startswith("## Document\nDOCTEXT\n\n## Question\n") and q in c
    assert "## Document" not in q_prompt(SPAN)


class WSTok:
    """Whitespace fake tokenizer: one token per maximal non-space run (offsets exclude spaces)."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kw):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        out = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


def _check_chunks(tok, text, ct):
    ch = chunk_document(tok, text, ct)
    n = len(tok(text, add_special_tokens=False, return_offsets_mapping=True)["input_ids"])
    assert "".join(c[2] for c in ch) == text, "chunks do not reconstruct the text"
    if not text:
        assert ch == []
        return ch
    assert ch[0][0] == 0 and ch[-1][1] == n
    for a, b in zip(ch, ch[1:]):
        assert a[1] == b[0], "token gap/overlap"
    assert all(0 < e - s <= ct for s, e, _ in ch) or n == 0
    return ch


def test_chunk_document_fake():
    tok = WSTok()
    rng = random.Random(0)
    words = ["alpha", "b", "gamma-delta", "x" * 12, "\n", "  ", "end."]
    for trial in range(200):
        text = "".join(rng.choice(words) + rng.choice([" ", "", "\n", "   "]) for _ in range(rng.randint(0, 80)))
        if rng.random() < 0.3:
            text = "  " + text + "  \n"
        for ct in (1, 3, 7, 512):
            _check_chunks(tok, text, ct)
    ch = _check_chunks(tok, "a b c d e f g", 3)
    assert [(s, e) for s, e, _ in ch] == [(0, 3), (3, 6), (6, 7)]
    assert [t for *_, t in ch] == ["a b c ", "d e f ", "g"]
    assert chunk_document(tok, "", 4) == []


# --------------------------------------------------------------------------- data tests (need prepared splits)


def _check_common(name):
    its = _items(name)
    index = load_docs_index(str(_split_dir(name)))
    for s, items in its.items():
        assert items, f"{name}/{s} empty"
        keys = [it["key"] for it in items]
        assert len(set(keys)) == len(keys), f"{name}/{s} duplicate keys"
        for it in items:
            validate_item(it)
            assert it["task"] == name
            assert it["doc_id"] in index, it["doc_id"]
        assert len(load_doc_split(str(_split_dir(name)), s)) == len(items)
    used = {it["doc_id"] for v in its.values() for it in v}
    assert used == set(index), f"{name}: docs.json and items disagree"
    for did, m in index.items():
        p = DATA / m["path"]
        assert p.exists() and p.stat().st_size > 0, p
        assert isinstance(m["n_tokens"], int) and m["n_tokens"] > 0 and isinstance(m["title"], str)
    return its, index


def test_quality():
    if not _have("quality"):
        return print("SKIP quality (quick mode or split absent)")
    its, index = _check_common("quality")
    for s, items in its.items():
        per = Counter(it["doc_id"] for it in items)
        assert min(per.values()) >= 10, (s, min(per.values()))
        assert all(len(it["choices"]) == 4 for it in items)
    docs = {s: {it["doc_id"] for it in v} for s, v in its.items()}
    assert not docs["val"] & docs["test"] and not docs["train"] & (docs["val"] | docs["test"])
    assert len(docs["train"]) == 150 and len(docs["val"]) + len(docs["test"]) == 115
    assert abs(len(docs["val"]) - len(docs["test"])) <= 1
    assert len(its["train"]) == 2523 and len(its["val"]) + len(its["test"]) == 2086
    med = _median([m["n_tokens"] for m in index.values()])
    assert _near(med, 6700), med


def test_longhealth():
    if not _have("longhealth"):
        return print("SKIP longhealth (quick mode or split absent)")
    its, index = _check_common("longhealth")
    assert len(index) == 20
    for s, items in its.items():
        assert len(items) == 400 and all(len(it["choices"]) == 5 for it in items)
        assert Counter(it["doc_id"] for it in items) == Counter({d: 20 for d in index})
    folds = json.load(open(_split_dir("longhealth") / "folds.json"))["folds"]
    assert len(folds) == 5 and all(len(f) == 4 for f in folds)
    assert sorted(d for f in folds for d in f) == sorted(index)
    fold_of = {d: i for i, f in enumerate(folds) for d in f}
    assert all(it["meta"]["fold"] == fold_of[it["doc_id"]] for it in its["test"])
    med = _median([m["n_tokens"] for m in index.values()])
    assert _near(med, 11700), med


LB_REWRITTEN = {"hotpotqa": (141, 200), "2wikimqa": (169, 200), "musique": (122, 200), "multifieldqa_en": (116, 150)}
LB_CF_MED = {"hotpotqa": 14976, "2wikimqa": 6439, "musique": 16716, "multifieldqa_en": 7315}


def test_lastingbench():
    if not _have("lastingbench"):
        return print("SKIP lastingbench (quick mode or split absent)")
    its, index = _check_common("lastingbench")
    split_of, by_pair = {}, defaultdict(dict)
    for s, items in its.items():
        for it in items:
            m = it["meta"]
            split_of[it["key"]] = s
            assert it["choices"] is None and isinstance(m["counterfactual"], bool) and isinstance(m["rewritten"], bool)
            by_pair[m["pair_key"]]["cf" if m["counterfactual"] else "orig"] = (s, it)
    rw, tot = Counter(), Counter()
    for pk, d in by_pair.items():
        assert set(d) == {"orig", "cf"}, pk
        (so, o), (sc, c) = d["orig"], d["cf"]
        assert so == sc, f"twins split apart: {pk}"
        assert c["doc_id"] == o["doc_id"] + "_cf" and o["question"] == c["question"]
        assert not o["meta"]["rewritten"] and c["meta"]["answers_original"] == o["answers"]
        sub = o["meta"]["subset"]
        tot[sub] += 1
        rw[sub] += c["meta"]["rewritten"]
        if not c["meta"]["rewritten"]:
            assert index[c["doc_id"]]["n_tokens"] == index[o["doc_id"]]["n_tokens"]
    for sub, (r, n) in LB_REWRITTEN.items():
        assert (rw[sub], tot[sub]) == (r, n), (sub, rw[sub], tot[sub])
        med = _median([index[o["doc_id"] + "_cf"]["n_tokens"] for d in by_pair.values()
                       for _, o in [d["orig"]] if o["meta"]["subset"] == sub])
        assert _near(med, LB_CF_MED[sub]), (sub, med)
    n = {s: len(v) // 2 for s, v in its.items()}
    assert sum(n.values()) == 750 and _near(n["train"], 300, 0.02) and _near(n["test"], 375, 0.02), n


def test_musique():
    if not _have("musique"):
        return print("SKIP musique (quick mode or split absent)")
    its, index = _check_common("musique")
    assert sum(len(v) for v in its.values()) == 2417
    hops = Counter()
    for items in its.values():
        for it in items:
            m = it["meta"]
            assert 17 <= m["n_paragraphs"] <= 20 and len(m["para_spans"]) == m["n_paragraphs"]
            assert m["supporting_idx"] and all(0 <= i < m["n_paragraphs"] for i in m["supporting_idx"])
            assert it["hop"] in (2, 3, 4) and str(it["hop"]) == m["hop_type"][0]
            assert len(m["supporting_idx"]) == it["hop"], (it["key"], m["supporting_idx"])
            hops[it["hop"]] += 1
    assert hops == Counter({2: 1252, 3: 760, 4: 405}), hops
    for it in random.Random(0).sample(its["test"], 30):  # marker/span consistency on a sample
        text = (DATA / index[it["doc_id"]]["path"]).read_text()
        for i, (a, b) in enumerate(it["meta"]["para_spans"]):
            assert text[a:b].startswith(f"[P{i}] ")
    med = _median([m["n_tokens"] for m in index.values()])
    assert _near(med, 2322 * 1.08), med  # paragraph-only median 2,322 + ~8% for [P{i}] title headers


def test_chunk_document_qwen():
    """chunk_document on a real document with the real Qwen3.5 tokenizer (skipped if unavailable)."""
    if not _have("quality"):
        return print("SKIP chunk/qwen (quick mode or split absent)")
    try:
        from xenc.doc_tasks import load_qwen_tokenizer
        tok = load_qwen_tokenizer()
    except Exception as e:  # noqa: BLE001
        return print(f"SKIP chunk/qwen ({type(e).__name__})")
    index = load_docs_index(str(_split_dir("quality")))
    did = sorted(index)[0]
    text = (DATA / index[did]["path"]).read_text()
    ch = _check_chunks(tok, text, 512)
    assert ch[-1][1] == index[did]["n_tokens"]


def main():
    global QUICK
    if "--quick" in sys.argv:
        QUICK = True
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  {t.__name__}: ok")
    print("OK")


if __name__ == "__main__":
    main()
