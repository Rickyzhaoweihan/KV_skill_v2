"""D20 tests: per-document registration adapter (xenc/d20_reg.py) and the leakage guard. CPU, tiny Qwen3.5."""

from __future__ import annotations

import copy
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402
from xenc.d13_read import D13Interface, D13Reader, EncTeacher  # noqa: E402
from xenc.d20_reg import DocAdapter, assert_no_test_questions, drop_test_matches  # noqa: E402

torch.set_num_threads(2)


def _ids(n, seed):
    return torch.randint(3, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _setup():
    from test_state_interface import _tiny_encoder
    r, e = tiny_qwen35(), _tiny_encoder()
    lmap = {1: 2, 5: 5}
    B = {li: torch.randn(64, 48) * 0.1 for li in lmap}
    itf = D13Interface(e, lmap, 64, 48, B=B, mlp_hidden=8)
    Tc = EncTeacher(e, sorted(set(lmap.values())))
    Tc.rec, Tc.kv_only = {}, True
    with torch.no_grad():
        e(input_ids=torch.tensor([_ids(25, 0)]), use_cache=True)
    op = {k: (v["k"], v["v"]) for k, v in Tc.rec.items()}
    Tc.rec = None
    Tc.detach()
    return r, itf, op


def _logits(r, R, op, ad, rows):
    R.load(ad.transform_op(op), 25)
    R.mode = "e2e"
    try:
        return r(input_ids=torch.tensor([rows])).logits
    finally:
        R.mode = "off"


def test_adapter_starts_at_base_trains_resets_and_snapshots():
    r, itf, op = _setup()
    R = D13Reader(r, itf)
    rows = _ids(6, 1)
    with torch.no_grad():
        R.load(op, 25)
        R.mode = "e2e"
        base = r(input_ids=torch.tensor([rows])).logits
        R.mode = "off"
    for parts in (["dA", "gate"], ["dA", "dB", "bias", "gate"], ["kv"]):
        itf2 = copy.deepcopy(itf)
        R.itf = itf2
        ad = DocAdapter(itf2, parts, rank=4, kv_rank=2)
        assert ad.n_params() > 0
        with torch.no_grad():
            y0 = _logits(r, R, op, ad, rows)
        assert torch.allclose(y0, base, atol=1e-5), parts            # zero-init == base interface
        opt = torch.optim.Adam(ad.param_groups(1e-2, 1e-2))
        for _ in range(3):
            opt.zero_grad()
            _logits(r, R, op, ad, rows).float().logsumexp(-1).mean().backward()
            opt.step()
        # the base interface stays frozen
        assert itf2.A["1"].weight.grad is None and not itf2.A["1"].weight.requires_grad
        with torch.no_grad():
            y1 = _logits(r, R, op, ad, rows)
        assert float((y1 - base).abs().max()) > 1e-5, parts
        snap = ad.snapshot()
        ad.reset()
        with torch.no_grad():
            assert torch.allclose(_logits(r, R, op, ad, rows), base, atol=1e-5)
            ad.load(snap)
            assert torch.allclose(_logits(r, R, op, ad, rows), y1, atol=1e-5)
    R.itf = itf
    R.detach()


def test_kv_adapter_leaves_operator_untouched():
    r, itf, op = _setup()
    ad = DocAdapter(itf, ["kv"], kv_rank=2)
    ref = {e: (k.clone(), v.clone()) for e, (k, v) in op.items()}
    with torch.no_grad():
        for p in ad.kv.values():
            p.normal_(0, 0.1)
    op2 = ad.transform_op(op)
    assert any(float((op2[e][0] - op[e][0]).abs().max()) > 0 for e in op)
    for e in op:
        assert torch.equal(op[e][0], ref[e][0]) and torch.equal(op[e][1], ref[e][1])


def test_leakage_guard():
    test_qs = ["What is the name of the ship?",
               "Consider the following hypothesis about this non-disclosure agreement:\n\"Receiving Party shall destroy "
               "or return some Confidential Information upon the termination of Agreement.\"\nWhat does the agreement say?"]
    ok = {"key": "d_d20s0", "doc_id": "d", "question": "Who repaired the engine in part two?", "meta": {"d20selfq": True}}
    assert assert_no_test_questions([ok], test_qs) == 1
    for bad_q in ("what is the name of the ship", "Is it true? Receiving party shall destroy or return some confidential "
                                                   "information upon the termination of agreement."):
        bad = dict(ok, question=bad_q)
        try:
            assert_no_test_questions([bad], test_qs)
            raise RuntimeError("leak not caught")
        except AssertionError:
            pass
        kept, nd = drop_test_matches([dict(bad), dict(ok)], test_qs)
        assert nd == 1 and kept[0]["question"] == ok["question"]
    try:                                                   # a dataset item (not a D20 self-question) is refused
        assert_no_test_questions([dict(ok, meta={})], test_qs)
        raise RuntimeError("dataset item not caught")
    except AssertionError:
        pass


def test_selfq_parsers():
    from d20_selfq import parse
    t = ("Q1: Who left the ship first?\nA) Ann\nB) Bob\nC) Cy\nD) Di\nA1: B\nQ2: Why did the ship stop?\nA) fuel\nB) storm\nC) war\n"
         "D) rest\nA2: A\nQ3: Where?\nA) x\nB) y\nC) z\nD) w\nA3: D")
    q = parse("mcq", t)
    assert len(q) == 2 and q[0]["correct_label"] == "B" and len(q[0]["choices"]) == 4   # "Where?" too short
    n = parse("nli", "S1: The captain was born in Ohio in 1950.\nS2: The ship sank on its first voyage.\n"
                     "S3: The crew was paid in gold coins.")
    assert [x["correct_label"] for x in n] == ["A", "B", "C"] and len(n[0]["choices"]) == 3


if __name__ == "__main__":
    for nm, f in list(globals().items()):
        if nm.startswith("test_") and callable(f):
            f()
            print("ok", nm)
    print("ALL D20 TESTS PASSED")
