"""NOTES tests (xenc/notes.py): encoder-written note columns appended to the D13 operator.

CPU only, tiny random real Qwen3_5ForCausalLM (tests/_tiny.tiny_qwen35)."""

from __future__ import annotations

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402
from xenc.d13_read import D13Interface, EncTeacher  # noqa: E402
from xenc.notes import NoteStore, _pad_equal, generate_notes, make_note_rt  # noqa: E402
from xenc.state_skill import full_layers  # noqa: E402

torch.set_num_threads(2)


def _ids(n, seed):
    return torch.randint(3, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _rt(train=False, tmp=None, tail=4):
    from fit_d13 import D13Rt
    r, e = tiny_qwen35(seed=1), tiny_qwen35(seed=2)
    L = full_layers(e)
    itf = D13Interface(e, {li: li for li in full_layers(r)}, 64, 64, B=None)
    NoteRt = make_note_rt(D13Rt)
    rt = NoteRt(r, e, None, "tiny", itf, tail=tail, max_tokens=10_000)
    rt._pre = {"d0": _ids(30, 0), "d1": _ids(25, 1)}
    rt.docs = {"d0": "x", "d1": "y"}
    suffix = {"a": _ids(5, 7), "b": _ids(8, 8)}
    rt.setup_notes(NoteStore(tmp or tempfile.mkdtemp()), suffix, max_new=6, stop_ids={-1}, pad_id=3,
                   gen_types=["a", "b"], train=train)
    return rt, e, L


def _full_kv(e, L, ids):
    Tc = EncTeacher(e, L)
    Tc.rec, Tc.kv_only = {}, True
    with torch.no_grad():
        e(input_ids=torch.tensor([ids]), use_cache=False)
    out = {k: (v["k"], v["v"]) for k, v in Tc.rec.items()}
    Tc.rec = None
    Tc.detach()
    return out


def test_notes_off_is_plain_d13rt():
    from fit_d13 import D13Rt
    rt, e, L = _rt()
    rt.note = None
    op, n, _ = rt.operator("d0")
    op2, n2, _ = D13Rt.operator(rt, "d0")
    assert n == n2 == 26 and all(torch.equal(op[k][0], op2[k][0]) and torch.equal(op[k][1], op2[k][1]) for k in op)


def test_note_columns_equal_full_prefill_and_slice_by_length():
    rt, e, L = _rt()
    for Ln in (0, 3, 6):
        rt.note = {"type": "b", "L": Ln, "src": None}
        op, n, _ = rt.operator("d0")
        r = rt.store.get("b", "d0")
        seq = rt._pre["d0"][:-4] + r["suffix"] + r["gen"][:Ln]
        assert n == len(seq) and len(r["suffix"]) == 8 and len(r["gen"]) == 6
        ref = _full_kv(e, L, seq)
        for k in op:
            assert op[k][0].shape[1] == n
            assert torch.allclose(op[k][0], ref[k][0], atol=1e-4) and torch.allclose(op[k][1], ref[k][1], atol=1e-4)
    # padded suffix: type "a" (5 tokens) got 3 pad tokens prepended to match "b" (8)
    assert rt.store.get("a", "d0")["suffix"] == [3, 3, 3] + rt.suffix["a"]


def test_wrong_notes_are_the_other_docs_ids_in_own_context():
    rt, e, L = _rt()
    rt.note = {"type": "a", "L": 6, "src": "d1"}
    op, n, _ = rt.operator("d0")
    r1 = rt.store.get("a", "d1")
    seq = rt._pre["d0"][:-4] + r1["suffix"] + r1["gen"]
    ref = _full_kv(e, L, seq)
    assert n == len(seq)
    assert all(torch.allclose(op[k][0], ref[k][0], atol=1e-4) for k in op)


def test_train_mode_teacher_continues_after_notes():
    """Train mode keeps the encoder cache after [doc ; notes]: teacher rows == a full pass over doc + notes + rows."""
    rt, e, L = _rt(train=True)
    rt.note = {"type": "b", "L": 4, "src": None}
    op, n, cache = rt.operator("d0")
    r = rt.store.get("b", "d0")
    rows = _ids(5, 9)
    rec = rt.teacher("d0", rows)
    seq = rt._pre["d0"][:-4] + r["suffix"] + r["gen"][:4]
    assert n == len(seq)
    Tc = rt.Tc
    Tc.rec, Tc.kv_only = {}, False
    with torch.no_grad():
        e(input_ids=torch.tensor([seq + rows]), use_cache=False)
    full = dict(Tc.rec)
    Tc.rec = None
    for k in rec:
        assert torch.allclose(rec[k]["out"], full[k]["out"][-5:], atol=1e-4)
        assert torch.allclose(rec[k]["q"], full[k]["q"][:, -5:], atol=1e-4)


def test_batched_generation_equals_single_stream():
    e = tiny_qwen35(seed=2)
    doc = _ids(20, 0)
    with torch.no_grad():
        cache = e(input_ids=torch.tensor([doc]), use_cache=True).past_key_values
    sufs = _pad_equal([_ids(3, 4), _ids(6, 5)], 3)
    both = generate_notes(e, cache, sufs, 7, set(), 3)
    for s, o in zip(sufs, both):
        one = generate_notes(e, cache, [s], 7, set(), 3)[0]
        assert one["gen"] == o["gen"] and len(o["gen"]) == 7
    with torch.no_grad():                       # the cache passed in is never modified
        a = e(input_ids=torch.tensor([[5]]), past_key_values=__import__("copy").deepcopy(cache)).logits
        b = e(input_ids=torch.tensor([doc + [5]])).logits[:, -1:]
    assert torch.allclose(a, b, atol=1e-4)


if __name__ == "__main__":
    for f in [v for k, v in list(globals().items()) if k.startswith("test_")]:
        f()
    print("test_notes: all passed")
