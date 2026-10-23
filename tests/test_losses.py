"""[D] xenc.losses / xenc.backbone (CPU, TinyLM). Run: python tests/test_losses.py"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tiny import TinyLM  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from xenc.backbone import build_chat, generate, short_name, with_target  # noqa: E402
from xenc.depth_skill import TinyTokenizer  # noqa: E402
from xenc.losses import (answer_ce, answer_span_mask, kl_teacher_student, magnitude_penalty,  # noqa: E402
                         route_aux_loss, student_logits, teacher_logits, trace_target)


def test_kl():
    torch.manual_seed(0)
    a = torch.randn(5, 50)
    assert kl_teacher_student(a, a).abs() < 1e-6
    b = torch.randn(5, 50)
    kl = kl_teacher_student(a, b)
    ref = F.kl_div(F.log_softmax(b, -1), F.softmax(a, -1), reduction="none").sum(-1).mean()
    assert kl > 0 and torch.allclose(kl, ref, atol=1e-5)
    mask = torch.tensor([1, 0, 0, 0, 0], dtype=torch.bool)
    ref0 = F.kl_div(F.log_softmax(b[:1], -1), F.softmax(a[:1], -1), reduction="sum")
    assert torch.allclose(kl_teacher_student(a, b, mask), ref0, atol=1e-5)


def test_answer_ce():
    torch.manual_seed(1)
    logits = torch.randn(4, 50)          # m+1 rows for m=3
    tgt = torch.tensor([3, 7, 9])
    ref = F.cross_entropy(logits[:-1], tgt)
    assert torch.allclose(answer_ce(logits, tgt), ref)
    assert torch.allclose(answer_ce(logits[:-1], tgt), ref)
    m = torch.tensor([False, False, True])
    assert torch.allclose(answer_ce(logits, tgt, m), F.cross_entropy(logits[2:3], tgt[2:]))
    try:
        answer_ce(torch.randn(7, 50), tgt)
        raise AssertionError("row mismatch must raise")
    except ValueError:
        pass


def test_teacher_student_tiny():
    lm = TinyLM(d=32, n_layers=8).requires_grad_(False)
    tok = TinyTokenizer()
    ids, m, P = with_target(tok, "tiny", "sys", "the user question here", "<answer>w1</answer>")
    assert ids.shape == (1, P + m) and m == 1
    t = teacher_logits(lm, ids, m)
    s = student_logits(lm, ids, m)
    assert t.shape == (m + 1, 50) and s.shape == (m + 1, 50)
    assert not t.requires_grad
    assert torch.allclose(t, s)
    x = torch.randn(1, 6, 32)            # float-embedding path
    assert teacher_logits(lm, x, 2).shape == (3, 50)
    g, n = generate(lm, tok, "tiny", build_chat(tok, "tiny", "s", "u v"), 3, return_n=True)
    assert n == 3 and isinstance(g, str)
    assert short_name("Qwen/Qwen3.5-9B") == "Qwen3.5-9B"


class _FakeInj:
    def __init__(self):
        self.mag_terms, self.route_terms, self.last_route = [], [], None


def test_mag_and_route_aux():
    inj = _FakeInj()
    assert float(magnitude_penalty(inj)) == 0.0
    inj.mag_terms = [torch.tensor(0.2), torch.tensor(0.4)]
    assert abs(float(magnitude_penalty(inj)) - 0.3) < 1e-6
    logit = torch.zeros(1, 3, 4, requires_grad=True)
    w = torch.softmax(logit, -1)
    inj.route_terms = [(5, w), (7, w)]
    aux = route_aux_loss(inj, [1])
    assert abs(float(aux.detach()) - math.log(4)) < 1e-5
    aux.backward()
    assert logit.grad is not None and logit.grad.abs().sum() > 0
    assert abs(float(route_aux_loss(inj, [0, 1]).detach()) - math.log(2)) < 1e-5
    assert route_aux_loss(inj, []) is None
    inj2 = _FakeInj()
    inj2.last_route = w.detach()        # detached fallback still reports the value
    assert abs(float(route_aux_loss(inj2, [3])) - math.log(4)) < 1e-5


def test_trace_target_and_span():
    g = "  Reasoning: the doc says X.\n<answer>B</answer> trailing junk <answer>C</answer>"
    assert trace_target(g) == "Reasoning: the doc says X.\n<answer>B</answer>"
    assert trace_target("no tags here ") == "no tags here"
    tok = TinyTokenizer()
    tr = "step one two <answer>B</answer>"
    n = len(tok.encode(tr))
    mk = answer_span_mask(tok, tr, n)
    assert mk.tolist() == [False, False, False, True]
    assert answer_span_mask(tok, "a b c", 3).all()


def test_student_grads_through_library_injector():
    from xenc.interface import make_interface
    from xenc.depth_skill import derive_by_depth
    from xenc.injector import LibraryInjector
    from xenc.library import Library
    from xenc import DEPTHS
    reader = TinyLM(d=32, n_layers=8, seed=0).requires_grad_(False)
    enc = TinyLM(d=24, n_layers=8, seed=1).requires_grad_(False)
    tok = TinyTokenizer()
    doc = " ".join(f"w{i}" for i in range(40))
    cache = derive_by_depth(enc, tok, doc, DEPTHS, chunk_tokens=16)
    lib = Library.from_doc_cache(cache, reader, DEPTHS)
    itf = make_interface(32, 24, sorted(lib.K))
    inj = LibraryInjector(reader, itf)
    inj.load_library(lib)
    ids, m, _ = with_target(tok, "tiny", "s", "question words", "<answer>w3</answer>")
    s = student_logits(reader, ids, m)
    loss = answer_ce(s, ids[0, -m:]) + route_aux_loss(inj, [1])
    loss.backward()
    assert itf.B.weight.grad is not None and itf.B.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in reader.parameters())
    inj.unload()


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_") and callable(fn):
            fn()
    print("OK")
