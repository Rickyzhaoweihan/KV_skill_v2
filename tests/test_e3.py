"""E3 tests: retrieval-aligned read signals (xenc/e3_retrieval.py). CPU only, tiny random Qwen3_5ForCausalLM."""

from __future__ import annotations

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402
from xenc.e3_retrieval import (QKCap, chars_to_mask, dilate, kl_rows, mixture_logp, retrieval_metrics,  # noqa: E402
                               sharpen, span_nll)

torch.set_num_threads(2)


def test_qkcap_matches_eager_attention():
    m = tiny_qwen35()
    ids = torch.randint(3, 100, (1, 23), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = m(input_ids=ids, output_attentions=True, use_cache=False).attentions
    C = QKCap(m, [1, 3])
    C.keep = 5
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=False)
    C.keep = None
    att = [a for a in ref if a is not None]          # transformers returns one entry per FULL-attention layer
    assert len(att) == 4, len(att)
    for j, e in enumerate((1, 3)):
        p = torch.softmax(C.logits(e), -1)
        assert torch.allclose(p, att[j][0, :, -5:, :].float(), atol=1e-5), (e, (p - att[j][0, :, -5:]).abs().max())
    C.detach()
    with torch.no_grad():
        out2 = m(input_ids=ids, use_cache=False)
    assert torch.allclose(out.logits, out2.logits)


def test_mixture_logp_is_normalised_head_mixture():
    g = torch.Generator().manual_seed(1)
    lg = torch.randn(2, 3, 4, 20, generator=g)
    lg[..., -1] = float("-inf")
    lp = mixture_logp(lg, [1, 3], 2, 15)
    assert lp.shape == (2, 13)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(2), atol=1e-5)
    p = torch.softmax(lg.reshape(6, 4, 20), -1)[:, [1, 3], 2:15].mean(0)
    assert torch.allclose(lp.exp(), p / p.sum(-1, keepdim=True), atol=1e-5)


def test_kl_sharpen_span():
    g = torch.Generator().manual_seed(2)
    a = torch.log_softmax(torch.randn(3, 10, generator=g), -1)
    assert float(kl_rows(a, a)) < 1e-6
    s = sharpen(a, 3)
    assert torch.allclose(s.exp().sum(-1), torch.ones(3)) and int(torch.isfinite(s).sum()) == 9
    b = torch.log_softmax(torch.randn(3, 10, generator=g), -1)
    assert float(kl_rows(s, b)) > 0
    mask = torch.zeros(10, dtype=torch.bool)
    mask[[2, 3]] = True
    ref = -torch.log(b.exp()[:, [2, 3]].sum(-1)).mean()
    assert math.isclose(float(span_nll(b, mask)), float(ref), rel_tol=1e-5)
    # gradient flows into the student
    x = torch.randn(1, 2, 4, 10, requires_grad=True)
    span_nll(mixture_logp(x, [0, 1], 0, 10), mask).backward()
    assert x.grad is not None and float(x.grad.abs().sum()) > 0


def test_masks_and_metrics():
    off = [(0, 0), (0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
    m = chars_to_mask(off, [(4, 8)], 6, 1)            # columns 2,3 overlap chars 4..8
    assert m.tolist() == [False, True, True, False, False]
    assert chars_to_mask(off, [(100, 101)], 6, 1) is None
    assert dilate(torch.tensor([False, False, True, False, False]), 1).tolist() == [False, True, True, True, False]
    lp = torch.log(torch.tensor([[0.1, 0.6, 0.1, 0.1, 0.1], [0.5, 0.1, 0.2, 0.1, 0.1]]))
    r = retrieval_metrics(lp, m)
    assert r["hit1"] == 0.5 and abs(r["mass"] - 0.5) < 1e-6 and abs(r["uniform"] - 0.4) < 1e-6
    assert abs(r["mrr"] - (1.0 + 0.5) / 2) < 1e-6


def test_e3_terms_on_d13_read_text_capture_and_grads():
    """The register_d13 E3 path on a tiny model: QKCap chained over D13Reader (mode off) records the text pass; the
    e2e read's logits give a mixture over the same doc columns; the text / enc / gold terms are finite and train A.
    With reader = encoder, A = I, B = I, the student mixture equals the encoder teacher's (enc term ~ 0)."""
    import copy as _copy
    from test_d13 import _teacher
    from xenc.d13_read import D13Interface, D13Reader, teacher_logits
    from xenc.state_skill import full_layers
    m = tiny_qwen35()
    doc = torch.randint(3, 100, (30,), generator=torch.Generator().manual_seed(0)).tolist()
    rows = torch.randint(3, 100, (7,), generator=torch.Generator().manual_seed(1)).tolist()
    op, rec, _ = _teacher(m, doc, rows)
    L = full_layers(m)
    itf = D13Interface(m, {li: li for li in L}, 64, 64, B=None)
    r = _copy.deepcopy(m)
    R = D13Reader(r, itf)
    cap = QKCap(r, L)
    N, S = len(doc), 2
    loc = [3, 4]
    srows = [len(rows) - 6 + i for i in loc]              # window = last m+1 = 6 rows
    # text pass (doc + rows in context), cap active, read off
    cap.keep, cap.rec = 6, {}
    with torch.no_grad():
        r(input_ids=torch.tensor([doc + rows]), use_cache=False)
    cap.keep = None
    t_lp = {li: mixture_logp(cap.logits(li), loc, S, N) for li in L}
    # student e2e: with an identity map the read = the encoder's attention over [operator ; own rows]
    R.load(op, N)
    R.mode, R.need_logits, R.rec = "e2e", True, {}
    params = itf.trainable()
    with torch.enable_grad():
        r(input_ids=torch.tensor([rows]), use_cache=False)
        s_lp = {li: mixture_logp(R.rec[li]["logits"], srows, S, N) for li in L}
        e_lp = {li: mixture_logp(teacher_logits(rec[li]["q"], op[li][0], rec[li]["k"], itf.att[str(li)].scaling),
                                 srows, S, N) for li in L}
        mask = torch.zeros(N - S, dtype=torch.bool)
        mask[[5, 6]] = True
        tk = torch.stack([kl_rows(t_lp[li], s_lp[li]) for li in L]).mean()
        ek = torch.stack([kl_rows(e_lp[li], s_lp[li]) for li in L]).mean()
        gk = torch.stack([span_nll(s_lp[li], mask) for li in L]).mean()
        assert float(ek) < 1e-4, float(ek)              # identity read: student == encoder teacher at the first layer
        # (the deeper layers also match: e2e passes the read, which equals the encoder's own attention output)
        assert float(tk) < 1e-4, float(tk)              # reader == encoder model: text pass == teacher pass
        assert math.isfinite(float(gk)) and float(gk) > 0
        (gk + kl_rows(sharpen(t_lp[L[0]], 3), s_lp[L[0]])).backward()
    g = itf.A[str(L[0])].weight.grad
    assert g is not None and float(g.abs().sum()) > 0
    R.mode = "off"
    cap.detach()
    R.detach()


class _CharTok:
    """Minimal fast-tokenizer stand-in: tokens = fixed chunks, with offsets (exercises tag-merging tokens)."""
    def __init__(self, cuts):
        self.cuts = cuts

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        b = [0] + [c for c in self.cuts if c < len(text)] + [len(text)]
        off = [(b[i], b[i + 1]) for i in range(len(b) - 1)]
        return {"input_ids": [hash(text[s:e]) % 1000 for s, e in off], "offset_mapping": off}


def test_answer_content_rows_handles_merged_tags():
    from xenc.e3_retrieval import answer_content_rows
    t = "<answer>Betty Cohen</answer>"                   # tokens: "<answer" | ">B" | "etty Cohen" | "</answer>"
    ids, loc = answer_content_rows(_CharTok([7, 9, 19]), t)
    assert loc == [1, 2] and len(ids) == 4
    ids, loc = answer_content_rows(_CharTok([8, 19]), t)  # clean split
    assert loc == [1]
    assert answer_content_rows(_CharTok([5]), "no tags")[1] == []


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("test_e3: all passed")
