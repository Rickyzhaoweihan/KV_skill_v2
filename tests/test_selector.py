"""LayerSelector: sparse top-k weights, matched update norm, gradients, reset/T=1. CPU, fast."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tiny  # noqa: E402,F401  (bootstraps xenc + deltaskill)
from deltaskill.derive import rms_norm  # noqa: E402
from xenc.selector import LayerSelector  # noqa: E402

D = 40
CAND = [3, 5, 7, 9, 11, 13, 15, 17]


def _sel(k=2, st=False, seed=0):
    torch.manual_seed(seed)
    return LayerSelector(D, CAND, k=k, alpha_init=0.02, straight_through=st)


def test_weights_sum_and_sparsity():
    for st in (False, True):
        for k in (1, 2, 3):
            sel = _sel(k=k, st=st)
            h = torch.randn(3, 7, D)
            sel.compute(rms_norm(h))
            w = sel.weights()
            assert w.shape == (3, 7, len(CAND))
            assert torch.allclose(w.sum(-1), torch.ones(3, 7), atol=1e-6)
            assert ((w != 0).sum(-1) == k).all(), (k, st)
            assert (w >= 0).all()
            assert sel.n_cand == 8 and sel.pos[CAND[3]] == 3
            ent = sel.entropy()
            assert 0.0 <= float(ent) <= math.log(k) + 1e-5


def test_matched_norm():
    for st in (False, True):
        sel = _sel(st=st)
        alpha = float(sel.log_alpha.detach().exp())
        h = torch.randn(2, 5, D) * 3.0
        base = torch.randn(2, 5, D)
        for scale in (1.0, 100.0):  # raw delta norms differ by 100x
            sel.reset()
            sel.compute(rms_norm(h))  # explicit compute is honoured at the first layer
            w = sel.weights().detach().clone()
            for li in CAND:
                out = sel.apply(base * scale, h, li)
                ratio = out.norm(dim=-1) / h.norm(dim=-1)
                exp = w[..., sel.pos[li]] * alpha
                assert torch.allclose(ratio, exp, atol=1e-5), (st, scale, li, (ratio - exp).abs().max())


def test_gradients():
    sel = _sel()
    h = torch.randn(2, 4, D)
    delta = torch.randn(2, 4, D, requires_grad=True)
    sel.reset()
    loss = sum(sel.apply(delta, h, li).sum() for li in CAND)
    loss.backward()
    assert sel.u.weight.grad is not None and sel.u.weight.grad.abs().sum() > 0
    assert sel.u.bias.grad is not None and sel.u.bias.grad.abs().sum() > 0
    assert sel.log_alpha.grad is not None and sel.log_alpha.grad.abs() > 0
    assert delta.grad is not None
    # straight-through: gradient reaches every logit (dense soft backward)
    st = _sel(st=True)
    st.reset()
    loss = sum((st.apply(torch.randn(2, 4, D), h, li) * (i + 1)).sum() for i, li in enumerate(CAND))
    loss.backward()
    assert st.u.bias.grad.abs().min() > 0


def test_reset_and_decode_steps():
    sel = _sel()
    h = torch.randn(1, 6, D)
    for li in CAND:
        sel.apply(torch.randn(1, 6, D), h, li)
    assert sel.weights().shape == (1, 6, 8)
    sel.reset()
    assert sel.weights() is None and sel.entropy() is None
    # T=1 decode steps WITHOUT reset: weights must be recomputed at the first candidate layer
    prev = None
    for step in range(3):
        h1 = torch.randn(1, 1, D) * (step + 1)
        for li in CAND:
            out = sel.apply(torch.randn(1, 1, D), h1, li)
            assert out.shape == (1, 1, D) and torch.isfinite(out).all()
        w = sel.weights()
        exp = sel._weights_from_logits(sel.u(rms_norm(h1.float())))
        assert torch.allclose(w, exp), step
        if prev is not None:
            assert not torch.equal(w, prev)
        prev = w.detach().clone()
    # explicit compute at the first layer is honoured (not overwritten)
    sel.reset()
    hn0 = torch.randn(1, 1, D)
    sel.compute(hn0)
    w0 = sel.weights().clone()
    sel.apply(torch.randn(1, 1, D), torch.randn(1, 1, D), CAND[0])
    assert torch.equal(sel.weights(), w0)
    # non-candidate layer -> KeyError
    try:
        sel.apply(torch.randn(1, 1, D), torch.randn(1, 1, D), 4)
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_dtype_preserved():
    sel = _sel()
    h = torch.randn(1, 3, D).to(torch.bfloat16)
    out = sel.apply(torch.randn(1, 3, D).to(torch.bfloat16), h, CAND[0])
    assert out.dtype == torch.bfloat16


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
    print("OK")
