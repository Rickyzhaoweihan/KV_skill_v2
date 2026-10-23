"""D18 direction B tests: the trained router (xenc/d18_router.py) on tiny models. CPU only."""

from __future__ import annotations

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from test_d18_library import LMAP, _enc, _ids, _itf, _op  # noqa: E402
from xenc.d18_library import lib_read, question_route  # noqa: E402
from xenc.d18_router import D18Router  # noqa: E402

torch.set_num_threads(2)


def _setup():
    e = _enc()
    itf = _itf(e)
    ops = [_op(e, _ids(n, s)) for n, s in ((20, 1), (31, 2), (17, 3))]
    x = torch.randn(6, 64, generator=torch.Generator().manual_seed(9))
    return itf, ops, x


def test_init_equals_lse_norm_route():
    itf, ops, x = _setup()
    skip = 2
    logit_rows = {}
    for li, en in LMAP.items():
        res = lib_read(itf, li, x, [o[en] for o in ops], skip=skip)
        logit_rows[li] = (res["r"], res["rmax"], res["n"])
    rows = slice(2, 6)
    want = question_route(logit_rows, rows)["lse_norm"]
    R = D18Router(itf, rank=4)
    pos = torch.arange(6)[rows]
    xs = {li: x[rows] for li in LMAP}
    Kdocs = {en: [o[en][0] for o in ops] for en in set(LMAP.values())}
    got = R(xs, pos, Kdocs, skip)
    assert torch.allclose(got, want, atol=1e-4), (got, want)
    got_ck = R(xs, pos, Kdocs, skip, ckpt=True)
    assert torch.allclose(got_ck, want, atol=1e-4)
    print("ok test_init_equals_lse_norm_route")


def test_grad_save_load_and_wfix():
    itf, ops, x = _setup()
    for p in itf.parameters():
        p.requires_grad_(False)
    R = D18Router(itf, rank=4)
    xs = {li: x for li in LMAP}
    Kdocs = {en: [o[en][0] for o in ops] for en in set(LMAP.values())}
    opt = torch.optim.Adam(R.parameters(), lr=5e-2)
    l0 = None
    for _ in range(25):                                    # InfoNCE toward document 2 must be learnable
        s = R(xs, torch.arange(6), Kdocs, 0, ckpt=True)
        loss = torch.nn.functional.cross_entropy(s.unsqueeze(0), torch.tensor([2]))
        l0 = float(loss) if l0 is None else l0
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(loss) < l0 and any(float(p.grad.abs().sum()) > 0 for p in R.U.values())
    with tempfile.TemporaryDirectory() as d:
        R.save(os.path.join(d, "r.pt"), {"x": 1})
        R2, extra = D18Router.load(os.path.join(d, "r.pt"), itf)
    assert extra == {"x": 1}
    assert torch.allclose(R2(xs, torch.arange(6), Kdocs, 0), R(xs, torch.arange(6), Kdocs, 0), atol=1e-6)
    li, en = 5, 5
    lst = [o[en] for o in ops]
    a = lib_read(itf, li, x, lst, force=1)["y"]
    b = lib_read(itf, li, x, lst, wfix=torch.tensor([0.0, 1.0, 0.0]))["y"]
    assert torch.allclose(a, b, atol=1e-6)
    print("ok test_grad_save_load_and_wfix")


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f()
    print("ALL D18 ROUTER TESTS PASSED")
