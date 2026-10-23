"""D24 tests: amortized registration hypernetwork (xenc/d24_hyper.py). CPU, tiny Qwen3.5."""

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
from xenc.d13_read import D13Interface, D13Reader, EncTeacher  # noqa: E402
from xenc.d24_hyper import D24Hyper, attach, load_h, save_h, set_doc  # noqa: E402

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
    ops = []
    for s in (0, 1):
        Tc.rec, Tc.kv_only = {}, True
        with torch.no_grad():
            e(input_ids=torch.tensor([_ids(25, 10 + s)]), use_cache=True)
        ops.append({k: (v["k"], v["v"]) for k, v in Tc.rec.items()})
    Tc.rec = None
    Tc.detach()
    return r, itf, ops


def _logits(r, R, itf, H, op, rows, use_h=True):
    set_doc(itf, H(op) if use_h else None)
    R.load(op, 25)
    R.mode = "e2e"
    try:
        return r(input_ids=torch.tensor([rows])).logits
    finally:
        R.mode = "off"
        set_doc(itf, None)


def test_hyper_starts_at_base_trains_and_conditions_on_operator():
    r, itf, ops = _setup()
    itf.requires_grad_(False)
    sd0 = {k: v.clone() for k, v in itf.state_dict().items()}
    H = D24Hyper(itf, rank=4, n_q=2, d_proj=8, hidden=16, bias_k=3, db_rank=2)
    attach(itf, H)
    assert not any(k.startswith("_d24") for k in itf.state_dict()), "H must not be an interface submodule"
    R = D13Reader(r, itf)
    rows = _ids(6, 1)
    with torch.no_grad():
        base = _logits(r, R, itf, H, ops[0], rows, use_h=False)
        y0 = _logits(r, R, itf, H, ops[0], rows)
    assert torch.allclose(y0, base, atol=1e-5)                    # zero-init heads == base interface
    opt = torch.optim.Adam(H.parameters(), lr=3e-2)
    for _ in range(4):
        opt.zero_grad()
        _logits(r, R, itf, H, ops[0], rows).float().logsumexp(-1).mean().backward()
        opt.step()
    assert all(not p.requires_grad for p in itf.parameters())     # the base interface stays frozen
    for k, v in itf.state_dict().items():
        assert torch.equal(v, sd0[k]), k
    with torch.no_grad():
        y1 = _logits(r, R, itf, H, ops[0], rows)
        y1b = _logits(r, R, itf, H, ops[1], rows)
        base2 = _logits(r, R, itf, H, ops[0], rows, use_h=False)
        a0, a1 = H(ops[0]), H(ops[1])
    assert float((y1 - base).abs().max()) > 1e-5                   # H moved the read
    assert torch.allclose(base2, base, atol=1e-6)                 # set_doc(None) restores the base exactly
    assert float((a0["M"][1] - a1["M"][1]).abs().max()) > 0         # adapter depends on the operator
    assert float((y1 - y1b).abs().max()) > 0
    # save / load round trip
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "h.pt")
        save_h(H, p, {"x": 1})
        H2, ex = load_h(p, itf)
        assert ex == {"x": 1}
        with torch.no_grad():
            b0, b2 = H(ops[1]), H2(ops[1])
        assert torch.allclose(b0["M"][5], b2["M"][5]) and torch.allclose(b0["dg"], b2["dg"])
    R.detach()


def test_zero_input_control_is_document_independent():
    r, itf, ops = _setup()
    H = D24Hyper(itf, rank=4, n_q=2, d_proj=8, hidden=16, input_mode="zero")
    with torch.no_grad():
        for m in H.headM.values():
            m.weight.normal_(0, 0.1)
        a0, a1 = H(ops[0]), H(ops[1])
    assert torch.allclose(a0["M"][1], a1["M"][1]) and float(a0["M"][1].abs().max()) > 0
    assert H.adapter_numel() == 2 * (16 + 1)


def test_v2_standardised_global_plus_residual():
    r, itf, ops = _setup()
    itf.requires_grad_(False)
    H = D24Hyper(itf, rank=4, n_q=2, d_proj=8, hidden=16, bias_k=3, arch="v2")
    H.init_stats(ops)
    attach(itf, H)
    R = D13Reader(r, itf)
    rows = _ids(6, 1)
    with torch.no_grad():
        base = _logits(r, R, itf, H, ops[0], rows, use_h=False)
        assert torch.allclose(_logits(r, R, itf, H, ops[0], rows), base, atol=1e-5)   # zero-init == base
        f = H.standardise(H.summary(ops[0]))
        assert all(float(v.abs().max()) <= 6 for v in f.values())
    opt = torch.optim.Adam(H.parameters(), lr=3e-2)
    for _ in range(4):
        opt.zero_grad()
        _logits(r, R, itf, H, ops[0], rows).float().logsumexp(-1).mean().backward()
        opt.step()
    assert float(H.gM.abs().max()) > 0                              # the global part trains
    with torch.no_grad():
        H.eval()
        a0, a1 = H(ops[0]), H(ops[1])
    assert float((a0["M"][1] - a1["M"][1]).abs().max()) > 0         # and the output still depends on the operator
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "h.pt")
        save_h(H, p)
        H2, _ = load_h(p, itf)
        H2.eval()
        with torch.no_grad():
            assert torch.allclose(H2(ops[1])["M"][5], a1["M"][5], atol=1e-6)
    R.detach()


if __name__ == "__main__":
    test_v2_standardised_global_plus_residual()
    test_hyper_starts_at_base_trains_and_conditions_on_operator()
    test_zero_input_control_is_document_independent()
    print("test_d24: OK")
