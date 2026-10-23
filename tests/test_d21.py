"""D21 tests: post-hoc compression of a D13 interface state dict (xenc/d21_compress.py). CPU only, random tensors."""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import torch  # noqa: E402

from xenc.d21_compress import compress, parse, quant_int4, quant_int8  # noqa: E402

torch.set_num_threads(2)


def _sd(d_r=32, d_e=24, h=8, layers=(3, 7)):
    g = torch.Generator().manual_seed(0)
    sd = {}
    for li in layers:
        sd[f"A.{li}.weight"] = torch.randn(d_e, d_r, generator=g)
        sd[f"A.{li}.bias"] = torch.randn(d_e, generator=g)
        sd[f"M.{li}.0.weight"] = torch.randn(h, d_r, generator=g)
        sd[f"M.{li}.0.bias"] = torch.randn(h, generator=g)
        sd[f"M.{li}.2.weight"] = torch.randn(d_e, h, generator=g)
        sd[f"M.{li}.2.bias"] = torch.randn(d_e, generator=g)
        sd[f"B.{li}"] = torch.randn(d_r, d_e, generator=g)
        sd[f"Bb.{li}"] = torch.randn(d_r, generator=g)
    sd["g"] = torch.ones(len(layers))
    return sd


def test_full_counts_and_identity():
    sd = _sd()
    new, r = compress(sd, "full")
    assert new == {} and r["params"] == sum(v.numel() for v in sd.values()) and r["bytes"] == 4 * r["params"]


def test_quant_errors():
    W = torch.randn(16, 128, generator=torch.Generator().manual_seed(1))
    q8, b8 = quant_int8(W)
    q4, b4 = quant_int4(W, 64)
    e8 = float((q8 - W).norm() / W.norm())
    e4 = float((q4 - W).norm() / W.norm())
    assert e8 < 0.01 < e4 < 0.2, (e8, e4)
    assert b8 == 16 * 128 + 2 * 16 and b4 == 16 * 128 // 2 + 4 * 16 * 2


def test_svd_rank_and_size():
    sd = _sd()
    new, r = compress(sd, "svd4_A")
    W, Wr = sd["A.3.weight"], new["A.3.weight"]
    assert torch.linalg.matrix_rank(Wr) == 4 and set(k for k in new) == {"A.3.weight", "A.7.weight"}
    full = compress(sd, "full")[1]["params"]
    assert r["params"] == full - 2 * (24 * 32 - 4 * (24 + 32))
    S = torch.linalg.svdvals(W)
    assert abs(float((Wr - W).norm()) - float(S[4:].norm())) < 1e-3          # Eckart-Young optimum
    new2, _ = compress(sd, "svd100")                                          # rank above break-even: unchanged
    assert new2 == {}


def test_drop_and_noM():
    sd = _sd()
    new, r = compress(sd, "drop7+noM")
    assert r["layers"] == [3] and new["g"].tolist() == [1.0, 0.0]
    assert float(new["M.3.2.weight"].abs().sum()) == 0 and float(new["M.3.2.bias"].abs().sum()) == 0
    exp = 24 * 32 + 24 + 32 * 24 + 32 + 1                                     # layer 3: A, A bias, B, Bb, gate
    assert r["params"] == exp, (r["params"], exp)
    assert parse("svd64+int8")["dtype"] == "int8" and parse("svd64_AB")["svd"] == {"A": 64, "B": 64}


def test_asvd_beats_svd_on_its_input_distribution():
    sd = _sd()
    g = torch.Generator().manual_seed(2)
    X = torch.randn(500, 32, generator=g) @ torch.diag(torch.logspace(0, -3, 32))    # anisotropic inputs
    covs = {k: (X.t() @ X) / 500 for k in sd if k.startswith(("A.", "M.")) and k.endswith("0.weight") or k.startswith("A.") and k.endswith("weight")}
    covs.update({k: torch.eye(sd[k].shape[1]) for k in sd if k.startswith("B.") or k.endswith("2.weight")})
    n_s, r_s = compress(sd, "svd4_A")
    n_a, r_a = compress(sd, "asvd4_A", covs=covs)
    assert r_s["params"] == r_a["params"]
    W = sd["A.3.weight"]
    e_s = float(((n_s["A.3.weight"] - W) @ X.t()).norm())
    e_a = float(((n_a["A.3.weight"] - W) @ X.t()).norm())
    assert e_a < 0.5 * e_s, (e_a, e_s)


if __name__ == "__main__":
    for nm, f in list(globals().items()):
        if nm.startswith("test_") and callable(f):
            f()
            print("ok", nm)
    print("ALL D21 TESTS PASSED")
