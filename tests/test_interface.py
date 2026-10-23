"""xenc.interface: make_interface no-op + shapes, EncoderLens, param groups, save/load, counts."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _tiny  # noqa: E402,F401
from deltaskill.derive import rms_norm  # noqa: E402
from xenc.interface import (EncoderLens, count_params, interface_param_groups,  # noqa: E402
                            load_interface, make_interface, save_interface)
from xenc.selector import LayerSelector  # noqa: E402

DR, DE, LAYERS = 40, 24, [2, 4, 6, 7]


def _branch(itf, h, li, K):
    hn = rms_norm(h.float())
    q = itf.query(hn)
    r = (q @ K.T) @ K / K.shape[0] ** 0.5
    return itf.gain(li) * itf.gate(hn, li) * itf.write(r)


def test_make_interface_noop_and_shapes():
    for rank in (0, 8):
        itf = make_interface(DR, DE, LAYERS, rank=rank)
        h = torch.randn(2, 5, DR)
        hn = rms_norm(h)
        assert itf.query(hn).shape == (2, 5, DE)
        assert itf.write(torch.randn(2, 5, DE)).shape == (2, 5, DR)
        assert itf.gate(hn, 6).shape == (2, 5, 1)
        K = torch.nn.functional.normalize(torch.randn(16, DE), dim=-1)
        for li in LAYERS:
            assert torch.equal(_branch(itf, h, li, K), torch.zeros(2, 5, DR))  # B zero -> no-op
        assert itf.d_m == DR and itf.d_s == DE and itf.inject_layers == LAYERS
    try:
        make_interface(DR, DE, [2, 2])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_encoder_lens():
    lens = EncoderLens(24, 16)
    x = torch.randn(3, 7, 24)
    y = lens(x)
    assert y.shape == (3, 7, 16)
    W = lens.proj.weight
    assert torch.allclose(W @ W.T, torch.eye(16), atol=1e-5)  # orthogonal rows


def _perturb(m, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 0.1)


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        for with_sel in (False, True):
            for rank in (0, 8):
                torch.manual_seed(1)
                itf = make_interface(DR, DE, LAYERS, rank=rank)
                _perturb(itf, 3)
                sel = LayerSelector(DR, LAYERS, k=2, alpha_init=0.05) if with_sel else None
                if sel is not None:
                    _perturb(sel, 4)
                path = f"{td}/itf_{with_sel}_{rank}.pt"
                save_interface(itf, path, selector=sel, extra={"step": 7})
                itf2, sel2, extra = load_interface(path, DR, DE, LAYERS)
                assert extra == {"step": 7}
                assert (sel2 is None) == (sel is None)
                h = torch.randn(2, 5, DR)
                K = torch.nn.functional.normalize(torch.randn(16, DE), dim=-1)
                for li in LAYERS:
                    a, b = _branch(itf, h, li, K), _branch(itf2, h, li, K)
                    assert a.abs().sum() > 0
                    if sel is not None:
                        a, b = sel.apply(a, h, li), sel2.apply(b, h, li)
                    assert torch.equal(a, b), (with_sel, rank, li)
                if sel is not None:
                    assert sel2.cand_layers == sel.cand_layers and sel2.k == sel.k
                    assert torch.equal(sel.weights(), sel2.weights())
        try:
            load_interface(path, DR, 32, LAYERS)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_param_groups_cover_once():
    for rank in (0, 8):
        itf = make_interface(DR, DE, LAYERS, rank=rank)
        sel = LayerSelector(DR, LAYERS, k=2)
        for s, slr in ((None, None), (sel, None), (sel, 1e-3)):
            groups = interface_param_groups(itf, 2e-4, selector=s, selector_lr=slr)
            ids = [id(p) for g in groups for p in g["params"]]
            expect = [id(p) for p in itf.parameters()] + (
                [id(p) for p in s.parameters()] if s is not None else [])
            assert len(ids) == len(set(ids)) == len(expect) and set(ids) == set(expect)
            assert groups[0]["lr"] == 2e-4
            if s is not None:
                assert groups[1]["lr"] == (slr if slr is not None else 2e-4)
            torch.optim.AdamW(groups)  # accepted by the optimizer
        c = count_params(itf, sel)
        assert c["total"] == c["interface"] + c["selector"]
        assert c["interface"] == sum(p.numel() for p in itf.parameters())
        assert c["selector"] == DR * len(LAYERS) + len(LAYERS) + 1
        assert c["A"] + c["B"] + c["gate_gain"] == c["interface"]


def test_param_groups_lr_A():  # D5: separate lr for the query map A
    for rank in (0, 8):
        itf = make_interface(DR, DE, LAYERS, rank=rank)
        sel = LayerSelector(DR, LAYERS, k=2)
        groups = interface_param_groups(itf, 2e-4, selector=sel, lr_A=2e-3)
        ids = [id(p) for g in groups for p in g["params"]]
        expect = [id(p) for p in itf.parameters()] + [id(p) for p in sel.parameters()]
        assert len(ids) == len(set(ids)) == len(expect) and set(ids) == set(expect)
        ga = [g for g in groups if g["name"] == "interface_A"][0]
        assert ga["lr"] == 2e-3 and {id(p) for p in ga["params"]} == {id(p) for p in itf.A.parameters()}
        torch.optim.AdamW(groups)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
    print("OK")
