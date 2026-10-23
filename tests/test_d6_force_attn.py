"""D6: LibraryInjector.force_attn (oracle addressing) plumbing. CPU only.
Run: python tests/test_d6_force_attn.py   (or pytest)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tiny import TinyLM, make_skill_rho  # noqa: E402

from xenc import DEPTHS  # noqa: E402
from deltaskill.derive import injection_layers, rms_norm  # noqa: E402
from deltaskill.interface import ModelInterface  # noqa: E402
from xenc.depth_skill import rho_layer_map  # noqa: E402
from xenc.injector import LibraryInjector, itf_gain, itf_gate, itf_write  # noqa: E402
from xenc.library import Library  # noqa: E402


def _setup():
    reader = TinyLM(d=32, n_layers=8)
    layers = injection_layers(reader, DEPTHS)
    itf = ModelInterface(32, 24, layers)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for p in itf.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    rmap = rho_layer_map(reader, DEPTHS)
    lib = Library([make_skill_rho(24, DEPTHS, n=n, seed=n) for n in (5, 9, 3)],
                  [rmap[r] for r in DEPTHS], DEPTHS)
    x = torch.randn(2, 7, 32, generator=torch.Generator().manual_seed(1))
    return reader, layers, itf, lib, x


def test_force_none_is_identity():
    reader, layers, itf, lib, x = _setup()
    inj = LibraryInjector(reader, itf)
    inj.load_library(lib)
    with torch.no_grad():
        a = reader(x)
        inj.force_attn = None
        b = reader(x)
    assert torch.equal(a, b)
    inj.detach()


def test_force_onehot_reads_that_slot():
    reader, layers, itf, lib, x = _setup()
    inj = LibraryInjector(reader, itf)
    inj.load_library(lib)
    li = layers[1]
    C, N = lib.mask[li].shape
    fa = torch.zeros(C, N)
    fa[1, 4] = 1.0
    inj.force_attn = fa
    h = torch.randn(2, 7, 32, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        d = inj._branch(h, li)
        hn = rms_norm(h.float())
        ref = itf_gain(itf, li, hn) * itf_gate(itf, hn, li) * itf_write(itf, lib.K[li][1, 4].expand(2, 7, -1))
    assert (d - ref).abs().max() < 1e-5
    # per-layer dict form + shape guard
    inj.force_attn = {l: torch.zeros(lib.mask[l].shape) for l in layers}
    with torch.no_grad():
        assert inj._branch(h, li).abs().max() == 0
    inj.force_attn = torch.zeros(C, N + 1)
    try:
        inj._branch(h, li)
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch must raise")
    inj.detach()


def test_force_norm_match():
    reader, layers, itf, lib, x = _setup()
    inj = LibraryInjector(reader, itf)
    inj.load_library(lib)
    li = layers[0]
    h = torch.randn(1, 5, 32, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        d0 = inj._branch(h, li)
        M = lib.mask[li]
        fa = torch.zeros(M.shape)
        fa[0, :2] = 0.5
        inj.force_attn, inj.force_norm_match = fa, True
        import xenc.injector as xi
        captured = {}
        orig = xi.itf_write
        xi.itf_write = lambda itf_, r: (captured.setdefault("r", []).append(r), orig(itf_, r))[1]
        try:
            inj._branch(h, li)
            inj.force_attn = None
            inj._branch(h, li)
        finally:
            xi.itf_write = orig
    r_forced, r_comp = captured["r"]
    assert torch.allclose(r_forced.norm(dim=-1), r_comp.norm(dim=-1), atol=1e-4)
    assert d0.shape == (1, 5, 32)
    inj.detach()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
    print("OK")
