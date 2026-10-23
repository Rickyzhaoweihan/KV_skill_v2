"""xenc.library + xenc.injector: no-op at init, guards, 1-chunk equivalence to ExternalSkillInjector,
top-k routing, oracle subset. Run: python tests/test_library.py   (or pytest). CPU only."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tiny import TinyLM, make_skill_rho  # noqa: E402

from xenc import DEPTHS  # noqa: E402
from deltaskill.derive import injection_layers  # noqa: E402
from deltaskill.injector import ExternalSkillInjector  # noqa: E402
from deltaskill.interface import ModelInterface  # noqa: E402
from xenc.depth_skill import TinyTokenizer, derive_by_depth, for_reader, rho_layer_map  # noqa: E402
from xenc.injector import LibraryInjector, XEncInjector  # noqa: E402
from xenc.library import Library  # noqa: E402


def _randomize(itf: ModelInterface, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in itf.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)


def _x(d, T=7, B=2, seed=1):
    return torch.randn(B, T, d, generator=torch.Generator().manual_seed(seed))


def _lib_from_skills(skills, model):
    rmap = rho_layer_map(model, DEPTHS)
    return Library(skills, [rmap[r] for r in DEPTHS], DEPTHS, device="cpu")


def test_xenc_noop_and_guards():
    reader = TinyLM(d=40, n_layers=8)
    layers = injection_layers(reader, DEPTHS)
    itf = ModelInterface(40, 24, layers)
    x = _x(40)
    with torch.no_grad():
        bare = reader(x)
    inj = XEncInjector(reader, itf)
    inj.load(make_skill_rho(24, DEPTHS))
    assert sorted(inj.skill.K) == layers
    with torch.no_grad():
        out = reader(x)
    assert torch.equal(out, bare), "zero-init B must be a no-op"
    assert len(inj.mag_terms) == len(layers) and all(float(m) == 0 for m in inj.mag_terms)
    try:
        inj.load(make_skill_rho(32, DEPTHS))
    except ValueError:
        pass
    else:
        raise AssertionError("d_s mismatch must raise ValueError")
    s = make_skill_rho(24, DEPTHS)
    li_skill = for_reader(s, reader)
    li_skill.K.pop(layers[-1])
    try:
        inj.load(li_skill)
    except KeyError:
        pass
    else:
        raise AssertionError("missing reader layer must raise KeyError")
    inj.detach()


def test_library_noop_at_init():
    reader = TinyLM(d=40, n_layers=8)
    itf = ModelInterface(40, 24, injection_layers(reader, DEPTHS))
    lib = _lib_from_skills([make_skill_rho(24, DEPTHS, n=n, seed=n) for n in (5, 9, 3)], reader)
    x = _x(40)
    with torch.no_grad():
        bare = reader(x)
    inj = LibraryInjector(reader, itf)
    inj.load_library(lib)
    with torch.no_grad():
        assert torch.equal(reader(x), bare)
    try:
        inj.load_library(_lib_from_skills([make_skill_rho(16, DEPTHS)], reader))
    except ValueError:
        pass
    else:
        raise AssertionError
    inj.detach()


def test_one_chunk_equals_external():
    reader = TinyLM(d=32, n_layers=8)
    layers = injection_layers(reader, DEPTHS)
    skill = make_skill_rho(32, DEPTHS, n=11, seed=4)
    x = _x(32)
    for read_softmax in (False, True):
        itf = ModelInterface(32, 32, layers)
        _randomize(itf, seed=7)
        assert itf.B.weight.abs().sum() > 0
        ext = ExternalSkillInjector(reader, itf, read_softmax=read_softmax)
        ext.load(for_reader(skill, reader))
        with torch.no_grad():
            ref = reader(x)
        ext.detach()
        lib_inj = LibraryInjector(reader, itf, read_softmax=read_softmax)
        lib_inj.load_library(_lib_from_skills([skill], reader))
        with torch.no_grad():
            out = reader(x)
        lib_inj.detach()
        with torch.no_grad():
            bare = reader(x)
        assert (ref - bare).abs().max() > 1e-3, "branch must actually do something"
        err = (out - ref).abs().max().item()
        assert err < 1e-5, (read_softmax, err)
        # mag_terms bookkeeping identical
        assert len(ext.mag_terms) == len(lib_inj.mag_terms) == len(layers)
        for a, b in zip(ext.mag_terms, lib_inj.mag_terms):
            assert abs(float(a) - float(b)) < 1e-5
    # also with the encoder == reader derive path (real derive_by_depth output, 1 chunk)
    tok = TinyTokenizer()
    d = derive_by_depth(reader, tok, " ".join(f"w{i}" for i in range(20)), DEPTHS, chunk_tokens=64)
    lib = Library.from_doc_cache(d, reader)
    assert lib.n_chunks == 1 and lib.layers == layers


def test_routing_topk():
    reader = TinyLM(d=32, n_layers=8)
    layers = injection_layers(reader, DEPTHS)
    itf = ModelInterface(32, 24, layers)
    _randomize(itf, seed=3)
    lib = _lib_from_skills([make_skill_rho(24, DEPTHS, n=4 + i, seed=i) for i in range(5)], reader)
    assert lib.n_chunks == 5 and lib.K[layers[0]].shape == (5, 8, 24)
    for top_k in (0, 2):
        inj = LibraryInjector(reader, itf, top_k=top_k)
        inj.load_library(lib)
        reader(_x(32))                        # grad enabled -> route_terms recorded
        assert len(inj.route_terms) == len(layers)
        for li, w in inj.route_terms:
            assert w.shape == (2, 7, 5)
            assert torch.allclose(w.sum(-1), torch.ones(2, 7), atol=1e-6)
            nz = (w > 0).sum(-1)
            assert bool((nz == (2 if top_k else 5)).all()), (top_k, nz)
            assert w.requires_grad
        assert torch.equal(inj.last_route, inj.route_terms[-1][1].detach())
        inj.detach()


class _CountingSelector:
    """Stand-in for agent [B]'s LayerSelector: records reset/apply calls, halves delta."""

    def __init__(self):
        self.resets, self.applied = 0, []

    def reset(self):
        self.resets += 1

    def apply(self, delta, h, li):
        self.applied.append(li)
        return 0.5 * delta


def test_selector_hooks():
    reader = TinyLM(d=32, n_layers=8)
    layers = injection_layers(reader, DEPTHS)
    itf = ModelInterface(32, 32, layers)
    _randomize(itf, seed=2)
    lib = _lib_from_skills([make_skill_rho(32, DEPTHS, n=6, seed=i) for i in range(3)], reader)
    x = _x(32)
    base = LibraryInjector(reader, itf)
    base.load_library(lib)
    with torch.no_grad():
        full = reader(x)
    base.detach()
    sel = _CountingSelector()
    inj = LibraryInjector(reader, itf, selector=sel)
    inj.load_library(lib)
    with torch.no_grad():
        reader(x)
        reader(x)
        halved = reader(x)
    assert sel.resets == 3 and sel.applied == layers * 3
    assert (halved - full).abs().max() > 1e-4, "selector output must be used"
    inj.detach()


def test_oracle_subset():
    reader = TinyLM(d=32, n_layers=8)
    skills = [make_skill_rho(32, DEPTHS, n=n, seed=i) for i, n in enumerate((6, 12, 3, 9))]
    lib = _lib_from_skills(skills, reader)
    for i in range(4):
        a, b = lib.subset([i]), _lib_from_skills([skills[i]], reader)
        assert a.n_chunks == 1 and a.layers == b.layers
        for li in a.layers:
            assert torch.equal(a.K[li], b.K[li]) and torch.equal(a.mask[li], b.mask[li])
            assert torch.allclose(a.S[li], b.S[li], atol=1e-6)
    ab = lib.subset([1, 3])
    ref = _lib_from_skills([skills[1], skills[3]], reader)
    for li in ab.layers:
        assert torch.equal(ab.K[li], ref.K[li]) and torch.allclose(ab.S[li], ref.S[li], atol=1e-6)
    # summary keys are unit-norm means of the valid rows
    li = lib.layers[0]
    rho = lib.rho_of[li]
    m = skills[2].K[rho].mean(0)
    assert torch.allclose(lib.S[li][2], m / m.norm(), atol=1e-5)


def test_from_doc_cache():
    enc = TinyLM(d=24, n_layers=12, seed=5)
    reader = TinyLM(d=40, n_layers=8)
    d = derive_by_depth(enc, TinyTokenizer(), " ".join(f"t{i % 37}" for i in range(90)), DEPTHS,
                        chunk_tokens=32, doc_id="dd", encoder="tiny-enc")
    lib = Library.from_doc_cache(d, reader)
    assert lib.n_chunks == 3 and lib.d_s == 24 and lib.layers == injection_layers(reader, DEPTHS)
    assert lib.mask[lib.layers[0]].sum(-1).tolist() == [32, 32, 26]
    assert lib.K[lib.layers[0]].dtype == torch.float32
    assert [m["start"] for m in lib.chunk_meta] == [0, 32, 64]
    itf = ModelInterface(40, 24, injection_layers(reader, DEPTHS))
    _randomize(itf)
    x = _x(40)
    with torch.no_grad():
        bare = reader(x)
    inj = LibraryInjector(reader, itf, top_k=2)
    inj.load_library(lib)
    with torch.no_grad():
        y = reader(x)
    assert torch.isfinite(y).all() and (y - bare).abs().max() > 1e-4
    inj.unload()
    with torch.no_grad():
        assert torch.equal(reader(x), bare), "unload restores the bare model"
    inj.detach()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name}: ok")
    print("OK")


def test_centering():
    """Centered Library: mu=0 is a no-op; nonzero mu re-normalizes rows and changes summaries."""
    import torch
    from xenc.library import Library
    from _tiny import make_skill_rho
    depths = [0.25, 0.5, 0.75, 1.0]
    chunks = [make_skill_rho(24, depths, n=8, seed=s) for s in range(3)]
    layers = [2, 4, 6, 7]
    base = Library(chunks, layers, depths)
    zero = Library(chunks, layers, depths, center={r: torch.zeros(24) for r in depths})
    for li in layers:
        assert torch.allclose(base.K[li], zero.K[li], atol=1e-6)
    mu = {r: torch.full((24,), 0.3) for r in depths}
    cen = Library(chunks, layers, depths, center=mu)
    for li in layers:
        M = cen.mask[li]
        norms = cen.K[li][M].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
        assert not torch.allclose(cen.K[li], base.K[li])
        assert torch.allclose(cen.K[li][~M], torch.zeros_like(cen.K[li][~M]))
    print("test_centering: ok")


if __name__ == "__main__" and "test_centering" in dir():
    test_centering()
