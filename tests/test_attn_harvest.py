"""D3: harvest="attn" — encoder attention keys/values as an UNTIED operator (U from keys, W from values).
Untied Library shapes, the read uses V (not K), cache round-trip with tied=False, attn-layer mapping on a
hybrid (linear/full attention) stack. Run: python tests/test_attn_harvest.py  (or pytest). CPU only."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _tiny import TinyAttnLM, TinyLM  # noqa: E402

from xenc import DEPTHS  # noqa: E402
from deltaskill.derive import injection_layers  # noqa: E402
from deltaskill.interface import ModelInterface  # noqa: E402
from xenc.depth_skill import (TinyTokenizer, attn_layer_map, cache_key, chunk_to_kvskill,  # noqa: E402
                              derive_by_depth, load_doc_cache, rho_layer_map, save_doc_cache)
from xenc.injector import LibraryInjector, XEncInjector  # noqa: E402
from xenc.library import Library  # noqa: E402

_TEXT = " ".join(f"t{(i * 7) % 41}" for i in range(90))


def _randomize(itf, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in itf.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)


def _derive(enc=None, **kw):
    enc = enc or TinyAttnLM(d=24, n_layers=8, seed=5)
    return derive_by_depth(enc, TinyTokenizer(), _TEXT, DEPTHS, chunk_tokens=32, doc_id="dd",
                           encoder="tiny-attn", harvest="attn", **kw)


def test_attn_derive_shapes():
    d = _derive()
    assert d["harvest"] == "attn" and d["tied"] is False and d["d_s"] == 16       # 2 kv heads x 8
    assert d["key_source"] == "k_proj"
    assert len(d["chunks"]) == 3 and [c["n_tokens"] for c in d["chunks"]] == [32, 32, 26]
    for c in d["chunks"]:
        for r in DEPTHS:
            K, V = c["K"][r], c["V"][r]
            assert K.shape == V.shape == (c["n_tokens"], 16) and K.dtype == torch.float16
            assert torch.allclose(K.float().norm(dim=-1), torch.ones(K.shape[0]), atol=2e-3)
            assert torch.allclose(V.float().norm(dim=-1), torch.ones(V.shape[0]), atol=2e-3)
            assert not torch.allclose(K, V)
    # resid harvest on the same model is unchanged (tied, d_s = hidden)
    r = derive_by_depth(TinyAttnLM(d=24, n_layers=8, seed=5), TinyTokenizer(), _TEXT, DEPTHS, chunk_tokens=32)
    assert r["d_s"] == 24 and "V" not in r["chunks"][0]


def test_attn_layer_map_hybrid():
    enc = TinyAttnLM(d=24, n_layers=16, seed=1, full_every=4)      # full attn at 3, 7, 11, 15
    assert list(rho_layer_map(enc, DEPTHS).values()) == [4, 8, 12, 15]
    amap = attn_layer_map(enc, DEPTHS)
    assert list(amap.values()) == [7, 11, 15, 15]
    d = _derive(enc)
    assert d["attn_layers"] == {0.25: 7, 0.5: 11, 0.75: 15, 1.0: 15}
    c = d["chunks"][0]
    assert torch.equal(c["K"][0.75], c["K"][1.0])                  # same physical layer
    try:
        derive_by_depth(TinyLM(d=24, n_layers=8), TinyTokenizer(), _TEXT, DEPTHS, harvest="attn")
    except ValueError:
        pass
    else:
        raise AssertionError("no attention modules must raise")


def test_cache_roundtrip_untied():
    d = _derive()
    with tempfile.TemporaryDirectory() as td:
        k_attn = cache_key(_TEXT, "tiny-attn", 32, DEPTHS, harvest="attn")
        k_res = cache_key(_TEXT, "tiny-attn", 32, DEPTHS)
        assert k_attn.endswith("_hattn") and k_res == k_attn[: -len("_hattn")]
        assert cache_key(_TEXT, "tiny-attn", 32, DEPTHS, harvest="resid") == k_res
        path = os.path.join(td, k_attn + ".pt")
        save_doc_cache(d, path)
        e = load_doc_cache(path)
    assert e["tied"] is False and e["harvest"] == "attn" and e["d_s"] == 16
    for a, b in zip(d["chunks"], e["chunks"]):
        for r in DEPTHS:
            assert torch.equal(a["K"][r], b["K"][r]) and torch.equal(a["V"][r], b["V"][r])
    s = chunk_to_kvskill(e["chunks"][1], e["d_s"], "dd", "tiny-attn")
    assert s.meta["tied"] is False and s.V is not s.K
    assert torch.equal(s.V[0.5], e["chunks"][1]["V"][0.5])


def test_untied_library_shapes():
    reader = TinyLM(d=40, n_layers=8)
    d = _derive()
    lib = Library.from_doc_cache(d, reader)
    layers = injection_layers(reader, DEPTHS)
    assert lib.V is not None and lib.layers == layers and lib.d_s == 16
    for li, rho in lib.rho_of.items():
        assert lib.V[li].shape == lib.K[li].shape == (3, 32, 16)
        assert torch.allclose(lib.V[li][2, :26], d["chunks"][2]["V"][rho].float())
        assert lib.V[li][2, 26:].abs().sum() == 0
        m = d["chunks"][0]["K"][rho].float().mean(0)                  # S comes from K, not V
        assert torch.allclose(lib.S[li][0], m / m.norm(), atol=1e-4)
    sub = lib.subset([2])
    assert sub.V is not None and sub.V[layers[0]].shape == (1, 26, 16)
    ref = Library([{**d["chunks"][2]}], lib.reader_layers, DEPTHS)
    for li in layers:
        assert torch.allclose(sub.V[li], ref.V[li]) and torch.allclose(sub.K[li], ref.K[li])
    lib.to(dtype=torch.float64)
    assert lib.V[layers[0]].dtype == torch.float64
    # tied caches still give V is None
    tied = Library.from_doc_cache(derive_by_depth(TinyLM(d=24, n_layers=12), TinyTokenizer(), _TEXT, DEPTHS,
                                                  chunk_tokens=32), reader)
    assert tied.V is None


def _run(reader, itf, lib, x, read_softmax=True):
    inj = LibraryInjector(reader, itf, read_softmax=read_softmax)
    inj.load_library(lib)
    with torch.no_grad():
        y = reader(x)
    inj.detach()
    return y


def test_read_uses_values():
    reader = TinyLM(d=40, n_layers=8)
    d = _derive()
    itf = ModelInterface(40, 16, injection_layers(reader, DEPTHS))
    _randomize(itf, 3)
    x = torch.randn(2, 7, 40, generator=torch.Generator().manual_seed(1))
    for rs in (True, False):
        untied = Library.from_doc_cache(d, reader)
        tied_chunks = [{"K": c["K"], "n_tokens": c["n_tokens"], "start": c["start"]} for c in d["chunks"]]
        tied = Library(tied_chunks, untied.reader_layers, DEPTHS)
        selfv = Library([{**c, "V": {r: t.clone() for r, t in c["K"].items()}} for c in tied_chunks],
                        untied.reader_layers, DEPTHS)
        assert selfv.V is not None
        y_u, y_t, y_s = (_run(reader, itf, L, x, rs) for L in (untied, tied, selfv))
        assert torch.allclose(y_s, y_t, atol=1e-5), "V := K copy must equal the tied read"
        assert (y_u - y_t).abs().max() > 1e-3, "untied read must use V, not K"
        # perturbing V alone changes the output; perturbing it must not change routing (S from K)
        pert = Library.from_doc_cache(d, reader)
        for li in pert.V:
            pert.V[li] = torch.roll(pert.V[li], 1, dims=1)
        assert (_run(reader, itf, pert, x, rs) - y_u).abs().max() > 1e-4


def test_one_chunk_untied_equals_single_skill():
    reader = TinyLM(d=32, n_layers=8)
    d = _derive()
    chunk = d["chunks"][0]
    itf = ModelInterface(32, 16, injection_layers(reader, DEPTHS))
    _randomize(itf, 7)
    x = torch.randn(2, 7, 32, generator=torch.Generator().manual_seed(2))
    s = chunk_to_kvskill({"K": {r: t.float() for r, t in chunk["K"].items()},
                          "V": {r: t.float() for r, t in chunk["V"].items()}, "start": 0}, 16, "dd", "tiny-attn")
    ext = XEncInjector(reader, itf, read_softmax=True)
    ext.load(s)
    with torch.no_grad():
        ref = reader(x)
    ext.detach()
    lib = Library.from_doc_cache({**d, "chunks": [chunk]}, reader)
    assert (ref - _run(reader, itf, lib, x)).abs().max() < 1e-4


def test_null_library_randomizes_values():
    from eval_doc import null_library
    reader = TinyLM(d=40, n_layers=8)
    lib = Library.from_doc_cache(_derive(), reader)
    nul = null_library(lib, "random", seed=3)
    for li in lib.K:
        assert nul.V[li].shape == lib.V[li].shape and not torch.allclose(nul.V[li], lib.V[li])
        assert nul.V[li][~nul.mask[li]].abs().sum() == 0
    z = null_library(lib, "zero")
    assert all(z.V[li].abs().sum() == 0 for li in lib.K)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name}: ok")
    print("OK")
