"""xenc.depth_skill: ρ->layer mapping, derive_by_depth on TinyLM, for_reader views, cache round-trip.
Run: python tests/test_depth_skill.py   (or pytest). CPU only."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tiny import FakeConfig, TinyLM  # noqa: E402

from xenc import DEPTHS  # noqa: E402
from deltaskill.derive import injection_layers  # noqa: E402
from xenc.depth_skill import (DepthCollisionError, TinyTokenizer, cache_key,  # noqa: E402
                              chunk_to_kvskill, derive_by_depth, for_reader, load_doc_cache,
                              rho_layer_map, save_doc_cache, tiny_embed_fn)

_ROOT = Path(__file__).resolve().parents[1]


class _Fake:
    def __init__(self, L, d=8):
        self.config = FakeConfig(L, d)


def _doc(n_words: int, seed: int = 0) -> str:
    g = torch.Generator().manual_seed(seed)
    return " ".join(f"w{int(i)}" for i in torch.randint(0, 500, (n_words,), generator=g))


def test_rho_mapping():
    for L, want in [(24, [6, 12, 18, 23]), (32, [8, 16, 24, 31]), (64, [16, 32, 48, 63])]:
        m = rho_layer_map(_Fake(L), DEPTHS)
        assert list(m.values()) == want, (L, m)
        assert list(m.keys()) == DEPTHS
        assert sorted(m.values()) == injection_layers(_Fake(L), DEPTHS)
        assert list(rho_layer_map(L, DEPTHS).values()) == want     # int L accepted too
    for exc in (ValueError, AssertionError):
        try:
            rho_layer_map(_Fake(3), DEPTHS)
        except exc:
            pass
        else:
            raise AssertionError("L=3 with 4 depths must raise a collision")
    try:
        for_reader(chunk_to_kvskill({"K": {r: torch.zeros(2, 4) for r in DEPTHS}}, 4, "d", "e"), _Fake(3))
    except DepthCollisionError:
        pass
    else:
        raise AssertionError("for_reader must guard collisions")


def test_derive_by_depth_tiny():
    model, tok = TinyLM(d=32, n_layers=8), TinyTokenizer()
    text = _doc(100)
    d = derive_by_depth(model, tok, text, DEPTHS, chunk_tokens=32, doc_id="doc0", encoder="tiny")
    assert len(d["chunks"]) == 4                                    # ceil(100/32)
    assert [c["n_tokens"] for c in d["chunks"]] == [32, 32, 32, 4]
    assert [c["start"] for c in d["chunks"]] == [0, 32, 64, 96]
    assert d["d_s"] == 32 and d["depths"] == DEPTHS and d["chunk_tokens"] == 32
    assert d["doc_id"] == "doc0" and d["encoder"] == "tiny"
    for c in d["chunks"]:
        assert sorted(c["K"]) == DEPTHS and all(isinstance(r, float) for r in c["K"])
        for t in c["K"].values():
            assert t.dtype == torch.float16 and t.device.type == "cpu"
            assert t.shape == (c["n_tokens"], 32)
            assert torch.allclose(t.float().norm(dim=-1), torch.ones(t.shape[0]), atol=2e-3)
    # NO pooling: concatenated chunks == the whole-document hook capture, row-normalized
    emb = tiny_embed_fn(model)(torch.tensor([tok.encode(text)]))
    x, layers = emb, dict(zip(range(8), [None] * 8))
    for i, lin in enumerate(model.model.layers):
        pre = lin(x)
        layers[i] = pre[0]
        x = torch.tanh(pre)
    for rho, li in rho_layer_map(model, DEPTHS).items():
        full = torch.cat([c["K"][rho] for c in d["chunks"]]).float()
        ref = layers[li] / layers[li].norm(dim=-1, keepdim=True)
        assert torch.allclose(full, ref, atol=2e-3), rho
    # max_doc_tokens truncates
    d2 = derive_by_depth(model, tok, text, DEPTHS, chunk_tokens=32, max_doc_tokens=40)
    assert [c["n_tokens"] for c in d2["chunks"]] == [32, 8]
    # explicit embed_fn path gives the same result
    d3 = derive_by_depth(model, tok, text, DEPTHS, chunk_tokens=32, embed_fn=tiny_embed_fn(model))
    assert torch.equal(d3["chunks"][1]["K"][0.5], d["chunks"][1]["K"][0.5])


def test_for_reader_view():
    model, tok = TinyLM(d=32, n_layers=8), TinyTokenizer()
    d = derive_by_depth(model, tok, _doc(50), DEPTHS, chunk_tokens=64)
    s = chunk_to_kvskill(d["chunks"][0], d["d_s"], "doc", "tiny")
    assert s.K is s.V and s.meta["keyed_by"] == "rho" and s.meta["tied"] and s.meta["depths"] == DEPTHS
    v = for_reader(s, model)
    assert sorted(v.K) == injection_layers(model, DEPTHS) == [2, 4, 6, 7]
    for rho, li in rho_layer_map(model, DEPTHS).items():
        assert v.K[li].data_ptr() == s.K[rho].data_ptr()           # a view, no copy
        assert v.V[li].data_ptr() == s.K[rho].data_ptr()
        assert v.K[li] is s.K[rho]
    assert v.d_s == 32 and v.rank == 50


def test_save_load_roundtrip():
    model, tok = TinyLM(d=32, n_layers=8), TinyTokenizer()
    text = _doc(70, seed=3)
    d = derive_by_depth(model, tok, text, DEPTHS, chunk_tokens=32, doc_id="x", encoder="tiny")
    with tempfile.TemporaryDirectory() as td:
        key = cache_key(text, "Qwen/Qwen3.5-2B", 32, DEPTHS)
        assert key.endswith("_Qwen3.5-2B_c32_d0.25-0.5-0.75-1.0") and len(key.split("_")[0]) == 12
        path = os.path.join(td, key + ".pt")
        save_doc_cache(d, path)
        e = load_doc_cache(path)
    assert {k: v for k, v in e.items() if k != "chunks"} == {k: v for k, v in d.items() if k != "chunks"}
    assert len(e["chunks"]) == len(d["chunks"])
    for a, b in zip(d["chunks"], e["chunks"]):
        assert a["n_tokens"] == b["n_tokens"] and a["start"] == b["start"]
        assert sorted(a["K"]) == sorted(b["K"])
        for r in a["K"]:
            assert b["K"][r].dtype == torch.float16 and torch.equal(a["K"][r], b["K"][r])


def test_encode_docs_cli_tiny():
    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, str(_ROOT / "scripts" / "encode_docs.py"), "--tiny", "--cache_dir", td]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert out.returncode == 0, out.stderr
        assert "encoded 3 new" in out.stdout
        assert len([f for f in os.listdir(td) if f.endswith(".pt")]) == 3
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert out.returncode == 0 and "skipped 3 existing" in out.stdout, out.stdout + out.stderr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name}: ok")
    print("OK")
