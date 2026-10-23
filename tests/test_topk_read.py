"""D8: question-dependent top-k selection read (xenc/topk_read.py), CPU.

  (a) selection shapes + equals a hand computation (reader queries at the selection layer of the BARE reader,
      per-head q·Kᵀ with GQA grouping, max over heads, top-k over valid slots);
  (b) k >= N: read identical to LibraryInjector; restricted flat read over ALL slots of one chunk == the
      standard 1-chunk read; no-op at init (B = 0);
  (c) gold-hit stats (hit / chance vs brute force);
  (d) decode-step reuse: one selection pass per multi-token forward, none for single-token forwards
      (fake TinyAttnLM via logits(); real tiny Qwen3.5 via model.generate with a KV cache);
  (e) gradients reach the interface; subset / null copies carry the selection keys.
Run: python tests/test_topk_read.py   (or pytest)
"""

from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS))
sys.path.insert(0, str(_TESTS.parent))
sys.path.insert(0, str(_TESTS.parent / "scripts"))
from _tiny import TinyAttnLM, tiny_qwen35  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import xenc  # noqa: E402,F401
from deltaskill.backbone_utils import decoder_layers  # noqa: E402
from xenc.injector import LibraryInjector  # noqa: E402
from xenc.interface import make_interface  # noqa: E402
from xenc.library import Library  # noqa: E402
from xenc.topk_read import (TopKLibrary, TopKLibraryInjector, TopKSelector, attach_sel_keys,  # noqa: E402
                            reader_queries)

D, L, NKV, HD, G = 32, 8, 2, 8, 2          # reader width, layers, kv heads, head dim, GQA groups
LAYERS = [1, 3, 5, 7]
SEL = 3
DS = 24                                    # encoder d_s
DEP = [0.2, 0.4, 0.6, 0.8]


def _model():
    m = TinyAttnLM(d=D, n_layers=L, full_every=1, n_kv=NKV, head_dim=HD).eval().requires_grad_(False)
    torch.manual_seed(7)
    for b in m.model.layers:                # give the fake attention a GQA query map
        att = b.self_attn
        att.head_dim, att.num_key_value_groups = HD, G
        att.q_proj = nn.Linear(D, NKV * G * HD).requires_grad_(False)
    return m


def _lib(C=3, N=16, seed=0, n_last=None):
    g = torch.Generator().manual_seed(seed)
    chunks = []
    for c in range(C):
        n = N if (c < C - 1 or n_last is None) else n_last
        K = {r: torch.nn.functional.normalize(torch.randn(n, DS, generator=g), dim=-1) for r in DEP}
        chunks.append({"K": K, "start": c * N, "n_tokens": n})
    lib = Library(chunks, LAYERS, DEP)
    ac = {"chunks": [{"K": {0.25: torch.randn(ch["n_tokens"], NKV * HD, generator=g)}} for ch in chunks]}
    return attach_sel_keys(lib, ac, 0.25)


def _itf(nonzero_B=True, seed=1):
    itf = make_interface(D, DS, LAYERS)
    if nonzero_B:
        torch.manual_seed(seed)
        with torch.no_grad():
            for p in itf.B.parameters():
                p.normal_(0, 0.3)
            itf.gate_b.fill_(2.0)
    return itf


def _inj(m, itf, k, **kw):
    return TopKLibraryInjector(m, itf, TopKSelector(m, SEL, k=k), read_temp=12.0, **kw)


def _manual_idx(m, x, lib, k, pos=-1, inj=None):
    got = {}
    if inj is not None:
        inj.enabled = False
    att = decoder_layers(m)[SEL].self_attn
    def _cap(mod, args):
        got.setdefault("x", args[0].detach().clone())

    h = att.register_forward_pre_hook(_cap)
    with torch.no_grad():
        m(x)
    h.remove()
    if inj is not None:
        inj.enabled = True
    q = reader_queries(att, got["x"], {"hd": HD, "gated": False})[:, :, pos]     # [B, n_q, hd]
    Kf = lib.sel_K.reshape(-1, NKV, HD)
    s = torch.stack([q[:, h_] @ Kf[:, h_ // G].T for h_ in range(NKV * G)], 1)  # [B, n_q, S]
    s = s.amax(1).masked_fill(~lib.mask[LAYERS[0]].reshape(-1), float("-inf"))
    return s.topk(k, -1).indices


def test_selection_shapes_and_manual():
    m = _model()
    lib = _lib(n_last=9)                                      # padded last chunk: 41 valid of 48
    inj = _inj(m, _itf(), k=7)
    inj.load_library(lib)
    x = torch.randn(1, 11, D)
    with torch.no_grad():
        m(x)
    idx = inj.last_idx
    assert idx.shape == (1, 7)
    assert lib.mask[LAYERS[0]].reshape(-1)[idx[0]].all()     # only valid slots
    ref = _manual_idx(m, x, lib, 7, inj=inj)
    assert set(idx[0].tolist()) == set(ref[0].tolist())
    assert inj.tk.last_scores.shape == (1, 48)
    # read attention recorded on [C,N], zero off the selected set, sums to 1
    inj.record_attn = True
    with torch.no_grad():
        m(x)
    for li in LAYERS:
        A = inj.last_attn[li]
        assert A.shape == (3, 16)
        assert abs(float(A.sum()) - 1) < 1e-5
        off = torch.ones(48, dtype=torch.bool)
        off[idx[0]] = False
        assert float(A.reshape(-1)[off].abs().sum()) == 0.0
    inj.unload()
    assert inj.last_idx is None


def test_full_k_equivalence_and_noop():
    m = _model()
    x = torch.randn(1, 9, D)
    lib = _lib()
    itf = _itf()
    ref_inj = LibraryInjector(m, itf, read_temp=12.0)
    ref_inj.load_library(lib)
    with torch.no_grad():
        ref = m.logits(x)
    ref_inj.unload()
    ref_inj.detach()
    inj = _inj(m, itf, k=10_000)                              # k >= N -> exactly the library read
    inj.load_library(_lib())
    with torch.no_grad():
        out = m.logits(x)
    assert inj.n_select_passes == 0
    assert torch.allclose(out, ref, atol=1e-6)
    inj.unload()
    inj.detach()

    # one chunk: the flat restricted read over ALL its slots == the standard routed read
    class _Flat(TopKLibraryInjector):
        @property
        def full_read(self):
            return False

    lib1 = _lib(C=1, N=16, seed=3)
    ref_inj = LibraryInjector(m, itf, read_temp=12.0)
    ref_inj.load_library(lib1)
    with torch.no_grad():
        ref1 = m.logits(x)
    ref_inj.unload()
    ref_inj.detach()
    flat = _Flat(m, itf, TopKSelector(m, SEL, k=16), read_temp=12.0)
    flat.load_library(_lib(C=1, N=16, seed=3))
    with torch.no_grad():
        out1 = m.logits(x)
    assert flat.n_select_passes == 1 and flat.last_idx.shape == (1, 16)
    assert torch.allclose(out1, ref1, atol=1e-5)
    flat.unload()
    flat.detach()

    # no-op at init (B zero) with a genuine top-k selection
    with torch.no_grad():
        bare = m.logits(x)
    inj0 = _inj(m, _itf(nonzero_B=False), k=5)
    inj0.load_library(_lib())
    with torch.no_grad():
        out0 = m.logits(x)
    assert inj0.n_select_passes == 1
    assert torch.equal(out0, bare)
    inj0.detach()


def test_gold_hit():
    m = _model()
    sel = TopKSelector(m, SEL, k=3)
    n = 10
    sel.valid = torch.ones(n, dtype=torch.bool)
    sel.last_scores = torch.arange(n, dtype=torch.float32).flip(0).unsqueeze(0)   # slot 0 best ... slot 9 worst
    gold = torch.zeros(n, dtype=torch.bool)
    gold[[4, 7]] = True
    st = sel.gold_hit(gold, ks=[3, 5])
    assert st["n_gold"] == 2 and st["best_rank"] == 4
    assert st[3]["hit"] == 0.0 and st[5]["hit"] == 1.0 and st[5]["n_gold_sel"] == 1
    for k in (3, 5):
        brute = sum(any(i in (4, 7) for i in c) for c in combinations(range(n), k)) / math.comb(n, k)
        assert abs(st[k]["chance_any"] - brute) < 1e-9
        assert abs(st[k]["chance_frac"] - k * 2 / n) < 1e-12
    assert sel.gold_hit(torch.zeros(n, dtype=torch.bool)) == {}


def test_decode_reuse_fake():
    m = _model()
    inj = _inj(m, _itf(), k=4)
    inj.load_library(_lib())
    x = torch.randn(1, 8, D)
    with torch.no_grad():
        m.logits(x)                                           # fake path: block-0 hook
    assert inj.n_select_passes == 1
    idx0 = inj.last_idx.clone()
    with torch.no_grad():
        m.logits(torch.randn(1, 1, D))                        # decode step: reuse
        m(torch.randn(1, 1, D))                               # __call__ path decode step: reuse
    assert inj.n_select_passes == 1 and torch.equal(inj.last_idx, idx0)
    with torch.no_grad():
        m(x)                                                  # __call__ path: one pass (not two)
    assert inj.n_select_passes == 2 and torch.equal(inj.last_idx, idx0)
    inj.enabled = False
    with torch.no_grad():
        m(x)
    assert inj.n_select_passes == 2                           # disabled: no selection
    inj.detach()


def test_selection_position_and_grads():
    m = _model()
    itf = _itf()
    inj = _inj(m, itf, k=6)
    lib = _lib()
    inj.load_library(lib)
    x = torch.randn(1, 10, D)
    inj.sel_pos = -4                                          # training: last PROMPT position
    inj.training_attn, inj.attn_pos = True, -4
    out = m.logits(x)
    ref = _manual_idx(m, x, lib, 6, pos=-4, inj=inj)
    assert set(inj.last_idx[0].tolist()) == set(ref[0].tolist())
    assert len(inj.attn_terms) == len(LAYERS) and inj.attn_terms[0].shape == (3, 16)
    with torch.enable_grad():
        itf.requires_grad_(True)
        out = m.logits(x)
        out.square().mean().backward()
    assert itf.A.weight.grad is not None and float(itf.A.weight.grad.abs().sum()) > 0
    assert float(itf.B.weight.grad.abs().sum()) > 0
    inj.detach()


def test_subset_and_null_carry_keys():
    from eval_doc import null_library
    lib = _lib(n_last=9)
    sub = lib.subset([2, 0])
    assert isinstance(sub, TopKLibrary) and sub.sel_K.shape[:2] == sub.mask[LAYERS[0]].shape
    assert torch.equal(sub.sel_K[1], lib.sel_K[0, : sub.sel_K.shape[1]])
    nul = null_library(lib)
    assert isinstance(nul, TopKLibrary) and nul.sel_K is lib.sel_K
    try:
        attach_sel_keys(_lib(n_last=9), {"chunks": [{"K": {0.25: torch.zeros(5, NKV * HD)}}] * 3}, 0.25)
        raise AssertionError("misaligned harvests must raise")
    except ValueError:
        pass


def test_hf_generate_decode_reuse():
    """Real tiny Qwen3.5: top-level pre-hook selects on the prefill forward, decode steps (KV cache) reuse."""
    m = tiny_qwen35()
    layers = [1, 3, 5, 7]
    sel_layer = 3
    att = decoder_layers(m)[sel_layer].self_attn
    hd = att.head_dim
    n_kv = att.k_proj.out_features // hd
    g = torch.Generator().manual_seed(0)
    chunks = [{"K": {r: torch.nn.functional.normalize(torch.randn(12, 20, generator=g), dim=-1) for r in DEP},
               "start": 12 * c, "n_tokens": 12} for c in range(2)]
    lib = Library(chunks, layers, DEP)
    lib = attach_sel_keys(lib, {"chunks": [{"K": {0.5: torch.randn(12, n_kv * hd, generator=g)}} for _ in range(2)]}, 0.5)
    itf = make_interface(m.config.hidden_size, 20, layers)
    with torch.no_grad():
        for p in itf.B.parameters():
            p.normal_(0, 0.3)
    inj = TopKLibraryInjector(m, itf, TopKSelector(m, sel_layer, k=5), read_temp=12.0)
    inj.load_library(lib)
    ids = torch.tensor([[5, 9, 11, 3, 7, 2]])
    with torch.no_grad():
        out = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=4, do_sample=False,
                         pad_token_id=0)
    assert out.shape[1] == ids.shape[1] + 4
    assert inj.n_select_passes == 1                           # prefill only; 3 decode steps reused
    # selection == the bare reader's queries at the last prompt position
    got = {}
    def _cap(mod, a, kw):
        got.setdefault("x", kw["hidden_states"].detach().clone())

    h = att.register_forward_pre_hook(_cap, with_kwargs=True)
    inj.enabled = False
    with torch.no_grad():
        m(input_ids=ids, use_cache=False)
    inj.enabled = True
    h.remove()
    geo = inj.tk.geo
    q = reader_queries(att, got["x"], geo)[:, :, -1]
    Kf = lib.sel_K.reshape(-1, n_kv, hd)
    s = torch.stack([q[0, h_] @ Kf[:, h_ // geo["groups"]].T for h_ in range(geo["n_q"])]).amax(0)
    assert set(inj.last_idx[0].tolist()) == set(s.topk(5).indices.tolist())
    # training-style forward (logits_to_keep, use_cache=False): one pass
    with torch.no_grad():
        m(input_ids=ids, use_cache=False, logits_to_keep=2)
    assert inj.n_select_passes == 2
    inj.detach()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
