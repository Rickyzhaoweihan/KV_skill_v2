"""D7: native-attention read (CPU, tiny randomly-initialised REAL Qwen3_5ForCausalLM).

  (a) harvest equivalence: derive_by_depth(harvest="attn_raw") == k_norm(k_proj(x)) / v_proj(x) recomputed
      through the module from the captured attention input (pre-RoPE), within 1e-3;
  (b) L/L branch == reference multi-head softmax(q Kᵀ·scaling) V with repeat_kv GQA grouping, the module's
      output gate and o_proj (null slot off, γ = g = 1);
  (c) no-op when the gate/gain is 0 (logits bit-identical to no library), and null logit ≫ lse ⇒ Δ ≈ 0;
  (d) shapes: attn_terms are [C,N], sum to 1 − null mass; slot_aux_loss backprops into the scalars;
  (e) cross-model maps (d_src != d_dst) + save/load round trip; null_like / subset keep shapes.
Run: python tests/test_native_attn.py   (or pytest)
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS))
sys.path.insert(0, str(_TESTS.parent))
from _tiny import tiny_qwen35  # noqa: E402

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from deltaskill.backbone_utils import decoder_layers  # noqa: E402
from xenc.depth_skill import TinyTokenizer, attn_layer_map, derive_by_depth  # noqa: E402
from xenc.native_attn import (NativeAttnInjector, NativeAttnInterface, NativeAttnLibrary,  # noqa: E402
                              attn_geometry, load_native_interface, reader_queries, save_native_interface)

DEPTHS3 = [0.25, 0.5, 0.75]           # 8 layers -> residual 2,4,6 -> full-attention 3,5,7
DOC = " ".join(f"w{i % 37}" for i in range(70))


def _setup(chunk=16):
    m = tiny_qwen35()
    tok = TinyTokenizer(vocab=100)
    cache = derive_by_depth(m, tok, DOC, DEPTHS3, chunk, harvest="attn_raw")
    lib = NativeAttnLibrary.from_doc_cache(cache, m, DEPTHS3)
    return m, tok, cache, lib


def _capture_attn_inputs(m, ids, layers):
    blocks, got, hs = decoder_layers(m), {}, []
    for li in layers:
        def pre(mod, args, kwargs, li=li):
            got[li] = kwargs.get("hidden_states", args[0] if args else None).detach().clone()
        hs.append(blocks[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
    with torch.no_grad():
        m(input_ids=ids, use_cache=False)
    for h in hs:
        h.remove()
    return got


def test_harvest_equivalence():
    m, tok, cache, lib = _setup()
    assert cache["harvest"] == "attn_raw" and cache["key_source"] == "k_norm"
    amap = attn_layer_map(m, DEPTHS3)
    assert sorted(amap.values()) == [3, 5, 7] and lib.layers == [3, 5, 7]
    ids = torch.tensor([tok.encode(DOC)])
    xs = _capture_attn_inputs(m, ids, list(amap.values()))
    blocks = decoder_layers(m)
    for rho, li in amap.items():
        att = blocks[li].self_attn
        g = attn_geometry(att)
        x = xs[li]
        T = x.shape[1]
        k = att.k_norm(att.k_proj(x).view(1, T, g["n_kv"], g["hd"]))[0].reshape(T, -1)
        v = att.v_proj(x)[0]
        Kc = torch.cat([c["K"][rho] for c in cache["chunks"]]).float()
        Vc = torch.cat([c["V"][rho] for c in cache["chunks"]]).float()
        assert Kc.shape == k.shape == (T, g["n_kv"] * g["hd"])
        assert (Kc - k).abs().max() < 1e-3 * max(1.0, float(k.abs().max())), float((Kc - k).abs().max())
        assert (Vc - v).abs().max() < 1e-3 * max(1.0, float(v.abs().max()))
        # library is the same tensor, padded per chunk
        Kl = lib.K[li].reshape(-1, Kc.shape[-1])[lib.mask[li].flatten()]
        assert torch.allclose(Kl, Kc)
        # partial RoPE at position > 0 changes the rotary dims: the harvest is the PRE-RoPE key
        assert g["hd"] == 16


def _reference(att, x, K, V, g):
    from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
    q, gate = reader_queries(att, x, g)                                   # [1,nq,T,hd]
    Kh = repeat_kv(K.permute(1, 0, 2).unsqueeze(0), g["groups"])           # [1,nq,S,hd]
    Vh = repeat_kv(V.permute(1, 0, 2).unsqueeze(0), g["groups"])
    a = torch.softmax(q.float() @ Kh.transpose(-1, -2) * g["scaling"], -1)
    o = (a @ Vh).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
    o = o * torch.sigmoid(gate.float())
    return att.o_proj(o)


def test_branch_matches_reference():
    m, tok, cache, lib = _setup()
    itf = NativeAttnInterface(lib.layers, lib.d_s, lib.d_s)
    inj = NativeAttnInjector(m, itf)
    inj.load_library(lib)
    ids = torch.tensor([tok.encode("what is w3 ?")])
    xs = _capture_attn_inputs(m, ids, lib.layers)
    for li in lib.layers:
        g = inj.geo[li]
        assert g["gated"] and g["groups"] == 2 and g["n_kv"] == 2
        K, V, _ = inj._operator(li, xs[li])
        with torch.no_grad():
            out = inj._branch(xs[li], li)
            ref = _reference(inj.attn[li], xs[li], K, V, g)
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4), float((out - ref).abs().max())
    inj.unload()
    inj.detach()


def test_noop_and_null():
    m, tok, cache, lib = _setup()
    ids = torch.tensor([tok.encode("what is w3 ?")])
    with torch.no_grad():
        base = m(input_ids=ids, use_cache=False).logits
    itf = NativeAttnInterface(lib.layers, lib.d_s, lib.d_s)
    inj = NativeAttnInjector(m, itf)
    inj.load_library(lib)
    with torch.no_grad():
        on = m(input_ids=ids, use_cache=False).logits
        assert (on - base).abs().max() > 1e-4                      # branch does something
        itf.gate.zero_()
        off = m(input_ids=ids, use_cache=False).logits
        assert torch.equal(off, base)                               # gain/gate 0 => exact no-op
        itf.gate.fill_(1.0)
        itf.null_logit.fill_(1e4)
        nul = m(input_ids=ids, use_cache=False).logits
        assert (nul - base).abs().max() < 1e-5                      # all mass on the null slot
    inj.unload()
    with torch.no_grad():
        assert torch.equal(m(input_ids=ids, use_cache=False).logits, base)   # unloaded => no-op
    inj.detach()


def test_attn_terms_and_grad():
    from xenc.losses import slot_aux_loss
    m, tok, cache, lib = _setup()
    itf = NativeAttnInterface(lib.layers, lib.d_s, lib.d_s, null_init=0.0)
    inj = NativeAttnInjector(m, itf)
    inj.load_library(lib)
    inj.record_attn = True
    inj.training_attn = True
    inj.attn_pos = -1
    ids = torch.tensor([tok.encode("what is w3 ?")])
    logits = m(input_ids=ids, use_cache=False).logits
    C, N = lib.mask[3].shape
    assert len(inj.attn_terms) == 3 and all(t.shape == (C, N) for t in inj.attn_terms)
    for li in lib.layers:
        A = inj.last_attn[li]
        assert A.shape == (C, N) and (A[~lib.mask[li]] == 0).all()
        assert abs(float(A.sum()) + inj.last_null[li] - 1.0) < 1e-4
    gm = torch.zeros(C, N, dtype=torch.bool)
    gm[1, 3] = True
    loss = slot_aux_loss(inj, gm) + logits[0, -1].logsumexp(-1) + sum(inj.mag_terms)
    loss.backward()
    for p in (itf.null_logit, itf.log_temp, itf.gate, itf.log_gain):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    inj.unload()
    inj.detach()


def test_maps_and_roundtrip():
    m, tok, cache, lib = _setup()
    d_dst = lib.d_s
    # pretend the encoder had half the KV width: project the library down, map back up
    P = torch.randn(d_dst, d_dst // 2) / (d_dst ** 0.5)
    small = NativeAttnLibrary({li: k @ P for li, k in lib.K.items()}, {li: v @ P for li, v in lib.V.items()},
                              {li: mk.clone() for li, mk in lib.mask.items()}, lib.chunk_meta, lib.rho_of)
    itf = NativeAttnInterface(lib.layers, d_dst // 2, d_dst, maps=True)
    try:
        NativeAttnInterface(lib.layers, d_dst // 2, d_dst, maps=False)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    inj = NativeAttnInjector(m, itf)
    inj.load_library(small)
    ids = torch.tensor([tok.encode("what is w3 ?")])
    out = m(input_ids=ids, use_cache=False).logits
    out[0, -1].logsumexp(-1).backward()
    assert itf.kmap["3"].weight.grad is not None and itf.vmap["7"].bias.grad is not None
    inj.unload()
    inj.detach()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "itf.pt")
        save_native_interface(itf, p, extra={"args": {"read": "native_attn"}})
        itf2, extra = load_native_interface(p)
        assert extra["args"]["read"] == "native_attn" and itf2.has_maps
        for (k1, v1), (k2, v2) in zip(itf.state_dict().items(), itf2.state_dict().items()):
            assert k1 == k2 and torch.equal(v1, v2)
    nl = lib.null_like("random", seed=1)
    assert all(nl.K[li].shape == lib.K[li].shape for li in lib.layers)
    sub = lib.subset([1])
    assert sub.n_chunks == 1 and sub.chunk_meta[0]["start"] == 16


def test_gold_slot_mask_compat():
    from xenc.gold_slots import gold_slot_mask
    m, tok, cache, lib = _setup()
    gm = gold_slot_mask(DOC, ["w5"], tok, 16, lib)
    assert gm is not None and gm.shape == lib.mask[3].shape and int(gm.sum()) >= 1


def test_runtime_build_native():
    """register_doc glue: DocRuntime (L/L, encodes itself with attn_raw) + rt.native + build_native."""
    import types
    sys.path.insert(0, str(_TESTS.parent / "scripts"))
    from register_doc import DocRuntime
    from xenc.native_attn import build_native, native_param_groups
    m = tiny_qwen35()
    tok = TinyTokenizer(vocab=100)
    with tempfile.TemporaryDirectory() as d:
        rt = DocRuntime(m, tok, "tiny", {"doc0": DOC}, d, "tinyq", 16, DEPTHS3, encoder_same=True)
        rt.harvest = "attn_raw"
        d_enc = rt.d_encoder(["doc0"])
        args = types.SimpleNamespace(native_layers="5,7", kv_map=None, init=None, null_init=-30.0, gain_init=0.0,
                                     out_gate=1, train_maps=0)
        itf, inj, save_fn = build_native(m, rt, args, d_enc, "cpu")
        rt.injector = inj
        lib = rt.library("doc0")
        assert isinstance(lib, NativeAttnLibrary) and lib.layers == [5, 7] and itf.inject_layers == [5, 7]
        inj.load_library(lib)
        out = m(input_ids=torch.tensor([tok.encode("what is w3 ?")]), use_cache=False).logits
        assert torch.isfinite(out).all() and len(inj.mag_terms) == 2
        groups = native_param_groups(itf, 1e-2, None)
        assert sum(p.numel() for g in groups for p in g["params"]) == 8
        inj.unload()
        inj.detach()


def test_rope_read_matches_in_context():
    """rope_mode='doc': the branch's per-head slot attention for a question placed (positionally) after the
    document == the real in-context attention of the last token over the document tokens (renormalized),
    computed by teacher_attention on the concatenated [doc; question] sequence."""
    from xenc.native_attn import teacher_attention
    m, tok, cache, lib = _setup()
    d_ids, q_ids = tok.encode(DOC), tok.encode("what is w3 ?")
    full = torch.tensor([d_ids + q_ids])
    N = len(d_ids)
    teach = teacher_attention(m, full, lib.layers)
    xs = _capture_attn_inputs(m, full, lib.layers)
    itf = NativeAttnInterface(lib.layers, lib.d_s, lib.d_s)
    inj = NativeAttnInjector(m, itf)
    inj.rope_mode = "doc"
    inj.record_attn = True
    inj.load_library(lib)
    for li in lib.layers:
        x = xs[li][:, N:]
        inj._qpos[li] = torch.arange(x.shape[1]).view(1, -1)
        with torch.no_grad():
            inj._branch(x, li)
        mine = inj.last_attn_heads[li]                                    # [n_q, S]
        ref = teach[li][:, :N]
        ref = ref / ref.sum(-1, keepdim=True)
        assert mine.shape == ref.shape
        assert torch.allclose(mine, ref, atol=2e-4, rtol=1e-3), float((mine - ref).abs().max())
        inj.rope_mode = "none"                                            # position-free differs
        inj._kv = {}
        with torch.no_grad():
            inj._branch(x, li)
        assert (inj.last_attn_heads[li] - ref).abs().max() > 1e-4
        inj.rope_mode = "doc"
        inj._kv = {}
    inj.unload()
    inj.detach()


if __name__ == "__main__":
    test_rope_read_matches_in_context()
    test_runtime_build_native()
    test_harvest_equivalence()
    test_branch_matches_reference()
    test_noop_and_null()
    test_attn_terms_and_grad()
    test_maps_and_roundtrip()
    test_gold_slot_mask_compat()
    print("test_native_attn: all passed")
