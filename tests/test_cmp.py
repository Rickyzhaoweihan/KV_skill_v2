"""CMP tests: per-token operator bottleneck (xenc/cmp_op.py). CPU, tiny random real Qwen3_5ForCausalLM.

  (a) unrope is the exact inverse of RoPE, and rope == transformers' apply_rotary_pos_emb (partial rotary);
  (b) full-rank 'all' / 'per' specs reconstruct the operator exactly (fp32 z) and the D13 read through the wired
      bottleneck equals the base read; bf16 z is close;
  (c) truncation error grows as d shrinks; cross-layer ('L<e>') regression runs and reports R²;
  (d) gradients reach the decoders U, b through an e2e read; the raw cached operator is untouched;
  (e) save / load round trip; bytes_per_token accounting.
Run: python tests/test_cmp.py   (or pytest)
"""

from __future__ import annotations

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402
from xenc.cmp_op import (OpCompressor, OpStats, load_cmp, rope, rope_tables, save_cmp, unrope,  # noqa: E402
                         wire)
from xenc.d13_read import D13Interface, D13Reader, EncTeacher  # noqa: E402
from xenc.state_skill import full_layers  # noqa: E402

torch.set_num_threads(2)


def _enc():
    from test_state_interface import _tiny_encoder
    return _tiny_encoder()


def _op(m, n, seed):
    Tc = EncTeacher(m, full_layers(m))
    Tc.rec, Tc.kv_only = {}, True
    ids = torch.randint(3, 100, (1, n), generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        m(input_ids=ids, use_cache=False)
    op = {e: (r["k"], r["v"]) for e, r in Tc.rec.items()}
    Tc.rec = None
    Tc.detach()
    return op


def _setup(spec="all:999", n_docs=6):
    e = _enc()
    L = full_layers(e)
    itf = D13Interface(e, {li: li for li in L}, 48, 48, B=None)
    att = itf.att[str(L[0])]
    Hkv, hd = att.config.num_key_value_heads, att.head_dim
    st = OpStats(L, Hkv * hd)
    for i in range(n_docs):
        st.add(_op(e, 70, i), itf.rot, doc_id=f"d{i}")
    return e, L, itf, st, Hkv, hd


def test_rope_inverse_and_matches_transformers():
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    e = _enc()
    itf = D13Interface(e, {li: li for li in full_layers(e)}, 48, 48, B=None)
    k = torch.randn(2, 30, 16)
    cos, sin = rope_tables(itf.rot, 30, "cpu")
    assert cos.shape[-1] < 16                                   # partial rotary is exercised
    pos = torch.arange(30).view(1, 1, 30).expand(3, 1, 30)
    c2, s2 = itf.rot(torch.zeros(1, 30, 1), pos)
    _, ref = apply_rotary_pos_emb(k.unsqueeze(0), k.unsqueeze(0), c2, s2)
    assert torch.allclose(rope(k, cos, sin), ref[0], atol=1e-6)
    assert torch.allclose(unrope(rope(k, cos, sin), cos, sin), k, atol=1e-5)
    op = _op(e, 40, 0)
    K = op[full_layers(e)[0]][0]
    c, s = rope_tables(itf.rot, K.shape[1], "cpu")
    assert torch.allclose(rope(unrope(K, c, s), c, s), K, atol=1e-5)


def test_full_rank_reconstructs_and_read_equals_base():
    e, L, itf, st, Hkv, hd = _setup()
    D = len(L) * 2 * Hkv * hd
    op = _op(e, 50, 99)
    for spec in (f"all:{D}", f"per:{2 * Hkv * hd}"):
        m = OpCompressor.from_stats(st.state(), Hkv, hd, spec)
        m.quant = "fp32"
        new = m(op, itf.rot)
        for li in L:
            assert torch.allclose(new[li][0], op[li][0], atol=1e-4), (spec, float((new[li][0] - op[li][0]).abs().max()))
            assert torch.allclose(new[li][1], op[li][1], atol=1e-4)
        m.quant = "bf16"
        new = m(op, itf.rot)
        rel = float((new[L[0]][0] - op[L[0]][0]).norm() / op[L[0]][0].norm())
        assert rel < 2e-2, rel
    # the read through the wired bottleneck == the base read
    r = tiny_qwen35()
    itf2 = D13Interface(e, {1: L[0], 5: L[1]}, 64, 48, B={1: torch.randn(64, 48) * .1, 5: torch.randn(64, 48) * .1})
    R = D13Reader(r, itf2)
    rows = torch.randint(3, 100, (1, 7), generator=torch.Generator().manual_seed(5))

    class RT:
        pass
    rt = RT()
    rt.R, rt.itf = R, itf2
    R.mode = "student"
    R.load(op, 50)
    with torch.no_grad():
        base = r(input_ids=rows).logits
    m = OpCompressor.from_stats(st.state(), Hkv, hd, f"all:{D}")
    m.quant = "fp32"
    wire(rt, m, log=lambda *_: None)
    R.load(op, 50)
    with torch.no_grad():
        got = r(input_ids=rows).logits
    assert torch.allclose(base, got, atol=1e-4), float((base - got).abs().max())
    R.detach()


def test_truncation_error_monotone_and_cross_layer():
    e, L, itf, st, Hkv, hd = _setup()
    op = _op(e, 50, 77)
    errs = []
    for d in (4, 16, 64, 128):
        m = OpCompressor.from_stats(st.state(), Hkv, hd, f"all:{d}")
        m.quant = "fp32"
        new = m(op, itf.rot)
        errs.append(sum(float((new[li][1] - op[li][1]).norm() + (new[li][0] - op[li][0]).norm()) for li in L))
        assert 0 < m.info["var_kept"][0] <= 1.0 + 1e-9
    assert errs[0] > errs[1] > errs[2] > errs[3], errs
    m = OpCompressor.from_stats(st.state(), Hkv, hd, f"L{L[1]}:32")
    assert L[0] in m.info["cross_r2"] and m.src[L[0]] == [0]
    new = m(op, itf.rot)
    assert new[L[0]][0].shape == op[L[0]][0].shape
    assert m.bytes_per_token("bf16") == 64 and m.bytes_per_token("int8") == 34
    m = OpCompressor.from_stats(st.state(), Hkv, hd, f"{L[1]}|all:8")       # sub-layer compressor: L[0] passes through
    assert m.layers == [L[1]]
    new = m(op, itf.rot)
    assert torch.equal(new[L[0]][0], op[L[0]][0]) and not torch.allclose(new[L[1]][0], op[L[1]][0], atol=1e-4)


def test_grads_reach_decoders_and_cache_untouched():
    e, L, itf, st, Hkv, hd = _setup()
    r = tiny_qwen35()
    itf2 = D13Interface(e, {1: L[0], 5: L[1]}, 64, 48, B={1: torch.randn(64, 48) * .1, 5: torch.randn(64, 48) * .1})
    R = D13Reader(r, itf2)

    class RT:
        pass
    rt = RT()
    rt.R, rt.itf = R, itf2
    m = OpCompressor.from_stats(st.state(), Hkv, hd, "all:16")
    wire(rt, m, log=lambda *_: None)
    op = _op(e, 40, 3)
    K0 = op[L[0]][0].clone()
    R.load(op, 40)
    R.mode, R.need_logits = "e2e", True
    with torch.enable_grad():
        out = r(input_ids=torch.randint(3, 100, (1, 6)))
        out.logits.float().logsumexp(-1).mean().backward()
    for li in L:
        assert m.U[str(li)].grad is not None and float(m.U[str(li)].grad.abs().sum()) > 0
        assert float(m.b[str(li)].grad.abs().sum()) > 0
    assert torch.equal(op[L[0]][0], K0)
    d = tempfile.mkdtemp()
    with torch.no_grad():
        m.U[str(L[0])].add_(0.1)
    m.quant = "int8"
    save_cmp(m, os.path.join(d, "c.pt"))
    m2 = load_cmp(os.path.join(d, "c.pt"))
    assert m2.quant == "int8" and torch.equal(m2.U[str(L[0])], m.U[str(L[0])]) and m2.groups == m.groups
    a, b = m(op, itf.rot), m2(op, itf.rot)
    assert torch.allclose(a[L[0]][0], b[L[0]][0])
    R.detach()


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
    print("test_cmp: ALL OK")
