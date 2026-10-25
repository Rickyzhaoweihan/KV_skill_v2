"""CMP hybrid tests (xenc/cmp_hyb.py): deep-layer native memory channel next to the D13 read. CPU, tiny real Qwen3.5.

  (a) DeepMem.read with the document's own K/V as memory == the reader's stock attention output of the full sequence
      [document; rows] at that layer (the channel IS native attention over extra columns);
  (b) HybRuntime on a real D13Rt: gates 0 -> logits identical to the base read; gates > 0 change them; e2e grads reach
      the gates and the cross-model decoders; generation-style incremental rows == one full pass;
  (c) oracle mode (memory = the reader's own true K/V of the document) runs and its memory equals a reader pass.
Run: python tests/test_cmp_hyb.py   (or pytest)
"""

from __future__ import annotations

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402
from deltaskill.backbone_utils import decoder_layers  # noqa: E402
from xenc.cmp_hyb import RKEY, DeepMem, build_hyb, detach_hyb  # noqa: E402
from xenc.cmp_op import OpCompressor, OpStats, save_cmp  # noqa: E402
from xenc.d13_read import D13Interface, EncTeacher  # noqa: E402

torch.set_num_threads(2)


def _ids(n, seed):
    return torch.randint(3, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _kv(m, layers, ids):
    Tc = EncTeacher(m, layers)
    Tc.rec, Tc.kv_only = {}, True
    with torch.no_grad():
        m(input_ids=torch.tensor([ids]), use_cache=False)
    op = {e: (r["k"], r["v"]) for e, r in Tc.rec.items()}
    Tc.rec = None
    Tc.detach()
    return op


def test_deep_read_is_native_attention():
    r = tiny_qwen35()
    li = 5
    doc, rows = _ids(20, 0), _ids(6, 1)
    cap = {}
    mod = decoder_layers(r)[li].self_attn
    h1 = mod.register_forward_pre_hook(lambda m, a, k: cap.__setitem__("x", (k.get("hidden_states", a[0] if a else None))),
                                       with_kwargs=True)
    h2 = mod.register_forward_hook(lambda m, a, k, o: cap.__setitem__("o", o[0]), with_kwargs=True)
    with torch.no_grad():
        r(input_ids=torch.tensor([doc + rows]), use_cache=False)
    h1.remove()
    h2.remove()
    mem = _kv(r, [li], doc)
    dm = DeepMem(r, [li])
    dm.mem, dm.n_mem = {li: mem[li]}, len(doc)
    with torch.no_grad():
        y, _ = dm.read(li, cap["x"][0, len(doc):].float())
    ref = cap["o"][0, len(doc):]
    assert torch.allclose(y, ref, atol=1e-4), float((y - ref).abs().max())


def _rt():
    from fit_d13 import D13Rt
    from test_state_interface import _tiny_encoder
    r, e = tiny_qwen35(), _tiny_encoder()
    itf = D13Interface(e, {1: 2, 3: 2}, 64, 48, B={1: torch.randn(64, 48) * .1, 3: torch.randn(64, 48) * .1})
    rt = D13Rt(r, e, None, "tiny", itf, tail=4, max_tokens=4000)
    return rt, r, e, itf


def _fit(r, e, itf, d=8):
    st = OpStats([5, RKEY + 5, RKEY + 7], 32)
    for i in range(8):
        ids = _ids(60, 10 + i)
        op = _kv(e, [5], ids)
        orr = _kv(r, [5, 7], ids)
        op.update({RKEY + k: v for k, v in orr.items()})
        st.add(op, itf.rot, doc_id=str(i))
    m = OpCompressor.from_stats(st.state(), 2, 16, f"5+{RKEY + 5}+{RKEY + 7}|L5:{d}")
    p = os.path.join(tempfile.mkdtemp(), "hyb_cmp.pt")
    save_cmp(m, p)
    return m, p


def _logits(rt, did, rows, mode="student"):
    op, n, _ = rt.operator(did)
    rt.R.load(op, n)
    rt.R.mode = mode
    try:
        return rt.reader(input_ids=torch.tensor([rows])).logits
    finally:
        rt.R.mode = "off"


def test_hyb_gate0_is_base_and_live():
    rt, r, e, itf = _rt()
    m, p = _fit(r, e, itf)
    assert set(m.info["cross_r2"]) == {RKEY + 5, RKEY + 7}
    rt._pre["d0"] = _ids(40, 3)
    rows = _ids(7, 4)
    with torch.no_grad():
        base = _logits(rt, "d0", rows)
    H = build_hyb(rt, p, layers=(5, 7), enc_layers=(5,), log=lambda *_: None)
    assert 5 in rt.Tc.layers
    with torch.no_grad():
        same = _logits(rt, "d0", rows)
    assert torch.equal(base, same) or float((base - same).abs().max()) < 1e-5
    assert set(H.dm.mem) == {5, 7} and H.dm.mem[5][0].shape == (2, 40 - 4, 16)
    H.dm.g.data.fill_(0.7)
    with torch.no_grad():
        diff = _logits(rt, "d0", rows)
    assert float((diff - base).abs().max()) > 1e-4
    for q in H.cmp.trainable() + [H.dm.g]:
        q.requires_grad_(True)
    with torch.enable_grad():
        out = _logits(rt, "d0", rows, mode="e2e")
        out.float().logsumexp(-1).mean().backward()
    assert H.dm.g.grad is not None and float(H.dm.g.grad.abs().sum()) > 0
    assert float(H.cmp.U[str(RKEY + 7)].grad.abs().sum()) > 0
    g_ref, u_ref = H.dm.g.grad.clone(), H.cmp.U[str(RKEY + 7)].grad.clone()     # checkpointed deep read: same grads
    H.dm.g.grad, H.cmp.U[str(RKEY + 7)].grad = None, None
    H.ckpt = True
    with torch.enable_grad():
        out = _logits(rt, "d0", rows, mode="e2e")
        out.float().logsumexp(-1).mean().backward()
    H.ckpt = False
    assert torch.allclose(H.dm.g.grad, g_ref, atol=1e-5) and torch.allclose(H.cmp.U[str(RKEY + 7)].grad, u_ref, atol=1e-5)
    # incremental (student, own past) == one pass
    with torch.no_grad():
        op, n, _ = rt.operator("d0")
        rt.R.load(op, n)
        rt.R.mode = "student"
        full = rt.reader(input_ids=torch.tensor([rows])).logits[0, -1]
        rt.R.load(op, n)
        out = rt.reader(input_ids=torch.tensor([rows[:-2]]), use_cache=True)
        out = rt.reader(input_ids=torch.tensor([rows[-2:]]), past_key_values=out.past_key_values, use_cache=True)
        rt.R.mode = "off"
    assert torch.allclose(full, out.logits[0, -1], atol=1e-4), float((full - out.logits[0, -1]).abs().max())
    detach_hyb(H)


def test_zcache_equals_live_encoding():
    """27B path: codes precomputed per document (scripts/cmp_zcache.py layout) give the same memory and logits as
    encoding the encoder's K/V live; zdim keeps a prefix of the PCA basis."""
    from xenc.cmp_hyb import zpath
    from xenc.cmp_op import rope_tables
    rt, r, e, itf = _rt()
    m, p = _fit(r, e, itf)
    rt._pre["d0"] = _ids(40, 3)
    rows = _ids(7, 4)
    H = build_hyb(rt, p, g=0.7, layers=(5, 7), enc_layers=(5,), log=lambda *_: None)
    with torch.no_grad():
        live = _logits(rt, "d0", rows)
        mem_live = {k: v[0].clone() for k, v in H.dm.mem.items()}
        op, n, _ = rt.operator("d0")
        cos, sin = rope_tables(itf.rot, n, "cpu")
        z = H.cmp.encode({5: op[5]}, cos, sin)[0]
    detach_hyb(H)
    zc = tempfile.mkdtemp()
    os.makedirs(os.path.dirname(zpath(zc, "d0")), exist_ok=True)
    torch.save({"z": z.half(), "n": n}, zpath(zc, "d0"))
    rt2, r2, e2, itf2 = _rt()
    itf2.load_state_dict(itf.state_dict())                 # same read interface (B is random in _rt)
    rt2._pre["d0"] = rt._pre["d0"]
    H2 = build_hyb(rt2, p, g=0.7, layers=(5, 7), enc_layers=(5,), zcache=zc, log=lambda *_: None)
    with torch.no_grad():
        got = _logits(rt2, "d0", rows)
    for k in mem_live:
        assert torch.allclose(H2.dm.mem[k][0].float(), mem_live[k].float(), atol=2e-2, rtol=2e-2)
    assert float((got - live).abs().max()) < 2e-2, float((got - live).abs().max())
    H2.zdim = 4                                            # prefix of the basis: runs, differs from the full code
    with torch.no_grad():
        trunc = _logits(rt2, "d0", rows)
    assert float((trunc - got).abs().max()) > 1e-5
    detach_hyb(H2)


def test_oracle_memory_is_reader_pass():
    rt, r, e, itf = _rt()
    rt._pre["d1"] = _ids(30, 5)
    H = build_hyb(rt, oracle=True, g=1.0, layers=(5, 7), enc_layers=(5,), log=lambda *_: None)
    with torch.no_grad():
        _logits(rt, "d1", _ids(5, 6))
    ref = _kv(r, [5, 7], rt.pre("d1")[:-4])
    assert torch.allclose(H.dm.mem[7][0], ref[7][0], atol=1e-5)
    detach_hyb(H)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
    print("test_cmp_hyb: ALL OK")
