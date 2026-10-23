"""D12p (copy of D13 for the 27B reader). D13 tests: the reader-query read of the encoder's operator (xenc/d13_read.py).

CPU only, tiny random real Qwen3_5ForCausalLM (tests/_tiny.tiny_qwen35)."""

from __future__ import annotations

import copy
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402
from xenc.d12p_read import (D13Interface, D13Reader, EncTeacher, d13_terms, load_d13, save_d13,  # noqa: E402
                           teacher_logits)
from xenc.state_skill import full_layers  # noqa: E402

torch.set_num_threads(2)


def _ids(n, seed):
    return torch.randint(3, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _teacher(m, doc, rows):
    """Encoder prefix pass (operator) + cached rows pass -> (op, rec, cache-copy rows rec)."""
    Tc = EncTeacher(m, full_layers(m))
    Tc.rec, Tc.kv_only = {}, True
    with torch.no_grad():
        out = m(input_ids=torch.tensor([doc]), use_cache=True)
    op = {e: (r["k"], r["v"]) for e, r in Tc.rec.items()}
    cache = out.past_key_values
    Tc.rec, Tc.kv_only = {}, False
    with torch.no_grad():
        m(input_ids=torch.tensor([rows]), past_key_values=copy.deepcopy(cache), use_cache=True)
    rec = dict(Tc.rec)
    Tc.rec = {}
    with torch.no_grad():                      # same rows from the full sequence (no cache)
        m(input_ids=torch.tensor([doc + rows]), use_cache=False)
    full = dict(Tc.rec)
    Tc.rec = None
    Tc.detach()
    return op, rec, full


def test_teacher_cache_copy_matches_full_pass():
    m = tiny_qwen35()
    doc, rows = _ids(30, 0), _ids(7, 1)
    _, rec, full = _teacher(m, doc, rows)
    for e in rec:
        assert torch.allclose(rec[e]["out"], full[e]["out"][-7:], atol=1e-4)
        assert torch.allclose(rec[e]["q"], full[e]["q"][:, -7:], atol=1e-4)


def test_identity_read_equals_encoder_attention():
    """reader = encoder, A = I, B = I, fed the encoder's own layer input: the D13 read IS the encoder's attention."""
    m = tiny_qwen35()
    doc, rows = _ids(30, 0), _ids(7, 1)
    op, rec, _ = _teacher(m, doc, rows)
    L = full_layers(m)
    itf = D13Interface(m, {li: li for li in L}, 64, 64, B=None)
    for li in L:
        K, V = op[li]
        res = itf.read(li, rec[li]["x"].float(), K, V, len(doc), need_logits=True)
        assert torch.allclose(res["o"], rec[li]["out"].float(), atol=1e-4), float((res["o"] - rec[li]["out"]).abs().max())
        att = itf.att[str(li)]
        lt = teacher_logits(rec[li]["q"], K, rec[li]["k"], att.scaling)
        t = d13_terms(res["logits"], lt, res["y"], rec[li]["out"].float(), len(doc), metrics=True)
        assert t["kl"] < 1e-5 and t["js"] < 1e-6 and t["cos"] > 0.9999 and t["top5"] == 1.0


def test_incremental_read_equals_full_read():
    m = tiny_qwen35()
    doc, rows = _ids(20, 0), _ids(6, 1)
    op, rec, _ = _teacher(m, doc, rows)
    li = full_layers(m)[1]
    itf = D13Interface(m, {li: li}, 64, 64, B=None)
    K, V = op[li]
    x = rec[li]["x"].float()
    full = itf.read(li, x, K, V, len(doc))["o"]
    r1 = itf.read(li, x[:4], K, V, len(doc))
    r2 = itf.read(li, x[4:5], K, V, len(doc), past=r1["past"])
    r3 = itf.read(li, x[5:], K, V, len(doc), past=r2["past"])
    assert torch.allclose(torch.cat([r1["o"], r2["o"], r3["o"]]), full, atol=1e-5)


def test_reader_modes_and_e2e_grads():
    """Cross-model shapes (64-d reader, 48-d encoder): student / teacher / train / e2e modes run; e2e trains A, B, g;
    save/load round-trip."""
    from test_state_interface import _tiny_encoder
    r, e = tiny_qwen35(), _tiny_encoder()
    doc, rows = _ids(25, 0), _ids(6, 1)
    op, rec, _ = _teacher(e, doc, rows)
    lmap = {1: 2, 5: 5}
    B = {li: torch.randn(64, 48) * 0.1 for li in lmap}
    itf = D13Interface(e, lmap, 64, 48, B=B, mlp_hidden=8)
    R = D13Reader(r, itf)
    R.load(op, len(doc))
    x = torch.tensor([rows])
    R.mode = "student"
    with torch.no_grad():
        a = r(input_ids=x, use_cache=True)
    assert set(R.past) == set(lmap) and R.past[1][0].shape[1] == 6
    R.mode = "teacher"
    R.inject = {li: itf.Bmap(li, rec[lmap[li]]["out"].float()) for li in lmap}
    with torch.no_grad():
        r(input_ids=x)
    assert R.xcap[1].shape == (6, 64)
    R.mode, R.need_logits = "e2e", True
    params = itf.trainable(train_B=True, train_g=True)
    with torch.enable_grad():
        out = r(input_ids=x)
        loss = out.logits.float().logsumexp(-1).mean() + sum(R.rec[li]["logits"].float().clamp_min(-50).mean() for li in lmap)
        loss.backward()
    for p in (itf.A["1"].weight, itf.B["1"], itf.g):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0
    R.mode = "off"
    with torch.no_grad():
        b = r(input_ids=x).logits
    assert float((a.logits - b).abs().max()) > 1e-4         # the read changes the reader
    with torch.no_grad():
        itf.A["1"].weight.add_(0.5)
        itf.g.fill_(0.7)
    d = tempfile.mkdtemp()
    save_d13(itf, os.path.join(d, "x.pt"), {"k": 1})
    itf2, extra = load_d13(os.path.join(d, "x.pt"), e, B)
    assert extra == {"k": 1} and torch.allclose(itf2.A["1"].weight, itf.A["1"].weight)
    assert torch.allclose(itf2.g, itf.g) and torch.allclose(itf2.B["1"], itf.B["1"])
    R.detach()


def test_low_rank_delta_starts_at_init_and_merges_on_save():
    from test_state_interface import _tiny_encoder
    e = _tiny_encoder()
    lmap = {1: 2, 5: 5}
    B = {li: torch.randn(64, 48) * 0.1 for li in lmap}
    itf = D13Interface(e, lmap, 64, 48, B=B)
    x, o = torch.randn(5, 64), torch.randn(5, 48)
    y0, b0 = itf.map_x(1, x), itf.Bmap(1, o)
    itf.add_delta(4)
    assert torch.allclose(itf.map_x(1, x), y0) and torch.allclose(itf.Bmap(1, o), b0) and float(itf.delta_reg()) == 0.0
    with torch.no_grad():
        for p in itf.D.parameters():
            p.normal_(0, 0.1)
        itf.Bb["1"].fill_(0.3)
    y1, b1 = itf.map_x(1, x), itf.Bmap(1, o)
    assert float(itf.delta_reg()) > 0
    d = tempfile.mkdtemp()
    save_d13(itf, os.path.join(d, "x.pt"))
    itf2, _ = load_d13(os.path.join(d, "x.pt"), e, B)
    assert itf2.delta_rank == 0
    assert torch.allclose(itf2.map_x(1, x), y1, atol=1e-5) and torch.allclose(itf2.Bmap(1, o), b1, atol=1e-5)


def test_d12p_load_B_any_b2_intercept():
    """fit_d12p.load_B_any: a D12 encside_B2 single-source blob y = W(x − μx) + μy becomes B = W, Bb = μy − W μx."""
    import tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))
    from fit_d12p import load_B_any
    g = torch.Generator().manual_seed(0)
    W, mx, my = torch.randn(5, 3, generator=g), torch.randn(3, generator=g), torch.randn(5, generator=g)
    blob = {"format": "xenc.encside_B2.v1", "level": "post",
            "layers": {"7": {"src": [[2, 0]], "match": 2, "W": W, "mx": mx, "my": my}}}
    with tempfile.TemporaryDirectory() as d:
        torch.save(blob, os.path.join(d, "b.pt"))
        B, Bb = load_B_any(os.path.join(d, "b.pt"), [7], {7: 2})
    x = torch.randn(4, 3, generator=g)
    assert torch.allclose(x @ B[7].t() + Bb[7], (x - mx) @ W.t() + my, atol=1e-5)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f()
            print("ok", n)
    print("ALL D13 TESTS PASSED")
