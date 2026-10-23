"""D16 tests: generalised reader-query read (xenc/d16_read.py). CPU only, tiny random Qwen3.5 models."""

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
from xenc.d13_read import D13Interface  # noqa: E402
from xenc.d16_read import (D16Interface, D16Reader, EncTeacher16, load_d16, pair_logits, save_d16)  # noqa: E402
from xenc.state_skill import full_layers  # noqa: E402

torch.set_num_threads(2)


def _ids(n, seed):
    return torch.randint(3, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _teacher(m, doc, rows, x_layers=()):
    import copy
    Tc = EncTeacher16(m, full_layers(m), x_layers)
    Tc.rec, Tc.mode = {}, "op"
    with torch.no_grad():
        out = m(input_ids=torch.tensor([doc]), use_cache=True)
    op = dict(Tc.rec)
    Tc.rec, Tc.mode = {}, "rows"
    with torch.no_grad():
        m(input_ids=torch.tensor([rows]), past_key_values=copy.deepcopy(out.past_key_values), use_cache=True)
    rec = dict(Tc.rec)
    Tc.rec = None
    Tc.detach()
    return op, rec


def test_default_equals_d13_and_identity():
    m = tiny_qwen35()
    doc, rows = _ids(30, 0), _ids(7, 1)
    L = full_layers(m)
    op, rec = _teacher(m, doc, rows, x_layers=L)
    d13 = D13Interface(m, {li: li for li in L}, 64, 64, B=None)
    for cfg in ({}, {"qhead": True}, {"tau": True}, {"moe": 2, "moe_rank": 4}, {"window": 3},
                {"feats": ["x", "L-1"]}, {"keymap": True}, {"kcat": {li: li for li in L}}, {"kcat": {li: L for li in L}}):
        itf = D16Interface(m, {li: li for li in L}, 64, 64, B=None, cfg=cfg)
        for li in L:
            x = rec[li]["x"].float()
            f = {"x": x, "L-1": torch.randn_like(x)}
            r16 = itf.read(li, f, op, len(doc), need_logits=True)
            r13 = d13.read(li, x, op[li]["K"], op[li]["V"], len(doc), need_logits=True)
            assert torch.allclose(r16["o"], r13["o"], atol=1e-4), (cfg, float((r16["o"] - r13["o"]).abs().max()))
            assert torch.allclose(r16["o"], rec[li]["out"].float(), atol=1e-4), cfg     # identity = encoder attention
            pl = pair_logits(itf.att[str(li)], itf.rot, x, op[li]["K"], len(doc))
            assert torch.allclose(pl, r13["logits"], atol=1e-4)


def test_incremental_window_and_kcat():
    m = tiny_qwen35()
    doc, rows = _ids(20, 0), _ids(6, 1)
    L = full_layers(m)
    op, rec = _teacher(m, doc, rows)
    li = L[1]
    itf = D16Interface(m, {li: li}, 64, 64, B=None, cfg={"window": 3, "kcat": {li: [L[2], L[0]]}, "tau": True})
    with torch.no_grad():
        itf.win_u[str(li)].normal_()
        itf.A[str(li)].weight[:, 64:].normal_(0, 0.05)
        itf.beta.normal_()
        itf.logtau.normal_(0, 0.2)
    x = rec[li]["x"].float()
    full = itf.read(li, {"x": x}, op, len(doc))["o"]
    r1 = itf.read(li, {"x": x[:4]}, op, len(doc))
    r2 = itf.read(li, {"x": x[4:5]}, op, len(doc), past=r1["past"])
    r3 = itf.read(li, {"x": x[5:]}, op, len(doc), past=r2["past"])
    assert torch.allclose(torch.cat([r1["o"], r2["o"], r3["o"]]), full, atol=1e-5)


def test_reader_modes_save_load():
    from test_state_interface import _tiny_encoder
    r, e = tiny_qwen35(), _tiny_encoder()
    doc, rows = _ids(25, 0), _ids(6, 1)
    op, rec = _teacher(e, doc, rows, x_layers=full_layers(e))
    lmap = {3: 2, 5: 5}
    B = {li: torch.randn(64, 48) * 0.1 for li in lmap}
    cfg = {"feats": ["x", "L-1", "q"], "mlp_hidden": 8, "moe": 2, "moe_rank": 4, "qhead": True, "keymap": True,
           "window": 2, "tau": True}
    itf = D16Interface(e, lmap, 64, 48, B=B, cfg=cfg, d_q=4 * 16)
    R = D16Reader(r, itf)
    R.load(op, len(doc))
    x = torch.tensor([rows])
    R.mode = "student"
    with torch.no_grad():
        a = r(input_ids=x, use_cache=True)
    assert set(R.past) == set(lmap) and R.past[3]["K"].shape[1] == 6
    R.mode = "teacher"
    R.inject = {li: itf.Bmap(li, rec[lmap[li]]["out"].float()) for li in lmap}
    with torch.no_grad():
        r(input_ids=x)
    assert R.fcap[3]["x"].shape == (6, 64) and R.fcap[3]["L-1"].shape == (6, 64) and R.fcap[3]["q"].shape == (6, 64)
    R.mode, R.need_logits = "e2e", True
    itf.trainable(train_B=True, train_g=True)
    with torch.enable_grad():
        out = r(input_ids=x)
        loss = out.logits.float().logsumexp(-1).mean() + sum(R.rec[li]["logits"].float().clamp_min(-50).mean() for li in lmap)
        loss.backward()
    for p in (itf.A["3"].weight, itf.B["3"], itf.g, itf.Qh["3"].weight, itf.Kp["3"].weight):
        assert p.grad is not None and float(p.grad.abs().sum()) > 0
    R.mode = "off"
    with torch.no_grad():
        b = r(input_ids=x).logits
    assert float((a.logits - b).abs().max()) > 1e-4
    with torch.no_grad():
        itf.A["3"].weight.add_(0.5)
    d = tempfile.mkdtemp()
    save_d16(itf, os.path.join(d, "x.pt"), {"k": 1})
    itf2, extra = load_d16(os.path.join(d, "x.pt"), e, B)
    assert extra == {"k": 1} and torch.allclose(itf2.A["3"].weight, itf.A["3"].weight)
    assert torch.allclose(itf2.Kp["3"].weight, itf.Kp["3"].weight) and itf2.cfg == itf.cfg
    R.detach()


def test_init_qhead_cross_device():
    """Regression (job 27939212): the encoder's attention weights live on another device than the fresh interface
    parameters (GPU encoder, CPU A at construction). Interface params on `meta`, encoder on CPU: init_qhead must
    move W to the parameters' device (it asserts the devices agree)."""
    m = tiny_qwen35()
    L = full_layers(m)
    itf = D16Interface(m, {L[0]: L[0]}, 64, 64, B=None, cfg={"qhead": True})
    s = str(L[0])
    itf.A[s].to("meta")
    itf.Qh[s].to("meta")
    itf.init_qhead(L[0])
    assert itf.Qh[s].weight.device.type == "meta" and itf.att[s].q_proj.weight.device.type == "cpu"
    if torch.cuda.is_available():                       # the real case, when a GPU is present
        itf2 = D16Interface(m.cuda(), {L[0]: L[0]}, 64, 64, B=None, cfg={"qhead": True})
        assert itf2.Qh[s].weight.device.type == "cpu"


def test_enc_layers16_kcat_lists():
    """build16 harvests the paired + all kcat encoder layers (kcat values are lists after D16Interface normalises)."""
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))
    from eval_d16 import enc_layers16
    lmap = {3: 3, 7: 7, 11: 7}
    assert enc_layers16(lmap, {"kcat": None}) == [3, 7]
    assert enc_layers16(lmap, {"kcat": {3: [7, 11, 23], 7: [3], 11: 15}}) == [3, 7, 11, 15, 23]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f()
            print("ok", n)
    print("ALL D16 TESTS PASSED")
