"""CAP tests: interface-capacity levers on the D13 read (xenc/d13_read.py add_mlp_B / train_att, fit_d13.extend_itf).

CPU only, tiny random real Qwen3_5ForCausalLM. Checks: step 0 == base (zero-init write-back MLP, untouched att copies,
deep layers at gate 0), the new parts get non-zero gradients in the e2e mode, save/load and extend keep them."""

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
from test_d13 import _ids, _teacher  # noqa: E402
from xenc.d13_read import D13Interface, D13Reader, load_d13, save_d13  # noqa: E402

torch.set_num_threads(2)


def _setup():
    from test_state_interface import _tiny_encoder
    torch.manual_seed(0)
    r, e = tiny_qwen35(), _tiny_encoder()
    doc, rows = _ids(25, 0), _ids(6, 1)
    op, _, _ = _teacher(e, doc, rows)
    B = {li: torch.randn(64, 48) * 0.1 for li in (1, 5)}
    itf = D13Interface(e, {1: 2}, 64, 48, B={1: B[1]}, mlp_hidden=8)
    with torch.no_grad():
        for p in itf.M.parameters():
            p.add_(0.05 * torch.randn_like(p))
        itf.g.fill_(0.8)
    return r, e, doc, rows, op, B, itf


def _logits(r, itf, op, doc, rows):
    R = D13Reader(r, itf)
    R.load(op, len(doc))
    R.mode = "student"
    with torch.no_grad():
        y = r(input_ids=torch.tensor([rows])).logits
    R.mode = "off"
    R.detach()
    return y


def test_cap_step0_equal_grads_live_and_roundtrip():
    from fit_d13 import extend_itf
    r, e, doc, rows, op, B, itf = _setup()
    base = _logits(r, itf, op, doc, rows)
    # deep layer (gate 0) + write-back MLP (zero output) + trainable att copies: step 0 == base
    ext = extend_itf(itf, e, {1: 2, 5: 5}, B, 1)
    ext.add_mlp_B(16)
    att_ps = ext.train_att()
    assert ext.config()["mlp_B"] == 16 and ext.config()["att_trained"] is True
    assert "mlp_B" not in itf.config() and "att_trained" not in itf.config()     # untouched default config
    assert torch.allclose(_logits(r, ext, op, doc, rows), base, atol=1e-5)
    # e2e: the new parts receive non-zero gradients (all layers, incl. the new one through its gate)
    with torch.no_grad():
        ext.g[ext._gi[5]] = 0.3
    R = D13Reader(r, ext)
    R.load(op, len(doc))
    R.mode = "e2e"
    for p in ext.MB.parameters():
        p.requires_grad_(True)
    with torch.enable_grad():
        out = r(input_ids=torch.tensor([rows]))
        out.logits.float().logsumexp(-1).mean().backward()
    R.mode = "off"
    R.detach()
    for li in ("1", "5"):
        assert ext.MB[li][2].weight.grad is not None and float(ext.MB[li][2].weight.grad.abs().sum()) > 0
        assert ext.att[li].q_proj.weight.grad is not None and float(ext.att[li].q_proj.weight.grad.abs().sum()) > 0
        assert ext.att[li].o_proj.weight.grad is not None and float(ext.att[li].o_proj.weight.grad.abs().sum()) > 0
    assert len(att_ps) > 0
    # move the new parts, then save / load / extend keep them exactly
    with torch.no_grad():
        for p in list(ext.MB.parameters()) + att_ps:
            p.add_(0.02 * torch.randn_like(p))
    moved = _logits(r, ext, op, doc, rows)
    assert float((moved - base).abs().max()) > 1e-4
    d = tempfile.mkdtemp()
    save_d13(ext, os.path.join(d, "cap.pt"))
    blob = torch.load(os.path.join(d, "cap.pt"), weights_only=False)
    assert any(k.startswith("MB.") for k in blob["state_dict"]) and any(k.startswith("att.") for k in blob["state_dict"])
    back, _ = load_d13(os.path.join(d, "cap.pt"), e, B)
    assert back.mlp_B == 16 and back.att_trained
    assert torch.allclose(_logits(r, back, op, doc, rows), moved, atol=1e-5)
    # a plain interface still saves no att / MB keys (default behaviour unchanged)
    save_d13(itf, os.path.join(d, "plain.pt"))
    sd = torch.load(os.path.join(d, "plain.pt"), weights_only=False)["state_dict"]
    assert not any(k.startswith(("MB.", "att.")) for k in sd)
    # extending an interface that already has CAP parts keeps them (and the read) on the old layers
    small = D13Interface(e, {1: 2}, 64, 48, B={1: B[1]}, mlp_hidden=8)
    small.load_state_dict(itf.state_dict())
    small.add_mlp_B(16)
    small.train_att()
    with torch.no_grad():
        for p in list(small.MB.parameters()) + list(small.att.parameters()):
            p.add_(0.02 * torch.randn_like(p))
    s_log = _logits(r, small, op, doc, rows)
    big = extend_itf(small, e, {1: 2, 5: 5}, B, 1)
    assert big.att_trained and big.mlp_B == 16 and float(big.MB["5"][2].weight.abs().sum()) == 0.0
    assert torch.allclose(_logits(r, big, op, doc, rows), s_log, atol=1e-5)


if __name__ == "__main__":
    test_cap_step0_equal_grads_live_and_roundtrip()
    print("ALL CAP TESTS PASSED")
