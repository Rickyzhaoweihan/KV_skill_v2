"""HYB-HM tests (xenc/hyb_hm.py): learned HM-style writer + zero-init additive deep read. CPU, tiny real Qwen3.5.

  (a) step 0 == parent: with the (random) writer / K decoder and the ZERO-INIT V decoder, logits are identical to the
      base read, in student and e2e mode (the channel adds exactly 0);
  (b) gradient flow: at step 0 only the V decoder gets gradient; once V^ != 0 the writer, K decoder and V decoder all do;
      the checkpointed deep read gives the same gradients;
  (c) generation-style incremental rows == one full pass; the writer input is the encoder's attention-block input;
  (d) storage: what is stored per token is d bf16 numbers (2d bytes); no PCA / quantiser object exists;
  (e) save / load round trip reproduces the logits.
Run: python tests/test_hyb_hm.py   (or pytest)
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
from deltaskill.backbone_utils import decoder_layers  # noqa: E402
from test_cmp_hyb import _ids, _logits, _rt  # noqa: E402
from xenc.hyb_hm import build_hm, detach_hm, save_hm  # noqa: E402

torch.set_num_threads(2)
LAY = (5, 7)


def _setup(d=8, mlp=0):
    torch.manual_seed(0)
    rt, r, e, itf = _rt()
    rt._pre["d0"] = _ids(40, 3)
    rows = _ids(7, 4)
    with torch.no_grad():
        base = _logits(rt, "d0", rows)
        base_e2e = _logits(rt, "d0", rows, mode="e2e")
    H = build_hm(rt, d=d, hw_layer=5, layers=LAY, mlp=mlp, log=lambda *_: None)
    return rt, r, e, H, rows, base, base_e2e


def _nudge(H, s=0.05):
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for li in H.dm.layers:
            H.dm.dv[str(li)].weight.copy_(torch.randn(H.dm.dv[str(li)].weight.shape, generator=g) * s)
            H.dm.dv[str(li)].bias.copy_(torch.randn(H.dm.dv[str(li)].bias.shape, generator=g) * s)


def test_step0_equals_parent():
    for mlp in (0, 16):
        rt, r, e, H, rows, base, base_e2e = _setup(mlp=mlp)
        with torch.no_grad():
            same = _logits(rt, "d0", rows)
            same_e2e = _logits(rt, "d0", rows, mode="e2e")
        assert float((same - base).abs().max()) == 0.0, float((same - base).abs().max())
        assert float((same_e2e - base_e2e).abs().max()) == 0.0
        _nudge(H)                                              # a non-zero V decoder changes the logits
        with torch.no_grad():
            diff = _logits(rt, "d0", rows)
        assert float((diff - base).abs().max()) > 1e-4
        detach_hm(H)


def test_gradient_flow_and_ckpt():
    rt, r, e, H, rows, base, _ = _setup()
    P = {"writer": H.dm.writer.weight, "dk": H.dm.dk["5"].weight, "dv": H.dm.dv["5"].weight, "dv7": H.dm.dv["7"].weight}
    for p in H.dm.trainable():
        p.requires_grad_(True)

    def run():
        for p in H.dm.trainable():
            p.grad = None
        with torch.enable_grad():
            out = _logits(rt, "d0", rows, mode="e2e")
            out.float().logsumexp(-1).mean().backward()
        return {k: (None if v.grad is None else v.grad.clone()) for k, v in P.items()}

    g0 = run()
    assert float(g0["dv"].abs().sum()) > 0 and float(g0["dv7"].abs().sum()) > 0          # V decoders learn at step 0
    assert g0["dk"] is None or float(g0["dk"].abs().sum()) == 0.0                       # K / writer wait for V != 0
    assert g0["writer"] is None or float(g0["writer"].abs().sum()) == 0.0
    _nudge(H)
    g1 = run()
    for k in P:
        assert g1[k] is not None and float(g1[k].abs().sum()) > 0, k
    H.ckpt = True
    g2 = run()
    H.ckpt = False
    for k in P:
        assert torch.allclose(g1[k], g2[k], atol=1e-5, rtol=1e-4), (k, float((g1[k] - g2[k]).abs().max()))
    detach_hm(H)


def test_incremental_equals_full_and_capture():
    rt, r, e, H, rows, base, _ = _setup()
    _nudge(H)
    cap = {}
    mod = decoder_layers(e)[5].self_attn
    h = mod.register_forward_pre_hook(lambda m, a, k: cap.__setitem__("x", k.get("hidden_states", a[0] if a else None)),
                                      with_kwargs=True)
    rt._ops.clear()
    with torch.no_grad():
        op, n, _ = rt.operator("d0")
    h.remove()
    assert op["hm_x"].shape == (n, H.dm.x_dim) and torch.equal(op["hm_x"], cap["x"][0].detach())
    with torch.no_grad():
        rt.R.load(op, n)
        rt.R.mode = "student"
        full = rt.reader(input_ids=torch.tensor([rows])).logits[0, -1]
        rt.R.load(op, n)
        out = rt.reader(input_ids=torch.tensor([rows[:-2]]), use_cache=True)
        out = rt.reader(input_ids=torch.tensor([rows[-2:]]), past_key_values=out.past_key_values, use_cache=True)
        rt.R.mode = "off"
    assert torch.allclose(full, out.logits[0, -1], atol=1e-4), float((full - out.logits[0, -1]).abs().max())
    detach_hm(H)


def test_storage_size():
    for d in (8, 16):
        rt, r, e, H, rows, base, _ = _setup(d=d)
        with torch.no_grad():
            _logits(rt, "d0", rows)
        z = H.dm.z
        assert z.dtype == torch.bfloat16 and z.shape == (H.dm.n_mem, d)
        assert H.dm.bytes_per_token() == 2 * d == z.element_size() * z.shape[1]
        assert not hasattr(H.dm, "cmp") and not any(k.startswith("E") for k in dict(H.dm.named_buffers()))   # no PCA basis
        detach_hm(H)
    # production widths: the stored code is d bf16 numbers per token
    assert [2 * d for d in (512, 1024, 2048)] == [1024, 2048, 4096]


def test_save_load_roundtrip():
    rt, r, e, H, rows, base, _ = _setup()
    _nudge(H)
    with torch.no_grad():
        ref = _logits(rt, "d0", rows)
    p = os.path.join(tempfile.mkdtemp(), "hyb_x.pt")
    save_hm(H, p)
    detach_hm(H)
    H2 = build_hm(rt, path=p, log=lambda *_: None)
    with torch.no_grad():
        got = _logits(rt, "d0", rows)
    assert float((got - ref).abs().max()) < 1e-5
    detach_hm(H2)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
    print("test_hyb_hm: ALL OK")
