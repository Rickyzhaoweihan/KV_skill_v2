"""E2 tests (xenc/e2_read.py): deep-value / multi-source extensions of the D13 read.

CPU only, tiny random Qwen3_5ForCausalLM models (tests/_tiny.tiny_qwen35, test_state_interface._tiny_encoder)."""

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
from xenc.d13_read import D13Interface, D13Reader, EncTeacher, load_d13, save_d13  # noqa: E402
from xenc.e2_read import E2Interface, E2Reader, _DVHarvest, upgrade  # noqa: E402

torch.set_num_threads(2)
LMAP = {1: 2, 5: 5}
DV, MS, RS = {1: 5}, {1: 5}, {1: 5}


def _ids(n, seed):
    return torch.randint(3, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _setup(dv=DV, ms=MS, rs=RS):
    from test_state_interface import _tiny_encoder
    r, e = tiny_qwen35(), _tiny_encoder()
    B = {li: torch.randn(64, 48) * 0.1 for li in LMAP}
    base = D13Interface(e, LMAP, 64, 48, B=B, mlp_hidden=8)
    with torch.no_grad():
        base.A["1"].weight.add_(0.05 * torch.randn_like(base.A["1"].weight))
        base.g.fill_(0.9)
    B2 = {li: {"W": torch.randn(64, 48) * 0.1, "b": torch.randn(64) * 0.01} for li in ms}
    B3 = {li: {"W": torch.randn(64, 48) * 0.1, "b": torch.randn(64) * 0.01} for li in rs}
    itf = upgrade(base, e, dv, ms, B2, rs, B3)
    for li in rs:                                  # rs path A3 / M3 start as copies of the base A / MLP
        assert torch.equal(itf.E2["A3"][str(li)].weight, base.A[str(li)].weight)
    return r, e, B, base, itf


def _operator(e, itf, doc):
    Tc = EncTeacher(e, itf.enc_layers())
    Tc.rec, Tc.kv_only = {}, True
    hk = itf.harvest_hooks(e)
    with torch.no_grad():
        e(input_ids=torch.tensor([doc]), use_cache=False)
    op = {k: (r["k"], r["v"]) for k, r in Tc.rec.items()}
    if hk is not None:
        op.update(hk.finish("cpu"))
    Tc.detach()
    return op


def test_harvest_deep_values():
    r, e, B, base, itf = _setup()
    doc = _ids(20, 0)
    op = _operator(e, itf, doc)
    assert set(op) == {2, 5, "dv1"}
    from deltaskill.backbone_utils import decoder_layers
    L = decoder_layers(e)
    # recompute directly: h_5 = output of decoder layer 5 (pre final norm), captured with a hook
    cap = {}
    h = L[5].register_forward_hook(lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0]))
    with torch.no_grad():
        e(input_ids=torch.tensor([doc]), use_cache=False)
    h.remove()
    att = L[2].self_attn
    V = att.v_proj(L[2].input_layernorm(cap["h"])).view(20, -1, att.head_dim).transpose(0, 1)
    assert op["dv1"].shape == op[2][1].shape and torch.allclose(op["dv1"], V, atol=1e-6)
    assert float((op["dv1"] - op[2][1]).abs().max()) > 1e-3          # deep values differ from the layer's own


def test_zero_init_equals_d13_and_mixing_is_live():
    """alpha = beta = 0: the E2 reader reproduces the D13 reader exactly (step 0 == base); e2e gradients reach
    alpha / beta / B2; nonzero mixing changes the output; rec['y'] stays the unmixed read-out."""
    r, e, B, base, itf = _setup()
    doc, rows = _ids(25, 0), _ids(6, 1)
    op = _operator(e, itf, doc)
    x = torch.tensor([rows])
    R0 = D13Reader(r, base)
    R0.load({k: v for k, v in op.items() if isinstance(k, int)}, len(doc))
    R0.mode = "student"
    with torch.no_grad():
        ref = r(input_ids=x).logits
    R0.mode = "off"
    R0.detach()
    R = E2Reader(r, itf)
    R.load(op, len(doc))
    R.mode = "student"
    with torch.no_grad():
        got = r(input_ids=x).logits
    assert torch.allclose(got, ref, atol=1e-5), float((got - ref).abs().max())
    R.past = {}
    R.mode, R.need_logits, R.rec = "e2e", True, {}
    ep = itf.e2_params()
    with torch.enable_grad():
        out = r(input_ids=x)
        loss = out.logits.float().logsumexp(-1).mean()
        loss.backward()
    # rs distillation term: non-zero, and it trains the new addressing A3 even while gamma = 0
    from xenc.d13_read import EncTeacher as _ET
    Tc = _ET(e, itf.enc_layers())
    import copy as _copy
    Tc.rec, Tc.kv_only = {}, True
    with torch.no_grad():
        o_ = e(input_ids=torch.tensor([doc]), use_cache=True)
    Tc.rec, Tc.kv_only = {}, False
    with torch.no_grad():
        e(input_ids=x, past_key_values=_copy.deepcopy(o_.past_key_values), use_cache=True)
    trec = dict(Tc.rec)
    Tc.rec = None
    Tc.detach()
    for p in ep["A3"]:
        p.grad = None
    with torch.enable_grad():
        R.rec = {}
        r(input_ids=x)
        d3 = itf.extra_dist(R.rec, trec, op, len(doc))
        assert float(d3.detach()) > 0
        d3.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in ep["A3"])
    for p in ep["mix"]:                          # the zero-init mixing weights are live at step 0
        assert p.grad is not None and float(p.grad.abs().sum()) > 0
    for p in list(itf.E2["B2"].parameters()):     # ms B2 is gated by beta = 0: no gradient until beta moves
        assert p.grad is None or float(p.grad.abs().sum()) == 0
    assert torch.allclose(R.rec[1]["y"], R.rec[1]["y_mix"], atol=1e-6)     # zero init: mixed == unmixed
    with torch.no_grad():
        itf.E2["alpha"]["1"].fill_(0.5)
        itf.E2["beta"]["1"].fill_(0.3)
        itf.E2["gamma"]["1"].fill_(0.4)
    R.mode, R.rec = "e2e", {}
    for p in ep["mix"] + ep["B2"]:
        p.grad = None
    with torch.enable_grad():
        r(input_ids=x).logits.float().logsumexp(-1).mean().backward()
    for p in ep["mix"] + ep["B2"]:
        assert p.grad is not None and float(p.grad.abs().sum()) > 0
    R.rec = {}
    with torch.no_grad():
        out2 = r(input_ids=x).logits
    assert float((out2 - ref).abs().max()) > 1e-4
    assert float((R.rec[1]["y"] - R.rec[1]["y_mix"]).abs().max()) > 1e-5
    # incremental (student, generation) == full with nonzero mixing (3-tuple past for ms)
    R.mode, R.past = "student", {}
    with torch.no_grad():
        full = r(input_ids=x).logits[0, -1]
        R.past = {}
        o1 = r(input_ids=x[:, :4], use_cache=True)
        o2 = r(input_ids=x[:, 4:], past_key_values=o1.past_key_values, use_cache=True).logits[0, -1]
    assert torch.allclose(full, o2, atol=1e-4), float((full - o2).abs().max())
    R.mode = "off"
    R.detach()


def test_save_load_roundtrip_and_flag_off():
    r, e, B, base, itf = _setup()
    with torch.no_grad():
        itf.E2["alpha"]["1"].fill_(0.25)
        itf.E2["B2"]["1"].add_(0.1)
    d = tempfile.mkdtemp()
    save_d13(itf, os.path.join(d, "e2.pt"))
    itf2, _ = load_d13(os.path.join(d, "e2.pt"), e, B)
    assert isinstance(itf2, E2Interface) and itf2.dv == DV and itf2.ms == MS and itf2.rs == RS
    for k, v in itf.E2.state_dict().items():
        assert torch.allclose(itf2.E2.state_dict()[k], v), k
    assert torch.allclose(itf2.A["1"].weight, itf.A["1"].weight)
    # a plain D13 file still loads as a plain D13Interface (flag off: default behaviour unchanged)
    save_d13(base, os.path.join(d, "d13.pt"))
    itf3, _ = load_d13(os.path.join(d, "d13.pt"), e, B)
    assert type(itf3) is D13Interface and not hasattr(itf3, "dv")


def test_dv_only_and_ms_only():
    for dv, ms, rs in (({1: 5}, {}, {}), ({}, {1: 5}, {}), ({}, {}, {1: 5})):
        r, e, B, base, itf = _setup(dv, ms, rs)
        op = _operator(e, itf, _ids(15, 2))
        R = E2Reader(r, itf)
        R.load(op, 15)
        R.mode = "student"
        with torch.no_grad():
            r(input_ids=torch.tensor([_ids(5, 3)]))
        R.mode = "off"
        R.detach()



def test_operator_translator():
    """tr: zero init == the D13 read (step 0 == base); the key map acts on PRE-RoPE keys (RoPE⁻¹ of the translated
    keys == k + k W + b at every position); gradients reach W / b; save/load keeps it."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    from test_state_interface import _tiny_encoder
    r, e = tiny_qwen35(), _tiny_encoder()
    B = {li: torch.randn(64, 48) * 0.1 for li in LMAP}
    base = D13Interface(e, LMAP, 64, 48, B=B, mlp_hidden=8)
    itf = upgrade(base, e, {}, {}, None, {}, None, tr=True)
    doc, rows = _ids(25, 0), _ids(6, 1)
    op = _operator(e, itf, doc)
    K, V = op[2]
    Kt, Vt = itf.translate(1, K, V)
    assert torch.equal(Kt, K.float()) and torch.equal(Vt, V.float())       # zero init is bit-exact
    x = torch.tensor([rows])
    R0 = D13Reader(r, base)
    R0.load(op, len(doc))
    R0.mode = "student"
    with torch.no_grad():
        ref = r(input_ids=x).logits
    R0.mode = "off"
    R0.detach()
    R = E2Reader(r, itf)
    R.load(op, len(doc))
    R.mode = "student"
    with torch.no_grad():
        got = r(input_ids=x).logits
    assert torch.allclose(got, ref, atol=1e-4), float((got - ref).abs().max())
    R.mode, R.need_logits, R.rec, R.past = "e2e", True, {}, {}
    with torch.enable_grad():
        r(input_ids=x).logits.float().logsumexp(-1).mean().backward()
    tr = itf.e2_params()["tr"]
    assert tr and all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in tr)
    with torch.no_grad():
        for p in tr:
            p.normal_(0, 0.05)
    Kt, Vt = itf.translate(1, K, V)
    N = K.shape[1]
    pos = torch.arange(N).view(1, 1, N).expand(3, 1, N)
    cos, sin = itf.rot(K[:1].transpose(0, 1).reshape(1, N, -1).float(), pos)
    un = lambda z: apply_rotary_pos_emb(z[None], z[None], cos.float(), -sin.float())[1][0]   # noqa: E731
    kp = un(K.float())
    P = itf.E2["tr"]
    assert torch.allclose(un(Kt), kp + torch.einsum("hnd,hde->hne", kp, P["Wk1"]) + P["bk1"][:, None], atol=1e-4)
    assert float((Kt - K).abs().max()) > 1e-3
    R.mode = "off"
    R.detach()
    d = tempfile.mkdtemp()
    save_d13(itf, os.path.join(d, "tr.pt"))
    itf2, _ = load_d13(os.path.join(d, "tr.pt"), e, B)
    assert itf2.tr and torch.allclose(itf2.E2["tr"]["Wv5"], itf.E2["tr"]["Wv5"])



def test_c2_teacher_targets_and_chaining():
    """C2 self-read teacher on a tiny reader that is its own encoder (identity B): targets == a direct D13 e2e read of
    the reader's own prefill operator; installing it over an E2 student reader leaves the student's read unchanged;
    the /scratch-style cache returns the same targets; c2_loss is 0 against itself and has gradients."""
    import types
    from xenc.d13_read import D13Interface as D13I
    from xenc.doc_tasks import load_qwen_tokenizer
    from xenc.e2_c2 import C2Teacher, c2_loss
    from xenc.state_skill import full_layers
    _scripts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from fit_d13 import D13Rt
    tok = load_qwen_tokenizer()
    V = len(tok)
    r = tiny_qwen35(vocab=V, d=64, seed=0)
    e = tiny_qwen35(vocab=V, d=48, seed=1)
    L = full_layers(r)                                       # [1, 3, 5, 7]
    lm = {1: 1, 5: 5}
    hub = "Qwen/Qwen3.5-9B"
    docs = {"quality_x": "The cat sat on the mat. " * 20}
    # C2 interface: reader == encoder, identity B
    c2 = D13I(r, lm, 64, 64, B=None, mlp_hidden=8)
    with torch.no_grad():
        c2.A["1"].weight.add_(0.02 * torch.randn_like(c2.A["1"].weight))
    d = tempfile.mkdtemp()
    save_d13(c2, os.path.join(d, "c2.pt"))
    # student: E2 (tr) 4B-like interface over a different tiny encoder
    B = {li: torch.randn(64, 48) * 0.1 for li in lm}
    base = D13I(e, lm, 64, 48, B=B)
    itf = upgrade(base, e, {}, {}, None, {}, None, tr=True)
    rt = D13Rt(r, e, tok, hub, itf, tail=8, max_tokens=4000, lru=1)
    rt.docs = docs
    it = {"key": "k1", "question": "Where did the cat sit?", "choices": None, "answers": ["mat"]}
    rows = rt.rows("quality_x", it, [5, 6])
    op, n_op, _ = rt.operator("quality_x")
    R = rt.R
    def student():
        R.load(op, n_op)
        R.mode, R.rec = "e2e", {}
        with torch.no_grad():
            out = r(input_ids=torch.tensor([rows]), use_cache=False).logits
        rec = R.rec
        R.mode, R.rec = "off", {}
        return out, rec
    ref, _ = student()
    cache = os.path.join(d, "cache")
    t = C2Teacher(r, tok, hub, os.path.join(d, "c2.pt"), 8, 4000, cache, docs)
    y1 = t.targets("quality_x", rows, "k1")
    got, rec = student()
    assert torch.allclose(got, ref, atol=1e-6), "C2 patches change the student read"
    # direct C2 read: the reader's own prefill operator, identity read-out
    t2 = D13Rt(r, r, tok, hub, t.itf, 8, 4000, lru=1)
    t2.docs = docs
    op2, n2, _ = t2.operator("quality_x")
    assert n2 == n_op
    t2.R.load(op2, n2)
    t2.R.mode, t2.R.rec = "e2e", {}
    with torch.no_grad():
        r(input_ids=torch.tensor([rows]), use_cache=False)
    for li in lm:
        assert torch.allclose(y1[li], t2.R.rec[li]["y"].float(), atol=1e-5)
    t2.R.mode, t2.R.rec = "off", {}
    y1b = t.targets("quality_x", rows, "k1")
    assert t.n_hit == 1 and all(torch.allclose(y1b[li], y1[li], atol=1e-2) for li in lm)
    assert float(c2_loss({li: {"y": y1[li]} for li in lm}, y1)) < 1e-9
    R.load(op, n_op)
    R.mode, R.rec = "e2e", {}
    with torch.enable_grad():
        r(input_ids=torch.tensor([rows]), use_cache=False)
        l = c2_loss(R.rec, y1)
        assert float(l.detach()) > 0
        l.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in itf.e2_params()["tr"])
    R.mode, R.rec = "off", {}


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print(n, "ok")
    print("test_e2: all ok")
