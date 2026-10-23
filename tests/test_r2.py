"""R2 tests (xenc/r2_read.py): query-side read forms of the D13 read (learned temperature, two-hop re-query, token
gate, context query, multi-query).

CPU only, tiny random Qwen3_5ForCausalLM models (tests/_tiny.tiny_qwen35, test_state_interface._tiny_encoder)."""

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
from xenc.d13_read import D13Interface, D13Reader, EncTeacher, load_d13, save_d13  # noqa: E402
from xenc.r2_read import R2Interface, R2Reader, parse_r2, upgrade_r2  # noqa: E402

torch.set_num_threads(2)
LMAP = {1: 2, 5: 5}
FORMS = {"temp": dict(temp=1), "hop": dict(hop=1), "tgate": dict(tgate=1), "ctx": dict(ctx=4),
         "mq": dict(mq=2, mq_rank=4), "all": dict(temp=1, hop=1, tgate=1, ctx=4, mq=2, mq_rank=4)}


def _ids(n, seed):
    return torch.randint(3, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _setup(**kw):
    from test_state_interface import _tiny_encoder
    torch.manual_seed(0)
    r, e = tiny_qwen35(), _tiny_encoder()
    B = {li: torch.randn(64, 48) * 0.1 for li in LMAP}
    base = D13Interface(e, LMAP, 64, 48, B=B, mlp_hidden=8)
    with torch.no_grad():
        for li in ("1", "5"):
            base.A[li].weight.add_(0.3 * torch.randn_like(base.A[li].weight))
            base.M[li][2].weight.normal_(0, 0.05)
        base.g.fill_(0.9)
    itf = upgrade_r2(base, e, parse_r2(**kw))
    return r, e, base, itf


def _operator(e, doc):
    Tc = EncTeacher(e, sorted(set(LMAP.values())))
    Tc.rec, Tc.kv_only = {}, True
    with torch.no_grad():
        e(input_ids=torch.tensor([doc]), use_cache=False)
    op = {k: (v["k"], v["v"]) for k, v in Tc.rec.items()}
    Tc.detach()
    return op


def _logits(r, itf, op, n, rows, reader_cls=None):
    R = (reader_cls or (R2Reader if isinstance(itf, R2Interface) else D13Reader))(r, itf)
    R.load(op, n)
    R.mode = "student"
    try:
        with torch.no_grad():
            return r(input_ids=torch.tensor([rows])).logits
    finally:
        R.mode = "off"
        R.detach()


def _perturb(itf, s=0.3):
    with torch.no_grad():
        for k, p in itf.R2.items():
            if k.startswith(("t", "b", "mb")):
                p.normal_(0, s)
            elif k.startswith("w"):
                p.normal_(0, 1.0)
            elif k.startswith("cU"):
                p.normal_(0, 0.3)


def test_parse_off_and_defaults():
    assert parse_r2() == {}
    c = parse_r2(temp=1)
    assert c["temp"] == 1 and c["hop"] == 0 and c["ctx"] == 0 and c["mq"] == 0


def test_zero_init_equals_base_exactly_and_forms_are_live():
    doc, rows = _ids(25, 0), _ids(7, 1)
    for name, kw in FORMS.items():
        r, e, base, itf = _setup(**kw)
        op = _operator(e, doc)
        a = _logits(r, base, op, len(doc), rows)
        b = _logits(r, itf, op, len(doc), rows)
        assert torch.equal(a, b), (name, float((a - b).abs().max()))
        _perturb(itf)
        c = _logits(r, itf, op, len(doc), rows)
        assert float((c - a).abs().max()) > 1e-4, name                   # live once the new parameters move
        itf.r2_on = False                                                  # ablation switch: back to base exactly
        assert torch.equal(_logits(r, itf, op, len(doc), rows), a), name


def test_temp_matches_explicit_reference():
    r, e, base, itf = _setup(temp=1)
    _perturb(itf)
    op = _operator(e, _ids(23, 0))
    x = torch.randn(5, 64)
    for li in (1, 5):
        K, V = op[LMAP[li]]
        out = itf.read(li, x, K, V, 23, need_logits=True)
        s0 = D13Interface.read(itf, li, x, K, V, 23, need_logits=True)["logits"]
        assert torch.allclose(out["logits"], s0)                           # raw logits returned (distillation side)
        Hkv, G = s0.shape[:2]
        p = torch.softmax(s0 * itf.R2[f"t{li}"].exp().view(Hkv, G, 1, 1), -1)
        att = itf.att[str(li)]
        hd = att.head_dim
        V_ = torch.cat([V.float(), att.v_proj(itf.map_x(li, x)).view(5, -1, hd).transpose(0, 1)], 1)
        o = torch.einsum("kgtc,kcd->kgtd", p, V_).reshape(-1, 5, hd).transpose(0, 1).reshape(5, -1)
        xe = itf.map_x(li, x)
        gate = torch.chunk(att.q_proj(xe).view(5, -1, hd * 2), 2, dim=-1)[1].reshape(5, -1)
        y = itf.Bmap(li, att.o_proj(o * torch.sigmoid(gate)))
        assert torch.allclose(out["y"], y, atol=1e-5), float((out["y"] - y).abs().max())


def test_hop_query_is_the_renormalised_residual_after_read1():
    """With beta = 1 the hop read equals a manual second read whose input is RMSNorm(h + own + g (y1 - own))."""
    r, e, base, itf = _setup(hop=1)
    with torch.no_grad():
        for li in itf.layers:
            itf.R2[f"b{li}"].fill_(1.0)
    op = _operator(e, _ids(23, 0))
    li = 5
    K, V = op[LMAP[li]]
    x, h, own = torch.randn(4, 64), torch.randn(4, 64), torch.randn(4, 64)
    from deltaskill.backbone_utils import decoder_layers
    nm = decoder_layers(r)[li].input_layernorm
    g = itf.gate(li)
    out = itf.read(li, x, K, V, 23, ext={"h": h, "own": own, "g": g, "norm": nm})
    y1 = D13Interface.read(itf, li, x, K, V, 23)["y"]
    x2 = nm(h + own + g * (y1 - own))
    # manual hop-2 read: base read of x2's query, but the own-row keys / values come from x (hop 1)
    att = itf.att[str(li)]
    hd = att.head_dim
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    xe, xe2 = itf.map_x(li, x), itf.map_x(li, x2)
    q2 = att.q_norm(torch.chunk(att.q_proj(xe2).view(4, -1, hd * 2), 2, dim=-1)[0]).transpose(0, 1)
    gate1 = torch.chunk(att.q_proj(xe).view(4, -1, hd * 2), 2, dim=-1)[1].reshape(4, -1)
    k = att.k_norm(att.k_proj(xe).view(4, -1, hd)).transpose(0, 1)
    v = att.v_proj(xe).view(4, -1, hd).transpose(0, 1)
    pos = torch.arange(23, 27).view(1, 1, 4).expand(3, 1, 4)
    cos, sin = itf.rot(xe.unsqueeze(0), pos)
    q2r, kr = apply_rotary_pos_emb(q2[None], k[None], cos, sin)
    from xenc.d13_read import _attend
    o2 = _attend(q2r[0], K.float(), V.float(), kr[0], v, att.scaling)["o"]
    y2 = itf.Bmap(li, att.o_proj(o2 * torch.sigmoid(gate1)))
    assert torch.allclose(out["y"], y2, atol=1e-4), float((out["y"] - y2).abs().max())
    assert torch.allclose(out["y_base"], y1, atol=1e-5)


def test_incremental_decode_equals_full_pass():
    doc, rows = _ids(25, 0), _ids(9, 1)
    for name in ("temp", "hop", "ctx", "mq", "all"):
        r, e, base, itf = _setup(**FORMS[name])
        _perturb(itf)
        op = _operator(e, doc)
        full = _logits(r, itf, op, len(doc), rows)[0]
        R = R2Reader(r, itf)
        R.load(op, len(doc))
        R.mode = "student"
        outs = []
        with torch.no_grad():
            o = r(input_ids=torch.tensor([rows[:5]]), use_cache=True)
            outs.append(o.logits[0])
            for t in rows[5:]:
                o = r(input_ids=torch.tensor([[t]]), past_key_values=o.past_key_values, use_cache=True)
                outs.append(o.logits[0])
        R.mode = "off"
        R.detach()
        inc = torch.cat(outs)
        assert torch.allclose(inc, full, atol=2e-4), (name, float((inc - full).abs().max()))


def _e2e_grads(r, itf, op, n, rows, ckpt):
    R = R2Reader(r, itf)
    R.load(op, n)
    R.mode, R.need_logits, R.rec, R.ckpt = "e2e", True, {}, ckpt
    itf.zero_grad(set_to_none=True)
    for p in itf.R2.parameters():
        p.requires_grad_(True)
    with torch.enable_grad():
        out = r(input_ids=torch.tensor([rows]))
        loss = out.logits.float().logsumexp(-1).mean() + sum(
            R.rec[li]["logits"].float().clamp_min(-50).mean() + R.rec[li]["y"].pow(2).mean() for li in LMAP)
        loss.backward()
    R.mode, R.rec = "off", {}
    R.detach()
    return {k: (p.grad.clone() if p.grad is not None else None) for k, p in itf.R2.items()}, float(loss)


def test_new_parameters_get_gradient_at_init_and_ckpt_matches():
    doc, rows = _ids(25, 0), _ids(7, 1)
    r, e, base, itf = _setup(**FORMS["all"])
    op = _operator(e, doc)
    g0, l0 = _e2e_grads(r, itf, op, len(doc), rows, ckpt=False)
    for k, gr in g0.items():
        if k.startswith(("t", "b", "w", "cU", "mb")):         # live at init (zero-init mixing / scales / gate weights)
            assert gr is not None and float(gr.abs().sum()) > 0, k
    g1, l1 = _e2e_grads(r, itf, op, len(doc), rows, ckpt=True)
    assert abs(l0 - l1) < 1e-5
    for k in g0:
        if g0[k] is None:
            assert g1[k] is None or float(g1[k].abs().sum()) == 0, k
        else:
            assert torch.allclose(g0[k], g1[k], atol=1e-5, rtol=1e-4), k


def test_ckpt_and_sub_ckpt_match_plain_grads_after_perturbation():
    """--mem_safe path (outer read checkpoint + one checkpoint per extra read) == plain autograd, with live extra reads."""
    doc, rows = _ids(25, 0), _ids(7, 1)
    r, e, base, itf = _setup(**FORMS["all"])
    _perturb(itf)
    op = _operator(e, doc)
    g0, l0 = _e2e_grads(r, itf, op, len(doc), rows, ckpt=False)
    assert itf.sub_ckpt is False
    g1, l1 = _e2e_grads(r, itf, op, len(doc), rows, ckpt=True)
    assert itf.sub_ckpt is True
    assert abs(l0 - l1) < 1e-5
    for k in g0:
        assert g0[k] is not None and float(g0[k].abs().sum()) > 0, k          # every R2 tensor is live once perturbed
        assert torch.allclose(g0[k], g1[k], atol=1e-5, rtol=1e-4), (k, float((g0[k] - g1[k]).abs().max()))


def test_save_load_roundtrip_and_docadapter():
    from xenc.d20_reg import DocAdapter
    doc, rows = _ids(25, 0), _ids(7, 1)
    r, e, base, itf = _setup(**FORMS["all"])
    _perturb(itf)
    op = _operator(e, doc)
    a = _logits(r, itf, op, len(doc), rows)
    d = tempfile.mkdtemp()
    save_d13(itf, os.path.join(d, "x.pt"), {"k": 1})
    B = {li: itf.B[str(li)].detach().clone() for li in LMAP}
    itf2, extra = load_d13(os.path.join(d, "x.pt"), e, B)
    assert isinstance(itf2, R2Interface) and itf2.r2 == itf.r2 and extra == {"k": 1}
    for k, p in itf.R2.items():
        assert torch.equal(p, itf2.R2[k]), k
    assert torch.equal(_logits(r, itf2, op, len(doc), rows), a)
    # per-document registration (D20) on top: zero-init deltas == the base read; R2 parameters frozen
    ad = DocAdapter(itf2, ["dA", "gate"], rank=4)
    assert all(not p.requires_grad for p in itf2.R2.parameters())
    assert torch.allclose(_logits(r, itf2, op, len(doc), rows), a, atol=1e-6)
    with torch.no_grad():
        for li in itf2.layers:
            itf2.D[f"Ua{li}"].normal_(0, 0.3)
    assert float((_logits(r, itf2, op, len(doc), rows) - a).abs().max()) > 1e-4    # dA flows through the R2 read
    assert ad.n_params() > 0


def test_private_generator_leaves_global_rng_untouched():
    from test_state_interface import _tiny_encoder
    e = _tiny_encoder()
    base = D13Interface(e, LMAP, 64, 48, B=None, mlp_hidden=0)
    torch.manual_seed(123)
    upgrade_r2(base, e, parse_r2(ctx=4, mq=2, mq_rank=4))
    a = torch.rand(3)
    torch.manual_seed(123)
    b = torch.rand(3)
    assert torch.equal(a, b)


def test_temp_init_equals_fixed_temperature():
    from xenc.r2_read import set_temp_init
    doc, rows = _ids(25, 0), _ids(7, 1)
    r, e, base, itf = _setup(temp=1)
    op = _operator(e, doc)
    set_temp_init(itf, 1.25)
    assert all(torch.allclose(itf.R2[f"t{li}"].exp(), torch.full((itf.r2_hq,), 1.25)) for li in itf.layers)
    # == the base read with every read logit x 1.25 (E4's temp1.25 variant)
    from xenc import e4_read
    e4_read.install()
    try:
        e4_read.set_from("temp1.25")
        ref = _logits(r, base, op, len(doc), rows)
    finally:
        e4_read.Variant.reset()
    assert torch.allclose(_logits(r, itf, op, len(doc), rows), ref, atol=1e-5)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            print("ok", k)
    print("test_r2: all passed")
