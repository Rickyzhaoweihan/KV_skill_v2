"""L1 tests (xenc/l1_read.py): length-aware read forms of the D13 read.

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
from xenc.l1_read import L1Interface, parse_l1, upgrade_l1  # noqa: E402

torch.set_num_threads(2)
LMAP = {1: 2, 5: 5}
FORMS = {"logn": dict(logn=1, nref=8), "chunk": dict(chunk=5), "topk": dict(sel="topk", k=4),
         "hier": dict(sel="hier", block=4, nblk=2), "logn+hier": dict(logn=1, nref=8, sel="hier", block=4, nblk=2),
         "chunk+topk": dict(chunk=5, sel="topk", k=6)}


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
        base.g.fill_(0.9)
    itf = upgrade_l1(base, e, parse_l1(**kw))
    return r, e, base, itf


def _operator(e, doc):
    Tc = EncTeacher(e, sorted(set(LMAP.values())))
    Tc.rec, Tc.kv_only = {}, True
    with torch.no_grad():
        e(input_ids=torch.tensor([doc]), use_cache=False)
    op = {k: (v["k"], v["v"]) for k, v in Tc.rec.items()}
    Tc.detach()
    return op


def _logits(r, itf, op, n, rows):
    R = D13Reader(r, itf)
    R.load(op, n)
    R.mode = "student"
    with torch.no_grad():
        y = r(input_ids=torch.tensor([rows])).logits
    R.mode = "off"
    R.detach()
    return y


def _perturb(itf, s=0.4):
    with torch.no_grad():
        for p in itf.L1.values():
            p.normal_(0, s)


def test_parse_off_and_defaults_unchanged():
    assert parse_l1() == {} and parse_l1(sel="") == {}
    a = parse_l1(logn=1)
    assert a["logn"] == 1 and a["chunk"] == 0 and a["sel"] == ""


def test_zero_init_and_noop_selection_equal_base_exactly():
    doc, rows = _ids(25, 0), _ids(6, 1)
    for kw in (dict(logn=1, nref=8), dict(chunk=5), dict(logn=1, chunk=5), dict(sel="topk", k=25),
               dict(sel="topk", k=40), dict(sel="hier", block=4, nblk=7), dict(sel="hier", block=30, nblk=1)):
        r, e, base, itf = _setup(**kw)
        op = _operator(e, doc)
        a = _logits(r, base, op, len(doc), rows)
        b = _logits(r, itf, op, len(doc), rows)
        assert torch.equal(a, b), (kw, float((a - b).abs().max()))
        if itf.l1["logn"] or itf.l1["chunk"]:
            _perturb(itf)
            c = _logits(r, itf, op, len(doc), rows)
            assert float((c - a).abs().max()) > 1e-4, kw                  # live once sigma / c != 0
            itf.l1_on = False                                               # ablation switch: back to base exactly
            assert torch.equal(_logits(r, itf, op, len(doc), rows), a)


def test_active_selection_changes_the_read():
    doc, rows = _ids(25, 0), _ids(6, 1)
    for name in ("topk", "hier"):
        r, e, base, itf = _setup(**FORMS[name])
        op = _operator(e, doc)
        assert float((_logits(r, itf, op, len(doc), rows) - _logits(r, base, op, len(doc), rows)).abs().max()) > 1e-5


def _manual(itf, li, x, K, V, n, form):
    """Reference: full logits -> per-form probabilities written out explicitly (no shared helpers)."""
    out = D13Interface.read(itf, li, x, K, V, n, need_logits=True)        # base read gives the raw logits
    s0 = out["logits"]                                                     # [Hkv,G,T,C]
    Hkv, G, T, C = s0.shape
    s = s0.clone()
    if itf.l1["logn"]:
        m = (1 + itf.L1[f"s{li}"] * torch.log(torch.tensor(n / itf.l1["nref"]))).clamp_min(0.05)
        s = s * m.view(Hkv, G, 1, 1)
    if itf.l1["chunk"]:                                                    # explicit chunk-local softmax + merge
        cs, beta = itf.l1["chunk"], 1 + itf.L1[f"c{li}"].view(Hkv, G, 1)
        bounds = [(a, min(a + cs, n)) for a in range(0, n, cs)] + [(n, C)]
        lses = torch.stack([s[..., a:b].logsumexp(-1) for a, b in bounds], -1)      # [Hkv,G,T,nc]
        w = torch.softmax(beta.unsqueeze(-1) * lses, -1)
        p = torch.zeros_like(s)
        for i, (a, b) in enumerate(bounds):
            p[..., a:b] = w[..., i:i + 1] * torch.softmax(s[..., a:b], -1)
        s = torch.where(p > 0, p.log(), torch.full_like(p, float("-inf")))
    if itf.l1["sel"] == "topk":
        k = itf.l1["k"]
        so = s[..., :n]
        thr = so.sort(-1, descending=True).values[..., k - 1:k]
        s = torch.cat([so.masked_fill(so < thr, float("-inf")), s[..., n:]], -1)
    if itf.l1["sel"] == "hier":
        Bk, nb = itf.l1["block"], itf.l1["nblk"]
        so = s[..., :n]
        blocks = [(a, min(a + Bk, n)) for a in range(0, n, Bk)]
        sc = torch.stack([so[..., a:b].mean(-1) for a, b in blocks], -1)
        top = sc.topk(nb, -1).indices
        keep = torch.zeros_like(so, dtype=torch.bool)
        for i, (a, b) in enumerate(blocks):
            keep[..., a:b] = (top == i).any(-1, keepdim=True)
        s = torch.cat([so.masked_fill(~keep, float("-inf")), s[..., n:]], -1)
    return torch.softmax(s, -1), s0


def test_forms_match_explicit_reference():
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb  # noqa: F401
    doc = _ids(23, 0)
    for name, kw in FORMS.items():
        r, e, base, itf = _setup(**kw)
        _perturb(itf)
        op = _operator(e, doc)
        x = torch.randn(5, 64)
        for li in (1, 5):
            K, V = op[LMAP[li]]
            K, V = K.float(), V.float()
            got = itf.read(li, x, K, V, len(doc), need_logits=True)
            p, s0 = _manual(itf, li, x, K, V, len(doc), name)
            assert torch.allclose(got["logits"], s0, atol=1e-6), name              # raw logits returned
            # reconstruct o from p with the module's own projections
            att = itf.att[str(li)]
            hd = att.head_dim
            Hkv, G, T, C = p.shape
            xe = itf.map_x(li, x)
            v = att.v_proj(xe).view(T, -1, hd).transpose(0, 1)
            Vall = torch.cat([V, v], 1)
            o = torch.einsum("kgtc,kcd->kgtd", p, Vall).reshape(Hkv * G, T, hd).transpose(0, 1).reshape(T, -1)
            gate = torch.chunk(att.q_proj(xe).view(T, -1, hd * 2), 2, dim=-1)[1].reshape(T, -1)
            y = itf.Bmap(li, att.o_proj(o * torch.sigmoid(gate)))
            assert torch.allclose(got["y"], y, atol=1e-4), (name, li, float((got["y"] - y).abs().max()))


def test_gradients_reach_l1_params_and_A():
    doc, rows = _ids(25, 0), _ids(6, 1)
    for name in ("logn", "chunk", "logn+hier", "chunk+topk", "topk"):
        r, e, base, itf = _setup(**FORMS[name])
        op = _operator(e, doc)
        R = D13Reader(r, itf)
        R.load(op, len(doc))
        for ck in (False, True):
            R.ckpt = ck
            for p in itf.parameters():
                p.grad = None
            itf.trainable(train_B=True, train_g=True)
            for p in itf.l1_params():
                p.requires_grad_(True)
            R.mode, R.need_logits, R.rec = "e2e", True, {}
            with torch.enable_grad():
                out = r(input_ids=torch.tensor([rows]))
                loss = out.logits.float().logsumexp(-1).mean() + 0.1 * R.rec[5]["logits"][..., :len(doc)].mean()
                loss.backward()
            for p in itf.l1_params():
                assert p.grad is not None and float(p.grad.abs().sum()) > 0, (name, ck)
            assert float(itf.A["5"].weight.grad.abs().sum()) > 0, (name, ck)
            R.mode = "off"
        R.detach()


def test_incremental_decode_equals_full_pass():
    doc, rows = _ids(22, 0), _ids(7, 1)
    for name, kw in FORMS.items():
        r, e, base, itf = _setup(**kw)
        _perturb(itf)
        op = _operator(e, doc)
        R = D13Reader(r, itf)
        R.load(op, len(doc))
        R.mode = "student"
        with torch.no_grad():
            full = r(input_ids=torch.tensor([rows])).logits[0]
        R.load(op, len(doc))
        with torch.no_grad():
            o = r(input_ids=torch.tensor([rows[:4]]), use_cache=True)
            steps = [o.logits[0]]
            for t in rows[4:]:
                o = r(input_ids=torch.tensor([[t]]), past_key_values=o.past_key_values, use_cache=True)
                steps.append(o.logits[0])
        inc = torch.cat(steps, 0)
        assert torch.allclose(inc, full, atol=1e-4), (name, float((inc - full).abs().max()))
        R.mode = "off"
        R.detach()


def test_save_load_roundtrip():
    doc, rows = _ids(25, 0), _ids(6, 1)
    for name in ("logn+hier", "chunk+topk"):
        r, e, base, itf = _setup(**FORMS[name])
        _perturb(itf)
        op = _operator(e, doc)
        a = _logits(r, itf, op, len(doc), rows)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.pt")
            save_d13(itf, path)
            B = {li: itf.B[str(li)].detach().clone() for li in LMAP}
            it2, _ = load_d13(path, e, B)
        assert type(it2) is L1Interface and it2.l1 == itf.l1
        for k, p in itf.L1.items():
            assert torch.equal(p, it2.L1[k])
        assert torch.equal(a, _logits(r, it2, op, len(doc), rows))


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
    print("test_l1: all passed")
