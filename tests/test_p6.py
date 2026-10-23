"""P6 tests: residual-stream token injection extensions of the P3 channel (xenc/p3_tok.py: src unemb, adapters
lin / mlp / beta, injection after later decoder layers, gate floor) and register_d13's P6 helpers (oracle-p
override, logit-lens aux, α schedule), plus the eval-only native value injector (xenc/p6_native.py).
CPU only, tiny random Qwen3_5ForCausalLM models."""

from __future__ import annotations

import os
import sys
import tempfile
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402
from xenc.d13_read import D13Interface, D13Reader, EncTeacher, load_d13, save_d13  # noqa: E402
from xenc.p3_tok import TokInterface, TokReader, parse_tok, upgrade_tok  # noqa: E402

torch.set_num_threads(2)
LMAP = {1: 2, 5: 5}
GT = 11


def _ids(n, seed):
    return torch.randint(12, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _maps(V=100):
    nosp = torch.arange(V)
    nosp[50:60] = torch.arange(60, 70)
    gtm = torch.full((V,), GT)
    gtm[50:55] = torch.arange(70, 75)
    return nosp, gtm


def _setup(src="unemb", adapter="beta", inj="7", shifts="0,1", topk=3, heads="avg", rank=0, mlp_hidden=0, floor=0.0):
    from test_state_interface import _tiny_encoder
    torch.manual_seed(0)
    r, e = tiny_qwen35(), _tiny_encoder()
    B = {li: torch.randn(64, 48) * 0.1 for li in LMAP}
    base = D13Interface(e, LMAP, 64, 48, B=B, mlp_hidden=8)
    with torch.no_grad():
        base.A["5"].weight.add_(0.05 * torch.randn_like(base.A["5"].weight))
        base.g.fill_(0.9)
    cfg = parse_tok("5", rank, heads, shifts, topk, 0, 1.0, src, adapter, mlp_hidden, inj, floor)
    E = r.get_output_embeddings().weight if src == "unemb" else r.get_input_embeddings().weight
    itf = upgrade_tok(base, e, cfg, E)
    if src == "unemb":
        itf.set_tok_maps(*_maps(), GT)
    return r, e, B, base, itf


def _operator(e, doc):
    Tc = EncTeacher(e, sorted(set(LMAP.values())))
    Tc.rec, Tc.kv_only = {}, True
    with torch.no_grad():
        e(input_ids=torch.tensor([doc]), use_cache=False)
    op = {k: (v["k"], v["v"]) for k, v in Tc.rec.items()}
    Tc.detach()
    op["ids"] = torch.tensor(doc, dtype=torch.long)
    return op


def _logits(r, itf, op, n, R_cls, rows):
    R = R_cls(r, itf)
    R.load(op, n)
    R.mode = "student"
    with torch.no_grad():
        y = r(input_ids=torch.tensor([rows])).logits.float()
    R.mode = "off"
    R.detach()
    return y


def _set(itf, v=0.7):
    with torch.no_grad():
        for n in ("beta", "M2", "U"):
            if n in itf.TOK:
                for p in itf.TOK[n].parameters():
                    p.normal_(0, v)


CONFIGS = [dict(src="unemb", adapter="beta", inj="7"), dict(src="unemb", adapter="beta+mlp", inj="6,7", mlp_hidden=16),
           dict(src="emb", adapter="lin+mlp", inj="", rank=8, heads="kv", topk=0, mlp_hidden=16, shifts="-1,0,1"),
           dict(src="emb", adapter="mlp", inj="7", rank=8, heads="kv", topk=0, mlp_hidden=16, shifts="0")]


def test_zero_init_equals_base_and_live_after_change():
    for kw in CONFIGS:
        r, e, B, base, itf = _setup(**kw)
        doc, rows = _ids(25, 0), _ids(6, 1)
        op = _operator(e, doc)
        a = _logits(r, base, op, len(doc), D13Reader, rows)
        b = _logits(r, itf, op, len(doc), TokReader, rows)
        assert torch.equal(a, b), (kw, float((a - b).abs().max()))
        _set(itf)
        c = _logits(r, itf, op, len(doc), TokReader, rows)
        assert float((c - a).abs().max()) > 1e-4, kw
        itf.tok_on = False
        assert torch.equal(_logits(r, itf, op, len(doc), TokReader, rows), a), kw


def test_incremental_decode_equals_full_pass():
    for kw in CONFIGS:
        r, e, B, base, itf = _setup(**kw)
        _set(itf)
        doc, rows = _ids(20, 0), _ids(7, 1)
        op = _operator(e, doc)
        R = TokReader(r, itf)
        R.load(op, len(doc))
        R.mode = "student"
        with torch.no_grad():
            full = r(input_ids=torch.tensor([rows])).logits[0]
            R.load(op, len(doc))
            o = r(input_ids=torch.tensor([rows[:3]]), use_cache=True)
            st = [o.logits[0]]
            for t in rows[3:]:
                o = r(input_ids=torch.tensor([[t]]), past_key_values=o.past_key_values, use_cache=True)
                st.append(o.logits[0])
        assert torch.allclose(torch.cat(st), full, atol=1e-4), (kw, float((torch.cat(st) - full).abs().max()))
        R.mode = "off"
        R.detach()


def test_unemb_oracle_injection_equals_manual():
    """One-hot p on column c, beta on shift 0 only: the residual after layer 7 gains γ·β·scale·mean(W_U[id_c],
    W_U[nosp(id_c)], W_U[gtm(id_c)] if merged)."""
    r, e, B, base, itf = _setup(src="unemb", adapter="beta", inj="7", shifts="0", topk=1)
    with torch.no_grad():
        itf.TOK["beta"]["7"].fill_(2.0)
    doc = _ids(20, 0)
    doc[6] = 52                                      # an id with both a no-space and a merged variant
    rows = _ids(5, 1)
    op = _operator(e, doc)

    def onehot(li, p):
        q = torch.zeros_like(p)
        q[..., 6] = 1.0
        return q
    l_on = _logits(r, itf, op, len(doc), TokReader, rows)
    itf.tok_p_override = onehot
    l_or = _logits(r, itf, op, len(doc), TokReader, rows)
    itf.tok_p_override = None
    assert float((l_or - l_on).abs().max()) > 1e-4                 # the oracle p changes the injected residual
    Wu = r.get_output_embeddings().weight
    want = 2.0 * float(itf.TOK["proj"]["scale"]) * (Wu[52] + Wu[62] + Wu[72]) / 3.0
    tb = itf.doc_table(op["ids"], Wu)
    p = torch.zeros(2, 2, len(rows), len(doc) + len(rows))
    p[..., 6] = 1.0
    rr = itf.tok_read(p, tb)
    assert torch.allclose(itf.adapt("7", rr)[0], want, atol=1e-5)


def test_gradients_reach_adapters_and_addressing():
    for kw in CONFIGS[:2]:
        r, e, B, base, itf = _setup(**kw)
        doc, rows = _ids(25, 0), _ids(6, 1)
        op = _operator(e, doc)
        R = TokReader(r, itf)
        for ck in (False, True):
            R.ckpt = ck
            for p in itf.parameters():
                p.grad = None
            itf.trainable(train_B=True, train_g=True)
            for p in itf.tok_params()["U"] + itf.tok_params()["gt"]:
                p.requires_grad_(True)
            R.load(op, len(doc))
            R.mode, R.need_logits, R.rec = "e2e", True, {}
            with torch.enable_grad():
                loss = r(input_ids=torch.tensor([rows])).logits.float().logsumexp(-1).mean()
                loss.backward()
            g = [p.grad for p in itf.tok_params()["U"]]
            assert any(x is not None and float(x.abs().sum()) > 0 for x in g), (kw, ck)
            assert set(R.tok_z) == {str(l) for l in itf.tok_inj_layers}, R.tok_z.keys()
            R.mode, R.tok_r, R.tok_z = "off", None, {}
        _set(itf)
        for p in itf.parameters():
            p.grad = None
        R.ckpt = False
        R.load(op, len(doc))
        R.mode, R.need_logits, R.rec = "e2e", False, {}
        with torch.enable_grad():
            r(input_ids=torch.tensor([rows])).logits.float().logsumexp(-1).mean().backward()
        assert float(itf.A["5"].weight.grad.abs().sum()) > 0, kw          # through the channel's p
        R.mode = "off"
        R.detach()


def test_gate_floor_save_load_and_old_config():
    r, e, B, base, itf = _setup(src="unemb", adapter="beta", inj="7", floor=0.3)
    with torch.no_grad():
        itf.TOK["gt"]["7"].fill_(0.05)
    assert abs(float(itf.tok_gate("7")) - 0.05) < 1e-6
    with torch.no_grad():
        itf.TOK["floor_on"]["v"].fill_(1.0)
    assert abs(float(itf.tok_gate("7")) - 0.3) < 1e-6
    _set(itf)
    d = tempfile.mkdtemp()
    save_d13(itf, os.path.join(d, "t.pt"))
    itf2, _ = load_d13(os.path.join(d, "t.pt"), e, B)
    itf2.set_tok_maps(*_maps(), GT)
    assert itf2.config()["tok"] == itf.config()["tok"] and float(itf2.TOK["floor_on"]["v"]) == 1.0
    doc, rows = _ids(20, 0), _ids(5, 1)
    op = _operator(e, doc)
    assert torch.equal(_logits(r, itf, op, len(doc), TokReader, rows), _logits(r, itf2, dict(op), len(doc), TokReader, rows))
    # an old P3 config (no P6 keys) still builds the original channel
    old = {"layers": [5], "rank": 8, "heads": "kv", "shifts": [-1, 0, 1], "topk": 0, "topk_norm": 0, "temp": 1.0}
    t = TokInterface(e, LMAP, 64, 48, B=B, mlp_hidden=8, tok=old)
    assert t.tok_adapter == "lin" and not t.tok_inj_layers and set(dict(t.TOK.named_children())) == {"proj", "U", "gt"}


def test_register_p6_helpers():
    from register_d13 import p6_alpha, p6_aux, p6_override
    a = types.SimpleNamespace(tok_oracle_p=1.0, tok_oracle_steps=100)
    assert p6_alpha(a, 1) == 1.0 and abs(p6_alpha(a, 51) - 0.5) < 1e-9 and p6_alpha(a, 200) == 0.0
    a.tok_oracle_steps = 0
    assert p6_alpha(a, 999) == 1.0
    itf = types.SimpleNamespace(tok_layers=[5])
    p = torch.softmax(torch.randn(2, 2, 6, 10), -1)
    fn = p6_override({"alpha": 0.75, "plan": {4: [2, 3]}}, itf)
    q = fn(5, p)
    assert torch.equal(q[..., :4, :], p[..., :4, :]) and torch.equal(q[..., 5, :], p[..., 5, :])
    assert torch.allclose(q[..., 4, :].sum(-1), torch.ones(2, 2), atol=1e-6)
    assert torch.allclose(q[..., 4, 2], 0.25 * p[..., 4, 2] + 0.375)
    assert torch.equal(fn(1, p), p)
    Wu = torch.randn(30, 8)
    z = torch.randn(6, 8, requires_grad=True)
    aux = p6_aux({"Wu": Wu}, {"7": z}, {"rows": [(4, 3), (5, 7)]})
    want = torch.nn.functional.cross_entropy(z[[4, 5]] @ Wu.t(), torch.tensor([3, 7]))
    assert torch.allclose(aux, want, atol=1e-5)
    aux.backward()
    assert float(z.grad[[4, 5]].abs().sum()) > 0 and float(z.grad[:4].abs().sum()) == 0


def test_native_value_injector():
    """xenc/p6_native.py: g = 0 == base; injection at a row equals g·o_proj(v ⊙ σ(gate)) computed by hand."""
    from xenc.p6_native import NativeInjector
    r = tiny_qwen35()
    rows = _ids(6, 1)
    with torch.no_grad():
        base = r(input_ids=torch.tensor([rows])).logits
    P = NativeInjector(r, [3, 5], "emb")
    P.ids = torch.tensor(_ids(20, 0))
    P.cols = [4]
    with torch.no_grad():
        assert torch.equal(r(input_ids=torch.tensor([rows])).logits, base)        # g = 0
    P.g[3] = 1.5
    from deltaskill.backbone_utils import decoder_layers
    L = decoder_layers(r)
    cap = {}
    hk = L[3].self_attn.register_forward_hook(lambda m, a_, o: cap.__setitem__("o", o[0].clone()))
    hi = L[3].self_attn.register_forward_pre_hook(lambda m, a_, kw: cap.__setitem__("x", (a_[0] if a_ else kw["hidden_states"]).clone()),
                                                  with_kwargs=True)
    with torch.no_grad():
        out = r(input_ids=torch.tensor([rows])).logits
    hk.remove()
    hi.remove()
    assert float((out - base).abs().max()) > 1e-5
    att = L[3].self_attn
    x = cap["x"][0, -1:]
    hd = att.head_dim
    q, gate = torch.chunk(att.q_proj(x).view(1, -1, hd * 2), 2, dim=-1)
    v = att.v_proj(L[3].input_layernorm(r.get_input_embeddings().weight[int(P.ids[4])])).view(-1, hd)
    G = q.shape[1] // v.shape[0]
    o = v.repeat_interleave(G, 0).reshape(1, -1)
    want = 1.5 * att.o_proj(o * torch.sigmoid(gate.reshape(1, -1)))
    with torch.no_grad():
        P.g[3] = 0.0
        hk = L[3].self_attn.register_forward_hook(lambda m, a_, o: cap.__setitem__("o0", o[0].clone()))
        r(input_ids=torch.tensor([rows]))
        hk.remove()
    assert torch.allclose(cap["o"][0, -1] - cap["o0"][0, -1], want[0], atol=1e-5)
    assert torch.equal(cap["o"][0, :-1], cap["o0"][0, :-1])
    P.detach()


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f()
            print("ok", n)
    print("ALL P6 TESTS PASSED")
