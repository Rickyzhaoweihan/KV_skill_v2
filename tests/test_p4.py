"""P4 tests: the output-space pointer / copy head (xenc/p4_ptr.py) on the D13 read.

CPU only, tiny random real Qwen3_5ForCausalLM reader (tests/_tiny.tiny_qwen35) + a different tiny encoder."""

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
from xenc.p4_ptr import PtrReader, copy_nll, parse_ptr, upgrade_ptr  # noqa: E402

torch.set_num_threads(2)
LMAP = {1: 2, 5: 5}
PREFIX = [7, 8, 9]                      # fake chat-template ids (S = 3)


def _ids(n, seed):
    return torch.randint(10, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _setup(layers="5", shifts="0,1", chain=1):
    from test_state_interface import _tiny_encoder
    r, e = tiny_qwen35(), _tiny_encoder()
    B = {li: torch.randn(64, 48) * 0.1 for li in LMAP}
    base = D13Interface(e, LMAP, 64, 48, B=B, mlp_hidden=8)
    with torch.no_grad():
        base.A["5"].weight.add_(0.05 * torch.randn_like(base.A["5"].weight))
        base.g.fill_(0.9)
    itf = upgrade_ptr(base, e, parse_ptr(layers, shifts, chain, PREFIX))
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


def _doc(n=30, seed=0):
    return PREFIX + _ids(n - len(PREFIX), seed)


def _logits(r, itf, op, n, R_cls, rows, mode="student"):
    R = R_cls(r, itf)
    R.load(op, n)
    R.mode, R.need_logits, R.rec = mode, False, {}
    with torch.no_grad():
        y = r(input_ids=torch.tensor([rows])).logits[0].float()
    R.mode = "off"
    R.detach()
    return y


def _randomize(itf, a=0.6, s=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        P = itf.PTR
        P["a"].fill_(a)
        P["b"].fill_(0.3)
        P["c"].fill_(0.2)
        P["w"].copy_(torch.randn(P["w"].shape, generator=g) * 0.1)
        for k in ("head", "layer", "logk", "shift"):
            P[k].copy_(torch.randn(P[k].shape, generator=g) * s)
        P["rho"].fill_(0.7)


def _raises(fn, msg):
    try:
        fn()
    except AssertionError as ex:
        assert msg in str(ex), str(ex)
        return
    raise RuntimeError(f"expected an AssertionError containing {msg!r}")


def test_gate_zero_equals_base():
    """a = 0 (init): the pointer reader's log P equals log_softmax of the parent read's logits (student and e2e), the
    greedy argmax is identical; ptr_off gives the parent's raw logits exactly."""
    for layers in ("5", "1,5"):
        r, e, B, base, itf = _setup(layers)
        doc, rows = _doc(), _ids(6, 1)
        op = _operator(e, doc)
        for mode in ("student", "e2e"):
            a = _logits(r, base, op, len(doc), D13Reader, rows, mode)
            b = _logits(r, itf, op, len(doc), PtrReader, rows, mode)
            assert torch.allclose(torch.log_softmax(a, -1), b, atol=1e-5), float((torch.log_softmax(a, -1) - b).abs().max())
            assert torch.equal(a.argmax(-1), b.argmax(-1))
        _randomize(itf)
        c = _logits(r, itf, op, len(doc), PtrReader, rows)
        assert float((torch.log_softmax(a, -1) - c).abs().max()) > 1e-3         # live once a > 0
        itf.ptr_on = False
        d = _logits(r, itf, op, len(doc), PtrReader, rows)
        assert torch.equal(a, d)
        # the reader patch is fully removed on detach (stock lm_head again)
        R = PtrReader(r, itf)
        R.detach()
        assert "forward" not in r.lm_head.__dict__


def test_copy_distribution_sums_to_one_and_matches_manual():
    r, e, B, base, itf = _setup("1,5", shifts="0,1", chain=0)
    _randomize(itf)
    doc, rows = _doc(), _ids(6, 1)
    op = _operator(e, doc)
    R = PtrReader(r, itf)
    R.load(op, len(doc))
    assert R.ptr_tab["S"] == len(PREFIX)
    R.mode = "student"
    keep = {}
    orig = itf.ptr_mix

    def spy(logits, h, R_):
        keep["pi"] = {li: (v[0].clone(), v[1].clone()) for li, v in R_.ptr_pi.items()}
        return orig(logits, h, R_)
    itf.ptr_mix = spy
    with torch.no_grad():
        y = r(input_ids=torch.tensor([rows])).logits[0]
    del itf.ptr_mix
    Pc = R.ptr_out["Pc"]
    assert torch.allclose(Pc.sum(-1), torch.ones(Pc.shape[0], dtype=Pc.dtype), atol=1e-5)
    assert float(Pc.min()) >= 0
    assert torch.allclose(y.exp().sum(-1), torch.ones(y.shape[0]), atol=1e-4)    # log P is normalised
    # manual: layer-mixed π over the body columns, offset mixture β over shifts {0, 1}
    lw = torch.softmax(itf.PTR["layer"], -1)
    pi = sum(lw[i] * keep["pi"][li][0] for i, li in enumerate(itf.ptr_layers))
    for v in keep["pi"].values():
        assert torch.allclose(v[0].sum(-1), torch.ones(v[0].shape[0]), atol=1e-5)
    beta = torch.softmax(itf.PTR["shift"], -1)
    S, N, V = len(PREFIX), len(doc), Pc.shape[1]
    man = torch.zeros(len(rows), V)
    for t in range(len(rows)):
        for j in range(N - S):
            for i, s in enumerate((0, 1)):
                c = S + j + s
                if c < N:
                    man[t, doc[c]] += beta[i] * pi[t, j]
    man = man / man.sum(-1, keepdim=True)
    assert torch.allclose(Pc.float(), man, atol=1e-5)
    # the mixture: P = (1-g) P_reader + g P_copy
    g = R.ptr_out["g"]
    R.mode = "off"
    with torch.no_grad():
        R.mode = "student"
        R.load(op, len(doc))
        itf.ptr_on = False
        raw = r(input_ids=torch.tensor([rows])).logits[0].float()
        itf.ptr_on = True
    want = ((1 - g).unsqueeze(-1) * torch.softmax(raw, -1) + g.unsqueeze(-1) * Pc.float()).log()
    assert torch.allclose(y, want, atol=1e-4)
    assert float(g.min()) > 0 and float(g.max()) < 1
    R.mode = "off"
    R.detach()


def test_ids_align_with_columns():
    """D13Rt.operator with a pointer interface stores ids == the exact prefill ids (column c = ids[c]); the pointer
    table's token at body column j and shift s is ids[S + j + s]; misaligned ids are rejected."""
    from fit_d13 import D13Rt
    r, e, B, base, itf = _setup("5")
    doc = _doc(34, 3)

    class Rt(D13Rt):
        def pre(self, did):
            return doc
    rt = Rt(r, e, None, "x", itf, tail=4, max_tokens=4000)
    op, n, _ = rt.operator("d")
    assert n == 30 and op["ids"].tolist() == doc[:-4]
    assert all(op[e_][0].shape[1] == n for e_ in set(LMAP.values()))
    R = rt.R
    assert isinstance(R, PtrReader)
    R.load(op, n)
    S = R.ptr_tab["S"]
    assert S == len(PREFIX)
    for i, s in enumerate(itf.ptr_shifts):
        for j in range(n - S):
            c = S + j + s
            assert bool(R.ptr_tab["valid"][i][j]) == (c < n)
            if c < n:
                assert int(R.ptr_tab["tok"][i][j]) == doc[c]
    # a truncated (cut-from-the-left) operator has no template prefix: every column is body
    op2 = dict(op)
    op2["ids"] = op["ids"].clone()
    op2["ids"][0] = 99
    R.load(op2, n)
    assert R.ptr_tab["S"] == 0
    bad = dict(op)
    bad["ids"] = op["ids"][:-1]
    _raises(lambda: R.load(bad, n), "do not align")
    # a plain D13 interface stores no ids (default behaviour unchanged)
    rt2 = Rt(r, e, None, "x", base, tail=4, max_tokens=4000)
    op3, _, _ = rt2.operator("d")
    assert "ids" not in op3
    for x in (rt.R, rt2.R):
        x.detach()
    rt.Tc.detach()
    rt2.Tc.detach()


def test_incremental_decode_equals_full_pass():
    """Generation (prefill + 1-token steps, chain state carried across forwards) == one full pass over the same rows,
    in student mode and against the teacher-forced e2e pass."""
    for layers, shifts, chain in (("5", "0,1", 1), ("1,5", "0", 1), ("1,5", "-1,0,1", 0), ("5", "1", 1)):
        r, e, B, base, itf = _setup(layers, shifts, chain)
        _randomize(itf)
        doc, rows = _doc(24, 0), _ids(8, 1)
        rows = rows[:3] + doc[10:15]          # copy a document span: the chain is exercised
        op = _operator(e, doc)
        R = PtrReader(r, itf)
        R.load(op, len(doc))
        R.mode = "student"
        with torch.no_grad():
            full = r(input_ids=torch.tensor([rows])).logits[0]
        R.load(op, len(doc))
        with torch.no_grad():
            o = r(input_ids=torch.tensor([rows[:3]]), use_cache=True)
            steps = [o.logits[0]]
            for t in rows[3:]:
                o = r(input_ids=torch.tensor([[t]]), past_key_values=o.past_key_values, use_cache=True)
                steps.append(o.logits[0])
        inc = torch.cat(steps, 0)
        assert torch.allclose(inc, full, atol=1e-4), (layers, shifts, chain, float((inc - full).abs().max()))
        R.load(op, len(doc))
        R.mode, R.need_logits, R.rec = "e2e", True, {}
        with torch.no_grad():
            tf = r(input_ids=torch.tensor([rows]), logits_to_keep=5).logits[0]
        assert torch.allclose(tf, full[-5:], atol=1e-4), float((tf - full[-5:]).abs().max())
        R.mode = "off"
        R.detach()


def test_chain_moves_mass_to_the_next_column():
    """Next-position pointer: after the row whose input token matches document column j, mass moves to j + 1."""
    r, e, B, base, itf = _setup("5", "0", 1)
    _randomize(itf)
    doc = _doc(24, 0)
    op = _operator(e, doc)
    R = PtrReader(r, itf)
    R.load(op, len(doc))
    S, N = R.ptr_tab["S"], len(doc)
    Nb = N - S
    j = 6
    tok = doc[S + j]
    assert doc.count(tok) == 1
    pi = torch.full((2, Nb), 1.0 / Nb)
    pi[0] = 0.0
    pi[0, j] = 1.0                                           # row 0 points exactly at column j
    R.ptr_pi = {5: (pi, torch.zeros(2))}
    R.ptr_x = torch.tensor([50, tok])                       # row 1's input = the token copied from column j
    R.mode, R.ptr_fresh = "e2e", True
    with torch.no_grad():
        itf.PTR["shift"].zero_()
        lp = itf.ptr_mix(torch.zeros(2, 100), torch.randn(2, 64), R)
    Pc = R.ptr_out["Pc"]
    mu = float(itf.PTR["rho"].detach().exp())
    want = (1.0 / Nb + mu) / (1 + mu)                         # mass of column j + 1 at row 1
    nxt = doc[S + j + 1]
    assert abs(float(Pc[1, nxt]) - (want + (doc[S:].count(nxt) - 1) / Nb / (1 + mu))) < 1e-5
    assert torch.isfinite(lp).all()
    R.mode = "off"
    R.detach()


def test_gradients_reach_gate_pointer_and_addressing():
    """Step 0 (a = 0): a gets a non-zero gradient. Once a > 0: w, b, c, head / layer / κ / β / μ and the read's A
    (through π) get gradients. Plain and --mem_safe checkpointed reads give the same gradients."""
    r, e, B, base, itf = _setup("1,5", "0,1", 1)
    doc = _doc(26, 0)
    rows = _ids(4, 1) + doc[8:12]
    op = _operator(e, doc)
    R = PtrReader(r, itf)
    params = itf.trainable(train_B=True, train_g=True)
    pp = itf.ptr_params()
    for p in pp["w"] + pp["scal"]:
        p.requires_grad_(True)
    tgt = torch.tensor(rows[1:])

    def run(ck):
        for p in list(itf.parameters()):
            p.grad = None
        R.load(op, len(doc))
        R.ckpt = ck
        R.mode, R.need_logits, R.rec = "e2e", True, {}
        with torch.enable_grad():
            lp = r(input_ids=torch.tensor([rows])).logits[0]
            loss = torch.nn.functional.cross_entropy(lp[:-1], tgt)
            loss.backward()
        R.mode = "off"
        return float(loss), {n: (p.grad.clone() if p.grad is not None else None) for n, p in itf.named_parameters()}

    l0, g0 = run(False)
    assert g0["PTR.a"] is not None and abs(float(g0["PTR.a"])) > 0
    assert all(v is None or bool(torch.isfinite(v).all()) for v in g0.values())       # no NaN / inf at a = 0
    _randomize(itf, a=0.5)
    l1, g1 = run(False)
    for k in ("PTR.a", "PTR.w", "PTR.b", "PTR.c", "PTR.head", "PTR.layer", "PTR.logk", "PTR.shift", "PTR.rho"):
        assert g1[k] is not None and float(g1[k].abs().sum()) > 0, k
    assert all(v is None or bool(torch.isfinite(v).all()) for v in g1.values())
    # the pointer path alone (frozen read write-back: gates 0) still sends gradient into the addressing A
    with torch.no_grad():
        itf.g.zero_()
    l2, g2 = run(False)
    assert g2["A.5.weight"] is not None and float(g2["A.5.weight"].abs().sum()) > 0
    assert g2["A.1.weight"] is not None and float(g2["A.1.weight"].abs().sum()) > 0
    with torch.no_grad():
        itf.g.fill_(0.9)
    l3, g3 = run(False)
    l4, g4 = run(True)
    assert abs(l3 - l4) < 1e-5
    for k in g3:
        assert (g3[k] is None) == (g4[k] is None), k
        if g3[k] is not None:
            assert torch.allclose(g3[k], g4[k], atol=1e-5, rtol=1e-4), (k, float((g3[k] - g4[k]).abs().max()))
    del params
    R.detach()


def test_save_load_roundtrip_and_copy_nll():
    r, e, B, base, itf = _setup("1,5", "0,1", 1)
    _randomize(itf)
    d = tempfile.mkdtemp()
    save_d13(itf, os.path.join(d, "p.pt"), {"k": 1})
    itf2, extra = load_d13(os.path.join(d, "p.pt"), e, B)
    assert type(itf2).__name__ == "PtrInterface" and extra == {"k": 1}
    assert itf2.config()["ptr"] == itf.config()["ptr"]
    for k, v in itf.PTR.items():
        assert torch.equal(v, itf2.PTR[k]), k
    doc, rows = _doc(), _ids(6, 1)
    op = _operator(e, doc)
    a = _logits(r, itf, op, len(doc), PtrReader, rows)
    b = _logits(r, itf2, op, len(doc), PtrReader, rows)
    assert torch.equal(a, b)
    Pc = torch.softmax(torch.randn(3, 10), -1)
    gold = torch.tensor([1, 2, 3])
    v = copy_nll(Pc, gold, torch.tensor([True, False, True]))
    assert abs(float(v) - float(-(Pc[0, 1].log() + Pc[2, 3].log()) / 2)) < 1e-6
    # a plain base checkpoint stays a plain D13Interface
    save_d13(base, os.path.join(d, "b.pt"))
    itf3, _ = load_d13(os.path.join(d, "b.pt"), e, B)
    assert type(itf3) is D13Interface


def _setup_k(layers="5", shifts="0", chain=0, topk=0, nbr=0, kappa0=1.0):
    from test_state_interface import _tiny_encoder
    r, e = tiny_qwen35(), _tiny_encoder()
    B = {li: torch.randn(64, 48) * 0.1 for li in LMAP}
    base = D13Interface(e, LMAP, 64, 48, B=B, mlp_hidden=8)
    itf = upgrade_ptr(base, e, parse_ptr(layers, shifts, chain, PREFIX, topk, nbr, kappa0))
    return r, e, B, base, itf


def test_topk_pointer_and_kappa0():
    """--ptr_topk k (+- nbr): P_copy has mass only on tokens of the kept columns; k >= N_body == dense; incremental ==
    full with top-k + chain; --ptr_kappa0 sets the initial sharpness and a = 0 still equals the base."""
    doc, rows = _doc(24, 0), _ids(3, 1)
    rows = rows + doc[10:14]
    S, N = len(PREFIX), len(doc)
    r, e, B, base, itf = _setup_k(topk=2, nbr=1, kappa0=1.25)
    assert abs(float(itf.PTR["logk"].exp()[0]) - 1.25) < 1e-6
    op = _operator(e, doc)
    a0 = _logits(r, base, op, N, D13Reader, rows)
    b0 = _logits(r, itf, op, N, PtrReader, rows)
    assert torch.allclose(torch.log_softmax(a0, -1), b0, atol=1e-5)
    _randomize(itf)
    R = PtrReader(r, itf)
    R.load(op, N)
    R.mode = "student"
    keep = {}
    orig = itf.ptr_mix

    def spy(logits, h, R_):
        keep["pi"] = R_.ptr_pi[5][0].clone()
        return orig(logits, h, R_)
    itf.ptr_mix = spy
    with torch.no_grad():
        full = r(input_ids=torch.tensor([rows])).logits[0]
    del itf.ptr_mix
    Pc = R.ptr_out["Pc"]
    for t in range(len(rows)):
        top = keep["pi"][t].topk(2).indices
        cols = {int(j) + d for j in top for d in (-1, 0, 1) if 0 <= int(j) + d < N - S}
        toks = {doc[S + j] for j in cols}
        nz = set(torch.nonzero(Pc[t] > 0).flatten().tolist())
        assert nz <= toks and nz, (t, nz, toks)
    R.load(op, N)
    with torch.no_grad():
        o = r(input_ids=torch.tensor([rows[:3]]), use_cache=True)
        steps = [o.logits[0]]
        for tk in rows[3:]:
            o = r(input_ids=torch.tensor([[tk]]), past_key_values=o.past_key_values, use_cache=True)
            steps.append(o.logits[0])
    assert torch.allclose(torch.cat(steps, 0), full, atol=1e-4)
    R.mode = "off"
    R.detach()
    # k >= N_body == dense
    r2, e2, B2, base2, d_itf = _setup_k(chain=1)
    _, _, _, _, k_itf = _setup_k(chain=1, topk=1000)
    k_itf.load_state_dict(d_itf.state_dict())
    _randomize(d_itf)
    k_itf.load_state_dict(d_itf.state_dict())
    op2 = _operator(e2, doc)
    x = _logits(r2, d_itf, op2, N, PtrReader, rows)
    y = _logits(r2, k_itf, op2, N, PtrReader, rows)
    assert torch.allclose(x, y, atol=1e-6)
    d = tempfile.mkdtemp()
    save_d13(itf, os.path.join(d, "k.pt"))
    itf3, _ = load_d13(os.path.join(d, "k.pt"), e, B)
    assert itf3.ptr_topk == 2 and itf3.ptr_nbr == 1 and torch.equal(itf3.PTR["logk"], itf.PTR["logk"])


def test_space_variant_first_token():
    """--ptr_nosp: on rows fed '>' a column emits nosp(id), on rows fed 'answer' gtm(id); other rows unchanged; the
    chain matches a fed variant; incremental == full; vocab_maps on a fake BPE vocab."""
    from xenc.p4_ptr import vocab_maps

    class FakeTok:
        def get_vocab(self):
            return {">": 0, "answer": 1, "\u0120Bob": 2, "Bob": 3, ">Bob": 4, "\u0120Ann": 5, "Ann": 6, "x": 7}
    ns, gm, gt, ans = vocab_maps(FakeTok(), 9)
    assert (gt, ans) == (0, 1) and ns[2] == 3 and ns[5] == 6 and ns[7] == 7 and ns[8] == 8
    assert gm[2] == 4 and gm[3] == 4 and gm[5] == 0 and gm[7] == 0
    GT, ANS = 60, 61
    doc = PREFIX + _ids(21, 0)
    doc = [t if t not in (GT, ANS) else 50 for t in doc]
    S, N = len(PREFIX), len(doc)
    nosp = torch.arange(100)
    gtm = torch.full((100,), GT)
    for v in set(doc[S:]):
        nosp[v] = 100 - 1 - v if (100 - 1 - v) not in doc else v
        gtm[v] = (v + 37) % 100
    rows = _ids(3, 1) + [ANS, GT, int(nosp[doc[S + 5]]), doc[S + 6]]
    rows = [t if t not in (GT, ANS) or i >= 3 else 50 for i, t in enumerate(rows)]
    for chain in (0, 1):
        r, e, B, base, itf = _setup_k(chain=chain, shifts="0")
        itf.ptr_nosp = 1
        itf.set_vocab_maps(nosp, gtm, GT, ANS)
        _randomize(itf)
        op = _operator(e, doc)
        R = PtrReader(r, itf)
        R.load(op, N)
        R.mode = "student"
        with torch.no_grad():
            full = r(input_ids=torch.tensor([rows])).logits[0]
        Pc = R.ptr_out["Pc"]
        body = doc[S:]
        assert set(torch.nonzero(Pc[4] > 0).flatten().tolist()) <= {int(nosp[v]) for v in body}        # fed '>'
        assert set(torch.nonzero(Pc[3] > 0).flatten().tolist()) <= {int(gtm[v]) for v in body}         # fed 'answer'
        assert set(torch.nonzero(Pc[2] > 0).flatten().tolist()) <= set(body)                           # other rows
        if chain:      # row 5 is fed nosp(doc[S+5]): its mass must flow to column S+6 (doc[S+6] gains vs chain off)
            itf.ptr_chain = 0
            R.load(op, N)
            with torch.no_grad():
                r(input_ids=torch.tensor([rows]))
            Pc0 = R.ptr_out["Pc"]
            itf.ptr_chain = 1
            assert float(Pc[5, doc[S + 6]]) > float(Pc0[5, doc[S + 6]]) + 1e-4
        R.load(op, N)
        with torch.no_grad():
            o = r(input_ids=torch.tensor([rows[:3]]), use_cache=True)
            st = [o.logits[0]]
            for tk in rows[3:]:
                o = r(input_ids=torch.tensor([[tk]]), past_key_values=o.past_key_values, use_cache=True)
                st.append(o.logits[0])
        assert torch.allclose(torch.cat(st, 0), full, atol=1e-4)
        R.mode = "off"
        R.detach()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
