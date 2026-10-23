"""P3 tests (xenc/p3_tok.py): the token-identity channel of the D13 read.

CPU only, tiny random Qwen3_5ForCausalLM models (tests/_tiny.tiny_qwen35, test_state_interface._tiny_encoder)."""

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
from xenc.p3_tok import TokInterface, TokReader, parse_tok, shift_table, upgrade_tok  # noqa: E402

torch.set_num_threads(2)
LMAP = {1: 2, 5: 5}


def _ids(n, seed):
    return torch.randint(3, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _setup(tok_layers="5", rank=8, heads="kv", shifts="-1,0,1", topk=0, topk_norm=0, temp=1.0):
    from test_state_interface import _tiny_encoder
    r, e = tiny_qwen35(), _tiny_encoder()
    B = {li: torch.randn(64, 48) * 0.1 for li in LMAP}
    base = D13Interface(e, LMAP, 64, 48, B=B, mlp_hidden=8)
    with torch.no_grad():
        base.A["5"].weight.add_(0.05 * torch.randn_like(base.A["5"].weight))
        base.g.fill_(0.9)
    itf = upgrade_tok(base, e, parse_tok(tok_layers, rank, heads, shifts, topk, topk_norm, temp), r.get_input_embeddings().weight)
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


def _logits(r, itf, op, n, R_cls, rows=None):
    rows = rows if rows is not None else _ids(6, 1)
    R = R_cls(r, itf)
    R.load(op, n)
    R.mode = "student"
    with torch.no_grad():
        y = r(input_ids=torch.tensor([rows])).logits
    R.mode = "off"
    R.detach()
    return y


def _raises(fn, msg):
    try:
        fn()
    except AssertionError as ex:
        assert msg in str(ex), str(ex)
        return
    raise RuntimeError(f"expected an AssertionError containing {msg!r}")


def _randomize_U(itf, s=0.3):
    with torch.no_grad():
        for p in itf.TOK["U"].values():
            p.normal_(0, s)


def test_zero_init_equals_base_exactly():
    for tl, topk in (("5", 0), ("1,5", 0), ("5", 4)):
        r, e, B, base, itf = _setup(tl, topk=topk)
        doc, rows = _ids(25, 0), _ids(6, 1)
        op = _operator(e, doc)
        a = _logits(r, base, op, len(doc), D13Reader)
        b = _logits(r, itf, op, len(doc), TokReader)
        assert torch.equal(a, b), float((a - b).abs().max())
        _randomize_U(itf)
        c = _logits(r, itf, op, len(doc), TokReader)
        assert float((c - a).abs().max()) > 1e-4                    # the channel is live once U != 0
        itf.tok_on = False                                          # --tok_off: back to the base exactly
        d = _logits(r, itf, op, len(doc), TokReader)
        assert torch.equal(a, d)


def test_gradients_reach_U_and_addressing():
    r, e, B, base, itf = _setup("1,5")
    doc, rows = _ids(25, 0), _ids(6, 1)
    op = _operator(e, doc)
    R = TokReader(r, itf)
    R.load(op, len(doc))
    for ck in (False, True):                                       # plain and --mem_safe checkpointed read
        R.ckpt = ck
        for p in itf.parameters():
            p.grad = None
        itf.trainable(train_B=True, train_g=True)
        for p in list(itf.TOK["U"].parameters()) + list(itf.TOK["gt"].parameters()):
            p.requires_grad_(True)
        R.mode, R.need_logits, R.rec = "e2e", True, {}
        with torch.enable_grad():
            out = r(input_ids=torch.tensor([rows]))
            loss = out.logits.float().logsumexp(-1).mean()
            loss.backward()
        for li in ("1", "5"):
            g = itf.TOK["U"][li].grad
            assert g is not None and float(g.abs().sum()) > 0, (ck, li)
        assert torch.equal(R.rec[5]["y"], R.rec[5]["y_mix"])       # U = 0: mixed == base read-out
        R.mode = "off"
    # after U moves, the token read sends gradient through p into the addressing (A)
    _randomize_U(itf)
    itf.trainable(train_B=False, train_g=False)
    for p in itf.parameters():
        p.grad = None
    R.ckpt = False
    R.mode, R.need_logits, R.rec = "e2e", False, {}
    with torch.enable_grad():
        y_tok_only = r(input_ids=torch.tensor([rows])).logits.float().logsumexp(-1).mean()
        y_tok_only.backward()
    assert float(itf.A["5"].weight.grad.abs().sum()) > 0
    assert not torch.equal(R.rec[5]["y"], R.rec[5]["y_mix"])
    assert itf.TOK["proj"]["P"].grad is None and not itf.TOK["proj"]["P"].requires_grad
    R.mode = "off"
    R.detach()


def test_incremental_decode_equals_full_pass():
    for heads, topk in (("kv", 0), ("avg", 0), ("kv", 5), ("avg", 0)):
        r, e, B, base, itf = _setup("1,5", heads=heads, topk=topk)
        _randomize_U(itf)
        doc, rows = _ids(20, 0), _ids(7, 1)
        op = _operator(e, doc)
        R = TokReader(r, itf)
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
        assert torch.allclose(inc, full, atol=1e-4), (heads, topk, float((inc - full).abs().max()))
        R.mode = "off"
        R.detach()


def test_ids_align_with_columns_and_table():
    """D13Rt.operator with a token-channel interface stores ids == the exact prefill ids (column c = ids[c]); the
    per-document table row j of shift block s is the projected embedding of ids[j + s]."""
    from fit_d13 import D13Rt
    r, e, B, base, itf = _setup("5", rank=8)
    doc = _ids(30, 3)

    class Rt(D13Rt):
        def pre(self, did):
            return doc
    rt = Rt(r, e, None, "x", itf, tail=4, max_tokens=4000)
    op, n, _ = rt.operator("d")
    assert n == 26 and op["ids"].tolist() == doc[:-4]
    assert all(op[e_][0].shape[1] == n for e_ in set(LMAP.values()))
    E = r.get_input_embeddings().weight
    tb = itf.doc_table(op["ids"], E)
    F, rk = tb["F"], 8
    pr = itf.TOK["proj"]
    for s_i, s in enumerate(itf.tok_shifts):
        for j in (0, 5, n - 1):
            blk = F[j, s_i * rk:(s_i + 1) * rk]
            if 0 <= j + s < n:
                want = (E[doc[j + s]].float() - pr["mu"]) @ pr["P"] * pr["scale"]
                assert torch.allclose(blk, want, atol=1e-5), (s, j)
            else:
                assert float(blk.abs().sum()) == 0.0
    # a plain D13 interface stores no ids (default behaviour unchanged)
    rt2 = Rt(r, e, None, "x", base, tail=4, max_tokens=4000)
    op2, _, _ = rt2.operator("d")
    assert "ids" not in op2
    rt.R.detach()
    rt2.R.detach()
    rt.Tc.detach()
    rt2.Tc.detach()


def test_misaligned_ids_are_rejected():
    r, e, B, base, itf = _setup("5")
    doc = _ids(20, 0)
    op = _operator(e, doc)
    R = TokReader(r, itf)
    op["ids"] = op["ids"][:-1]
    _raises(lambda: R.load(op, len(doc)), "do not align")
    del op["ids"]
    _raises(lambda: R.load(op, len(doc)), "carries no column token ids")
    R.detach()


def test_topk_full_k_equals_dense_and_pca_projection():
    r, e, B, base, itf_d = _setup("5", rank=8, topk=0)
    _, _, _, _, itf_k = _setup("5", rank=8, topk=10_000)
    itf_k.load_state_dict(itf_d.state_dict())
    _randomize_U(itf_d)
    itf_k.TOK["U"]["5"].data.copy_(itf_d.TOK["U"]["5"].data)
    doc, rows = _ids(20, 0), _ids(5, 1)
    op = _operator(e, doc)
    a = _logits(r, itf_d, op, len(doc), TokReader)
    b = _logits(r, itf_k, dict(op), len(doc), TokReader)
    assert torch.allclose(a, b, atol=1e-5)
    # P is orthonormal, mean feature variance ≈ 1 over the vocabulary
    P = itf_d.TOK["proj"]["P"]
    assert torch.allclose(P.t() @ P, torch.eye(8), atol=1e-4)
    E = r.get_input_embeddings().weight
    f = itf_d.features(torch.arange(E.shape[0]), E)
    assert abs(float(f.var(0, unbiased=False).mean()) - 1.0) < 1e-3
    # full-rank top-k (P = I) runs and starts at the base
    _, _, _, base2, itf_f = _setup("5", rank=0, topk=4)
    base2.load_state_dict({k: v for k, v in itf_f.state_dict().items() if not k.startswith("TOK.")}, strict=False)
    c = _logits(r, itf_f, dict(op), len(doc), TokReader)
    d = _logits(r, base2, dict(op), len(doc), D13Reader)
    assert torch.equal(c, d)


def test_save_load_roundtrip_and_shift_table():
    r, e, B, base, itf = _setup("1,5", heads="avg", shifts="0,1,2")
    _randomize_U(itf)
    with torch.no_grad():
        itf.TOK["gt"]["5"].fill_(0.7)
    d = tempfile.mkdtemp()
    save_d13(itf, os.path.join(d, "t.pt"), {"k": 2})
    itf2, extra = load_d13(os.path.join(d, "t.pt"), e, B)
    assert isinstance(itf2, TokInterface) and extra == {"k": 2}
    assert itf2.config()["tok"] == itf.config()["tok"]
    for k, v in itf.state_dict().items():
        if k.startswith(("TOK.", "A.", "g")):
            assert torch.equal(itf2.state_dict()[k], v), k
    doc, rows = _ids(20, 0), _ids(5, 1)
    op = _operator(e, doc)
    assert torch.equal(_logits(r, itf, op, len(doc), TokReader), _logits(r, itf2, dict(op), len(doc), TokReader))
    F0 = torch.arange(12.0).view(6, 2)
    T = shift_table(F0, [-1, 0, 2])
    assert torch.equal(T[0, :2], torch.zeros(2)) and torch.equal(T[1, :2], F0[0])
    assert torch.equal(T[:, 2:4], F0) and torch.equal(T[3, 4:], F0[5]) and torch.equal(T[4, 4:], torch.zeros(2))


def test_d20_transform_op_passes_extras_and_judge_entry_point():
    """register_d20's DocAdapter (kv part) keeps the ids / token table; TokInterface works under add_delta (dA)."""
    from xenc.d20_reg import DocAdapter
    r, e, B, base, itf = _setup("5")
    doc, rows = _ids(20, 0), _ids(5, 1)
    op = _operator(e, doc)
    ad = DocAdapter(itf, ["dA", "gate", "kv"], rank=4, kv_rank=2)
    op2 = ad.transform_op(op)
    assert op2["ids"] is op["ids"] and set(op2) == set(op)
    a = _logits(r, itf, op2, len(doc), TokReader)
    b = _logits(r, itf, op, len(doc), TokReader)
    assert torch.allclose(a, b, atol=1e-6)                         # zero-init per-doc parts == base


def test_cached_runtime_ids_recompute():
    """fit_d12p.CachedRt: an operator cache file without ids gets them re-derived from the doc text (length-checked)."""
    import fit_d12p
    r, e, B, base, itf = _setup("5")
    doc = _ids(30, 3)
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "op"))
    op = _operator(e, doc[:-4])
    torch.save({"kv": {k: v for k, v in op.items() if isinstance(k, int)}, "n": 26}, fit_d12p.op_path(d, "doc1"))
    rt = fit_d12p.CachedRt(r, None, "x", itf, d, tail=4)
    rt.pre = types.MethodType(lambda self, did: doc, rt)
    o, n, _ = rt.operator("doc1")
    assert n == 26 and o["ids"].tolist() == doc[:-4]
    rt.pre = types.MethodType(lambda self, did: doc + [5], rt)   # a different tokenization must not pass
    rt._ops.clear()
    _raises(lambda: rt.operator("doc1"), "does not match the cached operator")
    rt.R.detach()


def test_full_rank_topk_read_matches_manual():
    """rank 0 (P = I) top-k: r = Σ_{j in top-k(p̄)} p̄_j · scale · E[id_{j+s}] (zero outside the operator)."""
    r, e, B, base, itf = _setup("5", rank=0, heads="avg", shifts="-1,0,2", topk=3)
    E = r.get_input_embeddings().weight
    N, T, C = 9, 2, 11
    ids = torch.tensor(_ids(N, 4))
    p = torch.softmax(torch.randn(2, 2, T, C), -1)
    tb = itf.doc_table(ids, E)
    got = itf.tok_read(p, tb)
    pb = p[..., :N].mean((0, 1))                                    # [T,N]
    sc = float(itf.TOK["proj"]["scale"])
    want = []
    for t in range(T):
        top = pb[t].topk(3).indices.tolist()
        row = []
        for s in itf.tok_shifts:
            acc = torch.zeros(E.shape[1])
            for j in top:
                if 0 <= j + s < N:
                    acc += pb[t, j] * E[ids[j + s]] * sc
            row.append(acc)
        want.append(torch.cat(row))
    assert torch.allclose(got, torch.stack(want), atol=1e-5), float((got - torch.stack(want)).abs().max())


def test_val_attn_diag_items_default_unchanged_and_cap_deterministic():
    """register_d13.attn_diag_items: default == the original stride sample; --val_attn_max_op draws the same stride
    sample from items under the cap only (deterministic: identical in every arm)."""
    from register_d13 import attn_diag_items
    val = [{"key": f"k{i}", "n": (i * 7919) % 100000} for i in range(337)]
    for n in (30, 2, 1, 500):
        assert attn_diag_items(val, n) == val[:: max(1, len(val) // max(n, 1))][: n]
    cap = attn_diag_items(val, 30, lambda it: it["n"], 32000)
    assert len(cap) == 30 and all(it["n"] <= 32000 for it in cap)
    assert cap == attn_diag_items(list(val), 30, lambda it: it["n"], 32000)


def test_hard_topk_kvavg_and_temperature():
    """Hard top-k (renormalized), per-kv-head selection averaged (kvavg) and the channel-only temperature: zero-init ==
    base exactly, incremental == full, and the hard top-k read matches a manual computation."""
    for kw in (dict(rank=0, heads="kvavg", topk=2, topk_norm=1), dict(rank=8, heads="kv", topk=1, topk_norm=1),
               dict(rank=8, heads="kv", temp=1.25), dict(rank=0, heads="avg", topk=3, topk_norm=1, temp=1.25)):
        r, e, B, base, itf = _setup("1,5", **kw)
        doc, rows = _ids(20, 0), _ids(7, 1)
        op = _operator(e, doc)
        assert torch.equal(_logits(r, base, op, len(doc), D13Reader, rows), _logits(r, itf, op, len(doc), TokReader, rows))
        _randomize_U(itf)
        R = TokReader(r, itf)
        R.load(op, len(doc))
        R.mode = "student"
        with torch.no_grad():
            full = r(input_ids=torch.tensor([rows])).logits[0]
        R.load(op, len(doc))
        with torch.no_grad():
            o = r(input_ids=torch.tensor([rows[:3]]), use_cache=True)
            st = [o.logits[0]]
            for t in rows[3:]:
                o = r(input_ids=torch.tensor([[t]]), past_key_values=o.past_key_values, use_cache=True)
                st.append(o.logits[0])
        assert torch.allclose(torch.cat(st, 0), full, atol=1e-4), kw
        R.mode = "off"
        R.detach()
        assert itf.config()["tok"]["topk_norm"] == kw.get("topk_norm", 0) and itf.config()["tok"]["temp"] == kw.get("temp", 1.0)
    # manual hard top-2, kvavg, full rank
    r, e, B, base, itf = _setup("5", rank=0, heads="kvavg", shifts="0,1", topk=2, topk_norm=1)
    E = r.get_input_embeddings().weight
    N, T, C = 8, 2, 10
    ids = torch.tensor(_ids(N, 5))
    p = torch.softmax(torch.randn(2, 2, T, C), -1)
    got = itf.tok_read(p, itf.doc_table(ids, E))
    sc = float(itf.TOK["proj"]["scale"])
    pk = p[..., :N].mean(1)                                          # [Hkv,T,N]
    want = torch.zeros(T, 2 * E.shape[1])
    for t in range(T):
        for h in range(2):
            v, ix = pk[h, t].topk(2)
            v = v / v.sum()
            for b_, s_ in enumerate((0, 1)):
                for w_, j in zip(v.tolist(), ix.tolist()):
                    if j + s_ < N:
                        want[t, b_ * E.shape[1]:(b_ + 1) * E.shape[1]] += 0.5 * w_ * E[ids[j + s_]] * sc
    assert torch.allclose(got, want, atol=1e-5), float((got - want).abs().max())


def test_eval_only_p_override_hook_and_diag_helpers():
    """p3_diag: the eval-only p override (identity override == no override; a one-hot override changes the output) and
    the pure helpers (oracle columns, space variants, paragraph anatomy, gold occurrences)."""
    r, e, B, base, itf = _setup("5", rank=8)
    _randomize_U(itf)
    doc, rows = _ids(20, 0), _ids(6, 1)
    op = _operator(e, doc)
    a = _logits(r, itf, op, len(doc), TokReader, rows)
    itf.tok_p_override = lambda li, p: p
    b = _logits(r, itf, op, len(doc), TokReader, rows)
    assert torch.equal(a, b)

    def onehot(li, p):
        q = torch.zeros_like(p)
        q[..., 3] = 1.0
        return q
    itf.tok_p_override = onehot
    c = _logits(r, itf, op, len(doc), TokReader, rows)
    assert float((c - a).abs().max()) > 1e-5
    itf.tok_p_override = None
    E = r.get_input_embeddings().weight
    tb = itf.doc_table(op["ids"], E)
    p = torch.zeros(2, 2, 1, len(doc) + 1)
    p[..., 3] = 1.0
    assert torch.allclose(itf.tok_read(p, tb)[0], tb["F"][3].repeat(2), atol=1e-6)    # per-kv-head blocks
    from p3_diag import oracle_columns, occurrences, same_paragraph, space_variants
    # tokens: "The| Bob| Smith| went|." with end offsets
    ends = [3, 7, 13, 18, 19]
    assert oracle_columns(ends, [(4, 13)], 0, 5) == [1] and oracle_columns(ends, [(4, 13)], 3, 5) == [2]
    assert oracle_columns(ends, [(4, 13)], 9, 5) == [] and oracle_columns(ends, [(4, 13)], 0, 1) == []
    vocab = {"Bob": 1, "\u0120Bob": 2, ">Bob": 3, "x": 4}
    assert space_variants("Bob", vocab) == {1, 2} and space_variants(">Bob", vocab) == {1, 2, 3}
    assert space_variants("\u0120Bob", vocab) == {1, 2}
    d = "Alpha beta.\nGamma Bob Smith delta.\nOther Carl here."
    gs = [(d.find("Bob Smith"), d.find("Bob Smith") + 9)]
    assert same_paragraph(d, "delta", gs)[:2] == (True, True)
    assert same_paragraph(d, "Carl", gs)[:2] == (True, False)
    assert same_paragraph(d, "zeta", gs)[:2] == (False, False)
    pt = "SYS " + d + " TAIL"
    ans, sp = occurrences(pt, d, ["nope", "bob smith"], len(pt))
    assert ans == "bob smith" and sp == [(pt.find("Bob Smith"), pt.find("Bob Smith") + 9)]
    assert occurrences(pt, d, ["bob smith"], pt.find("Bob Smith") + 3) == (None, [])
    assert occurrences("x Bayes and monolingual", "Bayes and monolingual", ["es", "yes", "no"], 99) == (None, [])
    from p3_diag import tok_match
    assert tok_match("\u0120labeled", "l", True) and not tok_match("\u0120labeled", "l", False)
    assert tok_match("\u0120Bob", ">Bob", False) and not tok_match("\u0120Bobby", "Bob", False)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f()
            print("ok", n)
    print("ALL P3 TESTS PASSED")
