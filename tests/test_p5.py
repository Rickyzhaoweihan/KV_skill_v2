"""P5 tests: the dedicated copy head (xenc/p5_copy.py) on the D13 read.

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
from xenc.p5_copy import (P5Reader, best_span, occurrences, parse_p5, upgrade_p5)  # noqa: E402

torch.set_num_threads(2)
LMAP = {1: 2, 5: 5}
PREFIX = [7, 8, 9]
GT = 11                                   # fake '>' id
ANS = 10                                  # fake 'answer' id
V = 100


def _ids(n, seed, lo=12, hi=100):
    return torch.randint(lo, hi, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _nosp():
    t = torch.arange(V)
    t[50:60] = torch.arange(60, 70)       # "space" variants 50..59 -> 60..69
    return t


def _gtm():
    t = torch.full((V,), GT)
    t[50:60] = torch.arange(70, 80)       # merged '>'-variants of 50..59 (">Bob"); others: the bare '>'
    return t


def _setup(fuse="mix", chain=1):
    from test_state_interface import _tiny_encoder
    r, e = tiny_qwen35(), _tiny_encoder()
    B = {li: torch.randn(64, 48) * 0.1 for li in LMAP}
    base = D13Interface(e, LMAP, 64, 48, B=B, mlp_hidden=8)
    with torch.no_grad():
        base.A["5"].weight.add_(0.05 * torch.randn_like(base.A["5"].weight))
        base.g.fill_(0.9)
    itf = upgrade_p5(base, e, parse_p5("2,4", heads=2, dim=8, fuse=fuse, chain=chain, prefix_ids=PREFIX, gt_id=GT,
                                       span_max=6, nbkt=6, ans_id=ANS, merged=1))
    itf.set_nosp(_nosp(), _gtm())
    return r, e, base, itf


def _operator(e, itf, doc):
    Tc = EncTeacher(e, sorted(set(LMAP.values())))
    Tc.rec, Tc.kv_only = {}, True
    hk = itf.harvest_hooks(e)
    with torch.no_grad():
        e(input_ids=torch.tensor([doc]), use_cache=False)
    op = {k: (v["k"], v["v"]) for k, v in Tc.rec.items()}
    Tc.detach()
    op.update(hk.finish(torch.device("cpu")))
    op["ids"] = torch.tensor(doc, dtype=torch.long)
    return op


def _doc(n=30, seed=0):
    return PREFIX + _ids(n - len(PREFIX), seed)


def _randomize(itf, a=0.6, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        P = itf.P5
        P["a"].fill_(a)
        P["gb"].fill_(0.3)
        P["gw"].copy_(torch.randn(P["gw"].shape, generator=g) * 0.1)
        P["gf"].copy_(torch.randn(P["gf"].shape, generator=g) * 0.3)
        for k in ("hw", "pos", "cr"):
            P[k].copy_(torch.randn(P[k].shape, generator=g) * 0.5)
        P["logt"].fill_(0.5)
        P["rho"].fill_(0.7)
    itf._p5_ver += 1


def _student_logits(r, itf, op, n, R_cls, rows, keep=0):
    R = R_cls(r, itf) if R_cls is D13Reader else itf.make_reader(r)
    R.load(op, n)
    R.mode, R.need_logits, R.rec = "student", False, {}
    with torch.no_grad():
        y = r(input_ids=torch.tensor([rows]), logits_to_keep=keep).logits[0].float()
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


def test_zero_init_equals_base_exactly():
    r, e, base, itf = _setup()
    doc, rows = _doc(), _ids(6, 1)
    op = _operator(e, itf, doc)
    a = _student_logits(r, base, op, len(doc), D13Reader, rows)
    b = _student_logits(r, itf, op, len(doc), P5Reader, rows)
    assert torch.equal(a, b)
    _randomize(itf)
    c = _student_logits(r, itf, op, len(doc), P5Reader, rows, keep=3)
    assert float((torch.log_softmax(a[-3:], -1) - c).abs().max()) > 1e-3          # live once a > 0
    itf.p5_on = False
    d = _student_logits(r, itf, op, len(doc), P5Reader, rows)
    assert torch.equal(a, d)
    R = itf.make_reader(r)
    R.detach()
    assert "forward" not in r.lm_head.__dict__


def test_harvest_ids_align_with_columns():
    r, e, base, itf = _setup()
    doc = _doc(25, 3)
    op = _operator(e, itf, doc)
    with torch.no_grad():
        hs = e(input_ids=torch.tensor([doc]), output_hidden_states=True).hidden_states
    from deltaskill.backbone_utils import decoder_layers
    L = decoder_layers(e)
    want = torch.cat([L[l + 1].input_layernorm(hs[l + 1][0]) for l in (2, 4)], -1)
    assert op["p5h"].shape == (len(doc), 2 * 48) and op["p5h"].dtype == torch.bfloat16
    assert torch.allclose(op["p5h"].float(), want.to(torch.bfloat16).float(), atol=1e-2)
    tab = itf.p5_tables(op, len(doc))
    assert tab["S"] == 3 and tab["ids"].tolist() == doc[3:]
    assert torch.equal(tab["nosp"], _nosp()[torch.tensor(doc[3:])])
    bad = dict(op, ids=op["ids"][:-1])
    _raises(lambda: itf.p5_tables(bad, len(doc)), "do not align")
    trunc = dict(op, ids=torch.tensor([50] + doc[1:]))
    assert itf.p5_tables(trunc, len(doc))["S"] == 0                     # left-truncated doc: no template prefix


def test_incremental_decode_equals_full_pass():
    for fuse in ("mix", "poe"):
        r, e, base, itf = _setup(fuse)
        _randomize(itf)
        doc = _doc(30, 4)
        op = _operator(e, itf, doc)
        prompt, cont = _ids(6, 5), [GT] + doc[10:14]                       # '>' then a copied span
        full = _student_logits(r, itf, op, len(doc), P5Reader, prompt + cont, keep=len(cont) + 1)
        R = itf.make_reader(r)
        R.load(op, len(doc))
        R.mode = "student"
        outs = []
        with torch.no_grad():
            o = r(input_ids=torch.tensor([prompt]), use_cache=True, logits_to_keep=1)
            outs.append(o.logits[0, -1].float())
            for t in cont:
                o = r(input_ids=torch.tensor([[t]]), past_key_values=o.past_key_values, use_cache=True, logits_to_keep=1)
                outs.append(o.logits[0, -1].float())
        R.mode = "off"
        R.detach()
        inc = torch.stack(outs)
        assert torch.allclose(full, inc, atol=2e-4), (fuse, float((full - inc).abs().max()))


def test_copy_distribution_chain_and_first_row_mapping():
    r, e, base, itf = _setup()
    _randomize(itf)
    doc = _doc(30, 6)
    doc[12] = 55                                                         # a "space" token, nosp -> 65
    op = _operator(e, itf, doc)
    itf.make_reader(r).detach()
    tab = itf.p5_tables(op, len(doc))
    keys = itf.p5_keys(tab)
    K = 3
    h = torch.randn(K, 64)
    logits = torch.randn(K, V)
    x = torch.tensor([GT, 65, doc[13]])                                  # '>', then the no-space token, then the next
    res = itf.p5_forward(logits, h, x, keys, tab)
    assert torch.allclose(res["Pc"].sum(-1), torch.ones(K), atol=1e-5)
    manual = torch.zeros(K, V)
    ti = torch.stack([tab["nosp"], tab["ids"], tab["ids"]])
    manual.scatter_add_(1, ti, res["pi"])
    assert torch.allclose(manual, res["Pc"], atol=1e-6)
    j = 12 - 3                                                           # body column of doc[12]
    # chain: row 1 is fed 65 (= nosp of column j) -> extra mass on column j+1 vs the raw pointer
    assert float(res["pi"][1, j + 1]) > float(res["pi_raw"][1, j + 1])
    # row 2 is fed doc[13] (column j+1) -> mass moves to j+2
    assert float(res["pi"][2, j + 2]) > float(res["pi_raw"][2, j + 2])
    assert torch.allclose(res["pi"].sum(-1), torch.ones(K), atol=1e-5)
    # mixture / poe formulas
    g = res["g"]
    Pm = (1 - g).unsqueeze(-1) * torch.softmax(logits, -1) + g.unsqueeze(-1) * res["Pc"]
    assert torch.allclose(res["logp"].exp(), Pm, atol=1e-5)


def test_merged_tag_variant_rows_and_chain():
    """Rows fed 'answer' emit gtm(id) (">Bob"), rows fed '>' emit nosp(id), other rows ids; the chain follows a fed
    merged token but NOT the bare '>' (which gtm uses as its fallback); incremental == full on the merged route."""
    r, e, base, itf = _setup()
    _randomize(itf)
    doc = _doc(30, 9)
    doc[12], doc[20] = 55, 30                                            # 55 has a merged variant (75); 30 has none
    op = _operator(e, itf, doc)
    itf.make_reader(r).detach()
    tab = itf.p5_tables(op, len(doc))
    keys = itf.p5_keys(tab)
    x = torch.tensor([ANS, 75, doc[13], GT, 40])
    res = itf.p5_forward(torch.randn(5, V), torch.randn(5, 64), x, keys, tab)
    body = torch.tensor(doc[3:])
    assert set(torch.nonzero(res["Pc"][0] > 0).flatten().tolist()) <= set(_gtm()[body].tolist())
    assert set(torch.nonzero(res["Pc"][3] > 0).flatten().tolist()) <= set(_nosp()[body].tolist())
    assert set(torch.nonzero(res["Pc"][4] > 0).flatten().tolist()) <= set(body.tolist())
    assert bool(res["first"][0]) and bool(res["first"][3]) and not bool(res["first"][1])
    j = 12 - 3
    assert float(res["pi"][1, j + 1]) > float(res["pi_raw"][1, j + 1])   # fed 75 = gtm(55) -> mass to j+1
    # row 3 is fed the bare '>' (gtm's fallback for every column WITHOUT a merged variant, e.g. doc[20] = 30): no
    # column matches it, so the chain moves no mass (it would if gtm's fallback were used as a match)
    assert int((_gtm()[body] == GT).sum()) > 0
    assert torch.allclose(res["pi"][3], res["pi_raw"][3], atol=1e-7)
    # incremental == full on the merged route ('<', 'answer', '>Bob'-like, next token)
    prompt, cont = _ids(6, 5), [ANS, 75, doc[13], doc[14]]
    full = _student_logits(r, itf, op, len(doc), P5Reader, prompt + cont, keep=len(cont) + 1)
    R = itf.make_reader(r)
    R.load(op, len(doc))
    R.mode = "student"
    outs = []
    with torch.no_grad():
        o = r(input_ids=torch.tensor([prompt]), use_cache=True, logits_to_keep=1)
        outs.append(o.logits[0, -1].float())
        for t in cont:
            o = r(input_ids=torch.tensor([[t]]), past_key_values=o.past_key_values, use_cache=True, logits_to_keep=1)
            outs.append(o.logits[0, -1].float())
    R.mode = "off"
    R.detach()
    assert torch.allclose(full, torch.stack(outs), atol=2e-4)


def test_gradients_reach_the_new_params():
    r, e, base, itf = _setup()
    for n_, p in itf.named_parameters():
        p.requires_grad_(n_.startswith("P5."))
    doc = _doc(30, 7)
    op = _operator(e, itf, doc)
    R = itf.make_reader(r)
    rows = _ids(6, 8) + [GT] + doc[10:13]
    K = 4
    for a0 in (0.0, 0.5):
        with torch.no_grad():
            itf.P5["a"].fill_(a0)
        itf.zero_grad(set_to_none=True)
        R.load(op, len(doc))
        R.mode, R.p5_capture = "e2e", True
        with torch.no_grad():
            lg = r(input_ids=torch.tensor([rows]), logits_to_keep=K).logits[0].float()
        R.mode, R.p5_capture = "off", False
        assert torch.equal(lg, _student_logits(r, base, op, len(doc), D13Reader, rows, keep=K))   # capture: parent logits
        h = R.p5_h
        tab = R.p5_tab
        keys = itf.p5_keys(tab)
        res = itf.p5_forward(lg, h, torch.tensor(rows[-K:]), keys, tab)
        ce = torch.nn.functional.nll_loss(res["logp"][:-1], torch.tensor(rows[-K + 1:]))
        span = -res["pi"][1:, torch.tensor([7, 8, 9])].diagonal().clamp_min(1e-30).log().mean()
        pe = itf.p5_pi(itf.p5_scores(h[1:2], keys, tab, None, end=True))[0]
        (ce + span - pe[9].log() + torch.nn.functional.binary_cross_entropy(res["sig"], torch.ones(K) * 0.7)).backward()
        P = itf.P5
        for k in ("Wq", "Wqe", "Wkh", "Wkt", "a", "logt", "hw", "pos", "rho", "gw", "gf", "gb"):
            assert P[k].grad is not None and float(P[k].grad.abs().sum()) > 0, (a0, k)
        assert all(p.grad is None for n_, p in itf.named_parameters() if not n_.startswith("P5."))
        assert P["cr"].grad is not None
    R.detach()


def test_save_load_round_trip():
    r, e, base, itf = _setup()
    _randomize(itf)
    doc, rows = _doc(), _ids(6, 1)
    op = _operator(e, itf, doc)
    a = _student_logits(r, itf, op, len(doc), P5Reader, rows, keep=2)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.pt")
        save_d13(itf, p)
        itf2, _ = load_d13(p, e, {li: itf.B[str(li)].detach() for li in LMAP})
    itf2.set_nosp(_nosp(), _gtm())
    assert type(itf2).__name__ == "P5Interface"
    for k in itf.P5:
        assert torch.equal(itf.P5[k], itf2.P5[k]), k
    b = _student_logits(r, itf2, op, len(doc), P5Reader, rows, keep=2)
    assert torch.equal(a, b)


def test_occurrences_and_best_span():
    tab = {"ids": torch.tensor([5, 6, 7, 50, 8, 9, 6, 7, 51, 8]), "nosp": None}
    tab["nosp"] = _nosp()[tab["ids"]]
    g_ = _gtm()[tab["ids"]]
    tab["gtm_m"] = torch.where(g_ == GT, torch.full_like(g_, -1), g_)
    assert occurrences([71, 8], tab).tolist() == [8]                     # merged '>'-variant of column 8 (51 -> 71)
    assert occurrences([GT, 6], tab).tolist() == []                      # the bare '>' never matches a column
    assert occurrences([6, 7], tab).tolist() == [1, 6]
    assert occurrences([60, 8], tab).tolist() == [3]                     # first token up to its leading space
    assert occurrences([60, 9], tab).tolist() == []
    assert occurrences([7, 51, 8, 1], tab).tolist() == []                # runs past the end
    g = torch.Generator().manual_seed(0)
    for _ in range(20):
        ps, pe = torch.rand(15, generator=g), torch.rand(15, generator=g)
        ps, pe = ps / ps.sum(), pe / pe.sum()
        s, e_, c = best_span(ps, pe, 4)
        bf = max(((float(ps[i].log() + pe[j].log()), i, j) for i in range(15) for j in range(i, min(15, i + 4))))
        assert (s, e_) == (bf[1], bf[2]) and abs(c - bf[0]) < 1e-5


def test_content_rows_strict_real_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    except Exception:
        print("  (skipped: tokenizer not cached)")
        return
    from xenc.p5_copy import content_rows_strict, nospace_table, space_maps
    ids, rows = content_rows_strict(tok, "<answer>Bob Smith</answer>")
    assert [tok.decode([ids[i]]) for i in rows] == ["Bob", " Smith"]
    ids, rows = content_rows_strict(tok, "<answer>The test</answer>")
    assert rows == []                                                    # '>The' merged: rejected without allow_merged
    ids, rows = content_rows_strict(tok, "<answer>The test</answer>", allow_merged=True)
    assert [tok.decode([ids[i]]) for i in rows] == [">The", " test"] and tok.decode([ids[rows[0] - 1]]) == "answer"
    ns2, gm, gt, ans = space_maps(tok, len(tok))
    the = tok(" The", add_special_tokens=False)["input_ids"][0]
    assert tok.decode([int(gm[the])]) == ">The" and tok.decode([gt]) == ">" and tok.decode([ans]) == "answer"
    bert = tok(" BERT", add_special_tokens=False)["input_ids"]
    assert len(bert) != 1 or int(gm[bert[0]]) in (gt, tok.convert_tokens_to_ids(">BERT"))
    ids, rows = content_rows_strict(tok, "<answer>the 1990s</answer>")
    assert tok.decode([ids[i] for i in rows]) == "the 1990s"
    ns = nospace_table(tok, len(tok))
    sp = tok(" Bob", add_special_tokens=False)["input_ids"][0]
    assert tok.decode([int(ns[sp])]) == "Bob"


if __name__ == "__main__":
    n = 0
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
            print("ok", k)
    print(f"{n} passed")
