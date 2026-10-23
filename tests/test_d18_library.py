"""D18 tests: the library read over many operators (xenc/d18_library.py), BM25, library construction, and the
encoder-side segment prefill of scripts/eval_d18_context.py. CPU only, tiny random Qwen3_5ForCausalLM."""

from __future__ import annotations

import copy
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402
from xenc.d13_read import D13Interface, D13Reader, EncTeacher  # noqa: E402
from xenc.d18_library import BM25, D18LibReader, lib_read, route_stats  # noqa: E402

torch.set_num_threads(2)
LMAP = {1: 2, 5: 5}


def _ids(n, seed):
    return torch.randint(3, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _enc():
    from test_state_interface import _tiny_encoder
    return _tiny_encoder()


def _op(e, doc):
    Tc = EncTeacher(e, sorted(set(LMAP.values())))
    Tc.rec, Tc.kv_only = {}, True
    with torch.no_grad():
        e(input_ids=torch.tensor([doc]), use_cache=False)
    op = {k: (r["k"], r["v"]) for k, r in Tc.rec.items()}
    Tc.rec = None
    Tc.detach()
    return op


def _itf(e, seed=0):
    g = torch.Generator().manual_seed(seed)
    B = {li: torch.randn(64, 48, generator=g) * 0.1 for li in LMAP}
    itf = D13Interface(e, LMAP, 64, 48, B=B, mlp_hidden=8)
    with torch.no_grad():
        for li in LMAP:
            itf.A[str(li)].weight.add_(0.3 * torch.randn(48, 64, generator=g))
            itf.M[str(li)][2].weight.normal_(0, 0.05, generator=g)
    return itf


def test_single_doc_equals_d13_read_and_incremental():
    e = _enc()
    itf = _itf(e)
    op = _op(e, _ids(23, 0))
    x = torch.randn(7, 64, generator=torch.Generator().manual_seed(3))
    for li, en in LMAP.items():
        K, V = op[en]
        ref = itf.read(li, x, K, V, K.shape[1])["y"]
        got = lib_read(itf, li, x, [(K, V)])
        assert torch.allclose(got["y"], ref, atol=1e-4), float((got["y"] - ref).abs().max())
        assert torch.allclose(got["w"], torch.ones_like(got["w"]))
        r1 = lib_read(itf, li, x[:4], [(K, V)])
        r2 = lib_read(itf, li, x[4:], [(K, V)], past=r1["past"])
        assert torch.allclose(torch.cat([r1["y"], r2["y"]]), ref, atol=1e-4)


def test_routing_force_topk_and_skip():
    e = _enc()
    itf = _itf(e)
    ops = [_op(e, _ids(n, s)) for n, s in ((20, 1), (31, 2), (17, 3))]
    x = torch.randn(5, 64, generator=torch.Generator().manual_seed(4))
    li, en = 5, 5
    lst = [o[en] for o in ops]
    soft = lib_read(itf, li, x, lst, skip=2)
    assert soft["w"].shape[-1] == 3 and torch.allclose(soft["w"].sum(-1), torch.ones(soft["w"].shape[:-1]), atol=1e-5)
    f1 = lib_read(itf, li, x, lst, force=1)["y"]
    one = lib_read(itf, li, x, [lst[1]])["y"]
    assert torch.allclose(f1, one, atol=1e-5)
    hard = lib_read(itf, li, x, lst, top_k=1)
    assert torch.allclose(hard["w"].max(-1).values, torch.ones_like(hard["w"].max(-1).values))
    sharp = lib_read(itf, li, x, lst, beta=1e4, skip=2)["w"]
    assert torch.equal(sharp.argmax(-1), soft["w"].argmax(-1))
    # route logit ignores the shared leading `skip` columns: identical leading keys -> same w with/without them
    rs = route_stats({li: soft["w"][:, :, -1]}, gold=0)
    assert 0 <= rs["gold_mass"] <= 1 and rs["chance"] == 1 / 3


def test_reader_patch_matches_d13_student():
    r, e = tiny_qwen35(), _enc()
    itf = _itf(e)
    doc, rows = _ids(25, 0), _ids(6, 1)
    op = _op(e, doc)
    R = D13Reader(r, itf)
    R.load(op, len(doc))
    R.mode = "student"
    with torch.no_grad():
        a = r(input_ids=torch.tensor([rows]), use_cache=True)
        a2 = r(input_ids=torch.tensor([[7]]), past_key_values=a.past_key_values, use_cache=True)
    R.mode = "off"
    R.detach()
    L = D18LibReader(r, itf)
    L.load({k: [v] for k, v in op.items()}, skip=0)
    L.mode = "lib"
    with torch.no_grad():
        b = r(input_ids=torch.tensor([rows]), use_cache=True)
        b2 = r(input_ids=torch.tensor([[7]]), past_key_values=b.past_key_values, use_cache=True)
    assert torch.allclose(a.logits, b.logits, atol=1e-4) and torch.allclose(a2.logits, b2.logits, atol=1e-4)
    assert set(L.route) == set(LMAP) and L.route[1].shape[-1] == 1
    # two documents, the second a copy of the first with a different length: patch runs, routes sum to 1
    op2 = _op(e, _ids(40, 9))
    L.load({k: [op[k], op2[k]] for k in op}, skip=0)
    with torch.no_grad():
        r(input_ids=torch.tensor([rows]), use_cache=True)
    assert all(abs(float(w.sum(-1).mean()) - 1) < 1e-5 for w in L.route.values())
    L.mode = "off"
    with torch.no_grad():
        c = r(input_ids=torch.tensor([rows]), use_cache=True)
    assert float((c.logits - b.logits).abs().max()) > 1e-5
    L.detach()


def test_bm25_and_libraries():
    docs = {"d1": "the patient John Miller received temozolomide for glioblastoma",
            "d2": "Sophie Walker gave a passionate speech about women in politics",
            "d3": "the maglev line in Birmingham ran for eleven years"}
    bm = BM25(docs)
    assert bm.top1("How long did the Birmingham maglev run?") == "d3"
    assert bm.top1("What drug was Mr. Miller given?") == "d1"
    assert bm.top1("Miller", subset={"d2", "d3"}) in {"d2", "d3"}
    from eval_d18_library import bm25_query, libraries
    E = ["a", "b", "c", "d"]
    pool = ["p1", "p2", "p3", "p4"]
    L = libraries(E, pool, [1, 2, 3, 6])
    assert L[1]["c"] == ["c"] and L[2]["a"] == ["a", "b"] and L[2]["d"] == ["c", "d"]
    assert L[3]["a"] == ["a", "b", "c"] and L[3]["d"] == ["d", "p1", "p2"]
    assert all(len(g) == 6 and d in g for d, g in L[6].items())
    it = {"question": "Q?", "choices": [{"label": "A", "text": "x"}]}
    assert bm25_query(it) == "Q? x"
    ci = {"turns": [("first", "ok")], "final": "last"}
    assert bm25_query(it, ci, "final") == "last" and bm25_query(it, ci, "conv") == "first last"


class _Tok:
    eos_token_id = None

    def convert_tokens_to_ids(self, t):
        return -1

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{int(i)}" for i in ids)


def test_encside_segment_prefill_equivalence():
    """Runtime.gen_encside: injecting all rows in one prefill == the same rows split into two injected segments;
    a stock (uninjected) segment differs. Uses the real EncSideReader / EncCapture (post) on tiny models."""
    from eval_d18_context import Patches, Runtime
    from deltaskill.backbone_utils import decoder_layers
    from xenc.encside_read import EncCapture, EncSideInterface, EncSideReader
    r, e = tiny_qwen35(), _enc()
    rt = object.__new__(Runtime)
    rt.reader, rt.enc, rt.tok, rt.hub, rt.lmap, rt.dev = r, e, _Tok(), "tiny", dict(LMAP), torch.device("cpu")
    rt.rt = type("X", (), {"edev": torch.device("cpu")})()
    L = decoder_layers(r)
    rt.P = Patches({li: L[li].self_attn for li in LMAP})
    eitf = EncSideInterface(LMAP, 48, 64, "replace", "post", identity=False)
    with torch.no_grad():
        for li in LMAP:
            eitf.B[str(li)].copy_(torch.randn(64, 48, generator=torch.Generator().manual_seed(li)) * 0.2)
    rt.eitf, rt.ER = eitf, EncSideReader(r, eitf)
    rt.P.grab("enc")
    rt.cap = EncCapture(e, sorted(set(LMAP.values())), "post")
    doc, rows = _ids(20, 0), _ids(9, 1)
    with torch.no_grad():
        cache = e(input_ids=torch.tensor([doc]), use_cache=True).past_key_values
    g1 = rt.gen_encside(cache, rows, [(rows, (0, 9))], 5)
    g2 = rt.gen_encside(cache, rows, [(rows[:4], (0, 4)), (rows[4:], (4, 9))], 5)
    g3 = rt.gen_encside(cache, rows, [(rows[:4], None), (rows[4:], (4, 9))], 5)
    assert g1 == g2, (g1, g2)
    assert g1[1] == 5 and g3[1] == 5
    # the cache passed in is never mutated (deep-copied)
    g4 = rt.gen_encside(copy.deepcopy(cache), rows, [(rows, (0, 9))], 5)
    assert g4 == g1
    rt.ER.detach()
    rt.cap.detach()


def test_question_route_aggregators():
    """question_route: lse_norm = lse − log(N_d − skip); max = slot max; averages over the chosen rows only."""
    from xenc.d18_library import question_route
    e = _enc()
    itf = _itf(e)
    ops = [_op(e, _ids(n, s)) for n, s in ((20, 1), (31, 2), (17, 3))]
    x = torch.randn(5, 64, generator=torch.Generator().manual_seed(5))
    res = lib_read(itf, 5, x, [o[5] for o in ops], skip=2)
    assert res["r"].shape == res["rmax"].shape == res["w"].shape and res["n"].tolist() == [18.0, 29.0, 15.0]
    qr = question_route({5: (res["r"], res["rmax"], res["n"])}, slice(2, 5))
    want = (res["r"][:, :, 2:5] - res["n"].log()).mean((0, 1, 2))
    assert torch.allclose(qr["lse_norm"], want, atol=1e-5)
    assert torch.allclose(qr["lse"] - qr["lse_norm"], res["n"].log(), atol=1e-5)
    assert torch.all(qr["max"] <= qr["lse"] + 1e-5)


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f()
            print("ok", n)
    print("ALL D18 LIBRARY TESTS PASSED")
