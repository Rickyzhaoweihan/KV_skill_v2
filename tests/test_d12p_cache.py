"""D12p two-phase runtime: the cached-encoder runtime (CachedRt, phase B) reproduces the live-encoder runtime (D13Rt)
exactly — operator + question-pass teacher records written by the phase-A format (records keyed by the question rows,
holding rows + gold answer) and read back as prefixes. CPU, tiny random Qwen3_5ForCausalLM reader and encoder."""

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

torch.set_num_threads(2)


def _ids(n, seed):
    return torch.randint(3, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _setup(rt, docs, items):
    rt._pre = dict(docs)
    rt.suf = lambda it: list(it["q"])


def test_cached_runtime_matches_live():
    from fit_d12p import CachedRt, D13Rt, op_path, row_key
    from xenc.d12p_read import D13Interface
    reader = tiny_qwen35(d=96, n_heads=4, n_kv=2, head_dim=16, seed=1)
    enc = tiny_qwen35(d=64, seed=2)
    lmap = {1: 1, 3: 3, 5: 3, 7: 5}
    B = {li: torch.randn(96, 64) * 0.1 for li in lmap}
    itf = D13Interface(enc, lmap, 96, 64, B=B, mlp_hidden=8)
    with torch.no_grad():
        for li in lmap:
            itf.A[str(li)].weight.normal_(0, 0.1)
    docs = {"d0": _ids(40, 0), "d1": _ids(33, 1)}
    items = [{"q": _ids(6, 10 + i), "gold": _ids(3, 20 + i), "doc": d} for i, d in enumerate(["d0", "d0", "d1"])]
    live = D13Rt(reader, enc, None, "tiny", itf, tail=8)
    _setup(live, docs, items)
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "op")), os.makedirs(os.path.join(td, "rec"))
        for d in docs:                                  # phase-A format (scripts/cache_d12p.py)
            op, n, _ = live.operator(d)
            torch.save({"kv": {e: (K, V) for e, (K, V) in op.items()}, "n": n}, op_path(td, d))
        for it in items:
            rows = live.rows(it["doc"], it, it["gold"])
            rec = live.teacher(it["doc"], rows)
            torch.save({"rows": rows, "rec": {e: {f: r[f] for f in ("q", "k", "out", "x")} for e, r in rec.items()}},
                       os.path.join(td, "rec", row_key(it["doc"], live.rows(it["doc"], it)) + ".pt"))
        live.R.detach()
        cached = CachedRt(reader, None, "tiny", itf, td, tail=8)
        _setup(cached, docs, items)
        for it in items:
            for tgt in ((), it["gold"]):               # question-only rows (stage 1) and rows + gold (stage 2)
                rows = live.rows(it["doc"], it, tgt)
                a, b = live.teacher(it["doc"], rows), cached.teacher(it["doc"], rows)
                for e in a:
                    for f in ("q", "k", "out", "x"):
                        assert a[e][f].shape == b[e][f].shape, (f, a[e][f].shape, b[e][f].shape)
                        assert torch.allclose(a[e][f], b[e][f], atol=1e-4), (e, f)
        cached.R.detach()
        live.R = type(cached.R)(reader, itf)
        tl = live.s1_pass(items[0]["doc"], items[0], grad=False, metrics=True)
        live.R.detach()
        cached.R = type(cached.R)(reader, itf)
        tc = cached.s1_pass(items[0]["doc"], items[0], grad=False, metrics=True)
        for li in tl:
            for k in ("kl", "rel", "js", "cos"):
                assert abs(tl[li][k] - tc[li][k]) < 1e-4, (li, k, tl[li][k], tc[li][k])




def test_extended_interface_gate0_equals_base():
    """D12p deep: extending an interface to more reader layers (copy of the deepest layer, gate 0) leaves the
    student read (generation path) and the gate-aware teacher-forced stage-1 pass unchanged."""
    from fit_d12p import D13Rt, extend_itf
    from xenc.d12p_read import D13Interface, D13Reader
    reader = tiny_qwen35(d=96, n_heads=4, n_kv=2, head_dim=16, seed=1)
    enc = tiny_qwen35(d=64, seed=2)
    base_map, full_map = {1: 1, 3: 3}, {1: 1, 3: 3, 5: 5, 7: 7}
    itf = D13Interface(enc, base_map, 96, 64, B={li: torch.randn(96, 64) * 0.1 for li in base_map}, mlp_hidden=8)
    with torch.no_grad():
        for li in base_map:
            itf.A[str(li)].weight.normal_(0, 0.1)
            itf.M[str(li)][2].weight.normal_(0, 0.1)
        itf.g.copy_(torch.tensor([1.1, 0.9]))
    ext = extend_itf(itf, enc, full_map, 3)
    assert ext.layers == [1, 3, 5, 7] and float(ext.g[2]) == 0.0 and float(ext.g[3]) == 0.0
    assert torch.equal(ext.A["5"].weight, itf.A["3"].weight) and torch.equal(ext.B["7"], itf.B["3"])
    doc, it = _ids(30, 0), {"q": _ids(6, 5)}
    outs = []
    for I in (itf, ext):
        rt = D13Rt(reader, enc, None, "tiny", I, tail=8)
        rt._pre, rt.suf = {"d": doc}, (lambda it: list(it["q"]))
        op, n, _ = rt.operator("d")
        rows = rt.rows("d", it)
        rt.R.load(op, n)
        rt.R.mode = "student"
        with torch.no_grad():
            lo = reader(input_ids=torch.tensor([rows])).logits
        rt.R.mode = "off"
        rt.R.gate_tf = True
        t = rt.s1_pass("d", it, grad=False, metrics=True)
        outs.append((lo, t))
        rt.R.detach()
        rt.Tc.detach()
    assert torch.allclose(outs[0][0], outs[1][0], atol=1e-5)
    for li in base_map:
        assert abs(outs[0][1][li]["cos"] - outs[1][1][li]["cos"]) < 1e-5


if __name__ == "__main__":
    test_cached_runtime_matches_live()
    test_extended_interface_gate0_equals_base()
    print("test_d12p_cache: OK")
