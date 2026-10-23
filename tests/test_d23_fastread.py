"""D23 tests: the batched fast read (xenc/d23_fastread.py) equals the D13 read (xenc/d13_read.py) in fp32.

CPU only, tiny random real Qwen3.5 models (tests/_tiny.tiny_qwen35 reader, test_state_interface._tiny_encoder)."""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402
from xenc.d13_read import D13Interface, D13Reader, EncTeacher, _attend  # noqa: E402
from xenc.d23_fastread import FastRead, adapter_tensors, attend_two_part  # noqa: E402

torch.set_num_threads(2)
LMAP = {1: 2, 5: 5}


def _ids(n, seed):
    return torch.randint(3, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


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
    torch.manual_seed(seed)
    B = {li: torch.randn(64, 48) * 0.1 for li in LMAP}
    itf = D13Interface(e, LMAP, 64, 48, B=B, mlp_hidden=8)
    with torch.no_grad():
        for p in itf.M.parameters():
            p.normal_(0, 0.05)
        itf.g.copy_(torch.tensor([1.1, 0.9]))
    return itf


def test_two_part_attention_equals_concat_attention():
    torch.manual_seed(0)
    Hq, Hkv, hd, N, P, T = 4, 2, 16, 11, 3, 5
    for B, Bo in ((1, 1), (3, 1), (3, 3)):
        q = torch.randn(B, Hq, T, hd)
        Kop, Vop = torch.randn(Bo, Hkv, N, hd), torch.randn(Bo, Hkv, N, hd)
        Kw, Vw = torch.randn(B, Hkv, P + T, hd), torch.randn(B, Hkv, P + T, hd)
        for rb in (1 << 27, 7):                                   # also force row chunking
            got = attend_two_part(q, Kop, Vop, Kw, Vw, 0.25, P, row_budget=rb)
            for b in range(B):
                ob = b if Bo > 1 else 0
                ref = _attend(q[b], Kop[ob], Vop[ob], Kw[b], Vw[b], 0.25)["o"]
                assert torch.allclose(got[b], ref, atol=1e-5), float((got[b] - ref).abs().max())


def test_fast_reader_equals_d13_reader_incremental_and_batched():
    from test_state_interface import _tiny_encoder
    r, e = tiny_qwen35(), _tiny_encoder()
    docs = [_ids(25, 0), _ids(19, 5)]
    rows = [_ids(6, 1), _ids(6, 2)]
    ops = [_op(e, d) for d in docs]
    itf = _itf(e)
    itf.add_delta(4, seed=3)
    snaps = []
    for s in range(2):                                          # two documents' registered adapters (dA + gate)
        g = torch.Generator().manual_seed(10 + s)
        snaps.append({**{f"D.Ua{li}": torch.randn(48, 4, generator=g) * 0.1 for li in LMAP},
                      **{f"D.Va{li}": torch.randn(4, 64, generator=g) * 0.1 for li in LMAP},
                      "g": torch.tensor([1.0, 0.8]) + 0.1 * s})
    R = D13Reader(r, itf)
    ref_logits = []
    for b in range(2):                                          # reference: as-implemented, one document at a time
        with torch.no_grad():
            for li in LMAP:
                itf.D[f"Ua{li}"].copy_(snaps[b][f"D.Ua{li}"])
                itf.D[f"Va{li}"].copy_(snaps[b][f"D.Va{li}"])
            itf.g.copy_(snaps[b]["g"])
        R.load(ops[b], len(docs[b]))
        R.mode = "student"
        with torch.no_grad():
            o1 = r(input_ids=torch.tensor([rows[b][:4]]), use_cache=True)
            o2 = r(input_ids=torch.tensor([rows[b][4:]]), past_key_values=o1.past_key_values, use_cache=True)
        R.mode, R.past = "off", {}
        ref_logits.append(torch.cat([o1.logits, o2.logits], 1))
    # fast, single document, interface's own deltas (as loaded for document 1)
    F = FastRead(r, itf, dtype=torch.float32)
    F.load(ops[1], len(docs[1]))
    F.on = True
    with torch.no_grad():
        o1 = r(input_ids=torch.tensor([rows[1][:4]]), use_cache=True)
        o2 = r(input_ids=torch.tensor([rows[1][4:]]), past_key_values=o1.past_key_values, use_cache=True)
    assert torch.allclose(torch.cat([o1.logits, o2.logits], 1), ref_logits[1], atol=1e-4)
    # fast, batch of the two documents (equal operator lengths required for stacking -> pad doc 1 test via truncation)
    n = min(len(docs[0]), len(docs[1]))
    ops_t = [{k: (K[:, :n], V[:, :n]) for k, (K, V) in o.items()} for o in ops]
    ref_t = []
    for b in range(2):
        dA, gb = adapter_tensors([snaps[b]], LMAP, "cpu")
        F.load(ops_t[b], n, dA=dA, gb=gb)
        with torch.no_grad():
            ref_t.append(r(input_ids=torch.tensor([rows[b]]), use_cache=True).logits)
    stacked = {k: (torch.stack([ops_t[0][k][0], ops_t[1][k][0]]), torch.stack([ops_t[0][k][1], ops_t[1][k][1]]))
               for k in ops_t[0]}
    dA, gb = adapter_tensors(snaps, LMAP, "cpu")
    F.load(stacked, n, dA=dA, gb=gb)
    with torch.no_grad():
        got = r(input_ids=torch.tensor(rows), use_cache=True).logits
    for b in range(2):
        assert torch.allclose(got[b], ref_t[b][0], atol=1e-4), float((got[b] - ref_t[b][0]).abs().max())
    # per-request adapter path == interface-delta path for the same document
    with torch.no_grad():
        for li in LMAP:
            itf.D[f"Ua{li}"].copy_(snaps[0][f"D.Ua{li}"])
            itf.D[f"Va{li}"].copy_(snaps[0][f"D.Va{li}"])
        itf.g.copy_(snaps[0]["g"])
    F.load(ops_t[0], n)
    with torch.no_grad():
        own = r(input_ids=torch.tensor([rows[0]]), use_cache=True).logits
    assert torch.allclose(own, ref_t[0], atol=1e-4)
    F.on = False
    F.detach()
    R.detach()


def test_prefix_cache_restore_equals_fresh_continuation():
    """bench_d23.restore: after a question + decode on a document prefix cache, restoring the DeltaNet states and
    cropping the attention K/V gives the same next-question logits as a fresh prefix cache (Qwen3.5 hybrid cache)."""
    _scripts = os.path.join(os.path.dirname(_HERE), "scripts")
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from bench_d23 import cache_bytes, restore, snap_states
    m = tiny_qwen35()
    doc, q1, q2 = _ids(20, 0), _ids(5, 1), _ids(4, 2)
    with torch.no_grad():
        C = m(input_ids=torch.tensor([doc]), use_cache=True).past_key_values
        cb = cache_bytes(C)
        snap = snap_states(C)
        o = m(input_ids=torch.tensor([q1]), past_key_values=C, use_cache=True)
        m(input_ids=o.logits[:, -1:].argmax(-1), past_key_values=o.past_key_values, use_cache=True)
        restore(C, snap, len(doc))
        got = m(input_ids=torch.tensor([q2]), past_key_values=C, use_cache=True).logits
        ref = m(input_ids=torch.tensor([doc + q2]), use_cache=False).logits[:, -len(q2):]
    assert torch.allclose(got, ref, atol=1e-4), float((got - ref).abs().max())
    assert cb["attn_kv"] > 0 and cb["recurrent"] > 0 and cb["conv"] > 0


def test_truncated_harvest_equals_full():
    """bench_d23.Bench.harvest(trunc=True) stops the encoder after the deepest captured layer; the operator is identical."""
    import types
    _scripts = os.path.join(os.path.dirname(_HERE), "scripts")
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from bench_d23 import Bench
    from test_state_interface import _tiny_encoder
    e = _tiny_encoder()
    doc = _ids(90, 3)
    Tc = EncTeacher(e, [2])                       # layer 2 of 6: layers 3..5 are skipped
    fake = types.SimpleNamespace(rt=types.SimpleNamespace(pre=lambda did: doc, Tc=Tc), enc=e)
    calls = []
    hk = e.model.layers[5].register_forward_hook(lambda m, i, o: calls.append(1))
    with torch.no_grad():
        full, n1 = Bench.harvest(fake, "d", trunc=False)
        assert calls == [1]
        tr, n2 = Bench.harvest(fake, "d", trunc=True)
    assert calls == [1], "the truncated harvest must not run layers after the deepest read layer"
    hk.remove()
    assert n1 == n2 == len(doc) - 64
    for k in full:
        assert torch.equal(full[k][0], tr[k][0]) and torch.equal(full[k][1], tr[k][1])
    Tc.detach()


def test_vllm_plugin_interface_rebuild_equals_original():
    """xenc.d23_vllm_plugin.build_interface: the interface rebuilt from an export (state_dict + encoder text config,
    transformers' own attention / rotary modules) gives the identical read as the original D13Interface."""
    import tempfile
    from test_state_interface import _tiny_encoder
    from xenc.d23_vllm_plugin import build_interface
    e = _tiny_encoder()
    itf = _itf(e)
    doc, rows = _ids(25, 0), _ids(6, 1)
    op = _op(e, doc)
    blob = {"format": "d23_vllm_itf.v1", "layer_map": itf.layer_map, "d_r": itf.d_r, "d_e": itf.d_e,
            "mlp_hidden": itf.mlp_hidden, "identity_B": itf.identity_B, "enc_text_config": e.config.to_dict(),
            "state_dict": {k: v.detach().clone() for k, v in itf.state_dict().items()}}
    path = os.path.join(tempfile.mkdtemp(), "itf.pt")
    torch.save(blob, path)
    itf2, _ = build_interface(path, "cpu")
    x = torch.randn(len(rows), 64)
    for li, en in LMAP.items():
        K, V = op[en]
        a, b = itf.read(li, x, K, V, len(doc))["y"], itf2.read(li, x, K, V, len(doc))["y"]
        assert torch.allclose(a, b, atol=1e-6), float((a - b).abs().max())


def _ref_varlen(q, k, v, max_q, cu_q, max_k, seqused_k=None, softmax_scale=None, causal=False, block_table=None,
                return_softmax_lse=False, fa_version=2, **kw):
    """Reference for vllm_flash_attn.flash_attn_varlen_func (paged, GQA, causal bottom-right) -> (out, lse [Hq,n])."""
    Hq, hd = q.shape[1], q.shape[2]
    Hkv = k.shape[2]
    G = Hq // Hkv
    out = torch.zeros_like(q)
    lse = torch.zeros(Hq, q.shape[0])
    for i in range(len(cu_q) - 1):
        a, b = int(cu_q[i]), int(cu_q[i + 1])
        n = int(seqused_k[i])
        K = k[block_table[i].long()].reshape(-1, Hkv, hd)[:n].float()
        V = v[block_table[i].long()].reshape(-1, Hkv, hd)[:n].float()
        qi = q[a:b].float()                                     # [T, Hq, hd]
        T = b - a
        Kx, Vx = K.repeat_interleave(G, 1), V.repeat_interleave(G, 1)   # head h -> kv head h // G
        s_ = torch.einsum("thd,nhd->htn", qi, Kx) * softmax_scale
        if causal:
            col = torch.arange(n).view(1, n)
            row = torch.arange(T).view(T, 1)
            s_ = s_.masked_fill((col > n - T + row).unsqueeze(0), float("-inf"))
        lse[:, a:b] = torch.logsumexp(s_, -1)
        out[a:b] = torch.einsum("htn,nhd->thd", torch.softmax(s_, -1), Vx).to(q.dtype)
    return out, lse


def test_vllm_plugin_plan_matches_fast_read_mixed_batch_with_adapters():
    test_vllm_plugin_plan_matches_fast_read_mixed_batch(adapters=True)


def test_vllm_plugin_plan_matches_fast_read_mixed_batch_with_hyper_bias():
    """D24 hypernetwork adapters add a per-document write-back bias after B: y = Bmap(o) + b_doc."""
    test_vllm_plugin_plan_matches_fast_read_mixed_batch(adapters=True, bias=True)


def test_vllm_plugin_plan_matches_fast_read_mixed_batch(adapters=False, bias=False):
    """The plugin's planner + operator / own-row pools + LSE merge (with a reference FlashAttention) equal FastRead
    for a batch of two documents: prefill of both, then a decode step of both (different own lengths)."""
    import types as _t
    from test_state_interface import _tiny_encoder
    import xenc.d23_vllm_plugin as plug
    fake = _t.ModuleType("vllm.vllm_flash_attn")
    fake.flash_attn_varlen_func = _ref_varlen
    saved = {k: sys.modules.get(k) for k in ("vllm", "vllm.vllm_flash_attn")}
    sys.modules["vllm"] = sys.modules.get("vllm") or _t.ModuleType("vllm")
    sys.modules["vllm.vllm_flash_attn"] = fake
    try:
        e = _tiny_encoder()
        itf = _itf(e)
        docs = {"dA": _ids(21, 0), "dB": _ids(13, 5)}
        ops = {d: _op(e, ids) for d, ids in docs.items()}
        P = plug.Pools(itf, "cpu", block=4, own_max=16, max_reqs=4, dtype=torch.float32)
        for d, op in ops.items():                                 # load from in-memory ops (bypass files)
            n = len(docs[d])
            nb = -(-n // P.bs)
            P.docs[d] = {"start": sum(x["nblk"] for x in P.docs.values()), "nblk": nb, "n_op": n, "src": d}
        total = sum(x["nblk"] for x in P.docs.values())
        for en in P.enc_layers:
            P.op_k[en] = torch.zeros(total, P.bs, P.hkv, P.hd)
            P.op_v[en] = torch.zeros_like(P.op_k[en])
            for d, op in ops.items():
                st_ = P.docs[d]
                for X, Pl in ((op[en][0], P.op_k[en]), (op[en][1], P.op_v[en])):
                    Pl[st_["start"]: st_["start"] + st_["nblk"]].view(-1, P.hkv, P.hd)[: st_["n_op"]].copy_(X.transpose(0, 1))
        F = FastRead(tiny_qwen35(), itf, dtype=torch.float32)
        rows = {"dA": torch.randn(6, 64), "dB": torch.randn(4, 64)}
        nxt = {"dA": torch.randn(1, 64), "dB": torch.randn(1, 64)}
        snaps = {}
        if adapters:                                            # registered per-document dA (rank 4) + gate, doc dA only
            g_ = torch.Generator().manual_seed(7)
            snaps["dA"] = {**{f"D.Ua{li}": torch.randn(48, 4, generator=g_) * 0.2 for li in LMAP},
                           **{f"D.Va{li}": torch.randn(4, 64, generator=g_) * 0.2 for li in LMAP},
                           "g": torch.tensor([0.7, 1.3])}
            P.adapters["dA"] = {"Ua": {li: snaps["dA"][f"D.Ua{li}"] for li in LMAP},
                                "Va": {li: snaps["dA"][f"D.Va{li}"] for li in LMAP}, "g": snaps["dA"]["g"],
                                "b": ({li: torch.randn(64, generator=g_) * 0.3 for li in LMAP} if bias else None)}
        ref = {}
        for d in docs:                                          # reference: FastRead, one request at a time
            dA_, gb_ = adapter_tensors([snaps[d]], LMAP, "cpu") if d in snaps else (None, None)
            F.load(ops[d], len(docs[d]), dA=dA_, gb=gb_)
            ref[d] = {li: [F.read(li, rows[d].unsqueeze(0))[0]] for li in LMAP}
            F2 = {li: F.read(li, nxt[d].unsqueeze(0))[0] for li in LMAP}
            for li in LMAP:
                ref[d][li].append(F2[li])
        # plugin: step 1 = both prefills (positions 0..), step 2 = both decodes
        rid = {"dA": "d23|dA|x-u1", "dB": "d23|dB|x-u2"}
        for stepi, X in enumerate((rows, nxt)):
            plug.STATE["ctx"] = ([rid["dA"], rid["dB"]], {rid[d]: X[d].shape[0] for d in docs})
            p0 = {d: (0 if stepi == 0 else rows[d].shape[0]) for d in docs}
            pos = torch.cat([torch.arange(p0[d], p0[d] + X[d].shape[0]) for d in docs])
            plan = plug.make_plan(P, pos, pos.shape[0])
            xin = torch.cat([X["dA"], X["dB"]])
            for li in LMAP:
                y, g = plug.read_layer(P, li, xin.index_select(0, plan["idx"]), plan, 2)
                nA = X["dA"].shape[0]
                gi = itf._gi[li]
                ga = snaps["dA"]["g"][gi] if adapters else itf.g[gi]
                assert torch.allclose(g[:nA], ga.expand(nA)) and torch.allclose(g[nA:], itf.g[gi].expand(g.shape[0] - nA))
                refA = ref["dA"][li][stepi] + (P.adapters["dA"]["b"][li] if bias else 0)
                assert torch.allclose(y[:nA], refA, atol=1e-4), float((y[:nA] - refA).abs().max())
                assert torch.allclose(y[nA:], ref["dB"][li][stepi], atol=1e-4), float((y[nA:] - ref["dB"][li][stepi]).abs().max())
        assert P.req[rid["dA"]]["len"] == 7 and P.req[rid["dB"]]["len"] == 5
        P.release(rid["dA"])
        assert rid["dA"] not in P.req and len(P.free_slots) == 3
        F.detach()
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        plug.STATE["ctx"] = None


if __name__ == "__main__":
    test_two_part_attention_equals_concat_attention()
    test_fast_reader_equals_d13_reader_incremental_and_batched()
    test_prefix_cache_restore_equals_fresh_continuation()
    test_truncated_harvest_equals_full()
    test_vllm_plugin_interface_rebuild_equals_original()
    test_vllm_plugin_plan_matches_fast_read_mixed_batch()
    test_vllm_plugin_plan_matches_fast_read_mixed_batch_with_adapters()
    test_vllm_plugin_plan_matches_fast_read_mixed_batch_with_hyper_bias()
    print("test_d23_fastread: OK")
