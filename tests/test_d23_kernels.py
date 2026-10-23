"""D23 tests: the decode read kernels (xenc/d23_read_kernels.py) and the static (CUDA-graph) decode path of the vLLM
plugin (xenc/d23_vllm_plugin.py). CPU tests always run; the Triton kernel test runs only with CUDA + Triton."""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from xenc.d23_read_kernels import HAVE_TRITON, decode_read_ref  # noqa: E402

torch.set_num_threads(2)


def _pools(nb, bs, hkv, d, gen, dtype=torch.float32, device="cpu"):
    return (torch.randn(nb, bs, hkv, d, generator=gen).to(device, dtype),
            torch.randn(nb, bs, hkv, d, generator=gen).to(device, dtype))


def _case(N=3, hq=16, hkv=4, d=32, bs=4, nb_op=20, nb_own=10, seed=0, dtype=torch.float32, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    op_k, op_v = _pools(nb_op, bs, hkv, d, g, dtype, device)
    own_k, own_v = _pools(nb_own, bs, hkv, d, g, dtype, device)
    q = torch.randn(N, hq, d, generator=g).to(device, dtype)
    op_len = torch.tensor([13, 1, 37][:N] + [5] * max(0, N - 3), dtype=torch.int32)
    own_len = torch.tensor([3, 9, 1][:N] + [2] * max(0, N - 3), dtype=torch.int32)
    op_bt = torch.stack([torch.randperm(nb_op, generator=g)[:10] for _ in range(N)]).int()
    own_bt = torch.stack([torch.randperm(nb_own, generator=g)[:3] for _ in range(N)]).int()
    return q, op_k, op_v, op_bt.to(device), op_len.to(device), own_k, own_v, own_bt.to(device), own_len.to(device)


def test_decode_ref_equals_two_part_lse_merge():
    from xenc.d23_vllm_plugin import ref_varlen
    q, op_k, op_v, op_bt, op_len, own_k, own_v, own_bt, own_len = _case()
    N = q.shape[0]
    cu = torch.arange(N + 1, dtype=torch.int32)
    o1, l1 = ref_varlen(q, op_k, op_v, 1, cu, 64, seqused_k=op_len, softmax_scale=0.3, block_table=op_bt)
    o2, l2 = ref_varlen(q, own_k, own_v, 1, cu, 64, seqused_k=own_len, softmax_scale=0.3, causal=True, block_table=own_bt)
    lse = torch.logaddexp(l1, l2)
    merged = o1.float() * torch.exp(l1 - lse).t().unsqueeze(-1) + o2.float() * torch.exp(l2 - lse).t().unsqueeze(-1)
    ref = decode_read_ref(q, op_k, op_v, op_bt, op_len, own_k, own_v, own_bt, own_len, 0.3)
    assert torch.allclose(merged, ref, atol=1e-5), float((merged - ref).abs().max())


def test_triton_decode_kernel_matches_reference():
    if not (HAVE_TRITON and torch.cuda.is_available()):
        print("  (skipped: needs CUDA + Triton)")
        return
    from xenc.d23_read_kernels import DecodeReader
    for N, nb_op, seed in ((1, 40, 0), (3, 40, 1), (8, 300, 2)):
        q, op_k, op_v, op_bt, op_len, own_k, own_v, own_bt, own_len = _case(
            N=N, d=256, bs=16, nb_op=nb_op, nb_own=40, seed=seed, dtype=torch.bfloat16, device="cuda")
        if N == 8:                                                 # long operator crossing many splits / pages
            op_bt = torch.stack([torch.randperm(nb_op)[:200] for _ in range(N)]).int().cuda()
            op_len = torch.tensor([3200, 1, 17, 2048, 1024, 1023, 15, 3199], dtype=torch.int32).cuda()
            own_bt = torch.stack([torch.randperm(40)[:30] for _ in range(N)]).int().cuda()
            own_len = torch.tensor([1, 480, 33, 64, 2, 5, 300, 17], dtype=torch.int32).cuda()
        dec = DecodeReader(16, 16, 4, 256, max_keys=4096, device="cuda", chunk=512)
        out = dec(q, op_k, op_v, op_bt, op_len, own_k, own_v, own_bt, own_len, 0.0625).clone()
        ref = decode_read_ref(q, op_k, op_v, op_bt, op_len, own_k, own_v, own_bt, own_len, 0.0625)
        rel = float((out - ref).norm() / ref.norm())
        assert rel < 2e-3, (N, rel)
        # graph capture + replay with updated lengths
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            dec(q, op_k, op_v, op_bt, op_len, own_k, own_v, own_bt, own_len, 0.0625)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            o_g = dec(q, op_k, op_v, op_bt, op_len, own_k, own_v, own_bt, own_len, 0.0625)
        op_len.copy_(torch.clamp(op_len // 2, min=1))
        g.replay()
        torch.cuda.synchronize()
        ref2 = decode_read_ref(q, op_k, op_v, op_bt, op_len, own_k, own_v, own_bt, own_len, 0.0625)
        assert float((o_g - ref2).norm() / ref2.norm()) < 2e-3


def test_static_decode_path_loaded_under_bf16_default_dtype():
    """Regression (vllm4): the static pools / adapter rows are loaded from files while vLLM's default dtype is bf16;
    adapter rows must still be fp32 (the validated eager math) and the static path must match FastRead."""
    test_static_decode_path_matches_fast_read(from_files=True)


def test_static_decode_path_hyper_adapter_with_bias():
    """Static rows of a D24 hypernetwork adapter (rank R, gate, per-document write-back bias Bb), loaded from files."""
    test_static_decode_path_matches_fast_read(from_files=True, bias=True)


def test_export_hyper_rows_equal_attach():
    """scripts/d23_export_hyper.to_rows: plugin rows (Ua = U·M, Va = V, g0+dg, Ub·c) == xenc.d24_hyper.attach(itf, H)."""
    if torch.cuda.is_available():     # CPU-only test: on a GPU node the tiny model's GDN dispatches to FLA Triton
        print("  (skipped on a CUDA node: CPU-only test, run by scripts/run_tests.sh)")
        return
    _scripts = os.path.join(os.path.dirname(_HERE), "scripts")
    if _scripts not in sys.path:
        sys.path.insert(0, _scripts)
    from d23_export_hyper import self_check, to_rows
    from test_d23_fastread import _ids, _itf, _op
    from test_state_interface import _tiny_encoder
    from xenc.d24_hyper import D24Hyper
    e = _tiny_encoder()
    itf = _itf(e)
    op = _op(e, _ids(25, 0))
    torch.manual_seed(3)
    H = D24Hyper(itf, rank=4, n_q=2, d_proj=8, hidden=16, bias_k=3).eval()
    with torch.no_grad():
        for p in H.parameters():
            if p.dim() > 1:
                p.normal_(0, 0.1)
            else:
                p.normal_(0, 0.05)
        ad = H(op)
    rows = to_rows(H, itf, ad)
    assert rows["D.Ua1"].shape == (48, 4) and rows["D.Va1"].shape == (4, 64) and rows["Bb1"].shape == (64,)
    assert float(rows["Bb1"].abs().sum()) > 0 and float((rows["g"] - itf.g).abs().sum()) > 0
    self_check(H, itf, ad, rows)


def test_static_decode_path_matches_fast_read(from_files=False, bias=False):
    if torch.cuda.is_available():     # CPU-only test: on a GPU node the tiny model's GDN dispatches to FLA Triton
        print("  (skipped on a CUDA node: CPU-only test, run by scripts/run_tests.sh)")
        return
    """Plugin static path (graph mode): eager prefill of two documents, then one uniform decode step through
    Static.fill + read_static (reference attention), with registered adapter rows and padded rows -> equals FastRead."""
    import types as _t
    from test_d23_fastread import LMAP, _ids, _itf, _op
    from test_state_interface import _tiny_encoder
    from _tiny import tiny_qwen35
    import xenc.d23_vllm_plugin as plug
    from xenc.d23_fastread import FastRead, adapter_tensors
    fake = _t.ModuleType("vllm.vllm_flash_attn")
    fake.flash_attn_varlen_func = plug.ref_varlen
    saved = {k: sys.modules.get(k) for k in ("vllm", "vllm.vllm_flash_attn")}
    sys.modules["vllm"] = sys.modules.get("vllm") or _t.ModuleType("vllm")
    sys.modules["vllm.vllm_flash_attn"] = fake
    try:
        e = _tiny_encoder()
        itf = _itf(e)
        docs = {"dA": _ids(21, 0), "dB": _ids(13, 5)}
        ops = {d: _op(e, ids) for d, ids in docs.items()}
        P = plug.Pools(itf, "cpu", block=4, own_max=16, max_reqs=4, dtype=torch.float32, capacity_tokens=64,
                       max_adapters=2)
        g_ = torch.Generator().manual_seed(7)
        snap = {**{f"D.Ua{li}": torch.randn(48, 4, generator=g_) * 0.2 for li in LMAP},
                **{f"D.Va{li}": torch.randn(4, 64, generator=g_) * 0.2 for li in LMAP}, "g": torch.tensor([0.7, 1.3])}
        if bias:
            snap.update({f"Bb{li}": torch.randn(64, generator=g_) * 0.3 for li in LMAP})
        if from_files:                                           # the real load path, under vLLM's bf16 default dtype
            import tempfile
            dd = tempfile.mkdtemp()
            os.makedirs(os.path.join(dd, "ops"))
            os.makedirs(os.path.join(dd, "adapters"))
            for d, op in ops.items():
                torch.save({"op": op, "n_op": len(docs[d]), "doc": d}, os.path.join(dd, "ops", f"{d}.pt"))
            torch.save(snap, os.path.join(dd, "adapters", "dA.pt"))
            prev = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                P.load_docs(os.path.join(dd, "ops"), ["dA", "dB"], os.path.join(dd, "adapters"))
            finally:
                torch.set_default_dtype(prev)
            assert all(P.ad_Ua[li].dtype == torch.float32 and P.ad_Va[li].dtype == torch.float32 for li in LMAP)
            assert P.ad_rows == {"dA": 1}
        start = 0
        for d, op in (ops.items() if not from_files else ()):
            n = len(docs[d])
            nb = -(-n // P.bs)
            P.docs[d] = {"start": start, "nblk": nb, "n_op": n, "src": d}
            for en in P.enc_layers:
                for X, Pl in ((op[en][0], P.op_k[en]), (op[en][1], P.op_v[en])):
                    Pl[start: start + nb].view(-1, P.hkv, P.hd)[:n].copy_(X.transpose(0, 1))
            start += nb
        if not from_files:
            P.adapters["dA"] = {"Ua": {li: snap[f"D.Ua{li}"] for li in LMAP},
                                "Va": {li: snap[f"D.Va{li}"] for li in LMAP}, "g": snap["g"]}
            for li in LMAP:
                P.ad_Ua[li] = torch.zeros(3, 48, 4)
                P.ad_Va[li] = torch.zeros(3, 4, 64)
                P.ad_Ua[li][1], P.ad_Va[li][1] = snap[f"D.Ua{li}"], snap[f"D.Va{li}"]
            P.ad_g[1] = snap["g"]
            P.ad_rows["dA"] = 1
        S = plug.Static(P, max_n=4, max_doc_tokens=32, attn="ref")
        rows = {"dA": torch.randn(6, 64), "dB": torch.randn(4, 64)}
        nxt = {"dA": torch.randn(1, 64), "dB": torch.randn(1, 64)}
        F = FastRead(tiny_qwen35(), itf, dtype=torch.float32)
        ref = {}
        for d in docs:
            dA_, gb_ = adapter_tensors([snap], LMAP, "cpu") if d == "dA" else (None, None)
            F.load(ops[d], len(docs[d]), dA=dA_, gb=gb_)
            for li in LMAP:
                F.read(li, rows[d].unsqueeze(0))
            ref[d] = {li: F.read(li, nxt[d].unsqueeze(0))[0] for li in LMAP}
        rid = {"dA": "d23|dA|x-u1", "dB": "d23|dB|x-u2"}
        plug.STATE["ctx"] = ([rid["dA"], rid["dB"]], {rid[d]: rows[d].shape[0] for d in docs})   # eager prefill
        plan = plug.make_plan(P, torch.zeros(10, dtype=torch.long), 10)
        xin = torch.cat([rows["dA"], rows["dB"]])
        for li in LMAP:
            plug.read_layer(P, li, xin.index_select(0, plan["idx"]), plan, 2)
        seg, uniform, n_tok = plug.stage_step(P, [rid["dA"], rid["dB"]], {rid["dA"]: 1, rid["dB"]: 1})
        assert uniform and n_tok == 2
        S.fill(P, seg, n_tok)
        x = torch.cat([nxt["dA"], nxt["dB"], torch.randn(2, 64)])           # 2 padded rows (N = 4)
        for li in LMAP:
            y, g = plug.read_static(P, S, li, x, 2)
            refA = ref["dA"][li][0] + (snap[f"Bb{li}"] if bias else 0)
            assert torch.allclose(y[0], refA, atol=1e-4), float((y[0] - refA).abs().max())
            assert torch.allclose(y[1], ref["dB"][li][0], atol=1e-4), float((y[1] - ref["dB"][li][0]).abs().max())
            gi = itf._gi[li]
            assert float(g[0]) == float(snap["g"][gi]) and float(g[1]) == float(itf.g[gi]) and float(g[2:].abs().sum()) == 0
        F.detach()
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        plug.STATE["ctx"] = None


if __name__ == "__main__":
    test_decode_ref_equals_two_part_lse_merge()
    test_triton_decode_kernel_matches_reference()
    test_static_decode_path_matches_fast_read()
    test_static_decode_path_loaded_under_bf16_default_dtype()
    test_static_decode_path_hyper_adapter_with_bias()
    test_export_hyper_rows_equal_attach()
    print("test_d23_kernels: OK")
