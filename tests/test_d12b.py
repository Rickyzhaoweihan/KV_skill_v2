"""D12b: register_d12b's cached-operator runtime gives the SAME registration loss and gradients as the live-encoder
runtime (register_d20.D20.loss, dist_w 0), and the operator length check rejects a misaligned prefix. CPU, tiny
random Qwen3_5ForCausalLM reader and encoder."""

from __future__ import annotations

import copy
import os
import sys
import tempfile
from types import SimpleNamespace

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


def test_cached_registration_loss_matches_live():
    import fit_d12p
    import register_d20
    from register_d12b import cached_rt_factory
    from xenc.d12p_read import D13Interface
    from xenc.d20_reg import DocAdapter
    live_cls = fit_d12p.D13Rt
    reader = tiny_qwen35(d=96, n_heads=4, n_kv=2, head_dim=16, seed=1)
    enc = tiny_qwen35(d=64, seed=2)
    lmap = {1: 1, 3: 3, 5: 3, 7: 5}
    itf = D13Interface(enc, lmap, 96, 64, B={li: torch.randn(96, 64) * 0.1 for li in lmap}, mlp_hidden=8)
    with torch.no_grad():
        for li in lmap:
            itf.A[str(li)].weight.normal_(0, 0.1)
    docs = {"d0": _ids(40, 0)}
    it = {"q": _ids(6, 10)}
    tt = _ids(3, 20)
    live = live_cls(reader, enc, None, "tiny", itf, tail=8)
    live._pre, live.suf = dict(docs), (lambda it: list(it["q"]))
    losses, grads = [], []
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "op")), os.makedirs(os.path.join(td, "rec"))
        op, n, _ = live.operator("d0")
        torch.save({"kv": {e: (K, V) for e, (K, V) in op.items()}, "n": n}, fit_d12p.op_path(td, "d0"))
        rows = live.rows("d0", it, tt)
        t_log = torch.randn(len(tt), reader.config.vocab_size if hasattr(reader.config, "vocab_size")
                            else reader.config.get_text_config().vocab_size)
        live.R.detach()
        live.Tc.detach()
        itf0 = itf
        for mode in ("live", "cached"):
            itf = copy.deepcopy(itf0)                     # DocAdapter installs its delta on the interface
            if mode == "live":
                rt = live_cls(reader, enc, None, "tiny", itf, tail=8)
            else:
                rt = cached_rt_factory(td)(reader, enc, None, "tiny", itf, 8, 16000)
                assert isinstance(rt, fit_d12p.CachedRt) and rt.enc is None
            rt._pre, rt.suf = dict(docs), (lambda it: list(it["q"]))
            S = object.__new__(register_d20.D20)
            S.a = SimpleNamespace(kl=1.0, ce=0.5, dist_w=0.0, l2=0.0, rel_w=1.0)
            S.reader, S.itf, S.lmap, S.rt = reader, itf, dict(lmap), rt
            S.B0 = {li: itf.B[str(li)].detach().clone() for li in lmap}
            S.ad = DocAdapter(itf, ["dA", "gate"], 4, 2, 0)
            torch.manual_seed(7)
            with torch.no_grad():                         # a non-zero adapter so the read path matters
                for p in S.ad.parameters():
                    p.normal_(0, 0.05)
            x = {"rows": rows, "tt": tt, "t_log": t_log}
            with torch.enable_grad():
                loss, parts = S.loss("d0", x)
                loss.backward()
            assert parts["dist"] == 0.0 and parts["kl"] > 0
            losses.append(float(loss.detach()))
            grads.append([p.grad.detach().clone() for p in S.ad.parameters() if p.grad is not None])
            S.ad.zero_grad(set_to_none=True)
            rt.R.detach()
            if mode == "live":
                rt.Tc.detach()
        assert abs(losses[0] - losses[1]) < 1e-5, losses
        assert len(grads[0]) == len(grads[1]) > 0
        assert any(float(g.abs().sum()) > 0 for g in grads[1]), "no gradient reaches the per-document adapter"
        for a, b in zip(grads[0], grads[1]):
            assert torch.allclose(a, b, atol=1e-5)
        # misaligned prefix (a different --tail) must be rejected, not silently read
        bad = cached_rt_factory(td)(reader, enc, None, "tiny", itf, 4, 16000)
        bad._pre, bad.checked = dict(docs), set()
        try:
            bad.operator("d0")
            raise RuntimeError("misaligned cached operator was accepted")
        except AssertionError:
            pass
        bad.R.detach()


def test_cli_guards():
    import register_d12b
    for argv, msg in ((["--runtime", "d12p"], "--enc_cache"),
                      (["--enc_cache", "x", "--runtime", "d12p", "--dist_w", "1.0"], "--dist_w 0")):
        try:
            register_d12b.main(argv)
            raise RuntimeError("guard did not fire")
        except AssertionError as e:
            assert msg in str(e), e


def test_hyper_wiring_zero_init_grads_and_override():
    """D12b H on the d12p runtime: zero-init H == base read exactly; gradients reach H through the e2e read (with
    and without checkpointing); override(False) == base; a swapped adapter changes the read."""
    import fit_d12p
    from register_state import kept_logits
    from xenc.d12b_hyper import adapter_of, override, wire
    from xenc.d12p_read import D13Interface
    from xenc.d24_hyper import D24Hyper
    reader = tiny_qwen35(d=96, n_heads=4, n_kv=2, head_dim=16, seed=1)
    enc = tiny_qwen35(d=64, seed=2)
    lmap = {1: 1, 3: 3, 5: 3, 7: 5}
    itf = D13Interface(enc, lmap, 96, 64, B={li: torch.randn(96, 64) * 0.1 for li in lmap}, mlp_hidden=8)
    with torch.no_grad():
        for li in lmap:
            itf.A[str(li)].weight.normal_(0, 0.1)
        itf.g.fill_(1.0)
    for p_ in itf.parameters():
        p_.requires_grad_(False)
    rt = fit_d12p.D13Rt(reader, enc, None, "tiny", itf, tail=8)
    rt._pre, rt.suf = {"d0": _ids(40, 0), "d1": _ids(37, 1)}, (lambda it: list(it["q"]))
    it = {"q": _ids(6, 10)}
    rows = rt.rows("d0", it, _ids(3, 20))

    def logits(mode="e2e"):
        op, n, _ = rt.operator("d0")
        rt.R.load(op, n)
        rt.R.mode = mode
        try:
            return kept_logits(reader, torch.tensor([rows]), 4)
        finally:
            rt.R.mode = "off"

    with torch.no_grad():
        base = logits("student")
    H = D24Hyper(itf, rank=4, n_q=2, d_proj=16, hidden=32, bias_k=4, arch="v2")
    wire(rt, H)
    H.init_stats([rt.operator("d0")[0], rt.operator("d1")[0]])
    with torch.no_grad():
        assert torch.equal(logits("student"), base), "zero-init H must equal the base read"
    for m in list(H.headM.values()) + [H.headG] + list(H.headB.values()):
        torch.nn.init.normal_(m.weight, 0, 0.5)
    for ck in (False, True):
        rt.ckpt = rt.R.ckpt = ck
        H.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits().float().pow(2).mean().backward()
        gh = {n: float(p_.grad.abs().sum()) for n, p_ in H.named_parameters() if p_.grad is not None}
        assert gh.get("trunk.0.weight", 0) > 0 and gh.get("U.1", 0) > 0 and gh.get("headG.weight", 0) > 0, (ck, gh)
    with torch.no_grad():
        own = logits("student")
        assert not torch.allclose(own, base)
        with override(rt, False):
            assert torch.equal(logits("student"), base)
        with override(rt, adapter_of(rt, "d1")):
            assert not torch.allclose(logits("student"), own)
    rt.R.detach()
    rt.Tc.detach()


def test_slots_zero_init_grads_override_stats_and_roundtrip():
    """D12b H-slots: zero-init Wv => read == base read EXACTLY; at step 0 gradients reach Wv (only), after Wv moves
    they reach Wk / the Perceiver (with and without checkpointing); override(False) == base; another doc's slots
    change the read; slot stats are recorded; save/load round trip."""
    import fit_d12p
    from register_state import kept_logits
    from xenc.d12b_hyper import adapter_of, override, wire
    from xenc.d12b_slots import SlotH, load_slots, save_slots, slot_stats_summary
    from xenc.d12p_read import D13Interface
    reader = tiny_qwen35(d=96, n_heads=4, n_kv=2, head_dim=16, seed=1)
    enc = tiny_qwen35(d=64, seed=2)
    lmap = {1: 1, 3: 3, 5: 3, 7: 5}
    itf = D13Interface(enc, lmap, 96, 64, B={li: torch.randn(96, 64) * 0.1 for li in lmap}, mlp_hidden=8)
    with torch.no_grad():
        for li in lmap:
            itf.A[str(li)].weight.normal_(0, 0.1)
        itf.g.fill_(1.0)
    for p_ in itf.parameters():
        p_.requires_grad_(False)
    rt = fit_d12p.D13Rt(reader, enc, None, "tiny", itf, tail=8)
    rt._pre, rt.suf = {"d0": _ids(40, 0), "d1": _ids(37, 1)}, (lambda it: list(it["q"]))
    rows = rt.rows("d0", {"q": _ids(6, 10)}, _ids(3, 20))

    def logits(mode="e2e"):
        op, n, _ = rt.operator("d0")
        rt.R.load(op, n)
        rt.R.mode = mode
        try:
            return kept_logits(reader, torch.tensor([rows]), 4)
        finally:
            rt.R.mode = "off"

    with torch.no_grad():
        base = logits("student")
    H = SlotH(itf, n_slots=5, n_blocks=2)
    wire(rt, H)
    itf._slot_stats = []
    with torch.no_grad():
        assert torch.equal(logits("student"), base), "zero-init slots must equal the base read exactly"
    assert slot_stats_summary(itf)["share"] == 0.0
    with torch.enable_grad():
        logits().float().pow(2).mean().backward()
    g = {n: float(p_.grad.abs().sum()) if p_.grad is not None else 0.0 for n, p_ in H.named_parameters()}
    assert g["Wv.1.weight"] > 0 and g["Wv.7.weight"] > 0, g
    assert g["Wk.1.weight"] == 0 and g["blocks.1.0.Wq.weight"] == 0, "Wk / Perceiver get gradient only once Wv != 0"
    with torch.no_grad():
        for li in lmap:
            H.Wv[str(li)].weight.normal_(0, 0.3)
    for ck in (False, True):
        rt.ckpt = rt.R.ckpt = ck
        H.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits().float().pow(2).mean().backward()
        g = {n: float(p_.grad.abs().sum()) if p_.grad is not None else 0.0 for n, p_ in H.named_parameters()}
        assert g["Wk.3.weight"] > 0 and g["blocks.3.0.Wq.weight"] > 0 and g["lat.5"] > 0, (ck, g)
    with torch.no_grad():
        own = logits("student")
        assert not torch.allclose(own, base)
        ss = slot_stats_summary(itf)
        assert ss["share"] > 0 and 0 < ss["entropy"] <= 1.0 + 1e-6, ss
        with override(rt, False):
            assert torch.equal(logits("student"), base)
        with override(rt, adapter_of(rt, "d1")):
            assert not torch.allclose(logits("student"), own)
    with tempfile.TemporaryDirectory() as td:
        save_slots(H, os.path.join(td, "h.pt"), {"base": "x"})
        H2, ex = load_slots(os.path.join(td, "h.pt"), itf)
        assert ex["base"] == "x"
        op = rt.operator("d0")[0]
        a1, a2 = H(op), H2(op)
        for li in lmap:
            assert torch.allclose(a1["K"][li], a2["K"][li]) and torch.allclose(a1["V"][li], a2["V"][li])
    rt.R.detach()
    rt.Tc.detach()


def test_extra_val_items_train_split_takes_last_docs():
    import fit_d12p
    try:
        fit_d12p.extra_val_items("quality:train:3", 2, 0)
        raise RuntimeError("a train split with k>0 was accepted")
    except AssertionError:
        pass
    it, d = fit_d12p.extra_val_items("quality:train:-2", 2, 0)
    allq = sorted({fit_d12p._get(x, "doc_id") for x in fit_d12p.load_split_items("data/quality_split", "train")})
    assert sorted(d) == allq[-2:] and len(it) <= 4


def test_op_alias_reads_source_operator_and_checks_length():
    import fit_d12p
    from xenc.d12p_read import D13Interface
    reader = tiny_qwen35(d=96, n_heads=4, n_kv=2, head_dim=16, seed=1)
    enc = tiny_qwen35(d=64, seed=2)
    lmap = {1: 1, 3: 3}
    itf = D13Interface(enc, lmap, 96, 64, B={li: torch.randn(96, 64) * 0.1 for li in lmap}, mlp_hidden=8)
    live = fit_d12p.D13Rt(reader, enc, None, "tiny", itf, tail=8)
    live._pre = {"narrativeqa_x": _ids(30, 0)}
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "op"))
        op, n, _ = live.operator("narrativeqa_x")
        torch.save({"kv": {e: (K, V) for e, (K, V) in op.items()}, "n": n}, fit_d12p.op_path(td, "narrativeqa_x"))
        live.R.detach(), live.Tc.detach()
        old = dict(fit_d12p.OP_ALIAS)
        try:
            assert fit_d12p.find_op([td], "nqamc_x") is None             # no alias: not found
            fit_d12p.set_op_alias("nqamc=narrativeqa")
            assert fit_d12p.find_op([td], "nqamc_x") == fit_d12p.op_path(td, "narrativeqa_x")
            rt = fit_d12p.CachedRt(reader, None, "tiny", itf, td, tail=8)
            rt._pre = {"nqamc_x": _ids(30, 0)}
            o2, n2, _ = rt.operator("nqamc_x")
            assert n2 == n and all(torch.equal(o2[e][0], op[e][0].to(o2[e][0].device)) for e in op)
            rt._ops.clear()
            rt._pre = {"nqamc_x": _ids(31, 0)}                             # different prefix length -> refused
            try:
                rt.operator("nqamc_x")
                raise RuntimeError("misaligned alias accepted")
            except AssertionError:
                pass
            rt.R.detach()
        finally:
            fit_d12p.OP_ALIAS.clear()
            fit_d12p.OP_ALIAS.update(old)


if __name__ == "__main__":
    test_cached_registration_loss_matches_live()
    test_cli_guards()
    test_hyper_wiring_zero_init_grads_and_override()
    test_slots_zero_init_grads_override_stats_and_roundtrip()
    test_extra_val_items_train_split_takes_last_docs()
    test_op_alias_reads_source_operator_and_checks_length()
    print("test_d12b: OK")
