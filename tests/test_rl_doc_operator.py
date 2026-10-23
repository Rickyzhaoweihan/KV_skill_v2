"""[D4] RL-from-document plumbing (CPU, TinyLM): reward primitives (advantage normalization, k3 KL,
truncated IS, Frobenius clip), one policy-gradient update moves U/W and never the backbone, the
SVD init, and the operator cache round-trip into Library / eval_doc --operator_dir.
Run: python tests/test_rl_doc_operator.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_ROOT = _TESTS.parent
sys.path.insert(0, str(_TESTS))
sys.path.insert(0, str(_ROOT / "scripts"))
import _tiny  # noqa: E402,F401  (bootstraps xenc / deltaskill)

import torch  # noqa: E402

from xenc.reward import (frobenius_clip_, group_advantages, k3_kl, masked_mean, pg_loss,  # noqa: E402
                         token_logprobs, truncated_is_weights)

PY = sys.executable


# ---------------------------------------------------------------------------- reward primitives
def test_group_advantages():
    r = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    a = group_advantages(r, "grpo")
    assert abs(float(a.mean())) < 1e-6
    assert abs(float(a.std(unbiased=False)) - 1.0) < 1e-4
    assert (a[r == 1] > 0).all() and (a[r == 0] < 0).all()
    assert torch.allclose(group_advantages(torch.ones(8), "grpo"), torch.zeros(8))      # no variance -> no signal
    assert torch.allclose(group_advantages(r, "drgrpo"), r - r.mean())
    loo = group_advantages(torch.tensor([1.0, 0.0, 0.0]), "rloo")
    assert torch.allclose(loo, torch.tensor([1.0, -0.5, -0.5]))
    # batched groups normalize independently
    rb = torch.stack([r, 2 * r + 3])
    ab = group_advantages(rb, "grpo")
    assert torch.allclose(ab[0], ab[1], atol=1e-5)


def test_k3_kl_and_is():
    x = torch.randn(10)
    assert torch.allclose(k3_kl(x, x), torch.zeros(10))
    assert (k3_kl(x, x + torch.randn(10)) >= 0).all()
    cur = torch.tensor([0.0, 0.0, 0.0])
    beh = torch.tensor([0.0, -2.0, 1.0])            # ratios 1, e^2 (> cap), e^-1
    w = truncated_is_weights(cur.requires_grad_(), beh, cap=2.0)
    assert not w.requires_grad
    assert torch.allclose(w, torch.tensor([1.0, 2.0, float(torch.exp(torch.tensor(-1.0)))]))


def test_frobenius_clip():
    p1, p2 = torch.zeros(3, 4, requires_grad=True), torch.zeros(5, requires_grad=True)
    p1.grad = torch.full((3, 4), 3.0)
    p2.grad = torch.full((5,), 4.0)
    pre = (12 * 9 + 5 * 16) ** 0.5
    n = frobenius_clip_([p1, p2], 1.0)
    assert abs(n - pre) < 1e-4
    post = (p1.grad.norm() ** 2 + p2.grad.norm() ** 2) ** 0.5
    assert abs(float(post) - 1.0) < 1e-4
    p1.grad = torch.full((3, 4), 0.01)
    p2.grad = None
    frobenius_clip_([p1, p2], 1.0)                    # under the cap: unchanged
    assert torch.allclose(p1.grad, torch.full((3, 4), 0.01))


def test_pg_loss_direction():
    """A positive-advantage sample's log-prob must increase after a gradient step on pg_loss."""
    logits = torch.zeros(2, 3, 5, requires_grad=True)
    comp = torch.tensor([[1, 2, 3], [4, 4, 4]])
    mask = torch.ones(2, 3)
    adv = torch.tensor([1.0, -1.0])
    cur = token_logprobs(logits, comp)
    loss, met = pg_loss(cur, mask, adv, behav_tok_lp=cur.detach(), ref_tok_lp=cur.detach(), beta=0.1)
    loss.backward()
    with torch.no_grad():
        new = token_logprobs(logits - 1.0 * logits.grad, comp)
    assert masked_mean(new, mask)[0] > masked_mean(cur.detach(), mask)[0]
    assert masked_mean(new, mask)[1] < masked_mean(cur.detach(), mask)[1]
    assert abs(met["is_w_mean"] - 1.0) < 1e-6 and met["kl"] == 0.0


# ---------------------------------------------------------------------------- operator + update
def _setup(td):
    from register_doc import DocRuntime, build_injector, tiny_setup
    from xenc.interface import make_interface
    tz = tiny_setup(td, encoder_same=False)
    rt = DocRuntime(tz["reader"], tz["tok"], tz["hub"], tz["docs"], tz["cache_dir"], tz["enc_short"],
                    tz["chunk_tokens"], [0.25, 0.5, 0.75, 1.0])
    d_enc = rt.d_encoder(sorted(tz["docs"]))
    itf = make_interface(rt.d_reader, d_enc, rt.layers)
    with torch.no_grad():
        itf.B.weight.copy_(torch.randn(itf.B.weight.shape, generator=torch.Generator().manual_seed(0)) * 0.5)
        itf.gate_b.fill_(2.0)
    itf.requires_grad_(False)

    class H:
        read_temp, route_temp, top_k, linear_read = 12.0, 10.0, 0, False
    inj = build_injector(tz["reader"], itf, H)
    return tz, rt, itf, inj, d_enc


def test_svd_init():
    from xenc.doc_operator import svd_init
    with tempfile.TemporaryDirectory() as td:
        tz, rt, itf, inj, d_enc = _setup(td)
        lib = rt.library("tinydoc_0")
        U0 = svd_init(lib, m=8)
        for li in rt.layers:
            D = U0[li]
            assert D.shape == (8, d_enc)
            assert torch.allclose(D @ D.T, torch.eye(8), atol=1e-4)            # orthonormal directions
            X = lib.K[li][lib.mask[li]].float()
            # captured variance by the top-8 directions >= that of any random 8 orthonormal directions
            Q, _ = torch.linalg.qr(torch.randn(d_enc, 8, generator=torch.Generator().manual_seed(1)))
            assert (X @ D.T).pow(2).sum() >= (X @ Q).pow(2).sum() - 1e-4
        inj.detach()


def test_one_update_moves_operator_not_backbone():
    from rl_doc_operator import completion_logprobs, sample_group
    from xenc.backbone import build_chat
    from xenc.doc_operator import operator_library, svd_init
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        tz, rt, itf, inj, d_enc = _setup(td)
        model = tz["reader"]
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        itf_before = {k: v.detach().clone() for k, v in itf.state_dict().items()}
        U0 = svd_init(rt.library("tinydoc_0"), m=8)
        U = {li: torch.nn.Parameter(U0[li].clone()) for li in rt.layers}
        W = {li: torch.nn.Parameter(U0[li].clone()) for li in rt.layers}
        params = list(U.values()) + list(W.values())
        opt = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.0)
        it = tz["train"][0]
        ids = build_chat(tz["tok"], tz["hub"], "sys", it.question)
        lib = operator_library(U, W, rt.layers, rt.depths)
        inj.load_library(lib)
        comp, mask, blp, _ = sample_group(model, tz["tok"], tz["hub"], ids, G=6, budget=5, temp=1.0)
        assert comp.shape == (6, 5) and blp.shape == (6, 5) and (blp <= 0).all()
        r = (comp % 2 == 0).float().mean(1)
        if float(r.std()) == 0:
            r[0] += 1.0
        adv = group_advantages(r, "grpo")
        inj.enabled = False
        ref = completion_logprobs(model, ids, comp, mask, grad=False)
        inj.enabled = True
        cur = completion_logprobs(model, ids, comp, mask, grad=True)
        # behaviour log-probs from the sampler equal the trainer's (same model, fp32): IS ratio == 1
        assert torch.allclose(cur.detach(), blp, atol=1e-4)
        loss, _ = pg_loss(cur, mask, adv, blp, ref, beta=0.01, is_cap=2.0)
        loss.backward()
        assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in params)
        assert all(p.grad is None for p in model.parameters())
        assert all(p.grad is None for p in itf.parameters())
        frobenius_clip_(params, 1.0)
        opt.step()
        inj.unload()
        assert all(not torch.equal(U[li].detach(), U0[li]) for li in rt.layers)
        assert all(not torch.equal(W[li].detach(), U0[li]) for li in rt.layers)
        for k, v in model.state_dict().items():
            assert torch.equal(v, before[k]), k
        for k, v in itf.state_dict().items():
            assert torch.equal(v, itf_before[k]), k
        inj.detach()


def test_operator_cache_roundtrip_into_library():
    from xenc.doc_operator import load_operator, operator_cache, operator_library, save_operator
    from xenc.library import Library
    with tempfile.TemporaryDirectory() as td:
        tz, rt, itf, inj, d_enc = _setup(td)
        g = torch.Generator().manual_seed(3)
        U = {li: torch.randn(8, d_enc, generator=g) for li in rt.layers}
        W = {li: torch.randn(8, d_enc, generator=g) for li in rt.layers}
        rho_of = dict(zip(rt.layers, rt.depths))
        cache = operator_cache(U, W, rho_of, "tinydoc_1", "tiny-enc", {"m": 8})
        save_operator(cache, os.path.join(td, "ops"))
        d = load_operator(os.path.join(td, "ops"), "tinydoc_1")
        assert load_operator(os.path.join(td, "ops"), "nope") is None
        lib = Library.from_doc_cache(d, tz["reader"], rt.depths, device="cpu")
        assert lib.n_chunks == 1 and lib.V is not None
        for li in rt.layers:
            assert torch.allclose(lib.K[li][0], U[li]) and torch.allclose(lib.V[li][0], W[li])
            assert bool(lib.mask[li].all()) and lib.mask[li].shape == (1, 8)
        # the loaded Library and the in-memory operator view give the same injected forward
        x = torch.randn(1, 6, rt.d_reader, generator=g)
        inj.load_library(lib)
        y1 = tz["reader"](x)
        inj.unload()
        inj.load_library(operator_library(U, W, rt.layers, rt.depths))
        y2 = tz["reader"](x)
        inj.unload()
        y0 = tz["reader"](x)
        assert torch.allclose(y1, y2, atol=1e-5) and not torch.allclose(y1, y0)
        inj.detach()


def test_rl_then_eval_operator_dir_subprocess():
    def run(script, *args):
        r = subprocess.run([PY, str(_ROOT / "scripts" / script), *args], cwd=str(_ROOT),
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, f"{script} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
        return r.stdout
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "rl")
        run("rl_doc_operator.py", "--tiny", "--steps", "3", "--out", out)
        for sub in ("operators", "operators_init"):
            assert sorted(os.listdir(os.path.join(out, sub))) == [f"tinydoc_{i}.pt" for i in range(3)]
        h = json.load(open(os.path.join(out, "history_tinydoc_0.json")))
        assert len(h["steps"]) == 3 and any(s["n_signal"] > 0 for s in h["steps"])
        assert h["steps"][-1]["U_drift"] > 0
        run("eval_doc.py", "--tiny", "--n", "4", "--interface", os.path.join(out, "interface_ref.pt"),
            "--operator_dir", os.path.join(out, "operators"), "--conds", "kv,wrong_doc,null", "--out",
            os.path.join(td, "ev"))
        res = json.load(open(os.path.join(td, "ev", "eval_test.json")))
        assert res["summary"]["kv"]["n"] == 4 and res["summary"]["wrong_doc"]["n"] == 4
        assert all(r["wrong_doc"]["wrong_doc"] != r["doc_id"] for r in res["items"])


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_") and callable(fn):
            fn()
            print(f"  {k} ok", flush=True)
    print("OK")
