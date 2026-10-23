"""D12 tests: context write-back B (xenc/encside_ctx.py). CPU only, tiny random real Qwen3_5ForCausalLM."""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402

torch.set_num_threads(2)


def _ids(n, seed):
    return torch.randint(3, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _blob(spec: dict, d_in_per: int, d_out: int, eye=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    layers = {}
    for li, src in spec.items():
        D = d_in_per * len(src)
        W = torch.eye(d_out, D) if eye else torch.randn(d_out, D, generator=g)
        layers[str(li)] = {"src": [list(x) for x in src], "match": src[0][0], "W": W,
                           "mx": torch.zeros(D), "my": torch.zeros(d_out)}
    return {"format": "xenc.encside_B2.v1", "level": "post", "layers": layers}


def test_ctx_build_incremental_equals_full():
    """Causal-window features built over a prefix and then token by token == features built over the whole run."""
    from xenc.encside_ctx import CtxEncSideInterface
    d, T, T0 = 8, 11, 6
    spec = {5: [(1, 0), (1, 1), (1, 2), (3, 0)], 7: [(3, 0), (1, 0)]}
    itf = CtxEncSideInterface(_blob(spec, d, 4), [5, 7], w_dtype=torch.float32)
    assert itf.enc_layers == [1, 3] and itf.max_lag == 2
    o2 = {1: torch.randn(1, T, d), 3: torch.randn(1, T, d)}
    full = itf.build(o2, reset=True)
    assert full[5].shape == (1, T, 4 * d) and full[7].shape == (1, T, 2 * d)
    assert torch.equal(full[5][0, 4, d:2 * d], o2[1][0, 3]) and torch.equal(full[5][0, 4, 2 * d:3 * d], o2[1][0, 2])
    parts = [itf.build({e: o[:, :T0] for e, o in o2.items()}, reset=True)]
    for t in range(T0, T):
        parts.append(itf.build({e: o[:, t:t + 1] for e, o in o2.items()}))
    for li in (5, 7):
        assert torch.allclose(torch.cat([p[li] for p in parts], dim=1), full[li])


def test_ctx_identity_post_replace_is_noop_on_same_tokens():
    """L/L, W = I on the matched layer only, post level, replace: logits unchanged when the encoder ran on the
    reader's tokens; a W mixing in a second layer or a lagged row changes them."""
    from xenc.encside_ctx import CtxEncSideInterface
    from xenc.encside_read import EncCapture, EncSideReader
    from xenc.xattn_read import attn_dims
    m = tiny_qwen35()
    full = attn_dims(m)["full"]
    d = int(m.config.hidden_size)
    ids = _ids(12, 30)
    base = m(input_ids=torch.tensor([ids]), use_cache=False).logits
    cap = EncCapture(m, full, "post")
    cap.start()
    m(input_ids=torch.tensor([ids]), use_cache=False)
    o2 = cap.get()
    cap.detach()
    itf = CtxEncSideInterface(_blob({li: [(li, 0)] for li in full}, d, d, eye=True), full, w_dtype=torch.float32)
    R = EncSideReader(m, itf)
    R.cur = itf.build(o2, reset=True)
    got = m(input_ids=torch.tensor([ids]), use_cache=False).logits
    R.detach()
    assert torch.allclose(got, base, atol=1e-4), float((got - base).abs().max())
    itf2 = CtxEncSideInterface(_blob({li: [(li, 0), (li, 1)] for li in full}, d, d, eye=False, seed=1), full,
                               w_dtype=torch.float32)
    R = EncSideReader(m, itf2)
    R.cur = itf2.build(o2, reset=True)
    got2 = m(input_ids=torch.tensor([ids]), use_cache=False).logits
    R.detach()
    assert float((got2 - base).abs().max()) > 1e-3


def test_fit_b2_solve_recovers_window_map():
    """fit_encside_B2.solve: a target that is an affine map of (o_e[t], o_e[t-1]) is recovered by w3 (held R² ≈ 1)
    but not by m; the saved blob, applied through CtxEncSideInterface.build/map, reproduces the target."""
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))
    from fit_encside_B2 import solve
    from xenc.encside_ctx import CtxEncSideInterface
    g = torch.Generator().manual_seed(0)
    enc_full, d, dr, li = [1, 3, 5], 6, 5, 7
    A0, A1, b = torch.randn(d, dr, generator=g), torch.randn(d, dr, generator=g), torch.randn(dr, generator=g)
    data = []
    for k in range(12):
        n = 40
        E = {j: torch.randn(n + 2, d, generator=g) * (j + 1) for j in enc_full}
        Y = E[3][2:] @ A0 + E[3][1:-1] @ A1 + b
        data.append(("held" if k >= 10 else "fit", E, {li: Y}))
    out, _ = solve(data, [li], {li: 3}, enc_full, ["m", "w3", "n3"], [1e-6, 1e-2], torch.device("cpu"), "post", {})
    r = {c: out[c]["report"][li]["r2_held"] for c in out}
    assert r["w3"] > 0.999 and r["m"] < 0.9 and r["n3"] < 0.9, r
    assert out["n3"]["layers"][str(li)]["src"] == [[3, 0], [1, 0], [5, 0]]
    itf = CtxEncSideInterface(out["w3"], [li], w_dtype=torch.float32)
    E, Y = data[-1][1], data[-1][2][li]
    f = itf.build({j: E[j][None] for j in itf.enc_layers}, reset=True)[li][0, 2:]
    rel = float((itf.map(li, f) - Y).norm() / Y.norm())       # W is stored in bf16
    assert rel < 1e-2, rel


def test_train_ctx_init_equals_closed_form_and_trains():
    """TrainCtxEncSide: at init it maps exactly like the closed-form blob, reg = 0; a loss through a tiny reader
    gives gradients to U / bias / gain only; export() (gain folded in) reproduces the trained map."""
    from xenc.encside_ctx import CtxEncSideInterface, TrainCtxEncSide, from_tokenwise
    from xenc.encside_read import EncSideReader
    from xenc.xattn_read import attn_dims
    m = tiny_qwen35()
    full = attn_dims(m)["full"]
    d = int(m.config.hidden_size)
    g = torch.Generator().manual_seed(3)
    tw = {"level": "post", "enc_B": {str(li): torch.randn(d, d, generator=g) * 0.1 for li in full},
          "layer_pairs": {li: li for li in full}}
    blob = from_tokenwise(tw)
    ref = CtxEncSideInterface(blob, full, w_dtype=torch.float32)
    tr = TrainCtxEncSide(blob, full, rank=4)
    x = torch.randn(1, 5, d)
    for li in full:
        assert torch.allclose(tr.map(li, x), ref.map(li, x), atol=1e-5)
    assert float(tr.reg()) == 0.0
    R = EncSideReader(m, tr)
    R.cur = {li: torch.randn(1, 6, d) for li in full}
    lo = m(input_ids=torch.tensor([_ids(6, 40)]), use_cache=False).logits
    (lo.float().logsumexp(-1).mean() + tr.reg()).backward()
    assert all(tr.U[str(li)].grad is not None and tr.U[str(li)].grad.abs().sum() > 0 for li in full)
    assert tr.lg.grad is not None and all(p.grad is None for p in m.parameters())
    with torch.no_grad():
        for li in full:
            tr.U[str(li)].add_(0.01)
        tr.lg.add_(0.1)
    R.detach()
    ex = CtxEncSideInterface(tr.export(), full, w_dtype=torch.float32)
    for li in full:
        a, b = ex.map(li, x), tr.map(li, x)
        assert float((a - b).norm() / b.norm()) < 1e-2


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print(k, "ok")
    print("OK")
