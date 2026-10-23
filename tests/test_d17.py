"""D17 tests: the batched reader-query read (xenc/d17_read.py) equals D13's read row by row, and the RL sampler /
policy-log-prob plumbing of scripts/rl_d17.py is consistent. CPU only, tiny random Qwen3_5ForCausalLM."""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402
from test_d13 import _ids, _teacher  # noqa: E402
from xenc.d13_read import D13Interface, D13Reader  # noqa: E402
from xenc.d17_read import D17Reader, read_batched  # noqa: E402

torch.set_num_threads(2)


def _setup():
    from test_state_interface import _tiny_encoder
    r, e = tiny_qwen35(), _tiny_encoder()
    doc = _ids(25, 0)
    op, _, _ = _teacher(e, doc, _ids(6, 1))
    lmap = {1: 2, 5: 5}
    torch.manual_seed(3)
    B = {li: torch.randn(64, 48) * 0.1 for li in lmap}
    itf = D13Interface(e, lmap, 64, 48, B=B, mlp_hidden=8)
    with torch.no_grad():
        for li in lmap:
            itf.M[str(li)][2].weight.normal_(0, 0.1)
        itf.g.fill_(0.8)
    return r, e, doc, op, lmap, itf


def test_read_batched_equals_d13_rows_and_incremental():
    _, _, doc, op, lmap, itf = _setup()
    li = 5
    K, V = op[lmap[li]]
    x = torch.randn(3, 7, 64)
    full = read_batched(itf, li, x, K, V, len(doc), need_logits=True)
    for b in range(3):
        ref = itf.read(li, x[b], K, V, len(doc), need_logits=True)
        assert torch.allclose(full["y"][b], ref["y"], atol=1e-5)
        assert torch.allclose(full["logits"][b], ref["logits"], atol=1e-5)
    r1 = read_batched(itf, li, x[:, :4], K, V, len(doc))
    r2 = read_batched(itf, li, x[:, 4:], K, V, len(doc), past=r1["past"])
    assert torch.allclose(torch.cat([r1["y"], r2["y"]], 1), full["y"], atol=1e-5)


def test_d17_reader_matches_d13_reader_per_row():
    r, _, doc, op, lmap, itf = _setup()
    rows = [_ids(6, 1), _ids(6, 2)]
    R13 = D13Reader(r, itf)
    R13.load(op, len(doc))
    R13.mode = "student"
    ref = []
    with torch.no_grad():
        for rw in rows:
            R13.past = {}
            ref.append(r(input_ids=torch.tensor([rw])).logits[0])
    R13.mode = "off"
    R13.detach()
    R = D17Reader(r, itf)
    R.load(op, len(doc))
    R.mode = "student"
    with torch.no_grad():
        out = r(input_ids=torch.tensor(rows)).logits
    for b in range(2):
        assert torch.allclose(out[b], ref[b], atol=1e-4), float((out[b] - ref[b]).abs().max())
    R.mode, R.past = "e2e", {}
    with torch.enable_grad():
        lg = r(input_ids=torch.tensor(rows)).logits
    assert torch.allclose(lg, out, atol=1e-4)
    params = itf.trainable()
    lg.float().logsumexp(-1).mean().backward()
    assert all(p.grad is not None for p in params) and float(itf.A["1"].weight.grad.abs().sum()) > 0
    R.mode = "off"
    R.detach()


def test_sample_and_policy_logprobs_consistent():
    """Behaviour log-probs of the cached batched sampler == the full-forward e2e log-probs at the same temperature."""
    from rl_d17 import policy_logprobs, sample_batched
    r, _, doc, op, lmap, itf = _setup()
    R = D17Reader(r, itf)
    R.load(op, len(doc))
    prompt = _ids(8, 4)
    torch.manual_seed(0)
    comp, mask, blp, _ = sample_batched(r, R, None, None, prompt, G=3, budget=5, temp=1.3, stops=set(), stop_str=None)
    assert comp.shape == (3, 5) and float(mask.sum()) == 15
    R.load(op, len(doc))
    with torch.no_grad():
        cur = policy_logprobs(r, R, prompt, comp, 1.3)
    assert torch.allclose(cur, blp, atol=1e-4), float((cur - blp).abs().max())
    R.detach()


def test_mem_safe_checkpointed_read_same_logprobs_and_grads():
    """R.mem_safe (torch.utils.checkpoint of the grad-mode read) gives the same log-probs and A/M gradients."""
    from rl_d17 import policy_logprobs
    r, _, doc, op, lmap, itf = _setup()
    R = D17Reader(r, itf)
    prompt = _ids(8, 4)
    comp = torch.tensor([_ids(5, 7), _ids(5, 8)])
    out = []
    for ms in (False, True):
        itf.zero_grad(set_to_none=True)
        R.mem_safe = ms
        R.load(op, len(doc))
        with torch.enable_grad():
            lp = policy_logprobs(r, R, prompt, comp, 1.2)
            lp.sum().backward()
        grads = [p.grad.detach().clone() for p in list(itf.A.parameters()) + list(itf.M.parameters())]
        assert all(g is not None for g in grads) and any(float(g.abs().sum()) > 0 for g in grads)
        out.append((lp.detach(), grads))
    assert torch.allclose(out[0][0], out[1][0], atol=1e-5)
    for g0, g1 in zip(out[0][1], out[1][1]):
        assert torch.allclose(g0, g1, atol=1e-5, rtol=1e-4), float((g0 - g1).abs().max())
    R.detach()


def test_row_chunked_student_sampling_identical():
    """R.row_chunk (no-grad student read over row chunks) gives the same samples and behaviour log-probs."""
    from rl_d17 import sample_batched
    r, _, doc, op, lmap, itf = _setup()
    R = D17Reader(r, itf)
    prompt = _ids(8, 4)
    res = []
    for rc in (0, 2):
        R.row_chunk = rc
        R.load(op, len(doc))
        torch.manual_seed(0)
        comp, mask, blp, _ = sample_batched(r, R, None, None, prompt, G=5, budget=4, temp=1.3, stops=set(), stop_str=None)
        res.append((comp, blp))
    assert torch.equal(res[0][0], res[1][0])
    assert torch.allclose(res[0][1], res[1][1], atol=1e-5)
    R.detach()


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f()
            print("ok", n)
    print("ALL D17 TESTS PASSED")
