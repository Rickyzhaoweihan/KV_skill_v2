"""CAD tests (xenc/cad.py, scripts/eval_cad.py): context-aware decoding of the kv read.

CPU only, tiny random real Qwen3_5ForCausalLM reader + a different tiny encoder (as tests/test_d13.py).
  * a = 0 with the reference stream advancing (interleaving) reproduces fit_d13.D13Rt.gen token for token;
  * a > 0 per-step CAD equals a brute-force recompute (full uncached forwards of both conditions at every step);
  * forked streams leave the prefill untouched (two decodes from one prefill are identical);
  * forced-choice letter log-probs equal a full forward over rows + "<answer>";
  * cad_pick is hm_A_cad.py's formula; NQ-Swap adherence rule as scripts/d12_nqswap_adh.py."""

from __future__ import annotations

import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402
from xenc.cad import Stream, cad_decode, cad_pick, combine, fc_letter_logp  # noqa: E402
from xenc.d13_read import D13Interface, D13Reader, EncTeacher  # noqa: E402
from xenc.state_skill import full_layers  # noqa: E402

torch.set_num_threads(2)
HUB = "Qwen/Qwen3.5-9B"


class FakeTok:
    eos_token_id = None

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{int(i)}" for i in ids)

    def convert_tokens_to_ids(self, t):
        return None


def _ids(n, seed):
    return torch.randint(3, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _setup():
    from test_state_interface import _tiny_encoder
    torch.manual_seed(0)
    r, e = tiny_qwen35(), _tiny_encoder()
    doc, rows = _ids(25, 0), _ids(7, 1)
    Tc = EncTeacher(e, full_layers(e))
    Tc.rec, Tc.kv_only = {}, True
    with torch.no_grad():
        e(input_ids=torch.tensor([doc]), use_cache=True)
    op = {k: (v["k"], v["v"]) for k, v in Tc.rec.items()}
    Tc.rec = None
    lmap = {1: 2, 5: 5}
    B = {li: torch.randn(64, 48) * 0.3 for li in lmap}
    itf = D13Interface(e, lmap, 64, 48, B=B, mlp_hidden=8)
    with torch.no_grad():
        itf.g.fill_(0.8)
    R = D13Reader(r, itf)
    rt = types.SimpleNamespace(reader=r, R=R, dev=torch.device("cpu"), tok=FakeTok(), hub=HUB, itf=itf)
    rt.rows = lambda did, it, target_ids=(): list(rows)
    rt.operator = lambda did: (op, len(doc), None)
    return rt, op, len(doc), rows


def _streams(rt, op, n, rows):
    kv, ct = Stream(rt, rows, op, n, "kv"), Stream(rt, rows)
    kv.prefill()
    ct.prefill()
    return kv, ct


def test_alpha0_interleaved_equals_plain_gen():
    from fit_d13 import D13Rt
    rt, op, n, rows = _setup()
    plain = D13Rt.gen(rt, "d", {}, budget=8)
    kv, ct = _streams(rt, op, n, rows)
    a0 = cad_decode(kv, ct, 0.0, 8, rt.tok, HUB, set(), force_ref=True)
    a0b = cad_decode(kv, None, 0.0, 8, rt.tok, HUB, set())
    assert a0["gen"] == plain[0] == a0b["gen"] and a0["gen_tokens"] == plain[1] == 8
    closed = D13Rt.gen(rt, "d", {}, budget=8, read=False)
    c0 = cad_decode(ct, None, 0.0, 8, rt.tok, HUB, set())
    assert c0["gen"] == closed[0]
    assert plain[0] != closed[0]                                    # the read changes the reader here
    assert rt.R.mode == "off" and rt.R.past == {}


@torch.no_grad()
def test_cad_equals_bruteforce_and_fork_is_pure():
    rt, op, n, rows = _setup()
    R, r = rt.R, rt.reader
    kv, ct = _streams(rt, op, n, rows)
    for alpha in (0.5, 2.0):
        got = cad_decode(kv, ct, alpha, 6, rt.tok, HUB, set())
        again = cad_decode(kv, ct, alpha, 6, rt.tok, HUB, set())
        assert got["gen"] == again["gen"]                            # forks never touch the prefilled streams
        seq, new = list(rows), []
        for _ in range(6):
            R.load(op, n)
            R.mode = "student"
            lk = r(input_ids=torch.tensor([seq]), use_cache=False).logits[0, -1].float()
            R.mode, R.past = "off", {}
            lr = r(input_ids=torch.tensor([seq]), use_cache=False).logits[0, -1].float()
            nxt = int(((1 + alpha) * lk - alpha * lr).argmax())
            new.append(nxt)
            seq.append(nxt)
        assert got["gen"] == rt.tok.decode(new), (alpha, got["gen"], new)
    g0 = cad_decode(kv, None, 0.0, 6, rt.tok, HUB, set())["gen"]
    g8 = cad_decode(kv, ct, 8.0, 6, rt.tok, HUB, set())["gen"]
    assert g0 != g8                                                  # a strong contrast changes the decode


@torch.no_grad()
def test_fc_letter_logp_matches_full_forward():
    rt, op, n, rows = _setup()
    R, r = rt.R, rt.reader
    kv, ct = _streams(rt, op, n, rows)
    ansp, lids = [11, 12], [20, 21, 22, 23]
    lp = fc_letter_logp(kv, ansp, lids)
    R.load(op, n)
    R.mode = "student"
    full = r(input_ids=torch.tensor([rows + ansp]), use_cache=False).logits[0, -1].float()
    R.mode, R.past = "off", {}
    ref = torch.log_softmax(full[lids], -1)
    assert torch.allclose(torch.tensor(lp), ref, atol=1e-4)
    lp2 = fc_letter_logp(kv, ansp, lids)                             # the prefill is untouched by the fork
    assert lp2 == lp
    lpc = fc_letter_logp(ct, ansp, lids)
    fullc = r(input_ids=torch.tensor([rows + ansp]), use_cache=False).logits[0, -1].float()
    assert torch.allclose(torch.tensor(lpc), torch.log_softmax(fullc[lids], -1), atol=1e-4)


def test_cad_pick_and_combine():
    lm, lr = [-0.1, -2.5, -3.0], [-0.05, -4.0, -3.0]              # prior strongly favours option 0 too
    assert cad_pick(lm, lr, 0.0) == 0
    s = [(1 + 2.0) * x - 2.0 * y for x, y in zip(lm, lr)]          # hm_A_cad.py:41
    assert cad_pick(lm, lr, 2.0) == max(range(3), key=lambda i: s[i]) == 1
    a = torch.tensor([1.0, 3.0, 2.0], dtype=torch.bfloat16)
    assert combine(a, torch.zeros(3), 0.0) is a                      # a = 0: the main logits themselves
    assert int(combine(a, torch.tensor([0.0, 5.0, 0.0]), 1.0).argmax()) == 2
    p = combine(torch.tensor([5.0, 0.0, -9.0]), torch.tensor([9.0, 0.0, -20.0]), 1.0, plaus=0.1)
    assert int(p.argmax()) == 0 and torch.isinf(p[2])                # plausibility cut removes the implausible token


def test_nqswap_adh_rule():
    from eval_cad import nqswap_adh
    doc = "xx The youngest coach is Zac Taylor of the Bengals. yy"
    it = {"meta": {"span": [0, len(doc)], "prior_answers": ["Sean McVay"]}}
    assert nqswap_adh(it, "<answer>Zac Taylor</answer>", doc) == {"doc_adh": 1.0, "prior_adh": 0.0}
    assert nqswap_adh(it, "<answer>Sean McVay</answer>", doc) == {"doc_adh": 0.0, "prior_adh": 1.0}
    assert nqswap_adh(it, "<answer>Bill Belichick</answer>", doc) == {"doc_adh": 0.0, "prior_adh": 0.0}


class CallTok(FakeTok):
    """Callable fake tokenizer for the eval_cad plumbing test (ids in the tiny vocab; no chat template)."""
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [3 + (ord(c) % 90) for c in str(text)[-40:]] or [3]}


def test_eval_cad_main_plumbing(monkeypatch, tmp_path):
    """scripts/eval_cad.py end to end on CPU (tiny models, fake tokenizer / chat, real d25 FaithEval-CF + NQ-Swap
    items): writes the eval json, every a = 0 interleaving check is identical, MCQ items carry 6 forced-choice rows."""
    import json
    import fit_d13
    import register_state
    import xenc.state_skill as ss
    from test_state_interface import _tiny_encoder
    import eval_cad
    root = os.path.dirname(_HERE)
    if not os.path.exists(os.path.join(root, "data", "d25_fecf_split", "test")):
        return
    tok = CallTok()

    def fake_build(a, init=None):
        torch.manual_seed(0)
        r, e = tiny_qwen35(), _tiny_encoder()
        lmap = {1: 2, 5: 5}
        itf = D13Interface(e, lmap, 64, 48, B={li: torch.randn(64, 48) * 0.3 for li in lmap}, mlp_hidden=8)
        with torch.no_grad():
            itf.g.fill_(0.8)
        return r, e, tok, itf, lmap
    monkeypatch.setattr(fit_d13, "build", fake_build)
    monkeypatch.setattr(ss, "doc_prefix_text", lambda tok_, hub, doc, system: "S" + doc[:60])
    monkeypatch.setattr(ss, "question_suffix_ids", lambda tok_, hub, it, system: tok(it.question)["input_ids"])
    monkeypatch.setattr(register_state, "greedy", lambda *a_, **k_: ("<answer>A</answer>", 3))
    for sd, n in (("d25_fecf_split", 3), ("d25_nqswap_split", 2)):
        out = str(tmp_path / sd)
        assert eval_cad.main(["--arch", "d13", "--interface", "x.pt", "--split_dir", os.path.join(root, "data", sd),
                              "--n", str(n), "--device", "cpu", "--budget", "4", "--tail", "8", "--cad_alphas", "0,1",
                              "--text_alphas", "1", "--out", out]) == 0
        E = json.load(open(os.path.join(out, "eval_test.json")))
        assert len(E["items"]) == n and E["summary"]["cad0_match"] == {"n": n, "match": n}
        for r_ in E["items"]:
            assert {"kv", "closed_tail", "closed_book", "wrong_doc", "text", "kv_cad_ct_0", "kv_cad_ct_1",
                    "kv_cad_cb_1", "text_cad_cb_1"} <= set(r_["gen"])
            if sd == "d25_fecf_split":
                assert set(r_["fc"]["lp"]) == {"kv", "closed_tail", "closed_book", "text", "wrong_doc", "wrong_tail"}
                assert len(r_["fc"]["lp"]["kv"]) == len(r_["fc"]["labels"]) == 4
            else:
                assert "fc" not in r_ and "doc_adh" in r_["gen"]["kv"]


if __name__ == "__main__":
    for f in (test_alpha0_interleaved_equals_plain_gen, test_cad_equals_bruteforce_and_fork_is_pure,
              test_fc_letter_logp_matches_full_forward, test_cad_pick_and_combine, test_nqswap_adh_rule):
        f()
    print("test_cad: all passed")
