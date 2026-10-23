"""[D] end-to-end CPU plumbing for register_doc / eval_doc / ladder (TinyLM, fake tokenizer).

Runs `register_doc.py --tiny --steps 3`, `eval_doc.py --tiny --n 4` and `ladder.py --tiny` as
subprocesses and checks their outputs; unit-tests one registration step in-process (interface grads
nonzero, backbone grads None) and the eval condition builders. Run: python tests/test_register_plumbing.py
"""

from __future__ import annotations

import json
import math
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

PY = sys.executable


def _run(script: str, *args: str) -> str:
    r = subprocess.run([PY, str(_ROOT / "scripts" / script), *args], cwd=str(_ROOT),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"{script} failed ({r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r.stdout


def test_register_then_eval_subprocess():
    from xenc.interface import load_interface
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "run")
        _run("register_doc.py", "--tiny", "--steps", "3", "--out", out)
        for f in ("interface_best.pt", "interface_last.pt", "history.json", "args.json"):
            assert os.path.exists(os.path.join(out, f)), f
        h = json.load(open(os.path.join(out, "history.json")))
        assert len(h["steps"]) == 3 and h["evals"]
        for r in h["steps"]:
            assert all(math.isfinite(r[k]) for k in ("loss", "kl", "ce", "mag", "aux", "grad_norm"))
        args = json.load(open(os.path.join(out, "args.json")))
        itf, sel, extra = load_interface(os.path.join(out, "interface_best.pt"), args["d_reader"],
                                         args["d_encoder"], args["layers"])
        assert sel is None and extra["depths"] == args["depths"]
        assert itf.B.weight.abs().sum() > 0          # moved off the zero (no-op) init

        _run("eval_doc.py", "--tiny", "--n", "4", "--interface", os.path.join(out, "interface_best.pt"))
        res = json.load(open(os.path.join(out, "eval_test.json")))
        conds = ["closed_book", "text", "kv", "wrong_doc", "null", "oracle_chunk"]
        assert res["conds"] == conds and len(res["items"]) == 4
        for c in conds:
            s = res["summary"][c]
            assert set(s) >= {"score", "mean_gen_tokens", "mean_prompt_tokens", "n"} and s["n"] == 4, (c, s)
            assert 0.0 <= s["score"] <= 1.0
        for rec in res["items"]:
            assert rec["wrong_doc"]["wrong_doc"] != rec["doc_id"]
            assert rec["text"]["prompt_tokens"] > rec["closed_book"]["prompt_tokens"]


def test_trace_selector_same_subprocess():
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "run")
        _run("register_doc.py", "--tiny", "--steps", "2", "--out", out, "--target", "trace",
             "--selector", "1", "--encoder", "same")
        tr = json.load(open(os.path.join(out, "traces.json")))
        assert tr and all("trace" in v and "score" in v for v in tr.values())
        _run("eval_doc.py", "--tiny", "--n", "4", "--conds", "kv,oracle_chunk",
             "--interface", os.path.join(out, "interface_best.pt"))
        assert os.path.exists(os.path.join(out, "eval_test.json"))


def test_ladder_subprocess():
    with tempfile.TemporaryDirectory() as td:
        _run("ladder.py", "--tiny", "--out_dir", td)
        res = json.load(open(os.path.join(td, "tiny.json")))
        assert set(res["readers"]) == {"tiny-a", "tiny-b"}
        for r in res["readers"].values():
            assert r["closed_book"]["n"] == 4 and r["text"]["n"] == 4


def _runtime(td):
    from register_doc import DocRuntime, tiny_setup
    from xenc import DEPTHS
    tz = tiny_setup(td)
    rt = DocRuntime(tz["reader"], tz["tok"], tz["hub"], tz["docs"], tz["cache_dir"], tz["enc_short"],
                    tz["chunk_tokens"], DEPTHS)
    return tz, rt


def test_one_step_grads_in_process():
    from register_doc import SYSTEM_ANSWER, build_injector
    from xenc.doc_tasks import ctx_prompt, gold_text, q_prompt
    from xenc.backbone import with_target
    from xenc.interface import make_interface
    from xenc.losses import (answer_ce, kl_teacher_student, magnitude_penalty, route_aux_loss,
                             student_logits, teacher_logits)

    class H:
        read_temp, route_temp, top_k, linear_read = 12.0, 10.0, 0, False

    with tempfile.TemporaryDirectory() as td:
        tz, rt = _runtime(td)
        it = tz["train"][0]
        itf = make_interface(rt.d_reader, rt.d_encoder([it.doc_id]), rt.layers)
        inj = build_injector(rt.model, itf, H)
        tgt = gold_text(it)
        tids, m, _ = with_target(rt.tok, "tiny", SYSTEM_ANSWER, ctx_prompt(it, rt.docs[it.doc_id]), tgt)
        t = teacher_logits(rt.model, tids, m)
        sids, _, _ = with_target(rt.tok, "tiny", SYSTEM_ANSWER, q_prompt(it), tgt)
        inj.load_library(rt.library(it.doc_id))
        s = student_logits(rt.model, sids, m)
        gold = rt.gold_chunks(it)
        assert gold == [1]
        loss = (kl_teacher_student(t[:-1], s[:-1]) + 0.5 * answer_ce(s, sids[0, -m:])
                + 0.3 * magnitude_penalty(inj) + 0.1 * route_aux_loss(inj, gold))
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in itf.parameters())
        assert all(p.grad is None for p in rt.model.parameters())
        inj.unload()


def test_condition_builders():
    from eval_doc import build_condition_library, null_library, wrong_doc_id
    order = [f"d{i}" for i in range(10)]
    assert wrong_doc_id("d0", order) == "d7" and wrong_doc_id("d5", order) == "d2"
    for d in order:
        assert wrong_doc_id(d, order) != d
    assert wrong_doc_id("a", ["a", "b"]) == "b" and wrong_doc_id("a", ["a"]) is None
    for n in (7, 14):   # shift ≡ 0 mod n must still pick another doc
        o = [f"x{i}" for i in range(n)]
        assert all(wrong_doc_id(d, o) not in (d, None) for d in o)

    with tempfile.TemporaryDirectory() as td:
        tz, rt = _runtime(td)
        doc_order = sorted(tz["docs"])
        items = {i.key: i for i in tz["test"]}
        it = items["tiny_0"]
        own, _ = build_condition_library("kv", it, rt, doc_order)
        wrong, info = build_condition_library("wrong_doc", it, rt, doc_order)
        assert info["wrong_doc"] != it.doc_id and info["wrong_doc"] in doc_order
        li = own.layers[0]
        assert not torch.allclose(own.K[li], wrong.K[li])
        mcq = items["tiny_mcq"]                      # supporting_idx = [2, 0]
        orc, info = build_condition_library("oracle_chunk", mcq, rt, doc_order)
        assert orc.n_chunks == len(mcq.meta["supporting_idx"]) == 2 and info["gold_chunks"] == [0, 2]
        full, _ = build_condition_library("kv", mcq, rt, doc_order)
        assert torch.allclose(orc.K[li][0], full.K[li][0]) and torch.allclose(orc.K[li][1], full.K[li][2])
        nul, _ = build_condition_library("null", it, rt, doc_order)
        assert nul.K[li].shape == own.K[li].shape and not torch.allclose(nul.K[li], own.K[li])
        z = null_library(own, "zero")
        assert z.K[li].abs().sum() == 0
        assert build_condition_library("closed_book", it, rt, doc_order) == (None, {})
        # unknown gold chunk -> skipped
        from xenc.doc_tasks import DocItem
        nogold = DocItem(key="ng", doc_id=it.doc_id, question="q", choices=None, answers=["zzz-not-there"],
                         correct_label=None, task="tiny", meta={})
        lib, info = build_condition_library("oracle_chunk", nogold, rt, doc_order)
        assert lib is None and "skip" in info
        # supporting_idx as raw chunk indices when the doc has no [P{i}] markers
        from register_doc import gold_chunk_indices
        cache = rt.get_cache(it.doc_id)
        assert gold_chunk_indices({"meta": {"supporting_idx": [2, 1]}}, "no markers", cache, lambda t, c: 0) == [1, 2]
        # answer-string mapping
        ans = rt.docs[it.doc_id].split("\n")[2].split()[5]
        hit = gold_chunk_indices({"meta": {}, "answers": [ans], "choices": None}, rt.docs[it.doc_id], cache,
                                 rt.tok_offset)
        first = rt.docs[it.doc_id].find(ans)
        assert hit == [min(rt.tok_offset(rt.docs[it.doc_id], first) // tz["chunk_tokens"], 2)]


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_") and callable(fn):
            fn()
            print(f"  {k} ok", flush=True)
    print("OK")
