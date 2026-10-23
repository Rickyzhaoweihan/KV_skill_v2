"""eval_d13 --oom_skip: a CUDA OOM on an item records it and EXITS 75 (never continues in-process); a rerun with the same
args resumes from the per-item checkpoint, skips the OOM item and finishes. Default (flag off) still raises.

CPU only: the reader / greedy / chat builder are faked (text-only conds); the items are the real data/qasper_split test
items (first 4)."""

from __future__ import annotations

import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))

import pytest  # noqa: E402
import torch  # noqa: E402

import xenc  # noqa: E402,F401

SPLIT = os.path.join(os.path.dirname(_HERE), "data", "qasper_split")


def _patch(monkeypatch, oom_on: set, calls: list):
    import register_state
    import xenc.backbone as bb
    monkeypatch.setattr(bb, "load_reader", lambda name, device_map=None: (object(), object()))
    monkeypatch.setattr(bb, "build_chat", lambda tok, hub, system, prompt: [1, 2, 3])

    def fake_greedy(model, tok, hub, ids, budget, **kw):
        calls.append(1)
        if len(calls) in oom_on:
            raise torch.OutOfMemoryError("fake CUDA OOM")
        return "<answer>x</answer>", 3
    monkeypatch.setattr(register_state, "greedy", fake_greedy)


def _run(out, oom_skip):
    import eval_d13
    return eval_d13.main(["--split_dir", SPLIT, "--split", "test", "--n", "4", "--conds", "text", "--device", "cpu",
                          "--out", out] + (["--oom_skip", "1"] if oom_skip else []))


@pytest.mark.skipif(not os.path.exists(os.path.join(SPLIT, "test")), reason="data/qasper_split not materialized")
def test_oom_exit_75_then_resume(monkeypatch):
    out = tempfile.mkdtemp()
    calls: list = []
    _patch(monkeypatch, {3}, calls)                    # the 3rd generation (3rd item) OOMs
    with pytest.raises(SystemExit) as ex:
        _run(out, True)
    assert ex.value.code == 75
    skipped = json.load(open(os.path.join(out, "oom_skipped.json")))
    assert len(skipped) == 1 and skipped[0]["cond"] == "text"
    ck = [json.loads(x) for x in open(os.path.join(out, "eval_test.ckpt.jsonl")) if x.strip()]
    assert len(ck) == 2                                  # the two items finished before the OOM
    calls.clear()
    _patch(monkeypatch, set(), calls)
    _run(out, True)                                      # fresh "process": resume
    assert len(calls) == 1                               # only the 4th item is generated; done + OOM items skipped
    E = json.load(open(os.path.join(out, "eval_test.json")))
    keys = [r["key"] for r in E["items"]]
    assert len(keys) == 3 and skipped[0]["key"] not in keys and len(set(keys)) == 3
    ck = [json.loads(x) for x in open(os.path.join(out, "eval_test.ckpt.jsonl")) if x.strip()]
    assert len(ck) == 3


@pytest.mark.skipif(not os.path.exists(os.path.join(SPLIT, "test")), reason="data/qasper_split not materialized")
def test_oom_default_raises(monkeypatch):
    out = tempfile.mkdtemp()
    _patch(monkeypatch, {2}, [])
    with pytest.raises(torch.OutOfMemoryError):
        _run(out, False)
    assert not os.path.exists(os.path.join(out, "oom_skipped.json"))
