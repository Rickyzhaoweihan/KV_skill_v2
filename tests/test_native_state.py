"""D7 phase 2: hybrid-reader document memory (CPU, tiny randomly-initialised REAL Qwen3_5ForCausalLM).

  (a) keep kv+state+conv: continuing the prompt from the harvested prefix == in-context prefill of [doc; prompt]
      (logits within 1e-4), also for the decode step after it;
  (b) keep nothing == the prompt alone (closed book), exactly;
  (c) partial caches (kv only / state only) run, differ from both, and do not mutate the harvest;
  (d) tile_states shapes (head tiling, normalized-depth layer map); greedy_continue runs.
Run: python tests/test_native_state.py   (or pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS))
sys.path.insert(0, str(_TESTS.parent))
from _tiny import tiny_qwen35  # noqa: E402

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from xenc.native_state import (build_prefix_cache, continue_logits, harvest_states, layer_types,  # noqa: E402
                               tile_states)

torch.manual_seed(0)
DOC = torch.randint(0, 100, (1, 57))
PROMPT = torch.randint(0, 100, (1, 9))


def _full(m, ids):
    with torch.no_grad():
        return m(input_ids=ids, use_cache=False).logits


def test_keep_all_equals_in_context():
    m = tiny_qwen35()
    st = harvest_states(m, None, ids=DOC)
    assert st["n_doc"] == 57 and len(st["recurrent"]) == 4 and len(st["kv"]) == 4
    ref = _full(m, torch.cat([DOC, PROMPT], 1))[:, 57:]
    cache = build_prefix_cache(m, st, True, True, True)
    got = continue_logits(m, cache, PROMPT)
    assert (got - ref).abs().max() < 1e-4, float((got - ref).abs().max())
    nxt = got[:, -1:].argmax(-1)
    ref2 = _full(m, torch.cat([DOC, PROMPT, nxt], 1))[:, -1]
    got2 = continue_logits(m, cache, nxt)[:, -1]
    assert (got2 - ref2).abs().max() < 1e-4


def test_keep_nothing_equals_closed_book():
    m = tiny_qwen35()
    st = harvest_states(m, None, ids=DOC)
    ref = _full(m, PROMPT)
    got = continue_logits(m, build_prefix_cache(m, st, False, False, False), PROMPT)
    assert (got - ref).abs().max() < 1e-5


def test_partial_caches():
    m = tiny_qwen35()
    st = harvest_states(m, None, ids=DOC)
    snap = {k: v.clone() for k, v in st["recurrent"].items()}
    full = continue_logits(m, build_prefix_cache(m, st, True, True, True), PROMPT)
    none = continue_logits(m, build_prefix_cache(m, st, False, False, False), PROMPT)
    for kv, s in ((True, False), (False, True)):
        got = continue_logits(m, build_prefix_cache(m, st, kv, s, s), PROMPT)
        assert (got - full).abs().max() > 1e-4 and (got - none).abs().max() > 1e-4
    assert all(torch.equal(snap[k], st["recurrent"][k]) for k in snap)        # harvest untouched


def test_tile_states():
    src = tiny_qwen35(n_layers=6)
    dst = tiny_qwen35(n_layers=8)
    st = harvest_states(src, None, ids=DOC)
    H = st["recurrent"][0].shape[1]
    t = tile_states(st, layer_types(src), layer_types(dst), dst_heads=2 * H)
    assert sorted(t["recurrent"]) == [0, 2, 4, 6]
    assert t["recurrent"][0].shape[1] == 2 * H and t["layer_map"][0] == 0 and t["layer_map"][6] == 4
    t = tile_states(st, layer_types(src), layer_types(dst), dst_heads=H)
    out = continue_logits(dst, build_prefix_cache(dst, t, False, True, True), PROMPT)
    assert torch.isfinite(out).all()


def test_prefix_mask_and_scale():
    from xenc.native_state import PrefixMask
    m = tiny_qwen35()
    st = harvest_states(m, None, ids=DOC)
    full_layers = sorted(st["kv"])
    closed = _full(m, PROMPT)
    with PrefixMask(m, full_layers, st["n_doc"]):          # KV present but masked everywhere, no state
        got = continue_logits(m, build_prefix_cache(m, st, True, False, False), PROMPT)
    assert (got - closed).abs().max() < 1e-4, float((got - closed).abs().max())   # RoPE relative => closed book
    allc = continue_logits(m, build_prefix_cache(m, st, True, True, True), PROMPT)
    with PrefixMask(m, [], st["n_doc"]):
        same = continue_logits(m, build_prefix_cache(m, st, True, True, True), PROMPT)
    assert torch.equal(same, allc)
    with PrefixMask(m, full_layers[:2], st["n_doc"]):
        part = continue_logits(m, build_prefix_cache(m, st, True, True, True), PROMPT)
    assert (part - allc).abs().max() > 1e-4
    z = continue_logits(m, build_prefix_cache(m, st, True, True, True, state_scale=0.0), PROMPT)
    kc = continue_logits(m, build_prefix_cache(m, st, True, False, True), PROMPT)
    assert (z - kc).abs().max() < 1e-5                     # alpha=0 == recurrent dropped, conv kept
    # kv_layers subset with empty other layers (+ PrefixMask n_doc=0 for their shapes) == KV kept + masked
    keep = set(full_layers[:2])
    with PrefixMask(m, [l for l in full_layers if l not in keep], 0):
        a_ = continue_logits(m, build_prefix_cache(m, st, True, True, True, kv_layers=keep), PROMPT)
    with PrefixMask(m, [l for l in full_layers if l not in keep], st["n_doc"]):
        b_ = continue_logits(m, build_prefix_cache(m, st, True, True, True), PROMPT)
    assert (a_ - b_).abs().max() < 1e-4, float((a_ - b_).abs().max())
    lin = sorted(st["recurrent"])
    e = continue_logits(m, build_prefix_cache(m, st, False, True, True, state_layers=set(), conv_layers=set()), PROMPT)
    assert (e - closed).abs().max() < 1e-5                 # empty layer sets == nothing kept
    p1 = continue_logits(m, build_prefix_cache(m, st, False, True, True, state_layers={lin[0]}, conv_layers={lin[0]}), PROMPT)
    assert (p1 - closed).abs().max() > 1e-5


if __name__ == "__main__":
    test_prefix_mask_and_scale()
    test_keep_all_equals_in_context()
    test_keep_nothing_equals_closed_book()
    test_partial_caches()
    test_tile_states()
    print("test_native_state: all passed")
