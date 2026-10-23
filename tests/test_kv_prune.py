"""D8 phase 2: query-independent column pruning of the native full-attention K/V (xenc/kv_prune.py), CPU, tiny
randomly-initialised REAL Qwen3_5ForCausalLM (eager attention).

  (a) importance == the eager attention weights of the document prefill summed over queries and group heads;
  (b) frac = 1 (all columns, re-ordered by score) == the unpruned prefix cache;
  (c) pruned cache (re-rotated keys) == the FULL cache with the dropped columns masked per KV head, for
      kv_only and for all (DeltaNet states kept), acc / mean / rand scores;
  (d) per-layer budgets (ColumnBudget) == the full cache with per-layer masks.
Run: python tests/test_kv_prune.py   (or pytest)
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
from deltaskill.backbone_utils import decoder_layers  # noqa: E402
from xenc.kv_prune import ColumnBudget, harvest_with_importance, position_scores, prune_states  # noqa: E402
from xenc.native_state import build_prefix_cache, continue_logits  # noqa: E402

T_DOC = 40
DOC = torch.randint(3, 100, (1, T_DOC), generator=torch.Generator().manual_seed(0))
PROMPT = torch.tensor([[5, 9, 11, 3, 7, 2, 8]])


def _setup():
    m = tiny_qwen35()
    st, imp = harvest_with_importance(m, None, ids=DOC, block=16)
    return m, st, imp


class _HeadMask:
    """Full cache: at each full-attention layer mask (per query head) document columns not in keep[li] [n_kv,T]."""

    def __init__(self, m, keep: dict, n_doc: int):
        self.m, self.keep, self.n_doc, self._h = m, keep, n_doc, []

    def __enter__(self):
        blocks = decoder_layers(self.m)
        for li, kp in self.keep.items():
            att = blocks[li].self_attn

            def pre(mod, args, kwargs, kp=kp, li=li):
                x = kwargs["hidden_states"]
                cache = kwargs["past_key_values"]
                B, T = x.shape[:2]
                past = cache.layers[li].get_seq_length()
                kvn = past + T
                g = mod.num_key_value_groups
                col = torch.arange(kvn).view(1, -1)
                row = torch.arange(T).view(-1, 1) + past
                ok = ((col <= row).view(1, 1, T, kvn)).expand(1, kp.shape[0] * g, T, kvn).clone()
                doc_ok = kp.repeat_interleave(g, 0)                           # [n_q, n_doc]
                ok[:, :, :, : self.n_doc] &= doc_ok.view(1, -1, 1, self.n_doc)
                mk = torch.zeros(B, kp.shape[0] * g, T, kvn, dtype=x.dtype)
                mk.masked_fill_(~ok, torch.finfo(x.dtype).min)
                kwargs["attention_mask"] = mk
                return args, kwargs
            self._h.append(att.register_forward_pre_hook(pre, with_kwargs=True))
        return self

    def __exit__(self, *e):
        for h in self._h:
            h.remove()


def test_importance_matches_eager_attention():
    m, st, imp = _setup()
    blocks = decoder_layers(m)
    got, hs = {}, []
    for li in imp:
        def fh(mod, args, out, li=li):
            got[li] = out[1].detach()                                  # [1,n_q,T,T]
        hs.append(blocks[li].self_attn.register_forward_hook(fh))
    with torch.no_grad():
        m(input_ids=DOC, use_cache=False, output_attentions=True)
    for h in hs:
        h.remove()
    for li, A in got.items():
        n_q, n_kv = A.shape[1], imp[li].shape[0]
        ref = A[0].sum(1).reshape(n_kv, n_q // n_kv, T_DOC).sum(1)
        assert torch.allclose(imp[li], ref, atol=1e-4), (li, (imp[li] - ref).abs().max())
        assert abs(float(imp[li].sum()) - n_q * T_DOC) < 1e-2


def test_full_fraction_is_identity():
    m, st, imp = _setup()
    for keep_state in (False, True):
        ref = continue_logits(m, build_prefix_cache(m, st, True, keep_state, keep_state), PROMPT)
        stp, info = prune_states(m, st, imp, frac=1.0, score="acc")
        assert info["b_max"] == T_DOC
        got = continue_logits(m, build_prefix_cache(m, stp, True, keep_state, keep_state), PROMPT)
        assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max()


def test_pruned_equals_masked_full_cache():
    m, st, imp = _setup()
    for score in ("acc", "mean", "rand"):
        for keep_state in (False, True):
            stp, info = prune_states(m, st, imp, frac=0.3, score=score, seed=3)
            b = info["b_max"]
            assert b == 12
            keep = {}
            for li in st["kv"]:
                idx = position_scores(imp[li], score, seed=3 * 1000 + li).topk(b, -1).indices
                keep[li] = torch.zeros(idx.shape[0], T_DOC, dtype=torch.bool).scatter(1, idx, True)
                assert stp["kv"][li][0].shape[2] == b
            got = continue_logits(m, build_prefix_cache(m, stp, True, keep_state, keep_state), PROMPT)
            with _HeadMask(m, keep, T_DOC):
                ref = continue_logits(m, build_prefix_cache(m, st, True, keep_state, keep_state), PROMPT)
            assert torch.allclose(got, ref, atol=1e-4), (score, keep_state, (got - ref).abs().max())
            # and pruning actually changes the output vs the full cache
            full = continue_logits(m, build_prefix_cache(m, st, True, keep_state, keep_state), PROMPT)
            assert not torch.allclose(got, full, atol=1e-4)


def test_per_layer_budgets():
    m, st, imp = _setup()
    layers = sorted(st["kv"])
    budgets = {layers[0]: 0.1, layers[1]: 0.5}                          # others default frac 0.25
    stp, info = prune_states(m, st, imp, frac=0.25, score="acc", budgets=budgets)
    assert info["b_max"] == 20 and info["col_budget"][layers[0]] == 4
    keep = {}
    for li in st["kv"]:
        idx = imp[li].topk(info["col_budget"][li], -1).indices
        keep[li] = torch.zeros(idx.shape[0], T_DOC, dtype=torch.bool).scatter(1, idx, True)
    with ColumnBudget(m, info["col_budget"], info["b_max"]):
        got = continue_logits(m, build_prefix_cache(m, stp, True, False, False), PROMPT)
    with _HeadMask(m, keep, T_DOC):
        ref = continue_logits(m, build_prefix_cache(m, st, True, False, False), PROMPT)
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
