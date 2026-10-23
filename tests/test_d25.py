"""D25 tests: layer-role probes (xenc/d25_probe.py). CPU only, tiny random Qwen3_5ForCausalLM."""

from __future__ import annotations

import copy
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402
from xenc.d25_probe import AttnTap, LayerTap, first_stable, lens_logits, segment_mass  # noqa: E402
from xenc.state_skill import full_layers  # noqa: E402

torch.set_num_threads(2)


def _ids(n, seed):
    return torch.randint(3, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def test_attn_tap_prefix_cache_matches_eager_weights():
    """prefill the prefix once, deep-copy the cache, run the suffix with want_q: probs == stock eager weights
    of one full-sequence pass at the suffix rows."""
    m = tiny_qwen35()
    pre, suf = _ids(20, 0), _ids(5, 1)
    with torch.no_grad():
        full = m(input_ids=torch.tensor([pre + suf]), output_attentions=True)
    L = full_layers(m)
    T = AttnTap(m, L)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([pre]), use_cache=True)
        snap = T.snapshot()
        for _ in range(2):                          # two questions off the same prefix
            T.restore(snap)
            T.want_q = True
            m(input_ids=torch.tensor([suf]), past_key_values=copy.deepcopy(out.past_key_values), use_cache=True)
            for li in L:
                p = T.probs(li)                     # [Hkv,G,Tq,C]
                ref = full.attentions[L.index(li)][0, :, -5:]   # [Hq,Tq,C]; attentions = full layers only
                assert torch.allclose(p.reshape(ref.shape), ref, atol=1e-5)
                ms = segment_mass(p, {"a": (0, 10), "b": (10, 25)})
                assert torch.allclose(ms["a"] + ms["b"], torch.ones(5), atol=1e-5)
    T.detach()


def test_layer_tap_lens_last_layer_equals_logits():
    m = tiny_qwen35()
    tap = LayerTap(m)
    with torch.no_grad():
        out = m(input_ids=torch.tensor([_ids(12, 2)]))
    H = tap.stack()
    assert H.shape == (8, 64)
    ids = [5, 7, 11]
    lg = lens_logits(m, H, ids)
    assert torch.allclose(lg[-1], out.logits[0, -1, ids].float(), atol=1e-4)
    tap.detach()


def test_first_stable():
    assert first_stable([-1, 2, -1, 1, 3]) == 3
    assert first_stable([1, 1, 1]) == 0
    assert first_stable([1, 1, -1]) == 3


if __name__ == "__main__":
    test_attn_tap_prefix_cache_matches_eager_weights()
    test_layer_tap_lens_last_layer_equals_logits()
    test_first_stable()
    print("test_d25: 3 passed")
