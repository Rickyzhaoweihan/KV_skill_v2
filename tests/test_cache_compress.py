"""D14: compression of the shipped encoder operator (xenc/cache_compress.py), CPU, tiny real Qwen3_5ForCausalLM.

  (a) fake_quant: 16 bits is the identity; error int8 < int4; token groups reduce the int4 error; scales per channel;
  (b) compress_memory with the empty spec is the identity; pruning only some layers -> per-layer budgets + mask flag;
  (c) shipped_bytes: bf16 / int8 / int4 accounting, needed_upto drops deeper layers, conv / nostate;
  (d) the shipped operator (prefix cache of the document) + continuation over [tail; question] == one prefill of
      [document; question] at the attention outputs, and caches of layers deeper than the captured layer do not
      change it (lossless depth truncation);
  (e) parse_spec grammar.
Run: python tests/test_cache_compress.py   (or pytest)
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
from xenc.cache_compress import OpSpec, compress_memory, fake_quant, parse_spec, shipped_bytes  # noqa: E402
from xenc.kv_prune import harvest_with_importance  # noqa: E402
from xenc.native_state import build_prefix_cache  # noqa: E402

T_DOC = 40
DOC = torch.randint(3, 100, (1, T_DOC), generator=torch.Generator().manual_seed(0))
Q = [5, 9, 11, 3, 7, 2, 8]


def _setup():
    m = tiny_qwen35()
    st, imp = harvest_with_importance(m, None, ids=DOC, block=16)
    return m, st, imp


def test_fake_quant():
    x = torch.randn(1, 2, 300, 16) * torch.linspace(0.1, 5, 16)
    assert torch.equal(fake_quant(x, 16, 2), x)
    e8 = (fake_quant(x, 8, 2) - x).norm() / x.norm()
    e4 = (fake_quant(x, 4, 2) - x).norm() / x.norm()
    e4g = (fake_quant(x, 4, 2, group=32) - x).norm() / x.norm()
    assert e8 < 0.01 and e8 < e4 < 0.2 and e4g <= e4, (e8, e4, e4g)
    q = fake_quant(x, 4, 2)
    for c in range(16):                           # at most 15 distinct levels per channel
        assert q[0, 0, :, c].unique().numel() <= 15


def test_identity_and_budgets():
    m, st, imp = _setup()
    st2, info = compress_memory(m, st, imp, OpSpec())
    assert not info["masked"] and info["b_max"] == T_DOC
    for li in st["kv"]:
        assert torch.equal(st2["kv"][li][0], st["kv"][li][0])
    layers = sorted(st["kv"])
    st3, info3 = compress_memory(m, st, imp, OpSpec(score="acc", frac=0.25, prune_layers=(layers[-1],)))
    assert info3["masked"] and info3["b_max"] == T_DOC and info3["col_budget"][layers[-1]] == 10
    assert all(info3["col_budget"][li] == T_DOC for li in layers[:-1])


def test_bytes():
    m, st, imp = _setup()
    n_kv = sum(x.numel() for kv in st["kv"].values() for x in kv)
    full = shipped_bytes(st, compress_memory(m, st, imp, OpSpec())[1], OpSpec())
    assert full["kv"] == n_kv * st["kv"][1][0].element_size()
    sp8 = OpSpec(kv_bits=8, state_bits=8)
    b8 = shipped_bytes(st, compress_memory(m, st, imp, sp8)[1], sp8)
    assert n_kv <= b8["kv"] < 1.2 * n_kv
    sp = OpSpec(score="acc", frac=0.5)
    _, info = compress_memory(m, st, imp, sp)
    assert abs(shipped_bytes(st, info, sp)["kv"] - full["kv"] / 2) < 1e-6
    deep = max(st["kv"])
    nd = shipped_bytes(st, info, sp, needed_upto=deep - 2)
    assert nd["kv"] < shipped_bytes(st, info, sp)["kv"] and nd["recurrent"] < full["recurrent"]
    assert shipped_bytes(st, info, OpSpec(keep_state=False))["recurrent"] == 0
    z = shipped_bytes(st, info, OpSpec(keep_state=False, keep_conv=False))
    assert z["recurrent"] == 0 and z["conv"] == 0


def _attn_outputs(m, layers):
    out, hs = {}, []
    L = decoder_layers(m)
    for li in layers:
        def h(mod, inp, o, li=li):
            out[li] = (o[0] if isinstance(o, tuple) else o).detach()
        hs.append(L[li].self_attn.register_forward_hook(h))
    return out, hs


def test_prefix_operator_equals_single_prefill_and_truncation():
    m, st, imp = _setup()
    layers = sorted(st["kv"])
    cap_layer = layers[1]
    tail = DOC[0, -6:].tolist()
    body = DOC[:, :-6]
    stb, _ = harvest_with_importance(m, None, ids=body, block=16)
    out, hs = _attn_outputs(m, [cap_layer])
    with torch.no_grad():
        m(input_ids=torch.tensor([DOC[0].tolist() + Q]), use_cache=True)
        ref = out[cap_layer][0, -(6 + len(Q)):].clone()
        m(input_ids=torch.tensor([tail + Q]), past_key_values=build_prefix_cache(m, stb), use_cache=True)
        got = out[cap_layer][0].clone()
        # wipe every cache entry deeper than the captured layer: the captured outputs must not change
        stz = {**stb, "kv": {li: ((torch.zeros_like(K), torch.zeros_like(V)) if li > cap_layer else (K, V))
                             for li, (K, V) in stb["kv"].items()},
               "recurrent": {li: (torch.zeros_like(s) if li > cap_layer else s) for li, s in stb["recurrent"].items()}}
        m(input_ids=torch.tensor([tail + Q]), past_key_values=build_prefix_cache(m, stz), use_cache=True)
        got_z = out[cap_layer][0].clone()
    for h in hs:
        h.remove()
    assert torch.allclose(got, ref, atol=1e-4), (got - ref).abs().max()
    assert torch.allclose(got_z, got, atol=1e-6)


def test_parse_spec():
    s = parse_spec("late:11+15+19+23_acc10_q4g64_s8")
    assert s.prune_layers == (11, 15, 19, 23) and s.score == "acc" and abs(s.frac - 0.1) < 1e-9
    assert s.kv_bits == 4 and s.kv_group == 64 and s.state_bits == 8
    assert parse_spec("full") == OpSpec()
    c = parse_spec("conv_rand25_sc0.5")
    assert not c.keep_state and c.keep_conv and c.score == "rand" and c.state_scale == 0.5
    n = parse_spec("nostate")
    assert not n.keep_state and not n.keep_conv


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
