"""D10 tests: DeltaNet-state harvest, prefix cache anchor, and the cross-model state read branch.

CPU only, tiny random real Qwen3_5ForCausalLM (tests/_tiny.tiny_qwen35)."""

from __future__ import annotations

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402
from xenc.state_interface import (StateReader, load_state_interface, make_state_interface,  # noqa: E402
                                  save_state_interface)
from xenc.state_skill import build_prefix_cache, harvest_states, linear_layers  # noqa: E402

torch.set_num_threads(2)


def _ids(n, seed):
    return torch.randint(3, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _tiny_encoder(seed=1):
    """A DIFFERENT tiny Qwen3.5 (6 layers, 4 value heads) to exercise S/L shapes."""
    from transformers.models.qwen3_5 import configuration_qwen3_5 as C, modeling_qwen3_5 as M
    torch.manual_seed(seed)
    types = ["linear_attention", "linear_attention", "full_attention"] * 2
    cfg = C.Qwen3_5TextConfig(hidden_size=48, intermediate_size=96, num_hidden_layers=6, num_attention_heads=4,
                              num_key_value_heads=2, head_dim=16, vocab_size=100, layer_types=types,
                              linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                              linear_value_head_dim=16)
    cfg._attn_implementation = "eager"
    return M.Qwen3_5ForCausalLM(cfg).eval().requires_grad_(False)


def _soft_decay(m):
    """Random init draws A ~ U(0,16) => decay exp(-A·softplus) kills any state within a few tokens; use the
    slow-decay regime of a trained model so decay-dependent paths are actually exercised."""
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("A_log"):
                p.copy_(torch.log(torch.rand_like(p) * 0.05 + 0.01))
    return m


def _logits(model, ids, cache=None):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids]), past_key_values=cache, use_cache=cache is not None).logits[0].float()


def _kl(p_logits, q_logits):
    p = torch.log_softmax(p_logits, -1)
    q = torch.log_softmax(q_logits, -1)
    return float((p.exp() * (p - q)).sum(-1).mean())


def test_prefix_anchor_exact():
    """states + conv + full-attn K/V prefix == the document in context (L/L anchor)."""
    m = tiny_qwen35()
    doc, q = _ids(40, 0), _ids(9, 1)
    full = _logits(m, doc + q)[-len(q):]
    st = harvest_states(m, None, "", prefix_ids=doc, keep_kv=True)
    assert set(st["recurrent"]) == set(linear_layers(m)) and st["n_doc"] == 40
    got = _logits(m, q, build_prefix_cache(m, st, keep_kv=True, keep_state=True, keep_conv=True))
    assert torch.allclose(full, got, atol=2e-3), float((full - got).abs().max())
    closed = _logits(m, q)
    assert float((full - closed).abs().max()) > 1e-2      # the document matters in this tiny model


def test_branch_read_is_delta_rule_read():
    """With A = B = I the branch output is exactly the gated-delta-rule read Sᵀ(q/√dk) of the state."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
    m = tiny_qwen35()
    itf = make_state_interface(m, m, a_mode="none", b_mode="none", gate_init=1.0, decay=False)
    S = torch.randn(2, 16, 16)
    q = torch.randn(1, 3, 2, 16)
    qn = q / q.norm(dim=-1, keepdim=True)
    y = itf.read(itf.layers[0], qn, S)
    z = torch.zeros(1, 3, 2, 16)
    ref, _ = torch_recurrent_gated_delta_rule(q, z, z, g=torch.zeros(1, 3, 2), beta=torch.zeros(1, 3, 2),
                                              initial_state=S.unsqueeze(0), output_final_state=False,
                                              use_qk_l2norm_in_kernel=True)
    assert torch.allclose(y, ref, atol=1e-5), float((y - ref).abs().max())
    # full-matrix identity init == structural identity
    itf2 = make_state_interface(m, m, a_mode="full", b_mode="full", gate_init=1.0, decay=False)
    assert torch.allclose(itf2.read(itf.layers[0], qn, S), y, atol=1e-5)


def test_zero_gate_is_closed_book_and_unload():
    m = tiny_qwen35()
    doc, q = _ids(30, 2), _ids(7, 3)
    st = harvest_states(m, None, "", prefix_ids=doc)
    closed = _logits(m, q)
    itf = make_state_interface(m, m, gate_init=0.0, decay=False)
    R = StateReader(m, itf)
    R.load(st)
    assert torch.allclose(_logits(m, q), closed, atol=1e-5)
    with torch.no_grad():
        itf.g.fill_(1.0)
    assert float((_logits(m, q) - closed).abs().max()) > 1e-3
    R.unload()
    assert torch.allclose(_logits(m, q), closed, atol=1e-6)
    R.detach()
    assert torch.allclose(_logits(m, q), closed, atol=1e-6)


def test_identity_branch_moves_toward_in_context():
    """L/L identity read (+K/V prefix, decay) is closer to the in-context teacher than the K/V prefix alone."""
    m = _soft_decay(tiny_qwen35())
    doc, q = _ids(48, 4), _ids(8, 5)
    full = _logits(m, doc + q)[-len(q):]
    st = harvest_states(m, None, "", prefix_ids=doc, keep_kv=True)
    kv_only = _logits(m, q, build_prefix_cache(m, st, keep_kv=True, keep_state=False))
    itf = make_state_interface(m, m, a_mode="none", b_mode="none", gate_init=1.0, decay=True)
    R = StateReader(m, itf)
    R.load(st)
    br = _logits(m, q, build_prefix_cache(m, st, keep_kv=True, keep_state=False))
    R.detach()
    assert _kl(full, br) < _kl(full, kv_only), (_kl(full, br), _kl(full, kv_only))


def test_decode_matches_full_forward():
    """Cached greedy decode with the branch (incl. running decay) == one full forward of the same sequence."""
    m = _soft_decay(tiny_qwen35())
    st = harvest_states(m, None, "", prefix_ids=_ids(25, 6))
    itf = make_state_interface(m, m, gate_init=0.7, decay=True)
    R = StateReader(m, itf)
    R.load(st)
    q = _ids(6, 7)
    R.enabled = False
    base = _logits(m, q)
    R.enabled = True
    assert float((_logits(m, q) - base).abs().max()) > 1e-3     # the decayed branch is not vanishing
    with torch.no_grad():
        out = m.generate(input_ids=torch.tensor([q]), max_new_tokens=5, do_sample=False,
                         output_scores=True, return_dict_in_generate=True, pad_token_id=0)
    seq = out.sequences[0].tolist()
    full = _logits(m, seq[:-1])[len(q) - 1:]
    steps = torch.stack([s[0].float() for s in out.scores])
    assert torch.allclose(full, steps, atol=2e-3), float((full - steps).abs().max())
    R.detach()


def test_cross_model_grads_and_io():
    """S/L: encoder with other head counts; gradients reach only A/B/g; save/load round trip."""
    reader, enc = tiny_qwen35(), _tiny_encoder()
    st = harvest_states(enc, None, "", prefix_ids=_ids(30, 8))
    itf = make_state_interface(reader, enc, a_mode="full", b_mode="full", gate_init=0.1)
    assert set(itf.layer_map.values()) <= set(linear_layers(enc))
    R = StateReader(reader, itf)
    R.load(st)
    ids = torch.tensor([_ids(10, 9)])
    with torch.enable_grad():
        lo = reader(input_ids=ids, use_cache=False).logits
        loss = lo.float().logsumexp(-1).mean() + torch.stack(R.mag_terms).mean()
        loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in itf.parameters())
    assert all(p.grad is None for p in reader.parameters())
    for mode in ("head",):
        it2 = make_state_interface(reader, enc, a_mode=mode, b_mode=mode)
        assert it2.read(it2.layers[0], torch.randn(1, 2, 2, 16), st["recurrent"][itf.layer_map[it2.layers[0]]].float()).shape == (1, 2, 2, 16)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "itf.pt")
        save_state_interface(itf, p, extra={"x": 1})
        it3, ex = load_state_interface(p)
        assert ex["x"] == 1 and it3.layer_map == itf.layer_map
        assert all(torch.equal(a, b) for a, b in zip(it3.state_dict().values(), itf.state_dict().values()))
    R.detach()


# ============================================================================ attention-K/V branch (xattn_read)
def test_attn_branch_identity_equals_kv_prefix():
    """L/L, A = B = I, b = 0: the external attention read == the K/V prefix cache (DeltaNet layers fresh)."""
    from xenc.xattn_read import AttnReader, attn_dims, make_attn_interface
    m = tiny_qwen35()
    doc, q = _ids(37, 10), _ids(9, 11)
    st = harvest_states(m, None, "", prefix_ids=doc, keep_kv=True)
    ref = _logits(m, q, build_prefix_cache(m, st, keep_kv=True, keep_state=False))
    itf = make_attn_interface(m, attn_dims(m))
    R = AttnReader(m, itf)
    R.load(st)
    got = _logits(m, q)
    assert torch.allclose(ref, got, atol=1e-3), float((ref - got).abs().max())
    # cached decode with the branch == full forward
    with torch.no_grad():
        out = m.generate(input_ids=torch.tensor([q]), max_new_tokens=4, do_sample=False, output_scores=True,
                         return_dict_in_generate=True, pad_token_id=0)
    seq = out.sequences[0].tolist()
    full = _logits(m, seq[:-1])[len(q) - 1:]
    assert torch.allclose(full, torch.stack([x[0].float() for x in out.scores]), atol=2e-3)
    R.unload()
    assert torch.allclose(_logits(m, q), _logits(m, q, None), atol=1e-6)
    R.detach()


def test_attn_plus_state_identity_equals_full_prefix_first_token():
    """Both branches at identity: at the FIRST question token the state branch is exact too (no delta-rule
    interaction yet), so logits[0] == the full prefix cache (K/V + states + conv) up to the conv tail."""
    from xenc.xattn_read import AttnReader, attn_dims, make_attn_interface
    m = _soft_decay(tiny_qwen35())
    doc, q = _ids(33, 12), _ids(6, 13)
    st = harvest_states(m, None, "", prefix_ids=doc, keep_kv=True)
    ref = _logits(m, q, build_prefix_cache(m, st, keep_kv=True, keep_state=True, keep_conv=False))
    Ra = AttnReader(m, make_attn_interface(m, attn_dims(m)))
    Rs = StateReader(m, make_state_interface(m, m, "none", "none", gate_init=1.0, decay=True))
    Ra.load(st)
    Rs.load(st)
    got = _logits(m, q)
    kv_only = _logits(m, q, build_prefix_cache(m, st, keep_kv=True, keep_state=False))
    Ra.detach()
    Rs.detach()
    assert _kl(ref, got) < _kl(ref, kv_only), (_kl(ref, got), _kl(ref, kv_only))


def test_attn_branch_cross_model_grads():
    from xenc.xattn_read import AttnReader, attn_dims, load_attn_interface, make_attn_interface, save_attn_interface
    reader, enc = tiny_qwen35(), _tiny_encoder()
    reader_hd = attn_dims(reader)["hd"]
    ed = attn_dims(enc)
    if ed["hd"] != reader_hd:    # the tiny encoder has head_dim 12; build one with 16 for the attention branch
        from transformers.models.qwen3_5 import configuration_qwen3_5 as C, modeling_qwen3_5 as M
        torch.manual_seed(3)
        cfg = C.Qwen3_5TextConfig(hidden_size=48, intermediate_size=96, num_hidden_layers=6, num_attention_heads=2,
                                  num_key_value_heads=1, head_dim=16, vocab_size=100,
                                  layer_types=["linear_attention", "linear_attention", "full_attention"] * 2,
                                  linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                                  linear_value_head_dim=16)
        cfg._attn_implementation = "eager"
        enc = M.Qwen3_5ForCausalLM(cfg).eval().requires_grad_(False)
        ed = attn_dims(enc)
    st = harvest_states(enc, None, "", prefix_ids=_ids(20, 14), keep_kv=True)
    itf = make_attn_interface(reader, ed, bias_init=-1.0)
    R = AttnReader(reader, itf)
    R.load(st)
    with torch.enable_grad():
        lo = reader(input_ids=torch.tensor([_ids(8, 15)]), use_cache=False).logits
        lo.float().logsumexp(-1).mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in itf.parameters())
    assert all(p.grad is None for p in reader.parameters())
    with tempfile.TemporaryDirectory() as d:
        save_attn_interface(itf, os.path.join(d, "a.pt"))
        it2, _ = load_attn_interface(os.path.join(d, "a.pt"))
        assert it2.layer_map == itf.layer_map and it2.e_head == itf.e_head
    R.detach()


def test_attn_branch_native_tail_exact():
    """tail = T: branch over the first n_doc − T document columns + the reader reading the last T document tokens
    itself == K/V prefix cache of doc[:-T] followed by doc[-T:] + question (DeltaNet layers start at the tail)."""
    from xenc.xattn_read import AttnReader, attn_dims, make_attn_interface
    m = tiny_qwen35()
    doc, q, T = _ids(40, 16), _ids(7, 17), 6
    short = harvest_states(m, None, "", prefix_ids=doc[:-T], keep_kv=True)
    ref = _logits(m, doc[-T:] + q, build_prefix_cache(m, short, keep_kv=True, keep_state=False))
    full = harvest_states(m, None, "", prefix_ids=doc, keep_kv=True)
    R = AttnReader(m, make_attn_interface(m, attn_dims(m)))
    R.load(full)
    R.tail = T
    got = _logits(m, doc[-T:] + q)
    R.detach()
    assert torch.allclose(ref, got, atol=1e-3), float((ref - got).abs().max())


def test_encside_identity_replace_is_noop_on_same_tokens():
    """Encoder-side read, L/L, B = I, replace: when the encoder ran on exactly the reader's tokens, the injected
    output equals the reader's own, so logits are unchanged; a different encoder input changes them."""
    from xenc.encside_read import EncSideInterface, EncSideReader
    from xenc.xattn_read import attn_dims
    m = tiny_qwen35()
    ra = attn_dims(m)
    lm = {li: li for li in ra["full"]}
    itf = EncSideInterface(lm, ra["n_q"] * ra["hd"], ra["n_q"] * ra["hd"], "replace", "pre", identity=True)
    R = EncSideReader(m, itf)
    ids = _ids(12, 20)
    base = _logits(m, ids)
    R.capture = {}
    _logits(m, ids)
    own = dict(R.capture)
    R.capture = None
    R.cur = own
    assert torch.allclose(_logits(m, ids), base, atol=1e-4)
    R.cur = None
    R.capture = {}
    _logits(m, _ids(30, 21) + ids)
    ctx = {li: o[:, -len(ids):] for li, o in R.capture.items()}
    R.capture = None
    R.cur = ctx
    assert float((_logits(m, ids) - base).abs().max()) > 1e-3
    R.cur = None
    R.detach()


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print(k, "ok")
    print("OK")
