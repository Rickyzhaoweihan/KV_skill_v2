"""D26 tests: cross-family reader adapters (xenc/d26_read.py). CPU only: tiny random gpt-oss / GLM-4 / DeepSeek-V2
readers built from configs, tiny Qwen3.5 encoder (tests/_tiny.tiny_qwen35); real tokenizers only if they are in the
turbo HF cache (skipped otherwise)."""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402
from xenc.d13_read import D13Interface, D13Reader, EncTeacher  # noqa: E402
from xenc.d26_read import (D26Reader, align, bucket, chunked_eager, coords, d26_terms, install_chunked_eager,  # noqa: E402
                           pick_read_layers, reader_dims)

torch.set_num_threads(2)


def _ids(n, seed, hi=100):
    return torch.randint(3, hi, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def tiny_gpt_oss(seed=0):
    from transformers import GptOssConfig, GptOssForCausalLM
    torch.manual_seed(seed)
    cfg = GptOssConfig(hidden_size=64, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=16, num_local_experts=4, num_experts_per_tok=2, vocab_size=100,
                       sliding_window=4, layer_types=["sliding_attention", "full_attention"] * 2,
                       max_position_embeddings=512, pad_token_id=0)
    cfg._attn_implementation = "eager"
    m = GptOssForCausalLM(cfg).float().eval().requires_grad_(False)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("sinks"):
                p.normal_(0, 1.0)
    return m


def tiny_glm4(seed=0):
    from transformers import Glm4Config, Glm4ForCausalLM
    torch.manual_seed(seed)
    cfg = Glm4Config(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16, vocab_size=100, max_position_embeddings=512, pad_token_id=0)
    cfg._attn_implementation = "eager"
    return Glm4ForCausalLM(cfg).float().eval().requires_grad_(False)


def tiny_dsv2(seed=0):
    from transformers import DeepseekV2Config, DeepseekV2ForCausalLM
    torch.manual_seed(seed)
    cfg = DeepseekV2Config(hidden_size=64, intermediate_size=128, moe_intermediate_size=32, num_hidden_layers=3,
                           num_attention_heads=4, num_key_value_heads=4, kv_lora_rank=16, q_lora_rank=None,
                           qk_rope_head_dim=8, v_head_dim=16, qk_nope_head_dim=16, head_dim=8, n_routed_experts=4,
                           num_experts_per_tok=2, n_shared_experts=1, first_k_dense_replace=1, vocab_size=100,
                           n_group=1, topk_group=1, max_position_embeddings=512, pad_token_id=0)
    cfg._attn_implementation = "eager"
    return DeepseekV2ForCausalLM(cfg).float().eval().requires_grad_(False)


READERS = {"gpt_oss": tiny_gpt_oss, "glm4": tiny_glm4, "deepseek_v2": tiny_dsv2}


# ----------------------------------------------------------------------------- structure
def test_pick_read_layers():
    q32 = {"n_layers": 32, "full": [3, 7, 11, 15, 19, 23, 27, 31]}
    assert pick_read_layers(q32, q32) == {3: 3, 7: 7, 11: 11, 15: 15, 19: 19}           # = D13's 9B map
    oss = {"n_layers": 24, "full": list(range(1, 24, 2))}
    assert pick_read_layers(oss, q32) == {3: 3, 5: 7, 9: 11, 11: 15, 15: 19}
    assert pick_read_layers({"n_layers": 40, "full": list(range(40))}, q32) == {4: 3, 9: 7, 14: 11, 19: 15, 24: 19}
    assert pick_read_layers({"n_layers": 27, "full": list(range(27))}, q32) == {3: 3, 6: 7, 9: 11, 13: 15, 16: 19}
    d = reader_dims(tiny_gpt_oss())
    assert d["full"] == [1, 3] and d["family"] == "gpt_oss"
    assert reader_dims(tiny_glm4())["full"] == [0, 1, 2, 3]


# ----------------------------------------------------------------------------- alignment
def _char_tok(S, step):
    return [(i, min(i + step, len(S))) for i in range(0, len(S), step)]


def test_coords_align_identity_and_causal():
    S = "<s>HEADER" + "the quick brown fox" + "<t>"
    content = ["the quick brown fox"]
    o = _char_tok(S, 3)
    idx, valid = align(coords(S, o, content), coords(S, o, content))
    assert valid.all() and idx.tolist() == list(range(len(o)))
    # two tokenizations of the same content, different templates: aligned encoder rows never consumed more content
    Sr, Se = "[gMASK]" + content[0] + "<|assistant|>", "<|im_start|>" + content[0] + "<|im_end|>\nassistant\n"
    orr, oe = _char_tok(Sr, 2), _char_tok(Se, 5)
    cr, ce = coords(Sr, orr, content), coords(Se, oe, content)
    assert all(b >= a for a, b in zip(cr, cr[1:])) and all(b >= a for a, b in zip(ce, ce[1:]))
    idx, valid = align(cr, ce)
    for t in range(len(cr)):
        if valid[t]:
            assert ce[int(idx[t])] <= cr[t] + 1e-6
    assert idx[-1] == len(oe) - 1                       # the last template rows align to each other
    assert not valid[0]                                 # reader header row has no encoder counterpart


def test_real_tokenizers_alignment():
    from transformers import AutoTokenizer
    from xenc.d26_read import chat_str, encode, suffix_str
    from xenc.doc_tasks import ctx_prompt
    from register_doc import SYSTEM_ANSWER
    try:
        tok_e = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    except Exception:
        print("  (skipped: Qwen tokenizer not cached)")
        return
    item = {"question": "Which drug was stopped after the second visit?", "choices": None, "answers": ["warfarin"]}
    doc = "Patient ID 7. On admission, warfarin 5 mg daily; ﾃｽﾄ ü. Stopped after the 2nd visit (INR 4.1)."
    user = ctx_prompt(item, doc)
    Se = chat_str(tok_e, "Qwen/Qwen3.5-4B", SYSTEM_ANSWER, user)
    ide, oe = encode(tok_e, Se)
    ce = coords(Se, oe, [SYSTEM_ANSWER, user])
    for hub in ("openai/gpt-oss-20b", "zai-org/GLM-4-9B-0414", "deepseek-ai/DeepSeek-V2-Lite-Chat", "Qwen/Qwen3.5-9B"):
        try:
            tok = AutoTokenizer.from_pretrained(hub)
        except Exception:
            print(f"  (skipped {hub}: not cached)")
            continue
        Sr = chat_str(tok, hub, SYSTEM_ANSWER, user)
        idr, orr = encode(tok, Sr)
        cr = coords(Sr, orr, [SYSTEM_ANSWER, user])
        idx, valid = align(cr, ce)
        content = torch.tensor([c >= 1.0 for c in cr])
        assert bool(valid[content].all()) and valid.float().mean() > 0.5
        # causal + tight: the aligned encoder row's decoded content is a prefix of the reader row's
        n_ok = 0
        for t in range(len(idr)):
            if valid[t]:
                assert ce[int(idx[t])] <= cr[t] + 1e-6
                n_ok += abs(cr[t] - ce[int(idx[t])]) < 8
        assert n_ok / int(valid.sum()) > 0.9, hub
        s, qc = suffix_str(tok, hub, item, SYSTEM_ANSWER)
        assert s.startswith("\n\n## Question\n") and qc in s
        if "Qwen3.5-9B" in hub:
            assert idr == ide and idx.tolist() == list(range(len(ide)))       # same family: identity alignment


def test_bucket_terms_zero_when_equal():
    torch.manual_seed(0)
    Hkv, G, T, N = 2, 2, 5, 7
    s = torch.randn(Hkv, G, T, N + T)
    s = s.masked_fill(torch.arange(N + T).view(1, -1) > N + torch.arange(T).view(-1, 1), float("-inf"))
    y = torch.randn(T, 8)
    idx, valid = torch.arange(T), torch.ones(T, dtype=torch.bool)
    t = d26_terms(s, s.clone(), y, y.clone(), idx, valid, N, metrics=True)
    assert t["kl"] < 1e-6 and t["rel"] < 1e-9 and t["top5"] == 1.0
    b = bucket(s, N)
    assert torch.allclose(torch.softmax(b, -1)[..., -1], torch.softmax(s, -1)[..., N:].sum(-1), atol=1e-6)


# ----------------------------------------------------------------------------- attention
def test_chunked_eager_matches_stock():
    for fam, mk in (("gpt_oss", tiny_gpt_oss), ("deepseek_v2", tiny_dsv2)):
        m = mk()
        ids = torch.tensor([_ids(23, 3)])
        with torch.no_grad():
            ref = m(input_ids=ids).logits
        import importlib
        mod = importlib.import_module({"gpt_oss": "transformers.models.gpt_oss.modeling_gpt_oss",
                                       "deepseek_v2": "transformers.models.deepseek_v2.modeling_deepseek_v2"}[fam])
        orig = mod.eager_attention_forward
        try:
            mod.eager_attention_forward = lambda *a, **k: chunked_eager(*a, chunk=4, **k)   # noqa: E731
            with torch.no_grad():
                got = m(input_ids=ids).logits
                out = m(input_ids=ids[:, :20], use_cache=True)                     # + cached decode steps
                st = m(input_ids=ids[:, 20:], past_key_values=out.past_key_values).logits
        finally:
            mod.eager_attention_forward = orig
        assert torch.allclose(ref, got, atol=1e-4), (fam, float((ref - got).abs().max()))
        assert torch.allclose(ref[:, 20:], st, atol=1e-4), fam
    assert install_chunked_eager("glm4") is False


# ----------------------------------------------------------------------------- the read in foreign readers
def _setup(reader):
    enc = tiny_qwen35()
    from xenc.state_skill import full_layers
    r = reader_dims(reader)
    e = {"n_layers": 8, "full": full_layers(enc)}
    lmap = pick_read_layers(r, e, enc_layers=e["full"][: min(2, len(r["full"]))])
    B = {li: torch.randn(r["d"], 64) * 0.05 for li in lmap}
    itf = D13Interface(enc, lmap, r["d"], 64, B=B, mlp_hidden=8)
    with torch.no_grad():
        for li in lmap:
            itf.A[str(li)].weight.normal_(0, 0.1)
    Tc = EncTeacher(enc, sorted(set(lmap.values())))
    Tc.rec, Tc.kv_only = {}, True
    doc = _ids(30, 0)
    with torch.no_grad():
        enc(input_ids=torch.tensor([doc]))
    op = {k: (v["k"], v["v"]) for k, v in Tc.rec.items()}
    Tc.rec = None
    Tc.detach()
    return itf, op, len(doc)


def test_reader_patch_all_families():
    for fam, mk in READERS.items():
        reader = mk()
        itf, op, N = _setup(reader)
        rows = _ids(9, 5)
        x = torch.tensor([rows])
        with torch.no_grad():
            ref = reader(input_ids=x).logits
        R = D26Reader(reader, itf)
        R.load(op, N)
        with torch.no_grad():
            assert torch.allclose(reader(input_ids=x).logits, ref), fam               # mode off = stock
            R.mode = "student"
            full = reader(input_ids=x).logits
            R.past = {}
            o1 = reader(input_ids=x[:, :6], use_cache=True)                            # incremental == full
            o2 = reader(input_ids=x[:, 6:], past_key_values=o1.past_key_values, use_cache=True).logits
            R.past = {}
            assert not torch.allclose(full, ref, atol=1e-3), fam                        # the read changes the output
            assert torch.allclose(full[:, 6:], o2, atol=1e-4), (fam, float((full[:, 6:] - o2).abs().max()))
            itf.g.zero_()                                                               # gate 0 = stock
            assert torch.allclose(reader(input_ids=x).logits, ref, atol=1e-5), fam
            R.past = {}
            itf.g.fill_(1.0)
            R.keep0 = 2                                                                 # BOS rows keep stock output
            k2 = reader(input_ids=x).logits
            R.past = {}
            assert torch.allclose(k2[:, :2], ref[:, :2], atol=1e-5) and not torch.allclose(k2[:, 2:], ref[:, 2:], atol=1e-3)
            R.keep0 = 0
        R.mode, R.need_logits, R.rec = "e2e", True, {}
        itf.trainable(train_B=True, train_g=True)
        with torch.enable_grad():
            loss = reader(input_ids=x).logits.pow(2).mean()
            loss.backward()
        assert itf.A[str(itf.layers[0])].weight.grad is not None and itf.A[str(itf.layers[0])].weight.grad.abs().sum() > 0
        assert all(R.rec[li]["logits"].shape[-1] == N + 9 for li in itf.layers)
        R.mode = "cap"
        with torch.no_grad():
            reader(input_ids=x)
        assert all(R.ocap[li].shape == (9, reader_dims(reader)["d"]) for li in itf.layers)
        R.mode = "off"
        R.detach()


def test_d26reader_equals_d13reader_on_qwen():
    """Same model, same interface: the generic patch reproduces D13Reader exactly (student mode)."""
    reader = tiny_qwen35(seed=1)
    itf, op, N = _setup(reader)
    x = torch.tensor([_ids(8, 7)])
    out = {}
    for name, cls in (("d13", D13Reader), ("d26", D26Reader)):
        R = cls(reader, itf)
        R.load(op, N)
        R.mode = "student"
        with torch.no_grad():
            out[name] = reader(input_ids=x).logits
        R.mode = "off"
        R.detach()
    assert torch.allclose(out["d13"], out["d26"], atol=1e-6)


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
            print("ok", k)
    print("D26 tests passed")
