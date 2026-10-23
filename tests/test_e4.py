"""E4 tests (CPU): the read-variant patch of xenc.d13_read._attend, paragraph masses, and the text-attention
reconstruction used by scripts/e4_hopdiag.py."""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))
sys.path.insert(0, _HERE)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
import e4_hopdiag as E  # noqa: E402
from _tiny import tiny_qwen35  # noqa: E402

torch.set_num_threads(2)


def _qkv(seed=0, Hq=4, Hkv=2, T=5, N=30, P=3, hd=8):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    return r(Hq, T, hd), r(Hkv, N, hd), r(Hkv, N, hd), r(Hkv, P + T, hd), r(Hkv, P + T, hd)


def test_inactive_variant_is_the_original_read():
    q, Ko, Vo, Kw, Vw = _qkv()
    E.Variant.reset()
    a = E._ORIG_ATTEND(q, Ko, Vo, Kw, Vw, 0.3, True)
    b = E.attend_variant(q, Ko, Vo, Kw, Vw, 0.3, True)
    assert torch.equal(a["o"], b["o"])
    # the module-level patch is what D13Interface.read calls
    import xenc.d13_read as D
    assert D._attend is E.attend_variant
    # the active code path with no-op settings reproduces the original numerically
    E.Variant.topk = 10 ** 6
    c = E.attend_variant(q, Ko, Vo, Kw, Vw, 0.3, True)
    E.Variant.reset()
    E.Variant.colmask = torch.ones(Ko.shape[1], dtype=torch.bool)
    d = E.attend_variant(q, Ko, Vo, Kw, Vw, 0.3, True)
    E.Variant.reset()
    assert torch.allclose(a["o"], c["o"], atol=1e-6) and torch.allclose(a["o"], d["o"], atol=1e-6)


def test_topk_mask_and_temperature_act_on_operator_columns_only():
    q, Ko, Vo, Kw, Vw = _qkv(1)
    N = Ko.shape[1]
    E.Variant.reset()
    E.Variant.topk = 4
    s = E.attend_variant(q, Ko, Vo, Kw, Vw, 0.3, True)["logits"]
    fin = torch.isfinite(s[..., :N]).sum(-1)
    assert int(fin.max()) == 4 and int(fin.min()) == 4
    own = s[..., N:]
    assert torch.isfinite(own[..., 0]).all()                       # own earlier rows untouched
    E.Variant.reset()
    m = torch.zeros(N, dtype=torch.bool)
    m[5:9] = True
    E.Variant.colmask = m
    p = torch.softmax(E.attend_variant(q, Ko, Vo, Kw, Vw, 0.3, True)["logits"], -1)
    assert float(p[..., :5].sum()) == 0 and float(p[..., 9:N].sum()) == 0
    E.Variant.reset()
    E.Variant.temp = 2.0
    s2 = E.attend_variant(q, Ko, Vo, Kw, Vw, 0.3, True)["logits"]
    s1 = E._ORIG_ATTEND(q, Ko, Vo, Kw, Vw, 0.3, True)["logits"]
    E.Variant.reset()
    f = torch.isfinite(s1)
    assert torch.allclose(s2[f], 2 * s1[f], atol=1e-5)


def test_parse_variant_masks():
    ranges = {"hops": [(10, 20), (30, 35)], "header_end": 4}
    assert E.parse_variant("oracle_sup", ranges, 50)
    m = E.Variant.colmask
    assert int(m.sum()) == 4 + 10 + 5 and m[:4].all() and m[10:20].all() and not m[20:30].any()
    assert E.parse_variant("oracle_hop1", ranges, 50) and int(E.Variant.colmask.sum()) == 14
    assert E.parse_variant("mask_sup", ranges, 50) and int(E.Variant.colmask.sum()) == 50 - 15
    assert not E.parse_variant("mask_sup", None, 50)
    assert E.parse_variant("top64", None, 50) and E.Variant.topk == 64
    assert E.parse_variant("temp1.5", None, 50) and E.Variant.temp == 1.5
    E.Variant.reset()


def test_para_mass():
    p = torch.zeros(2, 3, 10)
    p[0, :, 2:4] = 0.5
    p[1, :, 7] = 1.0
    m = E.para_mass(p, [(0, 4), (4, 8), (8, 10)])
    assert torch.allclose(m, torch.tensor([[1.0, 0, 0], [0, 1.0, 0]]))


def test_text_attn_probs_reproduce_the_attention_module():
    """softmax over EncTeacher's recorded q/k (reader's own full attention) · v, gated, o_proj == module output."""
    from deltaskill.backbone_utils import decoder_layers
    from xenc.d13_read import EncTeacher
    from xenc.state_skill import full_layers
    m = tiny_qwen35()
    L = decoder_layers(m)
    fl = full_layers(m)
    Tc = EncTeacher(m, fl)
    Tc.rec, Tc.kv_only = {}, False
    ids = torch.randint(3, 100, (1, 17), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        m(input_ids=ids, use_cache=False)
    for li in fl:
        r = Tc.rec[li]
        att = L[li].self_attn
        rows = [5, 11, 16]
        p = E.text_attn_probs(r["q"], r["k"], rows, att.scaling)            # [Hq, R, T]
        assert torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-5)
        Hq, Hkv, hd = p.shape[0], r["k"].shape[0], att.head_dim
        v = r["v"].repeat_interleave(Hq // Hkv, 0)                           # [Hq, T, hd]
        o = torch.einsum("hrt,htd->hrd", p, v).transpose(0, 1).reshape(len(rows), Hq * hd)
        x = r["x"][rows]
        _, gate = torch.chunk(att.q_proj(x).view(len(rows), -1, hd * 2), 2, dim=-1)
        y = att.o_proj(o * torch.sigmoid(gate.reshape(len(rows), -1)))
        assert torch.allclose(y, r["out"][rows], atol=1e-4), float((y - r["out"][rows]).abs().max())
    Tc.rec = None
    Tc.detach()


def test_para_ranges_with_the_real_tokenizer():
    from transformers import AutoTokenizer
    from register_doc import SYSTEM_ANSWER
    from xenc.state_skill import doc_prefix_text
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    except Exception:                                  # offline without the tokenizer cached: skip
        return
    doc = "[P0] Alpha\nThe first paragraph text.\n[P1] Beta\nSecond one, with the answer Orion Pictures.\n[P2] Gamma\nThird."
    spans, k = [], 0
    for part in doc.split("\n["):
        s = doc.index(part if k == 0 else "[" + part)
        spans.append([s, s + len(part if k == 0 else "[" + part)])
        k += 1
    pt = doc_prefix_text(tok, "Qwen/Qwen3.5-9B", doc, SYSTEM_ANSWER)
    ids = [int(x) for x in tok(pt, add_special_tokens=False)["input_ids"]]
    r = E.para_ranges(tok, pt, ids, doc, spans, len(ids))
    for (a, b), (ca, cb) in zip(r["paras"], spans):
        txt = tok.decode(ids[a:b])
        assert doc[ca:cb].strip()[:8] in txt and b > a, (txt, doc[ca:cb])
    assert "Orion" in tok.decode(ids[r["paras"][1][0]:r["paras"][1][1]])
    assert "Orion" not in tok.decode(ids[r["paras"][0][0]:r["paras"][0][1]])
    assert r["header_end"] <= r["paras"][0][0]


def test_layer_filter_raw_logits_and_grads_through_the_real_read():
    """Variant restricted to one reader layer changes only that layer's read; raw_logits returns the unscaled scores;
    gradients reach A under the temperature; reset restores the exact D13 read."""
    from test_d13 import _teacher
    from xenc import e4_read
    from xenc.d13_read import D13Interface
    from xenc.state_skill import full_layers
    e4_read.install()
    m = tiny_qwen35()
    doc, rows = torch.randint(3, 100, (20,)).tolist(), torch.randint(3, 100, (6,)).tolist()
    op, rec, _ = _teacher(m, doc, rows)
    l1, l2 = full_layers(m)[1], full_layers(m)[2]
    itf = D13Interface(m, {l1: l1, l2: l2}, 64, 64, B=None)
    x1, x2 = rec[l1]["x"].float(), rec[l2]["x"].float()
    e4_read.Variant.reset()
    b1 = itf.read(l1, x1, *op[l1], len(doc), need_logits=True)
    b2 = itf.read(l2, x2, *op[l2], len(doc), need_logits=True)
    e4_read.set_from(f"temp2@{l2}")
    v1 = itf.read(l1, x1, *op[l1], len(doc), need_logits=True)
    v2 = itf.read(l2, x2, *op[l2], len(doc), need_logits=True)
    assert torch.equal(v1["o"], b1["o"])                                  # other layer untouched
    assert not torch.allclose(v2["o"], b2["o"], atol=1e-4)                # this layer sharpened
    f = torch.isfinite(b2["logits"])
    assert torch.allclose(v2["logits"][f], 2 * b2["logits"][f], atol=1e-4)
    e4_read.set_from(f"temp2@{l2}", raw_logits=True)
    r2 = itf.read(l2, x2, *op[l2], len(doc), need_logits=True)
    assert torch.allclose(r2["o"], v2["o"]) and torch.allclose(r2["logits"][f], b2["logits"][f], atol=1e-5)
    with torch.enable_grad():
        y = itf.read(l2, x2, *op[l2], len(doc), need_logits=True)["o"]
        y.pow(2).sum().backward()
    assert float(itf.A[str(l2)].weight.grad.abs().sum()) > 0
    e4_read.Variant.reset()
    assert torch.equal(itf.read(l2, x2, *op[l2], len(doc))["o"], b2["o"])
    assert e4_read.Variant.cur_layer is None


def test_parse_names():
    from xenc import e4_read
    assert e4_read.parse("base") == {"temp": 1.0, "topk": 0, "layers": None}
    assert e4_read.parse("temp1.5@3+19") == {"temp": 1.5, "topk": 0, "layers": {3, 19}}
    assert e4_read.parse("top64") == {"temp": 1.0, "topk": 64, "layers": None}


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
    print("test_e4: all passed")
