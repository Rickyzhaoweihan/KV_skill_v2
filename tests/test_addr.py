"""ADDR tests: SEG structure-aware keys / values (xenc/addr_seg.py), HID text-attention-output distillation
(xenc/addr_hid.py), and the pure helpers / hook logic of scripts/addr_diag.py.

CPU only, tiny random Qwen3_5ForCausalLM models (tests/_tiny.tiny_qwen35, test_state_interface._tiny_encoder)."""

from __future__ import annotations

import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "scripts"))

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from _tiny import tiny_qwen35  # noqa: E402
from xenc.addr_seg import SegInterface, parse_seg, segment_ids, upgrade_seg  # noqa: E402
from xenc.d13_read import D13Interface, D13Reader, EncTeacher, load_d13, save_d13  # noqa: E402

torch.set_num_threads(2)
LMAP = {1: 2, 5: 5}


def _ids(n, seed):
    return torch.randint(3, 100, (n,), generator=torch.Generator().manual_seed(seed)).tolist()


def _setup(parts="kP,kS,vP,vS"):
    from test_state_interface import _tiny_encoder
    torch.manual_seed(0)
    r, e = tiny_qwen35(), _tiny_encoder()
    B = {li: torch.randn(64, 48) * 0.1 for li in LMAP}
    base = D13Interface(e, LMAP, 64, 48, B=B, mlp_hidden=8)
    with torch.no_grad():
        for li in ("1", "5"):
            base.A[li].weight.add_(0.3 * torch.randn_like(base.A[li].weight))
        base.g.fill_(0.9)
    itf = upgrade_seg(base, e, parse_seg(1, 4, parts))
    return r, e, base, itf


def _seg(n):
    """synthetic segment ids for n columns: 2 template columns, then paragraphs of 5 columns; a level-1 header line at
    columns 2-3 and a level-2 header line at the first 2 columns of every later 9-column block."""
    par = torch.tensor([-1, -1] + [(c - 2) // 5 for c in range(2, n)])
    blk = [-1, -1] + [(c - 2) // 9 for c in range(2, n)]
    head_id = torch.tensor([b if c >= 2 and ((c - 2) % 9) < 2 else -1 for c, b in enumerate(blk)])
    nb = max(blk) + 1
    sec = torch.tensor(blk)                       # tuple of block 0 = (0,), of block k = (0, k)
    M = torch.zeros(nb, nb)
    M[0, 0] = 1.0
    for k in range(1, nb):
        M[k, 0] = M[k, k] = 0.5
    return {"par": par, "sec": sec, "head_id": head_id, "head": head_id >= 0, "sec_mat": M, "n_par": int(par.max()) + 1,
            "n_sec": nb, "n_hdr": nb, "par_mode": "blank"}


def _operator(e, doc):
    Tc = EncTeacher(e, sorted(set(LMAP.values())))
    Tc.rec, Tc.kv_only = {}, True
    with torch.no_grad():
        e(input_ids=torch.tensor([doc]), use_cache=False)
    op = {k: (v["k"], v["v"]) for k, v in Tc.rec.items()}
    Tc.detach()
    op["seg"] = _seg(len(doc))
    return op


def _logits(r, itf, op, n, rows):
    R = itf.make_reader(r) if hasattr(itf, "make_reader") else D13Reader(r, itf)
    R.load(op, n)
    R.mode = "student"
    with torch.no_grad():
        y = r(input_ids=torch.tensor([rows])).logits
    R.mode = "off"
    R.detach()
    return y


def _perturb(itf, s=0.3):
    with torch.no_grad():
        for p in itf.SEG.values():
            p.normal_(0, s)


# ----------------------------------------------------------------------------- segment ids
def test_segment_ids_paragraphs_sections_heads_and_truncation():
    text = "SYS <doc>\n## Intro\nAlpha beta.\n\nGamma delta.\n\n## Methods\nEpsilon zeta eta.\n\nTheta.</doc>Q"
    d0 = text.index("## Intro")
    d1 = text.index("</doc>")
    # word-level fake tokens with char offsets (whitespace kept inside the following token)
    import re
    offs = [(m.start(), m.end()) for m in re.finditer(r"\s*\S+", text)]
    n = len(offs)
    s = segment_ids(text, offs, d0, d1, n)
    words = [text[a:b].strip() for a, b in offs]
    par, sec, head = s["par"].tolist(), s["sec"].tolist(), s["head"].tolist()
    i_alpha, i_gamma, i_eps, i_theta = (next(i for i, w in enumerate(words) if w.startswith(x))
                                        for x in ("Alpha", "Gamma", "Epsilon", "Theta"))
    assert par[0] == -1 and sec[0] == -1                              # template column outside the document
    assert par[i_alpha] != par[i_gamma] and par[i_gamma] != par[i_eps] and par[i_eps] != par[i_theta]
    assert sec[i_alpha] == sec[i_gamma] == 0 and sec[i_eps] == sec[i_theta] == 1
    hw = [w for w, h in zip(words, head) if h]
    assert hw == ["##", "Intro", "##", "Methods"], hw
    assert s["par_mode"] == "blank" and s["n_sec"] == 2
    # truncated (left) operator: base > 0 -> ids renumbered densely, a section without its header line is dropped
    base = i_gamma
    s2 = segment_ids(text, offs, d0, d1, n - base, base)
    assert s2["par"].tolist()[0] == 0 and s2["sec"].tolist()[0] == -1 and s2["n_sec"] == 1
    # no blank lines -> fixed blocks
    t3 = "x " * 50
    o3 = [(m.start(), m.end()) for m in re.finditer(r"\S+", t3)]
    s3 = segment_ids(t3, o3, 0, len(t3), len(o3), 0, block=8)
    assert s3["par_mode"] == "block" and s3["par"].tolist()[:9] == [0] * 8 + [1] and s3["n_sec"] == 0


def test_segment_ids_hierarchical_headers_and_long_paragraph_cap():
    import re
    text = "ACT I\nScene I. Hall\nA line here.\n\nB line.\n\nScene II. Road\nC line.\n\nACT II\nScene I. Hall\nD line."
    offs = [(m.start(), m.end()) for m in re.finditer(r"\s*\S+", text)]
    s = segment_ids(text, offs, 0, len(text), len(offs))
    words = [text[a:b].strip() for a, b in offs]
    col = {w: i for i, w in reversed(list(enumerate(words)))}
    hdr_cols = {}
    for c, h in enumerate(s["head_id"].tolist()):
        if h >= 0:
            hdr_cols.setdefault(h, []).append(words[c])
    names = {h: " ".join(v) for h, v in hdr_cols.items()}
    assert sorted(names.values()) == sorted(["ACT I", "Scene I. Hall", "Scene II. Road", "ACT II", "Scene I. Hall"])
    def tup(w):
        r = s["sec_mat"][s["sec"][col[w]]]
        return sorted(names[h] for h in (r > 0).nonzero().flatten().tolist())
    assert tup("A") == ["ACT I", "Scene I. Hall"] and tup("B") == ["ACT I", "Scene I. Hall"]
    assert tup("C") == ["ACT I", "Scene II. Road"] and tup("D") == ["ACT II", "Scene I. Hall"]
    assert s["sec"][col["A"]] != s["sec"][col["D"]]                  # same scene name, different act
    # a long blank-line paragraph is cut into pieces of 2*block tokens
    t2 = "w " * 30 + "\n\n" + "v " * 3
    o2 = [(m.start(), m.end()) for m in re.finditer(r"\S+", t2)]
    s2 = segment_ids(t2, o2, 0, len(t2), len(o2), 0, block=4)
    assert s2["par"].tolist() == [i // 8 for i in range(30)] + [4] * 3


def test_hdr_regex_covers_benchmark_formats():
    from xenc.addr_seg import HDR
    for line in ("## Introduction", "[P3] Cork (city)", "Passage 2: The Real Glory", "ACT III", "SCENE II. A room",
                 "Scene IV. The same", "# Title"):
        assert HDR.match(line), line
    for line in ("Gamma delta.", "The act of", "ANTIPHOLUS.", "#hashtag"):
        assert not HDR.match(line), line


# ----------------------------------------------------------------------------- SEG read
def test_zero_init_equals_base_exactly_and_switch():
    doc, rows = _ids(25, 0), _ids(6, 1)
    r, e, base, itf = _setup()
    op = _operator(e, doc)
    a = _logits(r, base, op, len(doc), rows)
    b = _logits(r, itf, op, len(doc), rows)
    assert torch.equal(a, b), float((a - b).abs().max())
    _perturb(itf)
    c = _logits(r, itf, op, len(doc), rows)
    assert float((c - a).abs().max()) > 1e-4                             # live once W != 0
    itf.seg_on = False
    assert torch.equal(_logits(r, itf, op, len(doc), rows), a)           # ablation switch == base
    itf.seg_on = True
    for parts in ("kP", "kS", "vP", "vS"):                                # every term alone is live
        r, e, base, itf = _setup(parts)
        op = _operator(e, doc)
        _perturb(itf)
        assert float((_logits(r, itf, op, len(doc), rows) - a).abs().max()) > 1e-5, parts


def test_seg_kv_matches_explicit_reference():
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    doc = _ids(23, 0)
    r, e, base, itf = _setup()
    _perturb(itf)
    op = _operator(e, doc)
    itf._seg_op, itf._seg_cache = op, {}
    sg = op["seg"]
    N = len(doc)
    for li in (1, 5):
        K, V = op[LMAP[li]]
        K, V = K.float(), V.float()
        pos = torch.arange(N).view(1, 1, N).expand(3, 1, N)
        cos, sin = itf.rot(K[:1].transpose(0, 1).reshape(1, N, -1), pos)
        kpre = apply_rotary_pos_emb(K[None], K[None], cos, -sin)[1][0]
        assert torch.allclose(apply_rotary_pos_emb(kpre[None], kpre[None], cos, sin)[1][0], K, atol=1e-5)
        dk, dv = torch.zeros_like(K), torch.zeros_like(V)
        for j in range(N):
            p, s, = int(sg["par"][j]), int(sg["sec"][j])
            if p >= 0:
                cols = (sg["par"] == p).nonzero().flatten()
                dk[:, j] += torch.einsum("hd,hde->he", kpre[:, cols].mean(1), itf.SEG[f"kP{li}"])
                dv[:, j] += torch.einsum("hd,hde->he", V[:, cols].mean(1), itf.SEG[f"vP{li}"])
            if s >= 0:
                ks = sum(float(sg["sec_mat"][s, h]) * kpre[:, (sg["head_id"] == h).nonzero().flatten()].mean(1)
                         for h in range(sg["n_hdr"]) if float(sg["sec_mat"][s, h]) > 0)
                vs = sum(float(sg["sec_mat"][s, h]) * V[:, (sg["head_id"] == h).nonzero().flatten()].mean(1)
                         for h in range(sg["n_hdr"]) if float(sg["sec_mat"][s, h]) > 0)
                dk[:, j] += torch.einsum("hd,hde->he", ks, itf.SEG[f"kS{li}"])
                dv[:, j] += torch.einsum("hd,hde->he", vs, itf.SEG[f"vS{li}"])
        Kref = K + apply_rotary_pos_emb(dk[None], dk[None], cos, sin)[1][0]
        Kg, Vg = itf.seg_kv(li, K, V)
        assert torch.allclose(Kg, Kref, atol=1e-5) and torch.allclose(Vg, V + dv, atol=1e-5), li
        with torch.no_grad():                                             # cached path == recomputed path
            Kc, Vc = itf.seg_kv(li, K, V)
            Kc2, Vc2 = itf.seg_kv(li, K, V)
        assert torch.allclose(Kc, Kref, atol=1e-5) and Kc2 is Kc


def test_gradients_reach_seg_and_A_plain_and_checkpointed():
    doc, rows = _ids(25, 0), _ids(6, 1)
    r, e, base, itf = _setup()
    op = _operator(e, doc)
    R = itf.make_reader(r)
    R.load(op, len(doc))
    for ck in (False, True):
        R.ckpt = ck
        for p in itf.parameters():
            p.grad = None
        itf.trainable(train_B=True, train_g=True)
        for p in itf.seg_params():
            p.requires_grad_(True)
        R.mode, R.need_logits, R.rec = "e2e", True, {}
        with torch.enable_grad():
            out = r(input_ids=torch.tensor([rows]))
            out.logits.float().logsumexp(-1).mean().backward()
        for k, p in itf.SEG.items():                                     # zero-init maps: non-zero grad at step 0
            assert p.grad is not None and float(p.grad.abs().sum()) > 0, (k, ck)
        assert float(itf.A["5"].weight.grad.abs().sum()) > 0, ck
        R.mode = "off"
    R.detach()


def test_incremental_decode_equals_full_pass():
    doc, rows = _ids(22, 0), _ids(7, 1)
    r, e, base, itf = _setup()
    _perturb(itf)
    op = _operator(e, doc)
    full = _logits(r, itf, op, len(doc), rows)
    R = itf.make_reader(r)
    R.load(op, len(doc))
    R.mode = "student"
    with torch.no_grad():
        out = r(input_ids=torch.tensor([rows[:3]]), use_cache=True)
        steps = [out.logits[0]]
        for t in rows[3:]:
            out = r(input_ids=torch.tensor([[t]]), past_key_values=out.past_key_values, use_cache=True)
            steps.append(out.logits[0])
    R.mode = "off"
    R.detach()
    inc = torch.cat(steps, 0)
    assert torch.allclose(inc, full[0], atol=2e-4), float((inc - full[0]).abs().max())


def test_save_load_round_trip_and_missing_seg_rejected():
    doc, rows = _ids(25, 0), _ids(6, 1)
    r, e, base, itf = _setup()
    _perturb(itf)
    op = _operator(e, doc)
    a = _logits(r, itf, op, len(doc), rows)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.pt")
        save_d13(itf, p)
        B = {li: itf.B[str(li)].detach().clone() for li in LMAP}
        it2, _ = load_d13(p, e, B)
    assert isinstance(it2, SegInterface) and it2.seg == itf.seg
    assert torch.equal(_logits(r, it2, op, len(doc), rows), a)
    op2 = {k: v for k, v in op.items() if k != "seg"}
    R = it2.make_reader(r)
    try:
        R.load(op2, len(doc))
        raise AssertionError("operator without segment ids accepted")
    except AssertionError as ex:
        assert "segment ids" in str(ex)
    R.detach()


def test_docadapter_passes_seg_through():
    from xenc.d20_reg import DocAdapter
    doc, rows = _ids(25, 0), _ids(6, 1)
    r, e, base, itf = _setup()
    _perturb(itf)
    op = _operator(e, doc)
    a = _logits(r, itf, op, len(doc), rows)
    ad = DocAdapter(itf, ("dA", "gate", "kv"), rank=4, kv_rank=2)
    op2 = ad.transform_op(op)
    assert op2["seg"] is op["seg"]
    assert torch.allclose(_logits(r, itf, op2, len(doc), rows), a, atol=1e-5)   # zero-init adapter == base


# ----------------------------------------------------------------------------- HID
def test_hid_capture_and_gradient():
    from xenc.addr_hid import AttnOutCap, hid_loss, hid_rows
    doc, rows = _ids(25, 0), _ids(6, 1)
    r, e, base, _ = _setup()
    itf = base
    op = _operator(e, doc)
    R = D13Reader(r, itf)
    cap = AttnOutCap(r, [1, 5])
    # teacher: plain reader over doc + rows; capture the last T rows
    T = len(rows)
    cap.T, cap.detach, cap.rec = T, True, {}
    with torch.no_grad():
        r(input_ids=torch.tensor([doc + rows]))
    yt = dict(cap.rec)
    assert set(yt) == {1, 5} and yt[1].shape == (T, 64) and not yt[1].requires_grad
    # student: e2e read; captured rows carry the graph; loss gradient reaches A
    itf.trainable(train_B=True, train_g=True)
    R.load(op, len(doc))
    R.mode, R.rec = "e2e", {}
    cap.T, cap.detach, cap.rec = T, False, {}
    with torch.enable_grad():
        r(input_ids=torch.tensor([rows]))
        cap.T = None
        ys = dict(cap.rec)
        assert ys[5].requires_grad
        L = hid_loss(ys, yt, hid_rows(T, 2, 3, "all"))
        L.backward()
    assert float(L) > 0 and float(itf.A["5"].weight.grad.abs().sum()) > 0
    assert float(hid_loss(yt, yt, hid_rows(T, 2, 3, "qa"))) == 0.0
    from xenc.addr_hid import common_suffix
    assert common_suffix([1, 2, 3, 4], [9, 3, 4]) == 2 and common_suffix([1, 2], [1, 2]) == 2 and common_suffix([1], [2]) == 0
    assert hid_rows(10, 4, 3, "qa") == list(range(4, 10)) and hid_rows(10, 4, 3, "ans") == list(range(6, 10))
    R.mode = "off"
    R.detach()
    cap.remove()


# ----------------------------------------------------------------------------- addr_diag helpers
def test_addr_diag_tf_metrics_and_groups():
    from addr_diag import group_layers, tf_metrics
    V = 7
    tt = [1, 2, 3, 4]
    lg = torch.full((5, V), -5.0)
    for i, t in enumerate(tt):
        lg[i, t] = 5.0
    lg[1, 2], lg[1, 6] = 0.0, 6.0                                      # row 1 wrong
    m = tf_metrics(lg, tt, [1, 2])
    assert m["top1_first"] == 0.0 and m["tf_em"] == 0.0 and m["acc"] == 0.5
    m2 = tf_metrics(lg, tt, [2, 3])
    assert m2["top1_first"] == 1.0 and m2["tf_em"] == 1.0
    full, lin = [3, 7, 11, 15, 19, 23, 27, 31], [i for i in range(32) if i % 4 != 3]
    assert group_layers("R", full, lin) == {3, 7, 11, 15, 19} and group_layers("D", full, lin) == {23, 27, 31}
    assert group_layers("all", full, lin) == set(range(32))


def test_addr_diag_mixers_full_substitution_reproduces_full_context():
    """kv+all sanity in miniature: substituting EVERY token mixer's rows (recorded from a forward over context + rows)
    into a forward over the rows alone reproduces the context forward's logits on those rows."""
    from addr_diag import Mixers
    r = tiny_qwen35()
    ctx, rows = _ids(20, 0), _ids(6, 1)
    M = Mixers(r)
    T = len(rows)
    M.set(T, rec=True)
    with torch.no_grad():
        full = r(input_ids=torch.tensor([ctx + rows])).logits[0, -T:]
    rec = dict(M.rec)
    assert set(M.res) == set(range(8))
    M.set(T, sub={i: rec[i] for i in rec})
    with torch.no_grad():
        alone = r(input_ids=torch.tensor([rows])).logits[0]
    M.set(None)
    with torch.no_grad():
        plain = r(input_ids=torch.tensor([rows])).logits[0]
    assert torch.allclose(alone, full, atol=1e-4), float((alone - full).abs().max())
    assert float((plain - full).abs().max()) > 1e-3                   # the context matters in this model
    # fn path: identity replacement leaves the forward unchanged
    M.set(T, fn={1: lambda hs, o: o})
    with torch.no_grad():
        same = r(input_ids=torch.tensor([rows])).logits[0]
    M.set(None)
    assert torch.allclose(same, plain, atol=1e-6)


if __name__ == "__main__":
    import inspect
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_") and inspect.isfunction(f)]
    for f in fns:
        f()
        print(f"ok {f.__name__}")
    print(f"{len(fns)} passed")
