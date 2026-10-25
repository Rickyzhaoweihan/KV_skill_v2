"""SOFTCONF tests (agent SOFTCONF, 2026-10-25), CPU only: graded-conflict set construction (scripts/softconf_build.py),
the prior probe helpers (scripts/softconf_prior.py), the table logic (scripts/softconf_table.py) and the adaptive
CAD / read-gain knobs (xenc/softcad.py, scripts/softconf_eval.py)."""

from __future__ import annotations

import json
import math
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import softconf_build as B  # noqa: E402
import softconf_table as T  # noqa: E402
from softconf_prior import rotate, summarize  # noqa: E402


# ---------------------------------------------------------------------------- numbers
def test_numeric_align_render_roundtrip():
    orig = "Initial: 30 mg once daily; maximum: 30 mg/day. Children: 3 mg. Record 1:40.91 and 1,500 m."
    mod = "Initial: 3.0 mg once daily; maximum: 3.0 mg/day. Children: 3 mg. Record 1:40.91 and 1,500 m."
    idx = B.align_numeric(orig, mod, "30", "3")
    assert idx == [0, 1]
    assert B.render_numeric(orig, idx, 30.0, B.num_style("30")) == orig
    assert B.render_numeric(orig, idx, 36.0, B.num_style("30")) == orig.replace("30 mg", "36 mg")
    assert B.align_numeric(orig, mod.replace("Children", "Kids"), "30", "3") is None      # text changed elsewhere
    assert B.align_numeric(orig, orig.replace("3 mg.", "4 mg."), "30", "3") is None        # wrong token changed
    assert B.fmt_num(100.91, B.num_style("1:40.91")) == "1:40.91"
    assert B.fmt_num(7592.0, B.num_style("2:06:32")) == "2:06:32"
    assert B.fmt_num(1650.0, B.num_style("1,500")) == "1,650" and B.num_value("1:40.91") == 100.91


def test_numeric_ladder_levels():
    vals, txt = B.numeric_ladder(400.0, B.num_style("400"), "drugs", "down")
    assert txt == {"L0": "400", "L1": "360", "L2": "320", "L3": "200", "L4": "40"}
    vals, txt = B.numeric_ladder(2006.0, B.num_style("2006"), "years", "up")
    assert txt == {"L0": "2006", "L1": "2007", "L2": "2009", "L3": "2017", "L4": "2106"}
    assert B.numeric_ladder(3.0, B.num_style("3"), "news", "up") is None           # +10 % impossible on "3"
    vals, txt = B.numeric_ladder(9.63, B.num_style("9.63"), "records", "up")
    assert len(set(txt.values())) == 5 and txt["L1"] == "10.59"


def test_entity_alignment():
    o = "Born in Corby. Corby is a town. See Corby."
    assert B.align_entity(o, o.replace("Corby", "Kettering", 1), "Corby", "Kettering")
    assert B.align_entity(o, o.replace("Corby", "Kettering"), "Corby", "Kettering")
    assert not B.align_entity(o, o.replace("town", "city"), "Corby", "Kettering")


# ---------------------------------------------------------------------------- hedge
def test_hedge_extraction_and_ladder():
    t = ("Thiazide diuretics may cause hypokalemia and hyponatremia. Our findings do not support the hypothesis that "
         "ARBs increase the risk of stroke. This drug may cause drowsiness in children. Taper once leukocytosis "
         "improves markedly. Aspirin causes gastric bleeding in older adults.")
    cs = B.hedge_candidates(t)
    stm = {B.hedge_statement(c, c["o"]) for c in cs}
    assert stm == {"Thiazide diuretics may cause hypokalemia and hyponatremia.",
                   "Aspirin causes gastric bleeding in older adults."}, stm
    c = [c for c in cs if c["subj"] == "Aspirin"][0]
    assert c["o"] == 3 and B.render_hedge(t, c, 3) == t
    assert "Aspirin does not cause gastric" in B.render_hedge(t, c, 0)
    assert "Aspirin is likely to cause gastric" in B.render_hedge(t, c, 2)
    p = [c for c in cs if c["plural"]][0]
    assert B.hedge_statement(p, 2) == "Thiazide diuretics are likely to cause hypokalemia and hyponatremia."
    assert B.hedge_targets(1) == {"L0": 1, "L1": 2, "L3": 3, "L4": 0}
    assert B.hedge_targets(3) == {"L0": 3, "L1": 2, "L3": 1, "L4": 0}
    mono = "Oxybutynin: Drug information. Warnings: May cause heat prostration in hot weather."
    ci = [c for c in B.hedge_candidates(mono, "Oxybutynin") if c["implicit"]][0]
    assert B.render_hedge(mono, ci, 3).endswith("Causes heat prostration in hot weather.")
    assert B.hedge_statement(ci, 0) == "Oxybutynin does not cause heat prostration in hot weather."


# ---------------------------------------------------------------------------- persp
def test_pilot_seeds_valid_and_facts_fixed():
    seeds = json.load(open(B.SEEDS_PILOT))
    assert all(B.validate_seed(s) == [] for s in seeds)
    for s in seeds:
        facts = s["A"]["facts"] + s["B"]["facts"] + s["shared_facts"]
        for L, side in B.persp_variants():
            p = B.persp_passage(s, L, side)
            assert all(f in p for f in facts), (s["id"], L, side)        # facts verbatim at every framing level
    bad = dict(seeds[0])
    bad["A"] = {**bad["A"], "facts": bad["A"]["facts"][:1] + ["The immune view is primarily supported."]}
    assert any("framing" in e for e in B.validate_seed(bad))


# ---------------------------------------------------------------------------- end-to-end build (synthetic ClashEval)
def _fake_byq():
    byq = {}
    for i in range(12):
        v = 100 + 10 * i
        ctx = f"Drug{i}: Drug information. Dosing: Adult: Oral: {v} mg once daily. Max dose {v} mg/day. Note text {i}."
        q = f"What is the daily dose of Drug{i}?"
        rows = [{"dataset": "drugs", "question": q, "context_original": ctx, "context_mod": ctx,
                 "answer_original": str(v), "answer_mod": str(v), "mod_type": "0"}]
        for f in (0.5, 2.0):
            m = f"{v * f:.1f}"
            rows.append({"dataset": "drugs", "question": q, "context_original": ctx,
                         "context_mod": ctx.replace(f" {v} mg", f" {m} mg"), "answer_original": str(v),
                         "answer_mod": m, "mod_type": str(f)})
        byq[("drugs", q)] = rows
    return byq


def test_build_axis_num_end_to_end():
    qs = B.ce_num_questions(_fake_byq(), "drugs")
    assert len(qs) == 12 and all(q["idx"] == [0, 1] or len(q["idx"]) == 2 for q in qs)
    pool = [("Filler passage about an unrelated village. " * 60) + str(k) for k in range(10)]
    a = types.SimpleNamespace(val_frac=0.25, seed=1, pilot=0)
    docs, splits, wmap = B.build_axis("num", {"drugs": qs}, pool, a)
    ver = B.verify("num", docs, splits, wmap, None)
    assert {k: v for k, v in ver.items() if k.startswith("bad_")} == {k: 0 for k in ver if k.startswith("bad_")}, ver
    its = splits["test"] + splits["val"]
    assert len(its) == 12 * 5 and len(docs) == len({i["doc_id"] for i in its})
    for it in its:
        m = it["meta"]
        assert m["roles"][it["correct_label"]] == m["level"] and m["roles"][m["orig_label"]] == "L0"
        assert it["answers"][0] in docs[it["doc_id"]]["text"]
        w = wmap[it["doc_id"]]
        assert w != it["doc_id"] and w.endswith("_" + m["level"])
    l0 = [d for d in docs if d.endswith("_L0")][0]
    assert docs[l0]["text"].startswith("Passage 1:\nDrug")
    from xenc.doc_tasks import validate_item
    for it in its:
        validate_item(it)


# ---------------------------------------------------------------------------- table logic
def test_classify_over_under():
    meta = {"axis": "num", "roles": {"A": "L2", "B": "L0", "C": "L4", "D": "L1", "E": "L3"}, "doc_label": "A",
            "orig_label": "B", "order": ["B", "D", "A", "E", "C"]}
    assert T.classify(meta, "A", "L0")["doc"] == 1.0
    c = T.classify(meta, "B", "L0")
    assert c["prior"] == 1 and c["under"] == 1 and c["over"] == 0
    assert T.classify(meta, "C", "L0")["over"] == 1
    assert T.classify(meta, "D", "L0")["under"] == 1
    h = {"axis": "hedge", "roles": {"A": "m3", "B": "m2", "C": "m1", "D": "m0"}, "doc_label": "B", "orig_label": "C",
         "orig_marker_level": 1, "doc_marker_level": 2, "order": ["A", "B", "C", "D"]}
    assert T.classify(h, "A", None)["over"] == 1 and T.classify(h, "C", None)["under"] == 1
    assert T.classify(h, "D", None)["over"] == 0 and T.classify(h, "D", None)["under"] == 0


def test_fc_methods():
    from xenc.cad import cad_pick
    lm, lr = [math.log(x) for x in (.5, .3, .15, .05)], [math.log(x) for x in (.97, .02, .0099, .0001)]
    for a in (0.0, 0.5, 2.0):
        assert T.fc_method(f"cad_{a:g}", lm, lr) == cad_pick(lm, lr, a)
    assert T.fc_method("gate_2_0.99", lm, lr) == cad_pick(lm, lr, 0.0)          # prior .97 < .99: gate closed
    assert T.fc_method("gate_2_0.9", lm, lr) == cad_pick(lm, lr, 2.0)
    assert T.fc_method("cad_8", lm, lr) == 3                                     # strong CAD overshoots to a ~0 option
    assert T.fc_method("plaus_8_0.2", lm, lr) in (1, 2)                          # plausibility cut blocks it
    assert T.jsd2([.5, .5], [.5, .5]) < 1e-9 and abs(T.jsd2([1, 0], [0, 1]) - 1) < 1e-9
    assert T.fc_method("jsd_0", lm, lr) == 0


def test_prior_rotate_summarize():
    it = {"choices": [{"label": "A", "text": "x"}, {"label": "B", "text": "y"}, {"label": "C", "text": "z"}],
          "meta": {"roles": {"A": "L0", "B": "L1", "C": "L2"}}}
    r, l2r = rotate(it, 1)
    assert [c["text"] for c in r["choices"]] == ["y", "z", "x"] and l2r == {"A": "L1", "B": "L2", "C": "L0"}
    s = summarize([{"L0": .8, "L1": .1, "L2": .1}, {"L0": .6, "L1": .3, "L2": .1}])
    assert s["prior_role"] == "L0" and s["stable"] and abs(s["pmax"] - .7) < 1e-9
    assert not summarize([{"L0": .8, "L1": .2}, {"L0": .3, "L1": .7}])["stable"]


# ---------------------------------------------------------------------------- adaptive CAD / read gain
def test_alpha_rule():
    import torch
    from xenc.softcad import alpha_rule, jsd2
    lm, lr = torch.tensor([2.0, 0.0, -1.0]), torch.tensor([5.0, 0.0, 0.0])
    assert alpha_rule(lm, lr, 0.7, "const") == 0.7
    j = jsd2(torch.softmax(lm, -1), torch.softmax(lr, -1))
    assert 0 < j < 1 and abs(alpha_rule(lm, lr, 2.0, "jsd") - 2 * j) < 1e-9
    assert alpha_rule(lm, lr, 2.0, "jsd", floor=5.0) == 5.0
    assert alpha_rule(lm, lr, 1.0, "prior", tau=0.9) == 1.0 and alpha_rule(lm, lr, 1.0, "prior", tau=0.999) == 0.0
    assert alpha_rule(lm, lm, 3.0, "jsd") < 1e-6


def test_adaptive_decode_matches_cad_decode():
    import torch
    from test_cad import HUB, _setup, _streams
    from xenc.cad import cad_decode
    from xenc.softcad import adaptive_cad_decode
    rt, op, n, rows = _setup()
    kv, ct = _streams(rt, op, n, rows)
    for a in (0.5, 2.0):
        x = cad_decode(kv, ct, a, 6, rt.tok, HUB, set())
        y = adaptive_cad_decode(kv, ct, a, 6, rt.tok, HUB, set(), mode="const")
        assert x["gen"] == y["gen"] and set(y["alpha_trace"]) == {a}
    plain = cad_decode(kv, None, 0.0, 6, rt.tok, HUB, set())
    off = adaptive_cad_decode(kv, ct, 2.0, 6, rt.tok, HUB, set(), mode="prior", tau=1.01)   # gate never opens
    assert off["gen"] == plain["gen"] and set(off["alpha_trace"]) == {0.0}
    j = adaptive_cad_decode(kv, ct, 2.0, 6, rt.tok, HUB, set(), mode="jsd")
    assert all(0 <= t <= 2.0 for t in j["alpha_trace"])
    with torch.no_grad():
        pass


def test_gain_patch(monkeypatch):
    import torch
    import fit_d12p
    import fit_d13
    import softconf_eval

    class Itf:
        g = torch.nn.Parameter(torch.ones(3))
    monkeypatch.setattr(fit_d13, "build", lambda a, init=None: (None, None, None, Itf, {}))
    monkeypatch.setattr(fit_d12p, "build", lambda a, init=None: (None, None, None, Itf, {}))
    softconf_eval.patch(gain_scale=1.5)
    out = fit_d13.build(types.SimpleNamespace(), init="x")
    assert torch.allclose(out[3].g.detach(), torch.full((3,), 1.5))


def test_softconf_eval_end_to_end_on_pilot(monkeypatch, tmp_path):
    """scripts/softconf_eval.py -> eval_cad on a built SOFTCONF pilot split (tiny models, fake tokenizer; skipped if
    the pilot is absent), with the default SOFTCONF.sbatch condition set and the gain patch, then softconf_table."""
    import fit_d13
    import register_state
    import torch
    import xenc.state_skill as ss
    from test_cad import CallTok, D13Interface, tiny_qwen35
    from test_state_interface import _tiny_encoder
    import softconf_eval
    sd = "/scratch/drjieliu_root/drjieliu/rickyhan/softconf/pilot/softconf_pilot_num_split"
    if not os.path.exists(os.path.join(sd, "test", "items.json")):
        return
    tok = CallTok()

    def fake_build(a, init=None):
        torch.manual_seed(0)
        r, e = tiny_qwen35(), _tiny_encoder()
        lmap = {1: 2, 5: 5}
        itf = D13Interface(e, lmap, 64, 48, B={li: torch.randn(64, 48) * 0.3 for li in lmap}, mlp_hidden=8)
        return r, e, tok, itf, lmap
    import fit_d12p
    import xenc.cad as cad
    monkeypatch.setattr(fit_d13, "build", fake_build)
    monkeypatch.setattr(fit_d12p, "build", fit_d12p.build)          # restored after the test (patch() wraps it)
    monkeypatch.setattr(cad, "cad_decode", cad.cad_decode)          # idem
    monkeypatch.setattr(ss, "doc_prefix_text", lambda tok_, hub, doc, system: "S" + doc[:60])
    monkeypatch.setattr(ss, "question_suffix_ids", lambda tok_, hub, it, system: tok(it.question)["input_ids"])
    monkeypatch.setattr(register_state, "greedy", lambda *a_, **k_: ("<answer>A</answer>", 3))
    out = str(tmp_path / "sc")
    args = ["--gain_scale", "1.25", "--ad_mode", "priorjsd", "--ad_tau", "0.0", "--", "--arch", "d13", "--interface",
            "x.pt", "--split_dir", sd, "--split", "test", "--n", "3", "--device", "cpu", "--budget", "3", "--tail", "8",
            "--wrong_map", os.path.join(sd, "wrongmap.json"), "--plain", "kv,closed_book,wrong_doc", "--cad_refs",
            "closed_book", "--cad_alphas", "0.25,1", "--text", "1", "--text_alphas=", "--fc", "1", "--out", out]
    assert softconf_eval.main(args) == 0
    E = json.load(open(os.path.join(out, "eval_test.json")))
    assert len(E["items"]) == 3
    for r_ in E["items"]:
        assert {"kv", "closed_book", "wrong_doc", "text", "kv_cad_cb_0.25", "kv_cad_cb_1"} == set(r_["gen"])
        assert len(r_["fc"]["lp"]["kv"]) == 5 and "closed_book" in r_["fc"]["lp"]
    assert T.main(["--evals", os.path.join(out, "eval_test.json"), "--split_dir", sd, "--reader", "tiny",
                   "--out", str(tmp_path / "tab")]) == 0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
