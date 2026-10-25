"""RECON CPU tests: verbatim recite / altered-detail verify items (xenc/recon.py) and the default-off register_d13 flags.
Tokenizer-only (real Qwen3.5 tokenizer from the offline HF cache; skipped if unavailable)."""
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from xenc import recon as R  # noqa: E402

DOC = ("The Meridian expedition left Port Halvard on 14 March 1891 with forty-two sailors aboard. "
       "Captain Elsie Varga kept a daily log in which she recorded the weather, the ration counts and every quarrel. "
       "By the third week the ship had reached the Calloway Shoals, where 7 barrels of salted pork were lost overboard. "
       "Lieutenant Osric Penhallow argued that they should turn back, but Varga insisted on sailing north. "
       "In April the crew sighted the cliffs of Thessaly Island and anchored in a sheltered bay for nineteen days. "
       "The botanist Miriam Cole catalogued 312 species of moss, many of them previously unknown to science. "
       "When the expedition finally returned in September 1892, only thirty-one sailors were still alive.\n\n"
       "A second voyage in 1895 was financed by the Royal Geographic Circle of Aberdeen and led by Penhallow himself. "
       "It carried a steam launch, two surgeons, a printing press for a ship newspaper and a crate of homing pigeons. "
       "Storms near the Faroe banks split the mainmast, and the newspaper reported 23 separate repairs in one month. "
       "Cole joined again and this time collected lichens, beetles and the eggs of a rare grey petrel.\n\n"
       "Epilogue: Varga later wrote a memoir titled Salt and Silence, published in 1904 in Edinburgh. It sold "
       "poorly at first, yet a reprint in 1931 made her briefly famous among sailing enthusiasts in Glasgow and Leith.")


def _tok():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    except Exception:
        return None


def _prefix(tok, doc):
    from register_doc import SYSTEM_ANSWER
    from xenc.e3_retrieval import body_start
    from xenc.state_skill import doc_prefix_text
    ids = [int(x) for x in tok(doc_prefix_text(tok, "Qwen/Qwen3.5-9B", doc, SYSTEM_ANSWER),
                               add_special_tokens=False)["input_ids"]]
    return ids, body_start(tok, "Qwen/Qwen3.5-9B", SYSTEM_ANSWER)


def test_ramp_weight():
    assert R.ramp_weight(0, 1.0, 500) == 0.0
    assert R.ramp_weight(5, 0.0, 500) == 0.0
    assert abs(R.ramp_weight(250, 1.0, 500) - 0.5) < 1e-12
    assert R.ramp_weight(500, 0.8, 500) == 0.8 and R.ramp_weight(4000, 0.8, 500) == 0.8
    assert R.ramp_weight(1, 0.7, 0) == 0.7


def test_body_bounds():
    assert R.body_bounds(1000, 1000, 30, 64) == (30, 936)             # untruncated: template excluded, tail excluded
    assert R.body_bounds(900, 1000, 30, 64) == (0, 836)               # left-truncated by > S tokens: all body
    assert R.body_bounds(990, 1000, 30, 64) == (20, 926)


def test_alter_detail_swaps_one_detail_from_the_document():
    rng = random.Random(0)
    stmt = "the ship had reached the Calloway Shoals, where 7 barrels of salted pork were lost"
    for _ in range(50):
        alt, (old, new) = R.alter_detail(stmt, DOC, rng)
        assert alt != stmt and alt not in DOC and old in stmt and old != new
        assert alt == stmt.replace(old, new, 1) or new.isdigit()
        if old.isdigit():                                              # numbers first, drawn from the doc
            assert new in [m.group(0) for m in R._NUM.finditer(DOC)] or len(new) == len(old)
    assert R.alter_detail("tiny bits of text", "tiny bits of text here", rng) is None


def test_recite_and_verify_items_with_the_real_tokenizer():
    tok = _tok()
    if tok is None:
        return
    pre, S = _prefix(tok, DOC)
    lo, hi = R.body_bounds(len(pre), len(pre), S, 64)
    rng = random.Random(1)
    n_rec = n_ver = n_yes = 0
    for i in range(60):
        it = R.make_recite(tok, pre, lo, hi, rng, DOC, f"k{i}")
        if it is not None:
            n_rec += 1
            m = it["meta"]
            assert lo <= m["s"] and m["cont_end"] <= hi                      # spans only in operator body columns
            cue = re.match(r'.*?"(.*)"$', it["question"], re.S).group(1)
            assert DOC.count(cue) == 1                                        # the cue is unique in the document
            k = DOC.index(cue) + len(cue)
            assert DOC[k:].lstrip().startswith(it["answers"][0])             # the gold IS the true continuation
            td = R.teacher_doc(tok, pre, lo, m["s"], m["cont_end"], 256, DOC)
            assert cue in td and it["answers"][0] in td and len(td) < len(DOC)
            assert R.teacher_doc(tok, pre, lo, m["s"], m["cont_end"], 0, DOC) == DOC
        v = R.make_verify(tok, pre, lo, hi, rng, DOC, f"v{i}")
        if v is not None:
            n_ver += 1
            stmt = re.match(r'.*?"(.*)"\n', v["question"], re.S).group(1)
            assert [c["text"] for c in v["choices"]] == ["Yes", "No"]
            if v["meta"]["yes"]:
                n_yes += 1
                assert v["correct_label"] == "A" and stmt in DOC
            else:
                assert v["correct_label"] == "B" and stmt not in DOC
                old, new = v["meta"]["swap"]
                assert stmt.replace(new, old, 1) in DOC or new.isdigit()
    assert n_rec >= 40 and n_ver >= 40 and 0.3 < n_yes / n_ver < 0.7, (n_rec, n_ver, n_yes)
    from xenc.doc_tasks import gold_text, q_prompt, score
    assert score(v, gold_text(v)) == 1.0 and "Respond with ONLY the letter" in q_prompt(v)
    assert score(it, gold_text(it)) == 1.0


def test_recon_flags_default_off():
    from test_ds import _parse
    a = _parse(["--out", "x", "--init", "x"])
    assert a.recon_w == 0.0 and a.recon_val == 0 and a.recon_verify_p == 0.0 and a.recon_twin == 0


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_") and callable(f):
            f()
    print("test_recon: all passed")
