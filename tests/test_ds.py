"""DS (data scale-up) CPU tests: default-off flags of register_d13 and the strict leakage helpers."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _parse(argv):
    import argparse
    import register_d13
    cap = {}
    orig = argparse.ArgumentParser.parse_args

    def pa(self, args=None, ns=None):
        cap["a"] = orig(self, args, ns)
        raise SystemExit
    argparse.ArgumentParser.parse_args = pa
    try:
        register_d13.main(argv)
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = orig
    return cap["a"]


def test_flags_default_off():
    a = _parse(["--out", "x", "--init", "x"])
    assert a.save_every == 0 and a.train_doc_len == "" and a.extra_splits == ""


def test_periodic_ckpt_path():
    from register_d13 import periodic_ckpt_path
    assert periodic_ckpt_path("o", 1000, 0) is None
    assert periodic_ckpt_path("o", 999, 1000) is None
    assert periodic_ckpt_path("o", 0, 1000) is None
    assert periodic_ckpt_path("o", 2000, 1000) == os.path.join("o", "d13_step2000.pt")
    assert [s for s in range(1, 8001) if periodic_ckpt_path("o", s, 1000)] == list(range(1000, 8001, 1000))


def test_leakage_helpers():
    from ds_leakage import grams, norm_q, text_md5
    a = "The quick brown fox jumps over the lazy dog near the river bank today"
    assert grams(a) == grams(a.upper().replace(" ", "  "))          # case / whitespace insensitive
    assert len(grams(a)) == len(a.split()) - 7
    assert grams("too short text") == set()
    assert norm_q("Who is Mark Hunter?") == norm_q("who is mark hunter")
    assert text_md5("A, b.") == text_md5("a b")


def test_train_index_containment():
    from ds_judged_leak import TrainIndex
    t = " ".join(f"w{i}" for i in range(200))
    ix = TrainIndex({"d1": t}, {"q one"})
    j_in = " ".join(f"w{i}" for i in range(50, 100))                  # fully inside the train doc
    j_out = " ".join(f"x{i}" for i in range(50))
    r = ix.check(j_in, "other", 0.15)
    assert r["bad"] and r["c_jT"] == 1.0
    r2 = ix.check(j_out, "Q one?", 0.15)
    assert r2["bad"] and r2["q"] and r2["c_jT"] == 0.0
    assert not ix.check(j_out, "zzz", 0.15)["bad"]


if __name__ == "__main__":
    for f in (test_flags_default_off, test_periodic_ckpt_path, test_leakage_helpers, test_train_index_containment):
        f()
    print("test_ds: all passed")
