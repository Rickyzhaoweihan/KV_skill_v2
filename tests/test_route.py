"""ROUTE: register/don't-register router analysis + probe helpers (CPU, no model loads)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _row(bench, doc, kv0, kv_es, sig, n=2):
    r = {"bench": bench, "doc_id": doc, "kv0": kv0, "kv_es": kv_es, "n_items": n, "ho": {"0": sig, "50": sig / 2},
         "recipe": "tt", "es200": 50}
    import route_analyze as RA
    return RA.add_signals(r)


def test_router_fit_recovers_threshold():
    import route_analyze as RA
    rows = [_row("a", f"d{i}", 0.5, 0.5 + (0.2 if i >= 5 else -0.2), 0.01 * (i + 1)) for i in range(10)]
    bb = RA.by_bench(rows)
    d, t = RA.fit_tau(bb, "ho0")
    ch = RA.router("ho0", (d, t))
    assert [ch(r) for r in rows] == [False] * 5 + [True] * 5
    assert abs(RA.macro(bb, ch) - 0.6) < 1e-9                      # router = oracle here
    assert abs(RA.macro(bb, lambda r: True) - 0.5) < 1e-9 and abs(RA.macro(bb, lambda r: False) - 0.5) < 1e-9
    assert rows[0]["gain"] == -0.2 and abs(rows[0]["sig"]["hodrop_rel"] - 0.5) < 1e-3


def test_rf_runs_keep_ho_separate():
    import route_analyze as RA
    r = RA.add_signals({"bench": "a", "doc_id": "x", "kv0": 0.1, "kv_es": 0.2, "n_items": 1, "ho": {"0": 0.7},
                        "recipe": "rf", "es200": 0})
    assert "ho0" not in r["sig"] and r["sig"]["rf_ho0"] == 0.7


def test_probe_items_holdout_first_and_scorable():
    import route_probe as RP
    its = [{"key": f"d_s{i}", "question": f"q{i}", "answers": [f"a{i}"], "choices": None, "correct_label": None,
            "meta": {"d20selfq": True, "template": "span"}} for i in range(10)]
    its.append({"key": "d_s10", "question": "q10", "answers": ["x"], "choices": [{"label": "A", "text": "x"}],
                "correct_label": None, "meta": {"d20selfq": True, "template": "mcq"}})   # unscorable MCQ: dropped
    P, nh = RP.probe_items(its, "d", k=4, holdout=0.2)
    assert len(P) == 4 and nh == 2 and all(p["key"] != "d_s10" for p in P)
    P2, _ = RP.probe_items(its, "d", k=4, holdout=0.2)
    assert [p["key"] for p in P] == [p["key"] for p in P2]          # deterministic


def test_register_d20_namespace_defaults():
    import route_probe as RP
    ns = RP.register_d20_namespace(["--split_dir", "data/x", "--selfq", "s.jsonl", "--out", "/nonexistent/route"])
    assert ns.split_dir == "data/x" and ns.parts == "dA,gate" and ns.holdout == 0.0 and ns.budget == 32
    import argparse
    assert argparse.ArgumentParser.parse_args.__name__ == "parse_args"   # patch restored


if __name__ == "__main__":
    test_router_fit_recovers_threshold()
    test_rf_runs_keep_ho_separate()
    test_probe_items_holdout_first_and_scorable()
    test_register_d20_namespace_defaults()
    print("test_route OK")
