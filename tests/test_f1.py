"""F1: read-scale flags (default off) and the clause-level self-NLI parser / leakage guard."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def test_read_scale_mult_default_off():
    from xenc.d13_read import D13Interface

    class O:
        pass
    o = O()
    assert D13Interface.read_scale_mult(o, 100000) == 1.0          # attributes absent = off
    o.read_tau, o.read_logn_ref = 1.0, 0
    assert D13Interface.read_scale_mult(o, 100000) == 1.0
    o.read_logn_ref = 16000
    assert D13Interface.read_scale_mult(o, 8000) == 1.0           # shorter than ref: unchanged
    assert abs(D13Interface.read_scale_mult(o, 16000 ** 2) - 2.0) < 1e-9
    o.read_logn_ref, o.read_tau = 0, 0.5
    assert D13Interface.read_scale_mult(o, 10) == 2.0


def test_nlic_parse_and_leak_guard():
    import f1_nli_selfq as F
    from d20_selfq import task_questions
    from xenc.d20_reg import drop_test_matches
    txt = "\n".join(f"H{i}: A sufficiently long made-up hypothesis number {i} about the agreement terms." for i in range(1, 7))
    qs = F.parse_nlic(txt)
    assert [q["correct_label"] for q in qs] == list("AABBCC")
    frame, choices = F.task_format()
    assert "{h}" in frame and [c["label"] for c in choices] == list("ABC")
    tq = task_questions("data/contractnli_split")
    assert drop_test_matches([dict(q) for q in qs], tq)[1] == 0
    h = re.search(r"\"(.+)\"", json.load(open(ROOT / "data/contractnli_split/test/items.json"))[0]["question"]).group(1)
    assert drop_test_matches([dict(q) for q in F.parse_nlic("H1: " + h)], tq)[1] == 1




def test_chunked_terms_equal_full():
    import torch
    from xenc.d13_read import d13_terms, d13_terms_chunked
    torch.manual_seed(0)
    Hkv, G, T, C, d = 2, 3, 11, 40, 8
    ls = torch.randn(Hkv, G, T, C, requires_grad=True)
    lt = torch.randn(Hkv, G, T, C)
    mask = torch.arange(C).view(1, C) > (C - T) + torch.arange(T).view(T, 1)     # causal own-row mask, as in reads
    ls2 = ls.masked_fill(mask.view(1, 1, T, C), float("-inf"))
    lt = lt.masked_fill(mask.view(1, 1, T, C), float("-inf"))
    ys, yt = torch.randn(T, d, requires_grad=True), torch.randn(T, d)
    full = d13_terms(ls2, lt, ys, yt, C - T)
    g_full = torch.autograd.grad(full["kl"] + full["rel"], (ls, ys))
    for ch in (1, 4, 11, 64):
        ls2 = ls.masked_fill(mask.view(1, 1, T, C), float("-inf"))
        c = d13_terms_chunked(ls2, lt, ys, yt, C - T, ch)
        assert torch.allclose(c["kl"], full["kl"], atol=1e-5) and torch.allclose(c["rel"], full["rel"], atol=1e-5)
        g = torch.autograd.grad(c["kl"] + c["rel"], (ls, ys))
        assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(g, g_full))




def test_oom_event_exits_75_and_records(tmp_path=None):
    import subprocess
    import tempfile
    d = tmp_path or Path(tempfile.mkdtemp())
    code = ("import sys, types; sys.path[:0] = ['scripts', '.']; import register_d20 as R; "
            f"a = types.SimpleNamespace(oom_exit=1, out={str(d)!r}); "
            "R.oom_event(a, 'docA', 'docA_d20s7_p1', 'step'); print('NOT REACHED')")
    p = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert p.returncode == 75, (p.returncode, p.stderr[-2000:])
    assert "NOT REACHED" not in p.stdout
    ev = [json.loads(x) for x in open(Path(d) / "oom_events.jsonl")]
    assert ev == [{"doc_id": "docA", "key": "docA_d20s7", "where": "step", "t": ev[0]["t"]}]
    import register_d20 as R
    import types
    R.oom_event(types.SimpleNamespace(oom_exit=0, out=str(d)), "x", "y", "step")   # off: no-op
    assert len(R.oom_events(str(d))) == 1


if __name__ == "__main__":
    test_read_scale_mult_default_off()
    test_nlic_parse_and_leak_guard()
    test_chunked_terms_equal_full()
    test_oom_event_exits_75_and_records()
    print("test_f1 OK")
