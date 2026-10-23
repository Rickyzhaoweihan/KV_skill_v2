"""P1: the stored-top-k KL used for the label-free distillation corpus (scripts/p1_data.py)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p1_data import topk_kl  # noqa: E402
from xenc.losses import kl_teacher_student  # noqa: E402


def _store(t_logits, k):
    lp = torch.log_softmax(t_logits.float(), -1)
    v, i = lp.topk(k, -1)
    rest = torch.log1p(-v.exp().sum(-1).clamp(max=1 - 1e-6))
    return i.int(), v.half(), rest


def test_full_k_equals_exact_kl():
    torch.manual_seed(0)
    t, s = torch.randn(5, 50) * 3, torch.randn(5, 50) * 3
    i, v, r = _store(t, 50)
    # fp16 storage of the teacher log-probs -> tiny tolerance
    assert torch.allclose(topk_kl(i, v.float(), r, s), kl_teacher_student(t, s), atol=2e-3)


def test_topk_is_coarse_grained_kl_and_close_when_peaked():
    torch.manual_seed(1)
    t = torch.randn(4, 1000)
    t[:, :3] += 12.0                       # peaked teacher, like answer rows
    s = torch.randn(4, 1000, requires_grad=True)
    i, v, r = _store(t, 16)
    kt = topk_kl(i, v, r, s)
    ex = kl_teacher_student(t, s)
    assert float(kt) <= float(ex) + 1e-3   # coarse-graining never increases KL
    assert abs(float(kt) - float(ex)) < 0.05 * float(ex) + 1e-3
    kt.backward()
    assert s.grad is not None and float(s.grad.abs().sum()) > 0



def test_load_filters_and_weighted_subset(tmp_path=None):
    import json as _j
    import tempfile
    from p1_data import load_p1, weighted_subset
    d = tmp_path or __import__("pathlib").Path(tempfile.mkdtemp())
    (d / "teacher").mkdir(exist_ok=True)
    docs = {"a": {"path": "x", "n_tokens": 9000, "group": "new", "md5": "1"},
            "b": {"path": "y", "n_tokens": 2000, "group": "new", "md5": "2"}}
    _j.dump(docs, open(d / "docs.json", "w"))
    mk = lambda k, ch: {"key": k, "choices": ch, "conf": 0.9, "agree": True, "target": "<answer>x</answer>", "n_tids": 10}  # noqa: E731
    lines = [{"doc_id": "a", "chunk": "c.pt", "items": [mk("a0", None), mk("a1", [{"label": "A", "text": "t"}]), mk("a2", None)]},
             {"doc_id": "b", "chunk": "c.pt", "items": [mk("b0", None)]}]
    with open(d / "teacher" / "shard0.jsonl", "w") as f:
        for ln in lines:
            f.write(_j.dumps(ln) + "\n")
    t = {"ids": torch.zeros(1, 2, dtype=torch.int32), "lp": torch.zeros(1, 2).half(), "rest": torch.zeros(1),
         "tgt": torch.zeros(1, dtype=torch.int32)}
    torch.save({k: t for k in ("a0", "a1", "a2", "b0")}, d / "teacher" / "c.pt")
    q = lambda *x, **k: None  # noqa: E731
    assert len(load_p1(str(d), log=q)[0]) == 4                                       # default: everything
    it, _, _ = load_p1(str(d), log=q, span_only=True, min_tokens=8000)
    assert sorted(i["key"] for i in it) == ["a0", "a2"]                              # span-only, long docs only
    sub, w = weighted_subset(it, {"a0": 0.1, "a2": 0.5}, top_frac=0.5)
    assert [i["key"] for i in sub] == ["a2"] and w == [0.5]                          # top fraction by base KL
    sub, w = weighted_subset(it, {"a0": 0.1, "a2": 0.0}, top_frac=1.0)
    assert [i["key"] for i in sub] == ["a0"]                                          # zero / missing weight dropped


if __name__ == "__main__":
    test_full_k_equals_exact_kl()
    test_topk_is_coarse_grained_kl_and_close_when_peaked()
    test_load_filters_and_weighted_subset()
    print("test_p1: 3 passed")
