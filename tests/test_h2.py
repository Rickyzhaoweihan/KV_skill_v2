"""H2 / KV-RAG CPU tests: exact RoPE rebase on the real Qwen3.5-4B rotary, permutation-key mapping, retrieval
primitives (chunk grid, BM25, RRF, recall)."""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
for p in (str(_ROOT), str(_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("HF_HOME", "/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _rot():
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B")
    cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    return Qwen3_5TextRotaryEmbedding(cfg)


def test_rebase_equals_rotation_at_shifted_positions():
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    from register_h2 import rebase
    rot = _rot()
    itf = types.SimpleNamespace(rot=rot)
    torch.manual_seed(0)
    k = torch.randn(1, 4, 7, 256)

    def at(pos0):
        pos = torch.arange(pos0, pos0 + 7).view(1, 1, 7).expand(3, 1, 7)
        cos, sin = rot(k, pos)
        return apply_rotary_pos_emb(k, k, cos, sin)[0][0]
    for p, d in ((10, 5000), (90000, -88000), (300, 0)):
        got = rebase(itf, at(p), d)
        ref = at(p + d)
        rel = float((got - ref).norm() / ref.norm())     # fp32 angle rounding at ~1e5 positions only
        assert rel < 1e-3, (p, d, rel, float((got - ref).abs().max()))
        if abs(p) < 20000 and abs(p + d) < 20000:
            assert torch.allclose(got, ref, atol=1e-3), (p, d, float((got - ref).abs().max()))
    assert float(getattr(rot, "attention_scaling", 1.0)) == 1.0


def test_base_key():
    from register_h2 import base_key
    assert base_key("loogle_x_d20s3_p1") == "loogle_x_d20s3"
    assert base_key("clipper_test_00012") == "clipper_test_00012"


def test_retrieval_primitives():
    from h2_retrieval import BM25, chunk_spans, recall_item, rrf, words
    starts = list(range(0, 1000, 10))           # 100 tokens of 10 chars
    sp = chunk_spans(starts, 90, 32, 1000)       # body = 90 tokens (tail 10)
    assert sp == [(0, 320), (320, 640), (640, 900)]
    bm = BM25([words("the cat sat on the mat"), words("dogs bark loudly at night"), words("a cat and a dog")])
    s = bm.scores(words("cat"))
    assert s[1] == 0 and s[0] > 0 and s[2] > 0
    assert rrf([1, 2, 3], [3, 2, 1])[0] in (1, 2, 3) and set(rrf([1, 2], [2, 3])) == {1, 2, 3}
    u, al, cov = recall_item([(100, 200), (700, 710)], sp, [0])
    assert u == 0.5 and al == 0.0 and abs(cov - 100 / 110) < 1e-9


if __name__ == "__main__":
    test_rebase_equals_rotation_at_shifted_positions()
    test_base_key()
    test_retrieval_primitives()
    print("test_h2 OK")
