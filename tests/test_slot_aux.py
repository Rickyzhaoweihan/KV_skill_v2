"""D2: slot-level retrieval auxiliary loss (CPU, TinyLM).

  (a) Library of 3 chunks + a gold-slot mask: slot_aux_loss decreases after 20 Adam steps on the interface.
  (b) `register_doc.py --tiny --slot_aux 1.0` runs and records slot-aux item counts.
  (c) gold-slot masks: answer-string path (tiny docs) and MuSiQue supporting-paragraph path.
Run: python tests/test_slot_aux.py   (or pytest)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_ROOT = _TESTS.parent
sys.path.insert(0, str(_TESTS))
sys.path.insert(0, str(_ROOT / "scripts"))
from _tiny import TinyLM, make_skill_rho  # noqa: E402

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from xenc import DEPTHS  # noqa: E402


def test_slot_aux_decreases():
    from xenc.depth_skill import rho_layer_map
    from xenc.injector import LibraryInjector
    from xenc.interface import make_interface
    from xenc.library import Library
    from xenc.losses import slot_aux_loss

    torch.manual_seed(0)
    reader = TinyLM(d=32, n_layers=10).eval().requires_grad_(False)
    rmap = rho_layer_map(reader, DEPTHS)
    layers = [rmap[r] for r in DEPTHS]
    lib = Library([make_skill_rho(24, DEPTHS, n=8, seed=i) for i in range(3)], layers, DEPTHS)
    itf = make_interface(32, 24, layers)
    inj = LibraryInjector(reader, itf, read_temp=12.0)
    inj.load_library(lib)
    gold = torch.zeros(3, 8, dtype=torch.bool)
    gold[1, 2:4] = True
    x = torch.randn(1, 6, 32, generator=torch.Generator().manual_seed(1))
    opt = torch.optim.Adam(itf.parameters(), lr=1e-2)
    losses = []
    inj.training_attn = True
    inj.attn_pos = -1
    for _ in range(21):
        opt.zero_grad()
        inj.attn_terms.clear()
        reader(x)
        assert len(inj.attn_terms) == len(layers)
        assert inj.attn_terms[0].shape == (3, 8) and inj.attn_terms[0].requires_grad
        loss = slot_aux_loss(inj, gold)
        losses.append(float(loss.detach()))
        loss.backward()
        assert itf.A.weight.grad is not None and float(itf.A.weight.grad.abs().sum()) > 0
        opt.step()
    assert losses[-1] < losses[0] - 0.5, losses
    # off by default / under no_grad: nothing recorded
    inj.training_attn = False
    inj.attn_terms.clear()
    reader(x)
    assert inj.attn_terms == []
    assert slot_aux_loss(inj, gold) is None
    inj.unload()
    inj.detach()


def test_gold_slot_masks():
    from register_doc import DocRuntime, tiny_setup
    from xenc.depth_skill import TinyTokenizer
    from xenc.gold_slots import gold_slot_mask, item_gold_slot_mask
    from xenc.library import Library

    with tempfile.TemporaryDirectory() as td:
        tz = tiny_setup(td)
        rt = DocRuntime(tz["reader"], tz["tok"], tz["hub"], tz["docs"], tz["cache_dir"], tz["enc_short"],
                        tz["chunk_tokens"], DEPTHS)
        it = tz["train"][0]
        lib = rt.library(it.doc_id)
        gm = gold_slot_mask(tz["docs"][it.doc_id], it.answers, tz["tok"], tz["chunk_tokens"], lib)
        assert gm is not None and gm.shape == lib.mask[rt.layers[0]].shape
        assert bool(gm[1].any())                         # the answer lives in paragraph 1 == chunk 1
        assert not bool((gm & ~lib.mask[rt.layers[0]]).any())

    # MuSiQue-style: supporting paragraph spans -> every slot of those paragraphs is gold
    tok = TinyTokenizer()
    paras = ["a b c d", "e f g h", "i j k l"]
    text = "\n".join(paras)
    spans, p = [], 0
    for s in paras:
        spans.append([p, p + len(s)]); p += len(s) + 1
    chunks = [{"K": {r: torch.randn(4, 8) for r in DEPTHS}, "n_tokens": 4, "start": 4 * c} for c in range(3)]
    lib = Library(chunks, [1, 2, 3, 4], DEPTHS)
    item = {"answers": ["zzz"], "meta": {"supporting_idx": [2], "para_spans": spans}}
    gm = item_gold_slot_mask(item, text, tok, 4, lib)
    assert gm is not None and gm[2].all() and not gm[:2].any()
    assert item_gold_slot_mask({"answers": ["zzz"], "meta": {}}, text, tok, 4, lib) is None


def test_register_tiny_slot_aux():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "reg")
        r = subprocess.run([sys.executable, str(_ROOT / "scripts" / "register_doc.py"), "--tiny", "--steps", "3",
                            "--slot_aux", "1.0", "--out", out], capture_output=True, text=True, env=env, timeout=600)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        h = json.load(open(os.path.join(out, "history.json")))
        st = h["slot_aux_items"]
        assert st["with_gold"] + st["no_gold"] == len(h["steps"])
        assert st["with_gold"] > 0 or st["no_gold"] > 0
        assert all("slot" in s for s in h["steps"])


if __name__ == "__main__":
    test_slot_aux_decreases()
    test_gold_slot_masks()
    test_register_tiny_slot_aux()
    print("test_slot_aux: all passed")
