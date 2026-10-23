"""D9: dense distillation (xenc/distill.py) — CPU tests.

  (a) teacher-prompt document tokens map to the right Library slots (whitespace fake tokenizer, 3 chunks).
  (b) the teacher attention recompute equals HF eager `output_attentions` (tiny real Qwen3.5, GQA + gated q).
  (c) attention- and residual-distillation losses both decrease over 20 Adam steps on the interface (TinyLM).
  (d) `register_doc.py --tiny`: flags off => no Distiller, no new history keys; flags on => same step-1 kl/ce/mag
      (the capture hooks never change activations) and finite attn / hid terms.
Run: python tests/test_distill.py   (or pytest)
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

import torch  # noqa: E402

import xenc  # noqa: E402,F401


def _tiny_rt(td, mixing: bool = False):
    """tiny_setup fixtures; mixing=True swaps the reader for a TinyAttnLM (causal token mixing, so the
    document in context changes the residual at the question positions) encoding its own docs (L/L)."""
    from register_doc import DocRuntime, tiny_setup
    tz = tiny_setup(td, encoder_same=False)
    if mixing:
        from _tiny import TinyAttnLM
        tz["reader"] = TinyAttnLM(d=32, n_layers=10, seed=0).eval().requires_grad_(False)
        rt = DocRuntime(tz["reader"], tz["tok"], "tiny", tz["docs"], os.path.join(td, "mix_cache"), "tiny-mix",
                        tz["chunk_tokens"], xenc.DEPTHS, encoder_same=True)
        os.makedirs(os.path.join(td, "mix_cache"), exist_ok=True)
        return tz, rt
    rt = DocRuntime(tz["reader"], tz["tok"], "tiny", tz["docs"], tz["cache_dir"], tz["enc_short"],
                    tz["chunk_tokens"], xenc.DEPTHS)
    return tz, rt


def test_slot_mapping_whitespace():
    from register_doc import SYSTEM_ANSWER
    from xenc.backbone import build_chat
    from xenc.distill import prompt_text, slots_from_token_attn, teacher_slot_map
    from xenc.doc_tasks import ctx_prompt

    with tempfile.TemporaryDirectory() as td:
        tz, rt = _tiny_rt(td)
        tok, it = tz["tok"], tz["train"][0]
        doc = tz["docs"][it.doc_id]
        lib = rt.library(it.doc_id)
        C, N = next(iter(lib.K.values())).shape[:2]
        user = ctx_prompt(it, doc)
        ids = build_chat(tok, "tiny", SYSTEM_ANSWER, user)
        smap = teacher_slot_map(tok, "tiny", SYSTEM_ANSWER, user, doc, lib, tz["chunk_tokens"], n_prompt_ids=len(ids))
        assert smap is not None
        tpos, slot, w, C2, N2 = smap
        assert (C2, N2) == (C, N)
        p_words = prompt_text(tok, "tiny", SYSTEM_ANSWER, user).split()
        d_words = doc.split()
        assert len(tpos) == len(d_words) and torch.allclose(w, torch.ones_like(w))   # 1 word == 1 slot
        starts = [m["start"] for m in lib.chunk_meta]
        for t, s in zip(tpos.tolist(), slot.tolist()):
            c, n = divmod(s, N)
            assert p_words[t] == d_words[starts[c] + n]
        # the doc's first token sits at slot (0,0); paragraph 1 opens chunk 1 ([P1] marker)
        assert slot[0].item() == 0 and p_words[tpos[0]] == "[P0]"
        k = d_words.index("[P1]")
        assert divmod(slot[k].item(), N) == (1, 0)
        # a one-hot teacher attention on that prompt position becomes a one-hot slot distribution
        T = len(ids)
        a = torch.zeros(T)
        a[tpos[k]] = 0.7
        a[0] = 0.3                         # mass outside the document is dropped (renormalised)
        dist = slots_from_token_attn(a, smap)
        assert abs(float(dist.sum()) - 1) < 1e-6 and float(dist[N]) == 1.0
        # the student's [C,N] attention flattens with the same c*N+n convention
        assert lib.mask[lib.layers[0]].reshape(-1)[slot].all()


def test_query_attention_matches_eager():
    from _tiny import tiny_qwen35
    from xenc.distill import _query_attention, attention_windows
    m = tiny_qwen35()
    full = [i for i, b in enumerate(m.model.layers) if hasattr(b, "self_attn")]
    assert attention_windows(m, [2, 4, 7]) == {2: [1], 4: [3], 7: [5, 7]}
    cap, qpos = {}, 9
    hs = [m.model.layers[i].self_attn.register_forward_pre_hook(
        lambda mod, a, k, i=i: cap.__setitem__(i, _query_attention(mod, k["hidden_states"], k["position_embeddings"], qpos)),
        with_kwargs=True) for i in full]
    ids = torch.randint(0, 100, (1, 14), generator=torch.Generator().manual_seed(0))
    out = m(input_ids=ids, output_attentions=True)
    for h in hs:
        h.remove()
    atts = [a for a in out.attentions if a is not None] if len(out.attentions) != len(m.model.layers) else \
        [out.attentions[i] for i in full]
    for i, ref in zip(full, atts):
        assert torch.allclose(ref[0, :, qpos].sum(0).float(), cap[i], atol=1e-5), i
        assert float(cap[i][qpos + 1:].abs().sum()) == 0.0


def test_distill_losses_decrease():
    from register_doc import SYSTEM_ANSWER
    from xenc.backbone import with_target
    from xenc.distill import Distiller
    from xenc.doc_tasks import ctx_prompt, gold_text, q_prompt
    from xenc.injector import LibraryInjector
    from xenc.interface import make_interface
    from xenc.losses import student_logits, teacher_logits

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        tz, rt = _tiny_rt(td, mixing=True)
        model, tok, it = tz["reader"], tz["tok"], tz["train"][1]
        doc = tz["docs"][it.doc_id]
        lib = rt.library(it.doc_id)
        itf = make_interface(rt.d_reader, lib.d_s, rt.layers)
        with torch.no_grad():
            itf.log_gain.fill_(0.0)        # visible write from step 0
        inj = LibraryInjector(model, itf, read_temp=12.0)
        dist = Distiller(model, tok, "tiny", itf.inject_layers, attn_w=1.0, hid_w=1.0, sim_temp=4.0)
        assert all(v == [] for v in dist.windows.values())        # TinyLM: hidden-similarity fallback
        opt = torch.optim.Adam(itf.parameters(), lr=1e-2)
        tids, m, _ = with_target(tok, "tiny", SYSTEM_ANSWER, ctx_prompt(it, doc), gold_text(it))
        sids, _, _ = with_target(tok, "tiny", SYSTEM_ANSWER, q_prompt(it), gold_text(it))
        hist = {"attn": [], "hid": []}
        for _ in range(21):
            opt.zero_grad()
            inj.unload()
            dist.prepare(it.key, SYSTEM_ANSWER, ctx_prompt(it, doc), doc, lib, tz["chunk_tokens"], tids.shape[1] - m)
            with dist.teacher(tids.shape[1], m):
                teacher_logits(model, tids, m)
            assert set(dist.t_attn) == set(rt.layers) and set(dist.t_hid) == set(rt.layers)
            dist.baseline(sids, m)
            inj.load_library(lib)
            inj.training_attn, inj.attn_pos = True, -(m + 1)
            inj.attn_terms.clear()
            with dist.student():
                student_logits(model, sids, m)
            d = dist.losses(inj)
            loss = d["attn"] + d["hid"]
            loss.backward()
            assert itf.A.weight.grad is not None and float(itf.A.weight.grad.abs().sum()) > 0
            assert itf.B.weight.grad is not None and float(itf.B.weight.grad.abs().sum()) > 0
            opt.step()
            hist["attn"].append(float(d["attn"].detach()))
            hist["hid"].append(float(d["hid"].detach()))
            dist.clear()
        inj.training_attn = False
        inj.unload()
        inj.detach()
        assert hist["attn"][-1] < 0.9 * hist["attn"][0], hist["attn"]
        assert hist["hid"][-1] < 0.9 * hist["hid"][0], hist["hid"]


def _run(td, *extra):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    out = os.path.join(td, "run_" + "_".join(x.strip("-") for x in extra) if extra else "off")
    r = subprocess.run([sys.executable, str(_ROOT / "scripts" / "register_doc.py"), "--tiny", "--steps", "3",
                        "--out", out, *extra], capture_output=True, text=True, env=env, cwd=str(_ROOT))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return json.load(open(os.path.join(out, "history.json"))), r.stdout


def test_register_flags_off_identical():
    with tempfile.TemporaryDirectory() as td:
        h0, out0 = _run(td)
        h1, out1 = _run(td, "--attn_distill", "1.0", "--hid_distill", "1.0")
        assert "D9 distill" not in out0 and "D9 distill" in out1
        assert "distill" not in h0 and "attn" not in h0["steps"][0]
        s0, s1 = h0["steps"][0], h1["steps"][0]
        for k in ("kl", "ce", "mag", "aux"):
            assert s0[k] == s1[k], (k, s0[k], s1[k])
        assert s1["attn"] > 0 and s1["hid"] >= 0 and h1["distill"]["attn_items"] >= 1


if __name__ == "__main__":
    test_slot_mapping_whitespace()
    test_query_attention_matches_eager()
    test_distill_losses_decrease()
    test_register_flags_off_identical()
    print("test_distill: OK")
