"""Integration checks added by the integration agent (CPU only, TinyLM, < 60 s).

  - sharded-reader device plumbing: every tensor the branch touches goes through `_on(t, h)`; on one
    device `_on` is the identity (no copy, no `.to()`), and Library tensors are never replaced.
    What CANNOT be tested here: real cross-device moves (needs >= 2 GPUs). `test_two_gpu_sharded`
    runs only when torch.cuda.device_count() >= 2 (it is skipped on the login node / CPU).
  - no-op at init for the whole registration path (fresh interface + fresh selector => student
    logits == bare-model logits), with and without the selector.
  - 1-chunk Library (linear read) == ExternalSkillInjector after the device-aware rewrite.
  - register_doc writes history.json even when the loop raises; eval_doc resumes from .ckpt.jsonl.
  - doc_tasks.score edge cases; encode_docs --depths default/help; slurm launchers parse (bash -n).
Run: python tests/test_integration.py   (or pytest)
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

import xenc  # noqa: E402
from xenc import CAND, DEPTHS  # noqa: E402

PY = sys.executable


class _H:
    read_temp, route_temp, top_k, linear_read = 12.0, 10.0, 0, False


def _randomize(mod, seed=0, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in mod.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)


# ----------------------------------------------------------------------------- (a) devices
def test_device_helpers_identity_on_one_device():
    import xenc.injector as xi
    import xenc.selector as xs
    from deltaskill.derive import injection_layers
    from xenc.depth_skill import rho_layer_map
    from xenc.interface import make_interface
    from xenc.library import Library
    from xenc.selector import LayerSelector

    calls = {"inj": 0, "sel": 0}
    orig_i, orig_s = xi._on, xs._on

    def wrap(orig, key):
        def f(t, like):
            out = orig(t, like)
            calls[key] += 1
            assert out is t, "single device: _on must return the SAME tensor (no copy / no .to())"
            return out
        return f

    xi._on, xs._on = wrap(orig_i, "inj"), wrap(orig_s, "sel")
    try:
        reader = TinyLM(d=32, n_layers=10)
        layers = injection_layers(reader, CAND)
        rmap = rho_layer_map(reader, CAND)
        lib = Library([make_skill_rho(24, CAND, n=5 + i, seed=i) for i in range(3)],
                      [rmap[r] for r in CAND], CAND)
        for rank in (0, 4):
            itf = make_interface(32, 24, layers, rank=rank)
            _randomize(itf, 1)
            sel = LayerSelector(32, layers, k=2)
            inj = xi.LibraryInjector(reader, itf, selector=sel)
            inj.load_library(lib)
            ids = {li: (id(lib.K[li]), id(lib.mask[li]), id(lib.S[li])) for li in lib.K}
            calls.update(inj=0, sel=0)
            reader(torch.randn(2, 5, 32)).sum().backward()
            assert calls["inj"] > 0 and calls["sel"] > 0, "device-aware code path must be exercised"
            assert ids == {li: (id(lib.K[li]), id(lib.mask[li]), id(lib.S[li])) for li in lib.K}
            assert all(p.grad is not None for p in itf.parameters() if p.requires_grad)
            inj.detach()
        # single-skill path (XEncInjector) also routed through _on
        itf = make_interface(32, 24, layers)
        _randomize(itf, 2)
        inj = xi.XEncInjector(reader, itf)
        inj.load(make_skill_rho(24, CAND, n=7))
        calls["inj"] = 0
        with torch.no_grad():
            reader(torch.randn(1, 3, 32))
        assert calls["inj"] > 0
        inj.detach()
    finally:
        xi._on, xs._on = orig_i, orig_s


def test_two_gpu_sharded():
    """Real cross-device check; needs >= 2 GPUs (skipped otherwise; never run by the agents)."""
    if torch.cuda.device_count() < 2:
        print("    (skipped: needs 2 GPUs)")
        return
    from deltaskill.derive import injection_layers
    from xenc.injector import LibraryInjector
    from xenc.interface import make_interface
    from xenc.library import Library
    from xenc.depth_skill import rho_layer_map
    from xenc.selector import LayerSelector

    class Sharded(TinyLM):
        def forward(self, x):
            for i, layer in enumerate(self.model.layers):
                x = torch.tanh(layer(x.to(layer.weight.device)))
            return x

    m = Sharded(d=32, n_layers=10)
    for i, layer in enumerate(m.model.layers):
        layer.to("cuda:0" if i < 5 else "cuda:1")
    m.lm_head.to("cuda:1")
    layers = injection_layers(m, CAND)
    rmap = rho_layer_map(m, CAND)
    lib = Library([make_skill_rho(24, CAND, n=6, seed=i) for i in range(2)], [rmap[r] for r in CAND], CAND)
    itf = make_interface(32, 24, layers).to("cuda:0")
    _randomize(itf)
    inj = LibraryInjector(m, itf, selector=LayerSelector(32, layers, k=2).to("cuda:0"))
    inj.load_library(lib)
    m(torch.randn(1, 4, 32, device="cuda:0")).sum().backward()
    assert {lib.K[li].device.index for li in layers} == {0, 1}
    assert all(p.grad is not None and p.grad.device.index == 0 for p in itf.parameters())


# ----------------------------------------------------------------------------- (b) equivalence
def test_one_chunk_linear_equals_external_after_rewrite():
    from deltaskill.derive import injection_layers
    from deltaskill.injector import ExternalSkillInjector
    from xenc.depth_skill import for_reader, rho_layer_map
    from xenc.injector import LibraryInjector, XEncInjector
    from xenc.interface import make_interface
    from xenc.library import Library

    reader = TinyLM(d=32, n_layers=8)
    layers = injection_layers(reader, DEPTHS)
    rmap = rho_layer_map(reader, DEPTHS)
    skill = make_skill_rho(32, DEPTHS, n=9, seed=3)
    x = torch.randn(2, 6, 32, generator=torch.Generator().manual_seed(0))
    for rank in (0, 8):
        itf = make_interface(32, 32, layers, rank=rank)
        _randomize(itf, 5)
        outs = {}
        for name in ("ext", "xenc", "lib"):
            if name == "ext":
                inj = ExternalSkillInjector(reader, itf, read_softmax=False)
                inj.load(for_reader(skill, reader))
            elif name == "xenc":
                inj = XEncInjector(reader, itf, read_softmax=False)
                inj.load(skill)
            else:
                inj = LibraryInjector(reader, itf, read_softmax=False)
                inj.load_library(Library([skill], [rmap[r] for r in DEPTHS], DEPTHS))
            with torch.no_grad():
                outs[name] = reader(x)
            inj.detach()
        with torch.no_grad():
            bare = reader(x)
        assert (outs["ext"] - bare).abs().max() > 1e-3
        assert (outs["xenc"] - outs["ext"]).abs().max() < 1e-6, rank
        assert (outs["lib"] - outs["ext"]).abs().max() < 1e-5, rank


# ----------------------------------------------------------------------------- (c) no-op at init
def test_registration_path_noop_at_init():
    from register_doc import SYSTEM_ANSWER, DocRuntime, build_injector, tiny_setup
    from xenc.backbone import with_target
    from xenc.doc_tasks import gold_text, q_prompt
    from xenc.interface import make_interface
    from xenc.losses import student_logits
    from xenc.selector import LayerSelector

    for use_sel in (False, True):
        depths = CAND if use_sel else DEPTHS
        with tempfile.TemporaryDirectory() as td:
            tz = tiny_setup(td, depths=depths)
            rt = DocRuntime(tz["reader"], tz["tok"], tz["hub"], tz["docs"], tz["cache_dir"], tz["enc_short"],
                            tz["chunk_tokens"], depths)
            it = tz["train"][0]
            itf = make_interface(rt.d_reader, rt.d_encoder([it.doc_id]), rt.layers)
            sel = LayerSelector(rt.d_reader, rt.layers, k=2) if use_sel else None
            inj = build_injector(rt.model, itf, _H, sel)
            sids, m, _ = with_target(rt.tok, "tiny", SYSTEM_ANSWER, q_prompt(it), gold_text(it))
            bare = student_logits(rt.model, sids, m).detach()
            inj.load_library(rt.library(it.doc_id))
            s = student_logits(rt.model, sids, m)
            assert torch.equal(s.detach(), bare), f"fresh interface (+selector={use_sel}) must be a no-op"
            assert len(inj.mag_terms) == len(rt.layers) and all(float(t.detach()) == 0 for t in inj.mag_terms)
            s.sum().backward()                      # the branch is still differentiable at init
            assert itf.B.weight.grad is not None
            assert all(torch.isfinite(p.grad).all() for p in itf.parameters() if p.grad is not None)
            inj.unload()
            assert inj.mag_terms == [] and inj.route_terms == []
            inj.detach()


# ----------------------------------------------------------------------------- (d) register_doc
def test_register_history_written_on_exception():
    import register_doc
    import xenc.losses as L
    orig = L.answer_ce
    n = {"k": 0}

    def boom(*a, **k):
        n["k"] += 1
        if n["k"] >= 3:
            raise KeyboardInterrupt("simulated ^C")
        return orig(*a, **k)

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "run")
        L.answer_ce = boom
        try:
            register_doc.main(["--tiny", "--steps", "6", "--out", out, "--eval_every", "100"])
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("KeyboardInterrupt must propagate")
        finally:
            L.answer_ce = orig
        h = json.load(open(os.path.join(out, "history.json")))
        assert len(h["steps"]) == 2 and "KeyboardInterrupt" in h["error"]
        assert set(h["steps"][0]) >= {"loss", "kl", "ce", "mag", "aux", "grad_norm", "step"}
        assert json.load(open(os.path.join(out, "args.json")))["steps"] == 6


def test_register_encoder_same_cache_name():
    import register_doc
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "run")
        register_doc.main(["--tiny", "--steps", "1", "--out", out, "--encoder", "same"])
        args = json.load(open(os.path.join(out, "args.json")))
        assert args["enc_short"] == "tiny-reader" and args["d_encoder"] == args["d_reader"]
        files = os.listdir(os.path.join(out, "tiny_cache"))
        assert files and all("_tiny-reader_" in f for f in files), files


# ----------------------------------------------------------------------------- (e) eval_doc resume
def test_eval_resume():
    def run(out, *extra):
        r = subprocess.run([PY, str(_ROOT / "scripts" / "eval_doc.py"), "--tiny", "--n", "4", "--out", out,
                            "--conds", "oracle_chunk,kv,closed_book,wrong_doc", *extra],
                           cwd=str(_ROOT), capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        return r.stdout

    with tempfile.TemporaryDirectory() as td:
        o1 = run(td)
        assert "resume: 0 items" not in o1 and "resume:" not in o1
        r1 = json.load(open(os.path.join(td, "eval_test.json")))
        assert r1["conds"] == ["closed_book", "kv", "wrong_doc", "oracle_chunk"]   # canonical order
        o2 = run(td)
        assert "resume: 4 items" in o2, o2
        r2 = json.load(open(os.path.join(td, "eval_test.json")))
        assert r1["summary"] == r2["summary"] or json.dumps(r1["summary"]) == json.dumps(r2["summary"])
        assert "kv_minus_wrong_doc" in r2["summary"] and r2["summary"]["kv_minus_wrong_doc"]["n_paired"] == 4
        o3 = run(td, "--no_resume")
        assert "resume:" not in o3


# ----------------------------------------------------------------------------- (g) scoring
def test_score_edge_cases():
    from xenc.doc_tasks import DocItem, score
    mc = DocItem("k", "d", "q", [{"label": l, "text": t} for l, t in zip("ABCD", "wxyz")], ["x"], "B", "t")
    for g in ("<answer>B</answer>", "<answer>b</answer>", "<answer>(b)</answer>", "<answer>(B)</answer>",
              "<answer>B. x</answer>", "<answer>b) x</answer>", "The answer is B"):
        assert score(mc, g) == 1.0, g
    for g in ("", None, "<answer>A</answer>", "<answer></answer>"):
        assert score(mc, g) == 0.0, g
    sp = DocItem("k", "d", "q", None, ["New York City"], None, "t")
    assert score(sp, "<answer>new york city!</answer>") == 1.0
    assert score(sp, "<answer>The city: New York.</answer>") > 0.8
    for g in ("", None, "<answer></answer>", "   "):
        assert score(sp, g) == 0.0, g


# ----------------------------------------------------------------------------- (f) encode_docs CLI
def test_encode_docs_depth_defaults():
    r = subprocess.run([PY, str(_ROOT / "scripts" / "encode_docs.py"), "--help"], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0
    h = " ".join(r.stdout.split())
    assert ",".join(str(x) for x in CAND) in h and "--selector 1" in h
    import argparse
    import encode_docs
    seen = {}
    orig = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = lambda self, argv=None: seen.setdefault("a", orig(self, ["--tiny"])) and (_ for _ in ()).throw(SystemExit(0))
    try:
        encode_docs.main([])
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = orig
    # default --depths must be what register_doc --selector 0 reads (xenc.DEPTHS)
    assert [float(x) for x in seen["a"].depths.split(",")] == DEPTHS


# ----------------------------------------------------------------------------- (h) slurm
def test_slurm_launchers_parse():
    for f in sorted((_ROOT / "slurm").glob("*.sbatch")):
        r = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, (f, r.stderr)
        s = f.read_text()
        for needle in ("HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home", "HF_HUB_OFFLINE=1",
                       "PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels", "--account=drjieliu",
                       "--output=slurm/logs/"):
            assert needle in s, (f.name, needle)
    smoke = (_ROOT / "slurm" / "smoke.sbatch").read_text()
    assert smoke.index("encode_docs.py") < smoke.index("register_doc.py") < smoke.index("eval_doc.py")
    assert "--steps 5" in smoke and "--n 6" in smoke and "Qwen3.5-4B" in smoke and "Qwen3.5-2B" in smoke


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_") and callable(fn):
            fn()
            print(f"  {k} ok", flush=True)
    print("OK")
