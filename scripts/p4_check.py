"""P4 GPU correctness check (real 4B encoder -> 9B reader, VAL documents only; aborts the job chain on failure).

On a few held-out val items with the pointer interface built exactly as register_d13 builds it (--ptr_layers on the
--init checkpoint):
  1. ids: op["ids"] == the exact prefill ids, len == N, the 9B text prompt's first N ids == op["ids"] (column c = text
     position c), detected body start S == len(chat-template prefix) on non-truncated docs.
  2. a = 0 (init): pointer log P == log_softmax(parent read logits) on the prompt (max |Δ|), identical greedy answers
     with the pointer on and off.
  3. a = 0.5 (temporary): incremental decode == teacher-forced e2e pass on prompt + generated tokens (bf16 tolerance),
     the gate is > 0 and P_copy rows sum to 1.
  4. e2e backward at a = 0 on one item (gold answer, KL-free CE): PTR.a gets a finite non-zero gradient.
  5. cost: decode latency per token and peak memory with the pointer on vs off (same items).

    python scripts/p4_check.py --init outputs/D13_lever1_pilot/d13_best.pt --ptr_layers 19 --out outputs/P4_check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, _get, log  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--init", required=True)
    ap.add_argument("--ptr_layers", default="19")
    ap.add_argument("--ptr_shifts", default="0,1")
    ap.add_argument("--ptr_chain", type=int, default=1)
    ap.add_argument("--items", default="qasper:3,musique:2,loogle:1")
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=100000)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--enc_device", default="")
    ap.add_argument("--share_encoder", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from fit_d13 import D13Rt, build, select_items
    from register_state import kept_logits
    from xenc.backbone import build_chat, text_ids
    from xenc.doc_tasks import ctx_prompt, gold_text
    from xenc.e3_retrieval import body_start
    os.makedirs(a.out, exist_ok=True)
    reader, enc, tok, itf, lmap = build(a, init=a.init)
    assert type(itf).__name__ == "PtrInterface", type(itf)
    log(f"P4 check: init {a.init} | B {a.B} | layers {lmap} | ptr {itf.config()['ptr']['layers']} shifts "
        f"{itf.ptr_shifts} chain {itf.ptr_chain} S {itf.ptr_S} | GPU "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    S_e3 = body_start(tok, a.reader, SYSTEM_ANSWER)
    assert itf.ptr_S == S_e3, (itf.ptr_S, S_e3)
    rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens, lru=1)
    R = rt.R
    import glob
    from register_d13 import excluded_docs
    drop = excluded_docs(",".join(sorted(glob.glob("outputs/D22_sets/*.json"))))
    drop |= set(sum(json.load(open("outputs/D13_lever1_leakage.json")).values(), []))
    items, docs = [], {}
    for spec in a.items.split(","):
        name, k = spec.split(":")
        its, d = select_items(f"{name}:0", "val", {1}, 1, 7)
        its = [i for i in its if _get(i, "doc_id") not in drop][: int(k)]
        items += its
        docs.update({_get(i, "doc_id"): d[_get(i, "doc_id")] for i in its})
    log(f"  check items (val, excluded/judged docs removed): {[str(_get(i, 'key')) for i in items]}")
    rt.docs = docs
    rep = {"items": [], "S": itf.ptr_S}
    P = itf.PTR
    for it in items:
        did = _get(it, "doc_id")
        r = {"key": str(_get(it, "key")), "doc": did}
        op, N, _ = rt.operator(did)
        ids = op["ids"].tolist()
        assert ids == rt.pre(did)[:-a.tail] and len(ids) == N, "ids != prefill ids"
        tids = build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, docs[did]))
        r["text_prefix_equal"] = tids[:N] == ids
        full_pre = len(tok(docs[did], add_special_tokens=False)["input_ids"]) + S_e3 + 50 < a.max_tokens - 1024
        R.load(op, N)
        r["S_detected"], r["N"] = R.ptr_tab["S"], N
        if full_pre:
            assert r["text_prefix_equal"], f"{did}: text prompt ids[:N] != operator ids"
            assert R.ptr_tab["S"] == S_e3, (R.ptr_tab["S"], S_e3)
        rows = rt.rows(did, it)
        # 2. a = 0 -> base
        with torch.no_grad():
            R.load(op, N)
            R.mode = "student"
            lp = reader(input_ids=torch.tensor([rows], device=rt.dev)).logits[0, -8:].float()
            itf.ptr_on = False
            R.load(op, N)
            raw = reader(input_ids=torch.tensor([rows], device=rt.dev)).logits[0, -8:].float()
            itf.ptr_on = True
            R.mode = "off"
        r["a0_max_dlogp"] = float((torch.log_softmax(raw, -1) - lp).abs().max())
        assert r["a0_max_dlogp"] < 1e-3, r["a0_max_dlogp"]
        assert torch.equal(raw.argmax(-1), lp.argmax(-1))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        for on in (True, False):
            itf.ptr_on = on
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            t0 = time.time()
            g, n = rt.gen(did, it, a.budget, read=True)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            r[f"gen_{'on' if on else 'off'}"] = g
            r[f"s_per_tok_{'on' if on else 'off'}"] = round((time.time() - t0) / max(n, 1), 4)
            r[f"peak_GiB_{'on' if on else 'off'}"] = (round(torch.cuda.max_memory_allocated(rt.dev) / 2 ** 30, 2)
                                                     if torch.cuda.is_available() else None)
        itf.ptr_on = True
        assert r["gen_on"] == r["gen_off"], (r["gen_on"], r["gen_off"])
        # 3. a = 0.5: incremental == teacher-forced e2e
        with torch.no_grad():
            P["a"].fill_(0.5)
        gen_ids = []
        with torch.no_grad():
            R.load(op, N)
            R.mode = "student"
            o = reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=True, logits_to_keep=1)
            inc = [o.logits[0, -1].float()]
            gs = [float(R.ptr_out["g"][-1])]
            pcs = float((R.ptr_out["Pc"].sum(-1) - 1).abs().max())
            for _ in range(8):
                nxt = int(inc[-1].argmax())
                gen_ids.append(nxt)
                o = reader(input_ids=torch.tensor([[nxt]], device=rt.dev), past_key_values=o.past_key_values,
                           use_cache=True, logits_to_keep=1)
                inc.append(o.logits[0, -1].float())
                gs.append(float(R.ptr_out["g"][-1]))
            R.mode = "off"
            R.load(op, N)
            R.mode, R.need_logits, R.rec = "e2e", False, {}
            tf = kept_logits(reader, torch.tensor([rows + gen_ids]), len(gen_ids) + 1).float()
            R.mode, R.rec = "off", {}
        inc = torch.stack(inc)
        r["a05_inc_vs_full_max_dlogp_top"] = float((inc.gather(-1, inc.argmax(-1, keepdim=True))
                                                    - tf.gather(-1, inc.argmax(-1, keepdim=True))).abs().max())
        r["a05_argmax_agree"] = float((inc.argmax(-1) == tf.argmax(-1)).float().mean())
        r["a05_gate_mean"], r["a05_Pc_sum_err"] = sum(gs) / len(gs), pcs
        assert r["a05_gate_mean"] > 0 and pcs < 1e-4
        assert r["a05_argmax_agree"] >= 0.75, r
        with torch.no_grad():
            P["a"].zero_()
        rep["items"].append(r)
        log(f"  {did}: N {N} S {r['S_detected']} text==op {r['text_prefix_equal']} | a=0 max|dlogP| "
            f"{r['a0_max_dlogp']:.2e} gen equal | a=.5 inc-vs-full |d top logP| {r['a05_inc_vs_full_max_dlogp_top']:.3f} "
            f"argmax agree {r['a05_argmax_agree']:.2f} g {r['a05_gate_mean']:.3f} | s/tok on {r['s_per_tok_on']} off "
            f"{r['s_per_tok_off']} | peak GiB on {r['peak_GiB_on']} off {r['peak_GiB_off']}")
    # 4. gradient of a at a = 0 (one item, CE on the gold answer through the mixture)
    it = items[0]
    did = _get(it, "doc_id")
    op, N, _ = rt.operator(did)
    tt = text_ids(tok, gold_text(it))
    rows = rt.rows(did, it, tt)
    for p in itf.parameters():
        p.requires_grad_(False)
        p.grad = None
    P["a"].requires_grad_(True)
    R.load(op, N)
    R.ckpt = True
    R.mode, R.need_logits, R.rec = "e2e", True, {}
    with torch.enable_grad():
        s_log = kept_logits(reader, torch.tensor([rows]), len(tt) + 1)
        from xenc.losses import answer_ce
        ce = answer_ce(s_log, tt)
        ce.backward()
    R.mode, R.rec = "off", {}
    rep["grad_a_at_0"] = float(P["a"].grad)
    assert torch.isfinite(P["a"].grad).all() and abs(rep["grad_a_at_0"]) > 0
    log(f"  grad PTR.a at a=0 (CE on {did}): {rep['grad_a_at_0']:.4e} (negative = raising the gate lowers the CE)")
    keys = ("s_per_tok_on", "s_per_tok_off", "peak_GiB_on", "peak_GiB_off")
    rep["mean"] = {k: sum(x[k] for x in rep["items"]) / len(rep["items"]) for k in keys if rep["items"][0][k] is not None}
    json.dump(rep, open(os.path.join(a.out, "p4_check.json"), "w"), indent=1)
    log(f"P4 CHECK PASSED: {len(rep['items'])} items | mean {rep['mean']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
