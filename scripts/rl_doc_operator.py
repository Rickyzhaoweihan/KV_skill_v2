"""RL from document (direction D4): reward-learn a per-document COMPACT operator with the reader frozen.

The paper's second construction (arXiv 2608.05475, reward-learned operator, App. A.3 recipe): for each
document d a compact factorized operator M_d = W_d U_dᵀ (U_d, W_d ∈ R^{m×d_s} per injection layer,
independent, fp32) is optimized from TASK REWARD on d's own questions, read through a REGISTERED
interface (frozen by default; --train_interface co-trains it, as the paper does for reward learning).

  init      text-informed: top-m SVD directions of the document's harvested (centered) chunk slots
            (U = W = those directions) — the analogue of the paper's "RL from skill text"
  rollouts  reader + operator, q_prompt (no document), G samples / prompt at temperature T, budget
            tokens, stop on </answer>; P prompts per step (sampled without replacement per epoch)
  reward    xenc.doc_tasks.score (MCQ exact label / span token-F1)
  update    group-normalized advantage; token-level PG on completion log-probs with a truncated
            importance ratio min(pi/pi_sampler, is_cap) (detached) + beta·k3 KL(seq mean lp) to the
            bare reader; AdamW; global Frobenius grad-norm clip

Outputs (--out): operators/<doc_id>.pt (final, Library-compatible untied one-chunk caches),
operators_init/<doc_id>.pt (the SVD init, for the "no-RL" control), history_<doc>.json, args.json,
interface_ref.pt (copy of the frozen interface + its extra, so eval_doc can run with
`--interface <out>/interface_ref.pt --operator_dir <out>/operators`).

    python scripts/rl_doc_operator.py --reader Qwen/Qwen3.5-9B --split_dir data/quality_split \
        --interface outputs/p1_quality_9b_LL_center/interface_best.pt --max_docs 5 --steps 100 \
        --out outputs/D4_rl_quality_9b
    python scripts/rl_doc_operator.py --tiny --steps 2 --out /tmp/x       # CPU plumbing (TinyLM)

--tiny uses a synthetic verifiable reward (fraction of generated tokens with an even id), since a
random TinyLM never produces a correct answer; everything else is the real code path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402
from register_doc import (DocRuntime, _get, build_injector, load_split_items, log,  # noqa: E402
                          system_prompt, tiny_setup)
from xenc import DEPTHS  # noqa: E402
from xenc.doc_operator import operator_cache, operator_library, save_operator, svd_init  # noqa: E402
from xenc.reward import frobenius_clip_, group_advantages, pg_loss, token_logprobs  # noqa: E402

TINY_TARGET = 0   # --tiny reward: fraction of generated tokens with an even id (always has group variance)


def q_parity(item) -> int:
    return int(hashlib.sha1(str(_get(item, "key")).encode()).hexdigest(), 16) % 2


# ============================================================================ rollouts
def _stop_len(tok, hub, toks: list[int], stop: str, eos_ids: set, pad_id) -> int:
    """Number of valid completion tokens: through the first eos, or through the token that completes
    `stop`; trailing pad (rows finished early by generate) excluded."""
    from xenc.backbone import decode, is_fake  # noqa: F401
    for t, x in enumerate(toks):
        if x in eos_ids:
            return t + 1
        if stop and hasattr(tok, "decode") and stop in tok.decode(toks[max(0, t - 11): t + 1], skip_special_tokens=True):
            return t + 1
    n = len(toks)
    while n > 0 and pad_id is not None and toks[n - 1] == pad_id and pad_id not in eos_ids:
        n -= 1
    return n


@torch.no_grad()
def sample_group(model, tok, hub, ids: list[int], G: int, budget: int, temp: float, stop: str = "</answer>"):
    """-> (comp [G, L] long, mask [G, L] float, behav_lp [G, L] float, texts list[str]).

    Real HF model: one batched model.generate (num_return_sequences=G) with per-row stopping and
    output_scores (the processed sampling distribution -> behaviour log-probs for the IS ratio).
    Fake TinyLM: an explicit sampling loop over model.logits."""
    from xenc.backbone import decode, embed_ids, eval_mode, input_device, is_fake
    eval_mode(model)
    if is_fake(model):
        cur = torch.tensor([ids] * G, dtype=torch.long, device=model.device)
        lps, toks = [], []
        for _ in range(budget):
            lg = model.logits(embed_ids(model, cur))[:, -1].float() / max(temp, 1e-6)
            lp = torch.log_softmax(lg, -1)
            nxt = torch.multinomial(lp.exp(), 1)
            toks.append(nxt)
            lps.append(lp.gather(-1, nxt))
            cur = torch.cat([cur, nxt], 1)
        comp = torch.cat(toks, 1)
        blp = torch.cat(lps, 1)
        mask = torch.ones_like(blp)
        texts = [decode(tok, hub, comp[g].tolist()) for g in range(G)]
        return comp, mask, blp, texts
    from transformers import StoppingCriteria, StoppingCriteriaList
    dev = input_device(model)
    p = torch.tensor([ids], dtype=torch.long, device=dev)
    plen = p.shape[1]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    class _RowStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            tail = input_ids[:, plen:][:, -12:]
            return torch.tensor([stop in tok.decode(r, skip_special_tokens=True) for r in tail.tolist()],
                                dtype=torch.bool, device=input_ids.device)

    out = model.generate(input_ids=p, attention_mask=torch.ones_like(p), do_sample=True, temperature=float(temp),
                         top_p=1.0, top_k=0, min_p=None, repetition_penalty=1.0, max_new_tokens=budget,
                         num_return_sequences=G, pad_token_id=pad, output_scores=True,
                         return_dict_in_generate=True, stopping_criteria=StoppingCriteriaList([_RowStop()]))
    comp = out.sequences[:, plen:]                                   # [G, L]
    L = comp.shape[1]
    sc = torch.stack(out.scores, 1)[:, :L].float()                   # [G, L, V] processed logits
    blp = token_logprobs(sc, comp)                                   # [G, L]
    del sc
    eos = tok.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if gen_eos is not None:
        eos_ids |= set(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos])
    mask = torch.zeros(G, L, device=comp.device)
    texts = []
    for g in range(G):
        row = comp[g].tolist()
        n = _stop_len(tok, hub, row, stop, eos_ids, pad)
        mask[g, :n] = 1.0
        txt = decode(tok, hub, row[:n])
        if stop and stop in txt:
            txt = txt.split(stop, 1)[0] + stop
        texts.append(txt)
    blp = torch.where(mask > 0, blp, torch.zeros_like(blp))           # -inf rows of pads -> 0 (masked)
    return comp, mask, blp, texts


def completion_logprobs(model, ids: list[int], comp: torch.Tensor, mask: torch.Tensor, grad: bool) -> torch.Tensor:
    """Per-token log p(comp | prompt) under the CURRENT injector state: [G, L]."""
    from xenc.backbone import embed_ids, input_device, is_fake
    G, L = comp.shape
    dev = model.device if is_fake(model) else input_device(model)
    p = torch.tensor([ids] * G, dtype=torch.long, device=dev)
    full = torch.cat([p, comp.to(dev)], 1)
    with torch.set_grad_enabled(grad):
        if is_fake(model):
            lg = model.logits(embed_ids(model, full))[:, -L - 1:-1]
        else:
            attn = torch.cat([torch.ones_like(p), mask.to(dev).long()], 1)
            lg = model(input_ids=full, attention_mask=attn, logits_to_keep=L + 1, use_cache=False).logits[:, :-1]
        return token_logprobs(lg, comp.to(lg.device))


# ============================================================================ main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default=None, help="default: from the interface's extra (or 'same')")
    ap.add_argument("--split_dir", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--q_split", choices=["all", "train", "heldout"], default="train",
                    help="questions used for RL (sha1(key) parity, as eval_doc --q_split)")
    ap.add_argument("--max_docs", type=int, default=5, help="first N doc_ids (sorted) of the split")
    ap.add_argument("--doc_ids", default=None, help="comma list (overrides --max_docs)")
    ap.add_argument("--exclude_fold", type=int, default=None, help="drop items whose meta.fold == K")
    ap.add_argument("--interface", default=None, help="registered interface_*.pt (required unless --tiny)")
    ap.add_argument("--train_interface", action="store_true", help="co-train a per-document copy of the interface")
    ap.add_argument("--itf_lr", type=float, default=1e-5)
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--chunk_tokens", type=int, default=None)
    ap.add_argument("--center", default=None, help="center file for the INIT slots; default = the interface's")
    ap.add_argument("--m", type=int, default=64, help="operator slots")
    ap.add_argument("--init", choices=["svd", "random"], default="svd")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prompts", type=int, default=8,
                    help="prompt groups per step; may exceed the document's question count (questions are "
                         "cycled in shuffled epochs, so a question can appear in several groups of one step)")
    ap.add_argument("--group", type=int, default=8, help="samples per prompt")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.01, help="k3 KL to the bare reader")
    ap.add_argument("--kl_warmup", type=int, default=0, help="steps with beta = 0 (KL-free warm-up)")
    ap.add_argument("--is_cap", type=float, default=2.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--adv", choices=["grpo", "drgrpo", "rloo"], default="grpo")
    ap.add_argument("--eval_every", type=int, default=25, help="greedy reward on the RL questions (0 = off)")
    ap.add_argument("--snap_steps", default="", help="comma list of steps at which to also save operators_s<step>/ "
                    "(a document stopped early writes its final operator to every later snapshot)")
    ap.add_argument("--stop_patience", type=int, default=0,
                    help="stop a document after K consecutive steps with NO informative group (0 = never)")
    ap.add_argument("--resume", action="store_true", help="skip documents whose final operator + history exist")
    ap.add_argument("--read_temp", type=float, default=None)
    ap.add_argument("--route_temp", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_mem", default=None)
    ap.add_argument("--grad_ckpt", type=int, default=1, choices=[0, 1])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    if not a.tiny and not (a.split_dir and a.interface):
        ap.error("--split_dir and --interface are required unless --tiny")
    if a.out is None:
        a.out = "outputs/D4_rl_tiny" if a.tiny else "outputs/D4_rl_doc"
    return a


def main(argv=None) -> int:
    a = parse_args(argv)
    from xenc.backbone import build_chat, eval_mode, generate, input_device, load_reader, short_name, train_mode
    from xenc.doc_tasks import q_prompt, score
    from xenc.interface import load_interface, make_interface, save_interface

    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    extra = {}
    if a.interface:
        extra = torch.load(a.interface, map_location="cpu", weights_only=False).get("extra") or {}
    targs = extra.get("args") or {}
    depths = extra.get("depths") or DEPTHS
    encoder = a.encoder or targs.get("encoder") or "same"
    encoder_same = encoder == "same"
    style = extra.get("style", "answer")

    class _H:
        read_temp = a.read_temp if a.read_temp is not None else targs.get("read_temp", 12.0)
        route_temp = a.route_temp if a.route_temp is not None else targs.get("route_temp", 10.0)
        top_k = targs.get("top_k", 0)
        linear_read = bool(targs.get("linear_read", False))

    if a.tiny:
        torch.set_num_threads(1)
        tz = tiny_setup(a.out, encoder_same=False, depths=depths)
        model, tok, hub, docs = tz["reader"], tz["tok"], tz["hub"], tz["docs"]
        items = tz["train"]
        cache_dir, enc_short, chunk_tokens = tz["cache_dir"], tz["enc_short"], tz["chunk_tokens"]
        encoder_same = False
        a.budget = min(a.budget, 4)
        a.group = min(a.group, 4)
        a.prompts = min(a.prompts, 2)
        a.m = min(a.m, 8)
        a.grad_ckpt = 0
    else:
        from xenc.doc_tasks import load_docs
        docs = load_docs(a.split_dir)
        items = load_split_items(a.split_dir, a.split)
        cache_dir = a.cache_dir or targs.get("cache_dir") or extra.get("cache_dir") or xenc.DEFAULT_CACHE
        enc_short = short_name(a.reader) if encoder_same else short_name(encoder)
        chunk_tokens = a.chunk_tokens or extra.get("chunk_tokens") or 512
        hub = a.reader
        model, tok = load_reader(a.reader, device_map=a.device, max_mem=a.max_mem, grad_ckpt=bool(a.grad_ckpt))

    rt = DocRuntime(model, tok, hub, docs, cache_dir, enc_short, chunk_tokens, depths, encoder_same=encoder_same)
    center_path = a.center or (extra.get("center") if not a.tiny else None)
    if center_path:
        rt.center = {float(k): v for k, v in torch.load(center_path, map_location="cpu", weights_only=True)["center"].items()}
        log(f"init slots centered with {center_path}")

    if a.q_split != "all" and not a.tiny:
        want = 0 if a.q_split == "train" else 1
        items = [i for i in items if q_parity(i) == want]
    if a.exclude_fold is not None:   # e.g. LongHealth: keep the docs whose meta.fold != K
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") != a.exclude_fold]
    if a.doc_ids:
        doc_list = [d.strip() for d in a.doc_ids.split(",") if d.strip()]
    else:
        doc_list = sorted({_get(i, "doc_id") for i in items})
        if a.max_docs:
            doc_list = doc_list[: a.max_docs]
    doc_list = [d for d in doc_list if rt.has_cache(d)]
    if not doc_list:
        log("ERROR: no document with an encoder cache")
        return 2

    dev = input_device(model)
    d_enc = rt.d_encoder(doc_list)
    if a.interface:
        itf0, selector, _ = load_interface(a.interface, rt.d_reader, d_enc, rt.layers)
    else:   # --tiny: a random NON-zero interface (a zero-init B would block every gradient into U/W)
        itf0, selector = make_interface(rt.d_reader, d_enc, rt.layers, gain_init=0.0), None
        with torch.no_grad():
            g = torch.Generator().manual_seed(a.seed)
            itf0.B.weight.copy_(torch.randn(itf0.B.weight.shape, generator=g) * 0.5)
            itf0.gate_b.fill_(2.0)
    itf0 = itf0.to(dev).requires_grad_(False)
    save_interface(itf0, os.path.join(a.out, "interface_ref.pt"), selector,
                   extra={**extra, "d4_source_interface": a.interface, "center": extra.get("center") if not a.tiny else None,
                          "args": {**targs, "encoder": encoder if not a.tiny else targs.get("encoder")},
                          "depths": depths, "style": style, "chunk_tokens": chunk_tokens,
                          "cache_dir": cache_dir, "enc_short": enc_short})
    rho_of = dict(zip(rt.layers, sorted(float(r) for r in depths)))
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump({**vars(a), "depths": depths, "layers": rt.layers, "d_reader": rt.d_reader, "d_encoder": d_enc,
                   "docs": doc_list, "encoder": encoder, "enc_short": enc_short, "center": center_path}, f, indent=1)
    sysp = system_prompt(style)
    log(f"reader {hub} layers {rt.layers} d_s={d_enc} | docs {len(doc_list)} | m={a.m} steps={a.steps} "
        f"P={a.prompts} G={a.group} T={a.temp} budget={a.budget} lr={a.lr} beta={a.beta} is_cap={a.is_cap}")

    def reward_fn(it, text, comp_row, n_valid):
        if a.tiny:
            row = comp_row[: max(int(n_valid), 1)]
            return float((row % 2 == TINY_TARGET).float().mean())
        return float(score(it, text))

    summary = {}
    snaps = sorted(int(x) for x in a.snap_steps.split(",") if x.strip())
    sum_path = os.path.join(a.out, "rl_summary.json")
    if a.resume and os.path.exists(sum_path):
        summary = json.load(open(sum_path))
    t_all = time.time()
    for di, did in enumerate(doc_list):
        if a.resume and did in summary and os.path.exists(os.path.join(a.out, "operators", f"{did}.pt")):
            log(f"[{did}] resume: already done")
            continue
        pool = [i for i in items if _get(i, "doc_id") == did]
        if not pool:
            log(f"[{did}] no RL questions; skipping")
            continue
        lib0 = rt.library(did)
        if a.init == "svd":
            U0 = svd_init(lib0, a.m, device=dev if not a.tiny else None)
        else:
            gen = torch.Generator().manual_seed(a.seed + di)
            U0 = {li: (lambda R: R / R.norm(dim=-1, keepdim=True))(torch.randn(a.m, d_enc, generator=gen))
                  for li in rt.layers}
        U = {li: torch.nn.Parameter(U0[li].clone().float().to(dev)) for li in rt.layers}
        W = {li: torch.nn.Parameter(U0[li].clone().float().to(dev)) for li in rt.layers}
        save_operator(operator_cache(U, W, rho_of, did, enc_short, {"init": a.init, "m": a.m, "step": 0}),
                      os.path.join(a.out, "operators_init"))
        itf = copy.deepcopy(itf0).requires_grad_(bool(a.train_interface)) if a.train_interface else itf0
        inj = build_injector(model, itf, _H, selector)
        params = list(U.values()) + list(W.values())
        groups = [{"params": params, "lr": a.lr, "weight_decay": a.weight_decay}]
        if a.train_interface:
            groups.append({"params": [p for p in itf.parameters()], "lr": a.itf_lr, "weight_decay": 0.0})
        opt = torch.optim.AdamW(groups)
        meta = {"doc_id": did}
        hist = {"doc_id": did, "n_questions": len(pool), "steps": [], "evals": []}
        log(f"=== [{di + 1}/{len(doc_list)}] {did}: {len(pool)} RL questions")

        @torch.no_grad()
        def greedy_eval(step):
            lib = operator_library(U, W, rt.layers, depths, meta)
            tot, ntok = 0.0, 0
            for it in pool:
                ids = build_chat(tok, hub, sysp, q_prompt(it))
                inj.load_library(lib)
                try:
                    g, k = generate(model, tok, hub, ids, a.budget, return_n=True)
                finally:
                    inj.unload()
                tot += score(it, g)
                ntok += k
            r = {"step": step, "train_greedy": tot / len(pool), "gen_tokens": ntok / len(pool)}
            hist["evals"].append(r)
            log(f"  [eval @ {step}] greedy reward on RL questions {r['train_greedy']:.3f} (gen {r['gen_tokens']:.1f} tok)")

        if a.eval_every and not a.tiny:
            greedy_eval(0)
        order: list = []
        t0 = time.time()
        dry = 0
        stopped_at = None

        def snap(step_tag):
            save_operator(operator_cache(U, W, rho_of, did, enc_short,
                                         {"init": a.init, "m": a.m, "step": len(hist["steps"])}),
                          os.path.join(a.out, f"operators_s{step_tag}"))
        for step in range(1, a.steps + 1):
            opt.zero_grad(set_to_none=True)
            rec = {"step": step, "reward": 0.0, "kl": 0.0, "is_w": 0.0, "is_trunc": 0.0, "loss": 0.0,
                   "n_signal": 0, "gen_tokens": 0.0}
            P = a.prompts
            beta = 0.0 if step <= a.kl_warmup else a.beta
            for _ in range(P):
                if not order:
                    order = list(range(len(pool)))
                    rng.shuffle(order)
                it = pool[order.pop()]
                ids = build_chat(tok, hub, sysp, q_prompt(it))
                lib = operator_library(U, W, rt.layers, depths, meta)
                inj.load_library(lib)
                comp, mask, blp, texts = sample_group(model, tok, hub, ids, a.group, a.budget, a.temp)
                nval = mask.sum(1)
                r = torch.tensor([reward_fn(it, texts[g], comp[g].cpu(), nval[g]) for g in range(a.group)])
                rec["reward"] += float(r.mean()) / P
                rec["gen_tokens"] += float(nval.float().mean()) / P
                adv = group_advantages(r, a.adv)
                if float(adv.abs().max()) < 1e-8:                # no within-group signal
                    inj.unload()
                    continue
                rec["n_signal"] += 1
                ref = None
                if beta > 0:
                    inj.enabled = False
                    try:
                        ref = completion_logprobs(model, ids, comp, mask, grad=False)
                    finally:
                        inj.enabled = True
                inj.load_library(lib)
                inj.mag_terms.clear()
                inj.route_terms.clear()
                train_mode(model)
                cur = completion_logprobs(model, ids, comp, mask, grad=True)
                m_ = mask.to(cur.device)
                loss, met = pg_loss(cur, m_, adv.to(cur.device), blp.to(cur.device), ref, beta, a.is_cap)
                (loss / P).backward()
                eval_mode(model)
                inj.unload()
                rec["loss"] += float(loss.detach()) / P
                rec["kl"] += met.get("kl", 0.0) / P
                rec["is_w"] += met["is_w_mean"] / P
                rec["is_trunc"] += met["is_trunc_frac"] / P
            if rec["n_signal"]:          # IS / KL / loss stats: mean over the prompts that carried signal
                for k in ("loss", "kl", "is_w", "is_trunc"):
                    rec[k] *= P / rec["n_signal"]
            gn = frobenius_clip_(params + ([p for p in itf.parameters()] if a.train_interface else []), a.clip)
            if rec["n_signal"]:
                opt.step()
            rec["grad_norm"] = gn
            rec["frac_informative"] = rec["n_signal"] / P
            rec["U_drift"] = float(sum((U[li].detach().cpu() - U0[li]).norm() for li in U) / len(U))
            rec["W_drift"] = float(sum((W[li].detach().cpu() - U0[li]).norm() for li in W) / len(W))
            rec["t"] = round(time.time() - t0, 2)
            hist["steps"].append(rec)
            if not math.isfinite(rec["loss"]):
                log(f"ERROR: non-finite loss at step {step}: {rec}")
                break
            if step <= 3 or step % 5 == 0 or a.tiny:
                log(f"  [step {step}/{a.steps}] reward {rec['reward']:.3f} signal {rec['n_signal']}/{P} loss {rec['loss']:+.4f} "
                    f"kl {rec['kl']:.4f} isw {rec['is_w']:.3f} trunc {rec['is_trunc']:.3f} |g| {gn:.3f} "
                    f"dU {rec['U_drift']:.3f} dW {rec['W_drift']:.3f} gen {rec['gen_tokens']:.1f} "
                    f"({(time.time() - t0) / step:.1f}s/step)")
            if a.eval_every and not a.tiny and step % a.eval_every == 0:
                greedy_eval(step)
            if step in snaps:
                snap(step)
            dry = dry + 1 if rec["n_signal"] == 0 else 0
            if a.stop_patience and dry >= a.stop_patience and step < a.steps:
                stopped_at = step
                log(f"  early stop at step {step}: {dry} consecutive steps without an informative group")
                break
            if torch.cuda.is_available() and step % 10 == 0:
                torch.cuda.empty_cache()
        for s_ in snaps:                       # stopped early: later snapshots = the final operator
            if s_ > len(hist["steps"]):
                snap(s_)
        hist["stopped_at"] = stopped_at
        path = save_operator(operator_cache(U, W, rho_of, did, enc_short,
                                            {"init": a.init, "m": a.m, "step": len(hist["steps"])}),
                             os.path.join(a.out, "operators"))
        if a.train_interface:
            save_interface(itf, os.path.join(a.out, "interfaces", f"{did}.pt"), selector,
                           extra={**extra, "d4_doc": did})
        inj.detach()                                   # remove this document's hooks
        with open(os.path.join(a.out, f"history_{did}.json"), "w") as f:
            json.dump(hist, f, indent=1)
        last = hist["steps"][-5:]
        ns = sum(s_["n_signal"] for s_ in hist["steps"])
        summary[did] = {"n_questions": len(pool),
                        "frac_informative_groups": ns / max(1, a.prompts * len(hist["steps"])),
                        "frac_informative_last50": sum(s_["n_signal"] for s_ in hist["steps"][-50:]) / max(1, a.prompts * len(hist["steps"][-50:])), "reward_first": hist["steps"][0]["reward"] if hist["steps"] else None,
                        "reward_last5": sum(s["reward"] for s in last) / max(len(last), 1),
                        "greedy": [(e["step"], e["train_greedy"]) for e in hist["evals"]], "operator": path,
                        "steps_run": len(hist["steps"]), "stopped_at": stopped_at}
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=1)
        log(f"[{did}] done in {time.time() - t0:.0f}s -> {path}")
    with open(os.path.join(a.out, "rl_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if torch.cuda.is_available():
        log(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    log(f"done: {len(summary)} docs in {time.time() - t_all:.0f}s -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
