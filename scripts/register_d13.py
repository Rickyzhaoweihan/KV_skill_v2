"""D13 stage 2: end-to-end registration of the reader-query read (xenc/d13_read.py) with the paper's objective.

Same objective as register_doc.py / register_state.py (losses from xenc/losses.py): teacher = the reader with the
document in context (ctx_prompt + gold answer), student = the reader on its own doc tail + question + gold answer with
the D13 read of the encoder's operator replacing the attention output at the paired layers (gated: own + g·(read−own)):
    loss = kl·KL(p_text ‖ p_student)[answer tokens] + ce·CE(gold) + dist·(KL(p_enc-attn ‖ p_read) + relMSE(B o, B₀ o_enc))
The last term is stage 1's label-free attention distillation against the encoder's own pass over the same rows; it keeps
the read addressing the document (no collapse to a document-independent shift). A (+MLP), B, gates train; the operator
(the encoder's harvested K/V) and both backbones stay frozen. Init: stage-1 file (A) + D10's encoder-side ridge B.
Selection on HELD-OUT DOCUMENTS (QuALITY/MuSiQue/LastingBench val docs, LongHealth --val_fold): greedy kv / wrong_doc
every --eval_every steps.

    python scripts/register_d13.py --init outputs/D13_maps/d13_lin.pt --dist_w 1.0 --steps 2500 --out outputs/D13_reg_w1
"""

from __future__ import annotations

import argparse
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

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, _get, log  # noqa: E402


def excluded_docs(spec: str) -> set:
    """Doc ids of every D22 set in `spec` (judged_docs, reg_docs, and both sides of its wrong map)."""
    out: set = set()
    for f in [x for x in spec.split(",") if x]:
        S = json.load(open(f))
        out |= set(S.get("judged_docs", [])) | set(S.get("reg_docs", []))
        wm = S.get("wrong_map_file")
        if wm and os.path.exists(wm):
            W = json.load(open(wm))
            if isinstance(W, dict):
                out |= set(W) | {v for v in W.values() if isinstance(v, str)}
    return out


def doc_len_filter(spec: str):
    """'task:lo:hi,...' -> predicate(doc_id) keeping docs of those tasks with docs.json n_tokens in [lo, hi]."""
    if not spec:
        return lambda did: True
    from fit_d13 import TASK_SPLIT
    from xenc.doc_tasks import load_docs_index
    rng_, ntok = {}, {}
    for s in spec.split(","):
        t, lo, hi = s.split(":")
        rng_[t] = (int(lo), int(hi))
        ntok.update({d: int(m.get("n_tokens", 0)) for d, m in load_docs_index(TASK_SPLIT[t]).items()})

    def keep(did):
        t = did.split("_")[0]
        return t not in rng_ or rng_[t][0] <= ntok.get(did, -1) <= rng_[t][1]
    return keep


def attn_diag_items(val: list, n_val_attn: int, op_len=None, max_op: int = 0) -> list:
    """Val items used for the stage-1 attention diagnostic (metrics only; never the val score or selection).
    Default (max_op 0): the original stride sample, unchanged. max_op > 0: the same stride sample drawn only from items
    whose operator length op_len(item) <= max_op -- a deterministic, arm-independent rule (P3: the full-tensor JS
    metrics at ~96k-column operators OOM a 46 GB L40S)."""
    pool = val if not max_op else [it for it in val if op_len(it) <= max_op]
    return pool[:: max(1, len(pool) // max(n_val_attn, 1))][: n_val_attn]


def p6_alpha(a, step: int) -> float:
    if a.tok_oracle_p <= 0:
        return 0.0
    if a.tok_oracle_steps <= 0:
        return float(a.tok_oracle_p)
    return float(a.tok_oracle_p) * max(0.0, 1.0 - (step - 1) / float(a.tok_oracle_steps))


def p6_plan(p6: dict, did: str, it, target: str, tt: list, m: int, rows: list, step: int):
    """Answer-content rows of this training item -> {"rows": [(row, gold id)], "plan": {row: [gold-next-token cols]},
    "alpha"}. row indexes the FULL e2e forward (len(rows) rows); plan only when the answer occurs whole-word inside the
    operator columns (P3 diagnostic rules) and the cached prefix tokenization equals the operator ids."""
    from p3_diag import occurrences, oracle_columns
    from register_doc import SYSTEM_ANSWER as _SYS
    from xenc.e3_retrieval import answer_content_rows
    rt_, tok_, a = p6["rt"], p6["tok"], p6["a"]
    tt_c, loc = answer_content_rows(tok_, target)
    if tt_c != list(tt) or not loc:
        return None
    base0 = len(rows) - (m + 1)
    out = {"rows": [(base0 + i, int(tt[i])) for i in loc], "plan": {}, "alpha": p6_alpha(a, step)}
    if out["alpha"] > 0:
        p6["n_try"] += 1
        if did not in p6["pref"]:
            doc = rt_.docs[did]
            ptxt = p6["dpt"](tok_, a.reader, doc, _SYS)
            enc_ = tok_(ptxt, add_special_tokens=False, return_offsets_mapping=True)
            ok = list(enc_["input_ids"]) == rt_.pre(did)
            p6["pref"][did] = (ptxt, [e for _, e in enc_["offset_mapping"]], ok)
            if len(p6["pref"]) > 64:
                p6["pref"].pop(next(iter(p6["pref"])))
        ptxt, ends, ok = p6["pref"][did]
        N = len(rt_.pre(did)) - a.tail
        if ok:
            ans = target[len("<answer>"): -len("</answer>")]
            _, spans = occurrences(ptxt, rt_.docs[did], [ans], ends[N - 1])
            if spans:
                off = tok_(target, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
                for i in loc:
                    cs = oracle_columns(ends, spans, max(0, off[i][0] - len("<answer>")), N)
                    if cs:
                        out["plan"][base0 + i] = cs
                p6["n_plan"] += bool(out["plan"])
    return out


def p6_override(pr: dict, itf):
    """Channel-only p: on the planned rows, (1 − α)·p + α·one-hot(gold next-token columns) (every head)."""
    alpha, plan = pr["alpha"], pr["plan"]

    def fn(li, p):
        if li not in itf.tok_layers:
            return p
        q = p.clone()
        T = p.shape[2]
        for r, cs in plan.items():
            if 0 <= r < T:
                q[..., r, :] = (1 - alpha) * p[..., r, :]
                q[..., r, cs] = q[..., r, cs] + alpha / len(cs)
        return q
    return fn


def p6_aux(p6: dict, tok_z: dict, pr: dict):
    """Logit-lens CE of the adapter outputs at the answer-content rows: CE(z_row · W_Uᵀ, gold token)."""
    terms = []
    for k, z in tok_z.items():
        if z is None:
            continue
        rows = [(r, g) for r, g in pr["rows"] if 0 <= r < z.shape[0]]
        if not rows:
            continue
        Wu = p6["Wu"]
        zz = z[torch.tensor([r for r, _ in rows], device=z.device)].to(Wu.device)
        lg = (zz.to(Wu.dtype) @ Wu.t()).float()        # bf16 matmul on GPU: no fp32 copy of the 248k x d table
        terms.append(torch.nn.functional.cross_entropy(lg, torch.tensor([g for _, g in rows], device=lg.device)))
    return torch.stack(terms).mean() if terms else None


def periodic_ckpt_path(out: str, step: int, every: int) -> str | None:
    """DS --save_every: path of the fixed-step checkpoint to write after `step`, or None (every <= 0: never)."""
    if every and every > 0 and step > 0 and step % every == 0:
        return os.path.join(out, f"d13_step{step}.pt")
    return None


def lr_factor(step: int, steps: int, sched: str = "const", warmup: int = 0, min_frac: float = 0.0) -> float:
    """E1: lr multiplier at 0-based `step`. const -> 1.0 (old behaviour); cosine -> linear warmup then cosine to min_frac."""
    if sched != "cosine":
        return 1.0
    if warmup and step < warmup:
        return (step + 1) / warmup
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(steps - warmup, 1)))


def kl_tempered(t_logits, s_logits, T: float = 1.0):
    """E1: KL(softmax(t/T) || softmax(s/T)) * T^2 (Hinton scaling); T == 1 is exactly kl_teacher_student."""
    from xenc.losses import kl_teacher_student
    return kl_teacher_student(t_logits / T, s_logits / T) * (T ** 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--B", default="outputs/D10_maps/encB8_9b_2b_post.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--init", required=True, help="stage-1 d13 file")
    ap.add_argument("--train", default="quality:150,musique:150,lastingbench:150,longhealth:0")
    ap.add_argument("--task_p", default="quality:0.4,longhealth:0.2,musique:0.2,lastingbench:0.2")
    ap.add_argument("--val", default="quality:10,musique:20,lastingbench:20,longhealth:0")
    ap.add_argument("--val_q_per_doc", type=int, default=4)
    ap.add_argument("--val_fold", type=int, default=1)
    ap.add_argument("--test_fold", type=int, default=0)
    ap.add_argument("--n_val_attn", type=int, default=30)
    ap.add_argument("--val_attn_max_op", type=int, default=0, help="P3: >0 draws the val attention-diagnostic items only "
                    "from val items whose operator has <= this many columns (deterministic, the same in every arm); "
                    "0 = original sample")
    ap.add_argument("--val_max_op", type=int, default=0, help="P3: >0 drops val items whose operator has more columns than "
                    "this BEFORE step 0 (deterministic: identical val in every arm); 0 = keep all")
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--ce", type=float, default=0.5)
    ap.add_argument("--dist_w", type=float, default=1.0)
    ap.add_argument("--kl_ctx", type=float, default=0.0, help="weight of a teacher-forced per-token KL(text ‖ student) "
                    "over the shared NON-answer rows (doc tail + question), in addition to the answer rows")
    ap.add_argument("--kl_ctx_rows", type=int, default=192, help="max shared rows for --kl_ctx")
    ap.add_argument("--rel_w", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5, help="A")
    ap.add_argument("--lr_mlp", type=float, default=5e-5)
    ap.add_argument("--lr_B", type=float, default=1e-5)
    ap.add_argument("--lr_g", type=float, default=1e-3)
    ap.add_argument("--train_B", type=int, default=1)
    ap.add_argument("--train_g", type=int, default=1)
    ap.add_argument("--delta_rank", type=int, default=0, help=">0: freeze the init A / B, train rank-r deltas on both "
                    "+ write-back bias + gates, with --reg·relative drift ‖Δ‖²/‖W0‖² (D12 recipe)")
    ap.add_argument("--reg", type=float, default=1.0)
    ap.add_argument("--lr_delta", type=float, default=5e-5)
    ap.add_argument("--train_mlp", type=int, default=-1, help="-1: yes unless --delta_rank")
    ap.add_argument("--train_from_val", default="", help="e.g. quality:10 — val-split docs AFTER the first 10 "
                    "(sorted; the first 10 are the held-out val docs) are added to training")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_seed", type=int, default=1, help="fixed held-out val question sample (seed 0 runs used 1)")
    ap.add_argument("--mem_safe", type=int, default=0, help="1: torch.utils.checkpoint the e2e read and the per-layer "
                    "distillation terms (identical gradients, less memory; ported from D12p) and skip an item that "
                    "still OOMs. Default 0 = original behaviour")
    ap.add_argument("--exclude_sets", default="", help="comma list of D22 set json files: their judged_docs / reg_docs "
                    "(and wrong-map docs) are removed from train AND val (no judged document is ever trained on)")
    ap.add_argument("--exclude_docs", default="", help="json file {task: [doc_id, ...]} or [doc_id, ...] of extra train/val "
                    "docs to drop (near-duplicates of judged docs, scripts/d13_leakage.py)")
    ap.add_argument("--doc_len", default="", help='e.g. "loogle:32000:128000,clipper:32000:128000": keep only docs of that '
                    "task whose docs.json n_tokens lies in [lo, hi] (train and val)")
    ap.add_argument("--train_doc_len", default="", help="DS: like --doc_len but for TRAIN only (val keeps --doc_len, "
                    "so the held-out val stays identical to a parent run); default '' = --doc_len for both (old behaviour)")
    ap.add_argument("--save_every", type=int, default=0, help="DS: also save d13_step<N>.pt every N steps (checkpoint "
                    "sweeps; independent of val selection); 0 = off (old behaviour)")
    ap.add_argument("--extra_splits", default="", help="DS: 'name=data/<name>_split,...' extra TRAIN-only splits added to "
                    "fit_d13.TASK_SPLIT (e.g. musiquetr, narrativeqa); use them in --train / --task_p by name; default none")
    ap.add_argument("--select_min_step", type=int, default=0, help="val points before this step only set the bar (never "
                    "saved); a checkpoint is saved only when a later point BEATS it (run hygiene: >0 excludes step 0)")
    ap.add_argument("--select_task", default="", help="comma list of tasks: selection criterion = mean val kv over these "
                    "tasks' items (default: all val items)")
    ap.add_argument("--lru", type=int, default=3, help="operator/encoder-cache LRU size (lower for 128k operators)")
    ap.add_argument("--share_encoder", type=int, default=0, help="1: C2, --encoder == --reader, one shared instance")
    ap.add_argument("--enc_device", default="", help="put the encoder (and its operator prefill) on this device, e.g. "
                    "cuda:0 with --device cuda:1 (2-GPU long-operator runs); default = --device")
    ap.add_argument("--extend_src", type=int, default=0, help="lever 2a: layer copied into new --layers (default the "
                    "init's deepest); new layers start at gate 0")
    ap.add_argument("--lr_g_new", type=float, default=0.0, help="lever 2a: gate step size on the NEW layers (g = g0 + "
                    "s*theta, as D12p --lr_g_deep); 0 = one gate lr for all")
    ap.add_argument("--mlp_widen", type=int, default=0, help="lever 2b: widen the residual query MLPs to this many "
                    "hidden units (new units start with zero output weights: identical read at step 0)")
    ap.add_argument("--sched", default="const", choices=["const", "cosine"], help="E1: lr schedule (default const = old)")
    ap.add_argument("--warmup", type=int, default=0, help="E1: linear lr warmup steps (only with --sched cosine)")
    ap.add_argument("--lr_min_frac", type=float, default=0.0, help="E1: cosine floor as a fraction of the base lr")
    ap.add_argument("--kl_temp", type=float, default=1.0, help="E1: temperature T of the KL(text||kv) term "
                    "(softmax(logits/T) on both sides, times T^2); 1 = old")
    ap.add_argument("--ret_w", type=float, default=0.0, help="E3: weight of a retrieval-aligned read term over the "
                    "document-body operator columns at the answer-content rows (xenc/e3_retrieval.py); 0 = off")
    ap.add_argument("--ret_src", default="text", choices=("text", "enc", "gold"),
                    help="E3: text = KL(9B text-pass attention ‖ read), enc = KL(4B encoder attention ‖ read), "
                         "gold = −log read mass on the gold-answer-span columns (train items whose answer occurs in the doc)")
    ap.add_argument("--ret_topk", type=int, default=0, help="E3: keep only the teacher's top-k columns per row (0 = all)")
    ap.add_argument("--ret_layers", default="", help="E3: student read layers that get the term (default: all)")
    ap.add_argument("--ret_tlayer", type=int, default=-1, help="E3 text: one 9B teacher layer for every student layer "
                    "(-1 = the same layer)")
    ap.add_argument("--ret_tasks", default="", help="E3: comma list of tasks whose items get the term (default: all)")
    ap.add_argument("--e2_dv", default="", help="E2 axis 3 (xenc/e2_read.py): 'reader_layer:deep_enc_layer,...' deep "
                    "values V' = v_proj_e(LN_e(h_e')) mixed in by a per-kv-head alpha (zero init: step 0 == init)")
    ap.add_argument("--e2_ms", default="", help="E2 axis 2: 'reader_layer:enc_layer2,...' the same attention also reads "
                    "layer e2's operator values through e2's o_proj and a ridge B2 (--e2_B2), mixed by beta (zero init)")
    ap.add_argument("--e2_B2", default="", help="ridge init file for the --e2_ms maps (scripts/e2_probe.py --save_B2)")
    ap.add_argument("--lr_e2", type=float, default=2e-3, help="E2 mixing weights (alpha / beta / gamma) lr")
    ap.add_argument("--e2_rs", default="", help="E2 retrieval source: 'reader_layer:enc_layer3,...' an extra INDEPENDENT read "
                    "of a deeper encoder layer (own A3 = copy of A_l, e3's projections, ridge B3 from --e2_B2), mixed by "
                    "gamma (zero init), with its own distillation term to the encoder's layer-e3 attention")
    ap.add_argument("--lr_a3", type=float, default=1e-4, help="E2 rs: lr of the new path's A3 (its MLP gets 3x)")
    ap.add_argument("--e2_tr", type=int, default=0, help="E2 operator translator: 1 = per-read-layer zero-init map of the "
                    "operator K (pre-RoPE) / V per kv head (xenc/e2_read.py translate); document-independent")
    ap.add_argument("--lr_tr", type=float, default=1e-5, help="E2 operator translator lr")
    ap.add_argument("--c2_teacher", default="", help="E2: C2 (9B->9B, identity B) interface checkpoint; adds a feature-"
                    "matching loss pulling the student's per-layer read-out toward the C2 self-read on the same rows "
                    "(xenc/e2_c2.py), e.g. outputs/D13_c2_reg/d13_best.pt")
    ap.add_argument("--c2_w", type=float, default=1.0, help="weight of the C2 feature-matching loss")
    ap.add_argument("--c2_cache", default="/scratch/drjieliu_root/drjieliu/rickyhan/e2_c2_targets",
                    help="per-item C2 target cache (fp16) on /scratch")
    ap.add_argument("--save_items", type=int, default=0, help="E1: 1 = store per-item val scores (key, kv, wrong, tail) in "
                    "history.json evals so paired differences vs another run on the SAME items can be computed")
    ap.add_argument("--p1_corpus", default="", help="P1 data scale: label-free distillation corpus dir (scripts/p1_corpus.py + "
                    "p1_teacher.py; precomputed 9B text-teacher top-k on the answer rows). Default off")
    ap.add_argument("--p1_p", type=float, default=0.0, help="P1: probability a step item is a corpus item (0 = off; RNG untouched)")
    ap.add_argument("--p1_ce", type=float, default=0.5, help="P1: CE weight on a corpus item's target, applied only when the "
                    "target is the text teacher's own greedy path (agree) and conf >= --p1_ce_conf")
    ap.add_argument("--p1_ce_conf", type=float, default=0.5)
    ap.add_argument("--p1_min_conf", type=float, default=0.0, help="P1: drop corpus items whose teacher conf is below this")
    ap.add_argument("--p1_groups", default="", help="P1: comma list of corpus groups (bench_train,d15,new); default all")
    ap.add_argument("--p1_max_docs", type=int, default=0, help="P1: use at most this many corpus docs (seeded); 0 = all")
    ap.add_argument("--p1_check", type=int, default=0, help="P1 correctness: for the first N corpus items ALSO run the online "
                    "text teacher and log exact KL vs the stored top-k KL and max |d logp|")
    ap.add_argument("--tok_layers", default="", help="P3 token-identity channel (xenc/p3_tok.py): read layers that also "
                    "gather the reader's own input embeddings E[id_j] of the attended operator columns, written back by "
                    "a zero-init U (step 0 == init). '' = off")
    ap.add_argument("--tok_rank", type=int, default=256, help="P3: rank of the shared PCA projection of E (0 = full, "
                    "top-k only)")
    ap.add_argument("--tok_heads", default="kv", choices=("kv", "avg", "kvavg"), help="P3: per-kv-head token reads, "
                    "one head-averaged read, or (top-k) per-kv-head selection averaged into one read")
    ap.add_argument("--tok_topk_norm", type=int, default=0, help="P3: 1 = hard top-k (weights renormalized over the k "
                    "selected columns)")
    ap.add_argument("--tok_temp", type=float, default=1.0, help="P3: channel-only read temperature, p_tok = "
                    "softmax(logits·temp) (E4's lever; the V read is unchanged); 1 = off")
    ap.add_argument("--tok_shifts", default="-1,0,1", help="P3: neighbour offsets s (block s reads id_{j+s})")
    ap.add_argument("--tok_topk", type=int, default=0, help="P3: >0 keeps only the k highest-attention columns (0 = dense)")
    ap.add_argument("--lr_tok", type=float, default=5e-5, help="P3: lr of U (the token gates use --lr_g)")
    ap.add_argument("--lr_tok_g", type=float, default=0.0, help="P3: separate lr of the token gates γ (0 = --lr_g)")
    ap.add_argument("--tok_src", default="emb", choices=("emb", "unemb"), help="P6: token features from the reader's input "
                    "embedding (emb, P3) or its OUTPUT embedding / lm_head rows W_U (unemb: a residual-stream pointer the "
                    "frozen reader reads out natively; needs --tok_rank 0 --tok_topk > 0 --tok_heads avg|kvavg)")
    ap.add_argument("--tok_adapter", default="lin", help="P6: '+'-set of lin (U, P3), mlp (zero-init out), beta (per-shift "
                    "zero-init gains on the full-rank blocks)")
    ap.add_argument("--tok_mlp_hidden", type=int, default=0, help="P6: hidden width of the mlp adapter")
    ap.add_argument("--tok_inj_layers", default="", help="P6: inject the token read into the residual stream AFTER these "
                    "decoder layers (e.g. 23,27,31) instead of the read layer's attention output; '' = P3 behaviour")
    ap.add_argument("--tok_floor", type=float, default=0.0, help="P6: token-gate floor (no collapse), active from "
                    "--tok_floor_from")
    ap.add_argument("--tok_floor_from", type=int, default=0, help="P6: step at which the gate floor switches on")
    ap.add_argument("--tok_aux_w", type=float, default=0.0, help="P6: logit-lens CE on the injected vector itself, "
                    "CE((adapter output)·W_Uᵀ, gold next token) on the answer-content rows (direct gradient for the adapter)")
    ap.add_argument("--tok_oracle_p", type=float, default=0.0, help="P6 fair oracle: mix weight α of a one-hot p on the "
                    "next gold token's column into the CHANNEL's p on the answer-content rows of training items whose "
                    "answer occurs in the document (the V read keeps its own p); 0 = off")
    ap.add_argument("--tok_oracle_steps", type=int, default=0, help="P6: anneal α linearly to 0 over this many steps "
                    "(0 = constant α)")
    ap.add_argument("--tok_val_off_from", type=int, default=-1, help="P3: from this step on, val also scores kv with the "
                    "channel switched off on the same checkpoint (kv_tokoff; ablation); -1 = never")
    ap.add_argument("--ptr_layers", default="", help="P4 pointer / copy head (xenc/p4_ptr.py): read layers whose read "
                    "attention points at operator columns; the copy distribution over their token ids is mixed into the "
                    "reader's output, P = (1-g) P_reader + g P_copy, gate amplitude a = 0 at init (step 0 == init). '' = off")
    ap.add_argument("--ptr_shifts", default="0,1", help="P4: column offsets s (a pointer at column j copies ids[j+s])")
    ap.add_argument("--ptr_topk", type=int, default=0, help="P4: >0 keeps each row's top-k pointer columns (and their "
                    "+-ptr_nbr neighbours), renormalised; 0 = dense (default)")
    ap.add_argument("--ptr_nosp", type=int, default=0, help="P4: 1 = on rows fed the '>' of '<answer>' a column emits its "
                    "token WITHOUT the leading space ('<answer>Bob' vs the doc's ' Bob'), on rows fed 'answer' the '>'-merged "
                    "form ('>Bob'); the chain matches those variants too. 0 = off (default)")
    ap.add_argument("--ptr_nbr", type=int, default=0, help="P4: neighbours kept around each top-k column")
    ap.add_argument("--ptr_kappa0", type=float, default=1.0, help="P4: initial pointer sharpness κ (logits × κ; then "
                    "learned); 1 = the read's own softmax")
    ap.add_argument("--ptr_chain", type=int, default=1, help="P4: 1 = next-position pointer (after copying column j, "
                    "mass moves to j+1)")
    ap.add_argument("--lr_ptr", type=float, default=5e-3, help="P4: lr of the pointer scalars (a, b, c, head / layer "
                    "weights, sharpness, offset mixture, chain strength)")
    ap.add_argument("--lr_ptr_w", type=float, default=1e-4, help="P4: lr of the gate's hidden-state weights w")
    ap.add_argument("--lr_ptr_a", type=float, default=0.0, help="P4: own lr for the gate amplitude a (e.g. 10x --lr_ptr "
                    "when the gate stays shut); 0 = --lr_ptr (default, unchanged)")
    ap.add_argument("--ptr_copy_w", type=float, default=0.0, help="P4: weight of an extra copy-supervision term "
                    "-log P_copy(gold) on extractive items' answer-content rows whose gold token occurs in the document")
    ap.add_argument("--ptr_val_off_from", type=int, default=-1, help="P4: from this step on, val also scores kv with the "
                    "pointer switched off on the same checkpoint (kv_ptroff; ablation); -1 = never")
    ap.add_argument("--p1_span_only", type=int, default=0, help="P1-hard: 1 = only span-type corpus questions (no choices)")
    ap.add_argument("--p1_min_tokens", type=int, default=0, help="P1-hard: only corpus docs with >= this many tokens")
    ap.add_argument("--p1_weights", default="", help="P1-hard: json {key: base KL} (scripts/p1_score.py); corpus items are "
                    "then the top --p1_top_frac by weight, sampled proportional to weight (instead of uniform doc, item)")
    ap.add_argument("--p1_top_frac", type=float, default=1.0)
    ap.add_argument("--l1_logn", type=int, default=0, help="L1 (xenc/l1_read.py): 1 = scalable softmax, read logits x "
                    "(1 + sigma_h ln(N/l1_nref)) per read layer / query head, sigma trainable, init 0 (= base)")
    ap.add_argument("--l1_nref", type=int, default=16000, help="L1: reference operator length of --l1_logn")
    ap.add_argument("--l1_chunk", type=int, default=0, help="L1: >0 = chunk-local softmax over chunks of this many operator "
                    "columns merged with weights exp(beta_h lse_c); beta = 1 + c_h, c trainable, init 0 (= base)")
    ap.add_argument("--l1_sel", default="", choices=("", "topk", "hier"), help="L1: hard sparse read: topk = top --l1_k "
                    "operator columns per head and row; hier = top --l1_nblk blocks of --l1_block columns by mean-key score")
    ap.add_argument("--l1_k", type=int, default=128)
    ap.add_argument("--l1_block", type=int, default=256)
    ap.add_argument("--l1_nblk", type=int, default=32)
    ap.add_argument("--lr_l1", type=float, default=3e-3, help="L1: lr of sigma / c")
    ap.add_argument("--r2_temp", type=int, default=0, help="R2 (xenc/r2_read.py): 1 = learned per-head read temperature "
                    "exp(theta_h) on the read logits, theta init 0 (= base)")
    ap.add_argument("--r2_hop", type=int, default=0, help="R2: 1 = two-hop re-query: query again from RMSNorm(h + read-1 "
                    "write-back), heads = heads1 + beta_h (heads2 - heads1), beta init 0 (= base)")
    ap.add_argument("--r2_tgate", type=int, default=0, help="R2: 1 = token-conditional write-back gate g + w.x/sqrt(d), w init 0")
    ap.add_argument("--r2_ctx", type=int, default=0, help="R2: >0 = rank of a query term from the causal mean of the rows "
                    "so far (U init 0)")
    ap.add_argument("--r2_mq", type=int, default=0, help="R2: >0 = number of extra queries per head (small random low-rank "
                    "A offsets), mixed by beta_jh init 0")
    ap.add_argument("--r2_mq_rank", type=int, default=16)
    ap.add_argument("--r2_temp_init", type=float, default=1.0, help="R2 temp: initial per-head temperature multiplier "
                    "(1.0 = base; e.g. 1.25 = E4's fixed gain point, still learnable; step 0 then differs from the base)")
    ap.add_argument("--r2_mq_init", type=float, default=0.1, help="R2 mq: relative size of the extra queries' A offsets")
    ap.add_argument("--lr_r2", type=float, default=3e-3, help="R2: lr of the per-head scalars (theta, beta)")
    ap.add_argument("--lr_r2_w", type=float, default=1e-4, help="R2: lr of the vectors / matrices (tgate w, ctx / mq U, V)")
    ap.add_argument("--train_att", type=int, default=0, help="CAP: 1 = train the read's copies of the encoder attention "
                    "projections (q/k/v/o + q/k norms) at --lr_att; the operator (real encoder prefill) is unchanged")
    ap.add_argument("--lr_att", type=float, default=1e-5)
    ap.add_argument("--mlp_B", type=int, default=0, help="CAP: >0 adds a zero-init residual MLP (this many units) on the "
                    "write-back, y = B o + b + MLP(o/rms(o)); trained at --lr_mlpB")
    ap.add_argument("--lr_mlpB", type=float, default=5e-5)
    ap.add_argument("--seg", type=int, default=0, help="ADDR (xenc/addr_seg.py): 1 = structure-aware keys / values: paragraph "
                    "and section-header summaries of the operator added to every column's key / value through zero-init "
                    "per-kv-head maps (step 0 == the init)")
    ap.add_argument("--seg_block", type=int, default=128, help="ADDR SEG: paragraph = block of this many tokens in docs "
                    "without blank lines")
    ap.add_argument("--seg_parts", default="kP,kS,vP,vS", help="ADDR SEG: which summary terms (key/value x paragraph/section)")
    ap.add_argument("--lr_seg", type=float, default=1e-4)
    ap.add_argument("--seg_val_off_from", type=int, default=-1, help="ADDR SEG: from this step on, val also scores kv "
                    "with the SEG terms off (same checkpoint; ablation)")
    ap.add_argument("--hid_w", type=float, default=0.0, help="ADDR (xenc/addr_hid.py): weight of a relMSE between the read "
                    "layers' gated write-back and the 9B reader's own TEXT-mode attention output at the same rows (0 = off)")
    ap.add_argument("--hid_rows", default="all", choices=("all", "qa", "ans"), help="ADDR HID: rows of the term")
    ap.add_argument("--hid_layers", default="", help="ADDR HID: read layers of the term (default all)")
    ap.add_argument("--notes", default="", help="NOTES (xenc/notes.py): 'TYPE:L' = append the encoder's own K/V of a "
                    "question-independent note it writes after the document (type TYPE, L tokens) to every operator, in "
                    "training, val and wrong_doc alike (default '' = off: plain D13Rt)")
    ap.add_argument("--notes_dir", default="", help="NOTES: NoteStore base (default $XENC_CACHE/notes)")
    ap.add_argument("--cmp", default="", help="CMP (xenc/cmp_op.py): compressor file; every operator the reader reads "
                    "(train, val, wrong_doc) goes through the per-token bottleneck; the distillation teacher keeps the raw "
                    "operator (default '' = off)")
    ap.add_argument("--lr_cmp", type=float, default=1e-4, help="CMP: lr of the decoders U, b (0 = frozen closed form)")
    ap.add_argument("--hyb", default="", help="CMP hybrid (xenc/cmp_hyb.py): deep-layer memory channel read natively by "
                    "reader layers 23/27/31 next to the D13 read; a 6-layer compressor file (closed form), gates start at "
                    "--hyb_g0 (0 = step 0 == the init) (default '' = off)")
    ap.add_argument("--hyb_g0", type=float, default=0.0)
    ap.add_argument("--hyb_ckpt", type=int, default=0, help="CMP hybrid: 1 = gradient checkpointing of the deep reads "
                    "(same values; frees the [heads, rows, columns] attention of 3 layers between forward and backward)")
    ap.add_argument("--lr_hyb", type=float, default=1e-4, help="CMP hybrid: lr of the cross-model decoders")
    ap.add_argument("--lr_hyb_g", type=float, default=1e-2, help="CMP hybrid: lr of the deep gates")
    ap.add_argument("--hyb_hm", type=int, default=0, help="HYB-HM (xenc/hyb_hm.py): width d of an HM-style LEARNED writer "
                    "z=rmsnorm(W h_t) (bf16, 2d B/token; no PCA, no quantiser) read by the reader's deep layers through "
                    "learned K/V decoders; V decoder zero-init, FIXED scale 1, so step 0 == the init (default 0 = off)")
    ap.add_argument("--hm_layer", type=int, default=23, help="HYB-HM: 4B layer whose attention-block input is the writer input")
    ap.add_argument("--hm_layers", default="23,27,31", help="HYB-HM: reader layers that read the memory")
    ap.add_argument("--hm_mlp", type=int, default=0, help="HYB-HM: hidden width of an MLP writer (0 = linear)")
    ap.add_argument("--hm_g", type=float, default=1.0, help="HYB-HM: fixed read scale (not trained)")
    ap.add_argument("--lr_hm", type=float, default=3e-4, help="HYB-HM: lr of the writer and the K/V decoders")
    ap.add_argument("--recon_w", type=float, default=0.0, help="RECON (xenc/recon.py): weight of the verbatim "
                    "reconstruction aux items (HM's dense doc-LM fidelity signal in item form): per step, --recon_n extra "
                    "items on this step's training docs, 'recite the continuation of <cue>' (teacher-forced KL(text || kv) on "
                    "every continuation token + CE) or 'does the document state exactly <span>?' (one detail swapped for "
                    "half of them). Own RNG: the main item stream is unchanged. 0 = off (default)")
    ap.add_argument("--recon_n", type=int, default=1, help="RECON: aux items per optimizer step")
    ap.add_argument("--recon_ramp", type=int, default=500, help="RECON: linear weight ramp 0 -> --recon_w over N steps")
    ap.add_argument("--recon_ce", type=float, default=-1.0, help="RECON: CE weight on aux items (-1 = --ce)")
    ap.add_argument("--recon_cue", default="8:24", help="RECON recite: cue length range (tokens)")
    ap.add_argument("--recon_cont", default="12:32", help="RECON recite: continuation length range (tokens)")
    ap.add_argument("--recon_verify_p", type=float, default=0.0, help="RECON: share of aux items that are altered-detail "
                    "verification (Yes/No) instead of recite")
    ap.add_argument("--recon_span", default="16:40", help="RECON verify: statement length range (tokens)")
    ap.add_argument("--recon_twin", type=int, default=0, help="RECON: the text teacher reads a window of N document tokens "
                    "around the span (0 = the whole document)")
    ap.add_argument("--recon_val", type=int, default=0, help="RECON: held-out VAL docs for a teacher-free recite / verify "
                    "probe at every eval (fixed items; token acc / exact match / CE); 0 = off")
    ap.add_argument("--recon_seed", type=int, default=12345)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.train = ",".join(f"{s.split(':')[0]}:2" for s in a.train.split(","))
        a.val = ",".join(f"{s.split(':')[0]}:1" for s in a.val.split(","))
        a.steps, a.eval_every, a.val_q_per_doc, a.n_val_attn = 6, 3, 1, 2
    from eval_doc import wrong_doc_id
    from eval_d13 import summarize
    from fit_d13 import D13Rt, build, fmt, mean_terms, select_items
    from register_state import kept_logits
    from xenc.backbone import build_chat, text_ids
    from xenc.d13_read import save_d13
    from xenc.doc_tasks import ctx_prompt, gold_text, score
    from xenc.losses import answer_ce, kl_teacher_student
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    if a.extra_splits:                                 # DS: register extra train-only splits (default: none)
        import fit_d13 as _f13
        for _s in [x for x in a.extra_splits.split(",") if x]:
            _n, _d = _s.split("=")
            assert _n not in _f13.TASK_SPLIT or _f13.TASK_SPLIT[_n] == _d, f"--extra_splits would redefine {_n}"
            _f13.TASK_SPLIT[_n] = _d
        log(f"  extra_splits: {a.extra_splits}")
    reader, enc, tok, itf, lmap = build(a, init=a.init)
    if a.notes:                                       # NOTES: encoder-written note columns (default off)
        from xenc.notes import build_note_rt
        _nt, _nl = a.notes.split(":")
        assert int(_nl) > 0 and a.ret_w == 0 and a.hid_w == 0 and not a.c2_teacher and not a.p1_corpus, \
            "--notes supports the plain KL / CE / dist recipe only"
        rt = build_note_rt(D13Rt, reader, enc, tok, a.reader, a.encoder, itf, a.tail, a.max_tokens, SYSTEM_ANSWER,
                           [_nt], int(_nl), lru=a.lru, train=True, store_base=a.notes_dir or None)
        rt.note = {"type": _nt, "L": int(_nl), "src": None}
        log(f"NOTES on: type {_nt} L {_nl} | store {rt.store.root} | prompt {len(rt.suffix[_nt])} tokens "
            f"{tok.decode(rt.suffix[_nt])!r}")
    else:
        rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens, lru=a.lru)
    cmp = None
    if a.cmp:                                         # CMP operator bottleneck (default off)
        from xenc.cmp_op import load_cmp, save_cmp, wire
        cmp = load_cmp(a.cmp).to(rt.dev)
        wire(rt, cmp, log=log)
        cmp_U0 = {k: v.detach().clone() for k, v in cmp.U.items()}

    hyb = None
    if a.hyb:                                         # CMP hybrid deep channel (default off)
        from xenc.cmp_hyb import build_hyb, save_hyb
        hyb = build_hyb(rt, a.hyb, g=a.hyb_g0, log=log)
        hyb.ckpt = bool(a.hyb_ckpt)
        hyb_U0 = {k: v.detach().clone() for k, v in hyb.cmp.U.items()}

    hm = None
    if a.hyb_hm:                                      # HYB-HM learned writer + deep read (default off)
        assert not a.hyb and not a.cmp, "--hyb_hm replaces --hyb / --cmp"
        from xenc.hyb_hm import build_hm
        hm = build_hm(rt, d=a.hyb_hm, hw_layer=a.hm_layer, layers=[int(x) for x in a.hm_layers.split(",")], mlp=a.hm_mlp,
                      g=a.hm_g, log=log)
        hm.ckpt = bool(a.hyb_ckpt)
        hm_W0 = hm.dm.writer.weight.detach().clone() if not a.hm_mlp else None

    def save_cmp_as(d13_path):
        if hm is not None:
            from xenc.hyb_hm import save_hm
            save_hm(hm, d13_path.replace("d13_", "hyb_"), {"args": vars(a)})
        if cmp is not None:
            save_cmp(cmp, d13_path.replace("d13_", "cmp_"), {"args": vars(a)})
        if hyb is not None:
            save_hyb(hyb, d13_path.replace("d13_", "hyb_"), {"args": vars(a)})
    tr_folds = set(range(5)) - {a.test_fold, a.val_fold}
    train, d1 = select_items(a.train, "train", tr_folds, 0, a.seed)
    val, d2 = select_items(a.val, "val", {a.val_fold}, a.val_q_per_doc, a.val_seed)
    for spec in [x for x in a.train_from_val.split(",") if x]:
        name, skip = spec.split(":")
        assert name != "longhealth", "LongHealth val docs are a fold; use --val_fold"
        ex, d3 = select_items(f"{name}:0", "val", None, 0, a.seed)
        keep = set(sorted({_get(i, "doc_id") for i in ex})[int(skip):])
        train += [i for i in ex if _get(i, "doc_id") in keep]
        d1.update({k: v for k, v in d3.items() if k in keep})
    drop = excluded_docs(a.exclude_sets)
    if a.exclude_docs:
        X = json.load(open(a.exclude_docs))
        drop |= set(sum(X.values(), [])) if isinstance(X, dict) else set(X)
    keep_len = doc_len_filter(a.doc_len)
    keep_len_tr = doc_len_filter(a.train_doc_len) if a.train_doc_len else keep_len     # DS: train-only length window
    if drop or keep_len:
        n0 = (len(train), len(val))
        train = [i for i in train if _get(i, "doc_id") not in drop and keep_len_tr(_get(i, "doc_id"))]
        val = [i for i in val if _get(i, "doc_id") not in drop and keep_len(_get(i, "doc_id"))]
        log(f"  exclude_sets ({len(drop)} docs) / doc_len: train {n0[0]} -> {len(train)} items, val {n0[1]} -> {len(val)}")
    rt.docs = {**d1, **d2}
    diag_oom: list = []

    def _op_len(it):
        return len(rt.pre(_get(it, "doc_id"))) - a.tail

    if a.val_max_op:                                  # P3: deterministic val length cap (same items in every arm)
        n0 = len(val)
        val = [it for it in val if _op_len(it) <= a.val_max_op]
        log(f"  val_max_op {a.val_max_op}: val {n0} -> {len(val)} items")
    c2t = None
    if a.c2_teacher:                               # E2: C2 self-read teacher (shares the reader's weights)
        from xenc.e2_c2 import C2Teacher
        c2t = C2Teacher(reader, tok, a.reader, a.c2_teacher, a.tail, a.max_tokens, a.c2_cache, rt.docs)
        assert a.dist_w > 0, "--c2_teacher is applied inside the distillation block (needs --dist_w > 0)"
        log(f"E2 C2 teacher: {a.c2_teacher} layers {c2t.layers} (identity B) w {a.c2_w} cache {c2t.cache_dir}")
    tr_docs, va_docs = {_get(i, "doc_id") for i in train}, {_get(i, "doc_id") for i in val}
    assert not (tr_docs & drop) and not (va_docs & drop), "an excluded (judged) document reached train/val"
    assert not (tr_docs & va_docs), "val docs overlap train docs"
    rc = None
    if a.recon_w > 0 or a.recon_val > 0:               # RECON aux items / probe (default off: nothing below runs)
        assert not (a.notes or a.cmp or a.hyb or a.hyb_hm or a.c2_teacher or a.p1_corpus or a.ret_w > 0 or a.hid_w > 0
                    or getattr(itf, "rs", None)), "--recon_* supports the plain KL / CE / dist recipe only"
        from xenc import recon as _rc
        from xenc.e3_retrieval import body_start as _bs
        from xenc.state_skill import doc_prefix_text as _dpt_rc
        _rr = lambda s: tuple(int(x) for x in s.split(":"))                     # noqa: E731
        rc = {"S": _bs(tok, a.reader, SYSTEM_ANSWER), "nfull": {}, "rng": random.Random(a.recon_seed),
              "cue": _rr(a.recon_cue), "cont": _rr(a.recon_cont), "span": _rr(a.recon_span),
              "ce": a.ce if a.recon_ce < 0 else a.recon_ce, "hist": [], "n_try": 0, "skip": 0, "shown": 0}
        log(f"RECON: w {a.recon_w} ramp {a.recon_ramp} n/step {a.recon_n} ce {rc['ce']} | recite cue {rc['cue']} cont "
            f"{rc['cont']} | verify p {a.recon_verify_p} span {rc['span']} | teacher window {a.recon_twin or 'full doc'} | "
            f"val probe docs {a.recon_val} | body start S={rc['S']} | items only on TRAIN docs of this step (rng seed "
            f"{a.recon_seed}, separate from the main item stream)")

    def recon_make(did, rng_, key, kind=None):
        """One RECON item on document did (spans from its operator body columns) or None."""
        pre = rt.pre(did)
        if did not in rc["nfull"]:
            rc["nfull"][did] = len(tok(_dpt_rc(tok, a.reader, rt.docs[did], SYSTEM_ANSWER),
                                       add_special_tokens=False)["input_ids"])
        lo, hi = _rc.body_bounds(len(pre), rc["nfull"][did], rc["S"], a.tail)
        if kind is None:
            kind = "verify" if rng_.random() < a.recon_verify_p else "recite"
        if kind == "verify":
            it_ = _rc.make_verify(tok, pre, lo, hi, rng_, rt.docs[did], key, rc["span"])
        else:
            it_ = _rc.make_recite(tok, pre, lo, hi, rng_, rt.docs[did], key, rc["cue"], rc["cont"])
        if it_ is not None:
            mt = it_["meta"]
            it_["doc_id"] = did
            mt["tdoc"] = (lo, mt["s"], mt["e"])
            assert mt["tdoc"][2] <= len(pre) - a.tail, "RECON span reaches the visible tail"
        return it_

    def recon_loss(it_, w_):
        """Teacher-forced KL(text || kv) + CE (+ dist) on one RECON item, backward with weight w_ / batch."""
        R = rt.R
        did_ = it_["doc_id"]
        tt_ = text_ids(tok, gold_text(it_))
        m_ = len(tt_)
        lo_, s_, e_ = it_["meta"]["tdoc"]
        tids_ = build_chat(tok, a.reader, SYSTEM_ANSWER,
                           ctx_prompt(it_, _rc.teacher_doc(tok, rt.pre(did_), lo_, s_, e_, a.recon_twin, rt.docs[did_]))) + tt_
        if len(tids_) > a.max_tokens:
            return None
        R.mode = "off"
        try:
            with torch.no_grad():
                t_ = kept_logits(reader, torch.tensor([tids_]), m_ + 1).detach()
            rows_ = rt.rows(did_, it_, tt_)
            op_, n_op_, _ = rt.operator(did_)
            rec_ = rt.teacher(did_, rows_) if a.dist_w > 0 else None
            R.load(op_, n_op_)
            R.mode, R.need_logits, R.rec = "e2e", a.dist_w > 0, {}
            with torch.enable_grad():
                s_ = kept_logits(reader, torch.tensor([rows_]), m_ + 1)
                kl_ = kl_tempered(t_[:-1], s_[:-1], a.kl_temp)
                ce_ = answer_ce(s_, rows_[-m_:])
                loss_ = a.kl * kl_ + rc["ce"] * ce_
                dist_ = torch.zeros((), device=kl_.device)
                if a.dist_w > 0:
                    with torch.no_grad():
                        yref_ = {li: (rec_[e]["out"].float() if B0 is None else rec_[e]["out"].float() @ B0[li].t())
                                 for li, e in lmap.items()}
                    terms_ = rt.attn_terms(rec_, op_, n_op_, yref_)
                    dist_ = torch.stack([x["kl"] + a.rel_w * x["rel"] for x in terms_.values()]).mean()
                    loss_ = loss_ + a.dist_w * dist_.to(loss_.device)
                if not math.isfinite(float(loss_)):
                    return None
                assert loss_.requires_grad, "RECON term carries no gradient"
                (w_ * loss_ / a.batch).backward()
            with torch.no_grad():
                gold_ = torch.tensor(rows_[-m_:], device=s_.device)
                hit = (s_[:-1].argmax(-1) == gold_).float()
                thit = (t_[:-1].argmax(-1).to(s_.device) == gold_).float()
            return {"loss": float(loss_), "kl": float(kl_), "ce": float(ce_), "dist": float(dist_),
                    "acc": float(hit.mean()), "em": float(hit.min()), "t_acc": float(thit.mean()), "m": m_,
                    "kind": it_["meta"]["recon"], "w": w_}
        except torch.OutOfMemoryError:
            if not a.mem_safe:
                raise
            rt._ops.clear()
            torch.cuda.empty_cache()
            log(f"  OOM on RECON item {it_['key']}; skipped")
            return None
        finally:
            R.mode, R.rec = "off", {}

    @torch.no_grad()
    def recon_probe():
        """Teacher-free probe on held-out VAL docs: fixed recite + verify items (same in every arm and every eval)."""
        R = rt.R
        rng_v, out = random.Random(a.recon_seed + 1), {"recite": [], "verify": []}
        dv = sorted(va_docs)
        dv = [dv[int(i * len(dv) / a.recon_val)] for i in range(min(a.recon_val, len(dv)))]
        for i, did_ in enumerate(dv):
            for kind in ("recite", "verify"):
                it_ = recon_make(did_, rng_v, f"{did_}_rcv{i}{kind[0]}", kind)
                if it_ is None:
                    continue
                tt_ = text_ids(tok, gold_text(it_))
                m_ = len(tt_)
                rows_ = rt.rows(did_, it_, tt_)
                try:
                    op_, n_op_, _ = rt.operator(did_)
                    R.load(op_, n_op_)
                    R.mode, R.need_logits, R.rec = "e2e", False, {}
                    s_ = kept_logits(reader, torch.tensor([rows_]), m_ + 1)
                except torch.OutOfMemoryError:
                    rt._ops.clear()
                    torch.cuda.empty_cache()
                    continue
                finally:
                    R.mode, R.rec = "off", {}
                gold_ = torch.tensor(rows_[-m_:], device=s_.device)
                hit = (s_[:-1].argmax(-1) == gold_).float()
                out[kind].append({"acc": float(hit.mean()), "em": float(hit.min()),
                                  "ce": float(answer_ce(s_, rows_[-m_:])), "yes": it_["meta"].get("yes")})
        res = {}
        for kind, v in out.items():
            if v:
                res[kind] = {"n": len(v), **{k: round(sum(x[k] for x in v) / len(v), 4) for k in ("acc", "em", "ce")}}
                if kind == "verify":
                    for lab, f in (("yes", True), ("no", False)):
                        vv = [x for x in v if x["yes"] is f]
                        res[kind][f"em_{lab}"] = round(sum(x["em"] for x in vv) / max(len(vv), 1), 4)
        return res
    p1_items, p1_T, p1_by, p1_stats = [], {}, {}, {"n": 0, "kl": 0.0, "ce_used": 0, "skip_tok": 0, "skip_long": 0, "chk": []}
    if a.p1_corpus and a.p1_p > 0:
        assert a.kl_ctx == 0 and a.ret_w == 0 and a.kl_temp == 1.0, "P1 corpus items support only the answer-row KL at T=1"
        from p1_data import load_p1, norm_md5
        from xenc.doc_tasks import load_docs
        p1_items, p1_T, p1_docs = load_p1(a.p1_corpus, a.p1_groups, a.p1_min_conf, a.p1_max_docs, a.seed, log,
                                          span_only=bool(a.p1_span_only), min_tokens=a.p1_min_tokens)
        p1_w = None
        if a.p1_weights:
            from p1_data import weighted_subset
            n0 = len(p1_items)
            p1_items, p1_w = weighted_subset(p1_items, json.load(open(a.p1_weights)), a.p1_top_frac)
            log(f"P1-hard weights {a.p1_weights}: {n0} -> {len(p1_items)} items (top {a.p1_top_frac}); weight min/median/max "
                f"{min(p1_w):.4f} / {sorted(p1_w)[len(p1_w) // 2]:.4f} / {max(p1_w):.4f}; sampling proportional to weight")
        # HARD leakage assertion: no corpus doc is a judged / registered / excluded / val / test doc (by id or by text)
        from fit_d13 import TASK_SPLIT
        from register_doc import load_split_items
        nontrain_ids, nontrain_md5 = set(drop) | set(va_docs), set()
        for _t, _sd in TASK_SPLIT.items():
            _D = load_docs(_sd)
            for _sp in ("val", "test"):
                for _i in load_split_items(_sd, _sp):
                    if _t == "longhealth" and (_get(_i, "meta") or {}).get("fold") not in (0, 1):
                        continue
                    nontrain_ids.add(_get(_i, "doc_id"))
            nontrain_md5 |= {norm_md5(_D[d]) for d in nontrain_ids if d in _D}
        nontrain_md5 |= {norm_md5(rt.docs[d]) for d in va_docs}
        c_ids = {i["doc_id"] for i in p1_items}
        assert not (c_ids & nontrain_ids), f"P1 corpus contains {len(c_ids & nontrain_ids)} non-train (val/test/judged) docs"
        assert not ({p1_docs[d]["md5"] for d in c_ids} & nontrain_md5), "P1 corpus contains the TEXT of a non-train doc"
        log(f"P1 leakage assertion passed: {len(c_ids)} corpus docs vs {len(nontrain_ids)} non-train doc ids / "
            f"{len(nontrain_md5)} texts (val+test of {len(TASK_SPLIT)} splits, LH folds 0/1, D22/exclude sets, this val)")
        for i in p1_items:
            p1_by.setdefault(i["doc_id"], []).append(i)
        p1_dids = sorted(p1_by)
        p1_paths = {d: p1_docs[d]["path"] for d in p1_dids}
    pools: dict = {}
    for it in train:
        pools.setdefault(_get(it, "doc_id").split("_")[0], []).append(it)
    tp = {k: float(v) for k, v in (s.split(":") for s in a.task_p.split(",")) if k in pools}
    tasks, weights = list(tp), [tp[t] for t in tp]
    val_order = {}
    for d in sorted(va_docs):
        val_order.setdefault(d.split("_")[0], []).append(d)
    B0 = None if itf.identity_B else {li: itf.B[str(li)].detach().clone() for li in lmap}
    train_mlp = bool(a.train_mlp) if a.train_mlp >= 0 else not a.delta_rank
    if a.delta_rank:
        itf.trainable(train_B=False, train_g=bool(a.train_g))
        for p in itf.A.parameters():
            p.requires_grad_(False)
        itf.add_delta(a.delta_rank, a.seed)
        groups = [{"params": list(itf.D.parameters()), "lr": a.lr_delta}]
        if a.train_B:
            for p in itf.Bb.parameters():
                p.requires_grad_(True)
            groups.append({"params": list(itf.Bb.parameters()), "lr": a.lr_delta})
    else:
        itf.trainable(train_B=bool(a.train_B), train_g=bool(a.train_g))
        groups = [{"params": list(itf.A.parameters()), "lr": a.lr}]
        if a.train_B:
            groups.append({"params": list(itf.B.parameters()) + list(itf.Bb.parameters()), "lr": a.lr_B})
    for p in itf.M.parameters():
        p.requires_grad_(train_mlp)
    if itf.mlp_hidden and train_mlp:
        groups.append({"params": list(itf.M.parameters()), "lr": a.lr_mlp})
    newl = set(getattr(itf, "new_layers", []) or [])
    if a.train_g and a.lr_g_new > 0 and newl:
        # per-layer gate step size (ported from D12p): g = g0 + s * theta, theta by Adam at lr_g, s = lr_g_new / lr_g on
        # the new layers (Adam is invariant to gradient scale, so a separate step size needs this form)
        import types as _types
        g0 = itf.g.detach().clone()
        sc = torch.tensor([a.lr_g_new / a.lr_g if li in newl else 1.0 for li in itf.layers], device=g0.device)
        theta = torch.nn.Parameter(torch.zeros_like(g0))
        itf.g.requires_grad_(False)
        itf.gate = _types.MethodType(lambda self, li: g0[self._gi[li]] + sc[self._gi[li]] * theta[self._gi[li]], itf)

        def sync_g():
            with torch.no_grad():
                itf.g.copy_(g0 + sc * theta)
        groups.append({"params": [theta], "lr": a.lr_g})
        log(f"gates: new layers {sorted(newl)} step size {a.lr_g_new}, others {a.lr_g}")
    else:
        sync_g = lambda: None                                      # noqa: E731
        if a.train_g:
            groups.append({"params": [itf.g], "lr": a.lr_g})
    if hasattr(itf, "e2_params"):                  # E2: mixing weights (+ B2 maps with the B lr when --train_B)
        ep = itf.e2_params()
        for p in ep["mix"]:
            p.requires_grad_(True)
        groups.append({"params": ep["mix"], "lr": a.lr_e2})
        if ep["B2"]:
            for p in ep["B2"]:
                p.requires_grad_(bool(a.train_B))
            if a.train_B:
                groups.append({"params": ep["B2"], "lr": a.lr_B})
        if ep.get("tr"):
            for p in ep["tr"]:
                p.requires_grad_(True)
            groups.append({"params": ep["tr"], "lr": a.lr_tr})
            log(f"E2: operator translator on {itf.layers}: {sum(p.numel() for p in ep['tr']) / 1e6:.2f}M params lr {a.lr_tr}")
        if ep.get("A3"):
            groups.append({"params": list(itf.E2["A3"].parameters()), "lr": a.lr_a3})
            if itf.mlp_hidden:
                groups.append({"params": list(itf.E2["M3"].parameters()), "lr": 3 * a.lr_a3})
            for p in ep["A3"]:
                p.requires_grad_(True)
        log(f"E2: rs {itf.rs} (A3 params {sum(p.numel() for p in ep.get('A3', []))} lr {a.lr_a3}) B2/B3 init {a.e2_B2}")
        log(f"E2: dv {itf.dv} ms {itf.ms} | mix params {sum(p.numel() for p in ep['mix'])} lr {a.lr_e2} | B2 params "
            f"{sum(p.numel() for p in ep['B2'])} ({'train' if a.train_B else 'frozen'}) | init state {itf.e2_state()}")
    if hasattr(itf, "l1_params"):                  # L1 length-aware read forms: sigma / c (zero init)
        lp = itf.l1_params()
        for p in lp:
            p.requires_grad_(True)
        if lp:
            groups.append({"params": lp, "lr": a.lr_l1})
        log(f"L1 read form: {itf.l1} | params {sum(p.numel() for p in lp)} lr {a.lr_l1} | state {itf.l1_state()}")
    if hasattr(itf, "r2_params"):                  # R2 query-side read forms (zero-init mixing / scales)
        rp = itf.r2_params()
        for p in rp["scal"] + rp["mat"]:
            p.requires_grad_(True)
        if rp["scal"]:
            groups.append({"params": rp["scal"], "lr": a.lr_r2})
        if rp["mat"]:
            groups.append({"params": rp["mat"], "lr": a.lr_r2_w})
        log(f"R2 read form: {itf.r2} | params scal {sum(p.numel() for p in rp['scal'])} lr {a.lr_r2} / mat "
            f"{sum(p.numel() for p in rp['mat'])} lr {a.lr_r2_w} | state {itf.r2_state()}")
    if hasattr(itf, "tok_params"):                 # P3 token channel: U (zero init) + per-layer token gates
        tpp = itf.tok_params()
        for p in tpp["U"] + tpp["gt"]:
            p.requires_grad_(True)
        groups.append({"params": tpp["U"], "lr": a.lr_tok})
        groups.append({"params": tpp["gt"], "lr": a.lr_tok_g if a.lr_tok_g > 0 else a.lr_g})
        itf.tok_log = True
        log(f"P3 token channel: {itf.config()['tok']} | n_in {itf.tok_nin} | U params "
            f"{sum(p.numel() for p in tpp['U']) / 1e6:.2f}M lr {a.lr_tok} | proj init {getattr(itf, 'tok_init_info', None)} "
            f"| state {itf.tok_state()}")
    ptr_pp = []
    if hasattr(itf, "ptr_params"):                 # P4 pointer / copy head: own groups, own grad-norm clip
        pp_ = itf.ptr_params()
        for p in pp_["w"] + pp_["scal"]:
            p.requires_grad_(True)
        if a.lr_ptr_a > 0:                       # gate amplitude in its own group (default: shares --lr_ptr)
            groups.append({"params": [p for p in pp_["scal"] if p is itf.PTR["a"]], "lr": a.lr_ptr_a, "p4": True})
            groups.append({"params": [p for p in pp_["scal"] if p is not itf.PTR["a"]], "lr": a.lr_ptr, "p4": True})
        else:
            groups.append({"params": pp_["scal"], "lr": a.lr_ptr, "p4": True})
        groups.append({"params": pp_["w"], "lr": a.lr_ptr_w, "p4": True})
        ptr_pp = pp_["w"] + pp_["scal"]
        log(f"P4 pointer head: {itf.config()['ptr']['layers']} shifts {itf.ptr_shifts} chain {itf.ptr_chain} S "
            f"{itf.ptr_S} | params {sum(p.numel() for p in ptr_pp)} lr {a.lr_ptr} / w {a.lr_ptr_w} | copy_w "
            f"{a.ptr_copy_w} | state {itf.ptr_state()}")
    cap_pp = {}
    if a.mlp_B and not getattr(itf, "mlp_B", 0):        # CAP write-back MLP (zero output: step 0 == init)
        itf.add_mlp_B(a.mlp_B, a.seed)
    if getattr(itf, "mlp_B", 0):
        cap_pp["MB"] = list(itf.MB.parameters())
        for p in cap_pp["MB"]:
            p.requires_grad_(True)
        groups.append({"params": cap_pp["MB"], "lr": a.lr_mlpB})
    if a.train_att:                                     # CAP trainable read projections
        cap_pp["att"] = itf.train_att()
        groups.append({"params": cap_pp["att"], "lr": a.lr_att})
    if cap_pp:
        log(f"CAP: mlp_B {getattr(itf, 'mlp_B', 0)} ({sum(p.numel() for p in cap_pp.get('MB', [])) / 1e6:.2f}M, lr "
            f"{a.lr_mlpB}) | train_att {a.train_att} ({sum(p.numel() for p in cap_pp.get('att', [])) / 1e6:.2f}M, lr "
            f"{a.lr_att}) | layers {itf.layers} new {sorted(getattr(itf, 'new_layers', []) or [])}")
    if hasattr(itf, "seg_params"):                   # ADDR SEG maps (zero init)
        sp_ = itf.seg_params()
        for p in sp_:
            p.requires_grad_(True)
        groups.append({"params": sp_, "lr": a.lr_seg})
        log(f"ADDR SEG: {itf.seg} | params {sum(p.numel() for p in sp_) / 1e6:.2f}M lr {a.lr_seg} | state {itf.seg_state()}")
    hid = None
    if a.hid_w > 0:                                    # ADDR HID: text-attention-output distillation at the read layers
        from xenc.addr_hid import AttnOutCap
        hl = [int(x) for x in a.hid_layers.split(",")] if a.hid_layers else list(itf.layers)
        assert set(hl) <= set(itf.layers), hl
        hid = {"cap": AttnOutCap(reader, hl), "n": 0, "n_try": 0, "vals": []}
        log(f"ADDR HID: w {a.hid_w} layers {hl} rows {a.hid_rows}")
    p6 = None
    if hasattr(itf, "tok_params") and (a.tok_aux_w > 0 or a.tok_oracle_p > 0 or a.tok_floor > 0):
        from xenc.state_skill import doc_prefix_text as _dpt
        p6 = {"Wu": reader.get_output_embeddings().weight, "pref": {}, "n_plan": 0, "n_try": 0, "aux": [],
              "dpt": _dpt, "rt": rt, "tok": tok, "a": a}
        log(f"P6: aux_w {a.tok_aux_w} | oracle p α0 {a.tok_oracle_p} anneal {a.tok_oracle_steps} | floor {a.tok_floor} "
            f"from step {a.tok_floor_from} | inj layers {itf.tok_inj_layers} | src {itf.tok_src} adapter {itf.tok_adapter}")
    cmp_pp = []
    if cmp is not None and a.lr_cmp > 0:              # CMP decoders: own group, own grad-norm clip
        cmp_pp = cmp.trainable()
        for p in cmp_pp:
            p.requires_grad_(True)
        groups.append({"params": cmp_pp, "lr": a.lr_cmp, "p4": True})
        log(f"CMP decoders trainable: {sum(p.numel() for p in cmp_pp) / 1e6:.2f}M lr {a.lr_cmp}")
    elif cmp is not None:
        for p in cmp.trainable():
            p.requires_grad_(False)
    hyb_pp = []
    if hyb is not None:                               # CMP hybrid: decoders + deep gates, own group / own clip
        hyb_pp = hyb.cmp.trainable() + [hyb.dm.g]
        for p in hyb_pp:
            p.requires_grad_(True)
        groups.append({"params": hyb.cmp.trainable(), "lr": a.lr_hyb, "p4": True})
        groups.append({"params": [hyb.dm.g], "lr": a.lr_hyb_g, "p4": True})
        log(f"HYB trainable: decoders {sum(p.numel() for p in hyb.cmp.trainable()) / 1e6:.2f}M lr {a.lr_hyb} | gates "
            f"lr {a.lr_hyb_g} init {a.hyb_g0}")
    hm_pp = []
    if hm is not None:                                # HYB-HM: writer + K/V decoders, own group / own clip
        hm_pp = hm.dm.trainable()
        for p in hm_pp:
            p.requires_grad_(True)
        groups.append({"params": hm_pp, "lr": a.lr_hm, "p4": True})
        log(f"HYB-HM trainable: {sum(p.numel() for p in hm_pp) / 1e6:.2f}M lr {a.lr_hm} | d {a.hyb_hm} "
            f"({hm.dm.bytes_per_token()} B/token) writer layer {a.hm_layer} read layers {hm.dm.layers}")
    params = [p for g in groups if not g.get("p4") for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    for g_ in opt.param_groups:
        g_["base_lr"] = g_["lr"]

    def set_lr(step):                       # E1: optional warmup + cosine decay (--sched const leaves lr untouched)
        f = lr_factor(step, a.steps, a.sched, a.warmup, a.lr_min_frac)
        if a.sched == "cosine":
            for g_ in opt.param_groups:
                g_["lr"] = g_["base_lr"] * f
        return f

    log(f"D13 stage 2: {a.encoder} -> {a.reader} | layers {lmap} | init {a.init} | trainable "
        f"{sum(p.numel() for p in params) / 1e6:.1f}M | train {len(train)} items / {len(tr_docs)} docs "
        f"{ {t: len(v) for t, v in pools.items()} } | held-out val {len(val)} items / {len(va_docs)} docs | "
        f"kl {a.kl} ce {a.ce} dist {a.dist_w} | delta_rank {a.delta_rank} reg {a.reg} mlp {'train' if train_mlp else 'frozen'}")
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    ret = None
    if a.ret_w > 0:                                   # E3 retrieval-aligned term (default off: nothing below runs)
        from xenc.backbone import text_ids as _tid
        from xenc.e3_retrieval import QKCap, body_start
        ret = {"S": body_start(tok, a.reader, SYSTEM_ANSWER), "open": _tid(tok, "<answer>"), "close": _tid(tok, "</answer>"),
               "layers": [int(x) for x in a.ret_layers.split(",")] if a.ret_layers else list(itf.layers),
               "tasks": set(x for x in a.ret_tasks.split(",") if x), "n": 0, "n_try": 0}
        assert set(ret["layers"]) <= set(itf.layers), ret["layers"]
        ret["tmap"] = {li: (li if a.ret_tlayer < 0 else a.ret_tlayer) for li in ret["layers"]}
        if a.ret_src == "text":
            ret["cap"] = QKCap(reader, sorted(set(ret["tmap"].values())))
        log(f"E3 retrieval term: src {a.ret_src} w {a.ret_w} topk {a.ret_topk} layers {ret['layers']} teacher layers "
            f"{ret['tmap']} tasks {sorted(ret['tasks']) or 'all'} body start S={ret['S']}")
    if a.val_attn_max_op:
        log(f"  val attention diagnostic: {len(attn_diag_items(val, a.n_val_attn, _op_len, a.val_attn_max_op))} items with "
            f"<= {a.val_attn_max_op} operator columns")
    hist = {"steps": [], "evals": []}
    best, t0 = -1e9, time.time()
    best_path = os.path.join(a.out, "d13_best.pt")
    ct_cache: dict = {}

    def run_eval(step):
        nonlocal best
        rows = []
        for it in val:
            did = _get(it, "doc_id")
            w = wrong_doc_id(did, val_order[did.split("_")[0]])
            rec = {}
            for c, d, rd in (("kv", did, True), ("wrong_doc", w, True), ("closed_tail", did, False)):
                if c == "closed_tail" and _get(it, "key") in ct_cache:
                    rec[c] = ct_cache[_get(it, "key")]
                    continue
                if d is None:
                    continue
                g, n = rt.gen(d, it, a.budget, read=rd)
                rec[c] = {"score": float(score(it, g)), "gen_tokens": n}
                if c == "closed_tail":
                    ct_cache[_get(it, "key")] = rec[c]
            rows.append(rec)
        s = summarize(rows, ("closed_tail", "kv", "wrong_doc"))
        diag = []
        for it in attn_diag_items(val, a.n_val_attn, _op_len, a.val_attn_max_op):
            try:
                diag.append(rt.s1_pass(_get(it, "doc_id"), it, onpolicy=True, grad=False, metrics=True))
            except torch.OutOfMemoryError:
                # diagnostic only (not the val score, not selection): with --mem_safe an OOM drops this item from the
                # attention-metric mean, logged and counted; without --mem_safe it still raises (old behaviour)
                if not a.mem_safe:
                    raise
                torch.cuda.empty_cache()
                diag_oom.append((step, str(_get(it, "key"))))
                log(f"  OOM in the val attention diagnostic on {_get(it, 'key')} ({_op_len(it)} operator columns); "
                    f"dropped from the diagnostic mean only (total {len(diag_oom)})")
        att = mean_terms(diag)
        r = {"step": step, "kv": s["kv"]["score"], "wrong_doc": s["wrong_doc"]["score"],
             "closed_tail": s["closed_tail"]["score"],
             "gap": s.get("kv_minus_wrong_doc", {}).get("paired_gap", float("nan")),
             "gap_se": s.get("kv_minus_wrong_doc", {}).get("se", float("nan")), "gen": s["kv"]["mean_gen_tokens"],
             "gen_wrong": s["wrong_doc"]["mean_gen_tokens"], "attn": att,
             "gates": [round(float(x), 4) for x in itf.g], "n": len(rows), "attn_n": len(diag),
             "drift": float(itf.delta_reg()) if itf.delta_rank else None,
             "e2": itf.e2_state() if hasattr(itf, "e2_state") else None,
             "tok": itf.tok_state() if hasattr(itf, "tok_state") else None,
             **({"l1": itf.l1_state()} if hasattr(itf, "l1_state") else {}),
             **({"r2": itf.r2_state()} if hasattr(itf, "r2_state") else {}),
             "ptr": itf.ptr_state() if hasattr(itf, "ptr_state") else None}
        if hasattr(itf, "ptr_state"):
            r["ptr_g_val"] = round(rt.R.ptr_gsum / max(rt.R.ptr_gn, 1), 5)     # mean gate over kv + wrong_doc decode steps
            rt.R.ptr_gsum, rt.R.ptr_gn = 0.0, 0
        if hasattr(itf, "ptr_layers") and 0 <= a.ptr_val_off_from <= step:      # P4 ablation: same ckpt, pointer off
            itf.ptr_on = False
            try:
                for it, rec in zip(val, rows):
                    g, n = rt.gen(_get(it, "doc_id"), it, a.budget, read=True)
                    rec["kv_ptroff"] = {"score": float(score(it, g)), "gen_tokens": n}
            finally:
                itf.ptr_on = True
            r["kv_ptroff"] = sum(x["kv_ptroff"]["score"] for x in rows) / max(len(rows), 1)
        if hasattr(itf, "seg_params") and 0 <= a.seg_val_off_from <= step:     # ADDR SEG ablation: same ckpt, plain operator
            itf.seg_on = False
            try:
                for it, rec in zip(val, rows):
                    g, n = rt.gen(_get(it, "doc_id"), it, a.budget, read=True)
                    rec["kv_segoff"] = {"score": float(score(it, g)), "gen_tokens": n}
            finally:
                itf.seg_on = True
            r["kv_segoff"] = sum(x["kv_segoff"]["score"] for x in rows) / max(len(rows), 1)
        if hasattr(itf, "seg_params"):
            r["seg"] = itf.seg_state()
        if hasattr(itf, "tok_layers") and 0 <= a.tok_val_off_from <= step:     # P3 ablation: same ckpt, channel off
            itf.tok_on = False
            try:
                for it, rec in zip(val, rows):
                    g, n = rt.gen(_get(it, "doc_id"), it, a.budget, read=True)
                    rec["kv_tokoff"] = {"score": float(score(it, g)), "gen_tokens": n}
            finally:
                itf.tok_on = True
            r["kv_tokoff"] = sum(x["kv_tokoff"]["score"] for x in rows) / max(len(rows), 1)
        per_task = {}
        for it, rec in zip(val, rows):
            per_task.setdefault(_get(it, "doc_id").split("_")[0], []).append(rec)
        r["per_task"] = {t: {c: sum(x[c]["score"] for x in v if c in x) / len(v) for c in ("kv", "wrong_doc", "closed_tail")}
                         for t, v in per_task.items()}
        if a.save_items:
            r["items"] = [[str(_get(it, "key")), rec["kv"]["score"], rec["wrong_doc"]["score"] if "wrong_doc" in rec else None,
                           rec["closed_tail"]["score"]] + ([rec["kv_tokoff"]["score"]] if "kv_tokoff" in rec else [])
                          + ([rec["kv_ptroff"]["score"]] if "kv_ptroff" in rec else [])
                          for it, rec in zip(val, rows)]
        if rc is not None and a.recon_val > 0:      # RECON probe on held-out val docs (teacher-free, fixed items)
            r["recon_val"] = recon_probe()
            log(f"  [recon val @ {step}] {r['recon_val']}")
        hist["evals"].append(r)
        crit = r["kv"] + (1e-3 * r["gap"] if math.isfinite(r["gap"]) else 0.0)
        if a.select_task:
            sel = [x for it, x in zip(val, rows) if _get(it, "doc_id").split("_")[0] in a.select_task.split(",")]
            crit = sum(x["kv"]["score"] for x in sel) / max(len(sel), 1)
        r["crit"] = crit
        tag = ""
        if step < a.select_min_step:          # sets the bar but is never selected: later points must BEAT it
            best = max(best, crit)
            tag = " (bar; not selectable)"
        elif crit > best:
            best = crit
            save_d13(itf, best_path, {"stage": "register", "step": step, "val": r, "args": vars(a)})
            save_cmp_as(best_path)
            tag = " *best*"
        log(f"  [val @ {step}] held-out docs n={r['n']}: kv {r['kv']:.3f} wrong {r['wrong_doc']:.3f} closed_tail "
            f"{r['closed_tail']:.3f} gap {r['gap']:+.3f} ± {r['gap_se']:.3f} gen {r['gen']:.1f}/{r['gen_wrong']:.1f} "
            f"gates {r['gates']}{tag}" + (f" | e2 {r['e2']}" if r.get("e2") else "")
            + (f" | tok {r['tok']}" if r.get("tok") else "")
            + (f" | l1 {r['l1']}" if r.get("l1") else "")
            + (f" | r2 {r['r2']}" if r.get("r2") else "")
            + (f" | kv_tokoff {r['kv_tokoff']:.3f}" if "kv_tokoff" in r else "")
            + (f" | ptr {r['ptr']} g_val {r['ptr_g_val']}" if r.get("ptr") else "")
            + (f" | kv_ptroff {r['kv_ptroff']:.3f}" if "kv_ptroff" in r else "")
            + (f" | kv_segoff {r['kv_segoff']:.3f}" if "kv_segoff" in r else ""))
        log(f"            per task " + " ".join(f"{t}: {v['kv']:.2f}/{v['wrong_doc']:.2f}/{v['closed_tail']:.2f}"
                                             for t, v in r["per_task"].items()) + f" | attn {fmt(att)}")

    def dump():
        with open(os.path.join(a.out, "history.json"), "w") as f:
            json.dump(hist, f, indent=1)

    ptr_log = {"g": [], "cpy": []}
    run_eval(0)
    dump()
    R = rt.R
    rt.ckpt = R.ckpt = bool(a.mem_safe)
    skipped = 0
    e2_d3: list = []
    e2_c2: list = []
    for step in range(1, a.steps + 1):
        lr_f = set_lr(step - 1)
        opt.zero_grad(set_to_none=True)
        acc = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "dist": 0.0, "klc": 0.0, "ret": 0.0, "n": 0}
        step_docs: list = []                        # RECON: this step's training docs (operators already cached)
        for _ in range(a.batch):
            p1 = None
            if p1_items and rng.random() < a.p1_p:          # P1 corpus item: uniform doc, then uniform question
                if p1_w is not None:                        # P1-hard: proportional to the base's KL
                    p1 = it = rng.choices(p1_items, p1_w)[0]
                    d_ = it["doc_id"]
                else:
                    d_ = p1_dids[rng.randrange(len(p1_dids))]
                    p1 = it = p1_by[d_][rng.randrange(len(p1_by[d_]))]
                did = d_
                if did not in rt.docs:
                    rt.docs[did] = open(p1_paths[did]).read()
                target = it["target"]
            else:
                t = rng.choices(tasks, weights)[0]
                it = pools[t][rng.randrange(len(pools[t]))]
                did = _get(it, "doc_id")
                target = gold_text(it)
                step_docs.append(did)
            tt = text_ids(tok, target)
            m = len(tt)
            if p1 is not None:
                if tt != p1_T[p1["key"]]["tgt"].tolist():
                    p1_stats["skip_tok"] += 1
                    continue
                if p1["n_tids"] > a.max_tokens:
                    p1_stats["skip_long"] += 1
                    continue
            tids = build_chat(tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, rt.docs[did])) + tt
            if len(tids) > a.max_tokens:
                skipped += 1
                continue
            R.mode = "off"
            ret_ok = False
            if ret is not None and (not ret["tasks"] or did.split("_")[0] in ret["tasks"]):
                ret["n_try"] += 1
                from xenc.e3_retrieval import answer_content_rows
                tt_chk, loc = answer_content_rows(tok, target)     # (m+1)-window rows predicting the answer content
                Nr = len(rt.pre(did)) - a.tail
                ret_ok = (tt_chk == tt and len(loc) > 0 and tids[:Nr] == rt.pre(did)[:Nr] and Nr > ret["S"])
                if ret_ok:
                    if a.ret_src == "gold":
                        from xenc.e3_retrieval import gold_mask
                        from xenc.state_skill import doc_prefix_text
                        gm = gold_mask(tok, doc_prefix_text(tok, a.reader, rt.docs[did], SYSTEM_ANSWER), rt.docs[did],
                                       _get(it, "answers"), Nr, ret["S"])
                        ret_ok = gm is not None and not _get(it, "choices")
                if ret_ok and a.ret_src == "text":
                    ret["cap"].keep, ret["cap"].rec = m + 1, {}
            with torch.no_grad():
                from register_state import suffix_rows
                K = m + 1
                if a.kl_ctx > 0:
                    K = max(m + 1, min(suffix_rows(tids, rt.rows(did, it, tt), cap=a.kl_ctx_rows + m + 1), a.kl_ctx_rows + m + 1))
                hid_ok = False
                if hid is not None and p1 is None:
                    hid["n_try"] += 1
                    rows_h = rt.rows(did, it, tt)
                    from xenc.addr_hid import common_suffix
                    hid_k = common_suffix(tids, rows_h)          # rows aligned with the text pass (Qasper: all but 1)
                    hid_ok = hid_k >= m + 1
                    if hid_ok:
                        hid["cap"].T, hid["cap"].detach, hid["cap"].rec = hid_k, True, {}
                try:
                    t_log = (None if p1 is not None and len(p1_stats["chk"]) >= a.p1_check   # P1: stored teacher
                             else kept_logits(reader, torch.tensor([tids]), K).detach())
                except torch.OutOfMemoryError:
                    if not a.mem_safe:
                        raise
                    torch.cuda.empty_cache()
                    skipped += 1
                    log(f"  OOM in the text-teacher pass on {did} ({len(tids)} tokens); item skipped")
                    continue
                finally:
                    if ret is not None and "cap" in ret:
                        ret["cap"].keep = None
                    if hid is not None:
                        hid["cap"].T = None
                hid_t = hid["cap"].rec if hid_ok else None
                if hid is not None:
                    hid["cap"].rec = {}
                if ret_ok and a.ret_src == "text":
                    from xenc.e3_retrieval import mixture_logp
                    t_lp = {li: mixture_logp(ret["cap"].logits(te), loc, ret["S"], Nr) for li, te in ret["tmap"].items()}
                    ret["cap"].rec = {}
            rows = rt.rows(did, it, tt)
            try:
                op, n_op, _ = rt.operator(did)
            except torch.OutOfMemoryError:
                if not a.mem_safe:
                    raise
                rt._ops.clear()
                if hasattr(rt, "_cols"):                  # NOTES runtime: note-column cache too
                    rt._cols.clear()
                torch.cuda.empty_cache()
                skipped += 1
                log(f"  OOM in the operator prefill on {did}; item skipped")
                continue
            rec = rt.teacher(did, rows) if (a.dist_w > 0 or (ret_ok and a.ret_src == "enc")) else None
            y2 = None
            if c2t is not None:
                try:
                    y2 = c2t.targets(did, rows, str(_get(it, "key")))
                except torch.OutOfMemoryError:
                    if not a.mem_safe:
                        raise
                    c2t.rt._ops.clear()
                    torch.cuda.empty_cache()
                    skipped += 1
                    log(f"  OOM in the C2 teacher on {did}; item skipped")
                    continue
            if ret_ok:
                assert n_op == Nr, (n_op, Nr)
                srows = [len(rows) - (m + 1) + i for i in loc]
                if a.ret_src == "enc":
                    from xenc.d13_read import teacher_logits
                    from xenc.e3_retrieval import mixture_logp
                    with torch.no_grad():
                        t_lp = {li: mixture_logp(teacher_logits(rec[lmap[li]]["q"], op[lmap[li]][0], rec[lmap[li]]["k"],
                                                                itf.att[str(li)].scaling), srows, ret["S"], Nr)
                                for li in ret["layers"]}
                elif a.ret_src == "gold":
                    gm = gm.to(rt.dev)
            R.load(op, n_op)
            R.mode, R.need_logits, R.rec = "e2e", (a.dist_w > 0 or ret_ok), {}
            p6_rows = None
            if p6 is not None and p1 is None and not _get(it, "choices"):
                p6_rows = p6_plan(p6, did, it, target, tt, m, rows, step)
                if p6_rows is not None and p6_rows["alpha"] > 0 and p6_rows["plan"]:
                    itf.tok_p_override = p6_override(p6_rows, itf)
            if hid_t is not None:
                hid["cap"].T, hid["cap"].detach, hid["cap"].rec = len(rows), False, {}
            try:
                with torch.enable_grad():
                    s_all = kept_logits(reader, torch.tensor([rows]), K)
                    if hid is not None:
                        hid["cap"].T = None
                    s_log = s_all[-(m + 1):]
                    if p1 is None:
                        t_ans = t_log[-(m + 1):]
                        kl = kl_tempered(t_ans[:-1], s_log[:-1], a.kl_temp)
                    else:
                        from p1_data import topk_kl
                        T_ = p1_T[p1["key"]]
                        kl = topk_kl(T_["ids"], T_["lp"], T_["rest"], s_log[:-1])
                        if t_log is not None:          # --p1_check: stored top-k vs the online text teacher
                            with torch.no_grad():
                                tf = torch.log_softmax(t_log[-(m + 1):-1].float(), -1)
                                ex = kl_teacher_student(t_log[-(m + 1):-1], s_log[:-1])
                                dlp = (tf.gather(-1, T_["ids"].to(tf.device).long()) - T_["lp"].to(tf.device).float()).abs().max()
                            p1_stats["chk"].append((float(ex), float(kl), float(dlp)))
                            log(f"  [P1 check {len(p1_stats['chk'])}] {p1['key']} m={m}: exact KL {float(ex):.5f} | top-k KL "
                                f"{float(kl):.5f} | max |d logp| stored vs online {float(dlp):.4f}")
                    ce = answer_ce(s_log, rows[-m:])
                    if ptr_pp and R.ptr_out is not None:                  # P4: gate on the answer rows (logged)
                        ptr_log["g"].append(float(R.ptr_out["g"][-(m + 1):-1].detach().mean()))
                        if a.ptr_copy_w > 0 and p1 is None and not _get(it, "choices"):
                            from xenc.e3_retrieval import answer_content_rows
                            from xenc.p4_ptr import copy_nll
                            tt_chk, loc = answer_content_rows(tok, target)
                            if tt_chk == tt and loc:
                                gold = torch.tensor([tt[i] for i in loc], device=R.ptr_out["Pc"].device)
                                _i0 = R.itf.ptr_shifts.index(0) if 0 in R.itf.ptr_shifts else 0
                                _pool = R.ptr_tab["tok"][_i0]
                                if R.itf.ptr_nosp:          # first content token: its space / '>'-merged variants count
                                    _pool = torch.cat([_pool, R.ptr_tab["tok_ns"][_i0], R.ptr_tab["tok_gm"][_i0]])
                                in_doc = torch.isin(gold, _pool)
                                if bool(in_doc.any()):
                                    Pc_ = R.ptr_out["Pc"][-(m + 1):][loc]
                                    cpy = copy_nll(Pc_, gold, in_doc)
                                    ce = ce + (a.ptr_copy_w / a.ce if a.ce else 0.0) * cpy.to(ce.device)
                                    ptr_log["cpy"].append(float(cpy))
                    klc = torch.zeros((), device=kl.device)
                    if K > m + 1:
                        klc = kl_tempered(t_log[: K - m - 1], s_all[: K - m - 1], a.kl_temp)
                    ce_w = a.ce
                    if p1 is not None:
                        ce_w = a.p1_ce if (p1["agree"] and p1["conf"] >= a.p1_ce_conf) else 0.0
                        p1_stats["n"] += 1
                        p1_stats["kl"] += float(kl)
                        p1_stats["ce_used"] += int(ce_w > 0)
                    loss = a.kl * kl + ce_w * ce + a.kl_ctx * klc
                    dist = torch.zeros((), device=kl.device)
                    if a.dist_w > 0:
                        with torch.no_grad():
                            yref = {li: (rec[e]["out"].float() if B0 is None else rec[e]["out"].float() @ B0[li].t())
                                    for li, e in lmap.items()}
                        terms = rt.attn_terms(rec, op, n_op, yref)
                        dist = torch.stack([x["kl"] + a.rel_w * x["rel"] for x in terms.values()]).mean()
                        loss = loss + a.dist_w * dist.to(loss.device)
                        if y2 is not None:             # E2: C2 self-read feature matching
                            from xenc.e2_c2 import c2_loss
                            c2l = c2_loss(R.rec, y2)
                            loss = loss + a.c2_w * c2l.to(loss.device)
                            e2_c2.append(float(c2l))
                        if getattr(itf, "rs", None):   # E2 rs paths: their own distillation term (new addressing)
                            d3 = itf.extra_dist(R.rec, rec, op, n_op, a.rel_w)
                            loss = loss + a.dist_w * d3.to(loss.device)
                            e2_d3.append(float(d3))
                    rterm = torch.zeros((), device=kl.device)
                    if ret_ok:
                        from xenc.e3_retrieval import kl_rows, mixture_logp, sharpen, span_nll
                        rt_ = []
                        for li in ret["layers"]:
                            s_lp = mixture_logp(R.rec[li]["logits"], srows, ret["S"], Nr)
                            rt_.append(span_nll(s_lp, gm) if a.ret_src == "gold"
                                       else kl_rows(sharpen(t_lp[li].to(s_lp.device), a.ret_topk), s_lp))
                        rterm = torch.stack([x.to(kl.device) for x in rt_]).mean()
                        assert rterm.requires_grad, "E3 retrieval term carries no gradient"
                        loss = loss + a.ret_w * rterm
                        ret["n"] += 1
                    if hid_t is not None:              # ADDR HID: write-back vs the 9B text attention output
                        from xenc.addr_hid import hid_loss, hid_rows
                        _T = len(rows)
                        _ys = {li: v[_T - hid_k:] for li, v in hid["cap"].rec.items()}
                        _rr = [r_ - (_T - hid_k) for r_ in hid_rows(_T, a.tail, m, a.hid_rows) if r_ >= _T - hid_k]
                        hterm = hid_loss(_ys, hid_t, _rr)
                        assert hterm.requires_grad, "HID term carries no gradient"
                        loss = loss + a.hid_w * hterm.to(loss.device)
                        hid["n"] += 1
                        hid["vals"].append(float(hterm))
                        hid["cap"].rec = {}
                    if p6 is not None and a.tok_aux_w > 0 and p6_rows is not None and R.tok_z:
                        aux = p6_aux(p6, R.tok_z, p6_rows)
                        if aux is not None:
                            loss = loss + a.tok_aux_w * aux.to(loss.device)
                            p6["aux"].append(float(aux))
                    if a.delta_rank and a.reg > 0:
                        loss = loss + a.reg * itf.delta_reg().to(loss.device)
                    if not math.isfinite(float(loss)):
                        skipped += 1
                        continue
                    (loss / a.batch).backward()
            except torch.OutOfMemoryError:
                if not a.mem_safe:
                    raise
                R.rec, loss, kl, ce, dist, klc, s_all, s_log, rec, terms = {}, None, None, None, None, None, None, None, None, None
                torch.cuda.empty_cache()
                skipped += 1
                log(f"  OOM on {did} ({len(rows)} rows, {n_op} operator tokens); item skipped")
                continue
            finally:
                R.mode, R.rec = "off", {}
                if hasattr(R, "tok_z"):                     # P6: drop this forward's token read / adapter outputs
                    R.tok_r, R.tok_z = None, {}
                    itf.tok_p_override = None
                if hid is not None:
                    hid["cap"].T, hid["cap"].rec = None, {}
            for k, v in (("loss", loss), ("kl", kl), ("ce", ce), ("dist", dist), ("klc", klc), ("ret", rterm)):
                acc[k] += float(v)
            acc["n"] += 1
        if rc is not None and a.recon_w > 0 and step_docs:   # RECON aux items (separate rng; after the main items)
            w_rc = _rc.ramp_weight(step, a.recon_w, a.recon_ramp)
            for j_ in range(a.recon_n):
                rc["n_try"] += 1
                d_rc = step_docs[rc["rng"].randrange(len(step_docs))]
                it_rc = recon_make(d_rc, rc["rng"], f"{d_rc}_rc{step}_{j_}")
                o_rc = recon_loss(it_rc, w_rc) if it_rc is not None else None
                if o_rc is None:
                    rc["skip"] += 1
                    continue
                o_rc["step"] = step
                rc["hist"].append(o_rc)
                if rc["shown"] < 8:
                    rc["shown"] += 1
                    log(f"  [recon item {rc['shown']}] {it_rc['key']} {o_rc['kind']} | Q {it_rc['question'][:160]!r} | gold "
                        f"{it_rc['answers'][0][:120]!r} | kl {o_rc['kl']:.3f} ce {o_rc['ce']:.3f} acc {o_rc['acc']:.2f} "
                        f"teacher acc {o_rc['t_acc']:.2f} w {w_rc:.4f}")
        if acc["n"]:
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if hyb_pp:
                gnh = torch.nn.utils.clip_grad_norm_(hyb_pp, 1.0)
                if step <= 3 or step % 50 == 0:     # correctness (HYB): gates / decoders receive gradient and move
                    with torch.no_grad():
                        dr = {k: round(float((hyb.cmp.U[k] - hyb_U0[k]).norm() / hyb_U0[k].norm().clamp_min(1e-9)), 6)
                              for k in hyb.cmp.U if int(k) >= 100}
                    log(f"  [hyb live @ {step}] grad pre-clip {float(gnh):.5f} | g grad "
                        f"{[round(float(x), 6) for x in hyb.dm.g.grad] if hyb.dm.g.grad is not None else None} | gates "
                        f"{[round(float(x), 4) for x in hyb.dm.g]} | rel drift U {dr}")
            if hm_pp:
                gnm = torch.nn.utils.clip_grad_norm_(hm_pp, 1.0)
                if step <= 3 or step % 50 == 0:     # correctness (HYB-HM): V decoders move first, then K / writer follow
                    with torch.no_grad():
                        nv = {li: round(float(hm.dm.dv[str(li)].weight.norm()), 5) for li in hm.dm.layers}
                        dw = (round(float((hm.dm.writer.weight - hm_W0).norm() / hm_W0.norm()), 6)
                              if hm_W0 is not None else None)
                    log(f"  [hm live @ {step}] grad pre-clip {float(gnm):.5f} | |Uv| {nv} | writer rel drift {dw}")
            if cmp_pp:
                gnc = torch.nn.utils.clip_grad_norm_(cmp_pp, 1.0)
                if step <= 3 or step % 50 == 0:     # correctness (CMP): decoders receive gradient and move
                    with torch.no_grad():
                        dr = {k: round(float((cmp.U[k] - cmp_U0[k]).norm() / cmp_U0[k].norm().clamp_min(1e-9)), 6)
                              for k in cmp.U}
                    log(f"  [cmp live @ {step}] decoder grad pre-clip {float(gnc):.5f} | rel drift U {dr} | |b| "
                        f"{[round(float(v.norm()), 5) for v in cmp.b.values()]}")
            if ptr_pp:
                gnp = torch.nn.utils.clip_grad_norm_(ptr_pp, 1.0)
                if step <= 3 or step % 25 == 0:
                    # correctness (P4): the gate amplitude gets gradient from step 1; the rest once a > 0
                    P_ = itf.PTR
                    gr = {k: (round(float(P_[k].grad.norm()), 6) if P_[k].grad is not None else None) for k in P_}
                    gl, cl = ptr_log["g"][-50:], ptr_log["cpy"][-50:]
                    log(f"  [ptr live @ {step}] grad |.| {gr} (pre-clip total {float(gnp):.4f}) | train gate on answer "
                        f"rows mean last {len(gl)}: {sum(gl) / max(len(gl), 1):.5f}"
                        + (f" | copy nll {sum(cl) / len(cl):.4f}" if cl else "") + f" | state {itf.ptr_state()}")
            if (newl or a.mlp_widen) and (step <= 3 or step % 25 == 0):
                # correctness: the NEW parts are live (non-zero gradients; gates logged) -- lever 2a / 2b
                live = []
                if newl:
                    gg = theta.grad if a.train_g and a.lr_g_new > 0 else itf.g.grad
                    live.append("new-gate grad " + str([round(float(gg[itf._gi[li]]), 6) if gg is not None else None
                                                      for li in sorted(newl)]))
                    live.append("new-A grad |.| " + str([round(float(itf.A[str(li)].weight.grad.norm()), 6)
                                                       if itf.A[str(li)].weight.grad is not None else None
                                                       for li in sorted(newl)]))
                if a.mlp_widen:
                    old_h = int(getattr(itf, "mlp_hidden_init", 1024))
                    live.append("widened-MLP new-unit out-grad |.| " + str([round(float(m[2].weight.grad[:, old_h:].norm()), 6)
                                                                           if m[2].weight.grad is not None else None
                                                                           for m in itf.M.values()]))
                log(f"  [live @ {step}] " + " | ".join(live) + f" | gates {[round(float(x), 4) for x in itf.g]}")
            if cap_pp and (step <= 3 or step % 25 == 0):
                # correctness (CAP): the new parts get non-zero gradients; MB output norm shows they moved off zero
                lv = []
                if "MB" in cap_pp:
                    lv.append("MB out-grad |.| " + str([round(float(m[2].weight.grad.norm()), 6) if m[2].weight.grad is not None
                                                       else None for m in itf.MB.values()]))
                    lv.append("MB out-w |.| " + str([round(float(m[2].weight.norm()), 5) for m in itf.MB.values()]))
                if "att" in cap_pp:
                    lv.append("att q/o grad |.| " + str([(round(float(m.q_proj.weight.grad.norm()), 6) if m.q_proj.weight.grad is not None else None,
                                                         round(float(m.o_proj.weight.grad.norm()), 6) if m.o_proj.weight.grad is not None else None)
                                                        for m in itf.att.values()]))
                log(f"  [cap live @ {step}] " + " | ".join(lv))
            if hasattr(itf, "e2_params") and (step <= 3 or step % 25 == 0):
                if e2_c2:
                    log(f"  [e2 c2 @ {step}] feature-matching relMSE mean last {min(len(e2_c2), 50)}: "
                        f"{sum(e2_c2[-50:]) / len(e2_c2[-50:]):.4f} (first {e2_c2[0]:.4f}) | cache hit/miss {c2t.n_hit}/{c2t.n_miss}")
                if e2_d3:
                    log(f"  [e2 rs dist @ {step}] mean last {min(len(e2_d3), 50)}: {sum(e2_d3[-50:]) / len(e2_d3[-50:]):.4f}")
                # correctness (E2): the mixing weights get non-zero gradients and move; B2 grads when trained
                ep = itf.e2_params()
                log(f"  [e2 live @ {step}] mix grad |.| {[round(float(p.grad.norm()), 6) if p.grad is not None else None for p in ep['mix']]}"
                    f" | B2/B3 grad |.| {[round(float(p.grad.norm()), 6) if p.grad is not None else None for p in ep['B2']]}"
                    f" | A3 grad |.| {[round(float(p.grad.norm()), 6) if p.grad is not None else None for p in ep.get('A3', [])[:2]]}"
                    f" | tr grad |.| {[round(float(p.grad.norm()), 6) if p.grad is not None else None for p in ep.get('tr', [])[:4]]}"
                    f" | state {itf.e2_state()}")
            if hasattr(itf, "seg_params") and (step <= 3 or step % 25 == 0):
                # correctness (ADDR SEG): the zero-init maps get non-zero gradients and move
                log(f"  [seg live @ {step}] grad |.| " + str({k: (round(float(p.grad.norm()), 6) if p.grad is not None else None)
                                                         for k, p in list(itf.SEG.items())[:8]}) + f" | state {itf.seg_state()}")
            if hid is not None and (step <= 3 or step % 25 == 0):
                hv = hid["vals"][-50:]
                log(f"  [hid live @ {step}] term on {hid['n']}/{hid['n_try']} items | mean last {len(hv)}: "
                    f"{sum(hv) / max(len(hv), 1):.4f} (first {hid['vals'][0] if hid['vals'] else float('nan'):.4f})")
            if hasattr(itf, "l1_params") and itf.l1_params() and (step <= 3 or step % 25 == 0):
                log(f"  [l1 live @ {step}] grad |.| {[round(float(p.grad.norm()), 6) if p.grad is not None else None for p in itf.l1_params()]}"
                    f" | state {itf.l1_state()}")
            if hasattr(itf, "r2_params") and (step <= 3 or step % 25 == 0):
                # correctness (R2): the new parameters get non-zero gradients and move
                log(f"  [r2 live @ {step}] grad |.| {itf.r2_named_grads()} | state {itf.r2_state()}")
            if p6 is not None and a.tok_floor > 0 and step == a.tok_floor_from:
                with torch.no_grad():
                    itf.TOK["floor_on"]["v"].fill_(1.0)
                log(f"  [P6] token-gate floor {a.tok_floor} switched on at step {step}")
            if p6 is not None and (step <= 3 or step % 25 == 0):
                log(f"  [P6 @ {step}] oracle α {p6_alpha(a, step):.3f} | gold plan on {p6['n_plan']}/{p6['n_try']} items | "
                    f"aux mean last 50 {sum(p6['aux'][-50:]) / max(len(p6['aux'][-50:]), 1):.4f}")
            if hasattr(itf, "tok_params") and (step <= 3 or step % 25 == 0):
                # correctness (P3): U receives non-zero gradients and moves; gates / write-back ratio logged
                tpp = itf.tok_params()
                log(f"  [tok live @ {step}] U grad |.| {[round(float(p.grad.norm()), 6) if p.grad is not None else None for p in tpp['U']]}"
                    f" | gt grad {[round(float(p.grad), 6) if p.grad is not None else None for p in tpp['gt']]}"
                    f" | state {itf.tok_state()}")
            opt.step()
            sync_g()
            r = {k: (v / acc["n"] if k != "n" else v) for k, v in acc.items()}
            r.update(step=step, grad_norm=float(gn), t=round(time.time() - t0, 1), lr_f=lr_f)
            hist["steps"].append(r)
            if ret is not None and (step % 25 == 0 or step <= 3):
                log(f"  [E3 ret] step {step}: term applied to {ret['n']}/{ret['n_try']} eligible items so far "
                    f"(src {a.ret_src}); last-batch ret {r.get('ret', 0.0):.4f}")
            if step % 25 == 0 or step <= 3:
                w = hist["steps"][-25:]
                avg = {k: sum(x[k] for x in w) / len(w) for k in ("loss", "kl", "ce", "dist", "klc", "ret")}
                log(f"  [step {step}/{a.steps}] (avg last {len(w)}) loss {avg['loss']:.4f} kl {avg['kl']:.4f} "
                    f"ce {avg['ce']:.4f} dist {avg['dist']:.4f} klc {avg['klc']:.4f} ret {avg['ret']:.4f} |g| {r['grad_norm']:.3f} gates "
                    f"{[round(float(x), 3) for x in itf.g]} lr_f {lr_f:.3f} kl_temp {a.kl_temp} ({(time.time() - t0) / step:.2f}s/step)")
                if p1_items:
                    log(f"  [P1 @ {step}] corpus items used {p1_stats['n']} (mean top-k KL {p1_stats['kl'] / max(p1_stats['n'], 1):.4f}, "
                        f"CE on {p1_stats['ce_used']}) | skipped: target re-tokenization {p1_stats['skip_tok']}, over-length "
                        f"{p1_stats['skip_long']}")
        if rc is not None and a.recon_w > 0 and (step <= 3 or step % 25 == 0):
            hv = rc["hist"][-50:]
            byk = {}
            for x in hv:
                byk.setdefault(x["kind"], []).append(x)
            mk = lambda v, k: sum(x[k] for x in v) / max(len(v), 1)              # noqa: E731
            log(f"  [recon live @ {step}] w {_rc.ramp_weight(step, a.recon_w, a.recon_ramp):.4f} | items {len(rc['hist'])}/"
                f"{rc['n_try']} (skipped {rc['skip']}) | last {len(hv)}: loss {mk(hv, 'loss'):.4f} kl {mk(hv, 'kl'):.4f} ce "
                f"{mk(hv, 'ce'):.4f} dist {mk(hv, 'dist'):.4f} | " + " | ".join(
                    f"{k} n {len(v)} acc {mk(v, 'acc'):.3f} em {mk(v, 'em'):.3f} teacher acc {mk(v, 't_acc'):.3f}"
                    for k, v in sorted(byk.items())))
            hist["recon"] = rc["hist"]
        if step % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if step % a.eval_every == 0 or step == a.steps:
            run_eval(step)
            dump()
        ck = periodic_ckpt_path(a.out, step, a.save_every)
        if ck:                                     # DS --save_every: fixed-step checkpoints for a sweep
            sync_g()
            save_d13(itf, ck, {"stage": "register", "step": step, "args": vars(a)})
            save_cmp_as(ck)
            log(f"  saved {ck}")
    save_d13(itf, os.path.join(a.out, "d13_last.pt"), {"stage": "register", "step": a.steps, "args": vars(a)})
    save_cmp_as(os.path.join(a.out, "d13_last.pt"))
    dump()
    if torch.cuda.is_available():
        log(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if p1_items:
        hist["p1_stats"] = {k: v for k, v in p1_stats.items()}
        log(f"P1 stats: {hist['p1_stats']}")
        dump()
    log(f"done: {a.steps} steps, skipped {skipped}, best val crit {best:.4f} -> {best_path} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
