"""D20: per-document registration of the paper-form read (2B encoder operator -> 9B reader, D13 read), then the judge.

Every judged document (and every document used as a wrong_doc) is REGISTERED before it is read:
  base     the shared D13 interface (outputs/D13_reg_w10/d13_best.pt), frozen
  per-doc  xenc.d20_reg.DocAdapter (--parts: dA, dB, bias, gate, kv), zero-init = base; trained for up to
           max(--budgets) steps; a snapshot is kept at every budget (0 = kv_base)
  signals  label-free, from the document only (registration never sees a dataset question or answer):
           (a) --dist_w  stage-1 distillation: the encoder's own attention / read-out over the same rows
           (b) --kl      KL(text teacher ‖ student) on the answer tokens; teacher = the 9B reading the document as text
           (c) --ce      CE on the text teacher's greedy answer (pseudo-label)
           items = D20 self-questions (scripts/d20_selfq.py, written from the document) + --recite recitation prompts
           (quoted document excerpt -> "what follows"), both answered by the text teacher.
  guard    xenc.d20_reg.assert_no_test_questions(registration items, every question string of the task) — a hard
           assertion before any step.

Phases (--phase register,eval):
  register  per document: text-teacher targets (one 9B prefill of the document, shared by all items), encoder
            teacher rows, then the registration steps -> <out>/adapters/<doc_id>.pt (resumable; --shard/--nshard)
  eval      the judge on the selected items, for every --eval_budgets b: kv (own registered adapter), wrong_doc
            (another document's operator AND its registered adapter), plus closed_tail / text (once).

    python scripts/register_d20.py --split_dir data/longhealth_split --fold 0 --n 0 \
        --selfq outputs/D20_selfq/longhealth.jsonl --parts dA,gate --out outputs/D20_lh_f0_dA

REUSABLE ENTRY POINT (other directions judge through this; no unregistered kv numbers):
  any D13-format interface: --interface <ckpt> --reader <hub> --encoder <hub> --B <encB maps> --layers <reader layers>
  D12p module (27B reader): add --runtime d12p [--reader_mem 27GiB,27GiB --enc_device cuda:1 --itf_device cuda:1]
  any split: --split_dir data/<task>_split --split test [--fold k] [--n N] [--wrong_map json] [--max_tokens 60000]
  self-questions first: python scripts/d20_selfq.py --split_dir data/<task>_split --docs test:0 --out <file>
  (self-questions depend only on the 9B reader and the document text, so outputs/D20_selfq/<task>.jsonl is reusable
  by any interface whose teacher is the 9B; for another reader regenerate them with --reader).
  Report kv@<steps> (registered), kv@0 is the same adapter at step 0 = the base interface WITHOUT registration.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
import zlib
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402


# ============================================================================ items
def judge_items(a):
    """The judged items exactly as eval_d13.py selects them, plus the doc order used for wrong_doc."""
    items = load_split_items(a.split_dir, a.split)
    docs_all = sorted({_get(i, "doc_id") for i in items})
    if a.max_docs or a.skip_docs:
        keep = set(docs_all[a.skip_docs: a.skip_docs + a.max_docs if a.max_docs else None])
        items = [i for i in items if _get(i, "doc_id") in keep]
        docs_all = sorted(keep)
    if a.fold is not None:
        items = [i for i in items if (_get(i, "meta") or {}).get("fold") == a.fold]
    if a.n and len(items) > a.n:
        items = random.Random(a.seed).sample(items, a.n)
    return items, docs_all


def wrong_of(did, docs_all, wmap):
    from eval_doc import wrong_doc_id
    return wmap[did] if did in wmap else wrong_doc_id(did, docs_all)


def recite_items(tok, did, text, n, max_doc_tokens, seed, cont_tokens=0):
    """n recitation prompts at evenly spaced positions of the (operator-covered) document end."""
    if n <= 0:
        return []
    ids = tok(text, add_special_tokens=False)["input_ids"][-max_doc_tokens:]
    out, L = [], len(ids)
    rng = random.Random(seed)
    for j in range(n):
        lo = int((j + rng.random()) * L / n)
        lo = max(0, min(lo, L - 64))
        snip = tok.decode(ids[lo: lo + 24]).strip().replace('"', "'")
        if len(snip) < 20:
            continue
        cont = tok.decode(ids[lo + 24: lo + 24 + cont_tokens]).strip() if cont_tokens else ""
        if cont_tokens:     # whole-document signal: teacher-forced KL on the TRUE continuation (document text only)
            q = f"Quote the text of the document that comes right after this excerpt:\n\"{snip}\""
        else:
            q = f"In the document, which words come right after this excerpt?\n\"{snip}\""
        out.append({"key": f"{did}_d20r{j}", "doc_id": did, "question": q, "choices": None, "answers": [""],
                    "correct_label": None, "task": "d20recite", "hop": None,
                    "meta": {"d20recite": True, "cont": cont}})
    return out


def shuffle_choices(items, k, include_nli=False):
    """Each free-order MCQ self-question -> k copies with randomly permuted options (seeded by the item key);
    include_nli also permutes the 3-way statement-verification options (no fixed label-letter prior is taught)."""
    if k <= 0:
        return items
    out = []
    for it in items:
        tmpl = (it.get("meta") or {}).get("template")
        if not it.get("choices") or (tmpl == "nli" and not include_nli):
            out.append(it)
            continue
        rng = random.Random(zlib.crc32(it["key"].encode()))
        labs = [c["label"] for c in it["choices"]]
        for j in range(k):
            perm = list(it["choices"])
            rng.shuffle(perm)
            ch = [{"label": L, "text": c["text"]} for L, c in zip(labs, perm)]
            out.append(dict(it, key=f"{it['key']}_p{j}", choices=ch, correct_label=None))
    return out


class DocSkipped(RuntimeError):
    """F1 --oom_skip: a document whose registration hit too many CUDA OOMs; it gets no adapter and the judge drops
    every item whose document or wrong_doc is unregistered (from every condition)."""


def _base_key(k: str) -> str:
    return re.sub(r"_p\d+$", "", str(k))


def oom_events(out: str) -> list:
    p = os.path.join(out, "oom_events.jsonl")
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def oom_event(a, did, key, where):
    """F1 --oom_exit: record the OOM (doc, base item key, phase) and END THE PROCESS at once (exit 75), touching no
    CUDA state: a caught OOM can leave the CUDA context corrupted (later 'illegal memory access'). The caller re-runs
    the same command; resume then drops the recorded item up front (registration), or skips it in every judged
    condition (eval), in a fresh process."""
    if not getattr(a, "oom_exit", 0):
        return
    with open(os.path.join(a.out, "oom_events.jsonl"), "a") as f:
        f.write(json.dumps({"doc_id": did, "key": _base_key(key), "where": where, "t": time.strftime("%H:%M:%S")}) + "\n")
    log(f"OOM ({where}) on {did} / {key}: recorded in oom_events.jsonl; exiting 75 for a clean restart (--oom_exit)")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(75)


# ============================================================================ runtime
class D20:
    def __init__(self, a):
        from xenc.d20_reg import DocAdapter
        if a.runtime == "d12p":                  # D12p module copy (27B reader: --reader_mem / --enc_device / --itf_device)
            from fit_d12p import D13Rt, build
        else:
            from fit_d13 import D13Rt, build
        self.a = a
        reader, enc, tok, itf, lmap = build(a, init=a.interface)
        if itf.delta_rank:
            itf.merge_delta()
        self.reader, self.enc, self.tok, self.itf, self.lmap = reader, enc, tok, itf, lmap
        self.rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens)
        self.rt.ckpt = self.rt.R.ckpt = bool(a.mem_safe)   # checkpoint the e2e read + distillation terms (4B encoder)
        self.B0 = {li: itf.B[str(li)].detach().clone() for li in lmap}
        self.ad = DocAdapter(itf, [p for p in a.parts.split(",") if p], a.rank, a.kv_rank, a.seed)
        self.n_check = {"kept": 0, "dropped": 0}
        self.n_oom = {"teacher": 0, "step": 0, "holdout": 0, "eval": 0}

    # ------------------------------------------------------------------ teacher targets
    @torch.no_grad()
    def targets(self, did, items):
        """Text teacher (9B with the document in context): greedy answer, its logits on the answer tokens, its
        confidence; encoder teacher over the student's rows. One document prefill shared by all items."""
        from register_state import greedy, kept_logits
        from xenc.backbone import text_ids
        rt, R = self.rt, self.rt.R
        R.mode = "off"
        if self.a.enc_only:                     # ENCODER-ONLY registration: the reader never reads the document text
            res = []
            for it in items:
                rows = rt.rows(did, it)
                rec = {e: {k: v for k, v in r.items() if k in ("q", "k", "out")} for e, r in rt.teacher(did, rows).items()}
                res.append({"it": it, "target": "", "tt": [], "t_log": None, "conf": 1.0, "rows": rows, "rec": rec})
            return res
        pre = rt.pre(did)
        out = self.reader(input_ids=torch.tensor([pre], device=rt.dev), use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        del out
        res = []
        for it in items:
            try:
                suf = rt.suf(it)
                cont = (it.get("meta") or {}).get("cont")
                if cont:                                # recitation: the document's own continuation is the target
                    target = f"<answer>{cont}</answer>"
                else:
                    g, _ = greedy(self.reader, self.tok, self.a.reader, suf, self.a.ans_budget, cache=copy.deepcopy(cache))
                    if "<answer>" not in g:
                        continue
                    target = g[g.index("<answer>"):].split("</answer>")[0] + "</answer>"
                    want = (it.get("meta") or {}).get("intended")
                    if self.a.check_label and want and it.get("choices"):
                        from xenc.doc_tasks import extract_label
                        got = extract_label(target, "".join(c["label"] for c in it["choices"]))
                        if got != want:
                            self.n_check["dropped"] += 1
                            continue
                        self.n_check["kept"] += 1
                tt = text_ids(self.tok, target)
                m = len(tt)
                tl = kept_logits(self.reader, torch.tensor([suf + tt]), m + 1, cache=copy.deepcopy(cache))[:-1]
                lp = torch.log_softmax(tl.float(), -1).gather(-1, torch.tensor(tt, device=tl.device).view(-1, 1))
                conf = float(lp.mean().exp())
                if conf < self.a.conf:
                    continue
                rows = rt.rows(did, it, tt)
                rec = None
                if self.a.dist_w > 0:
                    rec = {e: {k: v for k, v in r.items() if k in ("q", "k", "out")} for e, r in rt.teacher(did, rows).items()}
                res.append({"it": it, "target": target, "tt": tt, "t_log": tl.to(torch.bfloat16), "conf": conf,
                            "rows": rows, "rec": rec})
            except torch.cuda.OutOfMemoryError:
                oom_event(self.a, did, it.get("key"), "teacher")
                if not self.a.oom_skip:
                    raise
                torch.cuda.empty_cache()
                self.n_oom["teacher"] += 1
                log(f"    OOM in teacher targets for {it.get('key')}; item skipped (--oom_skip)")
                continue
        del cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return res

    # ------------------------------------------------------------------ one registration loss
    def loss(self, did, x):
        from register_state import kept_logits
        from xenc.losses import answer_ce, kl_teacher_student
        a, rt, R = self.a, self.rt, self.rt.R
        op, n_op, _ = rt.operator(did)
        m = len(x["tt"])
        R.load(self.ad.transform_op(op), n_op)
        R.mode, R.need_logits, R.rec = "e2e", a.dist_w > 0, {}
        try:
            s_log = kept_logits(self.reader, torch.tensor([x["rows"]]), m + 1)
            if x["t_log"] is None:              # encoder-only: distillation term only
                kl = ce = torch.zeros((), device=s_log.device)
            else:
                kl = kl_teacher_student(x["t_log"].float(), s_log[:-1])
                ce = answer_ce(s_log, x["tt"])
            loss = a.kl * kl + a.ce * ce
            dist = torch.zeros((), device=kl.device)
            if a.dist_w > 0:
                rec = x["rec"]
                with torch.no_grad():
                    yref = {li: (rec[e]["out"].float() @ self.B0[li].t()) for li, e in self.lmap.items()}
                terms = rt.attn_terms(rec, op, n_op, yref)
                dist = torch.stack([t["kl"] + a.rel_w * t["rel"] for t in terms.values()]).mean()
                loss = loss + a.dist_w * dist.to(loss.device)
            if a.l2 > 0:
                loss = loss + a.l2 * self.ad.drift().to(loss.device)
        finally:
            R.mode, R.rec = "off", {}
        return loss, {"kl": float(kl.detach()), "ce": float(ce.detach()), "dist": float(dist.detach())}

    # ------------------------------------------------------------------ register one document
    def register(self, did, items):
        a = self.a
        budgets = sorted({int(b) for b in a.budgets.split(",")})
        t0 = time.time()
        X = self.targets(did, items)
        t_teacher = time.time() - t0
        H = []
        if a.holdout > 0 and len(X) >= 5:      # held-out SELF-questions (grouped by base question) for early stopping
            base = lambda x: x["it"]["key"].rsplit("_p", 1)[0] if "_p" in x["it"]["key"] else x["it"]["key"]   # noqa: E731
            keys = sorted({base(x) for x in X})
            random.Random(zlib.crc32(did.encode())).shuffle(keys)
            hk = set(keys[: max(1, int(round(a.holdout * len(keys))))])
            H = [x for x in X if base(x) in hk]
            X = [x for x in X if base(x) not in hk]
        self.ad.reset()
        snaps = {0: self.ad.snapshot()}
        ho = {0: self.ho_loss(did, H)} if H else {}
        opt = torch.optim.Adam(self.ad.param_groups(a.lr, a.lr_gate, a.lr_bias))
        params = [p for g in opt.param_groups for p in g["params"]]
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) if a.warmup > 0 else 1.0)
        rng = random.Random(a.seed + zlib.crc32(did.encode()) % 10007)
        hist, times = [], {0: 0.0}
        n_doc_oom = 0
        t1 = time.time()
        order = []
        for step in range(1, max(budgets) + 1):
            if not X:
                break
            opt.zero_grad(set_to_none=True)
            acc = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "dist": 0.0}
            n = 0
            for _ in range(a.batch):
                if not order:
                    order = list(range(len(X)))
                    rng.shuffle(order)
                x = X[order.pop()]
                try:
                    with torch.enable_grad():
                        loss, parts = self.loss(did, x)
                        if not math.isfinite(float(loss)):
                            continue
                        (loss / a.batch).backward()
                except torch.cuda.OutOfMemoryError:
                    oom_event(a, did, x["it"].get("key"), "step")
                    if not a.oom_skip:
                        raise
                    # drop this whole step (partial grads discarded) and the item for the rest of registration
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    self.n_oom["step"] += 1
                    n_doc_oom += 1
                    log(f"    OOM in registration step for {x['it'].get('key')}; step skipped, item dropped (--oom_skip)")
                    if n_doc_oom > a.oom_doc_max:   # repeated OOMs can leave the CUDA context unusable: give up on the doc
                        raise DocSkipped(f"{did}: {n_doc_oom} registration-step OOMs > --oom_doc_max {a.oom_doc_max}")
                    X = [y for y in X if y is not x]
                    order = []
                    acc = {k: 0.0 for k in acc}
                    n = 0
                    break
                acc["loss"] += float(loss)
                for k, v in parts.items():
                    acc[k] += v
                n += 1
            if n:
                torch.nn.utils.clip_grad_norm_(params, a.clip)
                opt.step()
                sched.step()
                hist.append({k: v / n for k, v in acc.items()})
            if step in budgets:
                snaps[step] = self.ad.snapshot()
                if H:
                    ho[step] = self.ho_loss(did, H)
                times[step] = round(time.time() - t1, 1)
            if step % 25 == 0:
                w = hist[-25:]
                log(f"    {did} step {step}: " + " ".join(f"{k} {sum(h[k] for h in w) / len(w):.4f}" for k in acc)
                    + f" | drift {float(self.ad.drift()):.2e} ({(time.time() - t1) / step:.2f}s/step)")
        for b in budgets:                       # an empty X keeps the base for every budget
            snaps.setdefault(b, snaps[0])
            times.setdefault(b, 0.0)
        return {"doc_id": did, "snaps": snaps, "n_items": len(X), "n_reg_items": len(items), "n_holdout": len(H),
                "ho_loss": ho,
                "targets": [{"q": x["it"]["question"][:200], "t": x["target"], "conf": round(x["conf"], 3),
                             "kind": (x["it"].get("meta") or {}).get("template", "recite")} for x in X],
                "t_teacher_s": round(t_teacher, 1), "t_reg_s": times, "hist": hist, "n_params": self.ad.n_params(),
                "parts": sorted(self.ad.parts), "rank": a.rank, "kv_rank": a.kv_rank,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}

    @torch.no_grad()
    def ho_loss(self, did, H) -> float:
        """Mean KL(text teacher ‖ student) + 0.5·CE(teacher answer) on held-out self-questions (label-free)."""
        from register_state import kept_logits
        from xenc.losses import answer_ce, kl_teacher_student
        rt, R = self.rt, self.rt.R
        op, n_op, _ = rt.operator(did)
        tot = 0.0
        if self.a.enc_only:                     # held-out distillation (attention KL + relMSE) instead
            for x in H:
                with torch.no_grad():
                    tot += float(self.loss(did, x)[1]["dist"])
            return tot / max(len(H), 1)
        n_ok = 0
        for x in H:
            m = len(x["tt"])
            R.load(self.ad.transform_op(op), n_op)
            R.mode, R.need_logits, R.rec = "e2e", False, {}
            try:
                s_log = kept_logits(self.reader, torch.tensor([x["rows"]]), m + 1)
            except torch.cuda.OutOfMemoryError:
                oom_event(self.a, did, x["it"].get("key"), "holdout")
                if not self.a.oom_skip:
                    raise
                torch.cuda.empty_cache()
                self.n_oom["holdout"] += 1
                continue
            finally:
                R.mode, R.rec = "off", {}
            tot += float(kl_teacher_student(x["t_log"].float(), s_log[:-1])) + 0.5 * float(answer_ce(s_log, x["tt"]))
            n_ok += 1
        return tot / max(n_ok if self.a.oom_skip else len(H), 1)

    # ------------------------------------------------------------------ generation with a registered adapter
    @torch.no_grad()
    def gen(self, did, it, snap, budget):
        from register_state import _stop_ids
        from xenc.backbone import decode
        rt, R = self.rt, self.rt.R
        self.ad.load(snap)
        rows = rt.rows(did, it)
        op, n_op, _ = rt.operator(did)
        R.load(self.ad.transform_op(op), n_op)
        R.mode = "student"
        try:
            out = self.reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=True, logits_to_keep=1)
            stops, new = _stop_ids(self.tok), []
            for _ in range(budget):
                nxt = int(out.logits[0, -1].argmax())
                new.append(nxt)
                if nxt in stops or "</answer>" in self.tok.decode(new[-12:], skip_special_tokens=True):
                    break
                out = self.reader(input_ids=torch.tensor([[nxt]], device=rt.dev), past_key_values=out.past_key_values,
                                  use_cache=True, logits_to_keep=1)
        finally:
            R.mode, R.past = "off", {}
        txt = decode(self.tok, self.a.reader, new)
        if "</answer>" in txt:
            txt = txt.split("</answer>", 1)[0] + "</answer>"
        return txt, len(new)


def load_selfq(path) -> dict:
    by = {}
    for line in open(path):
        r = json.loads(line)
        by[r["doc_id"]] = r["items"]
    return by


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--B", default="outputs/D10_maps/encB8_9b_2b_post.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--interface", default="outputs/D13_reg_w10/d13_best.pt", help="shared base interface")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max_docs", type=int, default=0, help="restrict to the first K sorted docs (dev sets)")
    ap.add_argument("--skip_docs", type=int, default=0, help="dev sets: skip the first K sorted docs (e.g. D13's val docs)")
    ap.add_argument("--seed", type=int, default=0, help="judge item sample seed (eval_d13: 0) and adapter init")
    ap.add_argument("--wrong_map", default=None)
    ap.add_argument("--selfq", required=True, help="scripts/d20_selfq.py output for this task")
    ap.add_argument("--max_items", type=int, default=48)
    ap.add_argument("--recite", type=int, default=0, help="recitation prompts per document")
    ap.add_argument("--recite_cont", type=int, default=0, help=">0: recitation target = the next N document tokens "
                    "(teacher-forced KL of the text teacher + CE on the true span); 0 = teacher's greedy answer")
    ap.add_argument("--shuffle", type=int, default=0, help="K>0: replace every free-order MCQ self-question (mcq / mix "
                    "templates) by K random option permutations (balances the answer letter; nli keeps its fixed order)")
    ap.add_argument("--ans_budget", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.0, help="drop items whose teacher confidence is below")
    ap.add_argument("--parts", default="dA,gate")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--kv_rank", type=int, default=8)
    ap.add_argument("--budgets", default="0,25,50,100,200,400")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_gate", type=float, default=3e-3)
    ap.add_argument("--lr_bias", type=float, default=None)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=0, help="linear lr warmup steps")
    ap.add_argument("--shuffle_nli", type=int, default=0, help="1: --shuffle also permutes the nli template's options")
    ap.add_argument("--holdout", type=float, default=0.0, help="fraction of self-questions held out for per-document "
                    "early stopping (kv_es@b = the budget <= b with the lowest held-out self-question loss)")
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--ce", type=float, default=0.5)
    ap.add_argument("--dist_w", type=float, default=1.0)
    ap.add_argument("--rel_w", type=float, default=1.0)
    ap.add_argument("--l2", type=float, default=0.0)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000, help="document prefix cap (Shakespeare-CF: 60000)")
    ap.add_argument("--runtime", default="d13", choices=("d13", "d12p"), help="interface module / runtime family")
    ap.add_argument("--enc_device", default=None, help="encoder device (d13 runtime too: e.g. --device cuda:1 "
                    "--enc_device cuda:0 puts the 4B encoder and its operator prefill on a second GPU); default = --device")
    ap.add_argument("--oom_doc_max", type=int, default=3, help="--oom_skip: give up on a document (no adapter; its "
                    "items are dropped from every judged condition) after this many registration-step OOMs")
    ap.add_argument("--dist_chunk", type=int, default=0, help="F1: >0 computes the distillation KL/relMSE in row chunks of "
                    "this size, each checkpointed (same loss, lower peak memory for 60k-128k operators); 0 = off")
    ap.add_argument("--oom_exit", type=int, default=0, help="F1: 1 = on ANY CUDA OOM, record (doc, item) in "
                    "<out>/oom_events.jsonl and exit 75 at once (a caught OOM can corrupt the CUDA context); re-run the "
                    "same command: registration resumes and drops the recorded items (a doc with > --oom_doc_max of them "
                    "is skipped), the judge resumes from its ckpt and drops recorded items from every condition")
    ap.add_argument("--oom_skip", type=int, default=0, help="F1: 1 skips (and counts) items that hit CUDA OOM in the "
                    "teacher pass, a registration step, the held-out loss or the judge, instead of crashing; 0 = off")
    ap.add_argument("--reader_mem", default=None, help="d12p: shard the reader, per-GPU budgets '27GiB,27GiB'")
    ap.add_argument("--itf_device", default=None, help="d12p: interface device")
    ap.add_argument("--enc_only", type=int, default=0, help="1: ENCODER-ONLY registration (stage-1 distillation on "
                    "self-questions only; no text teacher, the reader never reads the document as text)")
    ap.add_argument("--teacher", default="text", choices=("text", "none"), help="none == --enc_only 1: no 9B text "
                    "pass of the document at all (no prefill, no greedy answers); distillation term only")
    ap.add_argument("--mem_safe", type=int, default=0, help="1: torch.utils.checkpoint the read (identical gradients)")
    ap.add_argument("--phase", default="register,eval")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--eval_budgets", default=None, help="default: all --budgets")
    ap.add_argument("--conds", default="closed_tail,text,kv,wrong_doc")
    ap.add_argument("--eval_tag", default="", help="suffix for eval_<split>_<tag>.json / summary_<tag>.md")
    ap.add_argument("--reg_n", type=int, default=-1, help="register the docs of this item sample size (-1 = --n); "
                    "e.g. --reg_n 0 registers every document of the split once, then eval at --n 300 and --n 0")
    ap.add_argument("--budget", type=int, default=32, help="judge generation budget")
    ap.add_argument("--read_tau", type=float, default=1.0, help="F1: read-softmax temperature (<1 sharpens; 1 = off)")
    ap.add_argument("--read_logn_ref", type=int, default=0, help="F1: >0 scales read logits by log(N)/log(ref) for "
                    "operators longer than ref tokens (0 = off)")
    ap.add_argument("--check_label", type=int, default=0, help="F1: 1 drops self-questions whose text-teacher answer "
                    "letter differs from the writer's construction label (meta.intended); 0 = off")
    ap.add_argument("--tok_off", type=int, default=0, help="P3: 1 = judge a token-channel interface (xenc/p3_tok.py) "
                    "with the channel DISABLED (ablation; use its own --out); 0 = as saved")
    ap.add_argument("--ptr_off", type=int, default=0, help="P4: 1 = judge a pointer-head interface (xenc/p4_ptr.py) "
                    "with the pointer DISABLED (ablation; use its own --out); 0 = as saved")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.oom_exit:                              # --oom_exit implies the paired-skip judge of --oom_skip
        a.oom_skip = 1
    if a.teacher == "none" or a.enc_only:
        a.enc_only, a.teacher, a.kl, a.ce = 1, "none", 0.0, 0.0
        assert a.dist_w > 0, "encoder-only registration needs --dist_w > 0"
    from xenc.d20_reg import assert_no_test_questions
    from xenc.doc_tasks import load_docs
    from d20_selfq import task_questions
    os.makedirs(os.path.join(a.out, "adapters"), exist_ok=True)
    with open(os.path.join(a.out, f"args_{a.phase.replace(',', '_')}_s{a.shard}.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    torch.manual_seed(a.seed)
    items, docs_all = judge_items(a)
    wmap = json.load(open(a.wrong_map)) if a.wrong_map else {}
    reg_docs = sorted({_get(i, "doc_id") for i in items} | {wrong_of(_get(i, "doc_id"), docs_all, wmap) for i in items})
    docs = load_docs(a.split_dir)
    tq = task_questions(a.split_dir)
    S = D20(a)
    S.rt.docs = docs
    if hasattr(S.itf, "tok_layers"):                  # P3 token channel interface
        S.itf.tok_on = not a.tok_off
        log(f"P3 token channel: {'OFF (ablation)' if a.tok_off else 'ON'} | {S.itf.config()['tok']} | {S.itf.tok_state()}")
    else:
        assert not a.tok_off, "--tok_off needs a token-channel interface"
    if hasattr(S.itf, "ptr_layers"):                  # P4 pointer / copy head interface
        S.itf.ptr_on = not a.ptr_off
        log(f"P4 pointer head: {'OFF (ablation)' if a.ptr_off else 'ON'} | {S.itf.config()['ptr']['layers']} shifts "
            f"{S.itf.ptr_shifts} chain {S.itf.ptr_chain} | {S.itf.ptr_state()}")
    else:
        assert not a.ptr_off, "--ptr_off needs a pointer-head interface"
    S.itf.read_tau, S.itf.read_logn_ref = float(a.read_tau), int(a.read_logn_ref)
    S.rt.kl_chunk = int(a.dist_chunk)
    if a.dist_chunk:
        log(f"F1 dist_chunk {a.dist_chunk}: distillation terms computed in row chunks")
    if a.read_tau != 1.0 or a.read_logn_ref:
        log(f"F1 read scaling ON: tau {a.read_tau} logn_ref {a.read_logn_ref} (mult at 34k/80k/113k: "
            f"{S.itf.read_scale_mult(34000):.3f}/{S.itf.read_scale_mult(80000):.3f}/{S.itf.read_scale_mult(113000):.3f})")
    log(f"D20: {a.encoder} -> {a.reader} | base {a.interface} | per-doc parts {sorted(S.ad.parts)} rank {a.rank} "
        f"kv_rank {a.kv_rank}: {S.ad.n_params()} params/doc | judged {len(items)} items / "
        f"{len({_get(i, 'doc_id') for i in items})} docs; registered docs {len(reg_docs)} | signals kl {a.kl} ce {a.ce} "
        f"dist {a.dist_w} recite {a.recite} | budgets {a.budgets}")

    if "register" in a.phase:
        selfq = load_selfq(a.selfq)
        reg_set = reg_docs
        if a.reg_n >= 0 and a.reg_n != a.n:
            import copy as _c
            b = _c.copy(a)
            b.n = a.reg_n
            its2, _ = judge_items(b)
            reg_set = sorted(set(reg_docs) | {_get(i, "doc_id") for i in its2}
                             | {wrong_of(_get(i, "doc_id"), docs_all, wmap) for i in its2})
        mine = [d for k, d in enumerate(reg_set) if k % a.nshard == a.shard]
        skipped = set()
        if a.oom_exit and os.path.exists(os.path.join(a.out, "skipped_docs.jsonl")):
            skipped = {json.loads(x)["doc_id"] for x in open(os.path.join(a.out, "skipped_docs.jsonl"))}
        ev = [e for e in oom_events(a.out) if e["where"] in ("teacher", "step", "holdout")] if a.oom_exit else []
        for k, did in enumerate(mine):
            path = os.path.join(a.out, "adapters", f"{did}.pt")
            if os.path.exists(path) or did in skipped:
                continue
            bad_keys = {e["key"] for e in ev if e["doc_id"] == did}
            if len(bad_keys) > a.oom_doc_max:
                log(f"  [{k + 1}/{len(mine)}] SKIPPED {did}: {len(bad_keys)} OOM items > --oom_doc_max {a.oom_doc_max} (--oom_exit)")
                with open(os.path.join(a.out, "skipped_docs.jsonl"), "a") as fs:
                    fs.write(json.dumps({"doc_id": did, "reason": f"{len(bad_keys)} OOM items", "keys": sorted(bad_keys)}) + "\n")
                continue
            assert did in selfq, f"no self-questions for {did} in {a.selfq}"
            reg = shuffle_choices([dict(x) for x in selfq[did]][: a.max_items], a.shuffle, bool(a.shuffle_nli)) + recite_items(S.tok, did, docs[did], a.recite, a.max_tokens - 1100, a.seed, a.recite_cont)
            if a.oom_exit and bad_keys:          # items that OOMed in an earlier process: dropped up front
                n0 = len(reg)
                reg = [x for x in reg if _base_key(x["key"]) not in bad_keys]
                log(f"  {did}: dropping {n0 - len(reg)} registration items that OOMed before: {sorted(bad_keys)} (--oom_exit)")
            # LEAKAGE GUARD (hard): no registration prompt equals / contains any question string of this task
            # (train / val / test item files), and only D20 self-question / recitation items are registered.
            assert_no_test_questions(reg, tq)
            assert all(x["doc_id"] == did for x in reg)
            t0 = time.time()
            try:
                r = S.register(did, reg)
            except DocSkipped as ex:
                torch.cuda.empty_cache()
                log(f"  [{k + 1}/{len(mine)}] SKIPPED {did}: {ex}")
                with open(os.path.join(a.out, "skipped_docs.jsonl"), "a") as fs:
                    fs.write(json.dumps({"doc_id": did, "reason": str(ex), "oom": dict(S.n_oom)}) + "\n")
                continue
            torch.save(r, path + ".tmp")
            os.replace(path + ".tmp", path)
            log(f"  [{k + 1}/{len(mine)}] registered {did}: {r['n_items']} items (teacher {r['t_teacher_s']}s, "
                f"reg {r['t_reg_s']}) total {time.time() - t0:.0f}s"
                + (f" | check_label cumulative {S.n_check}" if a.check_label else "")
                + (f" | oom cumulative {S.n_oom}" if a.oom_skip else ""))

    if "eval" in a.phase:
        from eval_d13 import summarize
        from register_state import greedy
        from xenc.backbone import build_chat
        from xenc.doc_tasks import ctx_prompt, q_prompt, score
        missing = [d for d in reg_docs if not os.path.exists(os.path.join(a.out, "adapters", f"{d}.pt"))]
        dropped = {}
        if missing and a.oom_skip:
            # paired-skip design: an item is judged in EVERY condition or in none
            ms_ = set(missing)
            keep = [i for i in items if _get(i, "doc_id") not in ms_ and wrong_of(_get(i, "doc_id"), docs_all, wmap) not in ms_]
            dropped = {"docs": sorted(missing), "n_items_dropped": len(items) - len(keep), "n_items_kept": len(keep)}
            log(f"oom_skip: {len(missing)} unregistered docs {sorted(missing)} -> dropping {len(items) - len(keep)} of "
                f"{len(items)} items (doc or wrong_doc unregistered) from every condition")
            items = keep
            reg_docs = [d for d in reg_docs if d not in ms_]
        assert not (set(missing) & ({_get(i, "doc_id") for i in items} | {wrong_of(_get(i, "doc_id"), docs_all, wmap) for i in items})), \
            f"{len(missing)} documents not registered yet, e.g. {missing[:3]}"
        ads = {d: torch.load(os.path.join(a.out, "adapters", f"{d}.pt"), map_location="cpu", weights_only=False)
               for d in reg_docs}
        eb = sorted({int(b) for b in (a.eval_budgets or a.budgets).split(",")})
        conds = a.conds.split(",")
        rows_all, t0 = [], time.time()
        order = sorted(range(len(items)), key=lambda k: _get(items[k], "doc_id"))
        res = {}
        tag = f"{a.split}" + (f"_{a.eval_tag}" if a.eval_tag else "")
        prev = {}
        if a.oom_exit:
            if os.path.exists(os.path.join(a.out, f"eval_{tag}.json")):
                log(f"--oom_exit: eval_{tag}.json already exists; eval phase done")
                return 0
            ckp = os.path.join(a.out, f"eval_{tag}.ckpt.jsonl")
            if os.path.exists(ckp):              # resume a judge interrupted by an OOM restart
                for line in open(ckp):
                    try:
                        r = json.loads(line)
                        prev[r["key"]] = r
                    except Exception:
                        pass
            bad_eval = {e["key"] for e in oom_events(a.out) if e["where"] == "eval"}
            if bad_eval:
                n0 = len(order)
                order = [k for k in order if _get(items[k], "key") not in bad_eval]
                dropped["eval_oom_items"] = sorted(bad_eval)
                log(f"--oom_exit: dropping {n0 - len(order)} judged items that OOMed before, from every condition")
        ck = open(os.path.join(a.out, f"eval_{tag}.ckpt.jsonl"), "w")
        for j, k in enumerate(order):
            it = items[k]
            did = _get(it, "doc_id")
            w = wrong_of(did, docs_all, wmap)
            rec = {"key": _get(it, "key"), "doc_id": did, "wrong": w, "near_dup_train": (_get(it, "meta") or {}).get("near_dup_train")}
            if rec["key"] in prev:              # --oom_exit resume: re-use the judged record
                res[k] = prev[rec["key"]]
                rows_all.append(res[k])
                ck.write(json.dumps(res[k]) + "\n")
                ck.flush()
                continue
            try:
                if "closed_book" in conds:
                    g, n = greedy(S.reader, S.tok, a.reader, build_chat(S.tok, a.reader, SYSTEM_ANSWER, q_prompt(it)), a.budget)
                    rec["closed_book"] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
                if "closed_tail" in conds:
                    g, n = S.rt.gen(did, it, a.budget, read=False)
                    rec["closed_tail"] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
                if "text" in conds:
                    g, n = greedy(S.reader, S.tok, a.reader, build_chat(S.tok, a.reader, SYSTEM_ANSWER, ctx_prompt(it, docs[did])), a.budget)
                    rec["text"] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
                for b in eb:
                    if "kv" in conds:
                        g, n = S.gen(did, it, ads[did]["snaps"][b], a.budget)
                        rec[f"kv@{b}"] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
                    if "wrong_doc" in conds and w is not None:
                        g, n = S.gen(w, it, ads[w]["snaps"][b], a.budget)
                        rec[f"wrong_doc@{b}"] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
            except torch.cuda.OutOfMemoryError:
                oom_event(a, did, rec["key"], "eval")
                if not a.oom_skip:
                    raise
                torch.cuda.empty_cache()
                S.n_oom["eval"] += 1
                log(f"  OOM judging {rec['key']}; item dropped from EVERY condition (--oom_skip), total {S.n_oom['eval']}")
                continue
            res[k] = rec
            rows_all.append(rec)
            ck.write(json.dumps(rec) + "\n")
            ck.flush()
            if (j + 1) % 20 == 0:
                log(f"  [{j + 1}/{len(items)}] " + " ".join(
                    f"kv@{b}={sum(r[f'kv@{b}']['score'] for r in rows_all) / len(rows_all):.3f}" for b in eb)
                    + f" | {(time.time() - t0) / len(rows_all):.1f}s/item")
        rows = [res[k] for k in range(len(items)) if k in res]
        if a.oom_skip:
            log(f"oom_skip: {S.n_oom} (judged {len(rows)} of {len(items)} items)")
        out = {"args": vars(a), "budgets": {}, "cost": cost_summary(ads, items, eb)}
        if a.oom_skip:
            out["oom_skip"] = dict(S.n_oom)
            out["oom_dropped"] = dropped
        subsets = {"all": rows}
        if any(r["near_dup_train"] is not None for r in rows):
            subsets["clean"] = [r for r in rows if r["near_dup_train"] is False]
        def sel(d, b):              # per-document early stop: the budget <= b with the lowest held-out self-q loss
            ho = {int(k): v for k, v in (ads[d].get("ho_loss") or {}).items() if int(k) in eb and int(k) <= b}
            return min(sorted(ho), key=lambda k: ho[k]) if ho else b
        has_es = all(ads[d].get("ho_loss") for d in reg_docs)
        labels = [(str(b), b, False) for b in eb] + ([(f"es≤{b}", b, True) for b in eb if b != eb[0]] if has_es else [])
        out["es_choice"] = {d: {str(b): sel(d, b) for b in eb} for d in reg_docs} if has_es else {}
        for sname, rs in subsets.items():
            for lab, b, es in labels:
                rb, base_pair = [], []
                for r in rs:
                    x = {c: r[c] for c in ("closed_book", "closed_tail", "text") if c in r}
                    bk = sel(r["doc_id"], b) if es else b
                    if f"kv@{bk}" in r:
                        x["kv"] = r[f"kv@{bk}"]
                        base_pair.append(r[f"kv@{bk}"]["score"] - r[f"kv@{eb[0]}"]["score"])
                    if r.get("wrong") is not None:
                        bw = sel(r["wrong"], b) if es else b
                        if f"wrong_doc@{bw}" in r:
                            x["wrong_doc"] = r[f"wrong_doc@{bw}"]
                    rb.append(x)
                s = summarize(rb, [c for c in ("closed_book", "closed_tail", "text", "kv", "wrong_doc") if c in conds])
                if b != eb[0] and base_pair:            # registration gain vs kv_base (budget 0), paired
                    mu = sum(base_pair) / len(base_pair)
                    sd = (sum((p - mu) ** 2 for p in base_pair) / max(len(base_pair) - 1, 1)) ** 0.5
                    s["kv_minus_kv_base"] = {"paired_gap": mu, "se": sd / len(base_pair) ** 0.5, "n_paired": len(base_pair),
                                             "wins": sum(p > 0 for p in base_pair), "losses": sum(p < 0 for p in base_pair)}
                out["budgets"].setdefault(sname, {})[lab] = s
        out["items"] = rows
        json.dump(out, open(os.path.join(a.out, f"eval_{tag}.json"), "w"), indent=1)
        md = table_md(out, a)
        open(os.path.join(a.out, "summary.md" if not a.eval_tag else f"summary_{a.eval_tag}.md"), "w").write(md)
        log(md)
    return 0


def cost_summary(ads, items, eb) -> dict:
    judged = sorted({_get(i, "doc_id") for i in items})
    rs = [ads[d] for d in judged]
    out = {"n_docs": len(rs), "params_per_doc": rs[0]["n_params"] if rs else 0,
           "teacher_s_mean": sum(r["t_teacher_s"] for r in rs) / max(len(rs), 1),
           "items_per_doc": sum(r["n_items"] for r in rs) / max(len(rs), 1), "gpu": sorted({r["gpu"] for r in rs}),
           "reg_s_mean": {str(b): sum(float(r["t_reg_s"].get(b, 0.0)) for r in rs) / max(len(rs), 1) for b in eb}}
    return out


def table_md(out, a) -> str:
    c = out["cost"]
    L = [f"## D20 {a.split_dir} split={a.split} fold={a.fold} n={a.n} | parts {a.parts} rank {a.rank} | "
         f"kl {a.kl} ce {a.ce} dist {a.dist_w} recite {a.recite} batch {a.batch} lr {a.lr}",
         f"cost: {c['params_per_doc']} params/doc, {c['items_per_doc']:.1f} reg items/doc, teacher "
         f"{c['teacher_s_mean']:.0f}s/doc, reg s/doc by steps {c['reg_s_mean']} on {c['gpu']}", ""]
    for sname, bd in out["budgets"].items():
        L += [f"**{sname}**", "",
              "| steps | n | closed_tail | text | kv (gen) | wrong_doc | kv−wrong (W/L) | kv−tail | **kv−text** | kv−kv_base |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for b, s in bd.items():
            f = lambda k: (f"{s[k]['paired_gap']:+.3f}±{s[k]['se']:.3f}" + (f" ({s[k]['wins']}/{s[k]['losses']})" if k == "kv_minus_wrong_doc" else "")) if k in s else "—"   # noqa: E731
            sc = lambda k: f"{s[k]['score']:.3f}" if k in s else "—"   # noqa: E731
            L.append(f"| {b} | {s.get('kv', {}).get('n', '')} | {sc('closed_tail')} | {sc('text')} | {sc('kv')} "
                     f"({s.get('kv', {}).get('mean_gen_tokens', 0):.1f}) | {sc('wrong_doc')} | {f('kv_minus_wrong_doc')} | "
                     f"{f('kv_minus_closed_tail')} | {f('kv_minus_text')} | {f('kv_minus_kv_base')} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
