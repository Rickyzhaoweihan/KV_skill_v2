"""D26: per-document REGISTRATION (the D20 recipe) of the paper-form read for a CROSS-FAMILY reader (e.g. GLM-4-9B),
then the judge. Identical to scripts/register_d20.py (DocAdapter dA+gate rank 16, shuffled self-MCQs, held-out
self-questions, budgets 0..400, hard leakage assertion, wrong_doc = another document's operator AND its adapter) except
for the cross-tokenizer runtime (scripts/fit_d26.D26Rt):
  --teacher none   READER-FREE: stage-1 distillation only (the encoder's own attention / read-out over ITS tokenization
                   of the same rows, character-aligned to the reader rows); the reader never reads the document text
  --teacher text   + KL(reader-with-document ‖ student) + 0.5·CE on the reader's own greedy answer (the reader's
                   chat template; for GLM this is the "GLM-teacher" arm)
Self-questions come from scripts/d20_selfq.py (document text only; --reader Qwen/Qwen3.5-4B = encoder-written).
Hygiene (docs/PAPERFORM.md): budgets ≤ 400 (pilot); es≤b selects among budgets ≥ 1 only (step 0 excluded); the log
prints, per document, whether the held-out self-question loss beats step 0.

    python scripts/docreg_d26.py --reader glm4 --B outputs/D26_maps/B_glm4.pt --interface outputs/D26_glm4_reg/d13_best.pt \
        --split_dir data/longhealth_split --fold 0 --n 0 --selfq outputs/D26_selfq/longhealth_by4b.jsonl --teacher none \
        --out outputs/D26_glm4_reg_lh_rf
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
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
from register_doc import SYSTEM_ANSWER, _get, log  # noqa: E402


class D26Doc:
    """D20's per-document registration on a D26Rt (cross-tokenizer runtime)."""

    def __init__(self, a, rt, itf, lmap):
        from xenc.d20_reg import DocAdapter
        if itf.delta_rank:
            itf.merge_delta()
        self.a, self.rt, self.itf, self.lmap = a, rt, itf, lmap
        self.reader, self.tok = rt.reader, rt.tok_r
        self.rt.ckpt = self.rt.R.ckpt = bool(a.mem_safe)
        self.B0 = {li: itf.B[str(li)].detach().clone() for li in lmap}
        self.ad = DocAdapter(itf, [p for p in a.parts.split(",") if p], a.rank, a.kv_rank, a.seed)
        self._rpre: dict = {}

    def reader_pre(self, did) -> list[int]:
        """The reader's own `text` prompt up to the question (its template), left-truncated like D13Rt.pre."""
        if did not in self._rpre:
            from xenc.state_skill import doc_prefix_text
            ids = self.tok(doc_prefix_text(self.tok, self.a.reader, self.rt.docs[did], SYSTEM_ANSWER),
                           add_special_tokens=False)["input_ids"]
            self._rpre[did] = [int(x) for x in ids][-(self.a.max_tokens - 1024):]
        return self._rpre[did]

    @torch.no_grad()
    def _greedy_cache(self, ids, cache, budget):
        from xenc.backbone import decode
        dev = self.rt.dev
        out = self.reader(input_ids=torch.tensor([ids], device=dev), past_key_values=cache, use_cache=True,
                          logits_to_keep=1)
        new = []
        for _ in range(budget):
            nxt = int(out.logits[0, -1].argmax())
            new.append(nxt)
            if nxt in self.rt.stops or "</answer>" in self.tok.decode(new[-12:], skip_special_tokens=True):
                break
            out = self.reader(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=out.past_key_values,
                              use_cache=True, logits_to_keep=1)
        return decode(self.tok, self.a.reader, new)

    @staticmethod
    def _rec(rec):
        return {e: {k: v for k, v in r.items() if k in ("q", "k", "out")} for e, r in rec.items()}

    @torch.no_grad()
    def targets(self, did, items):
        from register_state import kept_logits
        from xenc.backbone import text_ids
        rt, R, a = self.rt, self.rt.R, self.a
        R.mode = "off"
        if a.enc_only:
            res = []
            for it in items:
                P = rt.pair(did, it)
                res.append({"it": it, "target": "", "tt": [], "t_log": None, "conf": 1.0, "P": P,
                            "rec": self._rec(rt.teacher(did, P["e_ids"]))})
            return res
        pre = self.reader_pre(did)
        out = self.reader(input_ids=torch.tensor([pre], device=rt.dev), use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        del out
        res = []
        for it in items:
            suf = rt.suf("r", it)[2]
            cont = (it.get("meta") or {}).get("cont")
            if cont:
                target = f"<answer>{cont}</answer>"
            else:
                g = self._greedy_cache(suf, copy.deepcopy(cache), a.ans_budget)
                if "<answer>" not in g:
                    continue
                target = g[g.index("<answer>"):].split("</answer>")[0] + "</answer>"
            tt = text_ids(self.tok, target)
            m = len(tt)
            tl = kept_logits(self.reader, torch.tensor([suf + tt]), m + 1, cache=copy.deepcopy(cache))[:-1]
            lp = torch.log_softmax(tl.float(), -1).gather(-1, torch.tensor(tt, device=tl.device).view(-1, 1))
            conf = float(lp.mean().exp())
            if conf < a.conf:
                continue
            P = rt.pair(did, it, target)
            assert P["r_ids"][-m:] == tt
            rec = self._rec(rt.teacher(did, P["e_ids"])) if a.dist_w > 0 else None
            res.append({"it": it, "target": target, "tt": tt, "t_log": tl.to(torch.bfloat16), "conf": conf, "P": P,
                        "rec": rec})
        del cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return res

    def loss(self, did, x):
        from register_state import kept_logits
        from xenc.losses import answer_ce, kl_teacher_student
        a, rt, R = self.a, self.rt, self.rt.R
        op, n_op, _ = rt.operator(did)
        m = len(x["tt"])
        R.load(self.ad.transform_op(op), n_op)
        R.mode, R.need_logits, R.rec = "e2e", a.dist_w > 0, {}
        try:
            s_log = kept_logits(self.reader, torch.tensor([x["P"]["r_ids"]]), m + 1)
            if x["t_log"] is None:
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
                terms = rt.attn_terms(rec, op, n_op, yref, x["P"])
                dist = torch.stack([t["kl"] + a.rel_w * t["rel"] for t in terms.values()]).mean()
                loss = loss + a.dist_w * dist.to(loss.device)
        finally:
            R.mode, R.rec = "off", {}
        return loss, {"kl": float(kl.detach()), "ce": float(ce.detach()), "dist": float(dist.detach())}

    @torch.no_grad()
    def ho_loss(self, did, H) -> float:
        if not H:
            return float("nan")
        tot = 0.0
        for x in H:
            _, p = self.loss(did, x)
            tot += p["dist"] if self.a.enc_only else p["kl"] + 0.5 * p["ce"]
        return tot / len(H)

    def register(self, did, items):
        a = self.a
        budgets = sorted({int(b) for b in a.budgets.split(",")})
        t0 = time.time()
        X = self.targets(did, items)
        t_teacher = time.time() - t0
        H = []
        if a.holdout > 0 and len(X) >= 5:
            base = lambda x: x["it"]["key"].rsplit("_p", 1)[0] if "_p" in x["it"]["key"] else x["it"]["key"]   # noqa: E731
            keys = sorted({base(x) for x in X})
            random.Random(zlib.crc32(did.encode())).shuffle(keys)
            hk = set(keys[: max(1, int(round(a.holdout * len(keys))))])
            H = [x for x in X if base(x) in hk]
            X = [x for x in X if base(x) not in hk]
        self.ad.reset()
        snaps = {0: self.ad.snapshot()}
        ho = {0: self.ho_loss(did, H)} if H else {}
        opt = torch.optim.Adam(self.ad.param_groups(a.lr, a.lr_gate))
        params = [p for g in opt.param_groups for p in g["params"]]
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup) if a.warmup > 0 else 1.0)
        rng = random.Random(a.seed + zlib.crc32(did.encode()) % 10007)
        hist, times, order = [], {0: 0.0}, []
        t1 = time.time()
        for step in range(1, max(budgets) + 1):
            if not X:
                break
            opt.zero_grad(set_to_none=True)
            acc, n = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "dist": 0.0}, 0
            for _ in range(a.batch):
                if not order:
                    order = list(range(len(X)))
                    rng.shuffle(order)
                x = X[order.pop()]
                with torch.enable_grad():
                    loss, parts = self.loss(did, x)
                    if not math.isfinite(float(loss)):
                        continue
                    (loss / a.batch).backward()
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
            if step % 50 == 0:
                w = hist[-50:]
                log(f"    {did} step {step}: " + " ".join(f"{k} {sum(h[k] for h in w) / len(w):.4f}" for k in acc)
                    + f" | drift {float(self.ad.drift()):.2e} gates {[round(float(g), 3) for g in self.itf.g]} "
                    f"({(time.time() - t1) / step:.2f}s/step)")
        for b in budgets:
            snaps.setdefault(b, snaps[0])
            times.setdefault(b, 0.0)
        if ho:
            log(f"    {did} held-out self-q loss by step: " + " ".join(f"{k}:{v:.4f}" for k, v in sorted(ho.items()))
                + f" | beats step 0: {any(v < ho[0] for k, v in ho.items() if k > 0)}")
        return {"doc_id": did, "snaps": snaps, "n_items": len(X), "n_reg_items": len(items), "n_holdout": len(H),
                "ho_loss": ho, "targets": [{"q": x["it"]["question"][:200], "t": x["target"], "conf": round(x["conf"], 3)}
                                           for x in X],
                "t_teacher_s": round(t_teacher, 1), "t_reg_s": times, "hist": hist, "n_params": self.ad.n_params(),
                "parts": sorted(self.ad.parts), "rank": a.rank, "teacher": a.teacher,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}

    @torch.no_grad()
    def gen(self, did, it, snap, budget):
        from xenc.backbone import decode
        rt, R = self.rt, self.rt.R
        self.ad.load(snap)
        rows = rt.rows(did, it)
        op, n_op, _ = rt.operator(did)
        R.load(self.ad.transform_op(op), n_op)
        R.mode = "student"
        try:
            out = self.reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=True, logits_to_keep=1)
            new = []
            for _ in range(budget):
                nxt = int(out.logits[0, -1].argmax())
                new.append(nxt)
                if nxt in rt.stops or "</answer>" in self.tok.decode(new[-12:], skip_special_tokens=True):
                    break
                out = self.reader(input_ids=torch.tensor([[nxt]], device=rt.dev), past_key_values=out.past_key_values,
                                  use_cache=True, logits_to_keep=1)
        finally:
            R.mode, R.past = "off", {}
        txt = decode(self.tok, self.a.reader, new)
        if "</answer>" in txt:
            txt = txt.split("</answer>", 1)[0] + "</answer>"
        return txt, len(new)


def es_choice(ho: dict, eb, b):
    """Per-document early stop among budgets 1..b (step 0 EXCLUDED, run-hygiene rule 2): lowest held-out loss."""
    c = {int(k): v for k, v in (ho or {}).items() if 0 < int(k) <= b and int(k) in eb and math.isfinite(v)}
    return min(sorted(c), key=lambda k: c[k]) if c else b


def main(argv=None) -> int:
    from fit_d26 import add_common
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_common(ap)
    ap.add_argument("--interface", required=True, help="shared base interface (D26 stage 2)")
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max_docs", type=int, default=0)
    ap.add_argument("--skip_docs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wrong_map", default=None)
    ap.add_argument("--selfq", required=True)
    ap.add_argument("--max_items", type=int, default=48)
    ap.add_argument("--shuffle", type=int, default=2)
    ap.add_argument("--shuffle_nli", type=int, default=0)
    ap.add_argument("--ans_budget", type=int, default=40)
    ap.add_argument("--conf", type=float, default=0.0)
    ap.add_argument("--parts", default="dA,gate")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--kv_rank", type=int, default=8)
    ap.add_argument("--budgets", default="0,25,50,100,200,400")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_gate", type=float, default=3e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--kl", type=float, default=1.0)
    ap.add_argument("--ce", type=float, default=0.5)
    ap.add_argument("--dist_w", type=float, default=1.0)
    ap.add_argument("--rel_w", type=float, default=1.0)
    ap.add_argument("--teacher", default="none", choices=("text", "none"))
    ap.add_argument("--mem_safe", type=int, default=1)
    ap.add_argument("--phase", default="register,eval")
    ap.add_argument("--conds", default="closed_tail,text,kv,wrong_doc")
    ap.add_argument("--reg_n", type=int, default=-1)
    ap.add_argument("--eval_tag", default="")
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    assert max(int(b) for b in a.budgets.split(",")) <= 500, "pilot rule: ≤ 500 registration steps"
    a.enc_only = int(a.teacher == "none")
    if a.enc_only:
        a.kl, a.ce = 0.0, 0.0
        assert a.dist_w > 0
    from d20_selfq import task_questions
    from fit_d26 import build, make_rt
    from register_d20 import cost_summary, judge_items, load_selfq, shuffle_choices, table_md, wrong_of
    from xenc.d20_reg import assert_no_test_questions
    from xenc.doc_tasks import load_docs
    os.makedirs(os.path.join(a.out, "adapters"), exist_ok=True)
    with open(os.path.join(a.out, f"args_{a.phase.replace(',', '_')}.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    torch.manual_seed(a.seed)
    items, docs_all = judge_items(a)
    wmap = json.load(open(a.wrong_map)) if a.wrong_map else {}
    reg_docs = sorted({_get(i, "doc_id") for i in items} | {wrong_of(_get(i, "doc_id"), docs_all, wmap) for i in items}
                      - {None})
    docs = load_docs(a.split_dir)
    tq = task_questions(a.split_dir)
    reader, enc, tok_r, tok_e, itf, lmap = build(a, init=a.interface)
    rt = make_rt(a, reader, enc, tok_r, tok_e, itf)
    rt.docs = docs
    S = D26Doc(a, rt, itf, lmap)
    log(f"D26 doc-registration: {a.encoder} -> {a.reader} | base {a.interface} | parts {sorted(S.ad.parts)} rank "
        f"{a.rank}: {S.ad.n_params()} params/doc | teacher {a.teacher} | judged {len(items)} items / "
        f"{len({_get(i, 'doc_id') for i in items})} docs; registered docs {len(reg_docs)} | budgets {a.budgets}")
    if "register" in a.phase:
        selfq = load_selfq(a.selfq)
        reg_set = reg_docs
        if a.reg_n >= 0 and a.reg_n != a.n:
            b = copy.copy(a)
            b.n = a.reg_n
            its2, _ = judge_items(b)
            reg_set = sorted(set(reg_docs) | {_get(i, "doc_id") for i in its2}
                             | ({wrong_of(_get(i, "doc_id"), docs_all, wmap) for i in its2} - {None}))
        n_beat = 0
        for k, did in enumerate(reg_set):
            path = os.path.join(a.out, "adapters", f"{did}.pt")
            if os.path.exists(path):
                continue
            assert did in selfq, f"no self-questions for {did} in {a.selfq}"
            reg = shuffle_choices([dict(x) for x in selfq[did]][: a.max_items], a.shuffle, bool(a.shuffle_nli))
            assert_no_test_questions(reg, tq)              # LEAKAGE GUARD (hard)
            assert all(x["doc_id"] == did for x in reg)
            t0 = time.time()
            r = S.register(did, reg)
            ho = r["ho_loss"]
            n_beat += bool(ho) and any(v < ho[0] for kk, v in ho.items() if kk > 0)
            torch.save(r, path + ".tmp")
            os.replace(path + ".tmp", path)
            log(f"  [{k + 1}/{len(reg_set)}] registered {did}: {r['n_items']} items (teacher {r['t_teacher_s']}s, "
                f"reg {r['t_reg_s']}) total {time.time() - t0:.0f}s | docs whose held-out loss beats step 0: {n_beat}")
    if "eval" in a.phase:
        from eval_d13 import summarize
        from xenc.d26_read import chat_ids, greedy_ids
        from xenc.doc_tasks import ctx_prompt, q_prompt, score
        missing = [d for d in reg_docs if not os.path.exists(os.path.join(a.out, "adapters", f"{d}.pt"))]
        assert not missing, f"{len(missing)} documents not registered yet, e.g. {missing[:3]}"
        ads = {d: torch.load(os.path.join(a.out, "adapters", f"{d}.pt"), map_location="cpu", weights_only=False)
               for d in reg_docs}
        eb = sorted({int(b) for b in a.budgets.split(",")})
        conds = a.conds.split(",")
        tag = a.split + (f"_{a.eval_tag}" if a.eval_tag else "")
        ck = open(os.path.join(a.out, f"eval_{tag}.ckpt.jsonl"), "w")
        res, t0 = {}, time.time()
        order = sorted(range(len(items)), key=lambda k: _get(items[k], "doc_id"))
        for j, k in enumerate(order):
            it = items[k]
            did = _get(it, "doc_id")
            w = wrong_of(did, docs_all, wmap)
            rec = {"key": _get(it, "key"), "doc_id": did, "wrong": w, "near_dup_train": (_get(it, "meta") or {}).get("near_dup_train")}
            if "closed_book" in conds:
                g, n = greedy_ids(reader, tok_r, a.reader, chat_ids(tok_r, a.reader, SYSTEM_ANSWER, q_prompt(it)), a.budget,
                                  stops=rt.stops)
                rec["closed_book"] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
            if "closed_tail" in conds:
                g, n = rt.gen(did, it, a.budget, read=False)
                rec["closed_tail"] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
            if "text" in conds:
                g, n = greedy_ids(reader, tok_r, a.reader, chat_ids(tok_r, a.reader, SYSTEM_ANSWER, ctx_prompt(it, docs[did])),
                                  a.budget, stops=rt.stops)
                rec["text"] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
            for b in eb:
                if "kv" in conds:
                    g, n = S.gen(did, it, ads[did]["snaps"][b], a.budget)
                    rec[f"kv@{b}"] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
                if "wrong_doc" in conds and w is not None:
                    g, n = S.gen(w, it, ads[w]["snaps"][b], a.budget)
                    rec[f"wrong_doc@{b}"] = {"gen": g, "score": float(score(it, g)), "gen_tokens": n}
            res[k] = rec
            ck.write(json.dumps(rec) + "\n")
            ck.flush()
            if (j + 1) % 20 == 0:
                done = list(res.values())
                log(f"  [{j + 1}/{len(items)}] " + " ".join(
                    f"kv@{b}={sum(r[f'kv@{b}']['score'] for r in done) / len(done):.3f}" for b in eb)
                    + f" | {(time.time() - t0) / len(done):.1f}s/item")
        rows = [res[k] for k in range(len(items))]
        out = {"args": vars(a), "budgets": {}, "cost": cost_summary(ads, items, eb)}
        has_es = all(ads[d].get("ho_loss") for d in reg_docs)
        labels = [(str(b), b, False) for b in eb] + ([(f"es≤{b}", b, True) for b in eb if b != eb[0]] if has_es else [])
        out["es_choice"] = {d: {str(b): es_choice(ads[d].get("ho_loss"), eb, b) for b in eb} for d in reg_docs} if has_es else {}
        for lab, b, es in labels:
            rb, base_pair = [], []
            for r in rows:
                x = {c: r[c] for c in ("closed_book", "closed_tail", "text") if c in r}
                bk = es_choice(ads[r["doc_id"]].get("ho_loss"), eb, b) if es else b
                if f"kv@{bk}" in r:
                    x["kv"] = r[f"kv@{bk}"]
                    base_pair.append(r[f"kv@{bk}"]["score"] - r[f"kv@{eb[0]}"]["score"])
                if r.get("wrong") is not None:
                    bw = es_choice(ads[r["wrong"]].get("ho_loss"), eb, b) if es else b
                    if f"wrong_doc@{bw}" in r:
                        x["wrong_doc"] = r[f"wrong_doc@{bw}"]
                rb.append(x)
            s = summarize(rb, [c for c in ("closed_book", "closed_tail", "text", "kv", "wrong_doc") if c in conds])
            if b != eb[0] and base_pair:
                mu = sum(base_pair) / len(base_pair)
                sd = (sum((p - mu) ** 2 for p in base_pair) / max(len(base_pair) - 1, 1)) ** 0.5
                s["kv_minus_kv_base"] = {"paired_gap": mu, "se": sd / len(base_pair) ** 0.5, "n_paired": len(base_pair),
                                         "wins": sum(p > 0 for p in base_pair), "losses": sum(p < 0 for p in base_pair)}
            out["budgets"].setdefault("all", {})[lab] = s
        out["items"] = rows
        json.dump(out, open(os.path.join(a.out, f"eval_{tag}.json"), "w"), indent=1)
        a2 = copy.copy(a)
        a2.recite = 0
        md = table_md(out, a2).replace("## D20 ", f"## D26 doc-registration ({a.reader}, teacher {a.teacher}) ")
        open(os.path.join(a.out, "summary.md" if not a.eval_tag else f"summary_{a.eval_tag}.md"), "w").write(md)
        log(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
