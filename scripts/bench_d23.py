"""D23: real-GPU efficiency benchmark of the paper-form read (small encoder operator + per-document registration,
large reader) against the text baselines. Benchmark only: no new accuracy claims for kv (accuracy comes from the
existing judged files, see outputs/D23_EFFICIENCY.md); the only accuracy this script produces is for the BM25 chunk-RAG
BASELINE (`rag_acc`), which has no judged number yet.

Methods (all on the same GPU, same documents / questions, bf16 weights, HF transformers as the repo runs them):
  text      the reader prefills document + question for every question (no cache)
  pcache    text + prefix caching: the reader prefills the document ONCE; per question it continues from its own cache
            (full-attention K/V + Gated-DeltaNet conv/recurrent states). Reuse is copy-free: the DeltaNet states are
            snapshotted and restored, the attention K/V cropped back (restore time reported separately)
  rag       BM25 over fixed-size token chunks of the document, top-k chunks (document order) + question in context
  kv        the paper-form read AS IMPLEMENTED (xenc.d13_read / d12p_read D13Reader "student" mode, fp32 read)
            with the registered per-document adapter attached (D20 parts dA,gate rank 16)
  kv_fast   the same read, batched bf16 implementation (xenc/d23_fastread.py; equivalence: tests + `validate`)
  One-time per document: pcache = reader document prefill; kv = encoder prefill (operator harvest) + registration
  (self-questions by the reader, text-teacher targets, ~200 adapter steps); rag = chunking + BM25 index.

Phases (--phases, comma list, run in this order):
  validate  (9B) greedy agreement fast vs as-implemented vs judged D20 rows (LongHealth f0, registered adapters);
            pcache vs text agreement; restore correctness
  harvest   encoder prefill + operator capture per document length (time, peak memory, bytes; H2D copy time)
  lat       batch-1 latency: TTFT / prefill, per-token decode, e2e at 6 (judged mean answer length) and 32 tokens,
            peak memory, FLOPs (torch FlopCounterMode, one prefill + one decode step) for every method
  reg       registration cost per length: self-question generation, teacher targets, adapter step time (x200)
  batch     throughput (questions/s) at batch sizes --batches (B different documents of the same length)
  lifetime  n real questions (real stopping, budget 32) end to end on one document, to check the composed costs
  rag_acc   BM25 chunk-RAG accuracy on LongHealth f0 (n=80) and QuALITY test (n=300, eval_d13's seed-0 sample)
  enc_other one-time harvest with other encoders (--other_encoders, e.g. Qwen/Qwen3.5-2B)

    python scripts/bench_d23.py --out outputs/D23_eff_9b                       # 9B reader, 4B encoder (D13_4b base)
    python scripts/bench_d23.py --reader Qwen/Qwen3.5-27B --runtime d12p --interface outputs/D12p_4b_s2/d13_best.pt \
        --B outputs/D10_maps/encB8_27b_4b_post.pt --layers 3,7,11,15,19,23,27,31 --phases harvest,lat,reg,batch,rag_acc \
        --out outputs/D23_eff_27b
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import statistics as st
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
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402

TASKS = ("shakespeare_cf", "longhealth", "contractnli", "lastingbench", "quality", "loogle", "clipper", "qasper",
         "musique", "shakespeare")
DEFAULT_DOCS = ("2k:quality_380c2df08912,8k:quality_67c8a9cb38f1,lh12k:longhealth_patient_02,"
                "16k:loogle_b9faf4fc6659,32k:loogle_cbeab30e24d7,64k:loogle_78d3ed194af9")


def sync():
    torch.cuda.synchronize()


def now():
    return time.perf_counter()


def med(x):
    return float(st.median(x)) if x else None


def task_of(did: str) -> str:
    for t in TASKS:
        if did.startswith(t + "_"):
            return t
    raise KeyError(did)


def items_for(did: str) -> list:
    sd = f"data/{task_of(did)}_split"
    out = []
    for sp in ("test", "val", "train"):
        try:
            out += [i for i in load_split_items(sd, sp) if _get(i, "doc_id") == did]
        except Exception:
            pass
    return out


class Rows:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def add(self, **kw):
        kw["t"] = time.strftime("%H:%M:%S")
        with open(self.path, "a") as f:
            f.write(json.dumps(kw) + "\n")
        log("  ROW " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in kw.items()
                                   if not isinstance(v, (list, dict))})[:600])


# ============================================================================ cache helpers (Qwen3.5 hybrid cache)
def cache_bytes(cache) -> dict:
    a = c = r = 0
    for L in cache.layers:
        k = getattr(L, "keys", None)
        if k is not None and getattr(L, "is_initialized", False) and k.numel():
            a += k.nbytes + L.values.nbytes
        if getattr(L, "is_conv_states_initialized", False):
            c += L.conv_states.nbytes
        if getattr(L, "is_recurrent_states_initialized", False):
            r += L.recurrent_states.nbytes
    return {"attn_kv": a, "conv": c, "recurrent": r, "total": a + c + r}


def snap_states(cache):
    out = []
    for i, L in enumerate(cache.layers):
        cs = L.conv_states.clone() if getattr(L, "is_conv_states_initialized", False) else None
        rs = L.recurrent_states.clone() if getattr(L, "is_recurrent_states_initialized", False) else None
        if cs is not None or rs is not None:
            out.append((i, cs, rs))
    return out


def restore(cache, snap, n: int):
    """Copy-free reuse of a document prefix cache: DeltaNet states back to the snapshot, attention K/V cropped to n."""
    for i, cs, rs in snap:
        L = cache.layers[i]
        if cs is not None:
            L.conv_states.copy_(cs)
        if rs is not None:
            L.recurrent_states.copy_(rs)
    for L in cache.layers:
        k = getattr(L, "keys", None)
        if k is not None and getattr(L, "is_initialized", False) and k.numel() and k.shape[-2] > n:
            L.keys = L.keys[..., :n, :]
            L.values = L.values[..., :n, :]


def q4_bytes(n_elems_per_tensor: list, scales_per_tensor: list) -> int:
    """4-bit symmetric per-channel quantization (D14 accounting): 0.5 B / element + fp16 scales."""
    return int(sum(n * 0.5 + 2 * s for n, s in zip(n_elems_per_tensor, scales_per_tensor)))


# ============================================================================ timing core
def forced_run(prefill, step, G: int):
    """Greedy forced decode (no early stop): -> (ttft_s, [step_s]). prefill() -> out; step(nxt, out) -> out."""
    sync()
    t0 = now()
    out = prefill()
    nxt = out.logits[:, -1].argmax(-1)
    sync()
    ttft = now() - t0
    steps = []
    for _ in range(G - 1):
        t = now()
        out = step(nxt, out)
        nxt = out.logits[:, -1].argmax(-1)
        sync()
        steps.append(now() - t)
    del out
    return ttft, steps


def measure(name, setup, prefill, step, teardown, G, reps, warm, flops_fn=None):
    """Median over reps of TTFT, per-token decode, e2e@6 / e2e@G; peak memory over one rep."""
    res = {"ttft": [], "tpot": [], "e2e6": [], "e2eG": [], "teardown": []}
    for k in range(warm + reps):
        setup()
        if k == warm:
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        ttft, steps = forced_run(prefill, step, G)
        if k == warm:
            peak = torch.cuda.max_memory_allocated()
        sync()
        t = now()
        teardown()
        sync()
        if k >= warm:
            res["ttft"].append(ttft)
            res["tpot"].append(med(steps))
            res["e2e6"].append(ttft + sum(steps[:5]))
            res["e2eG"].append(ttft + sum(steps))
            res["teardown"].append(now() - t)
    out = {k: med(v) for k, v in res.items()}
    out["ttft_all"] = [round(x, 5) for x in res["ttft"]]
    out["peak_total_GiB"] = peak / 2 ** 30
    out["peak_working_GiB"] = (peak - base) / 2 ** 30
    out["resident_before_GiB"] = base / 2 ** 30
    if flops_fn is not None:
        out.update(flops_fn())
    return out


def count_flops(fn) -> float:
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        fn()
    return float(fc.get_total_flops())


# ============================================================================ runtime
class _StopForward(Exception):
    pass


class _PreRt:
    """Encoder-only mode: the document-prefix tokenization of D13Rt (reader chat template) without loading the reader."""

    def __init__(self, tok, hub, max_tokens, Tc):
        self.tok, self.hub, self.max_tokens, self.Tc = tok, hub, max_tokens, Tc
        self.docs, self._pre = {}, {}

    def pre(self, did):
        if did not in self._pre:
            from xenc.state_skill import doc_prefix_text
            ids = [int(x) for x in self.tok(doc_prefix_text(self.tok, self.hub, self.docs[did], SYSTEM_ANSWER),
                                            add_special_tokens=False)["input_ids"]]
            self._pre[did] = ids[-(self.max_tokens - 1024):]
        return self._pre[did]


class Bench:
    def __init__(self, a):
        self.a = a
        self.docs, self.ops = {}, {}
        if a.encoder_only:                     # harvest only (cheap-hardware angle): encoder + tokenizer, no reader
            from transformers import AutoTokenizer
            from xenc.backbone import input_device, load_encoder
            from xenc.d13_read import EncTeacher
            self.enc = load_encoder(a.encoder, device_map="cuda")[0]
            self.tok = AutoTokenizer.from_pretrained(a.reader)
            self.reader = self.itf = self.S = None
            self.rt = _PreRt(self.tok, a.reader, a.max_tokens, EncTeacher(self.enc, [int(x) for x in a.enc_layers.split(",")]))
            self.dev = input_device(self.enc)
            return
        from register_d20 import D20
        ns = argparse.Namespace(
            reader=a.reader, encoder=a.encoder, B=a.B, layers=a.layers, interface=a.interface, device="cuda",
            runtime=a.runtime, enc_device=None, reader_mem=None, itf_device=None, mem_safe=1, tail=64,
            max_tokens=a.max_tokens, parts="dA,gate", rank=16, kv_rank=8, seed=0, kl=1.0, ce=0.5, dist_w=1.0,
            rel_w=1.0, l2=0.0, batch=2, lr=1e-3, lr_gate=3e-3, lr_bias=None, clip=1.0, warmup=25, holdout=0.0,
            budgets="0,2,12", ans_budget=40, conf=0.0, shuffle=2, enc_only=0, teacher="text")
        t0 = now()
        self.S = D20(ns)
        log(f"loaded reader+encoder+interface in {now() - t0:.0f}s")
        S = self.S
        self.reader, self.enc, self.tok, self.itf, self.rt = S.reader, S.enc, S.tok, S.itf, S.rt
        self.R = self.rt.R
        from xenc.d23_fastread import FastRead
        self.F = FastRead(self.reader, self.itf, dtype=torch.bfloat16)
        from xenc.backbone import input_device
        self.dev = input_device(self.reader)

    def doc_text(self, did):
        if did not in self.docs:
            from xenc.doc_tasks import load_docs
            self.docs.update(load_docs(f"data/{task_of(did)}_split"))
        self.rt.docs = self.docs
        return self.docs[did]

    def question(self, did, k: int = 0, pick="median"):
        its = items_for(did)
        its = sorted(its, key=lambda i: len(self.rt.suf(i)))
        if pick == "median":
            return its[len(its) // 2]
        return its[k % len(its)]

    # -------------------------------------------------------------- operator harvest (encoder one-time cost)
    @torch.no_grad()
    def harvest(self, did, enc=None, Tc=None, trunc: bool = True):
        """Operator harvest: encoder prefill of the document prefix minus the 64-token tail, capturing post-RoPE K/V at
        the read layers. trunc=True stops the forward after the deepest captured layer (early-exit hook): deeper
        encoder layers cannot affect the operator, so the result is identical (checked in `harvest`)."""
        enc = enc or self.enc
        Tc = Tc or self.rt.Tc
        ids = self.rt.pre(did)[:-64]
        from deltaskill.backbone_utils import decoder_layers
        from xenc.backbone import input_device
        x = torch.tensor([ids], device=input_device(enc))
        Tc.rec, Tc.kv_only = {}, True
        h = None
        if trunc:
            def _stop(mod, inp, out):
                raise _StopForward()
            h = decoder_layers(enc)[max(Tc.layers)].register_forward_hook(_stop)
        try:
            enc(input_ids=x, use_cache=False, logits_to_keep=1)
        except _StopForward:
            pass
        finally:
            if h is not None:
                h.remove()
        op = {e: (r["k"], r["v"]) for e, r in Tc.rec.items()}
        Tc.rec, Tc.kv_only = None, False
        return op, len(ids)

    def op_gpu(self, did):
        op, n = self.ops[did]
        return {e: (K.to(self.dev, non_blocking=True), V.to(self.dev, non_blocking=True)) for e, (K, V) in op.items()}, n


# ============================================================================ phases
def phase_validate(B: Bench, rows: Rows, a):
    """Greedy agreement on LongHealth f0 (registered D20 adapters at 200 steps): as-implemented vs fast vs judged."""
    from register_state import _stop_ids, greedy
    from xenc.backbone import decode
    from xenc.doc_tasks import score
    S, rt, tok = B.S, B.rt, B.tok
    judged = {r["key"]: r for r in json.load(open(a.validate_eval))["items"]}
    items = [i for i in load_split_items("data/longhealth_split", "test") if (_get(i, "meta") or {}).get("fold") == 0]
    by = {}
    for i in items:
        by.setdefault(_get(i, "doc_id"), []).append(i)
    sel = []
    for did in sorted(by)[: a.validate_docs]:
        sel += by[did][: a.validate_q]
    stops = _stop_ids(tok)

    def fast_gen(did, it, budget=32):
        rows_ = rt.rows(did, it)
        op, n_op = B.op_gpu(did)
        B.F.load(op, n_op)
        B.F.on = True
        try:
            out = B.reader(input_ids=torch.tensor([rows_], device=B.dev), use_cache=True, logits_to_keep=1)
            new = []
            for _ in range(budget):
                nxt = int(out.logits[0, -1].argmax())
                new.append(nxt)
                if nxt in stops or "</answer>" in tok.decode(new[-12:], skip_special_tokens=True):
                    break
                out = B.reader(input_ids=torch.tensor([[nxt]], device=B.dev), past_key_values=out.past_key_values,
                               use_cache=True, logits_to_keep=1)
        finally:
            B.F.on = False
            B.F.reset()
        txt = decode(tok, a.reader, new)
        return (txt.split("</answer>", 1)[0] + "</answer>") if "</answer>" in txt else txt

    agree_fi = agree_ij = agree_fj = 0
    sc = {"impl": 0.0, "fast": 0.0, "judged": 0.0, "text": 0.0, "pcache": 0.0}
    pc_agree = 0
    cur = None
    for it in sel:
        did = _get(it, "doc_id")
        B.doc_text(did)
        if did != cur:
            ad = torch.load(os.path.join(a.validate_adapters, f"{did}.pt"), map_location="cpu", weights_only=False)
            snap = ad["snaps"][200]
            op, n = B.harvest(did)
            B.ops[did] = ({e: (K.cpu(), V.cpu()) for e, (K, V) in op.items()}, n)
            del op
            pre = rt.pre(did)
            C = B.reader(input_ids=torch.tensor([pre], device=B.dev), use_cache=True, logits_to_keep=1).past_key_values
            snp = snap_states(C)
            cur = did
        g_impl, _ = S.gen(did, it, snap, 32)
        S.ad.load(snap)
        g_fast = fast_gen(did, it)
        g_j = judged[_get(it, "key")]["kv@200"]["gen"]
        g_text, _ = greedy(B.reader, tok, a.reader, pre + rt.suf(it), 32)
        g_pc, _ = greedy(B.reader, tok, a.reader, rt.suf(it), 32, cache=C)
        restore(C, snp, len(pre))
        agree_fi += g_fast == g_impl
        agree_ij += g_impl == g_j
        agree_fj += g_fast == g_j
        pc_agree += g_pc == g_text
        for k, g in (("impl", g_impl), ("fast", g_fast), ("judged", g_j), ("text", g_text), ("pcache", g_pc)):
            sc[k] += float(score(it, g))
    n = len(sel)
    rows.add(phase="validate", n=n, agree_fast_impl=agree_fi / n, agree_impl_judged=agree_ij / n,
             agree_fast_judged=agree_fj / n, agree_pcache_text=pc_agree / n,
             acc={k: v / n for k, v in sc.items()})
    S.ad.reset()
    B.ops.clear()


def phase_harvest(B: Bench, rows: Rows, a, docs):
    for tag, did in docs:
        B.doc_text(did)
        N = len(B.rt.pre(did))
        full = _time_harvest(B, did, a, trunc=False)
        times, peak, base = [], None, None
        for k in range(a.warmup + a.reps):
            torch.cuda.empty_cache()
            if k == a.warmup:
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
            sync()
            t = now()
            op, n = B.harvest(did, trunc=True)
            sync()
            if k == a.warmup:
                peak = torch.cuda.max_memory_allocated()
            if k >= a.warmup:
                times.append(now() - t)
            if k < a.warmup + a.reps - 1:
                del op
        fl = count_flops(lambda: B.harvest(did, trunc=True)) if a.flops else None
        op_full, _ = B.harvest(did, trunc=False)
        diff = max(float((op[e][i].float() - op_full[e][i].float()).abs().max()) for e in op for i in (0, 1))
        del op_full
        nbytes = sum(K.nbytes + V.nbytes for K, V in op.values())
        Hkv, hd = next(iter(op.values()))[0].shape[0], next(iter(op.values()))[0].shape[2]
        nel = [K.numel() for K, V in op.values() for _ in (0, 1)]
        q4 = q4_bytes(nel, [Hkv * hd] * len(nel))
        cpu = {e: (K.cpu().pin_memory(), V.cpu().pin_memory()) for e, (K, V) in op.items()}
        B.ops[did] = (cpu, n)
        del op
        torch.cuda.empty_cache()
        h2d = []
        for _ in range(3):
            sync()
            t = now()
            g = {e: (K.to(B.dev, non_blocking=True), V.to(B.dev, non_blocking=True)) for e, (K, V) in cpu.items()}
            sync()
            h2d.append(now() - t)
            del g
        rows.add(phase="harvest", L=tag, doc=did, N=N, n_op=n, encoder=a.encoder, enc_layers=sorted(cpu),
                 time_s=med(times), times=times, trunc=True, trunc_after_layer=max(cpu), full_time_s=full["time_s"],
                 full_flops=full["flops"], full_peak_working_GiB=full["peak_working_GiB"], trunc_vs_full_maxdiff=diff,
                 peak_total_GiB=peak / 2 ** 30, peak_working_GiB=(peak - base) / 2 ** 30,
                 op_bytes_bf16=nbytes, op_bytes_q4=q4, op_bytes_q4_acc25=int(q4 * 0.25), op_bytes_q4_acc10=int(q4 * 0.10),
                 h2d_s=med(h2d), flops=fl,
                 raw_text_bytes=len(B.docs[did].encode("utf-8")), adapter_params=B.S.ad.n_params() if B.S else None,
                 itf_params=sum(p.numel() for p in B.itf.parameters()) if B.itf is not None else None,
                 itf_bytes=sum(p.numel() * p.element_size() for p in B.itf.parameters()) if B.itf is not None else None,
                 gpu=torch.cuda.get_device_name(0))


def _time_harvest(B, did, a, trunc, enc=None, Tc=None):
    times, peak, base = [], 0, 0
    for k in range(a.warmup + a.reps):
        torch.cuda.empty_cache()
        if k == a.warmup:
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        sync()
        t = now()
        op, _ = B.harvest(did, enc=enc, Tc=Tc, trunc=trunc)
        sync()
        if k == a.warmup:
            peak = torch.cuda.max_memory_allocated()
        if k >= a.warmup:
            times.append(now() - t)
        del op
    fl = count_flops(lambda: B.harvest(did, enc=enc, Tc=Tc, trunc=trunc)) if a.flops else None
    return {"time_s": med(times), "flops": fl, "peak_working_GiB": (peak - base) / 2 ** 30}


def rag_prompt(B: Bench, did, it, chunk: int, k: int, cache: dict):
    from xenc.backbone import build_chat
    from xenc.d18_library import BM25
    from xenc.doc_tasks import ctx_prompt, q_prompt
    key = (did, chunk)
    t_index = None
    if key not in cache:
        t = now()
        ids = B.tok(B.doc_text(did), add_special_tokens=False)["input_ids"]
        chunks = {f"c{j:05d}": B.tok.decode(ids[s: s + chunk]) for j, s in enumerate(range(0, len(ids), chunk))}
        cache[key] = (chunks, BM25(chunks))
        t_index = now() - t
    chunks, bm = cache[key]
    t = now()
    sc = bm.scores(q_prompt(it))
    top = sorted(sorted(sc, key=lambda c: -sc[c])[:k])
    t_ret = now() - t
    ctx = "\n\n[...]\n\n".join(chunks[c] for c in top)
    return build_chat(B.tok, B.a.reader, SYSTEM_ANSWER, ctx_prompt(it, ctx)), t_ret, t_index


def phase_lat(B: Bench, rows: Rows, a, docs, methods):
    rd, dev, G = B.reader, B.dev, a.gen
    rag_cache = {}

    def step(nxt, out):
        return rd(input_ids=nxt.view(-1, 1), past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)

    for tag, did in docs:
        B.doc_text(did)
        it = B.question(did)
        pre, suf = B.rt.pre(did), B.rt.suf(it)
        N = len(pre)
        log(f"[lat] {tag} {did}: N={N} q={len(suf)}")
        common = dict(phase="lat", L=tag, doc=did, N=N, q_tokens=len(suf), G=G)
        nop = lambda: None  # noqa: E731
        for m in methods:
            torch.cuda.empty_cache()
            try:
                if m == "text":
                    x = torch.tensor([pre + suf], device=dev)
                    pf = lambda: rd(input_ids=x, use_cache=True, logits_to_keep=1)  # noqa: E731
                    fl = (lambda: {"flops_prefill": count_flops(pf)}) if a.flops else None
                    r = measure(m, nop, pf, step, nop, G, a.reps, a.warmup, fl)
                    rows.add(method=m, ctx_tokens=N + len(suf), **common, **r)
                elif m == "pcache":
                    xp = torch.tensor([pre], device=dev)
                    once = []
                    for _ in range(a.warmup + 2):
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        base0 = torch.cuda.memory_allocated()
                        sync()
                        t = now()
                        C = rd(input_ids=xp, use_cache=True, logits_to_keep=1).past_key_values
                        sync()
                        once.append(now() - t)
                        once_peak = (torch.cuda.max_memory_allocated() - base0) / 2 ** 30
                        if len(once) < a.warmup + 2:
                            del C
                    cb = cache_bytes(C)
                    snp = snap_states(C)
                    xs = torch.tensor([suf], device=dev)
                    pf = lambda: rd(input_ids=xs, past_key_values=C, use_cache=True, logits_to_keep=1)  # noqa: E731
                    td = lambda: restore(C, snp, N)  # noqa: E731

                    def fl():
                        f1 = count_flops(pf)
                        restore(C, snp, N)
                        f0 = count_flops(lambda: rd(input_ids=xp, use_cache=False, logits_to_keep=1))
                        return {"flops_prefill": f1, "flops_once": f0}
                    r = measure(m, nop, pf, step, td, G, a.reps, a.warmup, fl if a.flops else None)
                    t = now()
                    C2 = copy.deepcopy(C)
                    sync()
                    t_deep = now() - t
                    del C2
                    # H2D of the stored prefix cache (pinned CPU -> GPU), i.e. swapping a document in
                    cpu = [(L.keys.cpu().pin_memory(), L.values.cpu().pin_memory()) for L in C.layers
                           if getattr(L, "keys", None) is not None and getattr(L, "is_initialized", False) and L.keys.numel()]
                    sts = [s.cpu().pin_memory() for _, cs, rs in snp for s in (cs, rs) if s is not None]
                    h2d = []
                    for _ in range(3):
                        sync()
                        t = now()
                        g = [(k.to(dev, non_blocking=True), v.to(dev, non_blocking=True)) for k, v in cpu] + \
                            [s.to(dev, non_blocking=True) for s in sts]
                        sync()
                        h2d.append(now() - t)
                        del g
                    del cpu, sts
                    rows.add(method=m, ctx_tokens=N + len(suf), once_s=med(once[a.warmup:]), once_times=once[a.warmup:],
                             once_peak_working_GiB=once_peak, cache_bytes=cb,
                             deepcopy_s=t_deep, h2d_s=med(h2d), **common, **r)
                    del C, snp
                elif m.startswith("rag"):
                    chunk, k = (int(v) for v in m[3:].split("x"))
                    ids, t_ret, t_idx = rag_prompt(B, did, it, chunk, k, rag_cache)
                    x = torch.tensor([ids], device=dev)
                    pf = lambda: rd(input_ids=x, use_cache=True, logits_to_keep=1)  # noqa: E731
                    fl = (lambda: {"flops_prefill": count_flops(pf)}) if a.flops else None
                    r = measure(m, nop, pf, step, nop, G, a.reps, a.warmup, fl)
                    rets = [rag_prompt(B, did, it, chunk, k, rag_cache)[1] for _ in range(5)]
                    rows.add(method=m, ctx_tokens=len(ids), retrieval_s=med(rets), index_s=t_idx, **common, **r)
                elif m in ("kv", "kv_fast"):
                    if did not in B.ops:
                        op, n = B.harvest(did)
                        B.ops[did] = ({e: (K.cpu().pin_memory(), V.cpu().pin_memory()) for e, (K, V) in op.items()}, n)
                        del op
                    op, n_op = B.op_gpu(did)
                    rws = B.rt.rows(did, it)
                    x = torch.tensor([rws], device=dev)
                    if m == "kv":
                        def setup():
                            B.R.load(op, n_op)
                            B.R.mode = "student"

                        def td():
                            B.R.mode, B.R.past = "off", {}
                    else:
                        def setup():
                            B.F.load(op, n_op)
                            B.F.on = True

                        def td():
                            B.F.on = False
                            B.F.reset()
                    pf = lambda: rd(input_ids=x, use_cache=True, logits_to_keep=1)  # noqa: E731

                    def fl():
                        setup()
                        f1 = count_flops(pf)
                        td()
                        setup()
                        o = pf()
                        f2 = count_flops(lambda: step(o.logits[:, -1].argmax(-1), o))
                        td()
                        return {"flops_prefill": f1, "flops_decode_step": f2}
                    r = measure(m, setup, pf, step, td, G, a.reps, a.warmup, fl if a.flops else None)
                    rows.add(method=m, ctx_tokens=len(rws), n_op=n_op, op_resident_bytes=sum(K.nbytes + V.nbytes for K, V in op.values()),
                             **common, **r)
                    del op
                if m in ("text",) and a.flops:      # decode-step FLOPs over a cache of N (text = pcache decode)
                    o = rd(input_ids=torch.tensor([pre + suf], device=dev), use_cache=True, logits_to_keep=1)
                    f2 = count_flops(lambda: step(o.logits[:, -1].argmax(-1), o))
                    rows.add(phase="flops_decode", method="text/pcache", L=tag, N=N, flops_decode_step=f2)
                    del o
            except torch.cuda.OutOfMemoryError as e:
                B.R.mode, B.R.past, B.F.on = "off", {}, False
                B.F.reset()
                torch.cuda.empty_cache()
                rows.add(method=m, oom=True, err=str(e)[:200], **common)


def _selfq(model, tok, hub, did, text, N, dev):
    from d20_selfq import gen_streams, parse
    items = []
    for tmpl in ("mcq", "mix", "nli"):
        outs = gen_streams(model, tok, hub, text, tmpl, 6, 380, 0.9, 0.95, N + 10, dev)
        for o in outs:
            for q in parse(tmpl, o):
                q["meta"] = {"d20selfq": True, "template": tmpl}
                items.append(q)
    return [{"key": f"{did}_d23s{j}", "doc_id": did, **q, "task": "d20selfq", "hop": None} for j, q in enumerate(items[:48])]


def phase_reg(B: Bench, rows: Rows, a, docs):
    """Registration cost (label-free, from the document), per length:
      selfq_reader_s   self-questions written by the READER (it reads the document: D20's default source)
      selfq_enc_s      the same generator run on the small ENCODER (cost only: the reader never reads the document;
                       question quality from the 4B is untested)
      teacher=text     D20 default: text-teacher targets (the READER prefills the document once + answers every item),
                       then steps with KL + 0.5 CE + distillation
      teacher=none     D20 --enc_only 1: encoder distillation only (no reader document pass)
    Steps 2..12 timed -> s/step; x200 = the D20 default budget."""
    from register_d20 import shuffle_choices
    S = B.S
    from xenc.backbone import input_device
    for tag, did in docs:
        text = B.doc_text(did)
        N = len(B.rt.pre(did))
        row = dict(phase="reg", L=tag, doc=did, N=N)
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            sync()
            t = now()
            items = _selfq(B.reader, B.tok, a.reader, did, text, N, B.dev)
            sync()
            row.update(selfq_reader_s=now() - t, n_selfq=len(items),
                       peak_selfq_reader_GiB=torch.cuda.max_memory_allocated() / 2 ** 30)
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                sync()
                t = now()
                its_e = _selfq(B.enc, B.tok, a.encoder, did, text, N, input_device(B.enc))
                sync()
                row.update(selfq_enc_s=now() - t, n_selfq_enc=len(its_e),
                           peak_selfq_enc_GiB=torch.cuda.max_memory_allocated() / 2 ** 30)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                row.update(selfq_enc_oom=str(e)[:120])
            reg = shuffle_choices(items, 2, False)
            row["n_reg_items"] = len(reg)
            for mode in ("text", "none"):
                S.a.enc_only, S.a.teacher = (0, "text") if mode == "text" else (1, "none")
                S.a.kl, S.a.ce = (1.0, 0.5) if mode == "text" else (0.0, 0.0)
                pfx = "" if mode == "text" else "enc_"
                try:
                    B.rt._ops.clear()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    r = S.register(did, reg)
                    tr = r["t_reg_s"]
                    s_step = (tr[12] - tr[2]) / 10 if tr.get(12) else None
                    row.update({f"{pfx}n_targets": r["n_items"], f"{pfx}teacher_s": r["t_teacher_s"],
                                f"{pfx}s_per_step": s_step, f"{pfx}reg200_s": (s_step * 200 if s_step else None),
                                f"{pfx}peak_reg_GiB": torch.cuda.max_memory_allocated() / 2 ** 30})
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    row[f"{pfx}oom"] = str(e)[:160]
                finally:
                    S.ad.reset()
                    B.R.mode, B.R.rec = "off", {}
                    B.rt._ops.clear()
            row["adapter_params"] = S.ad.n_params()
            if row.get("reg200_s") is not None:
                row["total200_s"] = row["selfq_reader_s"] + row["teacher_s"] + row["reg200_s"]
            if row.get("enc_reg200_s") is not None and row.get("selfq_enc_s") is not None:
                row["enc_total200_s"] = row["selfq_enc_s"] + row["enc_teacher_s"] + row["enc_reg200_s"]
            rows.add(**row)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            rows.add(**row, oom=True, err=str(e)[:200])
        finally:
            S.a.enc_only, S.a.teacher, S.a.kl, S.a.ce = 0, "text", 1.0, 0.5
            S.ad.reset()
            B.R.mode, B.R.rec = "off", {}
            B.rt._ops.clear()


def _batch_one(B: Bench, a, m, bs, did, it, pre, suf, rag_cache, snaps):
    """One (method, batch size) throughput point; every large tensor is local and freed on return."""
    rd, dev, G = B.reader, B.dev, a.bgen
    nop = lambda: None  # noqa: E731
    st_ = {}

    def step(nxt, out):
        return rd(input_ids=nxt.view(-1, 1), past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)

    if m == "text":
        x = torch.tensor([pre + suf] * bs, device=dev)
        pf = lambda: rd(input_ids=x, use_cache=True, logits_to_keep=1)  # noqa: E731
        setup, td = nop, nop
    elif m.startswith("rag"):
        chunk, k = (int(v) for v in m[3:].split("x"))
        x = torch.tensor([rag_prompt(B, did, it, chunk, k, rag_cache)[0]] * bs, device=dev)
        pf = lambda: rd(input_ids=x, use_cache=True, logits_to_keep=1)  # noqa: E731
        setup, td = nop, nop
    elif m == "pcache":
        C0 = rd(input_ids=torch.tensor([pre], device=dev), use_cache=True, logits_to_keep=1).past_key_values
        xs = torch.tensor([suf] * bs, device=dev)

        def setup():
            st_["cache"] = None
            C = copy.deepcopy(C0)
            if bs > 1:                             # B documents' prefix caches (HF has no shared-prefix paging);
                C.reorder_cache(torch.zeros(bs, dtype=torch.long, device=dev))   # LinearAttentionLayer lacks repeat
            st_["cache"] = C

        def td():
            st_["cache"] = None
        pf = lambda: rd(input_ids=xs, past_key_values=st_["cache"], use_cache=True, logits_to_keep=1)  # noqa: E731
    elif m == "kv_fast":
        if did not in B.ops:
            op, n = B.harvest(did)
            B.ops[did] = ({e: (K.cpu().pin_memory(), V.cpu().pin_memory()) for e, (K, V) in op.items()}, n)
            del op
        op1, n_op = B.op_gpu(did)
        opb = {e: (K.unsqueeze(0).expand(bs, *K.shape).contiguous(), V.unsqueeze(0).expand(bs, *V.shape).contiguous())
               for e, (K, V) in op1.items()}          # B DIFFERENT documents' operators (memory B x)
        del op1
        from xenc.d23_fastread import adapter_tensors
        dA, gb = adapter_tensors([snaps[j % len(snaps)] for j in range(bs)], B.itf.layers, dev)
        x = torch.tensor([B.rt.rows(did, it)] * bs, device=dev)

        def setup():
            B.F.load(opb, n_op, dA=dA, gb=gb)
            B.F.on = True

        def td():
            B.F.on = False
            B.F.reset()
        pf = lambda: rd(input_ids=x, use_cache=True, logits_to_keep=1)  # noqa: E731
    else:
        raise ValueError(m)
    try:
        return measure(m, setup, pf, step, td, G, a.breps, 1)
    finally:
        B.F.on = False
        B.F.load({}, 0)
        st_.clear()


def phase_batch(B: Bench, rows: Rows, a, docs, methods):
    rag_cache = {}
    snaps = []                                  # per-request registered-adapter tensors (synthetic values, real shapes)
    g = torch.Generator().manual_seed(0)
    for j in range(8):
        s = {k: v.clone() for k, v in B.S.ad.snapshot().items()}
        for k, v in s.items():
            if k.startswith("D.Ua"):
                s[k] = torch.randn(v.shape, generator=g) * 1e-3
        snaps.append(s)
    for tag, did in docs:
        B.doc_text(did)
        it = B.question(did)
        pre, suf = B.rt.pre(did), B.rt.suf(it)
        N = len(pre)
        for m in methods:
            for bs in a.batches:
                torch.cuda.empty_cache()
                common = dict(phase="batch", method=m, L=tag, doc=did, N=N, B=bs, G=a.bgen)
                try:
                    r = _batch_one(B, a, m, bs, did, it, pre, suf, rag_cache, snaps)
                    rows.add(**common, qps=bs / r["e2eG"], e2e_s=r["e2eG"], ttft=r["ttft"], tpot=r["tpot"],
                             peak_total_GiB=r["peak_total_GiB"], peak_working_GiB=r["peak_working_GiB"],
                             resident_before_GiB=r["resident_before_GiB"])
                except torch.cuda.OutOfMemoryError as e:
                    rows.add(**common, oom=True, err=str(e)[:160])
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    rows.add(**common, error=repr(e)[:300])
                torch.cuda.empty_cache()


def phase_lifetime(B: Bench, rows: Rows, a, docs):
    """n real questions on one document, real stopping (budget 32), wall clock incl. one-time costs (registration
    excluded: reported by `reg`)."""
    from register_state import greedy
    from xenc.backbone import build_chat
    from xenc.doc_tasks import ctx_prompt
    S, rt = B.S, B.rt
    rag_cache = {}
    for tag, did in docs:
        if tag != a.lifetime_L:
            continue
        B.doc_text(did)
        its = items_for(did)
        qs = [its[j % len(its)] for j in range(a.lifetime_n)]
        pre = rt.pre(did)
        N = len(pre)
        res = {}
        for m in ("text", "pcache", "rag512x4", "kv", "kv_fast"):
            torch.cuda.empty_cache()
            sync()
            t0 = now()
            gens = 0
            if m == "text":
                for it in qs:
                    _, n = greedy(B.reader, B.tok, a.reader, pre + rt.suf(it), 32)
                    gens += n
            elif m == "pcache":
                C = B.reader(input_ids=torch.tensor([pre], device=B.dev), use_cache=True, logits_to_keep=1).past_key_values
                snp = snap_states(C)
                for it in qs:
                    _, n = greedy(B.reader, B.tok, a.reader, rt.suf(it), 32, cache=C)
                    restore(C, snp, N)
                    gens += n
                del C, snp
            elif m.startswith("rag"):
                for it in qs:
                    ids = rag_prompt(B, did, it, 512, 4, rag_cache)[0]
                    _, n = greedy(B.reader, B.tok, a.reader, ids, 32)
                    gens += n
            else:
                op, n_op = B.harvest(did)
                for it in qs:
                    if m == "kv":
                        rt._ops[did] = (op, n_op, None)
                        _, n = S.gen(did, it, S.ad.snapshot(), 32)
                    else:
                        from register_state import _stop_ids
                        B.F.load(op, n_op)
                        B.F.on = True
                        out = B.reader(input_ids=torch.tensor([rt.rows(did, it)], device=B.dev), use_cache=True, logits_to_keep=1)
                        new, stops = [], _stop_ids(B.tok)
                        for _ in range(32):
                            nxt = int(out.logits[0, -1].argmax())
                            new.append(nxt)
                            if nxt in stops or "</answer>" in B.tok.decode(new[-12:], skip_special_tokens=True):
                                break
                            out = B.reader(input_ids=torch.tensor([[nxt]], device=B.dev), past_key_values=out.past_key_values,
                                           use_cache=True, logits_to_keep=1)
                        B.F.on = False
                        B.F.reset()
                        n = len(new)
                    gens += n
                rt._ops.clear()
                del op
            sync()
            res[m] = {"wall_s": now() - t0, "mean_gen": gens / len(qs)}
        rows.add(phase="lifetime", L=tag, doc=did, N=N, n=len(qs), res=res)


def phase_rag_acc(B: Bench, rows: Rows, a):
    from register_state import greedy
    from xenc.doc_tasks import score
    rag_cache = {}
    sets = [("longhealth", "data/longhealth_split", 0, 0), ("quality", "data/quality_split", None, 300)]
    for name, sd, fold, n in sets:
        items = load_split_items(sd, "test")
        if fold is not None:
            items = [i for i in items if (_get(i, "meta") or {}).get("fold") == fold]
        if n and len(items) > n:
            items = random.Random(0).sample(items, n)
        for m in a.rag_methods:
            chunk, k = (int(v) for v in m[3:].split("x"))
            per, tot, ntok = [], 0.0, []
            for it in items:
                did = _get(it, "doc_id")
                ids = rag_prompt(B, did, it, chunk, k, rag_cache)[0]
                g, ng = greedy(B.reader, B.tok, a.reader, ids, 32)
                s = float(score(it, g))
                tot += s
                ntok.append(len(ids))
                per.append({"key": _get(it, "key"), "gen": g, "score": s})
            rows.add(phase="rag_acc", task=name, method=m, n=len(items), acc=tot / len(items),
                     mean_prompt_tokens=sum(ntok) / len(ntok), items=per)


def phase_enc_other(B: Bench, rows: Rows, a, docs):
    from xenc.backbone import load_encoder
    from xenc.d13_read import EncTeacher
    from xenc.xattn_read import attn_dims, full_depth_map
    keep = {int(x) for x in a.layers.split(",")}
    for hub in a.other_encoders:
        enc = load_encoder(hub, device_map="cuda")[0]
        lmap = {k: v for k, v in full_depth_map(attn_dims(B.reader), attn_dims(enc)).items() if k in keep}
        Tc = EncTeacher(enc, sorted(set(lmap.values())))
        for tag, did in docs:
            B.doc_text(did)
            times = []
            try:
                for k in range(a.warmup + a.reps):
                    torch.cuda.empty_cache()
                    if k == a.warmup:
                        torch.cuda.reset_peak_memory_stats()
                        base = torch.cuda.memory_allocated()
                    sync()
                    t = now()
                    op, n = B.harvest(did, enc=enc, Tc=Tc)
                    sync()
                    if k == a.warmup:
                        peak = torch.cuda.max_memory_allocated()
                    if k >= a.warmup:
                        times.append(now() - t)
                    if k < a.warmup + a.reps - 1:
                        del op
                fl = count_flops(lambda: B.harvest(did, enc=enc, Tc=Tc)) if a.flops else None
                full = _time_harvest(B, did, a, trunc=False, enc=enc, Tc=Tc)
                Hkv, hd = next(iter(op.values()))[0].shape[0], next(iter(op.values()))[0].shape[2]
                nel = [K.numel() for K, V in op.values() for _ in (0, 1)]
                rows.add(phase="harvest", L=tag, doc=did, N=len(B.rt.pre(did)), n_op=n, encoder=hub,
                         enc_layers=sorted(op), layer_map=lmap, time_s=med(times), times=times, flops=fl, trunc=True,
                         full_time_s=full["time_s"], full_flops=full["flops"],
                         peak_total_GiB=peak / 2 ** 30, peak_working_GiB=(peak - base) / 2 ** 30,
                         op_bytes_bf16=sum(K.nbytes + V.nbytes for K, V in op.values()),
                         op_bytes_q4=q4_bytes(nel, [Hkv * hd] * len(nel)))
                del op
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                rows.add(phase="harvest", L=tag, doc=did, encoder=hub, oom=True, err=str(e)[:160])
        Tc.detach()
        del enc, Tc
        torch.cuda.empty_cache()


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--interface", default="outputs/D13_4b_reg_w10/d13_best.pt")
    ap.add_argument("--runtime", default="d13", choices=("d13", "d12p"))
    ap.add_argument("--max_tokens", type=int, default=72000)
    ap.add_argument("--docs", default=DEFAULT_DOCS, help="tag:doc_id,...")
    ap.add_argument("--only", default="", help="restrict to these tags")
    ap.add_argument("--phases", default="validate,harvest,lat,reg,batch,lifetime,rag_acc,enc_other")
    ap.add_argument("--methods", default="text,pcache,rag512x4,rag512x8,kv,kv_fast")
    ap.add_argument("--batch_methods", default="text,pcache,rag512x4,kv_fast")
    ap.add_argument("--rag_methods", default="rag512x4,rag512x8")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--breps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--bgen", type=int, default=6)
    ap.add_argument("--flops", type=int, default=1)
    ap.add_argument("--validate_eval", default="outputs/D20_4b_lh_f0/eval_test.json")
    ap.add_argument("--validate_adapters", default="outputs/D20_4b_lh_f0/adapters")
    ap.add_argument("--validate_docs", type=int, default=4)
    ap.add_argument("--validate_q", type=int, default=5)
    ap.add_argument("--lifetime_L", default="16k")
    ap.add_argument("--lifetime_n", type=int, default=20)
    ap.add_argument("--other_encoders", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--encoder_only", type=int, default=0, help="1: load only the encoder (+ reader tokenizer); "
                    "phases harvest / enc_other only (e.g. the 4B harvest on an L40S)")
    ap.add_argument("--enc_layers", default="3,7,11,15,19", help="--encoder_only: encoder layers to capture "
                    "(4B: 9B reader 3,7,11,15,19; 27B reader 3,7,11,15)")
    ap.add_argument("--offload_encoder_after", default="", help="phase after which the encoder moves to CPU (27B)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.batches = [int(x) for x in a.batches.split(",") if x]
    a.rag_methods = [m for m in a.rag_methods.split(",") if m]
    a.other_encoders = [m for m in a.other_encoders.split(",") if m]
    docs = [tuple(x.split(":")) for x in a.docs.split(",") if x]
    if a.only:
        keep = set(a.only.split(","))
        docs = [d for d in docs if d[0] in keep]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"args_{time.strftime('%m%d_%H%M%S')}.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    rows = Rows(os.path.join(a.out, "rows.jsonl"))
    B = Bench(a)
    import transformers
    p = torch.cuda.get_device_properties(0)
    if a.encoder_only:
        rows.add(phase="env", gpu=torch.cuda.get_device_name(0), gpu_total_GiB=p.total_memory / 2 ** 30,
                 torch=torch.__version__, transformers=transformers.__version__, encoder_only=True, encoder=a.encoder,
                 enc_layers=a.enc_layers, reader=a.reader, attn_impl=getattr(B.enc.config, "_attn_implementation", None),
                 weights_encoder_GiB=sum(x.numel() * x.element_size() for x in B.enc.parameters()) / 2 ** 30,
                 host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"))
    else:
        rows.add(phase="env", gpu=torch.cuda.get_device_name(0), gpu_total_GiB=p.total_memory / 2 ** 30,
                 torch=torch.__version__, transformers=transformers.__version__,
                 attn_impl=getattr(B.reader.config, "_attn_implementation", None),
                 reader=a.reader, encoder=a.encoder, interface=a.interface, layers=a.layers, layer_map=B.itf.layer_map,
                 weights_reader_GiB=sum(x.numel() * x.element_size() for x in B.reader.parameters()) / 2 ** 30,
                 weights_encoder_GiB=sum(x.numel() * x.element_size() for x in B.enc.parameters()) / 2 ** 30,
                 weights_itf_GiB=sum(x.numel() * x.element_size() for x in B.itf.parameters()) / 2 ** 30,
                 resident_GiB=torch.cuda.memory_allocated() / 2 ** 30,
                 fla=bool(getattr(sys.modules.get("transformers.models.qwen3_5.modeling_qwen3_5"), "chunk_gated_delta_rule", None)),
                 host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"))
    phases = a.phases.split(",")
    for ph in phases:
        t0 = now()
        log(f"==== phase {ph}")
        try:
            run_phase(ph, B, rows, a, docs)
        except Exception as e:                  # keep going: one failing phase must not lose the others
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            rows.add(phase="error", failed_phase=ph, err=repr(e)[:400])
        log(f"==== phase {ph} done in {now() - t0:.0f}s")
        if ph == a.offload_encoder_after:
            B.enc.to("cpu")
            B.rt._ops.clear()
            torch.cuda.empty_cache()
            rows.add(phase="note", msg=f"encoder offloaded to CPU after {ph}", resident_GiB=torch.cuda.memory_allocated() / 2 ** 30)
    log("D23_BENCH_DONE")
    return 0


def run_phase(ph, B, rows, a, docs):
    # Inference phases under no_grad: the per-document adapter parameters require grad (registration), so a forward
    # outside no_grad would build an autograd graph and keep every intermediate alive (bug in the first 9B run).
    # `reg` manages grad itself (D20.register enables grad for the steps only).
    with (torch.enable_grad() if ph == "reg" else torch.no_grad()):
        if ph == "validate":
            phase_validate(B, rows, a)
        elif ph == "harvest":
            phase_harvest(B, rows, a, docs)
        elif ph == "lat":
            phase_lat(B, rows, a, docs, [m for m in a.methods.split(",") if m])
        elif ph == "reg":
            phase_reg(B, rows, a, docs)
        elif ph == "batch":
            phase_batch(B, rows, a, docs, [m for m in a.batch_methods.split(",") if m])
        elif ph == "lifetime":
            phase_lifetime(B, rows, a, docs)
        elif ph == "rag_acc":
            phase_rag_acc(B, rows, a)
        elif ph == "enc_other":
            phase_enc_other(B, rows, a, docs)
        else:
            raise ValueError(ph)


if __name__ == "__main__":
    sys.exit(main())
