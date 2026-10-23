"""D23 (vLLM): the efficiency benchmark under vLLM 0.32 (envs/vllm032), same documents / prompts as the HF run
(token ids exported by scripts/d23_export_vllm.py), one engine configuration per process.

  --mode text   APC off: text-in-context (document + question every time) and BM25 RAG x8
  --mode apc    automatic prefix caching on: the document prefix is prefilled once, questions reuse it
  --mode kv     the paper-form read plugin (xenc/d23_vllm_plugin.py; APC off, in-process engine, eager)
  --eager 1     enforce_eager (no CUDA graphs / torch.compile) -- the kv plugin always runs eager; --mode apc --eager 1
                is the same-execution-mode baseline for it

Measurements (greedy, forced lengths with ignore_eos; timestamps per generated token from the engine step loop):
  lat    batch 1: TTFT, e2e@6, e2e@32, decode ms/token = (e2e@32 - TTFT) / 31; median over --reps DISTINCT real
         questions of the document (so APC can reuse only the document prefix, never a whole prompt)
  once   apc: the first request of a document (document prefill); kv: operator pool load (H2D)
  batch  questions/s for B requests over B DIFFERENT documents of the same length, 6 generated tokens:
         text: B prompts; apc: B prefix variants (a distinct 2-token nonce in front), each warmed once (not timed);
         kv: B operator copies doc#0..doc#B-1 in the pool
  val    (9B) greedy agreement with the HF judged generations on 20 LongHealth f0 items: kv (registered adapters) vs
         D20 kv@200; apc vs the HF text condition
Memory: vLLM's KV-cache capacity (tokens), torch peak allocated, plugin pool bytes.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 python scripts/bench_d23_vllm.py --mode apc --reader Qwen/Qwen3.5-9B \
        --export /scratch/.../D23_vllm --out outputs/D23_vllm_9b
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")   # greedy decoding; avoids the FlashInfer sampling JIT
# JIT toolchain of THIS env (ninja + pip CUDA nvcc): torch.compile / FlashInfer JIT need them on PATH
_bin = os.path.join(sys.prefix, "bin")
os.environ["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")
_cu = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages",
                   "nvidia", "cu13")
if os.path.exists(os.path.join(_cu, "bin", "nvcc")):
    os.environ["CUDA_HOME"] = _cu        # force: the cluster preset CUDA 11.8 nvcc cannot build FlashInfer 0.7 JIT ops
    os.environ["PATH"] = os.path.join(_cu, "bin") + os.pathsep + os.environ["PATH"]
_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402


CG_SIZES = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]


def med(x):
    return float(st.median(x)) if x else None


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Rows:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def add(self, **kw):
        kw["t"] = time.strftime("%H:%M:%S")
        with open(self.path, "a") as f:
            f.write(json.dumps(kw) + "\n")
        log("ROW", json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in kw.items()
                               if not isinstance(v, (list, dict))})[:500])


class Eng:
    def __init__(self, llm):
        self.llm, self.e = llm, llm.llm_engine
        self.n = 0

    def run(self, reqs, max_tokens, stop=None, ignore_eos=True):
        """reqs: [(rid_prefix, token_ids)] -> (wall_s, {rid: {"t": [t_tok1, ...], "ids": [...], "text": str}})."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=ignore_eos, stop=stop,
                            include_stop_str_in_output=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ids = []
        for pfx, toks in reqs:
            self.n += 1
            rid = f"{pfx}|{self.n}"
            ids.append(rid)
            self.e.add_request(rid, TokensPrompt(prompt_token_ids=list(toks)), sp)
        res = {rid: {"t": [], "ids": [], "text": ""} for rid in ids}
        while self.e.has_unfinished_requests():
            outs = self.e.step()
            now = time.perf_counter() - t0
            for o in outs:
                r = res.get(o.request_id)
                if r is None:
                    continue
                c = o.outputs[0]
                while len(r["t"]) < len(c.token_ids):
                    r["t"].append(now)
                r["ids"], r["text"] = list(c.token_ids), c.text
        torch.cuda.synchronize()
        return time.perf_counter() - t0, res


def kv_capacity(llm) -> dict:
    cc = llm.llm_engine.vllm_config.cache_config
    out = {"num_gpu_blocks": getattr(cc, "num_gpu_blocks", None), "block_size": getattr(cc, "block_size", None)}
    if out["num_gpu_blocks"] and out["block_size"]:
        out["kv_tokens"] = out["num_gpu_blocks"] * out["block_size"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", required=True, choices=("text", "apc", "kv"))
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--itf_tag", default=None, help="kv: itf_<tag>.pt (default 9b / 27b from --reader)")
    ap.add_argument("--export", required=True)
    ap.add_argument("--eager", type=int, default=0)
    ap.add_argument("--gpu_util", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=66000)
    ap.add_argument("--max_num_seqs", type=int, default=64)
    ap.add_argument("--phases", default="val,lat,batch")
    ap.add_argument("--tags", default="2k,16k,64k")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--batch_tags", default=None, help="lengths for the batch phase (default: --tags)")
    ap.add_argument("--graph", type=int, default=0, help="kv: static decode path + FULL decode CUDA graphs (no compile)")
    ap.add_argument("--cg", type=int, default=1, help="kv --graph 1 --cg 0: static path but no graphs (D23_DEBUG check)")
    ap.add_argument("--decode_attn", default="fa", choices=("fa", "triton"), help="kv --graph 1: decode attention")
    ap.add_argument("--pool_tokens", type=int, default=2_100_000, help="kv --graph 1: static operator pool capacity")
    ap.add_argument("--cg_mode", default="default", choices=("default", "decode_only"),
                    help="apc/text: decode_only = FULL decode graphs, no torch.compile (the kv --graph 1 setting)")
    ap.add_argument("--adapters", default="adapters", help="kv: adapter subdir of --export (adapters = D20 registered "
                    "@200; adapters_h3 = D24 hypernetwork h3; adapters_h27r = random H of h3's shapes for the 27B)")
    ap.add_argument("--val_ref", default=None, help="kv val: JSON {key: HF generation} replacing the D20 kv@200 "
                    "references (e.g. <export>/val_ref_h3.json)")
    ap.add_argument("--gdn_prefill", default="triton", help="vLLM GDN prefill backend (triton = FLA kernels, no JIT; "
                    "auto picks the FlashInfer JIT kernel on Hopper)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from vllm import LLM
    import vllm
    import transformers
    P = json.load(open(os.path.join(a.export, "prompts.json")))
    bench = {b["tag"]: b for b in P["bench"]}
    tags = [t for t in a.tags.split(",") if t in bench]
    batches = [int(x) for x in a.batches.split(",") if x]
    tag = a.itf_tag or ("27b" if "27B" in a.reader else "9b")
    os.makedirs(a.out, exist_ok=True)
    rows = Rows(os.path.join(a.out, "rows.jsonl"))
    kw = dict(model=a.reader, dtype="bfloat16", max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util,
              enforce_eager=bool(a.eager or a.mode == "kv"), enable_prefix_caching=(a.mode == "apc"), seed=0,
              max_num_seqs=a.max_num_seqs, language_model_only=True, gdn_prefill_backend=a.gdn_prefill)
    if a.mode == "kv":
        from xenc import d23_vllm_plugin as plug
        arch = plug.register()
        val_docs = sorted({v["doc"] for v in P["val"]}) if "val" in a.phases and tag == "9b" else []
        os.environ["D23_ITF"] = os.path.join(a.export, f"itf_{tag}.pt")
        os.environ["D23_OPS"] = os.path.join(a.export, "ops")
        os.environ["D23_ADAPTERS"] = os.path.join(a.export, a.adapters)
        os.environ.setdefault("D23_MAX_ADAPTERS", "40")
        os.environ["D23_DOCS"] = ",".join(val_docs + [bench[t]["doc"] for t in tags])
        os.environ.setdefault("D23_OWN_MAX", "512")
        os.environ.setdefault("D23_MAX_REQS", "40")
        kw["hf_overrides"] = {"architectures": [arch]}
        kw["max_model_len"] = 4096
        kw["compilation_config"] = {"mode": 0}
        if a.graph:
            os.environ["D23_GRAPH"] = "1"
            os.environ["D23_POOL_TOKENS"] = str(a.pool_tokens)
            os.environ["D23_DECODE_ATTN"] = a.decode_attn
            os.environ.setdefault("D23_MAXN", "64")
            os.environ.setdefault("D23_MAX_DOC_TOKENS", "66000")
            if a.cg:
                kw["enforce_eager"] = False
                kw["compilation_config"] = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                            "cudagraph_capture_sizes": CG_SIZES}
    elif a.cg_mode == "decode_only":
        kw["enforce_eager"] = False
        kw["compilation_config"] = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": CG_SIZES}
    t0 = time.perf_counter()
    llm = LLM(**kw)
    load_s = time.perf_counter() - t0
    E = Eng(llm)
    env = dict(phase="env", mode=a.mode, eager=kw["enforce_eager"], reader=a.reader, vllm=vllm.__version__,
               torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0),
               gpu_total_GiB=torch.cuda.get_device_properties(0).total_memory / 2 ** 30, gpu_util=a.gpu_util,
               load_s=load_s, mem_allocated_GiB=torch.cuda.memory_allocated() / 2 ** 30,
               mem_reserved_GiB=torch.cuda.memory_reserved() / 2 ** 30, cap=kv_capacity(llm),
               host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"), gdn_prefill=a.gdn_prefill,
               cuda_home=os.environ.get("CUDA_HOME"), graph=a.graph, cg=a.cg, decode_attn=a.decode_attn,
               cg_mode=a.cg_mode, resolved_cudagraph_mode=str(getattr(llm.llm_engine.vllm_config.compilation_config,
                                                                   "cudagraph_mode", None)),
               resolved_compile_mode=str(getattr(llm.llm_engine.vllm_config.compilation_config, "mode", None)))
    if a.mode == "kv":
        env["plugin"] = plug.STATE.get("info")
    rows.add(**env)
    pfx_doc = (lambda d: f"d23|{d}") if a.mode == "kv" else (lambda d: f"{a.mode}|{d}")

    # ------------------------------------------------------------------ validation (9B)
    if "val" in a.phases and tag == "9b":
        sys.path.insert(0, str(_ROOT))
        from xenc.doc_tasks import score
        ok = agree = 0
        scs, gens = [], []
        vref = json.load(open(a.val_ref)) if a.val_ref else None
        for v in P["val"]:
            toks = v["rows"] if a.mode == "kv" else v["text"]
            if a.mode == "text":
                continue
            _, res = E.run([(pfx_doc(v["doc"]), toks)], 32, stop=["</answer>"], ignore_eos=False)
            g = next(iter(res.values()))["text"]
            g = (g.split("</answer>", 1)[0] + "</answer>") if "</answer>" in g else g
            ref = (vref[v["key"]] if vref is not None else v["judged_kv200"]) if a.mode == "kv" else v["judged_text"]
            agree += g.strip() == ref.strip()
            gens.append({"key": v["key"], "vllm": g, "hf": ref})
            s_ = float(score(v["item"], g))
            scs.append(s_)
            ok += 1
        if ok:
            rows.add(phase="val", n=ok, agree_with_hf=agree / ok, acc=sum(scs) / ok, gens=gens,
                     read_tokens=(plug.STATE.get("read_tokens", 0) if a.mode == "kv" else None),
                     debug=(plug.STATE.get("debug_log") if a.mode == "kv" else None),
                     ref=(("HF " + os.path.basename(a.val_ref)) if a.val_ref else "D20 judged kv@200 (registered adapters)")
                     if a.mode == "kv" else "HF judged text", adapters=a.adapters)

    # ------------------------------------------------------------------ batch-1 latency
    def prompt(b, k, variant=None):
        if a.mode == "kv":
            return b["pre"][-64:] + b["sufs"][k % len(b["sufs"])]
        return b["pre"] + b["sufs"][k % len(b["sufs"])]

    if "lat" in a.phases:
        for t in tags:
            b = bench[t]
            methods = ["text", "rag8"] if a.mode == "text" else [a.mode]
            for m in methods:
                torch.cuda.reset_peak_memory_stats()
                recs, once = [], None
                for k in range(a.reps + 1):
                    toks = b["rag8s"][k % len(b["rag8s"])] if m == "rag8" else prompt(b, k)
                    w, res = E.run([(pfx_doc(b["doc"]), toks)], 32)
                    r = next(iter(res.values()))
                    if k == 0:
                        once = w                       # warm-up; for apc this is the document's first (uncached) read
                        continue
                    tt = r["t"]
                    recs.append({"ttft": tt[0], "e2e6": tt[5], "e2e32": tt[31], "tpot": (tt[31] - tt[0]) / 31})
                rows.add(phase="lat", method=m, mode=a.mode, eager=kw["enforce_eager"], L=t, doc=b["doc"], N=b["N"],
                         ctx_tokens=len(toks), first_request_s=once, **{k_: med([x[k_] for x in recs]) for k_ in recs[0]},
                         ttft_all=[x["ttft"] for x in recs], peak_alloc_GiB=torch.cuda.max_memory_allocated() / 2 ** 30)
            if a.mode == "apc":                        # one-time document prefill, isolated: a fresh variant prefix
                nonce = [3000, 3001]
                w, _ = E.run([("apc_once", nonce + b["pre"] + b["sufs"][0])], 1)
                w2, _ = E.run([("apc_once", nonce + b["pre"] + b["sufs"][1])], 1)
                rows.add(phase="once", mode="apc", L=t, N=b["N"], first_read_s=w, second_question_ttft_s=w2)

    # ------------------------------------------------------------------ throughput, B different documents
    if "batch" in a.phases:
        from vllm import SamplingParams  # noqa: F401
        btags = [t for t in (a.batch_tags or a.tags).split(",") if t in bench]
        for t in btags:
            b = bench[t]
            methods = ["text", "rag8"] if a.mode == "text" else [a.mode]
            for m in methods:
                for B in batches:
                    try:
                        if a.mode == "kv":
                            pools = plug.STATE["pools"]
                            if not pools.capacity_blocks:             # eager path: reallocate the pool
                                pools.docs.clear()
                                pools.adapters.clear()
                                for d_ in list(pools.op_k):
                                    pools.op_k[d_] = pools.op_v[d_] = None
                                torch.cuda.empty_cache()
                            t1 = time.perf_counter()
                            pools.load_docs(os.environ["D23_OPS"], [f"{b['doc']}*{B}"])   # static: in place
                            torch.cuda.synchronize()
                            pool_load = time.perf_counter() - t1
                            reqs = [(f"d23|{b['doc']}#{j}", b["pre"][-64:] + b["sufs"][j % len(b["sufs"])]) for j in range(B)]
                        elif a.mode == "apc":
                            nonce = lambda j: [1000 + (j % 997), 2000 + B]  # noqa: E731   distinct prefix per "document"
                            warm = [(f"apc|w{j}", nonce(j) + b["pre"] + b["sufs"][0]) for j in range(B)]
                            E.run(warm, 1)                                     # one-time prefills (not timed)
                            nsuf = len(b["sufs"]) - 1
                            mk = lambda r: [(f"apc|q{j}", nonce(j) + b["pre"] + b["sufs"][1 + (j + 5 * r) % nsuf])  # noqa: E731
                                            for j in range(B)]
                            reqs = mk(0)
                            pool_load = None
                        else:
                            src = b["rag8s"] if m == "rag8" else [b["pre"] + s for s in b["sufs"]]
                            reqs = [(f"text|{j}", src[j % len(src)]) for j in range(B)]
                            pool_load = None
                        torch.cuda.reset_peak_memory_stats()
                        ws = []
                        for rep in range(2):
                            if a.mode == "apc" and rep:
                                reqs = mk(rep)                         # new questions: only the document prefix is cached
                            w, res = E.run(reqs, 6)
                            ws.append(w)
                        rows.add(phase="batch", method=m, mode=a.mode, eager=kw["enforce_eager"], L=t, N=b["N"], B=B,
                                 wall_s=min(ws), qps=B / min(ws), walls=ws, pool_load_s=pool_load,
                                 peak_alloc_GiB=torch.cuda.max_memory_allocated() / 2 ** 30,
                                 pool_bytes=(plug.STATE["pools"].bytes() if a.mode == "kv" else None))
                    except torch.cuda.OutOfMemoryError as e:
                        torch.cuda.empty_cache()
                        rows.add(phase="batch", method=m, mode=a.mode, L=t, N=b["N"], B=B, oom=True, err=str(e)[:200])
    if a.mode == "kv":                                   # guard against a silently skipped read
        rt = plug.STATE.get("read_tokens", 0)
        rows.add(phase="plugin_check", read_tokens=rt, read_forwards=plug.STATE.get("read_forwards", 0),
                 static_tokens=plug.STATE.get("static_tokens", 0), debug=plug.STATE.get("debug_log"), ok=bool(rt > 0))
        if not rt:
            log("D23_ERROR: the read branch never ran (no request context reached the model)")
    log("D23_VLLM_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
