"""D23: markdown tables from scripts/bench_d23.py rows (outputs/D23_eff_*/rows.jsonl). Prints to stdout.

    python scripts/d23_report.py outputs/D23_eff_9b [outputs/D23_eff_27b ...]
"""

from __future__ import annotations

import json
import sys

ORDER = ["2k", "8k", "lh12k", "16k", "32k", "64k"]
MIB = 2 ** 20
GIB = 2 ** 30


def load(d):
    rows = []
    for line in open(f"{d}/rows.jsonl"):
        rows.append(json.loads(line))
    return rows


def last(rows, **kw):
    hit = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    return hit[-1] if hit else None


def f(x, n=2, unit=""):
    if x is None:
        return "–"
    if isinstance(x, str):
        return x
    return f"{x:.{n}f}{unit}"


def ms(x):
    return "–" if x is None else f"{1000 * x:.0f}" if x >= 0.1 else f"{1000 * x:.1f}"


def report(d):
    rows = load(d)
    env = last(rows, phase="env")
    out = [f"## {d}", "", f"GPU {env['gpu']} ({env['gpu_total_GiB']:.0f} GiB), torch {env['torch']}, transformers "
           f"{env['transformers']}, attn `{env['attn_impl']}`, FLA kernels {env['fla']}; reader {env['reader']} "
           f"({env['weights_reader_GiB']:.1f} GiB), encoder {env['encoder']} ({env['weights_encoder_GiB']:.1f} GiB), "
           f"interface {env['weights_itf_GiB']:.2f} GiB; node {env['host']}, job {env['slurm_job']}", ""]
    Ls = [L for L in ORDER if last(rows, phase="lat", L=L) or last(rows, phase="harvest", L=L)]
    methods = []
    for r in rows:
        if r.get("phase") == "lat" and r.get("method") not in methods:
            methods.append(r["method"])
    # ---------------------------------------------------------------- latency
    out += ["### Batch-1 latency (median of reps; ms unless noted)", "",
            "| L | N | method | ctx tok | TTFT | per-token decode | e2e@6 | e2e@32 | peak working GiB |",
            "|---|---|---|---|---|---|---|---|---|"]
    for L in Ls:
        for m in methods:
            r = last(rows, phase="lat", L=L, method=m)
            if not r:
                continue
            if r.get("oom"):
                out.append(f"| {L} | {r['N']} | {m} | – | OOM | | | | |")
                continue
            extra = r.get("retrieval_s") or 0.0
            out.append(f"| {L} | {r['N']} | {m} | {r['ctx_tokens']} | {ms(r['ttft'] + extra)} | {ms(r['tpot'])} | "
                       f"{ms(r['e2e6'] + extra)} | {ms(r['e2eG'] + extra)} | {f(r['peak_working_GiB'])} |")
    # ---------------------------------------------------------------- one-time
    out += ["", "### One-time costs per document (s)", "",
            "| L | N | pcache: reader doc prefill (peak GiB) | kv: encoder harvest, truncated (full) | kv reg: self-q by reader | "
            "teacher targets (reader reads doc) | 200 steps (s/step) | **kv registered total** | enc-only reg: self-q by "
            "encoder | enc-only targets | enc-only 200 steps (s/step) | **kv enc-only total** | rag index |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for L in Ls:
        pc = last(rows, phase="lat", L=L, method="pcache")
        hv = [r for r in rows if r.get("phase") == "harvest" and r.get("L") == L and r.get("encoder") == env["encoder"]]
        hv = hv[-1] if hv else None
        rg = last(rows, phase="reg", L=L)
        rag = last(rows, phase="lat", L=L, method="rag512x4")
        g = lambda k: (rg or {}).get(k)  # noqa: E731
        hvt = (hv or {}).get("time_s")
        tot = hvt + g("total200_s") if hvt is not None and g("total200_s") is not None else None
        tot_e = hvt + g("enc_total200_s") if hvt is not None and g("enc_total200_s") is not None else None
        out.append(f"| {L} | {(pc or hv or {}).get('N')} | {f((pc or {}).get('once_s'))} ({f((pc or {}).get('once_peak_working_GiB'))}) | "
                   f"{f(hvt)}{(' (full ' + f((hv or {}).get('full_time_s')) + ')') if (hv or {}).get('full_time_s') else ''} | {f(g('selfq_reader_s'), 1)} | {f(g('teacher_s'), 1)} | {f(g('reg200_s'), 1)} ({f(g('s_per_step'), 3)}) | "
                   f"**{f(tot, 1)}** | {f(g('selfq_enc_s'), 1)} | {f(g('enc_teacher_s'), 1)} | "
                   f"{f(g('enc_reg200_s'), 1)} ({f(g('enc_s_per_step'), 3)}) | **{f(tot_e, 1)}** | {f((rag or {}).get('index_s'), 3)} |")
    # ---------------------------------------------------------------- break-even
    out += ["", "### Lifetime wall time per document, n questions (s; one-time + n x e2e@6; registration included for kv)",
            "", "| L | method | one-time | per question | n=1 | n=5 | n=20 | n=100 | break-even n vs pcache | vs text |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for L in Ls:
        per, once = {}, {}
        for m in methods:
            r = last(rows, phase="lat", L=L, method=m)
            if not r or r.get("oom"):
                continue
            per[m] = r["e2e6"] + (r.get("retrieval_s") or 0.0)
            if m == "pcache":
                once[m] = r["once_s"]
            elif m.startswith("rag"):
                once[m] = r.get("index_s") or 0.0
            elif m.startswith("kv"):
                hv = [x for x in rows if x.get("phase") == "harvest" and x.get("L") == L and x.get("encoder") == env["encoder"]]
                rg = last(rows, phase="reg", L=L) or {}
                if not hv:
                    continue
                h = hv[-1]["time_s"]
                per[m + " [base, no reg]"] = per[m]
                once[m + " [base, no reg]"] = h
                if rg.get("enc_total200_s") is not None:
                    per[m + " [enc-only reg]"] = per[m]
                    once[m + " [enc-only reg]"] = h + rg["enc_total200_s"]
                if rg.get("total200_s") is None:
                    per.pop(m)
                    continue
                once[m] = h + rg["total200_s"]
            else:
                once[m] = 0.0
        for m in list(per):
            if m not in once:
                continue
            tot = [once[m] + n * per[m] for n in (1, 5, 20, 100)]

            def be(b):
                if b not in per or m == b:
                    return "–"
                if per[m] >= per[b]:                       # dearer per question
                    if once[m] >= once[b]:
                        return "never"
                    return f"cheaper only for n < {(once[b] - once[m]) / max(per[m] - per[b], 1e-9):.0f}"
                n = (once[m] - once[b]) / (per[b] - per[m])
                return "always" if n <= 0 else f"{n:.0f}"
            out.append(f"| {L} | {m} | {f(once[m], 1)} | {f(per[m], 3)} | " + " | ".join(f(t, 1) for t in tot)
                       + f" | {be('pcache')} | {be('text')} |")
    # ---------------------------------------------------------------- storage
    out += ["", "### Storage per document (MiB)", "",
            "| L | N | raw text | token ids (int32) | reader prefix cache (attn K/V + conv + recurrent) | operator bf16 | "
            "op 4-bit | op 4-bit + keep 25% | op 4-bit + keep 10% | adapter (fp32) | cache / op-bf16 | cache / op-4bit |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for L in Ls:
        hv = [x for x in rows if x.get("phase") == "harvest" and x.get("L") == L and x.get("encoder") == env["encoder"]]
        pc = last(rows, phase="lat", L=L, method="pcache")
        if not hv:
            continue
        h = hv[-1]
        cb = (pc or {}).get("cache_bytes") or {}
        c = cb.get("total")
        out.append(f"| {L} | {h['N']} | {f(h['raw_text_bytes'] / MIB, 3)} | {f(4 * h['N'] / MIB, 3)} | "
                   f"{f(c / MIB if c else None, 1)} ({f(cb.get('attn_kv', 0) / MIB, 1)} + {f(cb.get('conv', 0) / MIB, 1)} + "
                   f"{f(cb.get('recurrent', 0) / MIB, 1)}) | {f(h['op_bytes_bf16'] / MIB, 1)} | {f(h['op_bytes_q4'] / MIB, 1)} | "
                   f"{f(h['op_bytes_q4_acc25'] / MIB, 1)} | {f(h['op_bytes_q4_acc10'] / MIB, 2)} | "
                   f"{f(h['adapter_params'] * 4 / MIB, 2)} | {f(c / h['op_bytes_bf16'] if c else None)} | "
                   f"{f(c / h['op_bytes_q4'] if c else None)} |")
    others = [x for x in rows if x.get("phase") == "harvest" and x.get("encoder") != env["encoder"] and not x.get("oom")]
    if others:
        out += ["", "Other encoders (one-time harvest):", "", "| encoder | L | N | harvest s | peak working GiB | op bf16 MiB | op 4-bit MiB | enc layers |",
                "|---|---|---|---|---|---|---|---|"]
        for x in others:
            out.append(f"| {x['encoder']} | {x['L']} | {x['N']} | {f(x['time_s'], 3)}{(' (full ' + f(x['full_time_s'], 3) + ')') if x.get('full_time_s') else ''} | {f(x['peak_working_GiB'])} | "
                       f"{f(x['op_bytes_bf16'] / MIB, 1)} | {f(x['op_bytes_q4'] / MIB, 1)} | {x['enc_layers']} |")
    # ---------------------------------------------------------------- swap-in
    out += ["", "### Swap-in of a stored document (pinned host -> GPU, ms)", "", "| L | prefix cache | operator bf16 |", "|---|---|---|"]
    for L in Ls:
        pc = last(rows, phase="lat", L=L, method="pcache")
        hv = [x for x in rows if x.get("phase") == "harvest" and x.get("L") == L and x.get("encoder") == env["encoder"]]
        out.append(f"| {L} | {ms((pc or {}).get('h2d_s'))} | {ms(hv[-1]['h2d_s'] if hv else None)} |")
    # ---------------------------------------------------------------- memory fit
    total = env["gpu_total_GiB"] * 0.95
    out += ["", f"### Documents resident in one GPU ({env['gpu_total_GiB']:.0f} GiB, 95% usable; serving weights only: "
            "reader for text/pcache, reader + interface for kv; minus the batch-1 working peak)", "",
            "| L | pcache docs | kv bf16 docs | kv 4-bit docs | text in-flight requests (KV of N each) |", "|---|---|---|---|---|"]
    for L in Ls:
        pc = last(rows, phase="lat", L=L, method="pcache")
        kv = last(rows, phase="lat", L=L, method="kv_fast")
        hv = [x for x in rows if x.get("phase") == "harvest" and x.get("L") == L and x.get("encoder") == env["encoder"]]
        tx = last(rows, phase="lat", L=L, method="text")
        if not (pc and kv and hv) or pc.get("oom") or kv.get("oom"):
            continue
        cbytes = pc["cache_bytes"]["total"] / GIB
        ob, oq = hv[-1]["op_bytes_bf16"] / GIB, hv[-1]["op_bytes_q4"] / GIB
        free_pc = total - env["weights_reader_GiB"] - pc["peak_working_GiB"]
        free_kv = total - env["weights_reader_GiB"] - env["weights_itf_GiB"] - kv["peak_working_GiB"]
        txt = (total - env["weights_reader_GiB"]) / tx["peak_working_GiB"] if tx and not tx.get("oom") else None
        out.append(f"| {L} | {int(free_pc / cbytes)} | {int(free_kv / ob)} | {int(free_kv / oq)} | {f(txt, 1)} |")
    # ---------------------------------------------------------------- throughput
    bm = []
    for r in rows:
        if r.get("phase") == "batch" and r.get("method") not in bm:
            bm.append(r["method"])
    Bs = sorted({r["B"] for r in rows if r.get("phase") == "batch"})
    if Bs:
        out += ["", "### Throughput (questions/s; batch of B different documents of length L; 6 generated tokens; "
                "peak GiB in parentheses)", "", "| L | method | " + " | ".join(f"B={b}" for b in Bs) + " |",
                "|---|---|" + "---|" * len(Bs)]
        for L in Ls:
            for m in bm:
                cells = []
                for b in Bs:
                    r = last(rows, phase="batch", L=L, method=m, B=b)
                    cells.append("–" if not r else "OOM" if r.get("oom") else f"{r['qps']:.2f} ({r['peak_total_GiB']:.0f})")
                out.append(f"| {L} | {m} | " + " | ".join(cells) + " |")
    # ---------------------------------------------------------------- flops
    out += ["", "### FLOPs (torch FlopCounterMode: matmul/SDPA; Gated-DeltaNet Triton kernels not counted, <1%)", "",
            "| L | text prefill | pcache q-prefill | pcache doc prefill (once) | decode step (text/pcache) | kv q-prefill | "
            "kv decode step | encoder harvest (once) | kv per question (prefill + 5 decode) / pcache |",
            "|---|---|---|---|---|---|---|---|---|"]
    for L in Ls:
        t = last(rows, phase="lat", L=L, method="text") or {}
        p = last(rows, phase="lat", L=L, method="pcache") or {}
        k = last(rows, phase="lat", L=L, method="kv_fast") or {}
        dd = last(rows, phase="flops_decode", L=L) or {}
        hv = [x for x in rows if x.get("phase") == "harvest" and x.get("L") == L and x.get("encoder") == env["encoder"]]
        T = lambda x: "–" if x is None else f"{x / 1e12:.2f}T" if x >= 1e11 else f"{x / 1e9:.1f}G"  # noqa: E731
        ratio = None
        if k.get("flops_prefill") and p.get("flops_prefill") and dd.get("flops_decode_step") and k.get("flops_decode_step"):
            ratio = (k["flops_prefill"] + 5 * k["flops_decode_step"]) / (p["flops_prefill"] + 5 * dd["flops_decode_step"])
        out.append(f"| {L} | {T(t.get('flops_prefill'))} | {T(p.get('flops_prefill'))} | {T(p.get('flops_once'))} | "
                   f"{T(dd.get('flops_decode_step'))} | {T(k.get('flops_prefill'))} | {T(k.get('flops_decode_step'))} | "
                   f"{T(hv[-1].get('flops') if hv else None)}{(' (full ' + T(hv[-1]['full_flops']) + ')') if hv and hv[-1].get('full_flops') else ''} | {f(ratio)} |")
    # ---------------------------------------------------------------- reg detail, lifetime, validate, rag acc
    for r in rows:
        if r.get("phase") == "validate":
            out += ["", f"Validation (LongHealth f0, registered adapters @200, n={r['n']}): agree fast/impl {r['agree_fast_impl']:.2f}, "
                    f"impl/judged {r['agree_impl_judged']:.2f}, fast/judged {r['agree_fast_judged']:.2f}, pcache/text "
                    f"{r['agree_pcache_text']:.2f}; acc {json.dumps({k: round(v, 3) for k, v in r['acc'].items()})}"]
        if r.get("phase") == "lifetime":
            out += ["", f"Lifetime check {r['L']} (N={r['N']}, n={r['n']} real questions, real stopping): " +
                    "; ".join(f"{m} {v['wall_s']:.1f}s (gen {v['mean_gen']:.1f})" for m, v in r["res"].items())]
        if r.get("phase") == "rag_acc":
            out += [f"- RAG acc {r['task']} {r['method']}: {r['acc']:.4f} (n={r['n']}, mean prompt {r['mean_prompt_tokens']:.0f} tok)"]
        if r.get("phase") == "reg":
            keys = ("n_selfq", "n_selfq_enc", "n_reg_items", "n_targets", "enc_n_targets", "peak_selfq_reader_GiB",
                    "peak_selfq_enc_GiB", "peak_reg_GiB", "enc_peak_reg_GiB", "oom", "enc_oom", "selfq_enc_oom", "err")
            out += [f"- reg {r['L']}: " + ", ".join(f"{k} {f(r[k]) if isinstance(r[k], float) else r[k]}" for k in keys if k in r)]
        if r.get("phase") == "error":
            out += [f"- ERROR in phase {r['failed_phase']}: {r['err']}"]
    return "\n".join(out)


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(report(d))
        print()
