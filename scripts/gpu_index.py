"""GPU-type index of every eval json (AUDIT 2026-10-25, CPU only, read-only except the output file).

eval_d13.py does not store the GPU in its json (only eval_cad.py / bl / a few others do), yet L40S and H100 greedy decodes
differ by ~.01-.03, so a kv-vs-text pair is only valid on the same GPU type. This maps  outputs/<dir>/eval_*.json  (and
outputs/<dir>/<sub>/eval_*.json)  ->  GPU type of the job that wrote it:

  1. the json's own "gpu" field, if present            (source "json")
  2. the launcher's  outputs/<dir>/gpu.txt              (source "gpu.txt"; written after a successful sweep command)
  3. the slurm job whose [Start, End] window (sacct) contains the file mtime AND whose log mentions the output dir
     (source "job"); the job's GPU = the "NVIDIA <name>" banner of its log(s), else the node's type learned from other
     logs, else the partition name.  Ambiguous (candidate jobs of different GPU types) or no candidate -> unattributed.

    python scripts/gpu_index.py                    # writes outputs/GPU_INDEX.json, prints a summary
    python scripts/gpu_index.py --out /tmp/x.json
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU_RE = re.compile(r"NVIDIA (H100|L40S|A100|A40|V100|L40|RTX[0-9A-Za-z ]*)")
JOB_RE = re.compile(r"_(\d+)(?:_lane\d+)?\.out$")
OUT_RE = re.compile(r"outputs/([A-Za-z0-9_.\-]+)")
SLACK_BEFORE, SLACK_AFTER = 120, 900          # seconds around the sacct window (clock skew / final flush)


def norm_gpu(name: str) -> str:
    name = name.strip()
    for k in ("H100", "L40S", "A100", "A40", "V100", "L40"):
        if k in name:
            return k
    return name


def part_gpu(part: str):
    p = part.lower()
    if "l40s" in p and "h100" not in p:
        return "L40S"
    if "h100" in p and "l40s" not in p:
        return "H100"
    if "v100" in p:
        return "V100"
    return None


def parse_logs(logdir):
    jobs = {}
    for p in sorted(glob.glob(os.path.join(logdir, "*.out"))):
        m = JOB_RE.search(os.path.basename(p))
        if not m:
            continue
        jid = m.group(1)
        txt = open(p, "rb").read().decode("utf-8", "ignore")
        j = jobs.setdefault(jid, {"logs": [], "gpus": collections.Counter(), "outs": set(), "txt": []})
        j["logs"].append(os.path.basename(p))
        j["name"] = re.sub(r"_%s(_lane\d+)?\.out$" % jid, "", os.path.basename(p))
        for g in GPU_RE.findall(txt):
            j["gpus"][norm_gpu(g)] += 1
        j["outs"] |= set(OUT_RE.findall(txt))
        j["txt"].append(txt)
    return jobs


def sacct(jids):
    out = {}
    jids = sorted(jids, key=int)
    for i in range(0, len(jids), 300):
        chunk = jids[i:i + 300]
        try:
            r = subprocess.run(["sacct", "-j", ",".join(chunk), "-X", "-P", "--noheader",
                                "--format=JobID,JobName%60,Partition,NodeList,Start,End,State"],
                               capture_output=True, text=True, timeout=120)
        except Exception as e:                                           # noqa: BLE001
            print("sacct failed:", e, file=sys.stderr)
            return out
        for line in r.stdout.splitlines():
            f = line.split("|")
            if len(f) < 7 or "_" in f[0] or "." in f[0]:
                continue
            try:
                st = dt.datetime.fromisoformat(f[4]).timestamp()
                en = dt.datetime.fromisoformat(f[5]).timestamp() if f[5] not in ("Unknown", "") else None
            except ValueError:
                st = en = None
            out[f[0]] = {"jobname": f[1], "partition": f[2], "node": f[3].split(",")[0], "start": st, "end": en, "state": f[6]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "GPU_INDEX.json"))
    ap.add_argument("--logs", default=os.path.join(REPO, "slurm", "logs"))
    a = ap.parse_args(argv)
    os.chdir(REPO)
    jobs = parse_logs(a.logs)
    sa = sacct(jobs.keys())
    # node -> GPU type, learned from logs that print one unambiguous banner
    node_votes = collections.defaultdict(collections.Counter)
    for jid, j in jobs.items():
        s = sa.get(jid)
        if s and len(j["gpus"]) == 1:
            node_votes[s["node"]][next(iter(j["gpus"]))] += 1
    node_gpu = {n: c.most_common(1)[0][0] for n, c in node_votes.items()}
    for jid, j in jobs.items():
        s = sa.get(jid, {})
        j.update({k: s.get(k) for k in ("partition", "node", "start", "end", "state")})
        if len(j["gpus"]) == 1:
            j["gpu"], j["gpu_src"] = next(iter(j["gpus"])), "log banner"
        elif len(j["gpus"]) > 1:
            j["gpu"], j["gpu_src"] = "MIXED:" + "/".join(sorted(j["gpus"])), "log banner"
        elif s.get("node") in node_gpu:
            j["gpu"], j["gpu_src"] = node_gpu[s["node"]], "node map"
        elif part_gpu(s.get("partition", "") or ""):
            j["gpu"], j["gpu_src"] = part_gpu(s["partition"]), "partition"
        else:
            j["gpu"], j["gpu_src"] = None, "unknown"

    files = sorted(glob.glob("outputs/*/eval_*.json") + glob.glob("outputs/*/*/eval_*.json"))
    index, unattr = {}, []
    for p in files:
        parts = p.split("/")
        d = parts[1]
        mt = os.path.getmtime(p)
        rec = {"mtime": dt.datetime.fromtimestamp(mt).isoformat(timespec="seconds")}
        # job-window attribution (computed for every file, used as a cross-check where a direct record exists)
        cands = []
        for jid, j in jobs.items():
            if j.get("start") is None:
                continue
            en = j["end"] if j["end"] is not None else j["start"] + 86400 * 2
            if not (j["start"] - SLACK_BEFORE <= mt <= en + SLACK_AFTER):
                continue
            named = d in j["outs"] or any(re.search(r"(?<![A-Za-z0-9_\-])%s(?![A-Za-z0-9_\-])" % re.escape(d), t) for t in j["txt"])
            if named:
                cands.append(jid)
        gp = {jobs[c]["gpu"] for c in cands}
        job_gpu = next(iter(gp)) if (cands and len(gp) == 1 and None not in gp and not next(iter(gp)).startswith("MIXED")) else None
        best = min(cands, key=lambda c: abs((jobs[c]["end"] or mt) - mt)) if job_gpu else None
        try:
            J = json.load(open(p))
            g = J.get("gpu") if isinstance(J, dict) else None
        except Exception:                                                # noqa: BLE001
            g = None
        gt = os.path.join("outputs", d, "gpu.txt")
        if g or (len(parts) == 3 and os.path.exists(gt)):
            rec.update(gpu=norm_gpu(g) if g else norm_gpu(open(gt).read()), source="json" if g else "gpu.txt")
            if job_gpu:
                rec.update(job_check=job_gpu, job_check_job=best, job_check_agrees=(job_gpu == rec["gpu"]))
            index[p] = rec
        elif job_gpu:
            rec.update(gpu=job_gpu, source="job", job=best, jobname=jobs[best]["name"], node=jobs[best]["node"],
                       n_candidate_jobs=len(cands), gpu_src=jobs[best]["gpu_src"])
            index[p] = rec
        else:
            rec.update(gpu=None, source="unattributed",
                       reason=("no job window+log match" if not cands else "candidate jobs disagree or GPU unknown"),
                       candidates=[{"job": c, "name": jobs[c]["name"], "gpu": jobs[c]["gpu"]} for c in cands][:8])
            index[p] = rec
            unattr.append(p)
    summary = collections.Counter((r["source"], r["gpu"]) for r in index.values())
    chk = [r for r in index.values() if "job_check_agrees" in r]
    out = {"generated": dt.datetime.now().isoformat(timespec="seconds"), "n_files": len(index),
           "n_unattributed": len(unattr),
           "cross_check_direct_vs_job": {"n": len(chk), "agree": sum(r["job_check_agrees"] for r in chk),
                                         "disagree": [k for k, r in index.items() if r.get("job_check_agrees") is False]}, "by_source_gpu": {f"{k[0]}|{k[1]}": v for k, v in sorted(summary.items(), key=str)},
           "unattributed": unattr, "files": index}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{len(index)} eval jsons; unattributed {len(unattr)}; direct-record vs job-window cross-check "
          f"{out['cross_check_direct_vs_job']['agree']}/{out['cross_check_direct_vs_job']['n']} agree")
    for k, v in sorted(summary.items(), key=str):
        print(f"  {k[0]:13s} {str(k[1]):6s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
