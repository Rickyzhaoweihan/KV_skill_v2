"""D21 (a): interface-fidelity screen of post-hoc compressed D13 base interfaces (xenc/d21_compress.py).

On the base's HELD-OUT VAL documents (args.json of the base: QuALITY 10 val docs, MuSiQue 20, LastingBench 20,
LongHealth fold 1; every question of those docs, the 96-item stage-2 val subset flagged), every variant is compared
with the full interface on the same items:
  KL       KL(p_full ‖ p_variant) on the full interface's greedy answer tokens (teacher-forced; mean and first token)
  cos      per read layer, cosine of the B-mapped read-out y_ℓ (variant vs full) over all reader rows (teacher-forced
           on the same rows; dropped layers excluded)
  agree    the variant's greedy answer equals the full interface's (MCQ label / normalized span)
  acc      greedy val accuracy (budget 32), paired difference to the full interface with se
These are interface-fidelity numbers on VALIDATION documents, not judge results (no test document is touched).

    python scripts/d21_compress.py --interface outputs/D13_4b_reg_w10/d13_best.pt --out outputs/D21_fid \
        --variants full,bf16,int8,svd256,drop19
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import _get, log  # noqa: E402

DEFAULT_VARIANTS = ",".join([
    "full", "bf16", "int8", "int4g128", "int4g64",
    "svd512", "svd256", "svd128", "svd64", "svd32",
    "svd128_A", "svd128_M", "svd128_B", "svd32_A", "svd32_B", "noM",
    "svd256+int8", "svd128+int8", "svd256+int4g64", "svd128+int4g64",
    "drop3", "drop7", "drop11", "drop15", "drop19",
    "drop3-7", "drop3-11", "drop3-15", "drop3-19", "drop7-11", "drop7-15", "drop7-19", "drop11-15", "drop11-19",
    "drop15-19"])


def answer_key(it, g: str) -> str:
    from xenc.doc_tasks import extract_answer, extract_label, normalize_answer
    ch = _get(it, "choices")
    if ch:
        return str(extract_label(g, "".join(str(c["label"]) for c in ch)))
    return normalize_answer(extract_answer(g))


def summarize_variant(rows: list[dict], layers: list[int]) -> dict:
    def mean(xs):
        xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return sum(xs) / len(xs) if xs else float("nan")

    def paired(rs):
        d = [r["score"] - r["full_score"] for r in rs]
        if not d:
            return float("nan"), float("nan")
        mu = sum(d) / len(d)
        sd = (sum((x - mu) ** 2 for x in d) / max(len(d) - 1, 1)) ** 0.5
        return mu, sd / len(d) ** 0.5

    out = {"n": len(rows), "acc": mean([r["score"] for r in rows]), "acc_full": mean([r["full_score"] for r in rows]),
           "agree": mean([float(r["agree"]) for r in rows]), "kl": mean([r["kl"] for r in rows]),
           "kl_first": mean([r["kl_first"] for r in rows]),
           "cos": {str(li): mean([r["cos"].get(str(li)) for r in rows]) for li in layers}}
    out["d_acc"], out["d_acc_se"] = paired(rows)
    sub = [r for r in rows if r["sub96"]]
    out["n96"], out["acc96"] = len(sub), mean([r["score"] for r in sub])
    out["d96"], out["d96_se"] = paired(sub)
    out["agree96"] = mean([float(r["agree"]) for r in sub])
    tasks = sorted({r["task"] for r in rows})
    out["per_task"] = {t: {"n": sum(r["task"] == t for r in rows),
                           "acc": mean([r["score"] for r in rows if r["task"] == t]),
                           "acc_full": mean([r["full_score"] for r in rows if r["task"] == t]),
                           "agree": mean([float(r["agree"]) for r in rows if r["task"] == t]),
                           "kl": mean([r["kl"] for r in rows if r["task"] == t])} for t in tasks}
    cs = [v for v in out["cos"].values() if not math.isnan(v)]
    out["cos_mean"] = sum(cs) / len(cs) if cs else float("nan")
    return out


def table(summ: dict, reports: dict) -> str:
    L = ["| setting | params (M) | MiB | val KL (mean / first tok) | read-out cos (mean; per layer) | agreement | "
         "val acc (n) | Δacc vs full ± se | val96 acc | QuALITY / LH / MuSiQue / LastingB acc |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for v, s in summ.items():
        r = reports[v]
        pt = s["per_task"]
        per = " / ".join(f"{pt[t]['acc']:.3f}" if t in pt else "—" for t in ("quality", "longhealth", "musique",
                                                                               "lastingbench"))
        cos = " ".join(f"{x:.3f}" if not math.isnan(x) else "–" for x in s["cos"].values())
        L.append(f"| `{v}` | {r['params'] / 1e6:.1f} | {r['MiB']:.1f} | {s['kl']:.4f} / {s['kl_first']:.4f} | "
                 f"{s['cos_mean']:.4f}; {cos} | {s['agree']:.3f} | {s['acc']:.3f} ({s['n']}) | "
                 f"{s['d_acc']:+.3f} ± {s['d_acc_se']:.3f} | {s['acc96']:.3f} | {per} |")
    return "\n".join(L)


@torch.no_grad()
def collect_covs(a, rt, itf, layers) -> dict:
    """Input covariances (uncentered second moments) of every A / M.0 / M.2 / B matrix of the FULL interface, from
    student-mode passes over TRAIN documents of the base (--cov_train; LongHealth folds ∉ {test 0, val}); label-free:
    the reader rows are doc tail + question only (no answers). Never val or judge documents."""
    from fit_d13 import select_items
    items, docs = select_items(a.cov_train, "train", set(range(5)) - {0, a.val_fold}, a.cov_q_per_doc, 0)
    rt.docs = {**docs, **rt.docs}
    acc: dict = {}

    def add(k, X):
        X = X.detach().double().reshape(-1, X.shape[-1])
        acc[k] = acc.get(k, 0) + X.t() @ X
        acc[k + "#n"] = acc.get(k + "#n", 0) + X.shape[0]

    hooks = []
    for li in layers:
        s = str(li)
        hooks.append(itf.A[s].register_forward_hook(lambda m, i, o, s=s: add(f"A.{s}.weight", i[0])))
        if itf.mlp_hidden:
            hooks.append(itf.M[s][0].register_forward_hook(lambda m, i, o, s=s: add(f"M.{s}.0.weight", i[0])))
            hooks.append(itf.M[s][2].register_forward_hook(lambda m, i, o, s=s: add(f"M.{s}.2.weight", i[0])))
    orig_B = itf.Bmap

    def Bmap(li, o):
        add(f"B.{li}", o)
        return orig_B(li, o)
    itf.Bmap = Bmap
    R = rt.R
    t0 = time.time()
    try:
        for it in sorted(items, key=lambda i: _get(i, "doc_id")):
            did = _get(it, "doc_id")
            op, n_op, _ = rt.operator(did)
            R.load(op, n_op)
            R.mode = "student"
            try:
                rt.reader(input_ids=torch.tensor([rt.rows(did, it)], device=rt.dev), use_cache=False, logits_to_keep=1)
            finally:
                R.mode, R.past = "off", {}
    finally:
        for h in hooks:
            h.remove()
        itf.Bmap = orig_B
    covs = {k: (v / acc[k + "#n"]).float() for k, v in acc.items() if not k.endswith("#n")}
    log(f"covariances from {len(items)} train items / {len(docs)} train docs "
        f"({acc[f'A.{layers[0]}.weight#n']} rows per matrix, {time.time() - t0:.0f}s)")
    return covs


def save_variant(itf, base: dict, new: dict, rep: dict, layers: list[int], path: str) -> str:
    """D13-format file of a compressed variant: kept layers only (the gate vector re-indexed), dense fp32 values."""
    from xenc.d13_read import FORMAT
    keep = rep["layers"]
    sd = {}
    for k, v in base.items():
        if k == "g":
            g = new.get("g", v).detach().float().cpu()
            sd["g"] = torch.stack([g[layers.index(li)] for li in keep])
        elif int(k.split(".")[1]) in keep:
            sd[k] = new.get(k, v).detach().float().cpu().clone()
    cfg = dict(itf.config())
    cfg["layer_map"] = {li: itf.layer_map[li] for li in keep}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"format": FORMAT, "config": cfg, "state_dict": sd, "extra": {"d21": rep}}, path + ".tmp")
    os.replace(path + ".tmp", path)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--B", default="outputs/D11_maps/encB5_9b_4b_post_xf01.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--interface", default="outputs/D13_4b_reg_w10/d13_best.pt")
    ap.add_argument("--val", default="quality:10,musique:20,lastingbench:20,longhealth:0",
                    help="the base's held-out val docs (its args.json)")
    ap.add_argument("--val_fold", type=int, default=1)
    ap.add_argument("--val_q_per_doc", type=int, default=0, help="0 = every question of the val docs")
    ap.add_argument("--val_seed", type=int, default=1)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="1 doc per task, 2 questions, 3 variants")
    ap.add_argument("--save_specs", default="", help="comma list of variants (must be in --variants) to save as "
                    "D13-format checkpoints (dense reconstruction; dropped layers removed from the layer map) for "
                    "scripts/register_d20.py --interface ... --layers <kept>")
    ap.add_argument("--save_dir", default="outputs/D21_ckpt")
    ap.add_argument("--cov_train", default="quality:40,musique:40,lastingbench:40,longhealth:0",
                    help="asvd: base TRAIN docs whose (label-free) reader rows give the input covariances")
    ap.add_argument("--cov_q_per_doc", type=int, default=2)
    ap.add_argument("--save_only", action="store_true", help="save --save_specs and exit (no fidelity screen)")
    a = ap.parse_args(argv)
    from fit_d13 import D13Rt, build, select_items
    from xenc.d21_compress import compress, interface_tensors
    from xenc.doc_tasks import score
    variants = [v for v in a.variants.split(",") if v]
    if a.smoke:
        a.val = ",".join(f"{s.split(':')[0]}:1" for s in a.val.split(","))
        a.val_q_per_doc = 2
        variants = variants[:3]
    if variants[0] != "full":
        variants = ["full"] + [v for v in variants if v != "full"]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    reader, enc, tok, itf, lmap = build(a, init=a.interface)
    rt = D13Rt(reader, enc, tok, a.reader, itf, a.tail, a.max_tokens, lru=1)
    layers = sorted(lmap)
    val, docs = select_items(a.val, "val", {a.val_fold}, a.val_q_per_doc, a.val_seed)
    sub, _ = select_items(a.val, "val", {a.val_fold}, 4, a.val_seed)      # the stage-2 selection subset (96 items)
    subkeys = {_get(i, "key") for i in sub}
    rt.docs = docs
    by_doc: OrderedDict = OrderedDict()
    for it in sorted(val, key=lambda i: _get(i, "doc_id")):
        by_doc.setdefault(_get(it, "doc_id"), []).append(it)
    log(f"D21 fidelity: {a.interface} | {a.encoder} -> {a.reader} layers {layers} | val {len(val)} items / "
        f"{len(by_doc)} docs ({len(subkeys & {_get(i, 'key') for i in val})} in the 96-item subset) | "
        f"{len(variants)} variants")

    # ---------------- variants (precomputed once; stored on CPU as dense fp32 reconstructions of changed tensors)
    live = itf.state_dict()
    base = {k: v.detach().clone() for k, v in interface_tensors(itf).items()}
    svd_cache: dict = {}
    store, reports = {}, {}
    covs = None
    if any("asvd" in v for v in variants):
        covs = collect_covs(a, rt, itf, layers)
    t0 = time.time()
    for v in variants:
        if "@" in v:            # <ckpt>@<label>: tensors from a D13 file (e.g. d21_recover.py), sizes as <label>
            path, lab = v.split("@", 1)
            sd_f = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
            _, rep = compress(base, lab, svd_cache, layers, covs)
            new = {k: t.float().to(base[k].device) for k, t in sd_f.items() if k in base}
            rep["spec"], rep["from_file"] = v, path
        else:
            new, rep = compress(base, v, svd_cache, layers, covs)
        err = {}
        for k, w in new.items():
            if k in base and base[k].dim() == 2:
                g = k[0]
                err.setdefault(g, []).append(float((w - base[k].float()).norm() / base[k].float().norm().clamp_min(1e-12)))
        rep["rel_err"] = {g: sum(e) / len(e) for g, e in err.items()}
        store[v] = {k: t.cpu() for k, t in new.items()}
        reports[v] = rep
        log(f"  variant {v}: {rep['params'] / 1e6:.2f}M params, {rep['MiB']:.1f} MiB, layers {rep['layers']}, "
            f"rel err {({g: round(e, 4) for g, e in rep['rel_err'].items()})}")
    svd_cache.clear()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    log(f"variants ready ({time.time() - t0:.0f}s)")
    if a.save_specs:
        for v in [s for s in a.save_specs.split(",") if s]:
            p = save_variant(itf, base, store[v], reports[v], layers, os.path.join(a.save_dir, f"{v}.pt"))
            log(f"  saved {v} -> {p} (layers {reports[v]['layers']})")
        if a.save_only:
            return 0
    dirty: set = set()

    @torch.no_grad()
    def apply(v):
        nonlocal dirty
        for k in dirty:
            live[k].copy_(base[k])
        for k, t in store[v].items():
            live[k].copy_(t.to(live[k].device, non_blocking=True))
        dirty = set(store[v])

    cap: dict = {"on": None}
    orig_read = itf.read

    def rec_read(li, *args, **kw):
        res = orig_read(li, *args, **kw)
        if cap["on"] is not None:
            cap["on"][li] = res["y"].detach()
        return res
    itf.read = rec_read
    R = rt.R

    @torch.no_grad()
    def gen(did, it):
        from register_state import _stop_ids
        from xenc.backbone import decode
        rows = rt.rows(did, it)
        op, n_op, _ = rt.operator(did)
        R.load(op, n_op)
        R.mode = "student"
        try:
            out = reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=True, logits_to_keep=1)
            stops, new = _stop_ids(tok), []
            for _ in range(a.budget):
                nxt = int(out.logits[0, -1].argmax())
                new.append(nxt)
                if nxt in stops or "</answer>" in tok.decode(new[-12:], skip_special_tokens=True):
                    break
                out = reader(input_ids=torch.tensor([[nxt]], device=rt.dev), past_key_values=out.past_key_values,
                             use_cache=True, logits_to_keep=1)
        finally:
            R.mode, R.past = "off", {}
        txt = decode(tok, a.reader, new)
        if "</answer>" in txt:
            txt = txt.split("</answer>", 1)[0] + "</answer>"
        return txt, new

    @torch.no_grad()
    def tf(did, it, ids):
        rows = rt.rows(did, it, ids)
        op, n_op, _ = rt.operator(did)
        R.load(op, n_op)
        R.mode, cap["on"] = "student", {}
        try:
            out = reader(input_ids=torch.tensor([rows], device=rt.dev), use_cache=False, logits_to_keep=len(ids) + 1)
            lp = torch.log_softmax(out.logits[0, :-1].float(), -1)
            ys = cap["on"]
        finally:
            R.mode, R.past, cap["on"] = "off", {}, None
        return lp, ys

    rows_v = {v: [] for v in variants}
    t0 = time.time()
    for di, (did, its) in enumerate(by_doc.items()):
        apply("full")
        refs = []
        for it in its:
            g, ids = gen(did, it)
            lp, ys = tf(did, it, ids)
            refs.append({"g": g, "ids": ids, "lp": lp, "ys": ys, "score": float(score(it, g)), "ak": answer_key(it, g)})
        for v in variants:
            apply(v)
            drop = set(reports[v]["dropped"])
            for it, ref in zip(its, refs):
                if v == "full":
                    g, lpv, ysv = ref["g"], ref["lp"], ref["ys"]
                else:
                    g, _ = gen(did, it)
                    lpv, ysv = tf(did, it, ref["ids"])
                kl_t = (ref["lp"].exp() * (ref["lp"] - lpv)).sum(-1)
                cos = {str(li): float(torch.nn.functional.cosine_similarity(ysv[li], ref["ys"][li], dim=-1).mean())
                       for li in layers if li not in drop}
                rows_v[v].append({"key": _get(it, "key"), "doc_id": did, "task": did.split("_")[0],
                                  "sub96": _get(it, "key") in subkeys, "gen": g, "score": float(score(it, g)),
                                  "full_score": ref["score"], "agree": answer_key(it, g) == ref["ak"],
                                  "kl": float(kl_t.mean()), "kl_first": float(kl_t[0]), "cos": cos})
        del refs
        el = time.time() - t0
        log(f"  [{di + 1}/{len(by_doc)} docs, {len(rows_v['full'])} items] {el:.0f}s | " + " ".join(
            f"{v}: agr {sum(r['agree'] for r in rows_v[v]) / len(rows_v[v]):.3f} "
            f"kl {sum(r['kl'] for r in rows_v[v]) / len(rows_v[v]):.4f}" for v in variants[1:4]))
        if (di + 1) % 5 == 0 or di + 1 == len(by_doc):
            summ = {v: summarize_variant(rows_v[v], layers) for v in variants}
            json.dump({"args": vars(a), "n_docs_done": di + 1, "reports": reports, "summary": summ},
                      open(os.path.join(a.out, "fidelity.json.tmp"), "w"), indent=1)
            os.replace(os.path.join(a.out, "fidelity.json.tmp"), os.path.join(a.out, "fidelity.json"))
    with open(os.path.join(a.out, "items.jsonl"), "w") as f:
        for v in variants:
            for r in rows_v[v]:
                f.write(json.dumps({"variant": v, **r}) + "\n")
    summ = {v: summarize_variant(rows_v[v], layers) for v in variants}
    md = table(summ, reports)
    open(os.path.join(a.out, "fidelity.md"), "w").write(
        f"## D21 fidelity screen: {a.interface}, val docs ({len(val)} items / {len(by_doc)} docs)\n\n{md}\n")
    log("\n" + md)
    if torch.cuda.is_available():
        log(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    log(f"done ({time.time() - t0:.0f}s) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
