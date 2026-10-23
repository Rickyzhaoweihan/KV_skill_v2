"""D16 stage 1 (screen): what the reader-side query map A sees and which encoder keys it addresses.

Same data / teacher / rows as D13 stage 1 (scripts/fit_d13.py): operator = the encoder's frozen prefill of the document
prefix (minus the reader's 64-token tail); teacher = the encoder continuing over tail + question (its per-head attention
over [operator ; own rows] and its post-o_proj output); student = the reader's own features -> A -> the encoder's frozen
attention projections -> read of the operator -> B (D10's encoder-side post B, fixed). Label-free. Rows are
teacher-forced (upstream read layers receive the teacher read-out), so one reader pass per question serves every variant.

  pass 0  ridge statistics for [1, x_ℓ, x_{ℓ-1}, x_{ℓ-2}, x_{ℓ-4}] and [1, q_ℓ] -> x_e at ALL encoder full layers
          => (a) the (reader layer, encoder layer) pairing grid: held-out R² and held-out attention JS of the ridge read
             against the encoder-e teacher; (b) ridge inits for the input-feature variants
  train   all variants concurrently, KL(p_teacher ‖ p_student) + rel_w·relMSE(B o_s, B o_t), teacher-forced
  held    after every epoch, held-out documents: teacher-forced and on-policy JS / read-out cosine; best epoch by
          on-policy cosine (D13's criterion) saved per variant; then a fixed τ / top-k sweep on chosen variants.

    python scripts/fit_d16.py --out outputs/D16_maps
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
from collections import OrderedDict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, _get, log  # noqa: E402

LAGS = (1, 2, 4)
# name -> (cfg, ridge feature set or None)
VARIANTS = {
    "base":   ({}, "x"),
    "mlp":    ({"mlp_hidden": 1024}, "x"),
    "cat2":   ({"feats": ["x", "L-1"]}, "x,L-1"),
    "cat3":   ({"feats": ["x", "L-1", "L-2"]}, "x,L-1,L-2"),
    "catfar": ({"feats": ["x", "L-4"]}, "x,L-4"),
    "qin":    ({"feats": ["q"]}, "q"),
    "win4":   ({"window": 4}, "x"),
    "moe4":   ({"moe": 4, "moe_rank": 256}, "x"),
    "qhead":  ({"qhead": True}, "x"),
    "tau":    ({"tau": True}, "x"),
    "norope": ({"rope": False}, "x"),
    "keymap": ({"keymap": True}, "x"),
    "kcat":   ({"kcat": "next"}, "x"),
    "kall":   ({"kcat": "all"}, "x"),          # scores against the keys of ALL six encoder full layers (D12's a8 idea)
}
LR = {"A": 3e-5, "M": 1e-4, "moe": 1e-4, "win": 1e-3, "qhead": 3e-5, "keymap": 1e-5, "kcat": 3e-5, "beta": 1e-3,
      "tau": 1e-3}
# stage 2 (register_d16.py): D13 stage-2 values for A / MLP (outputs/D13_reg_w10/args.json: lr 1e-5, lr_mlp 5e-5;
# B 1e-5, g 1e-3 via --lr_B / --lr_g); extras scaled alike
LR2 = {"A": 1e-5, "M": 5e-5, "moe": 5e-5, "win": 3e-4, "qhead": 1e-5, "keymap": 3e-6, "kcat": 1e-5, "beta": 3e-4,
       "tau": 3e-4}


# ============================================================================ runtime
class Rt16:
    def __init__(self, reader, enc, tok, hub, lmap, enc_layers, tail=64, max_tokens=16000, lru=3, x_layers=()):
        from xenc.d16_read import EncTeacher16
        from xenc.backbone import input_device
        self.reader, self.enc, self.tok, self.hub = reader, enc, tok, hub
        self.lmap, self.tail, self.max_tokens = lmap, int(tail), int(max_tokens)
        self.Tc = EncTeacher16(enc, enc_layers, x_layers)
        self.docs: dict = {}
        self._pre: dict = {}
        self._ops: OrderedDict = OrderedDict()
        self.lru = int(lru)
        self.dev, self.edev = input_device(reader), input_device(enc)
        self.R = None

    def pre(self, did):
        if did not in self._pre:
            from xenc.state_skill import doc_prefix_text
            ids = [int(x) for x in self.tok(doc_prefix_text(self.tok, self.hub, self.docs[did], SYSTEM_ANSWER),
                                            add_special_tokens=False)["input_ids"]]
            self._pre[did] = ids[-(self.max_tokens - 1024):]
        return self._pre[did]

    def suf(self, it):
        from xenc.state_skill import question_suffix_ids
        return question_suffix_ids(self.tok, self.hub, it, SYSTEM_ANSWER)

    def rows(self, did, it, target_ids=()):
        return self.pre(did)[-self.tail:] + self.suf(it) + list(target_ids)

    @torch.no_grad()
    def operator(self, did):
        if did in self._ops:
            self._ops.move_to_end(did)
            return self._ops[did]
        ids = self.pre(did)[: -self.tail] if self.tail else self.pre(did)
        self.Tc.rec, self.Tc.mode = {}, "op"
        out = self.enc(input_ids=torch.tensor([ids], device=self.edev), use_cache=True, logits_to_keep=1)
        op = {e: {k: v.to(self.dev) for k, v in r.items()} for e, r in self.Tc.rec.items()}
        self.Tc.rec = None
        self._ops[did] = (op, len(ids), out.past_key_values)
        while len(self._ops) > self.lru:
            self._ops.popitem(last=False)
        return self._ops[did]

    @torch.no_grad()
    def teacher(self, did, rows):
        _, _, cache = self.operator(did)
        self.Tc.rec, self.Tc.mode = {}, "rows"
        self.enc(input_ids=torch.tensor([rows], device=self.edev), past_key_values=copy.deepcopy(cache),
                 use_cache=True, logits_to_keep=1)
        rec = {e: {k: v.to(self.dev) for k, v in r.items()} for e, r in self.Tc.rec.items()}
        self.Tc.rec = None
        return rec

    @torch.no_grad()
    def capture(self, did, it, itf_any):
        """Teacher-forced reader pass -> (op, N, rec, fcap {li: feats}, y_t {li}, lt {li: teacher logits})."""
        from xenc.d13_read import teacher_logits
        op, N, _ = self.operator(did)
        rows = self.rows(did, it)
        rec = self.teacher(did, rows)
        R = self.R
        R.load(op, N)
        yt = {li: itf_any.Bmap(li, rec[e]["out"].float()) for li, e in self.lmap.items()}
        R.mode, R.inject, R.fcap = "teacher", yt, {}
        try:
            self.reader(input_ids=torch.tensor([rows], device=self.dev), use_cache=False, logits_to_keep=1)
        finally:
            R.mode = "off"
        lt = {li: teacher_logits(rec[e]["q"], op[e]["K"], rec[e]["k"], itf_any.att[str(li)].scaling)
              for li, e in self.lmap.items()}
        return op, N, rec, dict(R.fcap), yt, lt, rows

    @torch.no_grad()
    def onpolicy_terms(self, itf, did, it, tau=None, topk=0, rec=None):
        from xenc.d13_read import d13_terms, teacher_logits
        op, N, _ = self.operator(did)
        rows = self.rows(did, it)
        rec = rec if rec is not None else self.teacher(did, rows)
        R = self.R
        R.itf = itf
        R.load(op, N)
        R.mode, R.onpolicy, R.grad, R.rec, R.tau, R.topk = "train", True, False, {}, tau, topk
        try:
            self.reader(input_ids=torch.tensor([rows], device=self.dev), use_cache=False, logits_to_keep=1)
            out = {}
            for li, e in self.lmap.items():
                lt = teacher_logits(rec[e]["q"], op[e]["K"], rec[e]["k"], itf.att[str(li)].scaling)
                yt = itf.Bmap(li, rec[e]["out"].float())
                r = R.rec[li]
                out[li] = d13_terms(r["logits"], lt, r["y"], yt, N, metrics=True)
            return out
        finally:
            R.mode, R.rec, R.tau, R.topk, R.onpolicy = "off", {}, None, 0, False


class SharedGram:
    """X^T X and X^T Y accumulated in fp32 on GPU, flushed to fp64 on CPU (precision + memory)."""

    def __init__(self, dx, dy, dev, flush_rows=32768):
        self.xx = torch.zeros(dx, dx, device=dev)
        self.xy = torch.zeros(dx, dy, device=dev)
        self.XX = torch.zeros(dx, dx, dtype=torch.float64)
        self.XY = torch.zeros(dx, dy, dtype=torch.float64)
        self.yy = torch.zeros(dy, dtype=torch.float64)
        self.ys = torch.zeros(dy, dtype=torch.float64)
        self.n, self.pend, self.flush_rows = 0, 0, flush_rows

    def add(self, X, Y):
        X, Y = X.float(), Y.float()
        self.xx += X.t() @ X
        self.xy += X.t() @ Y
        self.yy += (Y.double() ** 2).sum(0).cpu()
        self.ys += Y.double().sum(0).cpu()
        self.n += X.shape[0]
        self.pend += X.shape[0]
        if self.pend >= self.flush_rows:
            self.flush()

    def flush(self):
        self.XX += self.xx.cpu().double()
        self.XY += self.xy.cpu().double()
        self.xx.zero_()
        self.xy.zero_()
        self.pend = 0

    def free_gpu(self):
        self.flush()
        del self.xx, self.xy

    def solve(self, idx, ycols, lam, dev):
        """Ridge on the sub-block idx (feature columns) -> targets ycols. -> W [len(idx), len(ycols)] fp32."""
        XX = self.XX[idx][:, idx].to(dev)
        XY = self.XY[idx][:, ycols].to(dev)
        d = XX.shape[0]
        reg = lam * torch.trace(XX) / d
        W = torch.linalg.solve(XX + reg * torch.eye(d, dtype=XX.dtype, device=dev), XY)
        return W.float()

    def r2_fit(self, idx, ycols, W):
        dev = W.device
        XX = self.XX[idx][:, idx].to(dev)
        XY = self.XY[idx][:, ycols].to(dev)
        Wd = W.double()
        sse = float((self.yy[ycols]).sum() - 2 * (Wd * XY).sum().cpu() + (Wd * (XX @ Wd)).sum().cpu())
        sst = float(self.yy[ycols].sum() - (self.ys[ycols] ** 2).sum() / max(self.n, 1))
        return 1 - sse / sst


def r2_rows(X, Y, W):
    P = X @ W
    return float(1 - ((P - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


# ============================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--B", default="outputs/D10_maps/encB8_9b_2b_post.pt")
    ap.add_argument("--layers", default="3,7,11,15,19")
    ap.add_argument("--pairs", default="", help="explicit reader:encoder pairing '3:3,7:3,...' (default: depth map); "
                                                  "--B must then hold B for those pairs (scripts/fit_d16_Bgrid.py)")
    ap.add_argument("--train", default="quality:150,musique:150,lastingbench:150,longhealth:0")
    ap.add_argument("--val", default="quality:6,musique:12,lastingbench:12,longhealth:0")
    ap.add_argument("--val_fold", type=int, default=1)
    ap.add_argument("--test_fold", type=int, default=0)
    ap.add_argument("--q_per_doc", type=int, default=8)
    ap.add_argument("--val_q_per_doc", type=int, default=4)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=16000)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr_scale", type=float, default=1.0)
    ap.add_argument("--rel_w", type=float, default=1.0)
    ap.add_argument("--sweep", default="base", help="variants for the fixed τ / top-k sweep (+ the best variant)")
    ap.add_argument("--taus", default="0.7,0.85,1.15")
    ap.add_argument("--topks", default="16,64,256,1024")
    ap.add_argument("--skip_grid", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    if a.smoke:
        a.train = ",".join(f"{s.split(':')[0]}:2" for s in a.train.split(","))
        a.val = ",".join(f"{s.split(':')[0]}:1" for s in a.val.split(","))
        a.epochs, a.q_per_doc, a.val_q_per_doc = 1, 2, 2
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    from fit_d13 import _group, fmt, mean_terms, select_items
    from xenc.backbone import load_encoder, load_reader
    from xenc.d13_read import d13_terms, load_B, teacher_logits
    from xenc.d16_read import D16Interface, D16Reader, pair_logits, save_d16
    from xenc.state_skill import text_config
    from xenc.xattn_read import _rotary, attn_dims, full_depth_map
    from deltaskill.backbone_utils import decoder_layers
    reader, tok = load_reader(a.reader, device_map=a.device)
    enc = load_encoder(a.encoder, device_map=a.device)[0]
    ra, ea = attn_dims(reader), attn_dims(enc)
    keep = [int(x) for x in a.layers.split(",")]
    lmap = {k: v for k, v in full_depth_map(ra, ea).items() if k in keep}
    if a.pairs:
        lmap = {int(k): int(v) for k, v in (t.split(":") for t in a.pairs.split(","))}
    efull = sorted(int(x) for x in ea["full"])
    B = load_B(a.B, sorted(lmap))
    d_r, d_e = int(text_config(reader).hidden_size), int(text_config(enc).hidden_size)
    d_q = ra["n_q"] * ra["hd"]
    nxt = {e: min([f for f in efull if f > e] or [max(efull)]) for e in efull}
    names = [v for v in a.variants.split(",") if v]
    rt = Rt16(reader, enc, tok, a.reader, lmap, efull, a.tail, a.max_tokens,
              x_layers=sorted(set(lmap.values())) if "keymap" in names else ())
    dev = rt.dev
    probe = D16Interface(enc, lmap, d_r, d_e, B=B).to(dev)          # base geometry, used for B / teacher scaling
    rt.R = D16Reader(reader, probe, extra_lags=LAGS, extra_q=True)
    tr_folds = set(range(5)) - {a.test_fold, a.val_fold}
    train, d1 = select_items(a.train, "train", tr_folds, a.q_per_doc, a.seed)
    val, d2 = select_items(a.val, "val", {a.val_fold}, a.val_q_per_doc, a.seed + 1)
    rt.docs = {**d1, **d2}
    assert not ({_get(i, "doc_id") for i in train} & {_get(i, "doc_id") for i in val}), "val docs overlap train"
    by_doc = _group(train)
    vgroups = _group(val)
    log(f"D16 stage 1: {a.encoder} -> {a.reader} | read layers {lmap} | enc full {efull} | train {len(train)} q / "
        f"{len(by_doc)} docs | held-out {len(val)} q / {len(vgroups)} docs | variants {names}")
    t0 = time.time()
    report = {"args": vars(a), "lmap": lmap, "efull": efull}

    # ---------------- pass 0: ridge statistics
    dz = 1 + d_r * (1 + len(LAGS))
    zcols = {"1": [0], "x": list(range(1, 1 + d_r))}
    for j, k in enumerate(LAGS):
        zcols[f"L-{k}"] = list(range(1 + d_r * (1 + j), 1 + d_r * (2 + j)))
    ecol = {e: list(range(i * d_e, (i + 1) * d_e)) for i, e in enumerate(efull)}
    GZ = {li: SharedGram(dz, d_e * len(efull), dev) for li in lmap}
    GQ = {li: SharedGram(1 + d_q, d_e, dev) for li in lmap}
    held = {li: {"Z": [], "Q": [], "Y": []} for li in lmap}

    def zfeat(f):
        T = f["x"].shape[0]
        return torch.cat([torch.ones(T, 1, device=f["x"].device)] + [f["x"]] + [f[f"L-{k}"] for k in LAGS], 1)

    for split, groups in (("fit", by_doc), ("held", vgroups)):
        for did, qs in groups.items():
            for it in qs:
                op, N, rec, fcap, yt, lt, rows = rt.capture(did, it, probe)
                Y = torch.cat([rec[e]["x"].float() for e in efull], 1)
                for li, e in lmap.items():
                    Z = zfeat(fcap[li])
                    Qf = torch.cat([torch.ones(Z.shape[0], 1, device=dev), fcap[li]["q"]], 1)
                    if split == "fit":
                        GZ[li].add(Z, Y)
                        GQ[li].add(Qf, rec[e]["x"].float())
                    else:
                        held[li]["Z"].append(Z.half().cpu())
                        held[li]["Q"].append(Qf.half().cpu())
                        held[li]["Y"].append(Y.half().cpu())
    for li in lmap:
        GZ[li].free_gpu()
        GQ[li].free_gpu()
        for k in held[li]:
            held[li][k] = torch.cat(held[li][k]).float()
    torch.cuda.empty_cache()
    log(f"pass 0 done ({time.time() - t0:.0f}s): {next(iter(GZ.values())).n} fit rows / layer")

    ridge_W: dict = {}            # (li, featset, e) -> W
    r2 = {}
    for li, e0 in lmap.items():
        for e in efull:
            idx = zcols["1"] + zcols["x"]
            W = GZ[li].solve(idx, ecol[e], a.lam, dev)
            ridge_W[(li, "x", e)] = W
            r2[f"{li}-{e}-x"] = {"fit": GZ[li].r2_fit(idx, ecol[e], W),
                                 "held": r2_rows(held[li]["Z"][:, idx].to(dev), held[li]["Y"][:, ecol[e]].to(dev), W)}
        for fs in ("x,L-1", "x,L-1,L-2", "x,L-4"):
            idx = zcols["1"] + sum((zcols[f] for f in fs.split(",")), [])
            W = GZ[li].solve(idx, ecol[e0], a.lam, dev)
            ridge_W[(li, fs, e0)] = W
            r2[f"{li}-{e0}-{fs}"] = {"fit": GZ[li].r2_fit(idx, ecol[e0], W),
                                     "held": r2_rows(held[li]["Z"][:, idx].to(dev), held[li]["Y"][:, ecol[e0]].to(dev), W)}
        W = GQ[li].solve(list(range(1 + d_q)), list(range(d_e)), a.lam, dev)
        ridge_W[(li, "q", e0)] = W
        r2[f"{li}-{e0}-q"] = {"fit": GQ[li].r2_fit(list(range(1 + d_q)), list(range(d_e)), W),
                              "held": r2_rows(held[li]["Q"].to(dev), held[li]["Y"][:, ecol[e0]].to(dev), W)}
    del GZ, GQ, held
    report["ridge_r2"] = r2
    log("ridge R² fit/held (reader feats -> encoder layer input):")
    for li, e0 in lmap.items():
        log(f"  L{li}: " + " ".join(f"E{e} {r2[f'{li}-{e}-x']['fit']:.3f}/{r2[f'{li}-{e}-x']['held']:.3f}" for e in efull)
            + " || paired E%d: " % e0 + " ".join(f"{fs} {r2[f'{li}-{e0}-{fs}']['held']:.3f}"
                                                for fs in ("x", "x,L-1", "x,L-1,L-2", "x,L-4", "q")))

    # ---------------- pairing grid: held-out JS of the ridge read against the encoder-e teacher
    if not a.skip_grid:
        atts = {e: copy.deepcopy(decoder_layers(enc)[e].self_attn).float().requires_grad_(False) for e in efull}
        rot = copy.deepcopy(_rotary(enc)).float()
        grid: dict = {}
        with torch.no_grad():
            for did, qs in vgroups.items():
                for it in qs:
                    op, N, rec, fcap, yt, lt, rows = rt.capture(did, it, probe)
                    for li in lmap:
                        x1 = torch.cat([torch.ones(fcap[li]["x"].shape[0], 1, device=dev), fcap[li]["x"]], 1)
                        for e in efull:
                            xe = x1 @ ridge_W[(li, "x", e)]
                            ls = pair_logits(atts[e], rot, xe, op[e]["K"], N)
                            lte = teacher_logits(rec[e]["q"], op[e]["K"], rec[e]["k"], atts[e].scaling)
                            o = torch.zeros(1, 1, device=dev)
                            m = d13_terms(ls, lte, o, o, N, metrics=True)
                            grid.setdefault(f"{li}-{e}", []).append({k: m[k] for k in ("js", "js_doc", "js_doc_uniform",
                                                                                         "mass_t", "mass_s", "top5")})
        G = {k: {m: sum(r[m] for r in v) / len(v) for m in v[0]} for k, v in grid.items()}
        report["pair_grid"] = G
        log("pairing grid (ridge read, held-out, teacher-forced): js_doc [teacher js_doc to uniform] top5")
        for li in lmap:
            log(f"  L{li}: " + " | ".join(f"E{e}{'*' if lmap[li] == e else ''} {G[f'{li}-{e}']['js_doc']:.3f} "
                                         f"[{G[f'{li}-{e}']['js_doc_uniform']:.3f}] {G[f'{li}-{e}']['top5']:.2f}"
                                         for e in efull))
        del atts
        with open(os.path.join(a.out, "report.json"), "w") as f:
            json.dump(report, f, indent=1)

    # ---------------- variants
    itfs, opts = OrderedDict(), {}
    for nm in names:
        cfg, fs = copy.deepcopy(VARIANTS[nm][0]), VARIANTS[nm][1]
        if cfg.get("kcat") == "next":
            cfg["kcat"] = {li: [nxt[e]] for li, e in lmap.items()}
        elif cfg.get("kcat") == "all":
            cfg["kcat"] = {li: [f for f in efull if f != e] for li, e in lmap.items()}
        itf = D16Interface(enc, lmap, d_r, d_e, B=B, cfg=cfg, d_q=d_q).to(dev)
        itf.att = probe.att                                  # frozen encoder projections: one shared copy
        torch.cuda.empty_cache()
        with torch.no_grad():
            for li, e in lmap.items():
                s = str(li)
                W = ridge_W[(li, fs, e)]                    # [1 + d_feats, d_e]
                itf.A[s].weight.zero_()
                itf.A[s].weight[:, :W.shape[0] - 1].copy_(W[1:].t())
                itf.A[s].bias.copy_(W[0])
                if cfg.get("qhead"):
                    itf.init_qhead(li)
                for e2 in (itf.cfg["kcat"] or {}).get(li, []):
                    W2 = ridge_W[(li, "x", e2)]
                    itf.A2[f"{li}_{e2}"].weight.zero_()
                    itf.A2[f"{li}_{e2}"].weight[:, :d_r].copy_(W2[1:].t())
                    itf.A2[f"{li}_{e2}"].bias.copy_(W2[0])
        itf.trainable()
        groups = [{"params": ps, "lr": LR[g] * a.lr_scale} for g, ps in itf.param_groups().items()]
        opts[nm] = torch.optim.AdamW(groups, weight_decay=0.0)
        itfs[nm] = itf
        log(f"  variant {nm}: cfg {itf.cfg} | A-family params {itf.n_params() / 1e6:.1f}M")
    del ridge_W
    torch.cuda.empty_cache()
    report["variants"] = {nm: {"cfg": itfs[nm].cfg, "params": itfs[nm].n_params(), "history": []} for nm in names}

    def held_eval():
        tf, opv = {nm: [] for nm in names}, {nm: [] for nm in names}
        with torch.no_grad():
            for did, qs in vgroups.items():
                for it in qs:
                    op, N, rec, fcap, yt, lt, rows = rt.capture(did, it, probe)
                    for nm, itf in itfs.items():
                        r = {}
                        for li in lmap:
                            res = itf.read(li, fcap[li], op, N, need_logits=True)
                            r[li] = d13_terms(res["logits"], lt[li], res["y"], yt[li], N, metrics=True)
                        tf[nm].append(r)
                        opv[nm].append(rt.onpolicy_terms(itf, did, it, rec=rec))
        rt.R.itf = probe
        return {nm: (mean_terms(tf[nm]), mean_terms(opv[nm])) for nm in names}

    best = {nm: -1e9 for nm in names}

    def do_eval(ep):
        H = held_eval()
        for nm in names:
            htf, hop = H[nm]
            report["variants"][nm]["history"].append({"epoch": ep, "tf": htf, "op": hop})
            tag = ""
            if hop["cos"] > best[nm]:
                best[nm] = hop["cos"]
                save_d16(itfs[nm], os.path.join(a.out, f"d16_{nm}.pt"), {"stage": "s1", "epoch": ep, "held_op": hop})
                tag = " *best*"
            log(f"[{nm} ep {ep}] held tf js {htf['js']:.4f} cos {htf['cos']:.4f} | on-policy {fmt(hop)}{tag}")
        with open(os.path.join(a.out, "report.json"), "w") as f:
            json.dump(report, f, indent=1)

    do_eval(0)
    for ep in range(1, a.epochs + 1):
        dids = list(by_doc)
        rng.shuffle(dids)
        acc, n = 0, 0
        tl = {nm: {"kl": 0.0, "rel": 0.0} for nm in names}
        for o in opts.values():
            o.zero_grad(set_to_none=True)
        for did in dids:
            for it in by_doc[did]:
                op, N, rec, fcap, yt, lt, rows = rt.capture(did, it, probe)
                bad = False
                for nm, itf in itfs.items():
                    for li in lmap:
                        with torch.enable_grad():
                            res = itf.read(li, fcap[li], op, N, need_logits=True)
                            t = d13_terms(res["logits"], lt[li], res["y"], yt[li], N)
                            loss = (t["kl"] + a.rel_w * t["rel"]) / (len(lmap) * a.accum)
                            if not math.isfinite(float(loss)):
                                bad = True
                                continue
                            loss.backward()
                        tl[nm]["kl"] += float(t["kl"]) / len(lmap)
                        tl[nm]["rel"] += float(t["rel"]) / len(lmap)
                if bad:
                    log(f"  non-finite loss on {did}")
                n += 1
                acc += 1
                if acc == a.accum:
                    for nm, o in opts.items():
                        torch.nn.utils.clip_grad_norm_([p for g in o.param_groups for p in g["params"]], 1.0)
                        o.step()
                        o.zero_grad(set_to_none=True)
                    acc = 0
                if n % 200 == 0:
                    log(f"  [ep {ep}] {n} q ({time.time() - t0:.0f}s): " +
                        " ".join(f"{nm} {tl[nm]['kl'] / n:.4f}" for nm in names))
        if acc:
            for o in opts.values():
                o.step()
                o.zero_grad(set_to_none=True)
        for nm in names:
            report["variants"][nm].setdefault("train", []).append({k: v / max(n, 1) for k, v in tl[nm].items()})
        do_eval(ep)

    # ---------------- fixed τ / top-k sweep on the best checkpoints
    from xenc.d16_read import load_d16
    ranked = sorted(names, key=lambda nm: -best[nm])
    sw = [v for v in a.sweep.split(",") if v in names]
    if ranked and ranked[0] not in sw:
        sw.append(ranked[0])
    report["sweep"] = {}
    for nm in sw:
        itf, _ = load_d16(os.path.join(a.out, f"d16_{nm}.pt"), enc, B)
        itf = itf.to(dev)
        itf.att = probe.att
        for kind, vals in (("tau", a.taus), ("topk", a.topks)):
            for v in [x for x in vals.split(",") if x]:
                kw = {"tau": float(v)} if kind == "tau" else {"topk": int(v)}
                rows_ = [rt.onpolicy_terms(itf, did, it, **kw) for did, qs in vgroups.items() for it in qs]
                m = mean_terms(rows_)
                report["sweep"][f"{nm}:{kind}={v}"] = m
                log(f"[sweep {nm} {kind}={v}] on-policy {fmt(m)}")
        del itf
        rt.R.itf = probe
    report["ranking_by_onpolicy_cos"] = [(nm, best[nm]) for nm in ranked]
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    log("summary (best epoch, held-out on-policy): variant params js js_doc cos top5")
    for nm in ranked:
        h = max(report["variants"][nm]["history"], key=lambda r: r["op"]["cos"])
        log(f"  {nm:8s} {report['variants'][nm]['params'] / 1e6:6.1f}M ep{h['epoch']} js {h['op']['js']:.4f} "
            f"js_doc {h['op']['js_doc']:.4f} cos {h['op']['cos']:.4f} top5 {h['op']['top5']:.3f}")
    if torch.cuda.is_available():
        log(f"max_memory_allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    log(f"done ({time.time() - t0:.0f}s) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
