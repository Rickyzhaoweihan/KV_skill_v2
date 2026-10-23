"""D25: aggregate d25_diag.py / d25_sweep.py outputs into markdown tables (D0, D2, D3, D6, D1).

  python scripts/d25_summary.py --diag 9B=outputs/D25_diag_9b,27B=outputs/D25_diag_27b,4B=outputs/D25_diag_4b \
      --sweep 9B=outputs/D25_sweep_9b,27B=outputs/D25_sweep_27b --split test > outputs/D25_tables.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import string
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def _norm(s):
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _ans(gen):
    m = re.findall(r"<answer>(.*?)(?:</answer>|$)", gen or "", re.DOTALL)
    return m[-1].strip() if m else (gen or "").strip()


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan"), 0
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5
    return mu, sd / math.sqrt(len(xs)), len(xs)


def f(mu_se_n, k=3):
    mu, se, n = mu_se_n
    return "—" if n == 0 else f"{mu:.{k}f}±{se:.{k}f}"


_DOCS = {}


def passage_text(set_name, did, span):
    key = (set_name, did)
    if key not in _DOCS:
        p = _ROOT / "data" / "docs" / set_name / f"{did}.md"
        _DOCS[key] = p.read_text() if p.exists() else ""
    t = _DOCS[key]
    return t[span[0]:span[1]] if span else t


_SPANS = {}


def _span(key):
    if not _SPANS:
        for sp in ("train", "val", "test"):
            p = _ROOT / "data" / "d25_nqswap_split" / sp / "items.json"
            if p.exists():
                for it in json.load(open(p)):
                    _SPANS[it["key"]] = it["meta"].get("span")
    return _SPANS.get(key)


def classify(r, cond):
    """-> 'doc' | 'prior' | 'other' for conflict items, using the reader's own closed-book choice as the prior when
    the set has no fixed prior label (FaithEval)."""
    m = r["meta"]
    c = r[cond]
    if "pred" in c:
        prior = m.get("prior_label") or (r["closed_book"]["pred"] if r["closed_book"]["pred"] != r["gold"] else None)
        if c["pred"] == r["gold"]:
            return "doc"
        if prior and c["pred"] == prior:
            return "prior"
        return "other"
    a = _norm(_ans(c["gen"]))
    pri = [_norm(x) for x in m.get("prior_answers") or []]
    if a and any(p and (p == a or re.search(rf"\b{re.escape(p)}\b", a)) for p in pri):
        return "prior"
    text = passage_text("d25_nqswap", r["doc_id"], _span(r["key"]))
    if a and a in _norm(text):
        return "doc"
    return "other"


def load_diag(d, split):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, f"*_{split}.json"))):
        name = os.path.basename(p)[: -len(f"_{split}.json")]
        out[name] = json.load(open(p))["rows"]
    return out


def strong_prior(r):
    """closed book already gives the prior answer."""
    m = r["meta"]
    cb = r["closed_book"]
    if "pred" in cb:
        if m.get("prior_label"):
            return cb["pred"] == m["prior_label"]
        return cb["pred"] != r["gold"]                     # FaithEval: the reader's prior disagrees with the doc
    return classify(r, "closed_book") == "prior"


def d0_table(diags):
    print("\n### D0. Doc- vs prior-adherence, text mode and closed book (forced-choice / greedy span)\n")
    print("conflict items only; prior = the memorized answer (SCF speaker, ConflictQA memory answer, NQ-Swap original "
          "answer) or, for FaithEval, the reader's own closed-book choice. 'strong prior' = closed book already gives "
          "the prior. ORIG rows: the same questions on the unedited document (gold = prior).\n")
    print("| reader | set | n | closed: prior-adh | text: doc-adh | text: prior-adh | text: other | strong-prior n | "
          "strong-prior text doc-adh | strong-prior text prior-adh |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for rd, sets in diags.items():
        for name, rows in sets.items():
            conf = [r for r in rows if r["meta"].get("conflict")]
            if conf:
                cls = {c: [classify(r, c) for r in conf] for c in ("closed_book", "text")}
                sp = [r for r in conf if strong_prior(r)]
                spc = [classify(r, "text") for r in sp]
                print(f"| {rd} | {name} | {len(conf)} | {f(ms([float(x == 'prior') for x in cls['closed_book']]))} | "
                      f"{f(ms([float(x == 'doc') for x in cls['text']]))} | {f(ms([float(x == 'prior') for x in cls['text']]))} | "
                      f"{f(ms([float(x == 'other') for x in cls['text']]))} | {len(sp)} | "
                      f"{f(ms([float(x == 'doc') for x in spc]))} | {f(ms([float(x == 'prior') for x in spc]))} |")
            orig = [r for r in rows if r["meta"].get("variant") == "orig"]
            if orig:
                acc = [float(r["text"]["pred"] == r["gold"]) for r in orig]
                cb = [float(r["closed_book"]["pred"] == r["gold"]) for r in orig]
                print(f"| {rd} | {name} ORIG (control) | {len(orig)} | closed acc {f(ms(cb))} | text acc {f(ms(acc))} | | | | | |")
            ctrl = [r for r in rows if not r["meta"].get("conflict") and r["meta"].get("variant") != "orig"
                    and "pred" in r["text"]]
            if ctrl:
                acc = [float(r["text"]["pred"] == r["gold"]) for r in ctrl]
                cb = [float(r["closed_book"]["pred"] == r["gold"]) for r in ctrl]
                print(f"| {rd} | {name} no-conflict control | {len(ctrl)} | closed acc {f(ms(cb))} | text acc {f(ms(acc))} | | | | | |")


def _prior_of(r):
    m = r["meta"]
    if m.get("prior_label"):
        return m["prior_label"]
    cb = r["closed_book"].get("pred")
    return cb if cb and cb != r["gold"] else None


def lens_curves(rows, cond, which):
    """per-layer: fraction of items with margin > 0 and the mean margin.
    which = 'doc_vs_prior' (logit[gold] - logit[prior]) or 'prior_top' (logit[prior] - max other)."""
    L = None
    frac, mean, depth = defaultdict(list), defaultdict(list), []
    from xenc.d25_probe import first_stable
    for r in rows:
        c = r[cond]
        if "lens" not in c:
            continue
        prior = _prior_of(r)
        if not prior:
            continue
        labs = "ABCDE"[: len(c["logits"])]
        gi, pi = labs.index(r["gold"]), labs.index(prior)
        lens = c["lens"]
        L = len(lens)
        if which == "doc_vs_prior":
            mg = [row[gi] - row[pi] for row in lens]
        else:
            mg = [row[pi] - max(v for k, v in enumerate(row) if k != pi) for row in lens]
        for l, v in enumerate(mg):
            frac[l].append(float(v > 0))
            mean[l].append(v)
        depth.append(first_stable(mg))
    return L, frac, mean, depth


def d2_table(diags):
    print("\n### D2. Logit lens (answer letters read from each layer's residual at the answer position)\n")
    print("commit depth (text, items answered with the DOC label): first layer from which logit[doc] > logit[prior] "
          "holds to the last layer. recall depth (closed book, items answered with the PRIOR): first layer from which "
          "the prior letter is top-1 to the end. Depth as layer index / normalized ρ = l/(L−1).\n")
    print("| reader | set | commit n | commit depth median (ρ) | IQR | recall n | recall depth median (ρ) | IQR |")
    print("|---|---|---|---|---|---|---|---|")
    curves = {}
    for rd, sets in diags.items():
        for name, rows in sets.items():
            conf = [r for r in rows if r["meta"].get("conflict") and "pred" in r["text"]]
            if not conf:
                continue
            tdoc = [r for r in conf if r["text"]["pred"] == r["gold"]]
            cpri = [r for r in conf if r["closed_book"]["pred"] == _prior_of(r)]
            L, fr, mn, dep = lens_curves(tdoc, "text", "doc_vs_prior")
            L2, fr2, mn2, dep2 = lens_curves(cpri, "closed_book", "prior_top")
            curves[(rd, name)] = (L, fr, mn, fr2, mn2)
            # also prior-following text items
            tpri = [r for r in conf if r["text"]["pred"] == _prior_of(r)]
            curves[(rd, name, "tpri")] = lens_curves(tpri, "text", "doc_vs_prior")

            def q(d, p):
                d = sorted(d)
                return d[min(len(d) - 1, int(p * len(d)))] if d else float("nan")
            Lr = (L or L2 or 1) - 1
            print(f"| {rd} | {name} | {len(dep)} | {q(dep, .5)} ({q(dep, .5) / Lr:.2f}) | {q(dep, .25)}–{q(dep, .75)} | "
                  f"{len(dep2)} | {q(dep2, .5)} ({q(dep2, .5) / Lr:.2f}) | {q(dep2, .25)}–{q(dep2, .75)} |")
    print("\nPer-layer curves (every 4th layer): text = P(logit[doc] > logit[prior]) on doc-answered items / mean margin; "
          "closed = P(prior is top-1) on prior-answered items / mean margin; text-prior = P(logit[doc] > logit[prior]) "
          "on text items that ended on the PRIOR.\n")
    for key, (L, fr, mn, fr2, mn2) in [(k, v) for k, v in curves.items() if len(k) == 2]:
        if not L:
            continue
        tp = curves.get((*key, "tpri"))
        print(f"\n**{key[0]} {key[1]}** (L={L})\n")
        print("| layer | ρ | text P(doc>prior) | text margin | closed P(prior top) | closed margin | text-prior-items P(doc>prior) |")
        print("|---|---|---|---|---|---|---|")
        for l in list(range(0, L, 4)) + ([L - 1] if (L - 1) % 4 else []):
            tpf = (sum(tp[1][l]) / len(tp[1][l])) if tp and tp[0] and tp[1][l] else float("nan")
            print(f"| {l} | {l / (L - 1):.2f} | {ms(fr[l])[0]:.2f} | {ms(mn[l])[0]:+.2f} | {ms(fr2[l])[0]:.2f} | "
                  f"{ms(mn2[l])[0]:+.2f} | {tpf:.2f} |")


def d3_table(diags):
    print("\n### D3. Context-attention profile (text mode, full-attention layers, mean over heads)\n")
    print("mass of the ANSWER-position query (last row) on: doc tokens, evidence (asked passage / quoted line ±60 "
          "chars), system/template head, own question rows. 'doc-adh' vs 'prior-adh' split by the text outcome.\n")
    for rd, sets in diags.items():
        for name, rows in sets.items():
            conf = [r for r in rows if r["meta"].get("conflict") and r.get("attn")]
            if not conf:
                continue
            layers = sorted(int(x) for x in conf[0]["attn"])
            cl = [classify(r, "text") for r in conf]
            print(f"\n**{rd} {name}** (n={len(conf)}; doc-adh n={cl.count('doc')}, prior-adh n={cl.count('prior')})\n")
            print("| layer | doc (last row) | evid (last row) | evid / doc | head | self | doc (mean rows) | "
                  "evid: doc-adh items | evid: prior-adh items |")
            print("|---|---|---|---|---|---|---|---|---|")
            for li in layers:
                g = lambda k, i, sel=None: ms([r["attn"][str(li)][k][i] for r, c in zip(conf, cl)   # noqa: E731
                                              if k in r["attn"][str(li)] and (sel is None or c == sel)])[0]
                dl, el = g("doc", 0), g("evid", 0)
                print(f"| {li} | {dl:.3f} | {el:.4f} | {el / max(dl, 1e-9):.3f} | {g('head', 0):.3f} | {g('self', 0):.3f} | "
                      f"{g('doc', 1):.3f} | {g('evid', 0, 'doc'):.4f} | {g('evid', 0, 'prior'):.4f} |")


def d6_table(diag_dirs, split):
    import torch
    print("\n### D6. Conflict direction h(CF doc) − h(original doc), last position (answer slot), text mode\n")
    print("rel‖Δ‖ = ‖mean Δ‖ / mean ‖h‖; sep = held-out accuracy of classifying CF vs ORIG by the projection on the "
          "mean-Δ direction fitted on the other half of the pairs (chance .5); cos½ = cosine between the two halves' "
          "directions.\n")
    for rd, d in diag_dirs.items():
        for p in sorted(glob.glob(os.path.join(d, f"h_*_{split}.pt"))):
            name = os.path.basename(p)[2: -len(f"_{split}.pt")]
            H = torch.load(p)
            rows = json.load(open(os.path.join(d, f"{name}_{split}.json")))["rows"]
            byk = {r["key"]: r for r in rows}
            pairs = []
            for r in rows:
                m = r["meta"]
                if m.get("variant") == "orig":
                    ck = m.get("pair_key")
                    if not ck:   # ConflictQA: pair by bundle + question order
                        continue
                    if ck in H and r["key"] in H:
                        pairs.append((H[ck].float(), H[r["key"]].float()))
            if not pairs:     # ConflictQA keys: cf item k, orig item k+1
                keys = [r["key"] for r in rows]
                for r in rows:
                    if r["meta"].get("variant") == "cf":
                        num = int(r["key"].rsplit("_", 1)[1])
                        ok = r["key"].rsplit("_", 1)[0] + f"_{num + 1:05d}"
                        if ok in H and r["key"] in H and byk.get(ok, {}).get("meta", {}).get("variant") == "orig":
                            pairs.append((H[r["key"]].float(), H[ok].float()))
            if len(pairs) < 10:
                continue
            A = torch.stack([x for x, _ in pairs])
            B = torch.stack([y for _, y in pairs])
            n, L = A.shape[0], A.shape[1]
            g = torch.Generator().manual_seed(0)
            perm = torch.randperm(n, generator=g)
            h1, h2 = perm[: n // 2], perm[n // 2:]
            print(f"\n**{rd} {name}** ({n} pairs, L={L})\n")
            print("| layer | ρ | rel‖Δ‖ | sep (held-out) | cos½ |")
            print("|---|---|---|---|---|")
            for l in list(range(0, L, 4)) + ([L - 1] if (L - 1) % 4 else []):
                D = A[:, l] - B[:, l]
                rel = float(D.mean(0).norm() / torch.cat([A[:, l], B[:, l]]).norm(dim=-1).mean())
                accs, dirs = [], []
                for tr, te in ((h1, h2), (h2, h1)):
                    u = D[tr].mean(0)
                    u = u / u.norm()
                    mid = 0.5 * ((A[tr, l] @ u).mean() + (B[tr, l] @ u).mean())
                    accs.append(0.5 * (float(((A[te, l] @ u) > mid).float().mean()) +
                                       float(((B[te, l] @ u) <= mid).float().mean())))
                    dirs.append(u)
                print(f"| {l} | {l / (L - 1):.2f} | {rel:.3f} | {sum(accs) / 2:.3f} | {float(dirs[0] @ dirs[1]):.3f} |")


def d1_table(sweeps, split, diag_dirs=None):
    print("\n### D1. Per-layer causal sweep of the base read (UNREGISTERED: diagnostic / step 0), forced choice\n")
    print("kv = reader prompt (doc tail + question) with the read at the listed layers only (others: the reader's "
          "own attention). VAL documents. acc = accuracy (= doc-adherence on conflict sets; Shakespeare-CF: speaker "
          "conflict items only), prior = prior-adherence (FaithEval: the reader's own closed-book choice).\n")
    for rd, d in sweeps.items():
        p = os.path.join(d, f"sweep_{split}.json")
        if not os.path.exists(p):
            continue
        J = json.load(open(p))
        rows = J["rows"]
        sets = sorted({r["set"] for r in rows})
        meta = {}
        for sname in sets:
            ip = _ROOT / "data" / f"{sname}_split" / split / "items.json"
            if ip.exists():
                for it in json.load(open(ip)):
                    meta[it["key"]] = it.get("meta") or {}
        cbp = {}
        if diag_dirs and rd in diag_dirs:
            for name, drows in load_diag(diag_dirs[rd], split).items():
                for r in drows:
                    if "pred" in r["closed_book"]:
                        cbp[r["key"]] = r["closed_book"]["pred"]
        for r in rows:
            m = meta.get(r["key"], {})
            if r["set"] == "shakespeare_cf" and not m.get("conflict"):
                r["prior"] = None
                r["noconf"] = True
            if r["set"] == "d25_fecf" and cbp.get(r["key"]) and cbp[r["key"]] != r["gold"]:
                r["prior"] = cbp[r["key"]]
        cfgs = ["all", "none"] + [f"only{l}" for l in J["layers"]] + [f"loo{l}" for l in J["layers"]]

        def stat(cfg, name, conf_only):
            rs = [r for r in rows if r["set"] == name and cfg in r["cfg"] and not (conf_only and r.get("noconf"))]
            acc, pri, wr, lp = [], [], [], []
            for r in rs:
                c, labs = r["cfg"][cfg], r["labels"]
                pred = labs[max(range(len(labs)), key=lambda k: c["lp"][k])]
                acc.append(float(pred == r["gold"]))
                lp.append(c["lp"][labs.index(r["gold"])])
                if r.get("prior"):
                    pri.append(float(pred == r["prior"]))
                if "w" in c:
                    wp = labs[max(range(len(labs)), key=lambda k: c["w"][k])]
                    wr.append(float(wp == r["gold"]))
            return ms(acc), ms(pri), ms(wr), ms(lp)
        print(f"\n**{rd}** layers {J['layers']}, base gates {[round(g, 2) for g in J['base_gates']]}\n")
        print("| config | " + " | ".join(f"{s} acc | {s} prior | {s} kv−wrong" for s in sets) + " | mean conflict acc | normal acc |")
        print("|---|" + "---|" * (3 * len(sets) + 2))
        for cfg in cfgs:
            cells, conf, norm = [], [], []
            for s in sets:
                acc, pri, wr, lp = stat(cfg, s, s == "shakespeare_cf")
                cells += [f"{acc[0]:.3f}", f"{pri[0]:.3f}" if pri[2] else "—",
                          f"{acc[0] - wr[0]:+.3f}" if wr[2] else "—"]
                (norm if s.startswith(("quality", "longhealth")) else conf).append(acc[0])
            print(f"| {cfg} | " + " | ".join(cells) + f" | {sum(conf) / max(len(conf), 1):.3f} | "
                  f"{sum(norm) / max(len(norm), 1):.3f} |")
        if J.get("greedy"):
            print(f"\nGreedy forward selection ({rd}, subsample): " +
                  " → ".join(f"+L{g['add']} (U {g['U']:.3f})" for g in J["greedy"]))


def gates_table(gates, diag_dirs):
    print("\n### Method: learned read gates (M2 / M5 / M2-deep), TEST documents, forced choice\n")
    print("kv = base interface UNREGISTERED (step 0) with base gates vs learned gates (trained on TRAIN conflict docs + "
          "QuALITY train; selected on val). text = the same reader with the document in context (D0 rows, same keys). "
          "Conflict sets: acc = doc-adherence. Paired differences ± se.\n")
    print("| run | set | n | text | kv base | kv learned | learned − base | learned − text | base − text | kv prior (base→learned) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for tag, (d, rd) in gates.items():
        p = os.path.join(d, "gates.json")
        if not os.path.exists(p):
            continue
        J = json.load(open(p))
        if "test_rows" not in J:
            continue
        txt, conf = {}, {}
        for name, rows in load_diag(diag_dirs[rd], "test").items():
            for r in rows:
                conf[r["key"]] = bool(r["meta"].get("conflict"))
                if "pred" in r["text"]:
                    txt[r["key"]] = float(r["text"]["pred"] == r["gold"])
        base = {r["key"]: r for r in J["test_rows"]["base"]}
        lrn = {r["key"]: r for r in J["test_rows"]["learned"]}
        for name in sorted({r["set"] for r in base.values()}):
            for sub, keep in (("conflict", lambda r: conf.get(r["key"])), ("all", lambda r: True)):
                ks = [k for k, r in base.items() if r["set"] == name and k in lrn and keep(r)]
                if name == "shakespeare_cf" and sub == "all":
                    ks = [k for k, r in base.items() if r["set"] == name and not conf.get(k)]
                    sub = "no-conflict"
                elif name != "shakespeare_cf" and sub == "conflict":
                    continue
                if not ks:
                    continue
                b = [float(base[k]["pred"] == base[k]["gold"]) for k in ks]
                l_ = [float(lrn[k]["pred"] == lrn[k]["gold"]) for k in ks]
                t = [txt.get(k) for k in ks]
                kt = [k for k, x in zip(ks, t) if x is not None]
                tt = [txt[k] for k in kt]
                lt = [float(lrn[k]["pred"] == lrn[k]["gold"]) - txt[k] for k in kt]
                bt = [float(base[k]["pred"] == base[k]["gold"]) - txt[k] for k in kt]
                pb = [float(base[k]["pred"] == base[k]["prior"]) for k in ks if base[k]["prior"] and conf.get(k)]
                pl = [float(lrn[k]["pred"] == lrn[k]["prior"]) for k in ks if lrn[k]["prior"] and conf.get(k)]
                pr = f"{ms(pb)[0]:.3f}→{ms(pl)[0]:.3f}" if pb else "—"
                print(f"| {tag} | {name} ({sub}) | {len(ks)} | {f(ms(tt)) if tt else '—'} | {ms(b)[0]:.3f} | {ms(l_)[0]:.3f} | "
                      f"{f(ms([x - y for x, y in zip(l_, b)]))} | {f(ms(lt)) if lt else '—'} | {f(ms(bt)) if bt else '—'} | {pr} |")
        print(f"\n{tag}: selected step {J['best']['step']}, gates {[round(x, 3) for x in J['best']['g']]}"
              + (f", beta {[round(x, 3) for x in J['best']['beta']]}" if any(J['best']['beta']) else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", default="")
    ap.add_argument("--sweep", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--sweep_split", default="val")
    ap.add_argument("--gates", default="", help="tag=dir@reader,...  (reader = a --diag key)")
    a = ap.parse_args()
    dd = dict(x.split("=", 1) for x in a.diag.split(",") if x)
    diags = {k: load_diag(v, a.split) for k, v in dd.items()}
    if diags:
        d0_table(diags)
        d2_table(diags)
        d3_table(diags)
        d6_table(dd, a.split)
    sw = dict(x.split("=", 1) for x in a.sweep.split(",") if x)
    if sw:
        d1_table(sw, a.sweep_split, dd)
    gt = {x.split("=", 1)[0]: tuple(x.split("=", 1)[1].split("@")) for x in a.gates.split(",") if x}
    if gt:
        gates_table(gt, dd)


if __name__ == "__main__":
    main()
