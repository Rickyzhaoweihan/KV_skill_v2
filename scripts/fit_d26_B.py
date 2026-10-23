"""D26: span-aligned ridge write-back B for a reader of ANOTHER family (D10/D11's encoder-side post B, cross-tokenizer).

Both models prefill the same `text` prompt (system + document + question) in their OWN chat template. At reader
read layer ℓ (paired with encoder layer e by xenc.d26_read.pick_read_layers) the reader's attention-module output
a_r[t] (d_r) is regressed on the encoder's attention-module output a_e[j(t)] (d_e) at the character-aligned encoder
row j(t) (xenc.d26_read.coords/align over the content = system prompt + user message): a_r ≈ B a_e, no intercept
(as encB5_*). Label-free; train documents only (QuALITY 60 / LongHealth folds ∉ --exclude_folds 12 / MuSiQue 40 /
LastingBench 30, D11's xf01 recipe); held-out R² on the last 10% of each task's documents; λ per layer from --lams.
The output loads with xenc.d13_read.load_B (keys = reader layers).

    python scripts/fit_d26_B.py --reader glm4 --out outputs/D26_maps/B_glm4.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from fit_state_maps import Gram, ridge  # noqa: E402
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="glm4")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layers", default="auto")
    ap.add_argument("--splits", default="quality:60,longhealth:12,musique:40,lastingbench:30")
    ap.add_argument("--exclude_folds", default="0,1")
    ap.add_argument("--max_tokens", type=int, default=9000, help="encoder tokens per prompt (reader rows cut to match)")
    ap.add_argument("--heldout_frac", type=float, default=0.1)
    ap.add_argument("--lams", default="1e-4,1e-3,1e-2,1e-1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mlp_hidden", type=int, default=0, help=">0: ALSO fit a label-free low-rank residual MLP on top of "
                    "the ridge B (d_e -> hidden -> d_r, GELU) and report its held-out R² (richer-B diagnostic; saved to "
                    "<out>_mlp.pt, not used by load_B)")
    ap.add_argument("--mlp_rows", type=int, default=1024, help="rows per document cached for the MLP fit")
    ap.add_argument("--mlp_epochs", type=int, default=8)
    a = ap.parse_args(argv)
    from fit_d26 import parse_layers
    from xenc.backbone import input_device, load_encoder
    from xenc.d13_read import EncTeacher
    from xenc.d26_read import D26Reader, align, chat_str, coords, encode, load_reader_d26, resolve_hub
    from xenc.doc_tasks import ctx_prompt, load_docs
    from xenc.state_skill import text_config
    a.reader = resolve_hub(a.reader)
    reader, tok_r = load_reader_d26(a.reader, device=a.device)
    enc, tok_e = load_encoder(a.encoder, device_map=a.device)
    lmap = parse_layers(a.layers, reader, enc)
    log(f"D26 B fit: {a.encoder} -> {a.reader} | layers {lmap}")
    R = D26Reader(reader, types.SimpleNamespace(layers=sorted(lmap), layer_map=lmap))
    Tc = EncTeacher(enc, sorted(set(lmap.values())))
    dev = input_device(reader)
    d_e, d_r = int(text_config(enc).hidden_size), int(text_config(reader).hidden_size)
    G = {s: {li: Gram(d_e, d_r, dev) for li in lmap} for s in ("fit", "held")}
    excl = {int(x) for x in a.exclude_folds.split(",") if x != ""}
    n_rows = {"fit": 0, "held": 0}
    cache_rows: dict = {}
    for spec in a.splits.split(","):
        name, nd = spec.split(":")
        sd = f"data/{name}_split"
        docs = load_docs(sd)
        items = [i for i in load_split_items(sd, "train") if (_get(i, "meta") or {}).get("fold") not in excl]
        first = {}
        for it in items:
            first.setdefault(_get(it, "doc_id"), it)
        dids = sorted(first)[: int(nd)]
        n_held = max(1, int(round(len(dids) * a.heldout_frac)))
        for k, did in enumerate(dids):
            s = "held" if k >= len(dids) - n_held else "fit"
            user = ctx_prompt(first[did], docs[did])
            Se, Sr = chat_str(tok_e, a.encoder, SYSTEM_ANSWER, user), chat_str(tok_r, a.reader, SYSTEM_ANSWER, user)
            e_ids, e_off = encode(tok_e, Se)
            r_ids, r_off = encode(tok_r, Sr)
            e_ids, e_off = e_ids[: a.max_tokens], e_off[: a.max_tokens]
            ce, cr = coords(Se, e_off, [SYSTEM_ANSWER, user]), coords(Sr, r_off, [SYSTEM_ANSWER, user])
            n_r = sum(1 for c in cr if c <= ce[-1] + 1e-6)                       # reader rows the encoder covers
            r_ids, cr = r_ids[:n_r], cr[:n_r]
            idx, valid = align(cr, ce)
            R.mode, R.xcap, R.ocap = "cap", {}, {}
            Tc.rec, Tc.kv_only = {}, False
            with torch.no_grad():
                reader(input_ids=torch.tensor([r_ids], device=dev), use_cache=False, logits_to_keep=1)
                enc(input_ids=torch.tensor([e_ids], device=input_device(enc)), use_cache=False, logits_to_keep=1)
            R.mode = "off"
            v = valid.to(dev)
            ii = idx.to(dev)[v]
            for li, e in lmap.items():
                G[s][li].add(Tc.rec[e]["out"].float().to(dev)[ii], R.ocap[li].float()[v])
                if a.mlp_hidden:
                    xe, yr = Tc.rec[e]["out"].to(dev)[ii], R.ocap[li][v]
                    sel = torch.randperm(xe.shape[0], device=dev)[: a.mlp_rows]
                    cache_rows.setdefault((s, li), []).append((xe[sel].half().cpu(), yr[sel].half().cpu()))
            n_rows[s] += int(v.sum())
            Tc.rec = None
        log(f"  {name}: {len(dids)} docs ({n_held} held out) | rows fit {n_rows['fit']} held {n_rows['held']}")
    B, rep = {}, {}
    lams = [float(x) for x in a.lams.split(",")]
    for li in lmap:
        cands = [(G["held"][li].r2(ridge(G["fit"][li], lm)), lm) for lm in lams]
        best = max(cands)[1]
        W = ridge(G["fit"][li], best)
        B[str(li)] = W.t().float().cpu()                                        # [d_r, d_e]
        rep[str(li)] = {"enc_layer": lmap[li], "lam": best, "r2_fit": G["fit"][li].r2(W), "r2_held": G["held"][li].r2(W),
                        "grid": {f"{lm:g}": r for r, lm in cands}}
        log(f"  L{li}<-E{lmap[li]}: λ {best:g} R² fit {rep[str(li)]['r2_fit']:.3f} held {rep[str(li)]['r2_held']:.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    mlp_sd = {}
    for li in (lmap if a.mlp_hidden else []):
        Xf = torch.cat([x for x, _ in cache_rows[("fit", li)]]).float()
        Yf = torch.cat([y for _, y in cache_rows[("fit", li)]]).float()
        Xh = torch.cat([x for x, _ in cache_rows[("held", li)]]).float().to(dev)
        Yh = torch.cat([y for _, y in cache_rows[("held", li)]]).float().to(dev)
        Bw = B[str(li)].to(dev)
        net = torch.nn.Sequential(torch.nn.Linear(d_e, a.mlp_hidden), torch.nn.GELU(),
                                  torch.nn.Linear(a.mlp_hidden, d_r)).to(dev)
        torch.nn.init.zeros_(net[2].weight)
        torch.nn.init.zeros_(net[2].bias)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
        sst = float(((Yh - Yh.mean(0)) ** 2).sum())
        with torch.no_grad():
            r2_lin = 1 - float(((Xh @ Bw.t() - Yh) ** 2).sum()) / sst
        best, best_sd = r2_lin, None
        for ep in range(a.mlp_epochs):
            perm = torch.randperm(Xf.shape[0])
            for i in range(0, Xf.shape[0], 1024):
                j = perm[i:i + 1024]
                x, y = Xf[j].to(dev), Yf[j].to(dev)
                loss = ((x @ Bw.t() + net(x) - y) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                r2 = 1 - float(((Xh @ Bw.t() + net(Xh) - Yh) ** 2).sum()) / sst
            if r2 > best:
                best, best_sd = r2, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        rep[str(li)].update({"mlp_hidden": a.mlp_hidden, "r2_held_rows_lin": r2_lin, "r2_held_rows_mlp": best,
                             "mlp_rows_fit": int(Xf.shape[0])})
        if best_sd is not None:
            mlp_sd[str(li)] = best_sd
        log(f"  [mlp {a.mlp_hidden}] L{li}: held-out R² (sampled rows) ridge {r2_lin:.3f} -> ridge+MLP {best:.3f}")
    if a.mlp_hidden:
        torch.save({"mlp_hidden": a.mlp_hidden, "mlp": mlp_sd, "layer_map": lmap, "B": a.out},
                   a.out.replace(".pt", "_mlp.pt"))
    torch.save({"level": "post", "mlp": None, "enc_B": B, "layer_map": lmap, "reader": a.reader, "encoder": a.encoder,
                "align": "char-span (xenc.d26_read.coords/align)", "rows": n_rows}, a.out)
    with open(a.out.replace(".pt", "_report.json"), "w") as f:
        json.dump({"args": vars(a), "layers": rep, "rows": n_rows}, f, indent=1)
    log(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
