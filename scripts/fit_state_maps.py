"""D10: ridge-fit the cross-model state-read maps A_ℓ, B_ℓ from paired activations (no document QA labels).

Both models read the SAME token ids (Qwen3.5 sizes share one vocabulary) — the `text` prompts of training
documents. At every token and every reader DeltaNet layer ℓ (paired with encoder layer ℓ' by normalized depth):
    A_ℓ : l2(q_reader,ℓ)   [Hk_r·dk]  ->  l2(q_encoder,ℓ')  [Hk_e·dk]     (query translation)
    B_ℓ : o_encoder,ℓ'     [Hv_e·dv]  ->  o_reader,ℓ        [Hv_r·dv]     (read-output translation;
                                                               o = pre-norm core_attn_out = Sᵀq of each model)
so that y = B Sᵀ_enc l2(A q) approximates the reader's own read of a state holding the same text. Ridge:
W = (XᵀX + λ·tr(XᵀX)/d·I)⁻¹XᵀY, accumulated as Gram matrices (fp64). R² on held-out documents (last 10%),
next to the structural-identity A for reference. Output loads via register_state.py --init <maps.pt>.

    python scripts/fit_state_maps.py --reader Qwen/Qwen3.5-9B --encoder Qwen/Qwen3.5-2B \
        --split_dir data/quality_split --max_docs 60 --out outputs/D10_maps/maps_9b_2b.pt
    python scripts/fit_state_maps.py --tiny --out /tmp/maps.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from register_doc import SYSTEM_ANSWER, _get, load_split_items, log  # noqa: E402

FORMAT = "xenc.state_maps.v1"


def _l2(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


class Gram:
    def __init__(self, dx, dy, dev):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)   # noqa: E731
        self.xx, self.xy, self.yy, self.ys, self.n = z(dx, dx), z(dx, dy), z(()), z(dy), 0

    def add(self, X, Y):
        """fp32 matmuls (fp64 is ~64x slower on L40S) accumulated into fp64 buffers, in 4096-row chunks."""
        X, Y = X.float(), Y.float()
        for i in range(0, X.shape[0], 4096):
            x, y = X[i:i + 4096], Y[i:i + 4096]
            self.xx += (x.t() @ x).double()
            self.xy += (x.t() @ y).double()
            self.yy += (y.double() * y.double()).sum()
            self.ys += y.double().sum(0)
        self.n += X.shape[0]

    def r2(self, W):
        """W [dx, dy]: Y ≈ X W."""
        sse = self.yy - 2 * (W * self.xy).sum() + (W * (self.xx @ W)).sum()
        sst = self.yy - (self.ys * self.ys).sum() / max(self.n, 1)
        return float(1 - sse / sst)


def ridge(g: Gram, lam: float):
    d = g.xx.shape[0]
    reg = lam * torch.trace(g.xx) / d
    return torch.linalg.solve(g.xx + reg * torch.eye(d, dtype=g.xx.dtype, device=g.xx.device), g.xy)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reader", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--encoder", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--split_dir", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_docs", type=int, default=60)
    ap.add_argument("--test_fold", type=int, default=None)
    ap.add_argument("--max_tokens", type=int, default=12000)
    ap.add_argument("--heldout_frac", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--branches", default="state,attn", help="which maps to fit: state (DeltaNet) and/or attn")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(argv)
    br = {b.strip() for b in a.branches.split(",") if b.strip()}
    from xenc.backbone import build_chat, input_device
    from xenc.doc_tasks import ctx_prompt
    from xenc.state_interface import StateReader, _struct_eye, make_state_interface
    from xenc.state_skill import depth_map, state_dims

    if a.tiny:
        from register_state import tiny_state_setup
        tz = tiny_state_setup()
        reader, enc, tok, hub, docs, items = tz["reader"], tz["encoder"], tz["tok"], tz["hub"], tz["docs"], tz["train"]
        a.heldout_frac = 0.34
    else:
        from xenc.backbone import load_encoder, load_reader
        from xenc.doc_tasks import load_docs
        reader, tok = load_reader(a.reader, device_map=a.device)
        enc = reader if a.encoder == "same" else load_encoder(a.encoder, device_map=a.device)[0]
        hub, docs = a.reader, load_docs(a.split_dir)
        items = load_split_items(a.split_dir, a.split)
        if a.test_fold is not None:
            items = [i for i in items if (_get(i, "meta") or {}).get("fold") != a.test_fold]
    first = {}
    for it in items:                      # one question per document (its suffix is a few dozen tokens)
        first.setdefault(_get(it, "doc_id"), it)
    doc_ids = sorted(first)[: a.max_docs] if a.max_docs else sorted(first)
    n_held = max(1, int(round(len(doc_ids) * a.heldout_frac)))
    rd, ed = state_dims(reader), state_dims(enc)
    lmap = depth_map(rd["linear"], rd["n_layers"], ed["linear"], ed["n_layers"])
    Rr = StateReader(reader, make_state_interface(reader, reader, "none", "none"), record_mag=False)
    Re = Rr if enc is reader else StateReader(enc, make_state_interface(enc, enc, "none", "none"), record_mag=False)
    dev = input_device(reader)
    dka, dkb = rd["n_k"] * rd["dk"], ed["n_k"] * ed["dk"]
    dvr, dve = rd["n_v"] * rd["dv"], ed["n_v"] * ed["dv"]
    G = {s: {li: (Gram(dka, dkb, dev), Gram(dve, dvr, dev)) for li in lmap} for s in ("fit", "held")} if "state" in br else {}
    if "attn" in br:            # per reader q-head h: reader q_h -> encoder q_h' (pre-RoPE), encoder o_h' -> reader o_h
        from xenc.xattn_read import AttnReader, attn_dims, full_depth_map, make_attn_interface
        ra, ea = attn_dims(reader), attn_dims(enc)
        amap = full_depth_map(ra, ea)
        Ar = AttnReader(reader, make_attn_interface(reader, ra), record_mag=False)
        Ae = Ar if enc is reader else AttnReader(enc, make_attn_interface(enc, ea), record_mag=False)
        hmap = [h * ea["n_q"] // ra["n_q"] for h in range(ra["n_q"])]
        GA = {s: {li: [(Gram(ra["hd"], ea["hd"], dev), Gram(ea["hd"], ra["hd"], dev)) for _ in range(ra["n_q"])]
                  for li in amap} for s in ("fit", "held")}
        log(f"attn map: {amap}; reader q-head -> encoder q-head {hmap}")
    log(f"fit maps: {len(doc_ids)} docs ({n_held} held out); layer map {lmap}")
    for k, did in enumerate(doc_ids):
        s = "held" if k >= len(doc_ids) - n_held else "fit"
        ids = build_chat(tok, hub, SYSTEM_ANSWER, ctx_prompt(first[did], docs[did]))[: a.max_tokens]
        x = torch.tensor([ids])
        Rr.capture, Re.capture = ({}, {}) if "state" in br else (None, None)
        if "attn" in br:
            Ar.capture, Ae.capture = {}, {}
        with torch.no_grad():
            reader(input_ids=x.to(dev), use_cache=False, logits_to_keep=1)
            if enc is not reader:
                enc(input_ids=x.to(input_device(enc)), use_cache=False, logits_to_keep=1)
        if "attn" in br:
            for li, e in amap.items():
                qr, orr = Ar.capture[li]
                qe, oe = Ae.capture[e]
                for h, h2 in enumerate(hmap):
                    ga, gb = GA[s][li][h]
                    ga.add(qr[0, h].float().to(dev), qe[0, h2].float().to(dev))
                    gb.add(oe[0, h2].float().to(dev), orr[0, h].float().to(dev))
            Ar.capture = Ae.capture = None
        for li, e in (lmap.items() if "state" in br else []):
            qr, orr = Rr.capture[li]
            qe, oe = Re.capture[e]
            T = qr.shape[1]
            ga, gb = G[s][li]
            ga.add(_l2(qr.float()).reshape(T, -1).to(dev), _l2(qe.float()).reshape(T, -1).to(dev))
            gb.add(oe.float().reshape(T, -1).to(dev), orr.float().reshape(T, -1).to(dev))
        Rr.capture = Re.capture = None
        if (k + 1) % 10 == 0 or a.tiny:
            log(f"  [{k + 1}/{len(doc_ids)}] {len(ids)} tokens")
    A, B, rep = {}, {}, {}
    Aeye = _struct_eye(ed["n_k"], rd["n_k"], rd["dk"]).to(dev).double() if rd["dk"] == ed["dk"] else None
    AA, AB, arep = {}, {}, {}
    if "attn" in br:
        for li in amap:
            WA, WB, r2 = [], [], []
            for h in range(ra["n_q"]):
                ga, gb = GA["fit"][li][h]
                ha, hb = GA["held"][li][h]
                wa, wb = ridge(ga, a.lam), ridge(gb, a.lam)
                eye = torch.eye(ra["hd"], dtype=torch.float64, device=dev)
                r2.append((ha.r2(wa), ha.r2(eye), hb.r2(wb), hb.r2(eye)))
                WA.append(wa.t().float().cpu())
                WB.append(wb.t().float().cpu())
            AA[str(li)], AB[str(li)] = torch.stack(WA), torch.stack(WB)
            mean = [sum(x[i] for x in r2) / len(r2) for i in range(4)]
            arep[li] = {"enc_layer": amap[li], "r2_A_held": mean[0], "r2_A_identity_held": mean[1],
                        "r2_B_held": mean[2], "r2_B_identity_held": mean[3]}
            log(f"  attn L{li:2d}<-E{amap[li]:2d}  A R² held {mean[0]:.3f} (identity {mean[1]:.3f}) | "
                f"B R² held {mean[2]:.3f} (identity {mean[3]:.3f})")
    for li in (lmap if "state" in br else []):
        ga, gb = G["fit"][li]
        Wa, Wb = ridge(ga, a.lam), ridge(gb, a.lam)
        ha, hb = G["held"][li]
        rep[li] = {"enc_layer": lmap[li], "r2_A_fit": ga.r2(Wa), "r2_A_held": ha.r2(Wa),
                   "r2_A_identity_held": ha.r2(Aeye.t()) if Aeye is not None else None,
                   "r2_B_fit": gb.r2(Wb), "r2_B_held": hb.r2(Wb)}
        A[str(li)] = Wa.t().float().cpu()
        B[str(li)] = Wb.t().float().cpu()
        log(f"  L{li:2d}<-E{lmap[li]:2d}  A R² fit {rep[li]['r2_A_fit']:.3f} held {rep[li]['r2_A_held']:.3f} "
            f"(identity {rep[li]['r2_A_identity_held'] if Aeye is None else round(rep[li]['r2_A_identity_held'], 3)}) | "
            f"B R² fit {rep[li]['r2_B_fit']:.3f} held {rep[li]['r2_B_held']:.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rep = {"state": rep, "attn": arep}
    torch.save({"format": FORMAT, "A": A, "B": B, "layer_map": lmap, "attn_A": AA, "attn_B": AB,
                "attn_map": amap if "attn" in br else {}, "report": rep, "args": vars(a),
                "docs": doc_ids}, a.out)
    with open(os.path.splitext(a.out)[0] + "_report.json", "w") as f:
        json.dump(rep, f, indent=1)
    log(f"saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
