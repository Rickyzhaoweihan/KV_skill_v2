"""E2: C2 SELF-READ TEACHER for the 4B -> 9B read (feature matching).

D13's C2 interface (9B encoder -> 9B reader, ONE shared instance, identity B; outputs/D13_c2_reg/d13_best.pt) reaches
text parity. On TRAIN items we run that C2 read over the SAME reader rows (doc tail + question + answer) and take its
per-read-layer read-out y^C2_ℓ (what replaces the 9B's attention output at ℓ). The 4B -> 9B student's (mixed /
translated) read-out at ℓ is pulled toward it with a per-row relative MSE:

    c2 = mean_ℓ mean_rows ‖y_ℓ − y^C2_ℓ‖² / ‖y^C2_ℓ‖²

The C2 pass shares the reader's weights (the 9B is its own encoder, as in D13's C2): the C2 operator is the 9B's own
prefill K/V of the document prefix; its patches chain over the student's (all modes "off" outside their own pass).
Targets are cached on /scratch per item (fp16), keyed by the C2 checkpoint, the item key and the row ids.
"""

from __future__ import annotations

import hashlib
import os

import torch


class C2Teacher:
    def __init__(self, reader, tok, hub: str, ckpt: str, tail: int, max_tokens: int, cache_dir: str = "", docs=None):
        import sys
        from pathlib import Path
        sp = str(Path(__file__).resolve().parent.parent / "scripts")
        if sp not in sys.path:
            sys.path.insert(0, sp)
        from fit_d13 import D13Rt
        from xenc.backbone import input_device
        from xenc.d13_read import load_d13
        itf2, _ = load_d13(ckpt, reader, None)                 # identity B (C2)
        assert itf2.identity_B, "C2 teacher expects the identity-B 9B->9B interface"
        itf2 = itf2.to(input_device(reader)).requires_grad_(False)
        self.itf = itf2
        self.rt = D13Rt(reader, reader, tok, hub, itf2, tail, max_tokens, lru=1)
        self.rt.docs = docs if docs is not None else {}
        self.reader = reader
        self.layers = list(itf2.layers)
        tag = hashlib.md5(os.path.abspath(ckpt).encode()).hexdigest()[:10]
        self.cache_dir = os.path.join(cache_dir, tag) if cache_dir else ""
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.n_hit = self.n_miss = 0

    def _path(self, key: str, rows) -> str:
        h = hashlib.md5((key + "|" + ",".join(map(str, rows))).encode()).hexdigest()
        return os.path.join(self.cache_dir, h + ".pt")

    @torch.no_grad()
    def targets(self, did: str, rows: list, key: str) -> dict:
        """{reader layer: y^C2 [len(rows), d_r] fp32} on the reader device."""
        dev = self.rt.dev
        p = self._path(key, rows) if self.cache_dir else ""
        if p and os.path.exists(p):
            try:
                blob = torch.load(p, map_location="cpu", weights_only=False)
                self.n_hit += 1
                return {li: blob[li].to(dev).float() for li in self.layers}
            except Exception:
                pass
        self.n_miss += 1
        op, n_op, _ = self.rt.operator(did)
        R = self.rt.R
        R.load(op, n_op)
        R.mode, R.need_logits, R.rec = "e2e", False, {}
        try:
            self.reader(input_ids=torch.tensor([rows], device=dev), use_cache=False, logits_to_keep=1)
            y = {li: R.rec[li]["y"].detach() for li in self.layers}
        finally:
            R.mode, R.rec = "off", {}
            self.rt._ops.clear()            # keep no 9B operator / cache resident (L40S headroom for the student step)
        if p:
            tmp = p + f".tmp{os.getpid()}"
            torch.save({li: v.half().cpu() for li, v in y.items()}, tmp)
            os.replace(tmp, p)
        return {li: v.float() for li, v in y.items()}


def c2_loss(Rrec: dict, y2: dict) -> torch.Tensor:
    """mean over layers of the per-row relative MSE between the student read-out (rec['y_mix'] for E2, else rec['y'])
    and the C2 read-out."""
    terms = []
    for li, yt in y2.items():
        r = Rrec.get(li)
        if r is None:
            continue
        ys = r.get("y_mix") if r.get("y_mix") is not None else r["y"]
        terms.append((((ys - yt) ** 2).sum(-1) / (yt ** 2).sum(-1).clamp_min(1e-6)).mean())
    assert terms, "C2 feature matching: no student read-outs recorded"
    return torch.stack(terms).mean()


__all__ = ["C2Teacher", "c2_loss"]
