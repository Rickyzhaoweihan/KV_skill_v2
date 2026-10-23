"""ADDR / HID: distil the read layers' WRITE-BACK toward the 9B reader's own TEXT-mode attention output. Default off.

The D13 read REPLACES the reader's attention output at the read layers (3/7/11/15/19). The stage-2 objective supervises
only the final logits (KL to the text teacher + CE) and, through the distillation regularizer, the 4B ENCODER's own
attention output mapped by B (a translation-limited target: held-out R^2 of a linear map from the 4B attention output
to the 9B's own text-mode attention output is only .54 at layer 19, outputs/E2_SUMMARY.md). The quantity the read is
meant to stand in for is the 9B's attention output at the same layer and row when it reads the document as text. That
tensor is available for free in the text-teacher pass the KL term already runs (rows t of the kv pass == the last T
positions of the text pass when the prefix is untruncated), so we add
    hid = mean_{read layers l} mean_{rows t} || y_kv(l, t) - a_text(l, t) ||^2 / || a_text(l, t) ||^2
where y_kv is the gated write-back (own + g (read - own)) that actually flows downstream. Nothing else changes.

AttnOutCap records the attention-module output rows with forward hooks (chained after the D13Reader patch), detached
for the teacher pass and with the graph for the student pass.
"""

from __future__ import annotations

import torch


class AttnOutCap:
    """Forward hooks on the reader's self_attn modules at `layers`. While self.T is set, a forward records
    out[0][0, -T:] per layer into self.rec (detached and fp32 when self.detach, else with the graph)."""

    def __init__(self, model, layers):
        from deltaskill.backbone_utils import decoder_layers
        L = decoder_layers(model)
        self.layers = sorted(int(x) for x in layers)
        self.T: int | None = None
        self.detach = True
        self.rec: dict = {}
        self._h = [L[li].self_attn.register_forward_hook(self._hook(li)) for li in self.layers]

    def _hook(self, li):
        def h(mod, args, out):
            if self.T is None:
                return None
            o = out[0] if isinstance(out, tuple) else out
            r = o[0, -self.T:]
            self.rec[li] = r.detach().float().clone() if self.detach else r.float()
            return None
        return h

    def remove(self):
        for h in self._h:
            h.remove()
        self._h = []


def common_suffix(a: list, b: list) -> int:
    """Length of the longest common suffix of two id lists (at most len(b))."""
    k = 0
    while k < len(a) and k < len(b) and a[-1 - k] == b[-1 - k]:
        k += 1
    return k


def hid_rows(T: int, tail: int, m: int, mode: str) -> list:
    """Row indices of the kv rows used by the term: all | qa (question + answer rows, no doc tail) | ans (answer rows)."""
    if mode == "all":
        return list(range(T))
    if mode == "qa":
        return list(range(min(tail, T), T))
    if mode == "ans":
        return list(range(max(T - (m + 1), 0), T))
    raise ValueError(mode)


def hid_loss(ys: dict, yt: dict, rows: list) -> torch.Tensor:
    """mean over layers (present in both) of the row-mean relative squared error."""
    terms = []
    for li in sorted(set(ys) & set(yt)):
        s, t = ys[li][rows], yt[li][rows].to(ys[li].device)
        terms.append((((s - t) ** 2).sum(-1) / (t ** 2).sum(-1).clamp_min(1e-6)).mean())
    assert terms, "HID: no layer recorded in both passes"
    return torch.stack(terms).mean()


__all__ = ["AttnOutCap", "common_suffix", "hid_rows", "hid_loss"]
