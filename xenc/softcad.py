"""SOFTCONF (agent SOFTCONF, 2026-10-25): graded-conflict arbitration knobs on top of xenc/cad.py (which is untouched).

Per-step adaptive CAD for generation (the forced-choice versions are CPU recomputes in scripts/softconf_table.py):
    s_t = (1 + a_t)·logits_main − a_t·logits_ref,       a_t = alpha_rule(p_main, p_ref)
  mode  const     a_t = a                                                  (= xenc.cad.cad_decode)
        jsd       a_t = max(floor, a·JSD2(p_main, p_ref))                  AdaCAD (Wang et al., NAACL'25, arXiv 2409.07394)
        prior     a_t = a·1[max p_ref >= tau]                              per-token prior-confidence gate
        priorjsd  a_t = a·JSD2·1[max p_ref >= tau]
        entropy   a_t = a·H(p_ref) / (H(p_ref) + H(p_main))                ACD-style (Kim et al. 2024, arXiv 2408.01084)
  plaus > 0: adaptive plausibility cut (Li et al. 2023) on p_main — the guard against OVERSHOOT (CAD boosting a token the
  read barely supports just because the prior gives it ~0).
JSD2 = Jensen-Shannon divergence in bits (in [0, 1]); H in nats.

Read-gain scaling (a single-stream, internal analogue of CAD against closed_tail): the D13 write-back is
    y = own + g·(read − own)        (xenc/d13_read.py:419, xenc/d12p_read.py:350)
so multiplying g by s > 1 extrapolates the read away from the reader's own attention output (s = 1 + alpha), s < 1 damps
it. scale_gain(itf, s) does that in place.
"""

from __future__ import annotations

import math
import time

import torch

MODES = ("const", "jsd", "prior", "priorjsd", "entropy")


def jsd2(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon divergence in bits between two probability vectors (fp32)."""
    p, q = p.float(), q.float()
    m = 0.5 * (p + q)

    def kl(a, b):
        a_ = a.clamp_min(1e-30)
        return float((a * (a_.log() - b.clamp_min(1e-30).log())).sum())
    return max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m)) / math.log(2)


def entropy(p: torch.Tensor) -> float:
    p = p.float()
    return float(-(p * p.clamp_min(1e-30).log()).sum())


def alpha_rule(lm: torch.Tensor, lr: torch.Tensor, a: float, mode: str = "const", tau: float = 0.0,
               floor: float = 0.0) -> float:
    """Per-step alpha from the two next-token logit vectors."""
    if mode == "const" or a == 0:
        return float(a)
    pm, pr = torch.softmax(lm.float(), -1), torch.softmax(lr.float().to(lm.device), -1)
    if mode == "jsd":
        return max(float(floor), a * jsd2(pm, pr))
    if mode == "prior":
        return float(a) if float(pr.max()) >= tau else 0.0
    if mode == "priorjsd":
        return a * jsd2(pm, pr) if float(pr.max()) >= tau else 0.0
    if mode == "entropy":
        hr, hm = entropy(pr), entropy(pm)
        return a * hr / max(hr + hm, 1e-12)
    raise ValueError(mode)


@torch.no_grad()
def adaptive_cad_decode(main, ref, alpha: float, budget: int, tok, hub: str, stops: set, stop: str = "</answer>",
                        plaus: float = 0.0, force_ref: bool = False, mode: str = "const", tau: float = 0.0,
                        floor: float = 0.0) -> dict:
    """xenc.cad.cad_decode with a per-step alpha (alpha_rule). mode == "const" is token-for-token cad_decode.
    Also returns the per-step alphas (alpha_trace)."""
    from xenc.backbone import decode
    from xenc.cad import combine
    t0 = time.time()
    m = main.fork()
    use_ref = ref is not None and (alpha != 0 or force_ref)
    r = ref.fork() if use_ref else None
    lm, lr = m.last, (r.last if r is not None else None)
    new, trace = [], []
    for _ in range(budget):
        a_t = alpha_rule(lm, lr, alpha, mode, tau, floor) if lr is not None else 0.0
        trace.append(round(a_t, 4))
        nxt = int(combine(lm, lr, a_t, plaus).argmax())
        new.append(nxt)
        if nxt in stops or stop in tok.decode(new[-12:], skip_special_tokens=True):
            break
        lm = m.step(nxt)
        if r is not None:
            lr = r.step(nxt)
    txt = decode(tok, hub, new)
    if stop in txt:
        txt = txt.split(stop, 1)[0] + stop
    return {"gen": txt, "gen_tokens": len(new), "t_loop": time.time() - t0, "alpha_trace": trace}


@torch.no_grad()
def scale_gain(itf, s: float):
    """Multiply the interface's per-layer read gate g in place (s = 1: no-op). Returns the new gates."""
    if s != 1.0:
        itf.g.mul_(float(s))
    return [round(float(x), 4) for x in itf.g.detach().float().cpu()]
