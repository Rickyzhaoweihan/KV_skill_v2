"""CAD: context-aware (contrastive) decoding for the kv read, inference only (agent CAD, 2026-10-23).

    logits_cad = (1 + a) * logits_main - a * logits_ref            (Shi et al. 2023, "context-aware decoding")

Borrowed from Hub Memory's conflict readout, /nfs/turbo/umms-drjieliu/usr/xiangzhg/kvshill/universal/hm_A_cad.py:32-43
(`s = [(1 + a_) * x - a_ * y for x, y in zip(r["lp"], C[k]["lp"])]`, x = option-letter logp WITH the document,
y = closed book), there applied only to forced-choice letter log-probs. Here it is applied to
  * forced choice: the option-letter logits after "<answer>" (fc_letter_logp + cad_pick), and
  * per-step greedy generation: every decoding step contrasts the two streams' next-token logits (cad_decode).
main = the kv read (own operator + doc tail + question; the reader never sees the document text) or, as a reference
       only, the text read (document in context);
ref  = closed_tail (the SAME reader prompt without the operator) or closed_book (chat(q_prompt), no document at all).
Log-softmax constants cancel in the argmax, so logits vs log-probs give the same pick (as in hm_A_cad.py).
a = 0 never consults the reference: the pick is the main stream's own argmax, bit-for-bit the plain decode.

A Stream wraps one reader conditioning on the shared, patched reader (fit_d13 / fit_d12p D13Rt.R): it owns its HF
cache and its own D13 read state (R.past), restoring them around every forward, so a kv stream and a reference stream
can be interleaved token by token without interfering.
"""

from __future__ import annotations

import copy
import time

import torch


class Stream:
    """One reader conditioning. kind: "kv" (D13 read of op), "plain" (stock reader). prefill() once, then fork() per
    decode (deep copy of the HF cache + read state) and step(token)."""

    def __init__(self, rt, ids, op=None, n_op: int = 0, name: str = ""):
        self.rt, self.ids, self.op, self.n_op, self.name = rt, [int(x) for x in ids], op, int(n_op), name
        self.cache, self.past, self.last, self.n_fwd = None, {}, None, 0

    @torch.no_grad()
    def _fwd(self, ids) -> torch.Tensor:
        rt, R = self.rt, self.rt.R
        if self.op is not None:
            R.op, R.n_op, R.past = self.op, self.n_op, self.past
            R.mode = "student"
        try:
            out = rt.reader(input_ids=torch.tensor([list(ids)], device=rt.dev), past_key_values=self.cache,
                            use_cache=True, logits_to_keep=1)
        finally:
            if self.op is not None:
                self.past = R.past
            R.mode, R.past = "off", {}
        self.cache = out.past_key_values
        self.n_fwd += 1
        self.last = out.logits[0, -1]
        return self.last

    def prefill(self) -> torch.Tensor:
        assert self.cache is None, "prefill once"
        return self._fwd(self.ids)

    def step(self, tok_id: int) -> torch.Tensor:
        return self._fwd([int(tok_id)])

    def extend(self, ids) -> torch.Tensor:
        return self._fwd(ids)

    def fork(self) -> "Stream":
        s = Stream(self.rt, self.ids, self.op, self.n_op, self.name)
        s.cache, s.past, s.last = copy.deepcopy(self.cache), dict(self.past), self.last
        return s


def combine(lm: torch.Tensor, lr: torch.Tensor | None, alpha: float, plaus: float = 0.0) -> torch.Tensor:
    """(1+a)·lm − a·lr in fp32; a == 0 returns lm itself (exactly the plain argmax). plaus > 0: adaptive plausibility
    (Li et al. 2023) — tokens with p_main < plaus·max p_main are excluded (default 0 = off, pure CAD)."""
    if alpha == 0 or lr is None:
        s = lm
    else:
        s = (1.0 + alpha) * lm.float() - alpha * lr.float().to(lm.device)
    if plaus and plaus > 0:
        p = torch.softmax(lm.float(), -1)
        s = s.float().masked_fill(p < plaus * p.max(), float("-inf"))
    return s


@torch.no_grad()
def cad_decode(main: Stream, ref: Stream | None, alpha: float, budget: int, tok, hub: str, stops: set,
               stop: str = "</answer>", plaus: float = 0.0, force_ref: bool = False) -> dict:
    """Greedy decode from forks of two PREFILLED streams. Same stop rule / ordering as fit_d13.D13Rt.gen.
    force_ref: also advance the reference at a = 0 (the interleaving check: must equal the plain decode)."""
    from xenc.backbone import decode
    t0 = time.time()
    m = main.fork()
    use_ref = ref is not None and (alpha != 0 or force_ref)
    r = ref.fork() if use_ref else None
    lm, lr = m.last, (r.last if r is not None else None)
    new: list[int] = []
    for _ in range(budget):
        nxt = int(combine(lm, lr, alpha, plaus).argmax())
        new.append(nxt)
        if nxt in stops or stop in tok.decode(new[-12:], skip_special_tokens=True):
            break
        lm = m.step(nxt)
        if r is not None:
            lr = r.step(nxt)
    txt = decode(tok, hub, new)
    if stop in txt:
        txt = txt.split(stop, 1)[0] + stop
    return {"gen": txt, "gen_tokens": len(new), "t_loop": time.time() - t0}


@torch.no_grad()
def fc_letter_logp(stream: Stream, ans_prefix_ids, letter_ids) -> list[float]:
    """Forced choice: log-softmax over the option letters of the next-token logits after "<answer>", continuing a
    fork of the prefilled stream (the prefill itself is left untouched)."""
    s = stream.fork()
    lg = s.extend(ans_prefix_ids)
    return torch.log_softmax(lg.float()[list(letter_ids)], -1).tolist()


def cad_pick(lp_main, lp_ref, alpha: float) -> int:
    """hm_A_cad.py:41-42 on one item: argmax_i (1+a)·lp_main[i] − a·lp_ref[i] (first index on ties)."""
    s = [(1 + alpha) * x - alpha * y for x, y in zip(lp_main, lp_ref)] if alpha else list(lp_main)
    return max(range(len(s)), key=lambda i: s[i])
