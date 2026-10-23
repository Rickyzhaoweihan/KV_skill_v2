"""Reward-learning primitives for compact operators (direction D4, "RL from document").

Minimal, dependency-free port of the policy-gradient pieces of the parent `kvskill` package
(NOT imported — copied with attribution) plus the paper's recipe details (arXiv 2608.05475,
App. A.3): group-normalized advantages, k3 KL to the bare reader, truncated importance ratio,
Frobenius (global-norm) gradient clipping.

    token_logprobs(logits, targets)            per-token log p(target)            [B, L]
    masked_mean(x, mask)                       per-row mean over valid tokens     [B]
    group_advantages(r, mode)                  'grpo' (r-mean)/std | 'drgrpo' r-mean | 'rloo'
    k3_kl(cur_lp, ref_lp)                      exp(ref-cur) - (ref-cur) - 1  (>= 0)
    truncated_is_weights(cur, behav, cap)      min(exp(cur - behav), cap), detached
    pg_loss(...)                               token-level TIS policy-gradient loss + β·k3 KL
    frobenius_clip_(params, max_norm)          global L2 (Frobenius) grad-norm clip, returns pre-clip norm
"""

from __future__ import annotations

from typing import Iterable

import torch


# --------------------------------------------------------------------------- log-probs
def token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log p(targets) from aligned logits.

    logits [B, L, V] where row t predicts targets[:, t]; targets [B, L] (int). Computed as
    logit_target - logsumexp (no full log-softmax kept). Ported from
    KV_skill/kvskill/utils.py::token_logprobs_from_logits (which additionally shifts by one; here
    the caller passes already-aligned rows)."""
    lg = logits.float()
    tgt = lg.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return tgt - torch.logsumexp(lg, dim=-1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-row mean of x over mask (length-normalized sequence log-prob).
    Port of kvskill/utils.py::masked_sequence_logprob(length_normalize=True)."""
    m = mask.to(x.dtype)
    return (x * m).sum(-1) / m.sum(-1).clamp_min(1.0)


# --------------------------------------------------------------------------- advantages
def group_advantages(rewards: torch.Tensor, mode: str = "grpo", eps: float = 1e-6) -> torch.Tensor:
    """Per-group advantage. Port of KV_skill/kvskill/algorithms/common.py::compute_advantages.

    'grpo'   : (r - mean) / (std + eps)      (group-normalized; the paper's choice)
    'drgrpo' / 'reinforce': r - mean
    'rloo'   : r - mean of the other G-1 samples
    A group with zero variance gives an all-zero advantage (no signal) in every mode."""
    r = rewards.float()
    g = r.shape[-1]
    if mode in ("reinforce", "drgrpo"):
        return r - r.mean(-1, keepdim=True)
    if mode == "rloo":
        if g == 1:
            return r - r.mean(-1, keepdim=True)
        loo = (r.sum(-1, keepdim=True) - r) / (g - 1)
        return r - loo
    if mode == "grpo":
        return (r - r.mean(-1, keepdim=True)) / (r.std(-1, unbiased=False, keepdim=True) + eps)
    raise ValueError(f"unknown advantage mode: {mode}")


# --------------------------------------------------------------------------- KL / IS
def k3_kl(cur_lp: torch.Tensor, ref_lp: torch.Tensor) -> torch.Tensor:
    """k3 KL estimator (Schulman) between the operator policy and the bare reader, per element.
    Port of KV_skill/kvskill/algorithms/common.py::kl_penalty. Always >= 0; 0 iff cur == ref."""
    d = ref_lp - cur_lp
    return torch.exp(d) - d - 1.0


def truncated_is_weights(cur_lp: torch.Tensor, behav_lp: torch.Tensor, cap: float = 2.0) -> torch.Tensor:
    """Truncated importance ratio min(pi_train / pi_behaviour, cap), DETACHED (a per-token weight on
    the score function; corrects the sampler/trainer mismatch, e.g. bf16 cached decode vs full fwd)."""
    return torch.exp(cur_lp.detach() - behav_lp.detach()).clamp(max=float(cap))


def pg_loss(cur_tok_lp: torch.Tensor, mask: torch.Tensor, adv: torch.Tensor,
            behav_tok_lp: torch.Tensor | None = None, ref_tok_lp: torch.Tensor | None = None,
            beta: float = 0.0, is_cap: float = 2.0) -> tuple[torch.Tensor, dict]:
    """Token-level policy-gradient loss for one group.

    cur_tok_lp [G, L] (grad), mask [G, L], adv [G] (detached), behav_tok_lp [G, L] (sampler
    log-probs, for the truncated IS weight; None => weight 1), ref_tok_lp [G, L] (bare reader,
    for the k3 KL; None or beta == 0 => no KL).

        loss = mean_g [ -A_g · mean_t w_gt · lp_gt  +  β · k3(mean_t lp_gt, mean_t ref_gt) ]

    Returns (scalar loss, metrics)."""
    adv = adv.detach().to(cur_tok_lp.device, cur_tok_lp.dtype)
    if behav_tok_lp is not None:
        w = truncated_is_weights(cur_tok_lp, behav_tok_lp.to(cur_tok_lp.device), is_cap)
    else:
        w = torch.ones_like(cur_tok_lp)
    pg_tok = masked_mean(w * cur_tok_lp, mask)                       # [G]
    loss_g = -adv * pg_tok
    met = {"is_w_mean": float(masked_mean(w, mask).mean()),
           "is_trunc_frac": float(masked_mean((w >= is_cap).to(w.dtype), mask).mean())}
    if ref_tok_lp is not None and beta > 0:
        kl = k3_kl(masked_mean(cur_tok_lp, mask), masked_mean(ref_tok_lp.to(cur_tok_lp.device), mask))
        loss_g = loss_g + beta * kl
        met["kl"] = float(kl.detach().mean())
    return loss_g.mean(), met


# --------------------------------------------------------------------------- clipping
def frobenius_clip_(params: Iterable[torch.Tensor], max_norm: float = 1.0) -> float:
    """Clip the global gradient norm (sqrt of the sum of squared Frobenius norms) to max_norm,
    in place. Returns the pre-clip norm."""
    ps = [p for p in params if p.grad is not None]
    if not ps:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(ps, max_norm))


__all__ = ["token_logprobs", "masked_mean", "group_advantages", "k3_kl", "truncated_is_weights",
           "pg_loss", "frobenius_clip_"]
