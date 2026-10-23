"""L1: LENGTH-AWARE read forms for the D13 paper-form read (xenc/d13_read.py). Default off; the operator (U, W), the
read layers, A / MLP and the write-back B are unchanged. Only the softmax over the operator columns changes.

Why (outputs/L1_SUMMARY.md): kv − text grows with document length (LongHealth parity, LooGLE −.15, LooGLE-long −.18,
CLIPPER −.14). The query passes the encoder's q_norm, so |q| and the max logit are bounded; a bounded-norm query over N
columns spreads its mass as N grows (softmax dilution).

Forms (all at the read layers, per query head h, per reader row t; N = number of operator columns; s_tj = the D13
logits incl. the base scale; own-row columns are kept in every form):

  logn   scalable softmax:  s'_tj = s_tj · m_h(N),  m_h(N) = 1 + σ_h · ln(N / nref)    σ trainable, init 0 -> base
         (length-dependent temperature only; a length-independent per-head temperature is agent R2's scope)
  chunk  chunk-local softmax + learned merge: the columns are cut into chunks of `chunk` operator columns (the own rows
         form one more chunk); each chunk is read with its own softmax, and the chunk reads are merged with weights
         w_c ∝ exp(β_h · lse_c), lse_c = logsumexp of the row's logits in chunk c. β = 1 is EXACTLY the flat softmax;
         β > 1 concentrates the read on the best chunks (soft two-stage read). Implemented as the exact identity
         softmax(s + (β_h − 1) · lse_c(j)).  β_h = 1 + c_h, c trainable, init 0 -> base.
  topk   hard top-k sparse read: per head and row keep the k highest operator logits, renormalise (k >= N: no-op).
  hier   two-stage hierarchical read: blocks of `block` operator columns with a summary score = the mean of the row's
         logits in the block (== q · mean-K of the block, i.e. a mean-key summary); keep the top `nblk` blocks per head
         and row, then the full softmax read inside them (nblk · block >= N: no-op). Mask implementation (the pilot
         measures accuracy, not speed).

The logits returned for the distillation term are the RAW D13 logits (causal mask only), as in E4's registration: the
stage-1 attention distillation constrains the query DIRECTION against the encoder teacher at the teacher's own
temperature, and the length-aware sharpening / selection is trained by the task loss (KL to text + CE).
N is the operator length only, constant over a document, so incremental decode == the full teacher-forced pass.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from xenc.d13_read import D13Interface

SELS = ("", "topk", "hier")


def parse_l1(logn: int = 0, nref: int = 16000, chunk: int = 0, sel: str = "", k: int = 128, block: int = 256,
             nblk: int = 32) -> dict:
    """CLI -> l1 config ({} = every form off)."""
    sel = str(sel or "")
    assert sel in SELS, sel
    if not (int(logn) or int(chunk) or sel):
        return {}
    assert int(nref) > 1 and int(k) > 0 and int(block) > 0 and int(nblk) > 0
    assert int(chunk) >= 0
    return {"logn": int(bool(int(logn))), "nref": int(nref), "chunk": int(chunk), "sel": sel, "k": int(k),
            "block": int(block), "nblk": int(nblk)}


def _chunk_lse(s_op: torch.Tensor, size: int) -> torch.Tensor:
    """s_op [..., N] -> logsumexp per chunk of `size` columns [..., ceil(N/size)]."""
    N = s_op.shape[-1]
    full = (N // size) * size
    parts = []
    if full:
        parts.append(s_op[..., :full].reshape(*s_op.shape[:-1], N // size, size).logsumexp(-1))
    if full < N:
        parts.append(s_op[..., full:].logsumexp(-1, keepdim=True))
    return torch.cat(parts, -1)


def _expand(per_chunk: torch.Tensor, size: int, N: int) -> torch.Tensor:
    """[..., nc] -> [..., N] (each chunk value repeated over its columns)."""
    return per_chunk.repeat_interleave(size, dim=-1)[..., :N]


@torch.no_grad()
def keep_mask(s_op: torch.Tensor, cfg: dict) -> torch.Tensor | None:
    """bool [..., N] operator columns kept by the sparse selection (None = keep all)."""
    N = s_op.shape[-1]
    if cfg.get("sel") == "topk":
        k = int(cfg["k"])
        if k >= N:
            return None
        thr = s_op.topk(k, dim=-1).values[..., -1:]
        return s_op >= thr
    if cfg.get("sel") == "hier":
        B, nb = int(cfg["block"]), int(cfg["nblk"])
        nblocks = -(-N // B)
        if nb >= nblocks:
            return None
        full = (N // B) * B
        parts = []
        if full:
            parts.append(s_op[..., :full].reshape(*s_op.shape[:-1], N // B, B).mean(-1))
        if full < N:
            parts.append(s_op[..., full:].mean(-1, keepdim=True))
        score = torch.cat(parts, -1)                                  # [..., nblocks] = q · mean-K per block
        idx = score.topk(nb, dim=-1).indices
        kb = torch.zeros_like(score, dtype=torch.bool).scatter_(-1, idx, True)
        return _expand(kb, B, N)
    return None


class L1Interface(D13Interface):
    """D13Interface + the length-aware read forms above (trainable σ / c in self.L1)."""

    def __init__(self, enc, layer_map: dict, d_r: int, d_e: int, B: dict | None = None, mlp_hidden: int = 0,
                 l1: dict | None = None):
        super().__init__(enc, layer_map, d_r, d_e, B=B, mlp_hidden=mlp_hidden)
        self.l1 = dict(l1 or {})
        assert self.l1, "L1Interface needs an l1 config (parse_l1)"
        a0 = self.att[str(self.layers[0])]
        self.l1_hq = a0.q_proj.out_features // (2 * a0.head_dim)
        self.l1_hkv = a0.k_proj.out_features // a0.head_dim
        P = {}
        for li in self.layers:
            if self.l1["logn"]:
                P[f"s{li}"] = nn.Parameter(torch.zeros(self.l1_hq))
            if self.l1["chunk"]:
                P[f"c{li}"] = nn.Parameter(torch.zeros(self.l1_hq))
        self.L1 = nn.ParameterDict(P)
        self.l1_on = True                    # False: the same checkpoint read with the plain D13 softmax (ablation)

    def config(self) -> dict:
        c = super().config()
        c["l1"] = dict(self.l1)
        return c

    def l1_params(self) -> list:
        return list(self.L1.parameters())

    def l1_mult(self, li, N: int):
        """[Hq] logit multiplier m_h(N) (None when logn is off)."""
        if not self.l1["logn"]:
            return None
        return (1.0 + self.L1[f"s{li}"] * math.log(max(N, 1) / self.l1["nref"])).clamp_min(0.05)

    @torch.no_grad()
    def l1_state(self) -> dict:
        out = {}
        for li in self.layers:
            d = {}
            if self.l1["logn"]:
                s = self.L1[f"s{li}"]
                d["sig"] = [round(float(s.min()), 4), round(float(s.mean()), 4), round(float(s.max()), 4)]
                d["m@4k/16k/64k"] = [round(float(self.l1_mult(li, n).mean()), 3) for n in (4000, 16000, 64000)]
            if self.l1["chunk"]:
                c = self.L1[f"c{li}"]
                d["beta"] = [round(1 + float(c.min()), 4), round(1 + float(c.mean()), 4), round(1 + float(c.max()), 4)]
            out[str(li)] = d
        return out

    def read(self, li, x, Kop, Vop, pos0: int, past=None, need_logits: bool = False):
        """Same signature / outputs as D13Interface.read; only the softmax over the columns differs."""
        if not self.l1_on:
            return super().read(li, x, Kop, Vop, pos0, past=past, need_logits=need_logits)
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        att = self.att[str(li)]
        hd = att.head_dim
        xe = self.map_x(li, x)
        T = xe.shape[0]
        q, gate = torch.chunk(att.q_proj(xe).view(T, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(T, -1)
        q = att.q_norm(q).transpose(0, 1)
        k = att.k_norm(att.k_proj(xe).view(T, -1, hd)).transpose(0, 1)
        v = att.v_proj(xe).view(T, -1, hd).transpose(0, 1)
        P = 0 if past is None else past[0].shape[1]
        pos = torch.arange(pos0 + P, pos0 + P + T, device=xe.device).view(1, 1, T).expand(3, 1, T)
        cos, sin = self.rot(xe.unsqueeze(0), pos)
        qr, kr = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos.float(), sin.float())
        qr, kr = qr[0], kr[0]
        Kown = kr if past is None else torch.cat([past[0], kr], 1)
        Vown = v if past is None else torch.cat([past[1], v], 1)
        out = self._attend_l1(li, qr, Kop.float(), Vop.float(), Kown, Vown,
                              att.scaling * self.read_scale_mult(Kop.shape[1]), need_logits)
        o = att.o_proj(out["o"] * torch.sigmoid(gate))
        return {"y": self.Bmap(li, o), "o": o, "logits": out.get("logits"),
                "past": (Kown.detach(), Vown.detach())}

    def _attend_l1(self, li, q, Kop, Vop, Kown, Vown, scale, need_logits=False):
        Hq, T, hd = q.shape
        Hkv, N = Kop.shape[0], Kop.shape[1]
        G = Hq // Hkv
        K = torch.cat([Kop, Kown], 1)
        V = torch.cat([Vop, Vown], 1)
        C = K.shape[1]
        qg = q.view(Hkv, G, T, hd)
        first_own = C - T
        col = torch.arange(C, device=q.device).view(1, C)
        row = torch.arange(T, device=q.device).view(T, 1)
        causal = (col > first_own + row).view(1, 1, T, C)
        m = self.l1_mult(li, N)
        s = s0 = None
        if m is None or need_logits:
            s0 = (torch.einsum("kgtd,kcd->kgtc", qg, K) * scale).masked_fill(causal, float("-inf"))   # raw (returned)
            s = s0
        if m is not None:
            # the multiplier scales the QUERY (no tensor x -inf products: their 0 * inf gradients would be NaN)
            s = (torch.einsum("kgtd,kcd->kgtc", qg * m.view(Hkv, G, 1, 1), K) * scale).masked_fill(causal, float("-inf"))
        if self.l1["chunk"]:
            cs = int(self.l1["chunk"])
            beta1 = self.L1[f"c{li}"].view(Hkv, G, 1, 1)                     # β − 1
            lse_op = _chunk_lse(s[..., :N], cs)                                 # [Hkv,G,T,nc]
            lse_own = s[..., N:].logsumexp(-1, keepdim=True)                    # own rows = one more chunk
            bias = torch.cat([_expand(lse_op, cs, N), lse_own.expand(*lse_own.shape[:-1], C - N)], -1)
            s = s + beta1 * bias
        if self.l1["sel"]:
            keep = keep_mask(s[..., :N].detach(), self.l1)
            if keep is not None:                                                # out-of-place: s0 stays raw
                s = torch.cat([s[..., :N].masked_fill(~keep, float("-inf")), s[..., N:]], -1)
        p = torch.softmax(s, -1)
        o = torch.einsum("kgtc,kcd->kgtd", p, V).reshape(Hq, T, hd).transpose(0, 1).reshape(T, Hq * hd)
        res = {"o": o}
        if need_logits:
            res["logits"] = s0
        return res


def upgrade_l1(itf: D13Interface, enc, l1: dict) -> L1Interface:
    """A loaded plain D13 base -> L1Interface with identical A / MLP / B / bias / gates and σ = c = 0."""
    assert type(itf) is D13Interface, f"L1 read forms on a {type(itf).__name__} init are not supported"
    assert not itf.delta_rank
    B = None if itf.identity_B else {li: itf.B[str(li)].detach().cpu() for li in itf.layers}
    new = L1Interface(enc, itf.layer_map, itf.d_r, itf.d_e, B=B, mlp_hidden=itf.mlp_hidden, l1=l1)
    sd = {k: v for k, v in itf.state_dict().items() if not k.startswith("D.")}
    missing, unexpected = new.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("L1.") for k in missing), missing
    for k in ("read_tau", "read_logn_ref"):
        if hasattr(itf, k):
            setattr(new, k, getattr(itf, k))
    return new.to(itf.g.device)


__all__ = ["L1Interface", "parse_l1", "upgrade_l1", "keep_mask"]
