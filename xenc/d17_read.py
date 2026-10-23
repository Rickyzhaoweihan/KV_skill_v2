"""D17: BATCHED version of D13's reader-query read (xenc/d13_read.py), for reward learning (GRPO over G samples).

Same read as D13Interface.read (the reader's own state -> A (+MLP) -> the encoder's frozen q/k/v/gate projections ->
attention over [operator ; own earlier rows] -> o_proj -> B; the result REPLACES the reader's attention output at the
paired layers, gated own + g·(read − own)), but for a batch of Bt sequences that share ONE operator and the same row
positions (G samples of one prompt). Equal to D13 row by row (tests/test_d17.py). Uses a D13Interface unchanged, so
the trained file is a plain d13 file (save_d13) that scripts/eval_d13.py judges as is.

D17Reader is API-compatible with D13Reader for what D13Rt uses (load / op / n_op / past / rec / mode "off" |
"student" | "e2e" / need_logits / itf / detach), so D13Rt.gen (greedy, batch 1) and D13Rt.attn_terms work with it.
"""

from __future__ import annotations

import types

import torch

from xenc.d13_read import D13Interface


def read_batched(itf: D13Interface, li: int, x: torch.Tensor, Kop, Vop, pos0: int, past=None,
                 need_logits: bool = False) -> dict:
    """x [Bt,T,d_r] fp32; Kop/Vop [Hkv,N,hd] (shared); past = (Kp, Vp) [Bt,Hkv,P,hd] own earlier rows.
    -> {"y": [Bt,T,d_r], "o": [Bt,T,d_e], "logits": [Bt,Hkv,G,T,N+P+T] (if asked), "past"}."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    att = itf.att[str(li)]
    hd = att.head_dim
    xe = itf.map_x(li, x)
    Bt, T = xe.shape[:2]
    q, gate = torch.chunk(att.q_proj(xe).view(Bt, T, -1, hd * 2), 2, dim=-1)
    gate = gate.reshape(Bt, T, -1)
    q = att.q_norm(q).transpose(1, 2)                                    # [Bt,Hq,T,hd]
    k = att.k_norm(att.k_proj(xe).view(Bt, T, -1, hd)).transpose(1, 2)   # [Bt,Hkv,T,hd]
    v = att.v_proj(xe).view(Bt, T, -1, hd).transpose(1, 2)
    P = 0 if past is None else past[0].shape[2]
    pos = torch.arange(pos0 + P, pos0 + P + T, device=xe.device).view(1, 1, T).expand(3, 1, T)
    cos, sin = itf.rot(xe[:1], pos)                                      # same positions for every row
    qr, kr = apply_rotary_pos_emb(q, k, cos.float(), sin.float())
    Kown = kr if past is None else torch.cat([past[0], kr], 2)
    Vown = v if past is None else torch.cat([past[1], v], 2)
    Hq, Hkv = qr.shape[1], Kop.shape[0]
    G = Hq // Hkv
    qg = qr.reshape(Bt, Hkv, G, T, hd)
    s_op = torch.einsum("bkgtd,kcd->bkgtc", qg, Kop.float()) * att.scaling          # [Bt,Hkv,G,T,N]
    s_own = torch.einsum("bkgtd,bkcd->bkgtc", qg, Kown) * att.scaling               # [Bt,Hkv,G,T,P+T]
    C = Kown.shape[2]
    col = torch.arange(C, device=x.device).view(1, C)
    row = torch.arange(T, device=x.device).view(T, 1)
    s_own = s_own.masked_fill((col > (C - T) + row).view(1, 1, 1, T, C), float("-inf"))
    s = torch.cat([s_op, s_own], -1)
    p = torch.softmax(s, -1)
    N = Kop.shape[1]
    o = (torch.einsum("bkgtc,kcd->bkgtd", p[..., :N], Vop.float())
         + torch.einsum("bkgtc,bkcd->bkgtd", p[..., N:], Vown))
    o = o.reshape(Bt, Hq, T, hd).transpose(1, 2).reshape(Bt, T, Hq * hd)
    o = att.o_proj(o * torch.sigmoid(gate))
    out = {"y": itf.Bmap(li, o), "o": o, "past": (Kown.detach(), Vown.detach())}
    if need_logits:
        out["logits"] = s
    return out


class D17Reader:
    """Patches the reader's attention at itf.layers (batched). mode:
        "off"      stock reader
        "student"  no-grad batched read replaces the output; own rows accumulate in R.past (sampling / greedy)
        "e2e"      read with grad replaces the output (policy log-probs; with need_logits the per-layer logits are
                   stored in R.rec[li] — squeezed to D13's [Hkv,G,T,C] / [T,d] shapes when Bt == 1)
    R.itf may be swapped (e.g. to a frozen reference interface) between forwards."""

    def __init__(self, model, itf: D13Interface):
        from deltaskill.backbone_utils import decoder_layers
        self.model, self.itf = model, itf
        L = decoder_layers(model)
        self.mods = {li: L[li].self_attn for li in itf.layers}
        self.mode = "off"
        self.op: dict = {}
        self.n_op = 0
        self.past: dict = {}
        self.rec: dict = {}
        self.need_logits = False
        self.mem_safe = False
        self.row_chunk = 0               # >0: the no-grad "student" read runs over batch rows in chunks of this size            # True: torch.utils.checkpoint the grad-mode read (identical gradients, less memory)
        self._orig = {}
        for li, m in self.mods.items():
            self._orig[li] = m.forward
            m.forward = types.MethodType(_d17_forward(self, li), m)

    def load(self, op: dict, n_op: int):
        self.op, self.n_op, self.past = op, int(n_op), {}

    def detach(self):
        for li, f in self._orig.items():
            self.mods[li].forward = f
        self._orig.clear()


def _d17_forward(R: D17Reader, li: int):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        out, w = type(self).forward(self, hidden_states, position_embeddings, attention_mask,
                                    past_key_values=past_key_values, **kwargs)
        if R.mode == "off":
            return out, w
        itf = R.itf
        K, V = R.op[itf.layer_map[li]]
        x = hidden_states.float()
        if R.mode == "student":
            with torch.no_grad():
                pst = R.past.get(li)
                Bt = x.shape[0]
                if R.row_chunk and Bt > R.row_chunk:
                    parts = [read_batched(itf, li, x[b:b + R.row_chunk], K, V, R.n_op,
                                          past=None if pst is None else (pst[0][b:b + R.row_chunk],
                                                                          pst[1][b:b + R.row_chunk]))
                             for b in range(0, Bt, R.row_chunk)]
                    res = {"y": torch.cat([q["y"] for q in parts], 0),
                           "past": (torch.cat([q["past"][0] for q in parts], 0),
                                    torch.cat([q["past"][1] for q in parts], 0))}
                else:
                    res = read_batched(itf, li, x, K, V, R.n_op, past=pst)
            R.past[li] = res["past"]
        elif R.mem_safe and not R.need_logits and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            n_op = R.n_op
            res = {"y": checkpoint(lambda x_: read_batched(itf, li, x_, K, V, n_op)["y"], x, use_reentrant=False)}
        else:
            res = read_batched(itf, li, x, K, V, R.n_op, need_logits=R.need_logits)
            if R.need_logits:
                R.rec[li] = ({"y": res["y"][0], "o": res["o"][0], "logits": res["logits"][0]}
                             if x.shape[0] == 1 else res)
        g = itf.gate(li)
        y = out.float() + g * (res["y"] - out.float())
        return y.to(out.dtype), w

    return forward


__all__ = ["read_batched", "D17Reader"]
