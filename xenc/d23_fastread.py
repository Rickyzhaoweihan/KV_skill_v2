"""D23 (efficiency benchmark): a BATCHED, low-precision implementation of the D13 paper-form read.

Same math as `xenc.d13_read.D13Interface.read` in "student" mode (the reader's own question state queries the
encoder's fixed operator; own rows are appended causally; B write-back; per-layer gate mixes the read into the
reader's own attention output). Differences are implementation only:
  * batched: x [B,T,d_r]; the operator is either shared by the batch ([1,Hkv,N,hd], broadcast without a copy) or one
    per request ([B,Hkv,N,hd]: B different documents);
  * the attention over the operator runs in `dtype` (bf16 on GPU) with fp32 softmax statistics, and never casts or
    concatenates the operator (the as-implemented path casts it to fp32 and concatenates it with the own rows on
    every forward, i.e. ~3 extra operator-sized copies per layer per decoded token);
  * per-request registered adapters (D20 `dA` low-rank delta on A + per-layer gate) as stacked tensors, so a batch
    can mix documents (multi-LoRA style); without them the interface's own (merged / single-document) deltas apply.
A/M/q/k/v/o projections stay in fp32 on the T new rows (cheap); only the O(N) part changes precision.
Equivalence with D13Interface.read: tests/test_d23_fastread.py (fp32, exact to 1e-5) and the GPU validation phase of
scripts/bench_d23.py (bf16: greedy answers vs the as-implemented read and vs the judged D20 rows).
"""

from __future__ import annotations

import types

import torch

import xenc  # noqa: F401


def attend_two_part(q, Kop, Vop, Kown, Vown, scale: float, P: int, row_budget: int = 1 << 27):
    """q [B,Hq,T,hd]; Kop/Vop [Bo,Hkv,N,hd] with Bo in {1, B}; Kown/Vown [B,Hkv,P+T,hd] (last T = these rows).
    Softmax over [operator ; own rows (causal)] without concatenating or casting the operator. -> [B,T,Hq*hd]."""
    B, Hq, T, hd = q.shape
    Hkv = Kown.shape[1]
    G = Hq // Hkv
    Bo, N = Kop.shape[0], Kop.shape[2]
    dt = Kop.dtype
    qg = q.to(dt).reshape(B, Hkv, G * T, hd)                 # row r = g*T + t
    Kw, Vw = Kown.to(dt), Vown.to(dt)
    C = Kw.shape[2]
    R = max(1, min(G * T, row_budget // max(1, B * Hkv * (N + C))))
    out = torch.empty(B, Hkv, G * T, hd, dtype=dt, device=q.device)
    col = torch.arange(C, device=q.device)
    for r0 in range(0, G * T, R):
        qc = qg[:, :, r0:r0 + R]
        Rc = qc.shape[2]
        if Bo == 1 and B > 1:                                 # shared operator: fold the batch into rows (no expand copy)
            qf = qc.transpose(0, 1).reshape(1, Hkv, B * Rc, hd)
            so = torch.matmul(qf, Kop.transpose(-1, -2)).float().mul_(scale)
            so = so.view(Hkv, B, Rc, N).transpose(0, 1)       # [B,Hkv,Rc,N] (view)
        else:
            so = torch.matmul(qc, Kop.transpose(-1, -2)).float().mul_(scale)
        sw = torch.matmul(qc, Kw.transpose(-1, -2)).float().mul_(scale)
        t = torch.arange(r0, r0 + Rc, device=q.device) % T
        sw = sw.masked_fill((col.view(1, C) > (P + t).view(Rc, 1)).view(1, 1, Rc, C), float("-inf"))
        m = torch.maximum(so.amax(-1, keepdim=True), sw.amax(-1, keepdim=True))
        po = (so - m).exp_()
        pw = (sw - m).exp_()
        den = po.sum(-1, keepdim=True) + pw.sum(-1, keepdim=True)
        if Bo == 1 and B > 1:
            pof = po.transpose(0, 1).reshape(1, Hkv, B * Rc, N).to(dt)
            oo = torch.matmul(pof, Vop).view(Hkv, B, Rc, hd).transpose(0, 1)
        else:
            oo = torch.matmul(po.to(dt), Vop)
        oc = oo.float() + torch.matmul(pw.to(dt), Vw).float()
        out[:, :, r0:r0 + Rc] = (oc / den).to(dt)
        del so, sw, po, pw
    return out.view(B, Hkv, G, T, hd).reshape(B, Hq, T, hd).transpose(1, 2).reshape(B, T, Hq * hd)


class FastRead:
    """Patches the reader's attention modules at itf.layers (on top of whatever forward is installed, e.g. an inactive
    D13Reader). on=False -> stock reader. load(op, n_op) with op {enc_layer: (K, V)}, K [Hkv,N,hd] | [Bo,Hkv,N,hd]."""

    def __init__(self, model, itf, dtype=torch.bfloat16):
        from deltaskill.backbone_utils import decoder_layers
        self.model, self.itf, self.dtype = model, itf, dtype
        L = decoder_layers(model)
        self.mods = {li: L[li].self_attn for li in itf.layers}
        self.on = False
        self.op: dict = {}
        self.n_op = 0
        self.past: dict = {}
        self.dA = None          # {li: (Ua [B,d_e,r], Va [B,r,d_r])} per-request deltas (None: itf's own)
        self.gb = None          # [B, n_layers] per-request gates (None: itf.g)
        self._orig = {}
        for li, m in self.mods.items():
            self._orig[li] = m.forward
            m.forward = types.MethodType(_fast_forward(self, li), m)

    def load(self, op: dict, n_op: int, dA: dict | None = None, gb: torch.Tensor | None = None):
        self.op = {e: tuple((X if X.dim() == 4 else X.unsqueeze(0)).to(self.dtype) for X in kv) for e, kv in op.items()}
        self.n_op, self.past, self.dA, self.gb = int(n_op), {}, dA, gb

    def reset(self):
        self.past = {}

    def detach(self):
        for li, f in self._orig.items():
            self.mods[li].forward = f
        self._orig.clear()

    def map_x(self, li, x):
        itf, k = self.itf, str(li)
        y = itf.A[k](x)
        if self.dA is not None:
            Ua, Va = self.dA[li]
            y = y + torch.bmm(torch.bmm(x, Va.transpose(1, 2)), Ua.transpose(1, 2))
        elif itf.delta_rank:
            y = y + (x @ itf.D[f"Va{k}"].t()) @ itf.D[f"Ua{k}"].t()
        if itf.mlp_hidden:
            y = y + itf.M[k](x)
        return y

    def read(self, li, x):
        """x [B,T,d_r] fp32 -> y [B,T,d_r] fp32 (B-mapped read-out)."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        itf = self.itf
        att = itf.att[str(li)]
        hd = att.head_dim
        xe = self.map_x(li, x)
        B, T, _ = xe.shape
        q, gate = torch.chunk(att.q_proj(xe).view(B, T, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = att.q_norm(q).transpose(1, 2)                                   # [B,Hq,T,hd]
        k = att.k_norm(att.k_proj(xe).view(B, T, -1, hd)).transpose(1, 2)    # [B,Hkv,T,hd]
        v = att.v_proj(xe).view(B, T, -1, hd).transpose(1, 2)
        pst = self.past.get(li)
        P = 0 if pst is None else pst[0].shape[2]
        pos = torch.arange(self.n_op + P, self.n_op + P + T, device=xe.device).view(1, 1, T).expand(3, B, T)
        cos, sin = itf.rot(xe, pos)
        qr, kr = apply_rotary_pos_emb(q, k, cos.float(), sin.float())
        kr, v = kr.to(self.dtype), v.to(self.dtype)
        Kown = kr if pst is None else torch.cat([pst[0], kr], 2)
        Vown = v if pst is None else torch.cat([pst[1], v], 2)
        self.past[li] = (Kown, Vown)
        K, V = self.op[itf.layer_map[li]]
        o = attend_two_part(qr, K, V, Kown, Vown, att.scaling, P)
        o = att.o_proj(o.float() * torch.sigmoid(gate))
        return itf.Bmap(li, o)

    def gate(self, li, B):
        i = self.itf._gi[li]
        if self.gb is not None:
            return self.gb[:, i].view(B, 1, 1)
        return self.itf.g[i]


def _fast_forward(R: FastRead, li: int):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        out, w = R._orig[li](hidden_states, position_embeddings, attention_mask, past_key_values=past_key_values, **kwargs)
        if not R.on:
            return out, w
        y = R.read(li, hidden_states.float())
        o = out.float()
        g = R.gate(li, o.shape[0])
        return (o + g * (y - o)).to(out.dtype), w

    return forward


def adapter_tensors(snaps: list, layers, device, dtype=torch.float32):
    """D20 DocAdapter snapshots (dA + gate) of B documents -> (dA {li: (Ua [B,d_e,r], Va [B,r,d_r])}, gb [B,nl])."""
    dA = {li: (torch.stack([s[f"D.Ua{li}"] for s in snaps]).to(device, dtype),
               torch.stack([s[f"D.Va{li}"] for s in snaps]).to(device, dtype)) for li in layers}
    gb = torch.stack([s["g"] for s in snaps]).to(device, dtype)
    return dA, gb


__all__ = ["FastRead", "attend_two_part", "adapter_tensors"]
