"""D23: fused DECODE kernel for the paper-form read in vLLM (one query row per request).

For every query row n and kv head h (GQA group of G query heads), attention over the CONCATENATION of
  (1) the document operator: K/V pages of a paged pool, block table op_bt[n], length op_len[n]
  (2) the request's own rows (reader rows mapped into the encoder's space, incl. the current one): own_bt[n], own_len[n]
with one softmax over both (the read's joint softmax; no separate LSE merge pass is needed).

Flash-decoding / split-K: stage 1 (grid N x Hkv x S) - each program streams a fixed CHUNK of the combined key range in
BLOCK_N tiles, keeping an online max / sum / accumulator for the G heads (padded to GP >= 16 rows for tl.dot);
stage 2 (grid N x Hkv) reduces the S partials by log-sum-exp. All shapes are static given (N, S), and nothing reads
device values on the host, so it is CUDA-graph capturable.

Pools: [num_blocks, BS, Hkv, D] (bf16), the same block size BS for both. Output: fp32 [N, Hq, D].
`decode_read_ref` is the pure-torch reference (also the CPU fallback).
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:          # CPU-only test environments
    HAVE_TRITON = False


def decode_read_ref(q, op_k, op_v, op_bt, op_len, own_k, own_v, own_bt, own_len, scale):
    """q [N,Hq,D]; pools [nb,BS,Hkv,D]; bt [N,maxb] int; len [N] int -> fp32 [N,Hq,D] (joint softmax)."""
    N, Hq, D = q.shape
    Hkv = op_k.shape[2]
    G = Hq // Hkv
    out = torch.zeros(N, Hq, D, dtype=torch.float32, device=q.device)
    for n in range(N):
        lo, lw = int(op_len[n]), int(own_len[n])
        K1 = op_k[op_bt[n].long()].reshape(-1, Hkv, D)[:lo]
        V1 = op_v[op_bt[n].long()].reshape(-1, Hkv, D)[:lo]
        K2 = own_k[own_bt[n].long()].reshape(-1, Hkv, D)[:lw]
        V2 = own_v[own_bt[n].long()].reshape(-1, Hkv, D)[:lw]
        K = torch.cat([K1, K2]).float().repeat_interleave(G, 1)        # [L, Hq, D]
        V = torch.cat([V1, V2]).float().repeat_interleave(G, 1)
        s = torch.einsum("hd,lhd->hl", q[n].float(), K) * scale
        out[n] = torch.einsum("hl,lhd->hd", torch.softmax(s, -1), V)
    return out


if HAVE_TRITON:
    @triton.jit
    def _split_kernel(Q, KOP, VOP, BTOP, LOP, KOWN, VOWN, BTOWN, LOWN, PACC, PM, PL, scale,
                      s_qn, s_qh, s_kb, s_ks, s_kh, s_btop, s_btown, s_pn, s_ph, s_ps, s_pg, s_mn, s_mh, s_ms,
                      G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
                      BLOCK_N: tl.constexpr, CHUNK: tl.constexpr):
        n = tl.program_id(0)
        h = tl.program_id(1)
        s = tl.program_id(2)
        lop = tl.load(LOP + n)
        lown = tl.load(LOWN + n)
        total = lop + lown
        offs_g = tl.arange(0, GP)
        offs_d = tl.arange(0, D)
        gmask = offs_g < G
        q = tl.load(Q + n * s_qn + (h * G + offs_g)[:, None] * s_qh + offs_d[None, :],
                    mask=gmask[:, None], other=0.0)
        m_i = tl.full([GP], float("-inf"), tl.float32)
        l_i = tl.zeros([GP], tl.float32)
        acc = tl.zeros([GP, D], tl.float32)
        start = s * CHUNK
        hi = tl.minimum(CHUNK, total - start)              # splits past this row's keys do no work
        for kk in range(0, hi, BLOCK_N):
            k_idx = start + kk + tl.arange(0, BLOCK_N)
            valid = k_idx < total
            is_op = k_idx < lop
            m_op = valid & is_op
            m_own = valid & (k_idx >= lop)
            op_page = tl.load(BTOP + n * s_btop + k_idx // BS, mask=m_op, other=0)
            own_i = tl.maximum(k_idx - lop, 0)
            own_page = tl.load(BTOWN + n * s_btown + own_i // BS, mask=m_own, other=0)
            op_off = op_page.to(tl.int64) * s_kb + (k_idx % BS) * s_ks + h * s_kh
            own_off = own_page.to(tl.int64) * s_kb + (own_i % BS) * s_ks + h * s_kh
            k = tl.load(KOP + op_off[:, None] + offs_d[None, :], mask=m_op[:, None], other=0.0) + \
                tl.load(KOWN + own_off[:, None] + offs_d[None, :], mask=m_own[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * scale                                  # [GP, BLOCK_N]
            qk = tl.where(valid[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(qk - m_safe[:, None])
            alpha = tl.exp(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(VOP + op_off[:, None] + offs_d[None, :], mask=m_op[:, None], other=0.0) + \
                tl.load(VOWN + own_off[:, None] + offs_d[None, :], mask=m_own[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        base = n * s_pn + h * s_ph + s * s_ps
        tl.store(PACC + base + offs_g[:, None] * s_pg + offs_d[None, :], acc, mask=gmask[:, None])
        mb = n * s_mn + h * s_mh + s * s_ms
        tl.store(PM + mb + offs_g, m_i, mask=gmask)
        tl.store(PL + mb + offs_g, l_i, mask=gmask)

    @triton.jit
    def _reduce_kernel(PACC, PM, PL, OUT, S, s_pn, s_ph, s_ps, s_pg, s_mn, s_mh, s_ms, s_on, s_oh,
                       G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr):
        n = tl.program_id(0)
        h = tl.program_id(1)
        offs_g = tl.arange(0, GP)
        offs_d = tl.arange(0, D)
        gmask = offs_g < G
        m_g = tl.full([GP], float("-inf"), tl.float32)
        for s in range(0, S):
            m = tl.load(PM + n * s_mn + h * s_mh + s * s_ms + offs_g, mask=gmask, other=float("-inf"))
            m_g = tl.maximum(m_g, m)
        m_safe = tl.where(m_g == float("-inf"), 0.0, m_g)
        l_tot = tl.zeros([GP], tl.float32)
        acc = tl.zeros([GP, D], tl.float32)
        for s in range(0, S):
            m = tl.load(PM + n * s_mn + h * s_mh + s * s_ms + offs_g, mask=gmask, other=float("-inf"))
            l = tl.load(PL + n * s_mn + h * s_mh + s * s_ms + offs_g, mask=gmask, other=0.0)
            w = tl.exp(m - m_safe)
            w = tl.where(l > 0, w, 0.0)
            a = tl.load(PACC + n * s_pn + h * s_ph + s * s_ps + offs_g[:, None] * s_pg + offs_d[None, :],
                        mask=gmask[:, None], other=0.0)
            l_tot += l * w
            acc += a * w[:, None]
        o = acc / tl.maximum(l_tot, 1e-30)[:, None]
        tl.store(OUT + n * s_on + (h * G + offs_g)[:, None] * s_oh + offs_d[None, :], o, mask=gmask[:, None])


class DecodeReader:
    """Static workspace for the fused decode read (allocate once; capture-safe calls).
    max_keys bounds op_len + own_len; chunk = keys per split program."""

    def __init__(self, max_n: int, hq: int, hkv: int, d: int, max_keys: int, device, chunk: int = 1024,
                 block_n: int = 64):
        self.G, self.hq, self.hkv, self.d = hq // hkv, hq, hkv, d
        self.GP = max(16, 1 << (self.G - 1).bit_length())
        self.chunk, self.block_n = int(chunk), int(block_n)
        self.S = max(1, math.ceil(max_keys / self.chunk))
        self.acc = torch.empty(max_n, hkv, self.S, self.G, d, dtype=torch.float32, device=device)
        self.m = torch.empty(max_n, hkv, self.S, self.G, dtype=torch.float32, device=device)
        self.l = torch.empty_like(self.m)
        self.out = torch.empty(max_n, hq, d, dtype=torch.float32, device=device)

    def __call__(self, q, op_k, op_v, op_bt, op_len, own_k, own_v, own_bt, own_len, scale):
        assert HAVE_TRITON and q.is_cuda
        N = q.shape[0]
        BS = op_k.shape[1]
        assert own_k.shape[1] == BS, "operator and own-row pools must share the block size"
        assert op_k.stride(1) == own_k.stride(1) and op_k.stride(2) == own_k.stride(2) and op_k.stride(0) == own_k.stride(0)
        acc, m, l, out = self.acc[:N], self.m[:N], self.l[:N], self.out[:N]
        _split_kernel[(N, self.hkv, self.S)](
            q, op_k, op_v, op_bt, op_len, own_k, own_v, own_bt, own_len, acc, m, l, float(scale),
            q.stride(0), q.stride(1), op_k.stride(0), op_k.stride(1), op_k.stride(2), op_bt.stride(0), own_bt.stride(0),
            acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3), m.stride(0), m.stride(1), m.stride(2),
            G=self.G, GP=self.GP, D=self.d, BS=BS, BLOCK_N=self.block_n, CHUNK=self.chunk, num_warps=4)
        _reduce_kernel[(N, self.hkv)](
            acc, m, l, out, self.S, acc.stride(0), acc.stride(1), acc.stride(2), acc.stride(3),
            m.stride(0), m.stride(1), m.stride(2), out.stride(0), out.stride(1),
            G=self.G, GP=self.GP, D=self.d, num_warps=4)
        return out


__all__ = ["decode_read_ref", "DecodeReader", "HAVE_TRITON"]
