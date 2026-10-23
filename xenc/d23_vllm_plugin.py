"""D23: the paper-form read as an out-of-tree vLLM model (vLLM 0.32, envs/vllm032). Imported only in that env.

Model: `Qwen3_5KVSkillForConditionalGeneration` = vLLM's Qwen3_5ForConditionalGeneration (hybrid Gated-DeltaNet +
full attention, unchanged) + the D13 read branch at the read layers. At a read layer ℓ (paired encoder layer e):

    out   = the reader's own attention output (vLLM's paged attention, unchanged)
    x     = the attention input (post input_layernorm), fp32
    xe    = A_ℓ x + M_ℓ(x) [+ per-request registered dA: Ua (Va x)]
    q,g   = q_norm(W_q xe) | W_g xe ;  k = k_norm(W_k xe) ; v = W_v xe          (the encoder's frozen projections)
    RoPE  at positions n_op(doc) + reader position                              (as xenc.d13_read: rows follow the op)
    read  = softmax over [operator(doc) ; own rows (causal)]  ->  B (W_o (σ(g) ⊙ ·))
    y     = out + gate_ℓ (read − out)                                            (gate: per request if registered)

Same math as xenc/d23_fastread.py. The O(N) part runs as ONE paged FlashAttention call over all requests' operators
(operator pool = the documents' K/V in fixed-size blocks; each request's block table points at its document; zero
copy) plus one paged call over the requests' own rows (own-row pool, causal), merged by their log-sum-exp. Small
projections are fp32 torch ops (as in the HF implementation).

Request -> document: the external request id is "d23|<doc_id>|<anything>" (vLLM appends "-<uuid>"). The runner is
patched to publish, before each forward, the request-id order of the flattened batch and the scheduled token counts;
finished requests free their own-row blocks. Requests without the prefix pass through unchanged.

Environment (read at model construction): D23_ITF (interface export of scripts/d23_export_vllm.py), D23_OPS (dir of
operators), D23_ADAPTERS (optional dir of registered adapters), D23_DOCS (comma list of doc ids to load; "doc*k"
loads k independent copies doc#0..doc#k-1, i.e. k DIFFERENT documents' worth of memory), D23_OWN_MAX (own rows per
request, default 1024), D23_MAX_REQS (default 64), D23_BLOCK (pool block size, default 16).
Run the engine in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) with enforce_eager=True (the per-step Python
bookkeeping is not CUDA-graph capturable).
"""

from __future__ import annotations

import os
import types

import torch
import torch.nn as nn

ARCH = "Qwen3_5KVSkillForConditionalGeneration"
CTX = {"pending": None, "nsched": None}          # published by the patched runner before each model forward
STATE = {"engine": None}


# ----------------------------------------------------------------------------- interface rebuild (HF modules)
def build_interface(path: str, device):
    """Rebuild the D13 interface with transformers' own Qwen3_5Attention / rotary modules from the exported state."""
    from transformers.models.qwen3_5 import configuration_qwen3_5 as C
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M

    from xenc.d13_read import D13Interface
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = C.Qwen3_5TextConfig.from_dict(blob["enc_text_config"])
    cfg._attn_implementation = "eager"
    lmap = {int(k): int(v) for k, v in blob["layer_map"].items()}

    class _Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = cfg
            n = max(lmap.values()) + 1
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module() for _ in range(n)])
            for e in set(lmap.values()):
                self.model.layers[e].self_attn = M.Qwen3_5Attention(cfg, e)
            self.model.rotary_emb = M.Qwen3_5TextRotaryEmbedding(cfg)

    with torch.device("cpu"):
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            enc = _Enc()
            B = None if blob["identity_B"] else {li: torch.zeros(blob["d_r"], blob["d_e"]) for li in lmap}
            itf = D13Interface(enc, lmap, blob["d_r"], blob["d_e"], B=B, mlp_hidden=blob["mlp_hidden"])
        finally:
            torch.set_default_dtype(prev)
    missing, unexpected = itf.load_state_dict(blob["state_dict"], strict=False)
    assert not unexpected and not missing, (missing[:5], unexpected[:5])
    itf = itf.float().to(device).eval().requires_grad_(False)
    return itf, blob


# ----------------------------------------------------------------------------- pools
class Pools:
    def __init__(self, itf, device, block: int, own_max: int, max_reqs: int, dtype=torch.bfloat16,
                 capacity_tokens: int = 0, max_adapters: int = 8):
        """capacity_tokens > 0: STATIC operator pool (allocated once, + 1 zero dummy block at the end; load_docs writes
        into it, so captured CUDA graphs keep valid pointers) and static adapter slots (row 0 = no adapter)."""
        self.itf, self.dev, self.bs, self.dtype = itf, device, int(block), dtype
        self.capacity_blocks = -(-int(capacity_tokens) // self.bs) if capacity_tokens else 0
        self.enc_layers = sorted(set(itf.layer_map.values()))
        a0 = itf.att[str(itf.layers[0])]
        self.hd = a0.head_dim
        self.hkv = a0.k_proj.out_features // self.hd
        self.hq = a0.q_proj.out_features // (2 * self.hd)
        self.docs: dict = {}               # doc -> {"start": blk, "nblk": n, "n_op": N}
        self.op_k: dict = {}
        self.op_v: dict = {}
        self.adapters: dict = {}           # doc -> {"Ua": {li}, "Va": {li}, "g": [nl]}
        self.own_blocks = -(-int(own_max) // self.bs)
        self.max_reqs = int(max_reqs)
        n_own = self.own_blocks * (self.max_reqs + 1)             # + 1 dummy slot (padded / non-read rows)
        self.dummy_slot = self.max_reqs
        self.own_k = {li: torch.zeros(n_own, self.bs, self.hkv, self.hd, dtype=dtype, device=device)
                      for li in itf.layers}
        self.own_v = {li: torch.zeros_like(self.own_k[li]) for li in itf.layers}
        self.free_slots = list(range(self.max_reqs))
        self.req: dict = {}                # req_id -> {"doc", "slot", "len"}
        if self.capacity_blocks:
            self.dummy_block = self.capacity_blocks
            for e in self.enc_layers:
                self.op_k[e] = torch.zeros(self.capacity_blocks + 1, self.bs, self.hkv, self.hd, dtype=dtype, device=device)
                self.op_v[e] = torch.zeros_like(self.op_k[e])
            nl = len(itf.layers)
            r = None
            self.ad_rows: dict = {}
            self.max_adapters = int(max_adapters)
            self.ad_g = itf.g.detach().float().view(1, nl).repeat(self.max_adapters + 1, 1).contiguous()
            self.ad_Ua, self.ad_Va, self.ad_b = {}, {}, {}
            self._ad_rank = r

    def load_docs(self, ops_dir: str, specs: list[str], adapters_dir: str | None = None):
        """specs: doc ids, or "doc*k" for k independent copies doc#0..doc#k-1."""
        if self.capacity_blocks:
            return self._load_static(ops_dir, specs, adapters_dir)
        entries = []
        for s in specs:
            if "*" in s:
                d, k = s.split("*")
                entries += [(d, f"{d}#{j}") for j in range(int(k))]
            else:
                entries.append((s, s))
        blobs, total = {}, 0
        for src, name in entries:
            if src not in blobs:
                blobs[src] = torch.load(os.path.join(ops_dir, f"{src}.pt"), map_location="cpu", weights_only=False)
            n = int(blobs[src]["n_op"])
            nb = -(-n // self.bs)
            self.docs[name] = {"start": total, "nblk": nb, "n_op": n, "src": src}
            total += nb
        for e in self.enc_layers:
            self.op_k[e] = torch.zeros(max(total, 1), self.bs, self.hkv, self.hd, dtype=self.dtype, device=self.dev)
            self.op_v[e] = torch.zeros_like(self.op_k[e])
        for name, d in self.docs.items():
            op = blobs[d["src"]]["op"]
            for e in self.enc_layers:
                K, V = op[e]                                     # [Hkv, N, hd]
                n = d["n_op"]
                for X, P in ((K, self.op_k[e]), (V, self.op_v[e])):
                    flat = P[d["start"]: d["start"] + d["nblk"]].view(-1, self.hkv, self.hd)
                    flat[:n].copy_(X.transpose(0, 1).to(self.dev, self.dtype))
            if adapters_dir and os.path.exists(os.path.join(adapters_dir, f"{d['src']}.pt")):
                sn = torch.load(os.path.join(adapters_dir, f"{d['src']}.pt"), map_location="cpu", weights_only=False)
                self.adapters[name] = {
                    "Ua": {li: sn[f"D.Ua{li}"].float().to(self.dev) for li in self.itf.layers},
                    "Va": {li: sn[f"D.Va{li}"].float().to(self.dev) for li in self.itf.layers},
                    "g": sn["g"].float().to(self.dev),
                    # D24 hypernetwork adapters also carry a per-document write-back bias offset (Ub·c)
                    "b": ({li: sn[f"Bb{li}"].float().to(self.dev) for li in self.itf.layers}
                          if f"Bb{self.itf.layers[0]}" in sn else None)}
        del blobs
        return total

    def _load_static(self, ops_dir, specs, adapters_dir):
        """(Re)load into the static pools: resets the document table, keeps every tensor's address."""
        entries = []
        for sp in specs:
            if "*" in sp:
                d, k = sp.split("*")
                entries += [(d, f"{d}#{j}") for j in range(int(k))]
            else:
                entries.append((sp, sp))
        self.docs.clear()
        self.adapters.clear()
        self.ad_rows.clear()
        blobs, total = {}, 0
        for src, name in entries:
            if src not in blobs:
                blobs[src] = torch.load(os.path.join(ops_dir, f"{src}.pt"), map_location="cpu", weights_only=False)
            n = int(blobs[src]["n_op"])
            nb = -(-n // self.bs)
            if total + nb > self.capacity_blocks:
                raise torch.cuda.OutOfMemoryError(f"D23 static operator pool capacity exceeded "
                                                  f"({(total + nb) * self.bs} > {self.capacity_blocks * self.bs} tokens)")
            self.docs[name] = {"start": total, "nblk": nb, "n_op": n, "src": src}
            total += nb
        for name, d in self.docs.items():
            op = blobs[d["src"]]["op"]
            for e in self.enc_layers:
                K, V = op[e]
                for X, P in ((K, self.op_k[e]), (V, self.op_v[e])):
                    flat = P[d["start"]: d["start"] + d["nblk"]].view(-1, self.hkv, self.hd)
                    flat[: d["n_op"]].copy_(X.transpose(0, 1).to(self.dev, self.dtype))
            if adapters_dir and os.path.exists(os.path.join(adapters_dir, f"{d['src']}.pt")):
                sn = torch.load(os.path.join(adapters_dir, f"{d['src']}.pt"), map_location="cpu", weights_only=False)
                self.adapters[name] = {
                    "Ua": {li: sn[f"D.Ua{li}"].float().to(self.dev) for li in self.itf.layers},
                    "Va": {li: sn[f"D.Va{li}"].float().to(self.dev) for li in self.itf.layers},
                    "g": sn["g"].float().to(self.dev),
                    # D24 hypernetwork adapters also carry a per-document write-back bias offset (Ub·c)
                    "b": ({li: sn[f"Bb{li}"].float().to(self.dev) for li in self.itf.layers}
                          if f"Bb{self.itf.layers[0]}" in sn else None)}
        for name, ad in self.adapters.items():                   # static adapter rows 1..max_adapters
            row = len(self.ad_rows) + 1
            assert row <= self.max_adapters, "D23_MAX_ADAPTERS exceeded"
            if not self.ad_Ua:
                for li in self.itf.layers:                      # bias rows: zero unless the adapter carries one
                    self.ad_b[li] = torch.zeros(self.max_adapters + 1, self.itf.d_r, dtype=torch.float32,
                                                device=self.dev)
                for li in self.itf.layers:
                    Ua, Va = ad["Ua"][li], ad["Va"][li]
                    # explicit fp32: vLLM loads the model under a bf16 default dtype (the eager path also uses fp32)
                    self.ad_Ua[li] = torch.zeros(self.max_adapters + 1, *Ua.shape, dtype=torch.float32, device=self.dev)
                    self.ad_Va[li] = torch.zeros(self.max_adapters + 1, *Va.shape, dtype=torch.float32, device=self.dev)
            for li in self.itf.layers:
                self.ad_Ua[li][row].copy_(ad["Ua"][li])
                self.ad_Va[li][row].copy_(ad["Va"][li])
                self.ad_b[li][row].copy_(ad["b"][li] if ad.get("b") is not None else torch.zeros_like(self.ad_b[li][row]))
            self.ad_g[row].copy_(ad["g"])
            self.ad_rows[name] = row
        del blobs
        return total

    def bytes(self) -> dict:
        op = sum(t.nbytes for t in list(self.op_k.values()) + list(self.op_v.values()))
        own = sum(t.nbytes for t in list(self.own_k.values()) + list(self.own_v.values()))
        return {"op_pool": op, "own_pool": own}

    def release(self, req_id):
        r = self.req.pop(req_id, None)
        if r is not None:
            self.free_slots.append(r["slot"])


def doc_of(req_id: str):
    if not req_id.startswith("d23|"):
        return None
    return req_id.split("|")[1]


# ----------------------------------------------------------------------------- per-step plan (shared by layers)
def stage_step(P: Pools, rids, ns):
    """Called ONCE per engine step, before the forward / graph replay (runner hook). Advances every read request's
    own-row state and returns (segments [(token offset, n tokens, rid, doc, first position)], uniform_decode, n_tok).
    Positions are tracked here (a request's rows are exactly the tokens it has been scheduled), so no device value is
    read; a request that starts over (preempted) is released by the hook first and restarts at 0."""
    seg, off, uniform = [], 0, True
    for rid in rids:
        t = int(ns.get(rid, 0))
        if t == 0:
            continue
        uniform &= t == 1
        doc = doc_of(rid.rsplit("-", 1)[0])
        if doc is not None and doc in P.docs:
            st = P.req.get(rid)
            if st is None:
                assert P.free_slots, "D23_MAX_REQS exceeded"
                st = P.req[rid] = {"doc": doc, "slot": P.free_slots.pop(), "len": 0}
            p0 = st["len"]
            assert p0 + t <= P.own_blocks * P.bs, "D23_OWN_MAX exceeded"
            seg.append((off, t, rid, doc, p0))
            st["len"] = p0 + t
        off += t
    STATE["read_tokens"] = STATE.get("read_tokens", 0) + sum(x[1] for x in seg)
    return seg, uniform, off


def make_plan(P: Pools, positions: torch.Tensor, n_tok: int):
    """-> plan dict for this forward, or None (no read requests: dummy / profiling runs). The context is the staged
    step (stage_step) or, for direct calls (tests), a raw (request ids, scheduled tokens) pair staged here."""
    ctx = STATE.get("ctx")
    if not ctx:
        return None
    if ctx[0] == "staged":
        seg = ctx[1]
    else:
        rids, ns = ctx
        if not rids or ns is None:
            return None
        seg = stage_step(P, rids, ns)[0]
    if STATE.get("debug", 0) > 0 and seg:                # cross-check the tracked positions against vLLM's
        pos = positions[0] if positions.dim() == 2 else positions
        pc = pos[:n_tok].tolist()
        for o, t, rid, doc, p0 in seg:
            assert pc[o] == p0, (rid, pc[o], p0)
    if not seg:
        return None
    bs = P.bs
    idx, slots, ropepos, q_lens, own_used, op_used, tok_req = [], [], [], [], [], [], []
    max_op_blk = max(P.docs[d]["nblk"] for _, _, _, d, _ in seg)
    op_bt = torch.zeros(len(seg), max_op_blk, dtype=torch.int32)
    own_bt = torch.zeros(len(seg), P.own_blocks, dtype=torch.int32)
    for j, (o, t, rid, doc, p0) in enumerate(seg):
        d = P.docs[doc]
        slot = P.req[rid]["slot"]
        idx += list(range(o, o + t))
        base = slot * P.own_blocks * bs
        slots += [base + p0 + i for i in range(t)]
        ropepos += [d["n_op"] + p0 + i for i in range(t)]
        q_lens.append(t)
        own_used.append(p0 + t)
        op_used.append(d["n_op"])
        tok_req += [j] * t
        op_bt[j, : d["nblk"]] = torch.arange(d["start"], d["start"] + d["nblk"], dtype=torch.int32)
        own_bt[j] = torch.arange(slot * P.own_blocks, (slot + 1) * P.own_blocks, dtype=torch.int32)
    dev = P.dev
    cu = torch.zeros(len(seg) + 1, dtype=torch.int32)
    cu[1:] = torch.cumsum(torch.tensor(q_lens, dtype=torch.int32), 0)
    plan = {"idx": torch.tensor(idx, device=dev), "slots": torch.tensor(slots, device=dev),
            "pos": torch.tensor(ropepos, device=dev), "cu_q": cu.to(dev), "max_q": max(q_lens),
            "op_used": torch.tensor(op_used, dtype=torch.int32, device=dev), "max_op": max(op_used),
            "own_used": torch.tensor(own_used, dtype=torch.int32, device=dev), "max_own": max(own_used),
            "op_bt": op_bt.to(dev), "own_bt": own_bt.to(dev), "tok_req": torch.tensor(tok_req, device=dev),
            "docs": [s[3] for s in seg]}
    ads = [P.adapters.get(d) for d in plan["docs"]]
    plan["adapters"] = ads if any(x is not None for x in ads) else None
    return plan


def ref_varlen(q, k, v, max_q, cu_q, max_k, seqused_k=None, softmax_scale=None, causal=False, block_table=None,
               return_softmax_lse=False, fa_version=2, **kw):
    """Pure-torch reference of vllm_flash_attn.flash_attn_varlen_func (paged, GQA, causal bottom-right, fp32)."""
    Hq, hd = q.shape[1], q.shape[2]
    Hkv = k.shape[2]
    G = Hq // Hkv
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    lse = torch.zeros(Hq, q.shape[0], device=q.device)
    cu, used, bt = cu_q.tolist(), seqused_k.tolist(), block_table.long()
    for i in range(len(cu) - 1):
        a, b, n = cu[i], cu[i + 1], used[i]
        K = k[bt[i]].reshape(-1, Hkv, hd)[:n].float().repeat_interleave(G, 1)
        V = v[bt[i]].reshape(-1, Hkv, hd)[:n].float().repeat_interleave(G, 1)
        s_ = torch.einsum("thd,nhd->htn", q[a:b].float(), K) * softmax_scale
        if causal:
            T = b - a
            col = torch.arange(n, device=q.device).view(1, n)
            row = torch.arange(T, device=q.device).view(T, 1)
            s_ = s_.masked_fill((col > n - T + row).unsqueeze(0), float("-inf"))
        lse[:, a:b] = torch.logsumexp(s_, -1)
        out[a:b] = torch.einsum("htn,nhd->thd", torch.softmax(s_, -1), V)
    return out.to(q.dtype), lse


def read_layer(P: Pools, li: int, x: torch.Tensor, plan: dict, fa_version: int, ref: bool = False):
    """x [n, d_r] (the plan's rows) -> y [n, d_r] fp32 (B-mapped read-out) and the per-token gate [n]."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    if ref:
        flash_attn_varlen_func = ref_varlen
    else:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    itf = P.itf
    k_ = str(li)
    att = itf.att[k_]
    hd, hq, hkv = P.hd, P.hq, P.hkv
    xf = x.float()
    xe = itf.A[k_](xf)
    if itf.mlp_hidden:
        xe = xe + itf.M[k_](xf)
    gi = itf._gi[li]
    gate_tok = itf.g[gi].expand(x.shape[0])
    if plan["adapters"] is not None:
        r = plan["tok_req"]
        Ua = torch.stack([a["Ua"][li] if a else torch.zeros_like(next(z for z in plan["adapters"] if z)["Ua"][li])
                          for a in plan["adapters"]])
        Va = torch.stack([a["Va"][li] if a else torch.zeros_like(next(z for z in plan["adapters"] if z)["Va"][li])
                          for a in plan["adapters"]])
        g = torch.stack([a["g"] if a else itf.g for a in plan["adapters"]])
        xe = xe + torch.bmm(torch.bmm(xf.unsqueeze(1), Va[r].transpose(1, 2)), Ua[r].transpose(1, 2)).squeeze(1)
        gate_tok = g[r, gi]
    n = xe.shape[0]
    q, gate = torch.chunk(att.q_proj(xe).view(n, -1, hd * 2), 2, dim=-1)
    gate = gate.reshape(n, -1)
    q = att.q_norm(q)                                                    # [n, Hq, hd]
    k = att.k_norm(att.k_proj(xe).view(n, -1, hd))                      # [n, Hkv, hd]
    v = att.v_proj(xe).view(n, -1, hd)
    pos = plan["pos"].view(1, 1, n).expand(3, 1, n)
    cos, sin = itf.rot(xe.unsqueeze(0), pos)
    qr, kr = apply_rotary_pos_emb(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), cos.float(), sin.float())
    qr = qr[0].transpose(0, 1).to(P.dtype).contiguous()                 # [n, Hq, hd]
    kr = kr[0].transpose(0, 1).to(P.dtype)
    # write own rows into the own-row pool, then two paged FlashAttention calls + LSE merge
    Kp = P.own_k[li].view(-1, hkv, hd)
    Vp = P.own_v[li].view(-1, hkv, hd)
    Kp.index_copy_(0, plan["slots"], kr)
    Vp.index_copy_(0, plan["slots"], v.to(P.dtype))
    e = itf.layer_map[li]
    o1, l1 = flash_attn_varlen_func(qr, P.op_k[e], P.op_v[e], plan["max_q"], plan["cu_q"], plan["max_op"],
                                    seqused_k=plan["op_used"], softmax_scale=att.scaling, causal=False,
                                    block_table=plan["op_bt"], return_softmax_lse=True, fa_version=fa_version)
    o2, l2 = flash_attn_varlen_func(qr, P.own_k[li], P.own_v[li], plan["max_q"], plan["cu_q"], plan["max_own"],
                                    seqused_k=plan["own_used"], softmax_scale=att.scaling, causal=True,
                                    block_table=plan["own_bt"], return_softmax_lse=True, fa_version=fa_version)
    l1, l2 = l1.float(), l2.float()                                      # [Hq, n]
    lse = torch.logaddexp(l1, l2)
    w1 = torch.exp(l1 - lse).t().unsqueeze(-1)
    w2 = torch.exp(l2 - lse).t().unsqueeze(-1)
    o = (o1.float() * w1 + o2.float() * w2).reshape(n, hq * hd)
    o = att.o_proj(o * torch.sigmoid(gate))
    y = itf.Bmap(li, o)
    if plan["adapters"] is not None and any(a is not None and a.get("b") is not None for a in plan["adapters"]):
        bias = torch.stack([a["b"][li] if (a is not None and a.get("b") is not None) else torch.zeros_like(y[0])
                            for a in plan["adapters"]])
        y = y + bias[plan["tok_req"]]
    return y, gate_tok


# ----------------------------------------------------------------------------- static decode path (CUDA graphs)
class Static:
    """Persistent per-token decode metadata (device buffers updated IN PLACE before every decode step / replay)."""

    def __init__(self, P: Pools, max_n: int, max_doc_tokens: int, attn: str = "fa"):
        dev, self.N = P.dev, int(max_n)
        self.max_op_blocks = -(-int(max_doc_tokens) // P.bs)
        self.max_doc_tokens = self.max_op_blocks * P.bs
        self.own_max = P.own_blocks * P.bs
        self.nl = len(P.itf.layers)
        self.ints = torch.zeros(self.N, 7, dtype=torch.int64, device=dev)   # op_start, op_len, own_slot, own_len,
        self.flts = torch.zeros(self.N, self.nl + 1, dtype=torch.float32, device=dev)  # wslot, rope, ad | gates, mask
        self.op_bt = torch.zeros(self.N, self.max_op_blocks, dtype=torch.int32, device=dev)
        self.own_bt = torch.zeros(self.N, P.own_blocks, dtype=torch.int32, device=dev)
        self.op_len = torch.ones(self.N, dtype=torch.int32, device=dev)
        self.own_len = torch.ones(self.N, dtype=torch.int32, device=dev)
        self.ar_op = torch.arange(self.max_op_blocks, dtype=torch.int64, device=dev).view(1, -1)
        self.ar_own = torch.arange(P.own_blocks, dtype=torch.int64, device=dev).view(1, -1)
        self.cu = torch.arange(self.N + 1, dtype=torch.int32, device=dev)
        self.attn = attn
        self.dec = None
        if attn == "triton":
            from xenc.d23_read_kernels import DecodeReader
            self.dec = DecodeReader(self.N, P.hq, P.hkv, P.hd, self.max_doc_tokens + self.own_max, dev)
        self.fill(P, [], 0)

    def fill(self, P: Pools, seg, n_tok: int):
        """Host side: per-token rows for a uniform decode step (seg from stage_step); padded rows -> dummy, mask 0."""
        import numpy as np
        ints = np.zeros((self.N, 7), dtype=np.int64)
        flts = np.zeros((self.N, self.nl + 1), dtype=np.float32)
        dummy_op = P.capacity_blocks if P.capacity_blocks else 0
        ints[:, 0], ints[:, 1] = dummy_op, 1
        ints[:, 2], ints[:, 3] = P.dummy_slot, 1
        ints[:, 4] = P.dummy_slot * P.own_blocks * P.bs
        base_g = P.ad_g[0].tolist() if P.capacity_blocks else P.itf.g.tolist()
        flts[:, : self.nl] = base_g
        for o, t, rid, doc, p0 in seg:
            d, st = P.docs[doc], P.req[rid]
            ints[o] = (d["start"], d["n_op"], st["slot"], p0 + 1, st["slot"] * P.own_blocks * P.bs + p0, d["n_op"] + p0,
                       P.ad_rows.get(doc, 0) if P.capacity_blocks else 0)
            if P.capacity_blocks and doc in P.ad_rows:
                flts[o, : self.nl] = P.ad_g[P.ad_rows[doc]].tolist()
            flts[o, self.nl] = 1.0
        self.ints.copy_(torch.from_numpy(ints))
        self.flts.copy_(torch.from_numpy(flts))
        torch.add(self.ints[:, 0:1], self.ar_op, out=self.op_bt)       # contiguous document pages
        self.op_bt.clamp_(max=P.capacity_blocks)                         # pages past a document: the dummy block
        torch.add(self.ints[:, 2:3] * P.own_blocks, self.ar_own, out=self.own_bt)
        self.op_len.copy_(self.ints[:, 1])
        self.own_len.copy_(self.ints[:, 3])


def read_static(P: Pools, S: Static, li: int, x: torch.Tensor, fa_version: int):
    """Capture-safe decode read for the first N rows (one row per request): static shapes, persistent buffers only."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    itf = P.itf
    k_ = str(li)
    att = itf.att[k_]
    hd, hq, hkv = P.hd, P.hq, P.hkv
    N = x.shape[0]
    xf = x.float()
    xe = itf.A[k_](xf)
    if itf.mlp_hidden:
        xe = xe + itf.M[k_](xf)
    if P.capacity_blocks and P.ad_Ua:
        ad = S.ints[:N, 6]
        xe = xe + torch.bmm(torch.bmm(xf.unsqueeze(1), P.ad_Va[li][ad].transpose(1, 2)),
                            P.ad_Ua[li][ad].transpose(1, 2)).squeeze(1)
    q, gate = torch.chunk(att.q_proj(xe).view(N, -1, hd * 2), 2, dim=-1)
    gate = gate.reshape(N, -1)
    q = att.q_norm(q)
    k = att.k_norm(att.k_proj(xe).view(N, -1, hd))
    v = att.v_proj(xe).view(N, -1, hd)
    pos = S.ints[:N, 5].view(1, 1, N).expand(3, 1, N)
    cos, sin = itf.rot(xe.unsqueeze(0), pos)
    qr, kr = apply_rotary_pos_emb(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), cos.float(), sin.float())
    qr = qr[0].transpose(0, 1).to(P.dtype).contiguous()
    kr = kr[0].transpose(0, 1).to(P.dtype)
    P.own_k[li].view(-1, hkv, hd).index_copy_(0, S.ints[:N, 4], kr)
    P.own_v[li].view(-1, hkv, hd).index_copy_(0, S.ints[:N, 4], v.to(P.dtype))
    e = itf.layer_map[li]
    if S.attn == "triton":
        o = S.dec(qr, P.op_k[e], P.op_v[e], S.op_bt[:N], S.op_len[:N], P.own_k[li], P.own_v[li], S.own_bt[:N],
                  S.own_len[:N], att.scaling)
    else:
        if S.attn == "ref":
            fn = ref_varlen
        else:
            from vllm.vllm_flash_attn import flash_attn_varlen_func as fn
        o1, l1 = fn(qr, P.op_k[e], P.op_v[e], 1, S.cu[: N + 1], S.max_doc_tokens, seqused_k=S.op_len[:N],
                    softmax_scale=att.scaling, causal=False, block_table=S.op_bt[:N], return_softmax_lse=True,
                    fa_version=fa_version)
        o2, l2 = fn(qr, P.own_k[li], P.own_v[li], 1, S.cu[: N + 1], S.own_max, seqused_k=S.own_len[:N],
                    softmax_scale=att.scaling, causal=True, block_table=S.own_bt[:N], return_softmax_lse=True,
                    fa_version=fa_version)
        l1, l2 = l1.float(), l2.float()
        lse = torch.logaddexp(l1, l2)
        o = o1.float() * torch.exp(l1 - lse).t().unsqueeze(-1) + o2.float() * torch.exp(l2 - lse).t().unsqueeze(-1)
    o = att.o_proj(o.reshape(N, hq * hd) * torch.sigmoid(gate))
    y = itf.Bmap(li, o)
    if P.capacity_blocks and P.ad_b:
        y = y + P.ad_b[li][S.ints[:N, 6]]                                 # per-document write-back bias (row 0 = 0)
    g = S.flts[:N, itf._gi[li]] * S.flts[:N, S.nl]                    # gate x mask (0 for padded / non-read rows)
    return y, g


# ----------------------------------------------------------------------------- vLLM wiring
def _release(scheduler_output):
    P = STATE.get("pools")
    if P is None:
        return
    for name in ("finished_req_ids", "preempted_req_ids"):
        for rid in getattr(scheduler_output, name, None) or ():
            P.release(rid)


def _stage(rids, ns):
    """Hook body: stage the step for the read and (graph mode) fill the static decode buffers."""
    P = STATE.get("pools")
    if P is None:
        CTX["pending"] = None
        return
    seg, uniform, n_tok = stage_step(P, rids, ns)
    S = STATE.get("static")
    static_ok = bool(S is not None and uniform and 0 < n_tok <= S.N)
    if static_ok:
        S.fill(P, seg, n_tok)
        STATE["static_tokens"] = STATE.get("static_tokens", 0) + sum(x[1] for x in seg)
    CTX["pending"] = ("staged", seg, static_ok)


def _patch_runner():
    """Publish (request-id order of the flattened batch, scheduled tokens per request) before each model forward,
    and free finished requests' own-row slots. vLLM 0.32 defaults to the V2 model runner (v1/worker/gpu); the V1
    runner is patched as well. The model's forward consumes the published context (see register())."""
    from vllm.v1.worker import gpu_model_runner as gmr
    R = gmr.GPUModelRunner
    if not getattr(R, "_d23_patched", False):
        orig_exec, orig_fwd = R.execute_model, R._model_forward

        def execute_model(self, scheduler_output, *args, **kwargs):
            _release(scheduler_output)
            CTX["nsched"] = dict(scheduler_output.num_scheduled_tokens)
            return orig_exec(self, scheduler_output, *args, **kwargs)

        def _model_forward(self, *args, **kwargs):
            _stage(list(self.input_batch.req_ids[: self.input_batch.num_reqs]), CTX.get("nsched") or {})
            return orig_fwd(self, *args, **kwargs)

        R.execute_model, R._model_forward, R._d23_patched = execute_model, _model_forward, True
    try:
        from vllm.v1.worker.gpu import model_runner as mr2
    except Exception:
        return
    R2 = mr2.GPUModelRunner
    if getattr(R2, "_d23_patched", False):
        return
    orig_prep, orig_fin = R2.prepare_inputs, R2.finish_requests

    def prepare_inputs(self, scheduler_output, batch_req_state, *args, **kwargs):
        ids = list(batch_req_state.req_ids)
        ns = [int(x) for x in batch_req_state.num_scheduled_tokens]
        _stage(ids, dict(zip(ids, ns)))
        return orig_prep(self, scheduler_output, batch_req_state, *args, **kwargs)

    def finish_requests(self, scheduler_output):
        _release(scheduler_output)
        return orig_fin(self, scheduler_output)

    R2.prepare_inputs, R2.finish_requests, R2._d23_patched = prepare_inputs, finish_requests, True


def _install(model):
    from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
    dev = next(model.parameters()).device
    itf, blob = build_interface(os.environ["D23_ITF"], dev)
    graph = os.environ.get("D23_GRAPH", "0") == "1"
    P = Pools(itf, dev, int(os.environ.get("D23_BLOCK", 16)), int(os.environ.get("D23_OWN_MAX", 1024)),
              int(os.environ.get("D23_MAX_REQS", 64)),
              capacity_tokens=int(os.environ.get("D23_POOL_TOKENS", 0)) if graph else 0,
              max_adapters=int(os.environ.get("D23_MAX_ADAPTERS", 8)))
    specs = [s for s in os.environ.get("D23_DOCS", "").split(",") if s]
    P.load_docs(os.environ["D23_OPS"], specs, os.environ.get("D23_ADAPTERS") or None)
    STATE["pools"], STATE["plan"] = P, None
    STATE["static"] = None
    if graph:
        assert P.capacity_blocks, "D23_GRAPH=1 needs D23_POOL_TOKENS (static operator pool)"
        STATE["static"] = Static(P, int(os.environ.get("D23_MAXN", 64)),
                                 int(os.environ.get("D23_MAX_DOC_TOKENS", 66000)),
                                 os.environ.get("D23_DECODE_ATTN", "fa"))
    STATE["debug"] = int(os.environ.get("D23_DEBUG", 0))
    try:
        from vllm.attention.utils.fa_utils import get_flash_attn_version
    except Exception:   # module moved across versions
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
    fa = get_flash_attn_version() or 2
    STATE["fa_version"] = fa
    found = []
    for name, m in model.named_modules():
        if not isinstance(m, Qwen3NextAttention):
            continue
        li = int(name.split(".layers.")[1].split(".")[0])
        if li in itf.layer_map:
            m.forward = types.MethodType(_read_forward(li, m.forward), m)
            found.append(li)
    assert sorted(found) == sorted(itf.layer_map), (found, itf.layer_map)
    STATE["info"] = {"layers": found, "fa_version": fa, "pool_bytes": P.bytes(), "docs": list(P.docs),
                     "graph": graph, "decode_attn": os.environ.get("D23_DECODE_ATTN", "fa") if graph else None,
                     "itf_bytes": sum(p.numel() * p.element_size() for p in itf.parameters())
                     + sum(b.numel() * b.element_size() for b in itf.buffers())}


def _read_forward(li: int, orig):
    def forward(self, positions, hidden_states, *args, **kwargs):
        out = orig(positions, hidden_states, *args, **kwargs)
        P = STATE.get("pools")
        if P is None:
            return out
        S = STATE.get("static")
        N = hidden_states.shape[0]
        if S is not None and N <= S.N and (torch.cuda.is_current_stream_capturing() or STATE.get("static_ok")):
            y, g = read_static(P, S, li, hidden_states, STATE["fa_version"])
            if STATE.get("debug", 0) > 0 and not torch.cuda.is_current_stream_capturing():
                STATE["debug"] -= 1
                attn, S.attn = S.attn, "ref"
                try:
                    yr, _ = read_static(P, S, li, hidden_states, STATE["fa_version"])
                finally:
                    S.attn = attn
                live = g != 0
                if bool(live.any()):
                    a_, b_ = y[live], yr[live]
                    STATE.setdefault("debug_log", []).append(
                        {"layer": li, "n": int(live.sum()), "path": "static/" + attn,
                         "rel_err": float((a_ - b_).norm() / b_.norm().clamp_min(1e-9)),
                         "min_cos": float(torch.nn.functional.cosine_similarity(a_, b_, dim=-1).min())})
                    print("D23_DEBUG", STATE["debug_log"][-1], flush=True)
            o = out.float()
            return (o + g.unsqueeze(-1) * (y - o)).to(out.dtype)
        plan = STATE.get("plan")
        if plan is None:
            plan = STATE["plan"] = make_plan(P, positions, hidden_states.shape[0]) or False
            if plan:
                STATE["read_forwards"] = STATE.get("read_forwards", 0) + 1
        if plan is False:
            return out
        idx = plan["idx"]
        xs = hidden_states.index_select(0, idx)
        y, g = read_layer(P, li, xs, plan, STATE["fa_version"])
        if STATE.get("debug", 0) > 0:                 # D23_DEBUG: compare against the pure-torch reference
            STATE["debug"] -= 1
            yr, _ = read_layer(P, li, xs, plan, STATE["fa_version"], ref=True)
            rel = float((y - yr).norm() / yr.norm().clamp_min(1e-9))
            cos = float(torch.nn.functional.cosine_similarity(y, yr, dim=-1).min())
            pos = positions[0] if positions.dim() == 2 else positions
            STATE.setdefault("debug_log", []).append(
                {"layer": li, "n": int(xs.shape[0]), "rel_err": rel, "min_cos": cos, "docs": plan["docs"],
                 "positions_head": pos[:4].tolist(), "pos_dim": positions.dim(), "rope_head": plan["pos"][:3].tolist(),
                 "gate_head": g[:2].tolist(), "y_norm": float(y.norm()), "out_norm": float(out.float().norm())})
            print("D23_DEBUG", STATE["debug_log"][-1], flush=True)
        o = out.index_select(0, idx).float()
        mixed = (o + g.unsqueeze(-1) * (y - o)).to(out.dtype)
        out = out.clone()
        out.index_copy_(0, idx, mixed)
        return out
    return forward


def register():
    """Register the architecture and patch the runner (call before constructing vllm.LLM, in-process engine)."""
    from vllm import ModelRegistry
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    class Qwen3_5KVSkillForConditionalGeneration(Qwen3_5ForConditionalGeneration):
        def forward(self, *args, **kwargs):
            ctx = CTX.pop("pending", None)                                   # fresh per forward (dummy runs: None)
            STATE["ctx"], STATE["plan"] = ctx, None
            STATE["static_ok"] = bool(ctx and ctx[0] == "staged" and ctx[2])
            try:
                return super().forward(*args, **kwargs)
            finally:
                STATE["ctx"], STATE["plan"], STATE["static_ok"] = None, None, False

        def load_weights(self, weights):
            loaded = super().load_weights(weights)
            _install(self)
            return loaded

    ModelRegistry.register_model(ARCH, Qwen3_5KVSkillForConditionalGeneration)
    _patch_runner()
    return ARCH


__all__ = ["register", "STATE", "CTX", "ARCH", "build_interface", "Pools", "make_plan", "read_layer", "stage_step",
           "Static", "read_static", "ref_varlen"]
