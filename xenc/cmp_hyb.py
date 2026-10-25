"""CMP hybrid (HYB): our D13 read at reader layers 3-19 PLUS a compact per-token document channel that the reader's
DEEP full-attention layers 23/27/31 read with their OWN attention (agent CMP, 2026-10-23).

Why: ADDR (outputs/ADDR_SUMMARY.md §1) located the whole kv-text span gap in the deep layers 23/27/31, which never see
the document in our read; Hub Memory (HM) writes a 128-d per-token code that the 9B's own layers (incl. 23/27/31) read
natively. outputs/HM_vs_OURS_METHOD.md §6 option 2 proposes inserting HM-style memory at the 9B layer-22 output. HYB is
the cheap form of that option: instead of running M placeholder positions through layers 23-31, the memory enters ONLY
the three deep full-attention layers, as extra key/value columns in the reader's own attention:

    z_t          = E (x_t − μ)                 x_t = the 4B encoder's pre-RoPE K/V at its layers 23/27/31 (the same
                                               frozen prefill as the operator); E = fixed PCA basis (xenc/cmp_op.py);
                                               z_t is what is STORED (d numbers / token)
    (K̂ℓ, V̂ℓ)_t   = decode_ℓ(z_t)               closed-form ridge to the 9B's OWN pre-RoPE K/V at layer ℓ of a text pass
                                               over the same training-document tokens (trainable U), RoPE at the column
    attn_ℓ(row)  = softmax(q_row·[K̂ℓ ; K_own]) [V̂ℓ ; V_own]   the reader's OWN q/k/v/o and output gate at ℓ
    out_ℓ        = stock_ℓ + g_ℓ (attn_ℓ − stock_ℓ)   g_ℓ = 0 at init: step 0 == the base exactly

The DeltaNet layers and layers ≤ 22 never see the memory; the memory columns are not updated by the reader (no reader
pass over the document). Reader cost per question: 3 layers × N columns of attention + the decode (N·d·2048·3 MACs),
vs a text prefill of N tokens × 32 layers.
Oracle (diagnostic, `oracle=True`): the memory is the 9B's TRUE K/V at 23/27/31 from a reader pass over the document
prefix (what text reading gives those layers) — the ceiling of the channel; it costs a reader prefill and is NOT a
method.

Feature keys: 4B layers e (23, 27, 31) and 9B reader layers ℓ under key 100+ℓ (123, 127, 131) in one OpStats / one
OpCompressor (spec '23+27+31+123+127+131|L23+L27+L31:d': the 4B layers are the encoder group, the 9B layers are
decoded from z by cross-model regression).
"""

from __future__ import annotations

import copy
import types

import torch
import torch.nn as nn

RKEY = 100            # 9B reader layer ℓ is stored under key RKEY + ℓ


def _rot(model):
    from xenc.xattn_read import _rotary
    return _rotary(model)


class DeepMem(nn.Module):
    """Gates + frozen fp32 copies of the reader's own attention at the deep layers; holds the per-document memory."""

    def __init__(self, reader, layers=(23, 27, 31), g_init: float = 0.0):
        super().__init__()
        from deltaskill.backbone_utils import decoder_layers
        L = decoder_layers(reader)
        self.layers = [int(x) for x in layers]
        self.att = nn.ModuleDict({str(li): copy.deepcopy(L[li].self_attn).float().requires_grad_(False)
                                  for li in self.layers})
        self.rot = copy.deepcopy(_rot(reader)).float()
        self.g = nn.Parameter(torch.full((len(self.layers),), float(g_init)))
        self._gi = {li: i for i, li in enumerate(self.layers)}
        self.mem: dict = {}                       # li -> (K post-RoPE [Hkv,N,hd], V [Hkv,N,hd])
        self.n_mem = 0

    def read(self, li, x, past=None):
        """x [T,d] fp32 attention-module input (post input_layernorm). -> (attn output [T,d], past)."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        from xenc.d13_read import _attend
        att = self.att[str(li)]
        hd = att.head_dim
        T = x.shape[0]
        q, gate = torch.chunk(att.q_proj(x).view(T, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(T, -1)
        q = att.q_norm(q).transpose(0, 1)
        k = att.k_norm(att.k_proj(x).view(T, -1, hd)).transpose(0, 1)
        v = att.v_proj(x).view(T, -1, hd).transpose(0, 1)
        P = 0 if past is None else past[0].shape[1]
        pos = torch.arange(self.n_mem + P, self.n_mem + P + T, device=x.device).view(1, 1, T).expand(3, 1, T)
        cos, sin = self.rot(x.unsqueeze(0), pos)
        qr, kr = apply_rotary_pos_emb(q.unsqueeze(0), k.unsqueeze(0), cos.float(), sin.float())
        qr, kr = qr[0], kr[0]
        Kown = kr if past is None else torch.cat([past[0], kr], 1)
        Vown = v if past is None else torch.cat([past[1], v], 1)
        Km, Vm = self.mem[li]
        out = _attend(qr, Km.float(), Vm.float(), Kown, Vown, att.scaling)
        return att.o_proj(out["o"] * torch.sigmoid(gate)), (Kown.detach(), Vown.detach())


class HybRuntime:
    """Wires DeepMem into a D13Rt (fit_d13.D13Rt): extends the encoder harvest to the 4B deep layers, wraps R.load so
    every operator also yields the deep memory, and patches the reader's deep attention modules. The patch follows the
    main reader's mode: off -> stock; student -> memory read with own past in R.past[('m', li)]; e2e -> with grad."""

    def __init__(self, rt, dm: DeepMem, cmp=None, enc_layers=(23, 27, 31), oracle: bool = False, log=print,
                 zcache: str = "", zdim: int = 0):
        """zcache (27B / cached-operator runtimes): the stored codes z of each document come from a directory written by
        scripts/cmp_zcache.py (<did>.pt = {"z": fp16 [N, d_max], "n": N}) instead of encoding live K/V; zdim < d_max
        keeps the first zdim PCA components (the rest = 0 = their mean), i.e. the d = zdim code of the same basis."""
        from deltaskill.backbone_utils import decoder_layers
        from xenc.d13_read import EncTeacher
        self.rt, self.dm, self.cmp, self.oracle = rt, dm, cmp, bool(oracle)
        self.zcache, self.zdim = str(zcache or ""), int(zdim or 0)
        self.enc_layers = [int(e) for e in enc_layers]
        assert self.oracle or cmp is not None, "HYB needs a compressor file (or oracle=True)"
        if cmp is not None:
            need = set(self.enc_layers) | {RKEY + li for li in dm.layers}
            assert set(cmp.layers) == need, (cmp.layers, need)
        if not self.zcache and not self.oracle:          # live: the encoder harvest also records the source layers
            old = sorted(rt.Tc.layers)
            rt.Tc.detach()
            rt.Tc = EncTeacher(rt.enc, sorted(set(old) | set(self.enc_layers)))
        rt._ops.clear()
        if self.oracle:
            self.Tr = EncTeacher(rt.reader, dm.layers)       # 9B K/V of a reader pass over the document prefix
            self._orc: dict = {}
        R = rt.R
        orig_load = R.load

        def load(op, n_op):
            orig_load(op, n_op)
            self.set_mem(op, n_op)

        R.load = load
        Ls = decoder_layers(rt.reader)
        self._orig = {}
        for li in dm.layers:
            m = Ls[li].self_attn
            self._orig[li] = m.forward
            m.forward = types.MethodType(self._forward(li), m)
        rt.hyb = self
        log(f"HYB on: deep reader layers {dm.layers} <- 4B {self.enc_layers} | "
            + ("ORACLE (the reader's true K/V of a reader pass over the document; diagnostic)" if self.oracle else
               (f"zcache {self.zcache} zdim {self.zdim or 'all'} | " if self.zcache else "") +
               f"code {cmp.spec} dims {cmp.dims} quant {cmp.quant} {cmp.bytes_per_token():.0f} B/token | "
               f"cross R2 {cmp.info.get('cross_r2')}") + f" | gates {[round(float(x), 4) for x in dm.g]}")

    @torch.no_grad()
    def _oracle_mem(self, did):
        if did not in self._orc:
            rt = self.rt
            ids = rt.pre(did)[: -rt.tail] if rt.tail else rt.pre(did)
            mode, rt.R.mode = rt.R.mode, "off"
            self.Tr.rec, self.Tr.kv_only = {}, True
            try:
                rt.reader(input_ids=torch.tensor([ids], device=rt.dev), use_cache=False, logits_to_keep=1)
                self._orc = {did: {li: (r["k"], r["v"]) for li, r in self.Tr.rec.items()}}
            finally:
                self.Tr.rec, self.Tr.kv_only = None, False
                rt.R.mode = mode
        return self._orc[did]

    def set_mem(self, op, n_op):
        dm = self.dm
        dm.n_mem = int(n_op)
        if self.oracle:
            did = op.get("_did")
            assert did is not None, "oracle HYB needs the document id in the operator (wrap of rt.operator)"
            dm.mem = dict(self._oracle_mem(did))
            return
        if self.zcache:
            did = op.get("_did")
            assert did is not None, "zcache HYB needs the document id in the operator (wrap of rt.operator)"
            b = torch.load(zpath(self.zcache, did), map_location="cpu", weights_only=False)
            assert int(b["n"]) == int(n_op), f"zcache: {did} has {b['n']} codes, operator has {n_op} columns"
            z = b["z"].to(self.rt.dev).float()
            if self.zdim and self.zdim < z.shape[1]:
                z = torch.cat([z[:, :self.zdim], torch.zeros_like(z[:, self.zdim:])], 1)
            assert z.shape[1] == self.cmp.dims[0], (z.shape, self.cmp.dims)
            from xenc.cmp_op import rope_tables
            cos, sin = rope_tables(self.cmp_rot, z.shape[0], z.device)
            new = self.cmp.decode([z], cos, sin, torch.bfloat16)
            dm.mem = {li: new[RKEY + li] for li in dm.layers}
            return
        src = {e: op[e] for e in self.enc_layers}
        src.update({RKEY + li: op[self.enc_layers[0]] for li in dm.layers})    # shape placeholders (decode targets)
        new = self.cmp(src, self.cmp_rot)
        dm.mem = {li: new[RKEY + li] for li in dm.layers}

    @property
    def cmp_rot(self):
        return self.rt.itf.rot

    def _forward(self, li):
        H = self

        def forward(mod, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
            out, w = H._orig[li](hidden_states, position_embeddings, attention_mask,
                                 past_key_values=past_key_values, **kwargs)
            R, dm = H.rt.R, H.dm
            if R.mode not in ("student", "e2e") or li not in dm.mem:
                return out, w
            x = hidden_states[0].float()
            if R.mode == "student":
                y, past = dm.read(li, x, R.past.get(("m", li)))
                R.past[("m", li)] = past
            elif getattr(H, "ckpt", False) and torch.is_grad_enabled():   # opt-in: recompute the deep read in backward
                from torch.utils.checkpoint import checkpoint
                y = checkpoint(lambda x_: dm.read(li, x_)[0], x, use_reentrant=False)
            else:
                y, _ = dm.read(li, x)
            g = dm.g[dm._gi[li]]
            o = out[0].float()
            return (o + g * (y - o)).to(out.dtype).unsqueeze(0), w

        return forward

    def wrap_operator_ids(self):
        """oracle: tag each operator dict with its document id (the reader pass needs the document)."""
        rt = self.rt
        orig = rt.operator

        def operator(did):
            op, n, cache = orig(did)
            op["_did"] = did
            return op, n, cache

        rt.operator = operator


def zpath(root: str, did: str) -> str:
    import hashlib
    import os
    return os.path.join(root, hashlib.md5(did.encode()).hexdigest()[:2], did.replace("/", "_") + ".pt")


def save_hyb(H: HybRuntime, path: str, extra: dict | None = None):
    import os
    from xenc.cmp_op import save_cmp
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"format": "xenc.cmp_hyb.v1", "layers": H.dm.layers, "enc_layers": H.enc_layers,
                "g": H.dm.g.detach().cpu(), "extra": extra or {}}, path + ".tmp")
    os.replace(path + ".tmp", path)
    if H.cmp is not None:
        save_cmp(H.cmp, path.replace(".pt", "_cmp.pt"), extra)


def build_hyb(rt, cmp_path: str = "", hyb_path: str = "", oracle: bool = False, g: float | None = None, log=print,
              layers=(23, 27, 31), enc_layers=(23, 27, 31), zcache: str = "", zdim: int = 0):
    """cmp_path: the closed-form / trained compressor (6-layer spec); hyb_path: a saved HYB state (gates + its _cmp file).
    g: override the gates (e.g. 1.0 for the oracle diagnostic)."""
    from xenc.cmp_op import load_cmp
    gv = None
    if hyb_path:
        blob = torch.load(hyb_path, map_location="cpu", weights_only=False)
        layers, enc_layers, gv = blob["layers"], blob["enc_layers"], blob["g"]
        cmp_path = cmp_path or hyb_path.replace(".pt", "_cmp.pt")
    dm = DeepMem(rt.reader, layers).to(rt.dev)
    if gv is not None:
        dm.g.data.copy_(gv)
    if g is not None:
        dm.g.data.fill_(float(g))
    cmp = None if oracle else load_cmp(cmp_path).to(rt.dev)
    H = HybRuntime(rt, dm, cmp, enc_layers, oracle=oracle, log=log, zcache=zcache, zdim=zdim)
    if oracle or zcache:
        H.wrap_operator_ids()
    return H


def detach_hyb(H: HybRuntime):
    from deltaskill.backbone_utils import decoder_layers
    Ls = decoder_layers(H.rt.reader)
    for li, f in H._orig.items():
        Ls[li].self_attn.forward = f
    H._orig.clear()


__all__ = ["DeepMem", "HybRuntime", "build_hyb", "save_hyb", "detach_hyb", "RKEY", "zpath"]
