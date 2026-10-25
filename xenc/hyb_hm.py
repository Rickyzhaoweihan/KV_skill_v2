"""HYB-HM: an HM-style LEARNED writer for the deep-layer channel (agent CMP, 2026-10-25). Default off; new file.

Hub Memory (HM) stores one small code per document token, z_t = f(h_t) (a projection of the 4B writer's hidden state at
one depth), and the 9B reader consumes that code natively. HYB (xenc/cmp_hyb.py) already lets the reader's deep
full-attention layers read an extra per-token memory with their OWN attention, but its code is a FIXED PCA of the
encoder's K/V and the decoder starts from a closed-form ridge. HYB-HM replaces both by learned maps and removes the
PCA entirely:

    x_t   = the 4B encoder's input to the attention block of layer `hw_layer` (post input-layernorm hidden state, one
            row per document token; captured during the same prefill that writes the operator)
    z_t   = rmsnorm( W x_t )                 W: trainable (linear, or a 1-hidden-layer MLP with --hm_mlp H); z_t is what is
                                             STORED: d numbers per token, bf16 (2d bytes/token); no quantisation
    K^_t  = RoPE_t( k_norm( Uk_l z_t + bk_l ) )     the reader's OWN k_norm at layer l, column position t
    V^_t  = Uv_l z_t + bv_l                  Uv, bv ZERO-INITIALISED
    out_l = stock_l + g * o_proj( sigmoid(gate) * sum_{t<N} p_t V^_t ),  p = softmax over [memory ; own rows] (joint)

The memory is an ADDITIVE residual on the stock attention output (g is a FIXED constant 1; no learned gate), so with
V^ = 0 the model is EXACTLY the parent: step 0 == outputs/DS_max8k/d13_step6000.pt with no channel. The first gradient
reaches Uv/bv (V^ = Uv z), then the key decoder and the writer W as soon as V^ != 0. Layers default to the deep
full-attention layers 23/27/31 (--hm_layers adds more, e.g. 15,19, which the D13 read also uses; the two stack).

Cost per question: for each read layer, one [N, d] x [d, 1024] decode for K^ and for V^ and attention over N extra
columns; the reader never sees the document text.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from xenc.cmp_hyb import DeepMem


def _rms(z, eps=1e-6):
    return z * torch.rsqrt(z.float().pow(2).mean(-1, keepdim=True) + eps).to(z.dtype)


class HmDeepMem(DeepMem):
    """DeepMem with a learned writer and learned linear K/V decoders (additive residual read)."""

    def __init__(self, reader, layers=(23, 27, 31), d: int = 512, x_dim: int = 2560, mlp: int = 0, g: float = 1.0):
        super().__init__(reader, layers, 0.0)
        self.g.requires_grad_(False)                   # unused (DeepMem's gate); the HM read has a FIXED scale hm_g
        self.hm_g = float(g)
        self.d, self.x_dim, self.mlp = int(d), int(x_dim), int(mlp)
        a0 = self.att[str(self.layers[0])]
        self.Hkv, self.hd = a0.k_proj.out_features // a0.head_dim, a0.head_dim
        F = self.Hkv * self.hd
        if mlp:
            self.writer = nn.Sequential(nn.Linear(x_dim, mlp, bias=False), nn.GELU(), nn.Linear(mlp, d, bias=False))
        else:
            self.writer = nn.Linear(x_dim, d, bias=False)
        self.dk = nn.ModuleDict({str(li): nn.Linear(d, F) for li in self.layers})
        self.dv = nn.ModuleDict({str(li): nn.Linear(d, F) for li in self.layers})
        for li in self.layers:                         # V decoder zero => the channel adds exactly 0 at step 0
            nn.init.zeros_(self.dv[str(li)].weight)
            nn.init.zeros_(self.dv[str(li)].bias)
            nn.init.zeros_(self.dk[str(li)].bias)
        self.x = None
        self._z0 = self._zg = None
        self._dec: dict = {}

    # ---------------------------------------------------------------- parameters / storage
    def trainable(self):
        return list(self.writer.parameters()) + list(self.dk.parameters()) + list(self.dv.parameters())

    def trainable_state(self):
        return {"writer": self.writer.state_dict(), "dk": self.dk.state_dict(), "dv": self.dv.state_dict()}

    def load_trainable(self, st):
        self.writer.load_state_dict(st["writer"])
        self.dk.load_state_dict(st["dk"])
        self.dv.load_state_dict(st["dv"])

    def bytes_per_token(self) -> int:
        return 2 * self.d                              # z_t in bf16: the only per-token object that is stored

    # ---------------------------------------------------------------- write / decode
    def write(self, x, chunk: int = 16384):
        """x [N, x_dim] (bf16/fp32) -> z [N, d] bf16 (storage precision), differentiable w.r.t. the writer."""
        dev = self.writer.parameters().__next__().device
        out = []
        for a in range(0, x.shape[0], chunk):
            xc = x[a:a + chunk].to(dev).float()
            out.append(_rms(self.writer(xc)).to(torch.bfloat16))
        return torch.cat(out, 0)

    def set_x(self, x, n_mem: int):
        """Load a document's writer inputs. z is computed lazily: without grad (eval / generation, cached) or with grad
        (training; once per loaded document, shared by the read layers)."""
        assert x.shape[0] == int(n_mem), (tuple(x.shape), n_mem)
        self.n_mem = int(n_mem)
        self.x = x
        self._z0 = self._zg = None
        self._dec = {}

    @property
    def z(self):
        """The stored code of the loaded document (no grad)."""
        if self._z0 is None and self.x is not None:
            with torch.no_grad():
                self._z0 = self.write(self.x)
        return self._z0

    def z_grad(self):
        if self._zg is None:
            self._zg = self.write(self.x)
        return self._zg

    def decode(self, li, z=None):
        """-> (K^ post-RoPE [Hkv,N,hd], V^ [Hkv,N,hd]) fp32, at reader layer li."""
        from xenc.cmp_op import rope, rope_tables
        z = self.z if z is None else z
        N = z.shape[0]
        zb = z.to(torch.bfloat16)
        dk, dv = self.dk[str(li)], self.dv[str(li)]
        k = Fn.linear(zb, dk.weight.to(torch.bfloat16), dk.bias.to(torch.bfloat16)).float().view(N, self.Hkv, self.hd)
        v = Fn.linear(zb, dv.weight.to(torch.bfloat16), dv.bias.to(torch.bfloat16)).float().view(N, self.Hkv, self.hd)
        k = self.att[str(li)].k_norm(k).transpose(0, 1)
        cos, sin = rope_tables(self.rot, N, z.device)
        return rope(k, cos, sin), v.transpose(0, 1)

    def read_add(self, li, x, past=None, z=None):
        """x [T,d_model] fp32 (attention-block input). -> (additive residual [T,d_model], past)."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
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
        if torch.is_grad_enabled():
            Km, Vm = self.decode(li, self.z_grad() if z is None else z)
        else:                                          # generation / eval: decode once per loaded document
            if li not in self._dec:
                self._dec[li] = self.decode(li)
            Km, Vm = self._dec[li]
        Hq = qr.shape[0]
        Hkv = Km.shape[0]
        G = Hq // Hkv
        N = Km.shape[1]
        K = torch.cat([Km, Kown], 1)
        s = torch.einsum("kgtd,kcd->kgtc", qr.view(Hkv, G, T, hd), K) * att.scaling
        C = K.shape[1]
        col = torch.arange(C, device=x.device).view(1, C)
        row = torch.arange(T, device=x.device).view(T, 1)
        s = s.masked_fill((col > (C - T) + row).view(1, 1, T, C), float("-inf"))
        p = torch.softmax(s, -1)[..., :N]              # joint softmax; only the memory columns carry V^ in the residual
        o = torch.einsum("kgtc,kcd->kgtd", p, Vm).reshape(Hq, T, hd).transpose(0, 1).reshape(T, Hq * hd)
        y = att.o_proj(o * torch.sigmoid(gate))
        return y, (Kown.detach(), Vown.detach())


class HmRuntime:
    """Wires HmDeepMem into a D13Rt: captures the 4B hidden state at `hw_layer` during the operator prefill (cached with
    the operator in rt._ops under op['hm_x']), computes z on R.load, and patches the reader's deep attention modules."""

    def __init__(self, rt, dm: HmDeepMem, hw_layer: int, log=print):
        from deltaskill.backbone_utils import decoder_layers
        self.rt, self.dm, self.hw_layer = rt, dm, int(hw_layer)
        self.ckpt = False
        self._cap = None
        self._xs = []
        rt._ops.clear()
        Le = decoder_layers(rt.enc)
        self._hook = Le[self.hw_layer].self_attn.register_forward_pre_hook(self._capture, with_kwargs=True)
        orig_op = self._orig_op = rt.operator

        def operator(did):
            if did in rt._ops:
                return orig_op(did)
            self._xs = []
            self._cap = True
            try:
                res = orig_op(did)
            finally:
                self._cap = None
            xs, self._xs = self._xs, []
            assert len(xs) == 1, f"HM writer capture: {len(xs)} hidden-state blocks"
            res[0]["hm_x"] = xs[0]
            assert xs[0].shape[0] == res[1], (tuple(xs[0].shape), res[1])
            return res

        rt.operator = operator
        R = rt.R
        orig_load = self._orig_load = R.load

        def load(op, n_op):
            orig_load(op, n_op)
            dm.set_x(op["hm_x"].to(rt.dev), n_op)

        R.load = load
        Ls = decoder_layers(rt.reader)
        self._orig = {}
        for li in dm.layers:
            m = Ls[li].self_attn
            self._orig[li] = m.forward
            m.forward = types.MethodType(self._forward(li), m)
        rt.hm = self
        log(f"HYB-HM on: writer = 4B layer-{self.hw_layer} attention-block input ({dm.x_dim}) -> "
            f"{'MLP' + str(dm.mlp) if dm.mlp else 'linear'} -> rmsnorm -> z[{dm.d}] bf16 = {dm.bytes_per_token()} B/token | "
            f"reader layers {dm.layers} | fixed scale {dm.hm_g} | V-decoder zero-init (step 0 == parent) | trainable "
            f"{sum(p.numel() for p in dm.trainable()) / 1e6:.2f}M")

    def _capture(self, mod, args, kwargs):
        if self._cap:
            h = kwargs.get("hidden_states", args[0] if args else None)
            self._xs.append(h[0].detach().to(self.rt.edev))
        return None

    def _forward(self, li):
        H = self

        def forward(mod, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
            out, w = H._orig[li](hidden_states, position_embeddings, attention_mask,
                                 past_key_values=past_key_values, **kwargs)
            R, dm = H.rt.R, H.dm
            if R.mode not in ("student", "e2e") or dm.x is None:
                return out, w
            x = hidden_states[0].float()
            if R.mode == "student":
                y, past = dm.read_add(li, x, R.past.get(("m", li)))
                R.past[("m", li)] = past
            elif H.ckpt and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                y = checkpoint(lambda x_, z_: dm.read_add(li, x_, z=z_)[0], x, dm.z_grad(), use_reentrant=False)
            else:
                y, _ = dm.read_add(li, x)
            return (out[0].float() + dm.hm_g * y).to(out.dtype).unsqueeze(0), w

        return forward


def save_hm(H: HmRuntime, path: str, extra: dict | None = None):
    import os
    dm = H.dm
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"format": "xenc.hyb_hm.v1",
                "config": {"d": dm.d, "x_dim": dm.x_dim, "mlp": dm.mlp, "layers": dm.layers, "hw_layer": H.hw_layer,
                           "g": dm.hm_g},
                "state": {k: {kk: vv.detach().cpu() for kk, vv in v.items()} for k, v in dm.trainable_state().items()},
                "extra": extra or {}}, path + ".tmp")
    os.replace(path + ".tmp", path)


def build_hm(rt, path: str = "", d: int = 512, hw_layer: int = 23, layers=(23, 27, 31), mlp: int = 0, g: float = 1.0,
             log=print):
    """Fresh (path='') or from a saved state. x_dim comes from the encoder config."""
    blob = None
    if path:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        assert blob.get("format") == "xenc.hyb_hm.v1", blob.get("format")
        c = blob["config"]
        d, hw_layer, layers, mlp, g = c["d"], c["hw_layer"], c["layers"], c["mlp"], c["g"]
    cfg = rt.enc.config
    x_dim = int(getattr(getattr(cfg, "text_config", cfg), "hidden_size"))
    dm = HmDeepMem(rt.reader, layers, d, x_dim, mlp, g).to(rt.dev)
    if blob is not None:
        dm.load_trainable(blob["state"])
    return HmRuntime(rt, dm, hw_layer, log=log)


def detach_hm(H: HmRuntime):
    from deltaskill.backbone_utils import decoder_layers
    Ls = decoder_layers(H.rt.reader)
    for li, f in H._orig.items():
        Ls[li].self_attn.forward = f
    H._orig.clear()
    H._hook.remove()
    H.rt.operator, H.rt.R.load = H._orig_op, H._orig_load
    H.rt._ops.clear()


__all__ = ["HmDeepMem", "HmRuntime", "build_hm", "save_hm", "detach_hm"]
