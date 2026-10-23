"""D25: layer-role probes for smart layer injection (docs/SMART_INJECTION.md step 1). Read-only, no training.

  LayerTap   forward hooks on every decoder block; keeps the residual stream at the LAST position of each forward
             ([n_layers, d], fp32) -> logit lens (D2) and the per-layer conflict direction (D6).
  lens_logits  final RMSNorm + lm_head applied to tapped residuals, restricted to a set of token ids.
  AttnTap    patches the full-attention modules: records post-RoPE keys of every forward (prefix pass + suffix pass,
             so a document prefix can be prefilled once and its cache deep-copied per question) and, when
             `want_q`, the post-RoPE queries of the current forward -> per-layer attention mass of the question rows
             over token segments (D3). Same arithmetic as the stock eager attention (tests/test_d25.py).
  commit_depth / first_stable  first layer from which a per-layer margin stays > 0 up to the last layer.
"""

from __future__ import annotations

import types

import torch

import xenc  # noqa: F401


def _final_norm(model):
    for path in ("model.norm", "model.language_model.norm", "language_model.model.norm", "model.model.norm"):
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise AttributeError("final norm not found")


class LayerTap:
    """h[l] = output of decoder block l at the last position of the most recent forward (fp32, CPU-free device)."""

    def __init__(self, model):
        from deltaskill.backbone_utils import decoder_layers
        self.layers = decoder_layers(model)
        self.h: dict = {}
        self.on = True
        self._hs = [blk.register_forward_hook(self._mk(i)) for i, blk in enumerate(self.layers)]

    def _mk(self, i):
        def hook(mod, inp, out):
            if self.on:
                t = out[0] if isinstance(out, tuple) else out
                self.h[i] = t[0, -1].detach().float()
        return hook

    def stack(self) -> torch.Tensor:
        return torch.stack([self.h[i].to(self.h[0].device) for i in range(len(self.layers))])

    def detach(self):
        for h in self._hs:
            h.remove()
        self._hs = []


@torch.no_grad()
def lens_logits(model, H: torch.Tensor, ids) -> torch.Tensor:
    """H [L, d] residuals -> [L, len(ids)] logits of final_norm(H) @ W_out[ids]^T (logit lens)."""
    norm = _final_norm(model)
    W = model.get_output_embeddings().weight
    idx = torch.tensor(list(ids), device=W.device)
    x = norm(H.to(device=W.device, dtype=W.dtype))
    return (x @ W[idx].t()).float()


def first_stable(margin) -> int:
    """Smallest l such that margin[l'] > 0 for every l' >= l (len(margin) if the last layer is <= 0)."""
    m = list(margin)
    L = len(m)
    k = L
    for i in range(L - 1, -1, -1):
        if m[i] > 0:
            k = i
        else:
            break
    return k


class AttnTap:
    """Patches the full-attention modules of `model` (Qwen3.5 gated attention). rec[li] = {"k": [Hkv,T,hd] post-RoPE,
    accumulated over forwards since reset_k(), "q": [Hq,Tq,hd] post-RoPE of the latest forward (want_q only)}."""

    def __init__(self, model, layers):
        from deltaskill.backbone_utils import decoder_layers
        L = decoder_layers(model)
        self.layers = sorted(int(x) for x in layers)
        self.mods = {li: L[li].self_attn for li in self.layers}
        self.k: dict = {}
        self.q: dict = {}
        self.on = True
        self.want_q = False
        self._orig = {}
        for li, m in self.mods.items():
            self._orig[li] = m.forward
            m.forward = types.MethodType(_tap_forward(self, li), m)

    def reset_k(self):
        self.k, self.q = {}, {}

    def snapshot(self):
        return {li: v for li, v in self.k.items()}

    def restore(self, snap):
        self.k = dict(snap)
        self.q = {}

    def detach(self):
        for li, f in self._orig.items():
            self.mods[li].forward = f
        self._orig.clear()

    @torch.no_grad()
    def probs(self, li) -> torch.Tensor:
        """softmax over all recorded key columns for the rows of the latest forward (causal within them):
        [Hkv, G, Tq, C]."""
        from xenc.d13_read import teacher_logits
        q = self.q[li]
        K = self.k[li]
        Tq = q.shape[1]
        s = teacher_logits(q, K[:, :-Tq], K[:, -Tq:], self.mods[li].scaling)
        return torch.softmax(s, -1)


def _tap_forward(T: AttnTap, li: int):
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        if T.on:
            shp = hidden_states.shape[:-1]
            hs = (*shp, -1, self.head_dim)
            with torch.no_grad():
                k = self.k_norm(self.k_proj(hidden_states).view(hs)).transpose(1, 2)
                q = None
                if T.want_q:
                    q, _ = torch.chunk(self.q_proj(hidden_states).view(*shp, -1, self.head_dim * 2), 2, dim=-1)
                    q = self.q_norm(q.reshape(hs)).transpose(1, 2)
                cos, sin = position_embeddings
                qq, kk = apply_rotary_pos_emb(q if q is not None else k, k, cos, sin)
                kk = kk[0].detach()
                T.k[li] = kk if li not in T.k else torch.cat([T.k[li], kk], 1)
                if T.want_q:
                    T.q[li] = qq[0].detach().float()
        return type(self).forward(self, hidden_states, position_embeddings, attention_mask,
                                  past_key_values=past_key_values, **kwargs)

    return forward


def segment_mass(p: torch.Tensor, segs: dict) -> dict:
    """p [Hkv,G,Tq,C]; segs {name: bool[C] or (a, b) column range} -> {name: [Tq] mean-over-heads mass}."""
    out = {}
    C = p.shape[-1]
    for name, s in segs.items():
        if isinstance(s, tuple):
            m = torch.zeros(C, dtype=torch.bool, device=p.device)
            m[s[0]:s[1]] = True
        else:
            m = s.to(p.device)
            if m.shape[0] < C:
                m = torch.cat([m, torch.zeros(C - m.shape[0], dtype=torch.bool, device=p.device)])
        out[name] = (p * m.view(1, 1, 1, C)).sum(-1).mean((0, 1))
    return out


__all__ = ["LayerTap", "lens_logits", "first_stable", "AttnTap", "segment_mass"]
