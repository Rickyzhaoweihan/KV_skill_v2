"""Encoder/reader-aware injectors (subclass deltaskill's ExternalSkillInjector, never patch it).

XEncInjector     : load() accepts ρ-keyed skills (remapped to the reader's layers via for_reader),
                   validates d_s against the interface and that every reader layer is covered.
LibraryInjector  : reads a Library of C chunk operators with per-token routing over chunks
                   (summary keys S), optional top-k routing, a masked within-chunk read, and an
                   optional LayerSelector (agent [B], xenc/selector.py) that is reset at the start of
                   every forward (pre-hook on the first injection layer) and applied to each delta.

Bookkeeping is identical to ExternalSkillInjector: per layer, `mag_terms.append(mean ||delta||/||h||)`
after the position mask. LibraryInjector additionally exposes `last_route` (detached [B,T,C] of the
last hooked layer) and `route_terms` (list of (layer, w) with grad, appended only while grad is
enabled; the caller clears it each step, like `mag_terms`; `unload()` also clears both).

Sharded readers (device_map="auto" / "cuda:0+cuda:1"): injection layers can live on different GPUs.
Every tensor the branch touches — the skill / Library K, mask, S for layer li, the interface weights
(A, B, gate_w, gate_b, log_gain) and the selector — is brought to `h.device` via `_on` (pattern of
cross_model/xskill/shared_interface.py `_on`). `_on` is the identity (no copy, no `.to()` call)
when the devices already match, so the single-device path is bit-identical to deltaskill's.
Library/skill tensors are moved lazily ONCE per layer and cached in place (not per decode step);
interface weights are moved per call (autograd routes the grads back to the master copy).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import xenc  # noqa: F401
from deltaskill.derive import rms_norm
from deltaskill.injector import ExternalSkillInjector
from deltaskill.interface import ModelInterface
from deltaskill.skill import KVSkill
from xenc.depth_skill import for_reader
from xenc.library import Library


# ----------------------------------------------------------------------------- device helpers
def _on(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """`t` on `like.device` (dtype unchanged). Identity — same object — when already there."""
    return t if t.device == like.device else t.to(like.device)


def _linear(mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(mod, nn.Sequential):
        for m in mod:
            x = _linear(m, x)
        return x
    b = None if mod.bias is None else _on(mod.bias, x)
    return F.linear(x, _on(mod.weight, x), b)


def itf_query(itf, hn: torch.Tensor) -> torch.Tensor:
    if not isinstance(itf, ModelInterface):
        return itf.query(hn)
    q = _linear(itf.A, hn)
    return q / (q.norm(dim=-1, keepdim=True) + 1e-6)


def itf_gate(itf, hn: torch.Tensor, li: int) -> torch.Tensor:
    if not isinstance(itf, ModelInterface):
        return itf.gate(hn, li)
    p = itf._pos[li]
    return torch.sigmoid(hn @ _on(itf.gate_w, hn) + _on(itf.gate_b, hn)[p]).unsqueeze(-1)


def itf_gain(itf, li: int, like: torch.Tensor) -> torch.Tensor:
    if not isinstance(itf, ModelInterface):
        return _on(itf.gain(li), like)
    return _on(itf.log_gain, like)[itf._pos[li]].exp()


def itf_write(itf, r: torch.Tensor) -> torch.Tensor:
    if not isinstance(itf, ModelInterface):
        return itf.write(r)
    return _linear(itf.B, r)


class XEncInjector(ExternalSkillInjector):
    def load(self, skill: KVSkill) -> None:
        if skill.meta.get("keyed_by") == "rho":
            skill = for_reader(skill, self.model)
        if skill.d_s != self.itf.d_s:
            raise ValueError(f"skill d_s={skill.d_s} (encoder) != interface d_s={self.itf.d_s}")
        missing = set(self.itf.inject_layers) - set(skill.K)
        if missing:
            raise KeyError(f"skill has no operator for reader layers {sorted(missing)}")
        super().load(skill)

    def _skill_kv(self, li: int, like: torch.Tensor):
        """Skill K/V for layer li on like.device (moved once, cached in the loaded skill)."""
        K, V = self.skill.K[li], self.skill.V[li]
        if K.device != like.device:
            K = self.skill.K[li] = K.to(like.device)
        if V.device != like.device:
            V = self.skill.V[li] = V.to(like.device)
        return K, V

    def _branch(self, h: torch.Tensor, li: int) -> torch.Tensor:
        """ExternalSkillInjector._branch, made device-aware for sharded readers (identical math)."""
        hn = rms_norm(h.float())
        q = itf_query(self.itf, hn)                       # [B,T,d_s]
        K, V = self._skill_kv(li, hn)
        a = torch.einsum("btd,rd->btr", q, K)
        if self.read_softmax:
            a = torch.softmax(a * self.read_temp, -1)
        else:
            a = a / (K.shape[0] ** 0.5)
        r = torch.einsum("btr,rd->btd", a, V)
        gcal = 1.0
        if self.calib is not None and li in self.calib:
            r = r * _on(self.calib[li]["scale"], r)
            gcal = self.calib[li]["gain"]
            if isinstance(gcal, torch.Tensor):
                gcal = _on(gcal, r)
        g = itf_gate(self.itf, hn, li)
        return self.gain_scale * itf_gain(self.itf, li, hn) * gcal * g * itf_write(self.itf, r)


class LibraryInjector(XEncInjector):
    def __init__(self, model, interface, read_softmax: bool = True, read_temp: float = 12.0,
                 route_temp: float = 10.0, top_k: int = 0, selector=None, read_fp32: bool = True):
        self.lib: Library | None = None
        self.route_temp = route_temp
        self.top_k = int(top_k)
        self.selector = selector
        self.last_route: torch.Tensor | None = None
        self.record_attn = False            # when True, keep per-layer slot attention of the LAST position
        self.last_attn: dict = {}            # li -> [C,N] (route x slot attention at last position, detached)
        # D6 diagnostic: forced addressing. None (default) = computed attention. Else a [C,N] distribution
        # (or {li: [C,N]}) that REPLACES route x slot attention at every position; with force_norm_match
        # the forced read is rescaled per token to the norm of the computed read (magnitude-matched).
        self.force_attn = None
        self.force_norm_match = False
        # D2 slot-level retrieval aux: when training_attn, keep aw[0, attn_pos] WITH grad per layer
        self.training_attn = False
        self.attn_pos: int = -1              # position whose (route x slot) attention is kept (last prompt pos)
        self.attn_terms: list = []           # list of [C,N] tensors (with grad), one per hooked layer/forward
        self.route_terms: list = []
        super().__init__(model, interface, read_fp32=read_fp32, read_softmax=read_softmax,
                         read_temp=read_temp)

    # ------------------------------------------------------------------ load / unload
    def load(self, skill: KVSkill) -> None:
        self.lib = None
        super().load(skill)

    def load_library(self, lib: Library) -> None:
        if lib.d_s != self.itf.d_s:
            raise ValueError(f"library d_s={lib.d_s} (encoder) != interface d_s={self.itf.d_s}")
        missing = set(self.itf.inject_layers) - set(lib.K)
        if missing:
            raise KeyError(f"library has no operator for reader layers {sorted(missing)}")
        dtype = torch.float32 if self.read_fp32 else next(self.model.parameters()).dtype
        self.skill = None
        self.lib = lib.to(self.model.device, dtype)   # per-layer device fix-up happens lazily in _lib_layer

    def unload(self) -> None:
        """Unload and clear the per-forward bookkeeping (mag_terms / route_terms / last_route), so long
        no-grad generation loops (eval) do not accumulate one scalar per layer per decode step."""
        self.skill = None
        self.lib = None
        self.mag_terms.clear()
        self.route_terms.clear()
        if hasattr(self, "attn_terms"):
            self.attn_terms.clear()
        self.last_route = None

    # ------------------------------------------------------------------ hooks
    def _active(self) -> bool:
        return self.enabled and (self.skill is not None or self.lib is not None)

    def _attach(self) -> None:
        from deltaskill.backbone_utils import decoder_layers
        blocks = decoder_layers(self.model)
        first = self.itf.inject_layers[0]

        def pre(mod, args):                     # start of every forward (incl. each decode step)
            if self.selector is not None:
                self.selector.reset()

        self._handles.append(blocks[first].register_forward_pre_hook(pre))
        for li in self.itf.inject_layers:
            def hook(mod, args, output, li=li):
                if not self._active():
                    return output
                is_tuple = isinstance(output, tuple)
                h = output[0] if is_tuple else output
                delta = self._branch(h, li).to(h.dtype)
                mask = self._pos_mask(h)
                if mask is not None:
                    delta = delta * mask
                rel = delta.float().norm(dim=-1) / (h.float().norm(dim=-1) + 1e-6)
                self.mag_terms.append(rel.mean())
                h = h + delta
                return (h,) + output[1:] if is_tuple else h

            self._handles.append(blocks[li].register_forward_hook(hook))

    # ------------------------------------------------------------------ branch
    def _lib_layer(self, li: int, like: torch.Tensor):
        """Library (K, mask, S) of layer li on like.device; moved once and cached in the Library."""
        lib = self.lib
        if lib.K[li].device != like.device:
            lib.K[li] = lib.K[li].to(like.device)
            lib.mask[li] = lib.mask[li].to(like.device)
            lib.S[li] = lib.S[li].to(like.device)
        return lib.K[li], lib.mask[li], lib.S[li]

    def _lib_values(self, li: int, like: torch.Tensor) -> torch.Tensor:
        """Values of layer li: lib.V[li] when the Library is untied (harvest="attn"), else K (tied)."""
        lib = self.lib
        if getattr(lib, "V", None) is None:
            return self._lib_layer(li, like)[0]
        if lib.V[li].device != like.device:
            lib.V[li] = lib.V[li].to(like.device)
        return lib.V[li]

    def route(self, q: torch.Tensor, li: int) -> torch.Tensor:
        """Per-token routing weights over chunks [B,T,C]; sums to 1, exactly top_k nonzeros if set."""
        _, _, S = self._lib_layer(li, q)
        logit = torch.einsum("btd,cd->btc", q, S) * self.route_temp
        C = logit.shape[-1]
        if self.top_k and self.top_k < C:
            _, i = logit.topk(self.top_k, -1)
            keep = torch.zeros_like(logit, dtype=torch.bool).scatter(-1, i, True)
            logit = logit.masked_fill(~keep, float("-inf"))
        return torch.softmax(logit, -1)

    def _branch(self, h: torch.Tensor, li: int) -> torch.Tensor:
        if self.lib is None:                    # single-skill path (ExternalSkillInjector semantics)
            delta = super()._branch(h, li)
        else:
            hn = rms_norm(h.float())
            q = itf_query(self.itf, hn)                              # [B,T,d_s]
            K, M, _ = self._lib_layer(li, hn)                        # [C,N,d_s], [C,N]
            w = self.route(q, li)                                    # [B,T,C]
            a = torch.einsum("btd,cnd->btcn", q, K)
            if self.read_softmax:
                a = torch.softmax((a * self.read_temp).masked_fill(~M, float("-inf")), -1)
            else:
                n = M.sum(-1).to(a.dtype).clamp_min(1)               # valid slots per chunk
                a = a.masked_fill(~M, 0.0) / n.sqrt().view(1, 1, -1, 1)
            aw = a * w.unsqueeze(-1)                                 # [B,T,C,N]
            if self.force_attn is not None:                          # D6: oracle / forced addressing
                fa = self.force_attn.get(li) if isinstance(self.force_attn, dict) else self.force_attn
                fa = _on(fa, aw).to(aw.dtype)
                if fa.shape != aw.shape[-2:]:
                    raise ValueError(f"force_attn {tuple(fa.shape)} != library [C,N] {tuple(aw.shape[-2:])}")
                fa = fa.expand_as(aw)
                if self.force_norm_match:
                    Kf = K.reshape(-1, K.shape[-1])
                    r0 = aw.reshape(*aw.shape[:2], -1) @ Kf
                    r1 = fa.reshape(*fa.shape[:2], -1) @ Kf
                    scale = r0.norm(dim=-1, keepdim=True) / (r1.norm(dim=-1, keepdim=True) + 1e-6)
                    fa = fa * scale.unsqueeze(-1)
                aw = fa
            if self.record_attn:
                self.last_attn[li] = aw[0, -1].detach().float().cpu()
            if self.training_attn and torch.is_grad_enabled():
                self.attn_terms.append(aw[0, self.attn_pos].float())
            B_, T_, C_, N_ = aw.shape
            Vt = self._lib_values(li, hn)                            # lib.V if untied, else K (tied)
            r = aw.reshape(B_, T_, C_ * N_) @ Vt.reshape(C_ * N_, -1)  # -> [B,T,d_s]
            g = itf_gate(self.itf, hn, li)
            delta = self.gain_scale * itf_gain(self.itf, li, hn) * g * itf_write(self.itf, r)
            self.last_route = w.detach()
            if torch.is_grad_enabled():
                self.route_terms.append((li, w))
        if self.selector is not None:
            delta = self.selector.apply(delta, h, li)
        return delta


__all__ = ["XEncInjector", "LibraryInjector", "itf_query", "itf_gate", "itf_gain", "itf_write"]
