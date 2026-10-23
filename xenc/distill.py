"""D9: dense distillation from the text-in-context teacher into the operator student (interface only).

Registration so far supervises ~6 answer tokens (KL + CE). This module adds two DENSE targets taken from
the SAME teacher forward (reader with the document in its prompt) that already produces the answer
logits, so the interface (A, B, gates, gains) gets a signal for *where to read* and *what to write*:

1. Attention-pattern distillation (teaches A to address). At each injection layer ℓ the teacher's
   softmax attention of the question's last prompt position over the DOCUMENT tokens is recorded at the
   full-attention layers in the window (ℓ_prev, ℓ] (Qwen3.5 hybrid: 3,7,11,…; linear-attention layers
   have no attention distribution), summed over heads (mass-weighted, GQA handled by repeating KV
   heads), mapped token -> operator slot (char offsets; same tokenizer; chunk starts from the cache) and
   renormalised over slots. Loss = KL(teacher ‖ student route×slot read attention) — the student term is
   `LibraryInjector.attn_terms` (D2, with grad). A model without softmax attention modules (the TinyLM
   test fake) falls back to hidden-similarity attention at the block output.
2. Residual-state distillation (teaches B / gates / gains to write). The residual after each injection
   block at the last m+1 positions (last prompt position + teacher-forced answer tokens; aligned because
   ctx_prompt and q_prompt share the suffix) of teacher h^T and student h^S (operator loaded), both
   RMS-normalised (scale is not the target). Default loss = ‖n(h^S) − n(h^T)‖² / ‖n(h^0) − n(h^T)‖²
   (h^0 = same student forward with no operator; denominator detached): the fraction of the context's
   effect on the residual stream that the branch has NOT reproduced (1.0 = no better than no skill).

Hooks never modify activations (the capture only stores references / no-grad copies), so with both
weights at 0 registration is bit-identical to before (no Distiller is built).

    dist = Distiller(model, tok, hub, inject_layers, attn_w=1.0, hid_w=1.0)
    dist.prepare(key, sysp, ctx_prompt_text, doc_text, lib, chunk_tokens)   # slot map (cached per key)
    with dist.teacher(T, m): teacher_logits(...)                              # captures attn + hidden
    dist.baseline(model, sids, m)                                             # h^0 (operator unloaded)
    inj.training_attn = True; inj.attn_pos = -(m+1)
    with dist.student(): student_logits(...)
    terms = dist.losses(inj)       # {"attn": tensor|None, "hid": tensor|None, + float diagnostics}
"""

from __future__ import annotations

import bisect
import contextlib
import sys

import torch

import xenc  # noqa: F401
from deltaskill.backbone_utils import decoder_layers
from deltaskill.derive import rms_norm


# ============================================================================ token -> slot mapping
def _offsets(tok, text: str) -> list[tuple[int, int]]:
    from xenc.gold_slots import _token_offsets
    return _token_offsets(tok, text)


def prompt_text(tok, hub: str, system: str | None, user: str) -> str:
    """The exact string whose tokenization is `xenc.backbone.build_chat(tok, hub, system, user)`."""
    if not hasattr(tok, "apply_chat_template"):
        return f"{system or ''}\n\n{user}"
    from deltaskill.backbone_utils import chat_template_kwargs
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **chat_template_kwargs(hub))


def doc_slot_map(prompt: str, prompt_offsets, doc_text: str, doc_offsets, lib, chunk_tokens: int,
                 doc_char_start: int | None = None):
    """Map the prompt tokens that lie inside the document to Library slots.

    prompt_offsets: char offsets of the teacher prompt's tokens; doc_offsets: char offsets of the
    document's own (whole-document) tokenization, which is what the cache chunks index
    (`chunk_meta[c]["start"/"n_tokens"]`). A prompt token overlapping k doc tokens spreads its mass 1/k.
    -> (tpos [P] long, slot [P] long (flat c*N+n), w [P] float, C, N) or None if the doc is not found."""
    ds = prompt.find(doc_text) if doc_char_start is None else doc_char_start
    if ds < 0 or not doc_text:
        return None
    de = ds + len(doc_text)
    C, N = next(iter(lib.K.values())).shape[:2]
    starts = [int(m.get("start", c * chunk_tokens)) for c, m in enumerate(lib.chunk_meta)]
    ntoks = [int(m.get("n_tokens", chunk_tokens)) for m in lib.chunk_meta]
    dends = [b for _, b in doc_offsets]

    def slot_of(j: int) -> int | None:
        c = bisect.bisect_right(starts, j) - 1
        if c < 0:
            return None
        n = j - starts[c]
        if n >= min(ntoks[c], N):
            return None
        return c * N + n

    tpos, slot, w = [], [], []
    for t, (a, b) in enumerate(prompt_offsets):
        if b <= ds or a >= de or b <= a:
            continue
        a, b = max(a, ds) - ds, min(b, de) - ds
        j = bisect.bisect_right(dends, a)          # first doc token whose end > a
        hit = []
        while j < len(doc_offsets) and doc_offsets[j][0] < b:
            s = slot_of(j)
            if s is not None:
                hit.append(s)
            j += 1
        for s in hit:
            tpos.append(t)
            slot.append(s)
            w.append(1.0 / len(hit))
    if not tpos:
        return None
    return (torch.tensor(tpos, dtype=torch.long), torch.tensor(slot, dtype=torch.long),
            torch.tensor(w, dtype=torch.float32), int(C), int(N))


def teacher_slot_map(tok, hub: str, system: str | None, user: str, doc_text: str, lib, chunk_tokens: int,
                     n_prompt_ids: int | None = None):
    """doc_slot_map for the teacher prompt build_chat(tok, hub, system, user). The rendered prompt is
    re-tokenized with offsets; if its length disagrees with the real prompt ids (n_prompt_ids) the map
    is refused (None) rather than silently shifted."""
    text = prompt_text(tok, hub, system, user)
    offs = _offsets(tok, text)
    if n_prompt_ids is not None and len(offs) != int(n_prompt_ids):
        return None
    k = text.find("## Document\n" + doc_text)
    ds = k + len("## Document\n") if k >= 0 else None
    return doc_slot_map(text, offs, doc_text, _offsets(tok, doc_text), lib, chunk_tokens, doc_char_start=ds)


def slots_from_token_attn(tok_attn: torch.Tensor, smap) -> torch.Tensor | None:
    """tok_attn [T] (teacher attention of one query over all positions) -> [C*N] distribution over slots
    restricted to the document (renormalised). None if the document gets no mass."""
    tpos, slot, w, C, N = smap
    a = tok_attn.float().to(tpos.device)[tpos] * w
    out = torch.zeros(C * N, dtype=torch.float32).index_add_(0, slot, a)
    z = out.sum()
    if not torch.isfinite(z) or float(z) <= 0:
        return None
    return out / z


# ============================================================================ teacher attention
def _attn_module(block):
    m = getattr(block, "self_attn", None)
    if m is None:
        return None
    return m if all(hasattr(m, k) for k in ("q_proj", "k_proj")) else None


def attention_windows(model, inject_layers) -> dict[int, list[int]]:
    """Injection layer -> softmax-attention layers in (previous injection layer, layer]. Empty list =
    no softmax attention in the window (fall back to hidden similarity at the block output)."""
    blocks = decoder_layers(model)
    full = [i for i, b in enumerate(blocks) if _attn_module(b) is not None and hasattr(_attn_module(b), "q_norm")]
    out, prev = {}, -1
    for li in sorted(int(x) for x in inject_layers):
        out[li] = [i for i in full if prev < i <= li]
        prev = li
    return out


def _query_attention(mod, hidden: torch.Tensor, pos_emb, qpos: int) -> torch.Tensor:
    """Softmax attention of query position `qpos` over keys [0, qpos], summed over heads -> [T] (zeros after
    qpos). Recomputes q (one row) and k (all rows) with the module's own projections / norms / RoPE
    (Qwen3/3.5 layout; gated q_proj detected from its width)."""
    cfg = mod.config
    hd = int(getattr(mod, "head_dim"))
    nh = int(cfg.num_attention_heads)
    nkv = int(cfg.num_key_value_heads)
    x = hidden[:1]
    T = x.shape[1]
    qp = mod.q_proj(x[:, qpos:qpos + 1])
    if qp.shape[-1] == 2 * nh * hd:                   # Qwen3.5 gated attention: (query, gate) per head
        qp = qp.view(1, 1, nh, 2 * hd)[..., :hd]
    q = mod.q_norm(qp.reshape(1, 1, nh, hd)).transpose(1, 2)            # [1,nh,1,hd]
    k = mod.k_norm(mod.k_proj(x).view(1, T, nkv, hd)).transpose(1, 2)    # [1,nkv,T,hd]
    cos, sin = pos_emb
    rope = getattr(sys.modules[type(mod).__module__], "apply_rotary_pos_emb")
    k = rope(k, k, cos, sin)[1]
    q = rope(q, q, cos[:, qpos:qpos + 1], sin[:, qpos:qpos + 1])[0]
    k = k.repeat_interleave(nh // nkv, dim=1)                            # GQA
    s = (q.float() @ k.float().transpose(-1, -2)).squeeze(2) * float(getattr(mod, "scaling", hd ** -0.5))
    s[..., qpos + 1:] = float("-inf")
    p = torch.softmax(s, -1)                                             # [1,nh,T]
    return p[0].sum(0)                                                   # mass-weighted head sum


# ============================================================================ distiller
class Distiller:
    def __init__(self, model, tok, hub: str, inject_layers, attn_w: float = 0.0, hid_w: float = 0.0,
                 hid_ratio: bool = True, hid_layer_w=None, sim_temp: float = 1.0):
        self.model, self.tok, self.hub = model, tok, hub
        self.layers = sorted(int(x) for x in inject_layers)
        self.attn_w, self.hid_w, self.hid_ratio = float(attn_w), float(hid_w), bool(hid_ratio)
        lw = [float(x) for x in (hid_layer_w or [1.0] * len(self.layers))]
        if len(lw) != len(self.layers):
            raise ValueError(f"hid layer weights {lw} for layers {self.layers}")
        self.hid_layer_w = dict(zip(self.layers, lw))
        self.sim_temp = float(sim_temp)
        self.windows = attention_windows(model, self.layers)
        self.blocks = decoder_layers(model)
        self._maps: dict = {}
        self.stats = {"attn_items": 0, "attn_skipped": 0}
        self._reset()

    def _reset(self):
        self.smap = None
        self.t_attn: dict = {}        # li -> [C*N] teacher slot distribution
        self.t_hid: dict = {}         # li -> [k, d] (no grad)
        self.b_hid: dict = {}         # li -> [k, d] baseline (no grad)
        self.s_out: dict = {}         # li -> student block output (reference, with grad)
        self.k = 0

    # ------------------------------------------------------------------ per item
    def prepare(self, key, system, user: str, doc_text: str, lib, chunk_tokens: int, n_prompt_ids: int):
        """Build (cached) the teacher-prompt -> slot map for this item. Must be called before teacher()."""
        self._reset()
        if not self.attn_w:
            return
        if key not in self._maps:
            self._maps[key] = teacher_slot_map(self.tok, self.hub, system, user, doc_text, lib, chunk_tokens,
                                               n_prompt_ids=n_prompt_ids)
        self.smap = self._maps[key]

    @contextlib.contextmanager
    def teacher(self, T: int, m: int):
        """Capture (no grad) teacher hidden states after every injection block (last m+1 positions) and the
        attention of the last prompt position (T-(m+1)) over all positions, per injection layer."""
        self.k = m + 1
        qpos = T - (m + 1)
        handles, tok_attn = [], {}
        want_attn = bool(self.attn_w) and self.smap is not None
        try:
            for li in self.layers:
                if want_attn:
                    for la in self.windows.get(li, []):
                        mod = _attn_module(self.blocks[la])

                        def pre(module, args, kwargs, la=la):
                            hs = kwargs.get("hidden_states", args[0] if args else None)
                            pe = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
                            with torch.no_grad():
                                tok_attn[la] = _query_attention(module, hs, pe, qpos).cpu()

                        handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))

                def post(module, args, output, li=li):
                    h = output[0] if isinstance(output, tuple) else output
                    with torch.no_grad():
                        self.t_hid[li] = h[0, -self.k:].detach().float().clone()
                        if want_attn and not self.windows.get(li):        # fallback: hidden similarity
                            hn = rms_norm(h[0].float())
                            s = (hn[:qpos + 1] @ hn[qpos]) / (hn.shape[-1] ** 0.5) * self.sim_temp
                            p = torch.zeros(h.shape[1], dtype=torch.float32, device=h.device)
                            p[:qpos + 1] = torch.softmax(s, -1)
                            tok_attn[("sim", li)] = p.cpu()

                handles.append(self.blocks[li].register_forward_hook(post))
            yield self
        finally:
            for hd in handles:
                hd.remove()
        if want_attn:
            for li in self.layers:
                src = self.windows.get(li) or []
                ta = [tok_attn[la] for la in src if la in tok_attn] if src else (
                    [tok_attn[("sim", li)]] if ("sim", li) in tok_attn else [])
                if not ta:
                    continue
                dist = slots_from_token_attn(torch.stack(ta).sum(0), self.smap)
                if dist is not None:
                    self.t_attn[li] = dist

    @torch.no_grad()
    def baseline(self, ids: torch.Tensor, m: int):
        """h^0: the student prompt with NO operator (caller unloads the injector first)."""
        if not (self.hid_w and self.hid_ratio):
            return
        from xenc.losses import _kept_logits
        cap = {}
        hs = [self.blocks[li].register_forward_hook(
            lambda mod, a, o, li=li: cap.__setitem__(li, (o[0] if isinstance(o, tuple) else o)[0, -(m + 1):]
                                                     .detach().float().clone())) for li in self.layers]
        try:
            _kept_logits(self.model, ids, 1)
        finally:
            for h in hs:
                h.remove()
        self.b_hid = cap

    @contextlib.contextmanager
    def student(self):
        """Keep a REFERENCE to each injection block's (post-injection) output; no ops inside the hook, so
        gradient checkpointing recomputation is unaffected."""
        hs = []
        if self.hid_w:
            for li in self.layers:
                hs.append(self.blocks[li].register_forward_hook(
                    lambda mod, a, o, li=li: self.s_out.__setitem__(li, o[0] if isinstance(o, tuple) else o)
                    if li not in self.s_out else None))
        try:
            yield self
        finally:
            for h in hs:
                h.remove()

    # ------------------------------------------------------------------ losses
    def attn_loss(self, injector) -> torch.Tensor | None:
        terms = list(getattr(injector, "attn_terms", None) or [])
        if not self.t_attn or not terms:
            self.stats["attn_skipped"] += 1
            return None
        vals = []
        for li, aw in zip(sorted(injector.itf.inject_layers), terms):
            t = self.t_attn.get(int(li))
            if t is None:
                continue
            s = aw.float().reshape(-1)
            t = t.to(s.device)
            if t.numel() != s.numel():
                raise ValueError(f"teacher slots {t.numel()} != student slots {s.numel()} at layer {li}")
            kl = (t * (t.clamp_min(1e-12).log() - s.clamp_min(1e-12).log())).sum()
            vals.append(kl)
            with torch.no_grad():                         # running diagnostics (history["distill"])
                st = self.stats
                st["t_ent_sum"] = st.get("t_ent_sum", 0.0) + float(-(t * t.clamp_min(1e-12).log()).sum())
                st["s_ent_sum"] = st.get("s_ent_sum", 0.0) + float(-(s * s.clamp_min(1e-12).log()).sum())
                st["top20_sum"] = st.get("top20_sum", 0.0) + topk_overlap(t, s.detach(), 20)
                st["n_layer_terms"] = st.get("n_layer_terms", 0) + 1
        if not vals:
            self.stats["attn_skipped"] += 1
            return None
        self.stats["attn_items"] += 1
        dev = vals[0].device
        return torch.stack([v.to(dev) for v in vals]).mean()

    def hid_loss(self) -> tuple[torch.Tensor | None, float | None]:
        """-> (loss, raw) where raw = mean ‖n(h^S) − n(h^T)‖²/d (diagnostic, == 2(1 − cos))."""
        if not self.hid_w or not self.s_out:
            return None, None
        num, den, raws = [], [], []
        for li in self.layers:
            if li not in self.s_out or li not in self.t_hid:
                continue
            hS = rms_norm(self.s_out[li][0, -self.k:].float())
            hT = rms_norm(self.t_hid[li].to(hS.device))
            d2 = (hS - hT).pow(2).sum(-1).mean()
            raws.append(float(d2.detach()) / hS.shape[-1])
            if self.hid_ratio and li in self.b_hid:
                h0 = rms_norm(self.b_hid[li].to(hS.device))
                ref = (h0 - hT).pow(2).sum(-1).mean().detach().clamp_min(1e-3 * hS.shape[-1])   # floor: 0.1% of ‖n(h)‖²
                num.append(self.hid_layer_w[li] * d2 / ref)
            else:
                num.append(self.hid_layer_w[li] * d2 / hS.shape[-1])
            den.append(self.hid_layer_w[li])
        if not num:
            return None, None
        dev = num[0].device
        loss = torch.stack([v.to(dev) for v in num]).sum() / max(sum(den), 1e-9)
        return loss, sum(raws) / len(raws)

    def losses(self, injector) -> dict:
        out = {"attn": None, "hid": None, "hid_raw": None}
        if self.attn_w:
            out["attn"] = self.attn_loss(injector)
        if self.hid_w:
            out["hid"], out["hid_raw"] = self.hid_loss()
        return out

    def clear(self):
        self.s_out = {}
        self.t_hid, self.b_hid, self.t_attn = {}, {}, {}


def topk_overlap(t: torch.Tensor, s: torch.Tensor, k: int = 20) -> float:
    """|top-k(teacher) ∩ top-k(student)| / k over the flat slot distributions (probe diagnostic)."""
    k = min(k, t.numel())
    a = set(t.reshape(-1).topk(k).indices.tolist())
    b = set(s.reshape(-1).topk(k).indices.tolist())
    return len(a & b) / k


__all__ = ["prompt_text", "doc_slot_map", "teacher_slot_map", "slots_from_token_attn", "attention_windows",
           "Distiller", "topk_overlap"]
