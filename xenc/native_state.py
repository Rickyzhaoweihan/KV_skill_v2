"""D7 (phase 2): where does document content live in a hybrid (Gated-DeltaNet + full-attention) reader?

Qwen3.5 keeps two kinds of document memory after a prefill:
  * full-attention layers: the K/V cache (post-RoPE keys, values)  [1, n_kv, T, hd] per layer — grows with T;
  * Gated-DeltaNet (linear) layers: a recurrent state S = Σ decay·β v kᵀ per value head
    [1, n_v_heads, d_k, d_v] (exactly the paper's operator M = W Uᵀ, read linearly as S q) plus a short
    causal-conv state [1, conv_dim, kernel] — fixed size, independent of T.

    st = harvest_states(model, tok, doc_text)            # one prefill of the raw document tokens
    cache = build_prefix_cache(model, st, keep_kv=True, keep_state=True, keep_conv=True)
    text = greedy_continue(model, tok, hub, cache, prompt_ids, budget)   # continue with the question

keep_kv / keep_state / keep_conv select which memory the reader continues from. keep all == in-context
prefill of [doc; prompt] (exact prefix caching; RoPE positions continue at n_doc); keep none == the prompt
alone (closed book). Without KV the prompt starts at position 0 (RoPE is relative, so only distances
among the prompt tokens matter). `tile_states` maps states of a smaller encoder onto a reader's layer/head
grid (normalized depth, head tiling) as a zero-training cross-model probe.
"""

from __future__ import annotations

import torch

import xenc  # noqa: F401
from deltaskill.backbone_utils import decoder_layers


def _text_config(model):
    c = model.config
    return c.get_text_config() if hasattr(c, "get_text_config") else c


def layer_types(model) -> list[str]:
    return list(_text_config(model).layer_types)


def new_cache(model):
    from transformers import DynamicCache
    return DynamicCache(config=_text_config(model))


def _ids(tok, text: str) -> torch.Tensor:
    enc = tok(text, add_special_tokens=False)
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    return torch.as_tensor(ids).view(1, -1)


@torch.no_grad()
def harvest_states(model, tok, doc_text: str | None = None, ids: torch.Tensor | None = None,
                   max_tokens: int = 0) -> dict:
    """One prefill of the document; returns the reader's memory of it:
        {"recurrent": {li: [1,H_v,d_k,d_v]}, "conv": {li: [1,C,k]}, "kv": {li: (K [1,n_kv,T,hd], V)},
         "n_doc": T, "layer_types": [...]}   (tensors on the model's device, model dtype)."""
    if ids is None:
        ids = _ids(tok, doc_text)
    if max_tokens and ids.shape[1] > max_tokens:
        ids = ids[:, :max_tokens]
    dev = model.get_input_embeddings().weight.device
    cache = new_cache(model)
    model(input_ids=ids.to(dev), past_key_values=cache, use_cache=True, logits_to_keep=1)
    types = layer_types(model)
    out = {"recurrent": {}, "conv": {}, "kv": {}, "n_doc": int(ids.shape[1]), "layer_types": types}
    for li, t in enumerate(types):
        L = cache.layers[li]
        if t == "linear_attention":
            out["recurrent"][li] = L.recurrent_states.detach().clone()
            out["conv"][li] = L.conv_states.detach().clone()
        else:
            out["kv"][li] = (L.keys.detach().clone(), L.values.detach().clone())
    return out


def state_nbytes(st: dict) -> dict:
    nb = lambda ts: sum(t.numel() * t.element_size() for t in ts)
    return {"recurrent": nb(st["recurrent"].values()), "conv": nb(st["conv"].values()),
            "kv": nb([x for kv in st["kv"].values() for x in kv])}


def build_prefix_cache(model, st: dict | None, keep_kv: bool = True, keep_state: bool = True,
                       keep_conv: bool = True, state_scale: float = 1.0, state_layers=None, conv_layers=None,
                       kv_layers=None):
    """Fresh DynamicCache the reader continues from. Tensors are CLONED (generation mutates the cache).
    A linear layer is marked as having previous state iff keep_state or keep_conv (the dropped part is zero).
    state_scale multiplies every recurrent state (ablation: magnitude vs content). For KV on a SUBSET of the
    full-attention layers keep all KV here and wrap the decode in `PrefixMask` (positions stay consistent).
    state_layers / conv_layers: optional sets of linear layers whose recurrent / conv state is kept (others zero;
    a linear layer with neither starts fresh, as if it never saw the document).
    kv_layers: optional set of full-attention layers that receive the document K/V; the others stay EMPTY, so the
    caller must wrap the forward in `PrefixMask(model, <the other full layers>, n_doc=0)` (builds their correctly
    shaped causal mask). The first full-attention layer must be in kv_layers (it sets the RoPE positions)."""
    cache = new_cache(model)
    if st is None:
        return cache
    for li, t in enumerate(layer_types(model)):
        L = cache.layers[li]
        if t == "linear_attention":
            ks = keep_state and (state_layers is None or li in state_layers)
            kc = keep_conv and (conv_layers is None or li in conv_layers)
            if not (ks or kc):
                continue
            r = st["recurrent"][li]
            c = st["conv"].get(li)
            if c is None:                       # e.g. tile_states without a conv map: zero conv context
                la = decoder_layers(model)[li].linear_attn
                c = torch.zeros(r.shape[0], la.conv1d.weight.shape[0], la.conv_kernel_size, dtype=r.dtype,
                                device=r.device)
                kc = False
            L.update_conv_state(c.clone() if kc else torch.zeros_like(c))
            L.update_recurrent_state(r * state_scale if ks else torch.zeros_like(r))
        elif keep_kv and (kv_layers is None or li in kv_layers):
            K, V = st["kv"][li]
            L.update(K.clone(), V.clone())
    return cache


class PrefixMask:
    """Context manager: at the given full-attention layers, the cached document prefix (first n_doc K/V
    positions) is masked out of attention for every new token (additive -inf mask built per call), so those
    layers read nothing from the document while RoPE positions stay those of [doc; prompt]."""

    def __init__(self, model, layers, n_doc: int):
        self.model, self.layers, self.n_doc = model, [int(x) for x in layers], int(n_doc)
        self._h = []

    def __enter__(self):
        blocks = decoder_layers(self.model)
        for li in self.layers:
            def pre(mod, args, kwargs, li=li):
                x = kwargs.get("hidden_states", args[0] if args else None)
                cache = kwargs.get("past_key_values")
                B, T = x.shape[:2]
                past = cache.layers[li].get_seq_length() if cache is not None else 0
                kv = past + T
                col = torch.arange(kv, device=x.device).view(1, -1)
                row = torch.arange(T, device=x.device).view(-1, 1) + past
                ok = (col <= row) & (col >= self.n_doc)
                m = torch.zeros(B, 1, T, kv, dtype=x.dtype, device=x.device)
                m.masked_fill_(~ok.view(1, 1, T, kv), torch.finfo(x.dtype).min)
                kwargs["attention_mask"] = m
                return args, kwargs
            self._h.append(blocks[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []
        return False


@torch.no_grad()
def continue_logits(model, cache, ids: torch.Tensor) -> torch.Tensor:
    dev = model.get_input_embeddings().weight.device
    return model(input_ids=ids.to(dev), past_key_values=cache, use_cache=True).logits


@torch.no_grad()
def greedy_continue(model, tok, cache, prompt_ids, max_new_tokens: int = 64, stop: str = "</answer>",
                    hub: str = "") -> tuple[str, int]:
    """Greedy decode continuing from `cache` (prompt ids appended after the cached prefix).
    -> (decoded final text, n generated)."""
    from xenc.backbone import decode
    dev = model.get_input_embeddings().weight.device
    ids = torch.as_tensor(prompt_ids).view(1, -1).to(dev)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    eos = set(x for x in [tok.eos_token_id, getattr(tok, "pad_token_id", None)] if x is not None)
    try:
        eos |= {tok.convert_tokens_to_ids("<|im_end|>")}
    except Exception:
        pass
    new = []
    for _ in range(max_new_tokens):
        nxt = int(out.logits[0, -1].argmax())
        if nxt in eos:
            break
        new.append(nxt)
        if stop and stop in tok.decode(new[-12:], skip_special_tokens=True):
            break
        out = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=cache, use_cache=True)
    txt = decode(tok, hub, new)
    if stop and stop in txt:
        txt = txt.split(stop, 1)[0] + stop
    return txt, len(new)


def tile_states(src: dict, src_types: list[str], dst_types: list[str], dst_heads: int,
                dst_conv_dim: int | None = None) -> dict:
    """Zero-training S/L probe: map a smaller encoder's linear-layer states onto the reader's grid.
    Layer alignment: each reader linear layer takes the encoder linear layer at the nearest normalized depth
    (index / count over linear layers). Heads: encoder value head h serves reader heads tiled
    (reader head j <- encoder head j * H_src // H_dst). d_k, d_v must match. Conv states are not mapped
    (dimension differs) -> returned only when dst_conv_dim matches. No KV."""
    s_lin = [i for i, t in enumerate(src_types) if t == "linear_attention"]
    d_lin = [i for i, t in enumerate(dst_types) if t == "linear_attention"]
    out = {"recurrent": {}, "conv": {}, "kv": {}, "n_doc": src["n_doc"], "layer_types": dst_types,
           "layer_map": {}}
    for j, li in enumerate(d_lin):
        rho = j / max(1, len(d_lin) - 1)
        k = min(range(len(s_lin)), key=lambda i: abs(i / max(1, len(s_lin) - 1) - rho))
        S = src["recurrent"][s_lin[k]]                                   # [1,H_s,d_k,d_v]
        Hs = S.shape[1]
        idx = torch.arange(dst_heads, device=S.device) * Hs // dst_heads
        out["recurrent"][li] = S.index_select(1, idx).clone()
        out["layer_map"][li] = s_lin[k]
        c = src["conv"][s_lin[k]]
        if dst_conv_dim is not None and c.shape[1] == dst_conv_dim:
            out["conv"][li] = c.clone()
    return out


__all__ = ["harvest_states", "build_prefix_cache", "PrefixMask", "greedy_continue", "continue_logits", "tile_states",
           "state_nbytes", "layer_types", "new_cache"]
