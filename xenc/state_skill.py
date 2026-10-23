"""D10: the document KV-Skill as the encoder's native Gated-DeltaNet states (harvest + cache + prefix cache).

Qwen3.5 is hybrid: 3 of every 4 decoder layers are Gated-DeltaNet, whose memory of everything read so far is
a fixed-size recurrent state S_ℓ ∈ R^{n_v_heads × d_k × d_v} = Σ_t (decayed, delta-corrected) k_t v_tᵀ, read as
o_t = S_ℓᵀ q_t. That state is the paper's associative operator M = W Uᵀ with the linear read W(Uᵀq), harvested by
ONE frozen prefill of the document and never trained. This module:

    harvest_states(model, tok, doc_text, hub=None, keep_kv=False) -> {"recurrent": {li: [Hv,dk,dv]},
        "conv": {li: [conv_dim, K]}, "kv": {li: (K [Hkv,T,hd], V)} (only if keep_kv), "n_doc": int, ...}
    build_prefix_cache(model, states, keep_kv, keep_state, keep_conv) -> transformers DynamicCache
    save_states / load_states / state_path (fp16 on disk, one file per (model, doc))

(Key names follow D7's planned xenc/native_state.py so the two can be reconciled.)

The prefix that is harvested is the chat-template rendering of `ctx_prompt(item, doc)` CUT right before
"\\n\\n## Question\\n" (`doc_prefix_text`): prefix ⊕ question-suffix == the `text` condition's prompt, so in
L/L the exact prefix cache (states + full-attention K/V) reproduces the in-context teacher (the anchor).
"""

from __future__ import annotations

import os

import torch

import xenc  # noqa: F401

FORMAT = "xenc.state_skill.v1"
STATE_ROOT = os.path.join(xenc.DEFAULT_CACHE, "states")
QSEP = "\n\n## Question\n"


# ============================================================================ model structure
def text_config(model):
    cfg = model.config
    return getattr(cfg, "text_config", None) or cfg


def layer_types(model) -> list[str]:
    return list(text_config(model).layer_types)


def linear_layers(model) -> list[int]:
    return [i for i, t in enumerate(layer_types(model)) if t == "linear_attention"]


def full_layers(model) -> list[int]:
    return [i for i, t in enumerate(layer_types(model)) if t != "linear_attention"]


def gdn_modules(model) -> dict:
    """reader layer index -> its Qwen3_5GatedDeltaNet module."""
    from deltaskill.backbone_utils import decoder_layers
    layers = decoder_layers(model)
    return {i: layers[i].linear_attn for i in linear_layers(model)}


def state_dims(model) -> dict:
    c = text_config(model)
    return {"n_k": int(c.linear_num_key_heads), "n_v": int(c.linear_num_value_heads),
            "dk": int(c.linear_key_head_dim), "dv": int(c.linear_value_head_dim),
            "n_layers": int(c.num_hidden_layers), "linear": linear_layers(model)}


def depth_map(reader_layers: list[int], L_reader: int, enc_layers: list[int], L_enc: int) -> dict:
    """reader linear layer -> encoder linear layer nearest in normalized depth (ℓ/(L−1))."""
    out = {}
    for li in reader_layers:
        tgt = li / max(L_reader - 1, 1) * max(L_enc - 1, 1)
        out[int(li)] = int(min(enc_layers, key=lambda e: (abs(e - tgt), e)))
    return out


# ============================================================================ prompts
def doc_prefix_text(tok, hub: str, doc_text: str, system: str | None) -> str:
    """Chat-template prefix of the `text` condition up to (excluding) "\\n\\n## Question\\n"."""
    from deltaskill.backbone_utils import chat_template_kwargs
    from xenc.doc_tasks import ctx_prompt
    dummy = {"question": "Q", "choices": None, "answers": ["a"]}
    user = ctx_prompt(dummy, doc_text)
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **chat_template_kwargs(hub))
    k = s.rfind(QSEP)
    assert k > 0, "question separator not found in the rendered chat"
    return s[:k]


def question_suffix_ids(tok, hub: str, item, system: str) -> list[int]:
    """Token ids of the `text` prompt AFTER the document prefix (starts with "\\n\\n## Question\\n")."""
    from deltaskill.backbone_utils import chat_template_kwargs
    from xenc.doc_tasks import ctx_prompt
    user = ctx_prompt(item, "\x00DOC\x00")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **chat_template_kwargs(hub))
    k = s.rfind(QSEP)
    return [int(x) for x in tok(s[k:], add_special_tokens=False)["input_ids"]]


# ============================================================================ harvest
@torch.no_grad()
def harvest_states(model, tok, doc_text: str, hub: str = "", system: str | None = None, keep_kv: bool = False,
                   prefix_ids: list[int] | None = None, max_tokens: int = 0) -> dict:
    """One frozen prefill of the document prefix (use_cache=True); returns fp16 CPU states."""
    from xenc.backbone import input_device
    if prefix_ids is None:
        if system is None:
            raise ValueError("pass system= (the registration system prompt) or prefix_ids=")
        prefix_ids = [int(x) for x in tok(doc_prefix_text(tok, hub, doc_text, system),
                                          add_special_tokens=False)["input_ids"]]
    if max_tokens and len(prefix_ids) > max_tokens:
        prefix_ids = prefix_ids[:max_tokens]
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=input_device(model))
    out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    pkv = out.past_key_values
    rec, conv, kv = {}, {}, {}
    for li in linear_layers(model):
        L = pkv.layers[li]
        rec[li] = L.recurrent_states[0].detach().float().cpu()
        conv[li] = L.conv_states[0].detach().float().cpu()
    if keep_kv:
        for li in full_layers(model):
            L = pkv.layers[li]
            kv[li] = (L.keys[0].detach().to(torch.float16).cpu(), L.values[0].detach().to(torch.float16).cpu())
    for li, S in rec.items():
        if not torch.isfinite(S).all():
            raise FloatingPointError(f"non-finite recurrent state at layer {li}")
    amax = max(float(S.abs().max()) for S in rec.values())
    dt = torch.float16 if amax < 6e4 else torch.float32
    return {"format": FORMAT, "recurrent": {k: v.to(dt) for k, v in rec.items()},
            "conv": {k: v.to(dt) for k, v in conv.items()}, "kv": kv, "n_doc": int(ids.shape[1]),
            "absmax": amax, "dims": state_dims(model), "adims": _attn_dims(model)}


def _attn_dims(model) -> dict:
    from xenc.xattn_read import attn_dims
    return attn_dims(model)


# ============================================================================ disk cache
def state_path(root: str | None, model_short: str, doc_id: str, kv: bool = False) -> str:
    return os.path.join(root or STATE_ROOT, model_short, f"{doc_id}{'_kv' if kv else ''}.pt")


def save_states(d: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(d, tmp)
    os.replace(tmp, path)


def load_states(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


# ============================================================================ prefix cache (L/L anchor)
def build_prefix_cache(model, states: dict, keep_kv: bool = True, keep_state: bool = True, keep_conv: bool = True,
                       device=None, dtype=None):
    """DynamicCache holding the document prefix: full-attention K/V (keep_kv), DeltaNet recurrent states
    (keep_state) and conv tails (keep_conv; zeros otherwise). keep_state=False leaves the linear layers
    un-initialised, so they start fresh on the question (has_previous_state False)."""
    from transformers import DynamicCache
    from xenc.backbone import input_device
    dev = device or input_device(model)
    dt = dtype or next(model.parameters()).dtype
    cache = DynamicCache(config=text_config(model))
    if keep_kv:
        if not states.get("kv"):
            raise ValueError("states have no full-attention K/V (harvest with keep_kv=True)")
        for li, (K, V) in states["kv"].items():
            cache.update(K.unsqueeze(0).to(dev, dt), V.unsqueeze(0).to(dev, dt), int(li))
    if keep_state:
        for li, S in states["recurrent"].items():
            L = cache.layers[int(li)]
            c = states["conv"][li].unsqueeze(0).to(dev, dt)
            L.update_conv_state(c if keep_conv else torch.zeros_like(c))
            L.update_recurrent_state(S.unsqueeze(0).to(dev, torch.float32))
    return cache


__all__ = ["harvest_states", "build_prefix_cache", "save_states", "load_states", "state_path", "linear_layers",
           "full_layers", "gdn_modules", "state_dims", "depth_map", "doc_prefix_text", "question_suffix_ids"]
