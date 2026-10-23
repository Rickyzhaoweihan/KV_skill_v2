"""ρ-keyed (normalized-depth) document operators — the encoder side of xenc.

One WHOLE-document prefill on the encoder; forward hooks on
`deltaskill.backbone_utils.decoder_layers(model)` at the encoder's `injection_layers` capture the
residual stream [T, d_enc] at each depth ρ. The captured states are cut on token boundaries into
windows of `chunk_tokens` (NO pooling — every token is a slot, each slot carries full causal
context from all preceding windows), each row is L2-normalized (as `deltaskill.derive.derive_skill`
does), and stored fp16 on CPU keyed by ρ. K = V (tied), so only K is stored.

Cache file format (one `.pt` per document, see docs/XENC_SPEC.md):
    {"chunks": [{"K": {rho: fp16 [n, d_s]}, "n_tokens": n, "start": tok_start}, ...],
     "d_s": int, "doc_id": str, "encoder": str, "chunk_tokens": int, "depths": [...]}

Fake-model detection (for CPU tests with `tests/_tiny.TinyLM`, whose forward takes a float
[B,T,d] tensor instead of input_ids): `derive_by_depth` accepts an explicit `embed_fn(ids)->[1,T,d]`.
If none is given and the model looks fake (`isinstance(model, nn.Module) and not hasattr(model,
"generate")`), a deterministic random embedding table (`tiny_embed_fn`) is used. Real HF models
always get `model(input_ids=..., use_cache=False)`.
"""

from __future__ import annotations

import hashlib
import os
from typing import Callable, Iterable

import torch
import torch.nn as nn

import xenc  # noqa: F401  (sys.path bootstrap for deltaskill)
from xenc import DEPTHS
from deltaskill.backbone_utils import decoder_layers
from deltaskill.derive import injection_layers
from deltaskill.skill import KVSkill


class DepthCollisionError(ValueError, AssertionError):
    """Two depths map to the same physical layer (e.g. 4 depths on an L=3 model)."""


class _StopForward(Exception):
    pass


# ----------------------------------------------------------------------------- depth mapping
def _num_layers(model_or_L) -> int:
    if isinstance(model_or_L, int):
        return model_or_L
    c = model_or_L.config
    c = c.get_text_config() if hasattr(c, "get_text_config") else c
    return int(c.num_hidden_layers)


def _hidden_size(model) -> int:
    c = model.config
    c = c.get_text_config() if hasattr(c, "get_text_config") else c
    return int(c.hidden_size)


class _LCfg:  # minimal stand-in so an int L can go through deltaskill.injection_layers
    def __init__(self, L):
        self.config = type("C", (), {"num_hidden_layers": L})()


def rho_layer_map(model_or_L, depths: Iterable[float] = DEPTHS) -> dict[float, int]:
    """ρ -> physical layer `min(L-1, int(ρ·L))` (deltaskill.derive.injection_layers, per depth).

    Raises DepthCollisionError (a ValueError and an AssertionError) when two depths collide."""
    depths = sorted(float(r) for r in depths)
    m = model_or_L if not isinstance(model_or_L, int) else _LCfg(model_or_L)
    out = {r: injection_layers(m, [r])[0] for r in depths}
    if len(set(out.values())) != len(depths):
        raise DepthCollisionError(
            f"depth collision on L={_num_layers(model_or_L)}: {depths} -> {list(out.values())}")
    return out


def _key(d: dict, rho: float):
    """Float-tolerant lookup of a ρ key."""
    if rho in d:
        return rho
    for k in d:
        if abs(float(k) - float(rho)) < 1e-6:
            return k
    raise KeyError(f"no operator for depth rho={rho}; have {sorted(d)}")


# ----------------------------------------------------------------------------- tiny helpers
def is_fake_model(model) -> bool:
    return isinstance(model, nn.Module) and not hasattr(model, "generate")


class TinyTokenizer:
    """Whitespace tokenizer hashing words into a small vocab (CPU tests / --tiny only)."""

    def __init__(self, vocab: int = 50):
        self.vocab = vocab

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [int(hashlib.sha1(w.encode()).hexdigest(), 16) % self.vocab for w in text.split()]

    def __call__(self, text: str, return_tensors=None, add_special_tokens: bool = False):
        ids = self.encode(text)
        return {"input_ids": torch.tensor([ids]) if return_tensors == "pt" else ids}


def tiny_embed_fn(model, vocab: int = 50, seed: int = 1234) -> Callable[[torch.Tensor], torch.Tensor]:
    d = _hidden_size(model)
    table = torch.randn(vocab, d, generator=torch.Generator().manual_seed(seed))
    return lambda ids: table[ids.cpu() % vocab].to(model.device)


def _tokenize(tok, text: str) -> torch.Tensor:
    enc = tok(text, return_tensors="pt")
    ids = enc["input_ids"] if isinstance(enc, dict) or hasattr(enc, "keys") else enc
    ids = torch.as_tensor(ids)
    return ids if ids.dim() == 2 else ids.unsqueeze(0)


# ----------------------------------------------------------------------------- attn harvest (D3)
HARVESTS = ("resid", "attn", "attn_raw")   # attn_raw (D7): attn K/V without the per-row L2 (native-attention read)


def _attn_module(block):
    """The block's softmax self-attention module (has k_proj & v_proj), or None (e.g. Gated-DeltaNet)."""
    m = getattr(block, "self_attn", None)
    return m if m is not None and hasattr(m, "k_proj") and hasattr(m, "v_proj") else None


def attn_layer_map(model, depths=DEPTHS) -> dict[float, int]:
    """ρ -> nearest FULL-attention layer at or after rho_layer_map(ρ) (falls back to the nearest one
    before it when none follows). Qwen3.5 hybrid: full-attn layers are 3, 7, 11, … (interval 4)."""
    blocks = decoder_layers(model)
    full = [i for i, b in enumerate(blocks) if _attn_module(b) is not None]
    if not full:
        raise ValueError("harvest='attn': encoder has no self-attention module with k_proj/v_proj")
    out = {}
    for rho, li in rho_layer_map(model, depths).items():
        after = [j for j in full if j >= li]
        out[rho] = after[0] if after else full[-1]
    return out


def _derive_attn(model, ids, depths, embed_fn):
    """One prefill; capture per-token pre-RoPE keys (k_norm output if the module has one, else
    k_proj) and values (v_proj) at each ρ's attention layer -> {rho: (K [T, H·hd], V [T, H·hd])}."""
    amap = attn_layer_map(model, depths)
    blocks = decoder_layers(model)
    want = set(amap.values())
    capK: dict[int, torch.Tensor] = {}
    capV: dict[int, torch.Tensor] = {}
    handles = []

    def _maybe_stop():
        if want <= set(capK) and want <= set(capV):
            raise _StopForward

    for li in want:
        att = _attn_module(blocks[li])
        kmod = att.k_norm if hasattr(att, "k_norm") else att.k_proj

        def khook(mod, args, output, li=li):
            capK[li] = output.detach()[0].reshape(output.shape[1], -1)   # [T, H·hd]
            _maybe_stop()

        def vhook(mod, args, output, li=li):
            capV[li] = output.detach()[0].reshape(output.shape[1], -1)
            _maybe_stop()
        handles.append(kmod.register_forward_hook(khook))
        handles.append(att.v_proj.register_forward_hook(vhook))
    try:
        if embed_fn is not None:
            model(embed_fn(ids))
        else:
            model(input_ids=ids.to(model.device), use_cache=False)
    except _StopForward:
        pass
    finally:
        for h in handles:
            h.remove()
    kname = "k_norm" if hasattr(_attn_module(blocks[next(iter(want))]), "k_norm") else "k_proj"
    return {rho: (capK[li], capV[li]) for rho, li in amap.items()}, amap, kname


# ----------------------------------------------------------------------------- derive
@torch.no_grad()
def derive_by_depth(model, tok, text: str, depths=DEPTHS, chunk_tokens: int = 512,
                    max_doc_tokens: int = 0, embed_fn: Callable | None = None,
                    doc_id: str = "", encoder: str | None = None, harvest: str = "resid") -> dict:
    """Encode `text` into the per-document cache dict (fp16 CPU, ρ-keyed, chunked, no pooling).

    harvest="resid" (default): residual stream at ρ·L, K = V (tied, only K stored).
    harvest="attn" (D3): the encoder's own attention keys (pre-RoPE, post k_norm) and values at the
    nearest full-attention layer ≥ ρ·L, KV heads concatenated, rows L2-normalized; stored untied
    (chunk "K" and "V"), d_s = n_kv_heads·head_dim; meta records the layers used."""
    if harvest not in HARVESTS:
        raise ValueError(f"harvest must be one of {HARVESTS}, got {harvest!r}")
    depths = sorted(float(r) for r in depths)
    if harvest in ("attn", "attn_raw"):
        ids = _tokenize(tok, text)
        if max_doc_tokens and ids.shape[1] > max_doc_tokens:
            ids = ids[:, :max_doc_tokens]
        T = int(ids.shape[1])
        if T == 0:
            raise ValueError("empty document")
        if embed_fn is None and is_fake_model(model):
            embed_fn = tiny_embed_fn(model)
        cap, amap, kname = _derive_attn(model, ids, depths, embed_fn)
        l2 = lambda h: (h.float() / (h.float().norm(dim=-1, keepdim=True) + 1e-6)).to("cpu", torch.float16)
        if harvest == "attn_raw":   # D7: keep the module's own K/V scale (k_norm output, v_proj output)
            l2 = lambda h: h.float().to("cpu", torch.float16)
        Ks = {rho: l2(cap[rho][0]) for rho in depths}
        Vs = {rho: l2(cap[rho][1]) for rho in depths}
        chunks = []
        for a in range(0, T, chunk_tokens):
            b = min(T, a + chunk_tokens)
            chunks.append({"K": {rho: Ks[rho][a:b].clone() for rho in depths},
                           "V": {rho: Vs[rho][a:b].clone() for rho in depths},
                           "n_tokens": b - a, "start": a})
        return {"chunks": chunks, "d_s": int(Ks[depths[0]].shape[-1]), "doc_id": doc_id,
                "encoder": encoder or getattr(model.config, "_name_or_path", "model"),
                "chunk_tokens": int(chunk_tokens), "depths": depths, "harvest": harvest,
                "tied": False, "attn_layers": {float(r): int(li) for r, li in amap.items()},
                "key_source": kname}
    rmap = rho_layer_map(model, depths)
    blocks = decoder_layers(model)
    last = max(rmap.values())
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for li in rmap.values():
        def hook(mod, args, output, li=li):
            hs = output[0] if isinstance(output, tuple) else output
            captured[li] = hs.detach()[0]          # [T, d]
            if li == last:                          # skip remaining layers + lm_head
                raise _StopForward
        handles.append(blocks[li].register_forward_hook(hook))

    ids = _tokenize(tok, text)
    if max_doc_tokens and ids.shape[1] > max_doc_tokens:
        ids = ids[:, :max_doc_tokens]
    T = int(ids.shape[1])
    if T == 0:
        raise ValueError("empty document")
    if embed_fn is None and is_fake_model(model):
        embed_fn = tiny_embed_fn(model)
    try:
        if embed_fn is not None:
            model(embed_fn(ids))
        else:
            model(input_ids=ids.to(model.device), use_cache=False)
    except _StopForward:
        pass
    finally:
        for h in handles:
            h.remove()

    states = {}
    for rho, li in rmap.items():
        h = captured[li].float()
        h = h / (h.norm(dim=-1, keepdim=True) + 1e-6)
        states[rho] = h.to("cpu", torch.float16)
    chunks = []
    for a in range(0, T, chunk_tokens):
        b = min(T, a + chunk_tokens)
        chunks.append({"K": {rho: states[rho][a:b].clone() for rho in depths},
                       "n_tokens": b - a, "start": a})
    return {"chunks": chunks, "d_s": _hidden_size(model), "doc_id": doc_id,
            "encoder": encoder or getattr(model.config, "_name_or_path", "model"),
            "chunk_tokens": int(chunk_tokens), "depths": depths}


# ----------------------------------------------------------------------------- views
def for_reader(skill_rho: KVSkill, reader_model, depths=None) -> KVSkill:
    """ρ-keyed -> reader-layer-keyed VIEW (same tensors, no copy)."""
    depths = sorted(float(r) for r in (depths or skill_rho.meta["depths"]))
    rmap = rho_layer_map(reader_model, depths)
    K = {li: skill_rho.K[_key(skill_rho.K, r)] for r, li in rmap.items()}
    V = {li: skill_rho.V[_key(skill_rho.V, r)] for r, li in rmap.items()}
    meta = {**skill_rho.meta, "keyed_by": "layer", "rho_of": {li: r for r, li in rmap.items()}}
    return KVSkill(K=K, V=V, d_s=skill_rho.d_s, text=skill_rho.text,
                   model_name=skill_rho.model_name, meta=meta)


def chunk_to_kvskill(chunk: dict, d_s: int, doc_id: str, encoder: str) -> KVSkill:
    """One cache chunk -> ρ-keyed KVSkill; K and V are the SAME dict/tensors (tied) unless the chunk
    carries its own "V" (harvest="attn")."""
    K = {float(r): t for r, t in chunk["K"].items()}
    V = {float(r): t for r, t in chunk["V"].items()} if chunk.get("V") is not None else K
    depths = sorted(K)
    return KVSkill(K=K, V=V, d_s=int(d_s), text=f"{doc_id}#t{chunk.get('start', 0)}",
                   model_name=encoder,
                   meta={"keyed_by": "rho", "depths": depths, "tied": V is K, "encoder": encoder,
                         "doc_id": doc_id, "start": int(chunk.get("start", 0)),
                         "n_tokens": int(chunk.get("n_tokens", next(iter(K.values())).shape[0])),
                         "num_tokens": int(next(iter(K.values())).shape[0])})


# ----------------------------------------------------------------------------- storage
def save_doc_cache(d: dict, path: str) -> None:
    """Atomic save (tmp + rename) so a killed job never leaves a truncated cache file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _f16 = lambda m: {float(r): t.detach().to("cpu", torch.float16).contiguous() for r, t in m.items()}
    out = {**d, "chunks": [{**c, "K": _f16(c["K"]), **({"V": _f16(c["V"])} if c.get("V") is not None else {})}
                           for c in d["chunks"]]}
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(out, tmp)
    os.replace(tmp, path)


def load_doc_cache(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def cache_key(doc_text: str, encoder_short: str, chunk_tokens: int, depths, harvest: str = "resid") -> str:
    h = hashlib.sha1(doc_text.encode()).hexdigest()[:12]
    short = encoder_short.rstrip("/").split("/")[-1]
    suffix = "" if harvest in (None, "resid") else f"_h{harvest}"     # resid keys unchanged (old caches valid)
    return f"{h}_{short}_c{int(chunk_tokens)}_d{'-'.join(str(float(r)) for r in depths)}{suffix}"


__all__ = ["derive_by_depth", "for_reader", "chunk_to_kvskill", "save_doc_cache", "load_doc_cache",
           "cache_key", "rho_layer_map", "attn_layer_map", "HARVESTS", "DepthCollisionError", "TinyTokenizer", "tiny_embed_fn",
           "is_fake_model"]
