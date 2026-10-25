"""BL / S5: question-agnostic compression of the READER's own document KV cache (agent BASE, 2026-10-23).

Label for every row produced with this module: "reader prefills the doc (ingest-cost-unmatched upper reference)".
The reader runs one full prefill of [chat template + document] (no question), its cache is compressed ONCE per
document, and every question of the document continues from the compressed cache.

Qwen3.5 is hybrid: only the full-attention layers hold a per-token KV cache; the Gated-DeltaNet (linear-attention)
layers hold a fixed-size recurrent + conv state, which is kept unchanged (it already summarizes the whole document;
its size is reported separately in the storage column).

Compression is applied by MASKING (per full-attention layer, per KV head, per cached position): dropped positions
get an additive -inf mask for every later query. Kept keys keep their original RoPE positions and new tokens continue
at position P (the full prefix length), exactly as after a physical eviction that keeps positions (kvpress / KVzip
convention), so the attention outputs equal those of a physically pruned cache; storage is accounted analytically
(kept positions x layers x KV heads x head_dim x K,V x 2 bytes). Prefix positions before the document (chat template,
system prompt) are always kept and not counted against the budget.

Scores (all question-agnostic; computed once per document):
  streaming  StreamingLLM: keep the template (attention sink) + the most recent keep_n document tokens.
  snapkv     SnapKV with the observation window = the LAST `window` (64) cached document tokens (no question exists
             at compression time): attention of the window queries to every cached key, summed over window rows,
             avg-pooled over positions (kernel 7), summed over each KV head's query group; per-head top-k
             (uniform head budget); the window itself is always kept.
  kvzip      KVzip-style context reconstruction: after the prefill, the reader is fed
             "\\n\\nRepeat the previous context exactly." + the assistant turn + the document again in chunks of
             `chunk` (2048) tokens (each chunk on the restored post-prefill cache); score(l, h, j) = max over all
             reconstruction queries and the query heads of KV head h of the softmax attention to cached position j.
             Non-uniform budget: global top keep_n * n_layers * n_kv over all (layer, head, position) triples.
Budget: keep_n = round(ratio * N_doc) document positions per (layer, KV head) on average; ratio 1 = no compression
(the reference "kvc_full" row: plain prefix cache, no mask hook at all).
"""

from __future__ import annotations

import torch

import xenc  # noqa: F401
from deltaskill.backbone_utils import decoder_layers
from xenc.native_attn import attn_geometry, reader_queries
from xenc.kv_prune import _apply_rope


def text_config(model):
    c = model.config
    return c.get_text_config() if hasattr(c, "get_text_config") else c


def full_layers(model) -> list[int]:
    return [i for i, t in enumerate(text_config(model).layer_types) if t != "linear_attention"]


# ============================================================================ prompt split
def split_prompt(tok, hub: str, system: str, user: str, doc_text: str, doc_offset: int) -> tuple[list[int], int, int]:
    """Token ids of the chat prompt (identical to xenc.backbone.build_chat) and the token boundaries
    (doc_start, P): tokens [0, doc_start) end before the document text, tokens [0, P) end at or before the end of
    the document text (P is the question-agnostic prefix; the token straddling the document end, if any, goes to
    the suffix). doc_offset = character offset of doc_text inside `user`."""
    from deltaskill.backbone_utils import chat_template_kwargs
    from xenc.backbone import build_chat
    ids = build_chat(tok, hub, system, user)
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    S = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **chat_template_kwargs(hub))
    enc = tok(S, add_special_tokens=False, return_offsets_mapping=True)
    assert list(enc["input_ids"]) == list(ids), "rendered chat template does not re-tokenize to build_chat ids"
    u0 = S.index(user)
    d0, d1 = u0 + doc_offset, u0 + doc_offset + len(doc_text)
    ends = [e for _, e in enc["offset_mapping"]]
    doc_start = sum(1 for e in ends if e <= d0)
    P = sum(1 for e in ends if e <= d1)
    return list(ids), int(doc_start), int(P)


def gen_prompt_suffix(tok, hub: str) -> str:
    """The chat template text that closes a user turn and opens the assistant turn (thinking off)."""
    from deltaskill.backbone_utils import chat_template_kwargs
    S = tok.apply_chat_template([{"role": "user", "content": "XQXQX"}], tokenize=False, add_generation_prompt=True,
                                **chat_template_kwargs(hub))
    return S.split("XQXQX", 1)[1]


# ============================================================================ capture of attention inputs
class _Capture:
    """Forward pre-hooks on the full-attention modules: store the (last `last` rows of the) hidden input and the
    RoPE (cos, sin) of those rows."""

    def __init__(self, model, layers, last: int | None = None):
        self.model, self.layers, self.last = model, list(layers), last
        self.x, self._h = {}, []

    def __enter__(self):
        blocks = decoder_layers(self.model)
        for li in self.layers:
            def pre(mod, args, kwargs, li=li):
                x = kwargs.get("hidden_states", args[0] if args else None)
                cos, sin = kwargs["position_embeddings"]
                if self.last:
                    x, cos, sin = x[:, -self.last:], cos[:, -self.last:], sin[:, -self.last:]
                self.x[li] = (x.detach(), cos.detach(), sin.detach())
            self._h.append(blocks[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []
        return False


def _queries(model, li, x, cos, sin):
    att = decoder_layers(model)[li].self_attn
    geo = attn_geometry(att)
    q = reader_queries(att, x, geo)[0]
    return _apply_rope(q.float(), cos.float(), sin.float()), geo


@torch.no_grad()
def attn_probs_to_prefix(q: torch.Tensor, K: torch.Tensor, scaling: float, q_pos0: int, P: int, groups: int,
                         reduce: str, block: int = 256) -> torch.Tensor:
    """Softmax attention of queries q [1,n_q,m,hd] (post-RoPE, absolute positions q_pos0 .. q_pos0+m-1) over keys
    K [1,n_kv,L,hd] (causal), restricted to key positions [0, P).
    reduce "max": max over queries and the query heads of each KV head -> [n_kv, P]
    reduce "sum": sum over queries -> [n_q, P]."""
    _, n_q, m, hd = q.shape
    n_kv, L = K.shape[1], K.shape[2]
    Kf = K[0].float()
    col = torch.arange(L, device=K.device)
    out = torch.zeros((n_kv if reduce == "max" else n_q), P, dtype=torch.float32, device=K.device)
    for s in range(0, m, block):
        e = min(m, s + block)
        qb = q[0, :, s:e].float().reshape(n_kv, groups, e - s, hd)
        lg = torch.einsum("kgbd,ktd->kgbt", qb, Kf) * scaling                 # [n_kv,g,B,L]
        row = torch.arange(q_pos0 + s, q_pos0 + e, device=K.device).view(-1, 1)
        lg = lg.masked_fill((col.view(1, -1) > row).view(1, 1, e - s, L), float("-inf"))
        p = torch.softmax(lg, -1)[..., :P]
        if reduce == "max":
            out = torch.maximum(out, p.amax(dim=(1, 2)))
        else:
            out += p.sum(2).reshape(n_q, P)
        del lg, p
    return out


# ============================================================================ prefill + scores
@torch.no_grad()
def prefill(model, prefix_ids: list[int], window: int = 64):
    """One prefill of the question-agnostic prefix. -> (cache, {li: (x_last, cos, sin)}) with the last `window`
    rows' attention inputs captured for SnapKV."""
    dev = model.get_input_embeddings().weight.device
    x = torch.tensor([prefix_ids], device=dev)
    with _Capture(model, full_layers(model), last=window) as cap:
        out = model(input_ids=x, use_cache=True, logits_to_keep=1)
    return out.past_key_values, cap.x


@torch.no_grad()
def snapkv_scores(model, cache, win: dict, P: int, window: int = 64, kernel: int = 7) -> dict:
    sc = {}
    for li, (x, cos, sin) in win.items():
        q, geo = _queries(model, li, x, cos, sin)
        K = cache.layers[li].keys[:, :, :P]
        w = q.shape[2]
        s = attn_probs_to_prefix(q, K, geo["scaling"], P - w, P, geo["groups"], reduce="sum")       # [n_q,P]
        s = torch.nn.functional.avg_pool1d(s.unsqueeze(1), kernel, stride=1, padding=kernel // 2).squeeze(1)
        sc[li] = s.reshape(geo["n_kv"], geo["groups"], P).sum(1)                                   # [n_kv,P]
    return sc


def _save_linear(cache, lin):
    return {li: {k: (v.clone() if torch.is_tensor(v) else v) for k, v in vars(cache.layers[li]).items()} for li in lin}


def _restore(cache, saved, full, P):
    for li, d in saved.items():
        for k, v in d.items():
            setattr(cache.layers[li], k, v.clone() if torch.is_tensor(v) else v)
    for li in full:
        L = cache.layers[li]
        L.keys, L.values = L.keys[:, :, :P].contiguous(), L.values[:, :, :P].contiguous()


@torch.no_grad()
def kvzip_scores(model, tok, hub: str, cache, prefix_ids: list[int], doc_start: int, P: int,
                 chunk: int = 2048, prompt: str = "\n\nRepeat the previous context exactly.") -> dict:
    """KVzip-style reconstruction scores {li: [n_kv, P]} (max attention received). The cache is restored to the
    post-prefill state after every chunk (attention layers cropped to P, linear-attention states restored)."""
    dev = model.get_input_embeddings().weight.device
    full = full_layers(model)
    lin = [i for i in range(len(cache.layers)) if i not in full]
    saved = _save_linear(cache, lin)
    pre = tok(prompt + gen_prompt_suffix(tok, hub), add_special_tokens=False)["input_ids"]
    doc = prefix_ids[doc_start:P]
    sc = {li: torch.zeros(cache.layers[li].keys.shape[1], P, device=cache.layers[li].keys.device) for li in full}
    for s in range(0, len(doc), chunk):
        ids = pre + doc[s:s + chunk]
        with _Capture(model, full) as cap:
            model(input_ids=torch.tensor([ids], device=dev), past_key_values=cache, use_cache=True, logits_to_keep=1)
        for li in full:
            x, cos, sin = cap.x[li]
            q, geo = _queries(model, li, x, cos, sin)
            K = cache.layers[li].keys                                           # [1,n_kv,P+m,hd]
            sc[li] = torch.maximum(sc[li], attn_probs_to_prefix(q, K, geo["scaling"], P, P, geo["groups"], "max"))
        del cap
        _restore(cache, saved, full, P)
    return sc


# ============================================================================ keep masks
def keep_masks(method: str, ratio: float, P: int, doc_start: int, n_kv: int, layers: list[int],
               scores: dict | None = None, window: int = 64) -> tuple[dict, dict]:
    """-> ({li: bool [n_kv, P]} kept positions, info). ratio >= 1 keeps everything."""
    N = P - doc_start
    keep_n = max(1, min(N, int(round(ratio * N))))
    masks = {}
    for li in layers:
        m = torch.zeros(n_kv, P, dtype=torch.bool)
        m[:, :doc_start] = True
        masks[li] = m
    if ratio >= 1:
        for li in layers:
            masks[li][:] = True
    elif method == "streaming":
        for li in layers:
            masks[li][:, P - keep_n:] = True
    elif method == "snapkv":
        w = min(window, keep_n)
        for li in layers:
            s = scores[li].float().cpu().clone()
            s[:, :doc_start] = float("-inf")
            s[:, P - w:] = float("-inf")
            masks[li][:, P - w:] = True
            if keep_n > w:
                top = s.topk(keep_n - w, dim=-1).indices
                masks[li].scatter_(1, top, True)
    elif method == "kvzip":
        flat = torch.cat([scores[li].float().cpu()[:, doc_start:].reshape(-1) for li in layers])
        k = keep_n * n_kv * len(layers)
        top = flat.topk(k).indices
        sel = torch.zeros_like(flat, dtype=torch.bool)
        sel[top] = True
        off = 0
        for li in layers:
            n = n_kv * N
            masks[li][:, doc_start:] = sel[off:off + n].view(n_kv, N)
            off += n
    else:
        raise ValueError(method)
    kept = sum(int(m[:, doc_start:].sum()) for m in masks.values())
    info = {"method": method, "ratio": ratio, "N_doc": N, "keep_n": keep_n, "doc_start": doc_start, "P": P,
            "kept_frac": kept / max(1, N * n_kv * len(layers))}
    return masks, info


class MaskHook:
    """Forward pre-hooks on the full-attention modules: cached prefix columns [0, P) are visible only where
    keep[li][kv_head, col] is True; columns >= P are causal. Additive float mask [1, n_q, T, past+T]."""

    def __init__(self, model, keep: dict, P: int):
        self.model, self.P, self._h = model, int(P), []
        self.keep = {}
        for li, m in keep.items():
            att = decoder_layers(model)[li].self_attn
            g = attn_geometry(att)["groups"]
            self.keep[li] = m.repeat_interleave(g, dim=0)          # [n_q,P] (head h reads KV head h // g)

    def __enter__(self):
        blocks = decoder_layers(self.model)
        for li, km in self.keep.items():
            def pre(mod, args, kwargs, li=li):
                x = kwargs.get("hidden_states", args[0] if args else None)
                cache = kwargs.get("past_key_values")
                T = x.shape[1]
                past = cache.layers[li].get_seq_length() if cache is not None else 0
                assert past >= self.P, "MaskHook is only for continuations of the compressed prefix"
                L = past + T
                k = self.keep[li].to(x.device)
                n_q = k.shape[0]
                col = torch.arange(L, device=x.device).view(1, -1)
                row = torch.arange(T, device=x.device).view(-1, 1) + past
                ok = (col <= row).view(1, T, L).expand(n_q, T, L).clone()
                ok[:, :, :self.P] &= k.view(n_q, 1, self.P)
                m = torch.zeros(1, n_q, T, L, dtype=x.dtype, device=x.device)
                m.masked_fill_(~ok.unsqueeze(0), torch.finfo(x.dtype).min)
                kwargs["attention_mask"] = m
                return args, kwargs
            self._h.append(blocks[li].self_attn.register_forward_pre_hook(pre, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []
        return False


def storage_bytes(model, cache, kept_positions_total: int, bytes_per: int = 2) -> dict:
    """Analytic storage of the compressed per-document memory: kept K,V of the full-attention layers (bf16) +
    the fixed DeltaNet conv / recurrent states as held in the cache."""
    full = full_layers(model)
    hd = cache.layers[full[0]].keys.shape[-1]
    kv = kept_positions_total * hd * 2 * bytes_per
    lin = 0
    for i, L in enumerate(cache.layers):
        if i in full:
            continue
        for v in vars(L).values():
            if torch.is_tensor(v):
                lin += v.numel() * v.element_size()
    return {"kv_bytes": int(kv), "linear_state_bytes": int(lin)}


__all__ = ["split_prompt", "gen_prompt_suffix", "prefill", "snapkv_scores", "kvzip_scores", "keep_masks",
           "MaskHook", "attn_probs_to_prefix", "storage_bytes", "full_layers"]
