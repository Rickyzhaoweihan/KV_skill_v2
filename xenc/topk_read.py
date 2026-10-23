"""D8: question-dependent top-k SELECTION inside the read branch (instead of SVD compression).

Two fixed one-prefill harvests of the SAME document, slot-aligned (same tokenization, same 512 chunking):
    keys for selection : the encoder's attention keys at the full-attention layer nearest ρ_sel
                         (pre-RoPE, post k_norm; `harvest="attn"` caches `*_hattn.pt`, or `attn_raw`)
    operator for read  : the standard residual Library (centered `*_c512_d…pt`), U = W = H̄ (tied)

Stage 1 (selection, once per forward): the READER's own attention queries at its full-attention layer for
ρ_sel (pre-RoPE, post q_norm; GQA: query head h scores KV head h // groups) at the last prompt position are
scored against the key cache, s_i = pool_h(q_h · k_{kv(h),i}) (pool = max over heads by default), and the
top-k valid slot indices are kept (`last_idx` [B,k], flat into [C·N]).
Stage 2 (read, every injection layer): the paper's read restricted to those k slots of the residual operator,
    q = L2(A·RMSNorm(h)),  a = softmax(read_temp · Uₛᵀq) over the k selected slots,  h' = h + γ g B (Wₛ a)
through the unchanged registered interface. No routing over chunks (the selected set replaces it).

The selection query comes from a separate, injector-disabled prefill pass of the bare reader that stops right
after the selection layer's q_proj (≈ ρ_sel of a forward). It is triggered by a forward pre-hook on the
top-level model whenever a forward has more than one position (prefill / teacher-forced training forward);
single-token decode forwards REUSE the cached index set. Fake models (tests/_tiny: `logits()` bypasses
`__call__`) are handled by a pre-hook on decoder block 0 that re-runs `model.forward(x)`.

k >= number of valid slots ⇒ selection is skipped and the read is exactly LibraryInjector's (so the 1-chunk
equivalence and the no-op-at-init (B = 0) behaviour are inherited unchanged).
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict

import torch

import xenc  # noqa: F401
from deltaskill.backbone_utils import decoder_layers
from deltaskill.derive import rms_norm
from xenc.depth_skill import _key, attn_layer_map
from xenc.injector import LibraryInjector, _on, itf_gain, itf_gate, itf_query, itf_write
from xenc.library import Library


# ============================================================================ attention geometry
def attn_geometry(att) -> dict:
    """(same convention as xenc.native_attn.attn_geometry; duplicated so D8 does not depend on D7's file)"""
    hd = int(getattr(att, "head_dim"))
    n_kv = att.k_proj.out_features // hd
    groups = int(getattr(att, "num_key_value_groups", 1))
    n_q = n_kv * groups
    gated = att.q_proj.out_features == 2 * n_q * hd
    if not gated and att.q_proj.out_features != n_q * hd:
        raise ValueError(f"q_proj out {att.q_proj.out_features} inconsistent with {n_q} heads x {hd}")
    return {"hd": hd, "n_kv": n_kv, "groups": groups, "n_q": n_q, "gated": gated}


def reader_queries(att, x: torch.Tensor, geo: dict) -> torch.Tensor:
    """The attention module's own pre-RoPE queries (post q_norm) -> [B, n_q, T, hd]."""
    B, T, _ = x.shape
    hd = geo["hd"]
    qg = att.q_proj(x)
    if geo["gated"]:
        q, _ = torch.chunk(qg.view(B, T, -1, 2 * hd), 2, dim=-1)
    else:
        q = qg.view(B, T, -1, hd)
    if hasattr(att, "q_norm"):
        q = att.q_norm(q)
    return q.transpose(1, 2)


class _StopSelect(Exception):
    pass


# ============================================================================ selection-key library
class TopKLibrary(Library):
    """A residual Library plus `sel_K` [C, N, d_k]: the slot-aligned attention keys used for selection."""

    sel_K: torch.Tensor

    def subset(self, idx) -> "TopKLibrary":
        new = super().subset(idx)
        idx = [int(i) for i in (idx if hasattr(idx, "__iter__") else [idx])]
        n = next(iter(new.K.values())).shape[1]
        new.__class__ = TopKLibrary
        new.sel_K = self.sel_K.index_select(0, torch.tensor(idx, device=self.sel_K.device))[:, :n].clone()
        return new

    def to(self, device=None, dtype=None) -> "TopKLibrary":
        super().to(device=device, dtype=dtype)
        self.sel_K = self.sel_K.to(device=device, dtype=dtype)
        return self


def attach_sel_keys(lib: Library, attn_cache: dict, sel_rho: float) -> TopKLibrary:
    """Turn `lib` (in place) into a TopKLibrary whose `sel_K` holds the attn cache's keys at `sel_rho`,
    padded to lib's [C, N]. Raises if the two harvests are not slot-aligned."""
    chunks = attn_cache["chunks"]
    C, N = next(iter(lib.mask.values())).shape
    if len(chunks) != C:
        raise ValueError(f"attn cache has {len(chunks)} chunks, residual library {C}")
    rows = [c["K"][_key(c["K"], sel_rho)] for c in chunks]
    n_lib = next(iter(lib.mask.values())).sum(-1).tolist()
    for c, r in enumerate(rows):
        if r.shape[0] != int(n_lib[c]):
            raise ValueError(f"chunk {c}: attn cache {r.shape[0]} slots vs residual {int(n_lib[c])} (not aligned)")
    d_k = rows[0].shape[-1]
    dev = next(iter(lib.K.values())).device
    S = torch.zeros(C, N, d_k, dtype=torch.float32, device=dev)
    for c, r in enumerate(rows):
        S[c, : r.shape[0]] = r.to(device=dev, dtype=torch.float32)
    lib.__class__ = TopKLibrary
    lib.sel_K = S
    return lib


class TopKSpec:
    """Attached to a DocRuntime as `rt.topk`: residual Library + slot-aligned attention keys (own LRU)."""

    def __init__(self, sel_rho: float, sel_harvest: str = "attn", lru: int = 8):
        self.sel_rho = float(sel_rho)
        self.sel_harvest = sel_harvest
        self._lru: OrderedDict = OrderedDict()
        self._lru_cap = lru

    def attn_path(self, rt, doc_id: str) -> str:
        from xenc.depth_skill import cache_key
        return os.path.join(rt.cache_dir, cache_key(rt.docs[doc_id], rt.enc_short, rt.chunk_tokens, rt.depths,
                                                    harvest=self.sel_harvest) + ".pt")

    def attn_cache(self, rt, doc_id: str) -> dict | None:
        if doc_id in self._lru:
            self._lru.move_to_end(doc_id)
            return self._lru[doc_id]
        from xenc.depth_skill import load_doc_cache
        p = self.attn_path(rt, doc_id)
        if not os.path.exists(p):
            print(f"WARN: D8 no {self.sel_harvest} cache for {doc_id} ({os.path.basename(p)})", flush=True)
            return None
        d = load_doc_cache(p)
        self._lru[doc_id] = d
        while len(self._lru) > self._lru_cap:
            self._lru.popitem(last=False)
        return d

    def library(self, rt, doc_id: str, resid_cache: dict):
        ac = self.attn_cache(rt, doc_id)
        if ac is None:
            return None
        lib = Library.from_doc_cache(resid_cache, rt.model, rt.depths, device="cpu", center=rt.center)
        return attach_sel_keys(lib, ac, self.sel_rho)


def load_sel_kmap(path: str, sel_rho: float) -> dict:
    """D7 fit_kv_map.py output -> {"W": [d_dst, d_src], "b": [d_dst]} for the key map at sel_rho (S/L)."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    e = blob["maps"][_key(blob["maps"], sel_rho)]
    return {"W": e["Wk"].float(), "b": e["bk"].float()}


# ============================================================================ selector
class TopKSelector:
    """Captures the reader's queries at `sel_layer` and scores them against the loaded selection keys.

    keys: [S, n_kv, hd] (flat over C·N; invalid slots masked). Scores per head [B, n_q, S]; pooled over heads
    ("max" | "mean") -> [B, S]; `last_idx` = top-k flat indices [B, k] (sorted by score)."""

    def __init__(self, model, sel_layer: int, k: int = 64, pool: str = "max", query: str = "last",
                 kmap: dict | None = None):
        blocks = decoder_layers(model)
        att = getattr(blocks[sel_layer], "self_attn", None)
        if att is None or not hasattr(att, "q_proj"):
            raise ValueError(f"selection layer {sel_layer} has no softmax self-attention with q_proj")
        self.model, self.sel_layer, self.att = model, int(sel_layer), att
        self.geo = attn_geometry(att)
        self.k, self.pool, self.query = int(k), pool, query
        self.kmap = kmap
        self.keys: torch.Tensor | None = None       # [S, n_kv, hd]
        self.valid: torch.Tensor | None = None      # [S] bool
        self.pos: int = -1
        self.last_idx: torch.Tensor | None = None   # [B, k]
        self.last_scores: torch.Tensor | None = None   # [B, S] pooled (invalid = -inf)
        self.last_head_scores: torch.Tensor | None = None  # [B, n_q, T_q, S] when keep_heads
        self.keep_heads = False
        self.armed = False

    # ---- keys
    def set_keys(self, sel_K: torch.Tensor, valid: torch.Tensor) -> None:
        K = sel_K.reshape(-1, sel_K.shape[-1]).float()
        if self.kmap is not None:
            K = K @ self.kmap["W"].to(K).T + self.kmap["b"].to(K)
        g = self.geo
        if K.shape[-1] != g["n_kv"] * g["hd"]:
            raise ValueError(f"selection keys d={K.shape[-1]} != reader KV dim {g['n_kv'] * g['hd']} "
                             "(cross-model selection needs a key map)")
        self.keys = K.reshape(-1, g["n_kv"], g["hd"])
        self.valid = valid.reshape(-1).bool()

    def clear(self) -> None:
        self.keys = self.valid = self.last_idx = self.last_scores = self.last_head_scores = None

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum()) if self.valid is not None else 0

    # ---- scoring
    def head_scores(self, q: torch.Tensor) -> torch.Tensor:
        """q [B, n_q, T, hd] -> per-head scores [B, n_q, T, S] (GQA grouping, no RoPE either side)."""
        g = self.geo
        K = _on(self.keys, q)
        B, _, T, hd = q.shape
        qh = q.float().reshape(B, g["n_kv"], g["groups"], T, hd)
        s = torch.einsum("bkgtd,skd->bkgts", qh, K)
        return s.reshape(B, g["n_q"], T, -1)

    def pool_scores(self, s: torch.Tensor, query: str | None = None, pool: str | None = None) -> torch.Tensor:
        """[B, n_q, T, S] -> [B, S]: positions per `query` (last | all_max | all_mean), heads per `pool`."""
        query, pool = query or self.query, pool or self.pool
        if query == "last":
            s = s[:, :, -1]
        elif query == "all_max":
            s = s.amax(2)
        elif query == "all_mean":
            s = s.mean(2)
        else:
            raise ValueError(f"unknown selection query {query}")
        s = s.amax(1) if pool == "max" else s.mean(1)
        return s.masked_fill(~_on(self.valid, s).unsqueeze(0), float("-inf"))

    def select_from_input(self, x: torch.Tensor) -> torch.Tensor:
        """x = input of the selection layer's attention (post input_layernorm) [B, T, d]."""
        T = x.shape[1]
        pos = self.pos if self.pos >= 0 else T + self.pos
        q = reader_queries(self.att, x[:, : pos + 1], self.geo)       # [B, n_q, pos+1, hd]
        if self.query == "last" and not self.keep_heads:
            q = q[:, :, -1:]
        s = self.head_scores(q)
        if self.keep_heads:
            self.last_head_scores = s.detach()
        sc = self.pool_scores(s)
        k = min(self.k, self.n_valid)
        self.last_scores = sc.detach()
        self.last_idx = sc.topk(k, -1).indices
        return self.last_idx

    # ---- stats
    def gold_hit(self, gold_flat: torch.Tensor, ks=None) -> dict:
        """gold_flat [C·N] bool. -> {k: {hit, n_gold_sel, rank_best, chance_any, chance_frac}} for batch row 0."""
        sc = self.last_scores[0]
        g = _on(gold_flat.reshape(-1).bool(), sc)[: sc.numel()]
        v = _on(self.valid, sc)
        g = g & v
        n, ng = int(v.sum()), int(g.sum())
        if ng == 0:
            return {}
        order = sc.argsort(descending=True)
        rank_of = torch.empty_like(order)
        rank_of[order] = torch.arange(order.numel(), device=order.device)
        best = int(rank_of[g].min())
        out = {"n_valid": n, "n_gold": ng, "best_rank": best, "best_rank_frac": best / max(n - 1, 1)}
        for k in (ks or [self.k]):
            k = min(int(k), n)
            top = order[:k]
            hits = int(g[top].sum())
            # P(top-k random subset contains >= 1 gold) = 1 - C(n-ng, k)/C(n, k)
            lp = (math.lgamma(n - ng + 1) - math.lgamma(n - ng - k + 1) - math.lgamma(n + 1) + math.lgamma(n - k + 1)) \
                if n - ng >= k else float("-inf")
            out[k] = {"hit": float(hits > 0), "n_gold_sel": hits, "chance_any": 1.0 - math.exp(lp),
                      "chance_frac": k * ng / n}
        return out


# ============================================================================ injector
class TopKLibraryInjector(LibraryInjector):
    def __init__(self, model, interface, selector: TopKSelector, read_softmax: bool = True,
                 read_temp: float = 12.0, route_temp: float = 10.0, top_k: int = 0, layer_selector=None,
                 read_fp32: bool = True):
        self.tk = selector
        self._selecting = False
        self._sel_this_forward = False
        self.sel_pos = -1                     # prompt position whose query selects (training: -(m+1))
        self.n_select_passes = 0              # number of selection pre-passes run (decode reuse check)
        super().__init__(model, interface, read_softmax=read_softmax, read_temp=read_temp,
                         route_temp=route_temp, top_k=top_k, selector=layer_selector, read_fp32=read_fp32)

    @property
    def last_idx(self):
        return self.tk.last_idx

    def _active(self) -> bool:
        return (not self._selecting) and super()._active()

    # ---- load / unload
    def load_library(self, lib) -> None:
        super().load_library(lib)
        if getattr(lib, "sel_K", None) is None:
            raise ValueError("TopKLibraryInjector needs a TopKLibrary (residual library + sel_K attention keys)")
        first = self.itf.inject_layers[0]
        self.tk.set_keys(lib.sel_K, lib.mask[first])
        self.tk.last_idx = None

    def unload(self) -> None:
        super().unload()
        self.tk.clear()

    @property
    def full_read(self) -> bool:
        return self.tk.k >= self.tk.n_valid

    # ---- hooks
    def _attach(self) -> None:
        super()._attach()
        blocks = decoder_layers(self.model)

        def att_pre(mod, args, kwargs):
            if self._selecting:
                x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                self.tk.pos = self.sel_pos
                with torch.no_grad():
                    self.tk.select_from_input(x)
                raise _StopSelect

        self._handles.append(self.tk.att.register_forward_pre_hook(att_pre, with_kwargs=True))

        def top_pre(mod, args, kwargs):           # HF path: every model(...) call (prefill / decode / train)
            if self._selecting or not self._active():
                return None
            x = kwargs.get("input_ids")
            if x is None:
                x = kwargs.get("inputs_embeds")
            if x is None and args:
                x = args[0]
            self._sel_this_forward = True
            if x is None or self.full_read:
                return None
            if x.shape[1] == 1 and self.tk.last_idx is not None:
                return None                        # decode step: reuse the selected set
            kw = dict(kwargs)
            if "past_key_values" in kw:
                kw["past_key_values"] = None
            if "use_cache" in kw or hasattr(self.model, "generation_config"):
                kw["use_cache"] = False
            kw.pop("logits_to_keep", None)
            self._run_select(lambda: mod(*args, **kw))
            return None

        self._handles.append(self.model.register_forward_pre_hook(top_pre, with_kwargs=True))

        from xenc.backbone import is_fake
        fake = is_fake(self.model)

        def block0_pre(mod, args):                 # fake models: logits() bypasses model.__call__
            if self._selecting or not self._active():
                return None
            if not fake:                           # HF: selection happens in top_pre (never in grad-ckpt recompute)
                self._sel_this_forward = False
                return None
            if self._sel_this_forward:
                self._sel_this_forward = False
                return None
            if self.full_read:
                return None
            x = args[0]
            if x.shape[1] == 1 and self.tk.last_idx is not None:
                return None
            self._run_select(lambda: self.model.forward(x))
            return None

        self._handles.append(blocks[0].register_forward_pre_hook(block0_pre))

    def _run_select(self, fn) -> None:
        self._selecting = True
        self.n_select_passes += 1
        try:
            with torch.no_grad():
                fn()
        except _StopSelect:
            pass
        finally:
            self._selecting = False
        if self.tk.last_idx is None:
            raise RuntimeError("D8 selection pass did not reach the selection layer")

    # ---- branch
    def _branch(self, h: torch.Tensor, li: int) -> torch.Tensor:
        if self.lib is None or self.full_read:
            return super()._branch(h, li)
        idx = self.tk.last_idx                                       # [B', k]
        hn = rms_norm(h.float())
        q = itf_query(self.itf, hn)                                  # [B,T,d_s]
        K, M, _ = self._lib_layer(li, hn)
        Kf = K.reshape(-1, K.shape[-1])
        Vf = self._lib_values(li, hn).reshape(-1, K.shape[-1])
        idx = _on(idx, hn)
        if idx.shape[0] != q.shape[0]:
            idx = idx[:1].expand(q.shape[0], -1)
        Ks, Vs = Kf[idx], Vf[idx]                                    # [B,k,d_s]
        a = torch.einsum("btd,bkd->btk", q, Ks)
        if self.read_softmax:
            a = torch.softmax(a * self.read_temp, -1)
        else:
            a = a / (idx.shape[-1] ** 0.5)
        C, N = M.shape
        if self.record_attn:            # [C,N] read attention at the last position (zero off the selected set)
            full = torch.zeros(C * N, dtype=a.dtype, device=a.device).index_put((idx[0],), a[0, -1].detach())
            self.last_attn[li] = full.view(C, N).float().cpu()
        if self.training_attn and torch.is_grad_enabled():
            full = torch.zeros(C * N, dtype=a.dtype, device=a.device).index_put((idx[0],), a[0, self.attn_pos])
            self.attn_terms.append(full.view(C, N).float())
        r = torch.einsum("btk,bkd->btd", a, Vs)
        g = itf_gate(self.itf, hn, li)
        delta = self.gain_scale * itf_gain(self.itf, li, hn) * g * itf_write(self.itf, r)
        self.last_route = None
        if self.selector is not None:
            delta = self.selector.apply(delta, h, li)
        return delta


# ============================================================================ script glue
def sel_layer_for(model, sel_rho: float) -> int:
    return int(attn_layer_map(model, [float(sel_rho)])[float(sel_rho)])


def build_topk(model, rt, args, itf):
    """register_doc / eval_doc: attach rt.topk and build the selector + injector. -> injector."""
    sel_rho = float(getattr(args, "select_layer_rho", 0.25))
    rt.topk = TopKSpec(sel_rho, getattr(args, "sel_harvest", "attn") or "attn")
    kmap = load_sel_kmap(args.sel_kmap, sel_rho) if getattr(args, "sel_kmap", None) else None
    sel = TopKSelector(model, sel_layer_for(model, sel_rho), k=int(args.topk),
                       pool=getattr(args, "sel_pool", "max") or "max",
                       query=getattr(args, "sel_query", "last") or "last", kmap=kmap)
    return TopKLibraryInjector(model, itf, sel, read_softmax=not getattr(args, "linear_read", False),
                               read_temp=args.read_temp, route_temp=args.route_temp, top_k=args.top_k)


def add_topk_args(ap) -> None:
    ap.add_argument("--topk", type=int, default=64, help="D8 (--read topk): slots selected per forward")
    ap.add_argument("--select_layer_rho", type=float, default=0.25,
                    help="D8: depth of the selection attention layer (reader queries vs encoder attention keys)")
    ap.add_argument("--sel_harvest", choices=["attn", "attn_raw"], default="attn",
                    help="D8: selection-key cache (attn = *_hattn.pt, L2 rows; attn_raw = *_hattn_raw.pt)")
    ap.add_argument("--sel_pool", choices=["max", "mean"], default="max", help="D8: pool scores over heads")
    ap.add_argument("--sel_query", choices=["last", "all_max", "all_mean"], default="last",
                    help="D8: query position(s): last prompt position, or pooled over all prompt positions")
    ap.add_argument("--sel_kmap", default=None,
                    help="D8 S/L: encoder->reader key map (scripts/fit_kv_map.py output) applied to the selection keys")


__all__ = ["TopKSelector", "TopKLibraryInjector", "TopKLibrary", "TopKSpec", "attach_sel_keys", "build_topk",
           "add_topk_args", "sel_layer_for", "load_sel_kmap", "attn_geometry", "reader_queries"]
