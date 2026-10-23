"""D6 diagnostics: shared helpers (content-presence / oracle-read / linear probes).

Everything here is read-only over the existing encoder cache: a document's slots are the cached
L2-normalized residual rows (fp16, rho-keyed, chunked), concatenated over chunks to [T, d] per depth
and optionally centered exactly like xenc.library.Library (slot -> L2(slot - mu[rho])).
"""
from __future__ import annotations

import bisect
import glob
import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import xenc  # noqa: E402,F401
from xenc.depth_skill import _key, cache_key, load_doc_cache  # noqa: E402

HF_HUB = os.environ.get("HF_HUB_CACHE", "/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home/hub")


def l2(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


def load_center(path: str | None) -> dict | None:
    if not path:
        return None
    return {float(k): v.float() for k, v in torch.load(path, map_location="cpu", weights_only=True)["center"].items()}


def center_rows(x: torch.Tensor, center: dict | None, rho: float) -> torch.Tensor:
    """x: L2-normalized rows [.., d] -> L2(x - mu[rho]) (identity + L2 if no center)."""
    x = x.float()
    if center is None:
        return l2(x)
    mu = center[_key(center, rho)].to(x.device)
    return l2(x - mu)


def doc_cache(docs: dict, doc_id: str, enc_short: str, cache_dir: str, depths, chunk_tokens: int = 512):
    path = os.path.join(cache_dir, cache_key(docs[doc_id], enc_short, chunk_tokens, depths) + ".pt")
    return load_doc_cache(path) if os.path.exists(path) else None


def doc_states(cache: dict, center: dict | None, depths, device="cpu") -> tuple[dict, list[int], torch.Tensor]:
    """-> ({rho: [T,d] centered L2 slots}, chunk starts, chunk_of_token [T])."""
    out = {}
    for rho in depths:
        x = torch.cat([c["K"][_key(c["K"], rho)] for c in cache["chunks"]], 0).to(device)
        out[float(rho)] = center_rows(x, center, rho)
    starts = [int(c.get("start", 0)) for c in cache["chunks"]]
    cid = torch.cat([torch.full((int(c["n_tokens"]),), i, dtype=torch.long) for i, c in enumerate(cache["chunks"])])
    return out, starts, cid


def chunk_summaries(states: dict, cid: torch.Tensor) -> dict:
    """{rho: [C,d]} = L2(mean of the chunk's (centered) slots) — xenc.library summary keys."""
    C = int(cid.max()) + 1
    out = {}
    for rho, x in states.items():
        s = torch.zeros(C, x.shape[-1], device=x.device).index_add_(0, cid.to(x.device), x)
        out[rho] = l2(s / torch.bincount(cid, minlength=C).to(x.device).unsqueeze(-1).float())
    return out


def doc_offsets(tok, text: str) -> list[tuple[int, int]]:
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return [tuple(o) for o in enc["offset_mapping"]]


def string_token_mask(text: str, strings, offs, T: int | None = None, min_len: int = 2) -> torch.Tensor:
    """bool [T]: tokens overlapping any case-insensitive occurrence of any of `strings`."""
    T = len(offs) if T is None else T
    m = torch.zeros(T, dtype=torch.bool)
    ends = [b for (_, b) in offs]
    for s in strings or []:
        s = str(s).strip()
        if len(s) < min_len:
            continue
        for mt in re.finditer(re.escape(s), text, flags=re.IGNORECASE):
            a, b = mt.start(), mt.end()
            i = bisect.bisect_right(ends, a)          # first token whose end > a
            while i < min(T, len(offs)) and offs[i][0] < b:
                m[i] = True
                i += 1
    return m


def first_token_positions(text: str, s: str, offs, T: int) -> list[int]:
    """Token index of the first token of every case-insensitive occurrence of `s`."""
    s = str(s).strip()
    if len(s) < 2:
        return []
    ends = [b for (_, b) in offs]
    out = []
    for mt in re.finditer(re.escape(s), text, flags=re.IGNORECASE):
        i = bisect.bisect_right(ends, mt.start())
        if i < T:
            out.append(i)
    return out


def choice_masks(item, text: str, offs, T: int) -> dict:
    """{label: bool[T]} for each MCQ choice text; the gold answer mask is masks[correct_label]."""
    out = {}
    for c in item.get("choices") or []:
        out[c["label"]] = string_token_mask(text, [c["text"]], offs, T)
    return out


def gold_mask(item, text: str, offs, T: int) -> torch.Tensor:
    ans = list(item.get("answers") or [])
    if item.get("choices") and item.get("correct_label"):
        ans += [c["text"] for c in item["choices"] if c["label"] == item["correct_label"]]
    return string_token_mask(text, ans, offs, T)


def wrong_doc_id(doc_id: str, doc_order: list[str], shift: int = 7) -> str:
    """Same rule as eval_doc.wrong_doc_id."""
    n = len(doc_order)
    i = doc_order.index(doc_id)
    for s in range(shift, shift + n):
        j = doc_order[(i + s) % n]
        if j != doc_id:
            return j
    raise ValueError("need >= 2 docs")


def topk_chance(T: int, g: int, k: int) -> float:
    """P(at least one of g gold among T in a uniformly random k-subset)."""
    if g <= 0:
        return 0.0
    p = 1.0
    for i in range(min(k, T)):
        p *= max(0, T - g - i) / (T - i)
    return 1.0 - p


def load_embeddings(hub: str) -> torch.Tensor:
    """Input embedding matrix [V, d] (fp32, CPU) straight from the safetensors snapshot."""
    from safetensors import safe_open
    base = os.path.join(HF_HUB, "models--" + hub.replace("/", "--"), "snapshots")
    idx = glob.glob(os.path.join(base, "*", "model.safetensors.index.json"))
    if not idx:
        raise FileNotFoundError(f"no safetensors index under {base}")
    wm = json.load(open(idx[0]))["weight_map"]
    key = next(k for k in wm if k.endswith("embed_tokens.weight"))
    with safe_open(os.path.join(os.path.dirname(idx[0]), wm[key]), framework="pt") as f:
        return f.get_tensor(key).float()


def load_items(split_dir: str, split: str, fold: int | None) -> list[dict]:
    items = json.load(open(os.path.join(split_dir, split, "items.json")))
    if fold is not None:
        items = [i for i in items if (i.get("meta") or {}).get("fold") == fold]
    return items


__all__ = ["l2", "load_center", "center_rows", "doc_cache", "doc_states", "chunk_summaries", "doc_offsets",
           "string_token_mask", "first_token_positions", "choice_masks", "gold_mask", "wrong_doc_id",
           "topk_chance", "load_embeddings", "load_items"]
