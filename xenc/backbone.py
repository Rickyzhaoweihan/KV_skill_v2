"""Frozen reader / encoder loading + chat / decoding helpers (agent [D]).

Real backbones go through HF transformers (imported lazily, so the --tiny / CPU-test paths never
import model code). Fake backbones (`tests/_tiny.TinyLM`: no `.generate`, forward takes float
embeddings [B,T,d]) are supported by every helper here so scripts can run end-to-end on CPU.

    model, tok = load_reader("Qwen/Qwen3.5-9B", device_map="cuda", grad_ckpt=True)
    ids = build_chat(tok, hub, system, user)          # list[int], enable_thinking=False for Qwen
    text = generate(model, tok, hub, ids, 64)          # greedy, stops on </answer>, final channel

Conventions ported from deltaskill.backbone_utils (chat_template_kwargs / final_channel /
skip_special_on_decode) and cross_model/xskill/multi_backbone.py (device_map + max_memory,
grad-ckpt only active in train(); generate() must run in eval() so the KV cache is used).
"""

from __future__ import annotations

import torch

import xenc  # noqa: F401  (sys.path bootstrap for deltaskill)
from deltaskill.backbone_utils import (chat_template_kwargs, final_channel, is_gpt_oss,
                                        skip_special_on_decode)


def short_name(hub: str) -> str:
    return str(hub).rstrip("/").split("/")[-1]


def is_fake(model) -> bool:
    """TinyLM-style test model: no HF `generate`, forward(x: float [B,T,d])."""
    return not hasattr(model, "generate")


# ----------------------------------------------------------------------------- loading
def _device_kwargs(device_map: str, max_mem: str | None) -> dict:
    dm = str(device_map or "cuda")
    if dm == "auto" or "+" in dm:
        kw = {"device_map": "auto"}
        if dm == "auto":
            n = torch.cuda.device_count()
            gpus = list(range(n))
        else:
            gpus = [int(d.split(":")[1]) if ":" in d else 0 for d in dm.split("+") if d]
        if max_mem:
            kw["max_memory"] = {g: max_mem for g in gpus}
        elif dm != "auto":
            # restrict placement to the listed GPUs (accelerate needs an explicit budget per device)
            kw["max_memory"] = {g: f"{int(torch.cuda.get_device_properties(g).total_memory * 0.92 / 2**30)}GiB"
                                for g in gpus}
        return kw
    if dm == "cuda":
        dm = "cuda:0"
    return {"device_map": {"": dm}}


def _load(hub: str, dtype=torch.bfloat16, device_map: str = "cuda", max_mem: str | None = None,
          grad_ckpt: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = {"dtype": dtype, **_device_kwargs(device_map, max_mem)}
    try:
        m = AutoModelForCausalLM.from_pretrained(hub, **kw)
    except Exception:   # unified / multimodal wrappers (deltaskill.backbone_utils.load_backbone)
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(hub, **kw)
    m.eval().requires_grad_(False)
    m._xenc_grad_ckpt = bool(grad_ckpt)
    m._xenc_hub = hub
    if grad_ckpt:
        # transformers only checkpoints in train(); see train_mode()/eval_mode().
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tok = AutoTokenizer.from_pretrained(hub)
    return m, tok


def load_reader(hub: str, dtype=torch.bfloat16, device_map: str = "cuda", max_mem: str | None = None,
                grad_ckpt: bool = False):
    """-> (model, tokenizer). Frozen (eval(), requires_grad False on every parameter).

    device_map: "cuda" / "cuda:N" / "cpu" (single device), "auto" (all visible GPUs) or
    "cuda:0+cuda:1" (shard over the listed GPUs). max_mem: per-GPU budget for sharded loads
    (e.g. "40GiB" on L40S, "76GiB" on H100)."""
    return _load(hub, dtype=dtype, device_map=device_map, max_mem=max_mem, grad_ckpt=grad_ckpt)


def load_encoder(hub: str, **kw):
    """Same loader as the reader (the encoder is only ever run under no_grad)."""
    kw.setdefault("grad_ckpt", False)
    return _load(hub, **kw)


def train_mode(model) -> None:
    """Enable activation checkpointing for a TRAINING forward (no-op unless loaded with grad_ckpt)."""
    if getattr(model, "_xenc_grad_ckpt", False):
        model.train()


def eval_mode(model) -> None:
    model.eval()


def input_device(model):
    if is_fake(model):
        return model.device
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return model.device


# ----------------------------------------------------------------------------- tiny embedding
_TINY_EMB: dict[int, object] = {}


def embed_ids(model, ids: torch.Tensor) -> torch.Tensor:
    """Fake models only: int ids [B,T] -> float embeddings [B,T,d] via a fixed random table
    (same table as xenc.depth_skill.tiny_embed_fn, cached per model)."""
    if ids.is_floating_point():
        return ids
    fn = _TINY_EMB.get(id(model))
    if fn is None:
        from xenc.depth_skill import tiny_embed_fn
        fn = _TINY_EMB[id(model)] = tiny_embed_fn(model, vocab=int(getattr(model, "vocab", 50)))
    return fn(ids)


# ----------------------------------------------------------------------------- chat
def _to_list(r) -> list[int]:
    if hasattr(r, "keys"):
        r = r["input_ids"]
    if isinstance(r, torch.Tensor):
        r = r.tolist()
    if r and isinstance(r[0], (list, tuple)):
        r = r[0]
    return [int(x) for x in r]


def build_chat(tok, hub: str, system: str | None, user: str, add_generation_prompt: bool = True) -> list[int]:
    """Chat-template token ids. Qwen/GLM get enable_thinking=False; gpt-oss reasoning_effort=low.
    A tokenizer without a chat template (the --tiny fake) gets `system\\n\\nuser` plain-encoded."""
    if not hasattr(tok, "apply_chat_template"):
        return _to_list(tok(f"{system or ''}\n\n{user}", add_special_tokens=False))
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    r = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=add_generation_prompt,
                                **chat_template_kwargs(hub))
    return _to_list(r)


def text_ids(tok, text: str) -> list[int]:
    return _to_list(tok(text, add_special_tokens=False))


def with_target(tok, hub: str, system: str | None, user: str, target: str, device=None):
    """-> (ids [1, P+m] LongTensor, m, P): prompt followed by the teacher-forced target."""
    p = build_chat(tok, hub, system, user)
    a = text_ids(tok, target)
    ids = torch.tensor([p + a], dtype=torch.long, device=device)
    return ids, len(a), len(p)


def decode(tok, hub: str, ids) -> str:
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    if not hasattr(tok, "decode"):                      # --tiny fake tokenizer: opaque tokens
        return " ".join(f"t{int(i)}" for i in ids)
    raw = tok.decode(ids, skip_special_tokens=skip_special_on_decode(hub))
    return final_channel(raw, hub)


# ----------------------------------------------------------------------------- generation
def _stop_criteria(tok, hub, plen, stop):
    from transformers import StoppingCriteria, StoppingCriteriaList
    oss = is_gpt_oss(hub)

    class _Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            if oss:   # only stop once the FINAL channel has begun (the analysis trace may quote tags)
                txt = tok.decode(input_ids[0, plen:], skip_special_tokens=False)
                return "<|channel|>final" in txt and stop in txt.rsplit("<|channel|>final", 1)[1]
            return stop in tok.decode(input_ids[0, plen:][-12:], skip_special_tokens=True)

    return StoppingCriteriaList([_Stop()])


@torch.no_grad()
def generate(model, tok, hub: str, ids, max_new_tokens: int, stop: str = "</answer>",
             return_n: bool = False):
    """Greedy decode. Returns the decoded FINAL text (or (text, n_generated) if return_n)."""
    if isinstance(ids, (list, tuple)):
        ids = torch.tensor([list(ids)], dtype=torch.long)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if is_fake(model):
        cur = ids.to(model.device)
        for _ in range(max_new_tokens):
            nxt = model.logits(embed_ids(model, cur))[:, -1].argmax(-1, keepdim=True)
            cur = torch.cat([cur, nxt], 1)
        new = cur[0, ids.shape[1]:]
        txt = decode(tok, hub, new)
        return (txt, int(new.numel())) if return_n else txt
    eval_mode(model)
    ids = ids.to(input_device(model))
    plen = ids.shape[1]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                         top_p=None, top_k=None, pad_token_id=pad,
                         stopping_criteria=_stop_criteria(tok, hub, plen, stop))
    new = out[0, plen:]
    txt = decode(tok, hub, new)
    if stop and stop in txt:
        txt = txt.split(stop, 1)[0] + stop
    return (txt, int(new.numel())) if return_n else txt


__all__ = ["short_name", "load_reader", "load_encoder", "train_mode", "eval_mode", "build_chat",
           "text_ids", "with_target", "decode", "generate", "embed_ids", "is_fake", "input_device"]
