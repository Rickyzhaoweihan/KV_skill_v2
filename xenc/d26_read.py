"""D26: CROSS-FAMILY readers for the paper-form read. The encoder (Qwen3.5-4B), its operator (post-RoPE K/V of the
document prefix) and the D13 read (xenc/d13_read.D13Interface: A -> the ENCODER's frozen q/k/v/gate projections ->
softmax over [operator ; own rows] -> W_o -> B) are unchanged. What changes when the reader is NOT a Qwen:

1. Where the read goes. The read replaces the output of a reader attention MODULE (post o_proj, d_r) and reads that
   module's input (the reader's input-RMSNorm'd hidden state). Nothing inside the reader's attention is used, so
   MLA (DeepSeek-V2: compressed latent KV, 192/128 head dims), attention sinks (gpt-oss) and GLM-4's sandwich norm
   (post_self_attn_layernorm is applied AFTER the module, i.e. after the replaced output) need no special case.
   Layers: only the reader's FULL-attention layers are candidates (gpt-oss alternates sliding-128 / full; GLM-4 and
   DeepSeek-V2 are all full; Qwen3.5 every 4th). `pick_read_layers` maps each encoder read layer e (D13: 3,7,11,15,19
   of 32) to the reader full-attention layer nearest the same normalized depth e/(L_e-1)·(L_r-1) (== identity for
   Qwen3.5-9B, i.e. D13's map).
   The module signature differs per family (DeepSeek-V2: (hidden, attention_mask, past, position_embeddings)), so
   the patch is signature-agnostic (*args, **kwargs).
2. Tokenizers differ. Every token-aligned quantity (stage-1 teacher rows, ridge A / B targets, distillation rows)
   is aligned through CHARACTER coordinates: `coords` maps each token's end offset to (# content characters
   consumed) (+ a fraction in [0, .5] inside a chat-template run, so differently formatted templates align
   proportionally); `align` pairs reader token t with the LAST encoder token that has consumed no more text
   (causal: the teacher never sees text the student has not). Reader rows before the first encoder row (reader
   BOS / template header) are marked invalid and excluded from every loss.
3. The distillation loss over columns. The teacher's own-row columns are ENCODER tokens and the student's are reader
   tokens, so the per-column KL of D13 is undefined across tokenizers. `bucket` collapses all own-row columns into
   one "own" mass: KL over [operator columns (identical: the encoder's tokens) ; own] — defined over operator
   positions plus the total own-row mass, at aligned rows. relMSE(B o_s, B o_t) is taken at aligned rows.
4. Prompts. gpt-oss: harmony template + forced final channel ("<|channel|>final<|message|>" appended to every
   generation prompt, else the analysis channel eats the 32-token budget); GLM-4-0414: "[gMASK]<sop>" header,
   stops on <|user|>/<|endoftext|>; DeepSeek-V2-Chat: BOS + "User: … Assistant:". The read prompt (doc tail +
   question suffix) gets the reader's BOS header (`reader_bos`), whose rows keep their stock attention output
   (R.keep0) so the reader's attention-sink token is not overwritten.
5. Long text prompts on gpt-oss (eager-only attention with sinks) and DeepSeek-V2 (q/k 192 ≠ v 128): the eager
   attention function is replaced by an exact query-chunked version (`chunked_eager`), bounding memory at 16k tokens.
"""

from __future__ import annotations

import bisect
import types

import torch

import xenc  # noqa: F401

READERS = {"gptoss": "openai/gpt-oss-20b", "glm4": "zai-org/GLM-4-9B-0414", "glm4_32b": "zai-org/GLM-4-32B-0414",
           "dsv2": "deepseek-ai/DeepSeek-V2-Lite-Chat", "qwen9b": "Qwen/Qwen3.5-9B"}
GPT_OSS_FINAL = "<|channel|>final<|message|>"
ENC_READ_LAYERS = (3, 7, 11, 15, 19)


def resolve_hub(name: str) -> str:
    return READERS.get(name, name)


def is_gpt_oss(hub: str) -> bool:
    from deltaskill.backbone_utils import is_gpt_oss as f
    return f(hub)


# ============================================================================ structure
def _tc(model):
    c = model.config
    return c.get_text_config() if hasattr(c, "get_text_config") else c


def reader_dims(model_or_cfg) -> dict:
    """{"n_layers", "full", "d", "family"}: full-attention layers from config.layer_types (sliding_attention and
    linear_attention excluded); a config without layer_types is all full attention (GLM-4, DeepSeek-V2)."""
    c = _tc(model_or_cfg) if hasattr(model_or_cfg, "config") else model_or_cfg
    n = int(c.num_hidden_layers)
    lt = getattr(c, "layer_types", None)
    full = [i for i, t in enumerate(lt) if t == "full_attention"] if lt else list(range(n))
    return {"n_layers": n, "full": full, "d": int(c.hidden_size), "family": str(getattr(c, "model_type", ""))}


def pick_read_layers(r: dict, e: dict, enc_layers=ENC_READ_LAYERS) -> dict:
    """{reader_layer: encoder_layer}: each encoder read layer -> the nearest unused reader full-attention layer at the
    same normalized depth (ties -> the shallower layer)."""
    out, used = {}, set()
    for el in enc_layers:
        tgt = el / max(e["n_layers"] - 1, 1) * max(r["n_layers"] - 1, 1)
        cand = [x for x in sorted(r["full"], key=lambda x: (abs(x - tgt), x)) if x not in used]
        if not cand:
            raise ValueError("reader has fewer full-attention layers than read layers")
        used.add(cand[0])
        out[int(cand[0])] = int(el)
    return dict(sorted(out.items()))


# ============================================================================ prompts
def chat_ids(tok, hub: str, system, user) -> list[int]:
    """Generation prompt ids (xenc.backbone.build_chat) + gpt-oss forced final channel."""
    from xenc.backbone import build_chat, text_ids
    ids = build_chat(tok, hub, system, user)
    return ids + (text_ids(tok, GPT_OSS_FINAL) if is_gpt_oss(hub) else [])


def chat_str(tok, hub: str, system, user) -> str:
    from deltaskill.backbone_utils import chat_template_kwargs
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **chat_template_kwargs(hub))
    return s + (GPT_OSS_FINAL if is_gpt_oss(hub) else "")


def suffix_str(tok, hub: str, item, system: str) -> tuple[str, str]:
    """(suffix, qc): the rendered `text` prompt after the document (starts with "\\n\\n## Question\\n", ends with the
    model's assistant header; gpt-oss: + forced final channel) and its CONTENT part qc (the user text after the doc)."""
    from xenc.doc_tasks import ctx_prompt
    from xenc.state_skill import QSEP
    user = ctx_prompt(item, "\x00DOC\x00")
    s = chat_str(tok, hub, system, user)
    k = s.rfind(QSEP)
    qc = user[user.rfind(QSEP):]
    suf = s[k:]
    assert suf.startswith(qc), "question content not verbatim in the rendered suffix"
    return suf, qc


def reader_bos(tok, hub: str) -> str:
    """Leading special header of the reader's chat template (attention-sink token), '' if none."""
    if not hasattr(tok, "apply_chat_template"):
        return ""
    s = chat_str(tok, hub, None, "X")
    if s.startswith("[gMASK]<sop>"):
        return "[gMASK]<sop>"
    b = getattr(tok, "bos_token", None)
    if b and s.startswith(b):
        return b
    return ""


def stop_ids(tok, model=None) -> set:
    """End-of-turn ids: tokenizer eos + generation_config eos list + known end tokens of the four families."""
    s = set()
    if getattr(tok, "eos_token_id", None) is not None:
        s.add(int(tok.eos_token_id))
    gc = getattr(model, "generation_config", None) if model is not None else None
    e = getattr(gc, "eos_token_id", None) if gc is not None else None
    for x in ([e] if isinstance(e, int) else (e or [])):
        s.add(int(x))
    for t in ("<|im_end|>", "<|endoftext|>", "<|return|>", "<|end|>", "<|call|>", "<|user|>", "<|observation|>",
              "<｜end▁of▁sentence｜>"):
        try:
            i = tok.convert_tokens_to_ids(t)
            if isinstance(i, int) and i >= 0 and i != getattr(tok, "unk_token_id", None):
                s.add(i)
        except Exception:
            pass
    return s


def encode(tok, s: str) -> tuple[list[int], list[tuple[int, int]]]:
    e = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    return [int(x) for x in e["input_ids"]], [(int(a), int(b)) for a, b in e["offset_mapping"]]


# ============================================================================ character alignment
def coords(S: str, offsets, contents) -> list[float]:
    """Per token (end offset e into S): # content characters in S[:e]; inside a template run (characters of S not in
    any content span) + 0.5·(fraction of the run consumed). `contents` are located in order in S."""
    spans, pos = [], 0
    for c in contents:
        if not c:
            continue
        k = S.find(c, pos)
        if k < 0:
            raise ValueError(f"content not found in S: {c[:40]!r}")
        spans.append((k, k + len(c)))
        pos = k + len(c)
    starts = [a for a, _ in spans]
    out, prev = [], 0.0
    for _, e in offsets:
        e = int(e)
        n = sum(min(max(e - a, 0), b - a) for a, b in spans)
        frac = 0.0
        if e > 0:
            i = bisect.bisect_right(starts, e - 1) - 1         # span starting at or before char e-1
            inside = i >= 0 and spans[i][0] <= e - 1 < spans[i][1]
            if not inside:
                ra = spans[i][1] if i >= 0 else 0
                rb = spans[i + 1][0] if i + 1 < len(spans) else len(S)
                frac = 0.5 * (e - ra) / max(rb - ra, 1)
        c = max(prev, n + frac)
        out.append(c)
        prev = c
    return out


def align(c_r, c_e, eps: float = 1e-6):
    """-> (idx [T_r] long, valid [T_r] bool): reader row t <- the last encoder row j with c_e[j] <= c_r[t]."""
    idx, valid = [], []
    for t, c in enumerate(c_r):
        j = bisect.bisect_right(c_e, c + eps) - 1
        if j >= 0 and abs(c_e[j] - c) <= eps:
            # exact tie (several tokens ending on the same character, e.g. a multi-byte char split into byte tokens):
            # pair the k-th reader token of its tie run with the k-th encoder token of the encoder's tie run
            lo = bisect.bisect_left(c_e, c - eps)
            k = 0
            while t - k - 1 >= 0 and abs(c_r[t - k - 1] - c) <= eps:
                k += 1
            j = min(lo + k, j)
        valid.append(j >= 0)
        idx.append(max(j, 0))
    return torch.tensor(idx, dtype=torch.long), torch.tensor(valid, dtype=torch.bool)


def bucket(logits: torch.Tensor, N: int) -> torch.Tensor:
    """[..., N + P] masked scores -> [..., N + 1]: operator columns + logsumexp over the own-row columns."""
    own = torch.logsumexp(logits[..., N:], -1, keepdim=True)
    return torch.cat([logits[..., :N], own], -1)


def d26_terms(ls_logits, lt_logits, y_s, y_t, idx, valid, N: int, metrics: bool = False):
    """Cross-tokenizer D13 terms: student [Hkv,G,T_r,N+T_r], teacher [Hkv,G,T_e,N+T_e] scores; y_s [T_r,d], y_t
    [T_e,d]. Rows: valid reader rows vs their aligned encoder rows; columns: operator + bucketed own mass."""
    from xenc.d13_read import d13_terms
    v = valid.to(ls_logits.device)
    ii = idx.to(ls_logits.device)[v]
    ls = bucket(ls_logits, N)[:, :, v]
    lt = bucket(lt_logits, N)[:, :, ii.to(lt_logits.device)].to(ls.device)
    return d13_terms(ls, lt, y_s[v], y_t[ii.to(y_t.device)].to(y_s.device), N, metrics)


# ============================================================================ generic reader patch
class D26Reader:
    """D13Reader for any reader family (signature-agnostic patch of the reader's attention modules at itf.layers).
    Modes as D13Reader ("off" / "student" / "train" / "teacher" / "e2e") plus "cap" (record the stock module output
    in R.ocap[li] and its input in R.xcap[li]; output unchanged). In "train"/"teacher" R.inject[li] [T,d_r] replaces
    the output on rows where R.valid (others keep the stock output). Rows at absolute positions < R.keep0 always keep
    the stock output (reader BOS / attention-sink header)."""

    def __init__(self, model, itf):
        from deltaskill.backbone_utils import decoder_layers
        self.model, self.itf = model, itf
        L = decoder_layers(model)
        self.mods = {li: L[li].self_attn for li in itf.layers}
        self.mode = "off"
        self.op: dict = {}
        self.n_op = 0
        self.past: dict = {}
        self.rec: dict = {}
        self.inject: dict = {}
        self.xcap: dict = {}
        self.ocap: dict = {}
        self.valid = None
        self.keep0 = 0
        self.onpolicy = False
        self.need_logits = False
        self.grad = True
        self.ckpt = False
        self._orig = {}
        for li, m in self.mods.items():
            self._orig[li] = m.forward
            m.forward = types.MethodType(_d26_forward(self, li), m)

    def load(self, op: dict, n_op: int):
        self.op, self.n_op, self.past = op, int(n_op), {}

    def detach(self):
        for li, f in self._orig.items():
            self.mods[li].forward = f
        self._orig.clear()


def _d26_forward(R: D26Reader, li: int):
    def forward(self, *args, **kwargs):
        res = type(self).forward(self, *args, **kwargs)
        if R.mode == "off":
            return res
        out = res[0] if isinstance(res, tuple) else res
        hs = args[0] if args else kwargs["hidden_states"]
        itf = R.itf
        x = hs[0].float()
        own = out[0].float()
        T = x.shape[0]
        if R.mode == "cap":
            R.xcap[li], R.ocap[li] = hs[0].detach(), out[0].detach()
            return res
        e = itf.layer_map[li]
        K, V = R.op[e]
        P = 0
        if R.mode == "student":
            P = 0 if R.past.get(li) is None else int(R.past[li][0].shape[1])
            r = itf.read(li, x, K, V, R.n_op, past=R.past.get(li))
            R.past[li] = r["past"]
            y = own + itf.gate(li) * (r["y"] - own)
        elif R.mode == "teacher":
            R.xcap[li] = x.detach()
            y = _masked(R.inject[li], own, R.valid)
        elif R.mode == "e2e":
            if torch.is_grad_enabled() and R.ckpt:
                from torch.utils.checkpoint import checkpoint
                nl = bool(R.need_logits)

                def _rd(x_):
                    r_ = itf.read(li, x_, K, V, R.n_op, need_logits=nl)
                    return (r_["y"], r_["o"], r_["logits"]) if nl else (r_["y"], r_["o"])
                t_ = checkpoint(_rd, x, use_reentrant=False)
                r = {"y": t_[0], "o": t_[1], "logits": t_[2] if nl else None}
            else:
                r = itf.read(li, x, K, V, R.n_op, need_logits=R.need_logits)
            R.rec[li] = r
            y = own + itf.gate(li) * (r["y"] - own)
        else:                                                    # "train" (stage 1)
            with torch.set_grad_enabled(R.grad):
                r = itf.read(li, x, K, V, R.n_op, need_logits=True)
            R.rec[li] = r
            y = r["y"].detach() if R.onpolicy else _masked(R.inject[li], own, R.valid)
        if R.keep0 > P:
            n = min(R.keep0 - P, T)
            y = torch.cat([own[:n], y[n:]], 0)
        y = y.to(out.dtype).unsqueeze(0)
        return (y,) + tuple(res[1:]) if isinstance(res, tuple) else y

    return forward


def _masked(inj, own, valid):
    if valid is None:
        return inj
    v = valid.to(own.device).view(-1, 1)
    return torch.where(v, inj.to(own.dtype), own)


# ============================================================================ attention for long text prompts
def chunked_eager(module, query, key, value, attention_mask, scaling, dropout: float = 0.0, s_aux=None,
                  sliding_window=None, chunk: int = 1024, **kwargs):
    """Exact eager attention (fp32 softmax) computed in query chunks; GQA without repeat; optional sinks (s_aux,
    gpt-oss) and additive / boolean masks. -> ([B,T,H,Dv], None) like transformers' eager_attention_forward."""
    B, H, T, D = query.shape
    Hkv, S = key.shape[1], key.shape[2]
    G = H // Hkv
    qg = query.view(B, Hkv, G, T, D)
    outs = []
    for i in range(0, T, chunk):
        q = qg[:, :, :, i:i + chunk]
        t = q.shape[3]
        s = torch.einsum("bkgtd,bksd->bkgts", q.float(), key.float()) * scaling
        if attention_mask is not None:
            m = attention_mask[:, :, i:i + t, :S]
            m = m.unsqueeze(2) if m.dim() == 4 else m
            s = s.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else s + m.float()
        elif t > 1 or T > 1:
            row = torch.arange(i, i + t, device=s.device).view(t, 1) + (S - T)
            col = torch.arange(S, device=s.device).view(1, S)
            bad = col > row
            if sliding_window:
                bad = bad | (col <= row - int(sliding_window))
            s = s.masked_fill(bad, float("-inf"))
        if s_aux is not None:
            sk = s_aux.float().view(1, Hkv, G, 1, 1).expand(B, Hkv, G, t, 1)
            c = torch.cat([s, sk], -1)
            c = c - c.max(-1, keepdim=True).values
            p = torch.softmax(c, -1)[..., :-1]
        else:
            p = torch.softmax(s, -1)
        o = torch.einsum("bkgts,bksd->bkgtd", p.to(value.dtype), value)
        outs.append(o)
    o = torch.cat(outs, 3).reshape(B, H, T, -1).transpose(1, 2).contiguous()
    return o, None


def install_chunked_eager(family: str) -> bool:
    """Route the eager attention of gpt-oss / DeepSeek-V2 through chunked_eager (module-global patch, idempotent)."""
    import importlib
    mod = {"gpt_oss": "transformers.models.gpt_oss.modeling_gpt_oss",
           "deepseek_v2": "transformers.models.deepseek_v2.modeling_deepseek_v2"}.get(family)
    if mod is None:
        return False
    m = importlib.import_module(mod)
    if getattr(m.eager_attention_forward, "_d26", False):
        return True
    f = lambda *a, **k: chunked_eager(*a, **k)   # noqa: E731
    f._d26 = True
    m.eager_attention_forward = f
    return True


def load_reader_d26(hub: str, device: str = "cuda"):
    """Frozen bf16 reader + tokenizer. gpt-oss: MXFP4 dequantized to bf16 (no `kernels` package in envs/qwen35; bf16
    also keeps the read's gradients exact), eager + chunked attention. DeepSeek-V2: native transformers port
    (no trust_remote_code), eager + chunked. GLM-4 / Qwen: sdpa."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    fam = str(AutoConfig.from_pretrained(hub).model_type)
    kw = {"dtype": torch.bfloat16, "device_map": {"": "cuda:0" if device == "cuda" else device}}
    if fam == "gpt_oss":
        from transformers import Mxfp4Config
        kw["quantization_config"] = Mxfp4Config(dequantize=True)
    if fam in ("gpt_oss", "deepseek_v2"):
        install_chunked_eager(fam)
        kw["attn_implementation"] = "eager"
    elif fam != "qwen3_5":
        kw["attn_implementation"] = "sdpa"
    m = AutoModelForCausalLM.from_pretrained(hub, **kw)
    m.eval().requires_grad_(False)
    m._xenc_hub = hub
    return m, AutoTokenizer.from_pretrained(hub)


@torch.no_grad()
def greedy_ids(model, tok, hub: str, ids, budget: int, stop: str = "</answer>", stops: set | None = None):
    """Manual greedy decode (any family) -> (final-channel text, n generated)."""
    from xenc.backbone import decode, input_device
    dev = input_device(model)
    stops = stop_ids(tok, model) if stops is None else stops
    out = model(input_ids=torch.tensor([list(ids)], device=dev), use_cache=True, logits_to_keep=1)
    new = []
    for _ in range(budget):
        nxt = int(out.logits[0, -1].argmax())
        new.append(nxt)
        if nxt in stops or stop in tok.decode(new[-12:], skip_special_tokens=True):
            break
        out = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=out.past_key_values,
                    use_cache=True, logits_to_keep=1)
    txt = decode(tok, hub, new)
    if stop and stop in txt:
        txt = txt.split(stop, 1)[0] + stop
    return txt, len(new)


__all__ = ["READERS", "GPT_OSS_FINAL", "resolve_hub", "reader_dims", "pick_read_layers", "chat_ids", "chat_str",
           "suffix_str", "reader_bos", "stop_ids", "encode", "coords", "align", "bucket", "d26_terms", "D26Reader",
           "chunked_eager", "install_chunked_eager", "load_reader_d26", "greedy_ids"]
