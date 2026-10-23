"""Shared CPU fixtures for xenc tests: a tiny fake decoder with `model.layers` (what
ExternalSkillInjector hooks) and synthetic ρ-keyed / layer-keyed skills shaped like
derive_by_depth output. Import from tests; do not copy.

    from _tiny import TinyLM, make_skill_layers, make_skill_rho, FakeConfig
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
import xenc  # noqa: E402,F401  (bootstraps delta-skill on sys.path)
from deltaskill.skill import KVSkill  # noqa: E402


class FakeConfig:
    def __init__(self, n_layers: int, hidden: int):
        self.num_hidden_layers = n_layers
        self.hidden_size = hidden
        self._name_or_path = f"tiny-L{n_layers}-d{hidden}"

    def get_text_config(self):
        return self


class _Stack(nn.Module):
    def __init__(self, d: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(d, d) for _ in range(n_layers))


class TinyLM(nn.Module):
    """Just enough of a decoder for the injectors: model.layers, .config, .device, forward(x)->x.

    forward returns the final hidden state [B,T,d]; `logits(x)` maps it to a 50-token vocab.
    """

    def __init__(self, d: int = 32, n_layers: int = 8, vocab: int = 50, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.model = _Stack(d, n_layers)
        self.config = FakeConfig(n_layers, d)
        self.lm_head = nn.Linear(d, vocab)
        self.d, self.n_layers, self.vocab = d, n_layers, vocab

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = torch.tanh(layer(x))
        return x

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.forward(x))


def unit_rows(n: int, d: int, g: torch.Generator) -> torch.Tensor:
    h = torch.randn(n, d, generator=g)
    return h / h.norm(dim=-1, keepdim=True)


def make_skill_layers(d: int, layers: list[int], n: int = 16, seed: int = 0) -> KVSkill:
    """Reader-layer-keyed skill (legacy deltaskill layout), K == V, unit rows."""
    g = torch.Generator().manual_seed(seed)
    K = {li: unit_rows(n, d, g) for li in layers}
    return KVSkill(K=K, V={li: t.clone() for li, t in K.items()}, d_s=d, text=f"skill{seed}",
                   model_name="tiny", meta={"num_tokens": n})


def make_skill_rho(d_s: int, depths: list[float], n: int = 16, seed: int = 0,
                   encoder: str = "tiny-enc") -> KVSkill:
    """ρ-keyed skill as produced by xenc.depth_skill (K is V, fp32 here for tests)."""
    g = torch.Generator().manual_seed(seed)
    K = {float(r): unit_rows(n, d_s, g) for r in depths}
    return KVSkill(K=K, V=K, d_s=d_s, text=f"doc{seed}", model_name=encoder,
                   meta={"keyed_by": "rho", "depths": sorted(float(r) for r in depths),
                         "tied": True, "encoder": encoder, "num_tokens": n})


# ----------------------------------------------------------------------------- D3: fake attention
class _FakeAttn(nn.Module):
    """Minimal attention stand-in: k_proj / v_proj ([d] -> [n_kv·hd]) + o_proj, so the harvest="attn"
    hook path (xenc.depth_skill._derive_attn) is exercised. Mixes values causally (prefix mean)."""

    def __init__(self, d: int, n_kv: int = 2, head_dim: int = 8):
        super().__init__()
        self.k_proj = nn.Linear(d, n_kv * head_dim)
        self.v_proj = nn.Linear(d, n_kv * head_dim)
        self.o_proj = nn.Linear(n_kv * head_dim, d)

    def forward(self, x):
        self.k_proj(x)
        v = self.v_proj(x)
        T = v.shape[1]
        csum = v.cumsum(1) / torch.arange(1, T + 1, device=v.device, dtype=v.dtype).view(1, T, 1)
        return self.o_proj(csum)


class _AttnBlock(nn.Module):
    def __init__(self, d: int, full: bool, n_kv: int, head_dim: int):
        super().__init__()
        self.mlp = nn.Linear(d, d)
        if full:
            self.self_attn = _FakeAttn(d, n_kv, head_dim)
        else:
            self.linear_attn = nn.Linear(d, d)       # Gated-DeltaNet stand-in: no k_proj / v_proj

    def forward(self, x):
        mix = self.self_attn(x) if hasattr(self, "self_attn") else self.linear_attn(x)
        return torch.tanh(self.mlp(x) + 0.5 * mix)


class TinyAttnLM(TinyLM):
    """TinyLM whose blocks carry attention modules; `full_every=k` makes only layers k-1, 2k-1, … full
    attention (Qwen3.5 hybrid pattern with k=4), the rest linear-attention stand-ins."""

    def __init__(self, d: int = 32, n_layers: int = 8, vocab: int = 50, seed: int = 0,
                 full_every: int = 1, n_kv: int = 2, head_dim: int = 8):
        super().__init__(d, n_layers, vocab, seed)
        torch.manual_seed(seed + 100)
        self.model.layers = nn.ModuleList(
            _AttnBlock(d, (i + 1) % full_every == 0, n_kv, head_dim) for i in range(n_layers))


# ----------------------------------------------------------------------------- D7: tiny REAL Qwen3.5
def tiny_qwen35(n_layers: int = 8, d: int = 64, n_heads: int = 4, n_kv: int = 2, head_dim: int = 16,
                vocab: int = 100, seed: int = 0):
    """Randomly initialised transformers Qwen3_5ForCausalLM (hybrid: full attention every 2nd layer,
    i.e. layers 1,3,5,7), fp32, CPU, eager attention. No checkpoint download. Used by D7 tests to exercise
    the real Qwen3_5Attention (gated q_proj, q_norm/k_norm, GQA, partial RoPE)."""
    from transformers.models.qwen3_5 import configuration_qwen3_5 as C, modeling_qwen3_5 as M
    torch.manual_seed(seed)
    types = ["linear_attention" if i % 2 == 0 else "full_attention" for i in range(n_layers)]
    cfg = C.Qwen3_5TextConfig(hidden_size=d, intermediate_size=2 * d, num_hidden_layers=n_layers,
                              num_attention_heads=n_heads, num_key_value_heads=n_kv, head_dim=head_dim,
                              vocab_size=vocab, layer_types=types, linear_num_key_heads=2,
                              linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16)
    cfg._attn_implementation = "eager"
    m = M.Qwen3_5ForCausalLM(cfg).eval().requires_grad_(False)
    with torch.no_grad():   # non-trivial norms (Qwen3.5 RMSNorm weights init at 0 => scale 1)
        for n, p in m.named_parameters():
            if n.endswith("norm.weight"):
                p.normal_(0, 0.2)
    return m
