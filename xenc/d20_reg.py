"""D20: per-document REGISTRATION of the paper-form read (the D13 read: the 9B reader's own state queries the 2B
encoder's fixed operator). Every document is registered (trained on, label-free) before it is read.

The operator (the encoder's harvested post-RoPE K/V of the document) is NEVER modified. The shared base interface
(D13: A, MLP, B, write-back bias, gates) is frozen; a small PER-DOCUMENT part is trained on label-free signals from
the document itself and then used for that document's questions. `DocAdapter` holds the per-document part:

    dA     rank-r delta on A at every read layer           x̃ = (A + Ua Va) RMSNorm(h) + MLP(...)
    dB     rank-r delta on the write-back B                 y = (B + Ub Vb) o + b
    bias   the write-back bias b (per layer, d_r)
    gate   the per-layer gate g (own + g·(read − own))
    kv     key/value-side adapter applied at READ time to a copy of the operator, per encoder layer and kv head:
           K' = K + (K Pk) Qk,  V' = V + (V Pv) Qv  (Q zero-init; the stored operator tensor is unchanged)

Every per-document part starts exactly at the base interface (zero-init deltas), so registration step 0 == kv_base.

Leakage guard: `assert_no_test_questions` (registration items vs every judged question string of the task) is called
by scripts/register_d20.py before any registration step, and `drop_test_matches` by scripts/d20_selfq.py.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

FORMAT = "xenc.d20_reg.v1"
PARTS = ("dA", "dB", "bias", "gate", "kv")


# ----------------------------------------------------------------------------- leakage guard
def norm_q(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def forbidden_strings(test_questions) -> tuple[set, list]:
    """(exact normalized test questions, long substrings: every normalized test question and every quoted span
    (e.g. a ContractNLI hypothesis) of ≥ 30 normalized characters)."""
    exact, subs = set(), set()
    for q in test_questions:
        n = norm_q(q)
        if not n:
            continue
        exact.add(n)
        if len(n) >= 30:
            subs.add(n)
        for m in re.findall(r"\"([^\"]{20,})\"", str(q)):
            nm = norm_q(m)
            if len(nm) >= 30:
                subs.add(nm)
    return exact, sorted(subs)


def leaks(question: str, exact: set, subs: list) -> bool:
    n = norm_q(question)
    return n in exact or any(s in n for s in subs)


def drop_test_matches(items: list, test_questions) -> tuple[list, int]:
    """Generator side: drop self-questions that coincide with (or contain) any judged question string."""
    exact, subs = forbidden_strings(test_questions)
    keep = [it for it in items if not leaks(it["question"], exact, subs)]
    return keep, len(items) - len(keep)


def assert_no_test_questions(reg_items: list, test_questions) -> int:
    """Registration side (hard assertion): no registration prompt equals or contains a judged question string,
    and every registration item is a D20 self-question / recitation item (never a dataset item)."""
    exact, subs = forbidden_strings(test_questions)
    for it in reg_items:
        meta = it.get("meta") or {}
        assert meta.get("d20selfq") or meta.get("d20recite"), f"non-D20 item in registration: {it.get('key')}"
        assert not leaks(it["question"], exact, subs), f"TEST QUESTION LEAK in registration item {it.get('key')}"
    return len(reg_items)


# ----------------------------------------------------------------------------- per-document adapter
class DocAdapter(nn.Module):
    """The per-document trainable part on top of a frozen D13Interface `itf` (see module doc)."""

    def __init__(self, itf, parts, rank: int = 16, kv_rank: int = 8, seed: int = 0):
        super().__init__()
        parts = set(parts)
        bad = parts - set(PARTS)
        assert not bad, f"unknown parts {bad}"
        self.itf, self.parts, self.rank, self.kv_rank = itf, parts, int(rank), int(kv_rank)
        itf.requires_grad_(False)
        if parts & {"dA", "dB"}:
            assert not itf.delta_rank, "base interface already carries deltas (merge them first)"
            itf.add_delta(self.rank, seed)
        dev = itf.g.device
        self.kv = nn.ParameterDict()
        if "kv" in parts:
            g = torch.Generator().manual_seed(seed + 17)
            for e in sorted(set(itf.layer_map.values())):
                att = next(itf.att[str(li)] for li, ee in itf.layer_map.items() if ee == e)
                hd = att.head_dim
                hkv = att.k_proj.out_features // hd
                for s in ("k", "v"):
                    self.kv[f"P{s}{e}"] = nn.Parameter((torch.randn(hkv, hd, self.kv_rank, generator=g) / hd ** 0.5).to(dev))
                    self.kv[f"Q{s}{e}"] = nn.Parameter(torch.zeros(hkv, self.kv_rank, hd, device=dev))
        self._named = self._collect()
        self._base = {k: p.detach().clone() for k, p in self._named.items()}

    def _collect(self) -> dict:
        itf, out = self.itf, {}
        for p in itf.D.parameters():
            p.requires_grad_(False)
        for li in itf.layers:
            k = str(li)
            if "dA" in self.parts:
                out[f"D.Ua{k}"], out[f"D.Va{k}"] = itf.D[f"Ua{k}"], itf.D[f"Va{k}"]
            if "dB" in self.parts and not itf.identity_B:
                out[f"D.Ub{k}"], out[f"D.Vb{k}"] = itf.D[f"Ub{k}"], itf.D[f"Vb{k}"]
            if "bias" in self.parts and not itf.identity_B:
                out[f"Bb.{k}"] = itf.Bb[k]
        if "gate" in self.parts:
            out["g"] = itf.g
        for k, p in self.kv.items():
            out[f"kv.{k}"] = p
        for p in out.values():
            p.requires_grad_(True)
        return out

    def param_groups(self, lr: float, lr_gate: float, lr_bias: float | None = None) -> list:
        lr_bias = lr if lr_bias is None else lr_bias
        g = {"main": [], "gate": [], "bias": []}
        for k, p in self._named.items():
            g["gate" if k == "g" else "bias" if k.startswith("Bb.") else "main"].append(p)
        return [{"params": v, "lr": {"main": lr, "gate": lr_gate, "bias": lr_bias}[n]} for n, v in g.items() if v]

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self._named.values()))

    @torch.no_grad()
    def reset(self):
        for k, p in self._named.items():
            p.copy_(self._base[k])

    @torch.no_grad()
    def snapshot(self) -> dict:
        return {k: p.detach().to("cpu", copy=True) for k, p in self._named.items()}

    @torch.no_grad()
    def load(self, snap: dict):
        assert set(snap) == set(self._named), (sorted(set(snap) ^ set(self._named)))[:5]
        for k, p in self._named.items():
            p.copy_(snap[k].to(p.device, p.dtype))

    def transform_op(self, op: dict) -> dict:
        """{enc_layer: (K, V)} -> the read-time operator view (identity unless the kv part is on)."""
        if "kv" not in self.parts:
            return op
        out = {}
        for e, kv in op.items():
            if not isinstance(e, int):          # extras (E2 "dv*", P3 "ids" / token table) pass through unchanged
                out[e] = kv
                continue
            K, V = kv
            Kf, Vf = K.float(), V.float()
            K2 = Kf + torch.einsum("hnr,hrd->hnd", torch.einsum("hnd,hdr->hnr", Kf, self.kv[f"Pk{e}"]), self.kv[f"Qk{e}"])
            V2 = Vf + torch.einsum("hnr,hrd->hnd", torch.einsum("hnd,hdr->hnr", Vf, self.kv[f"Pv{e}"]), self.kv[f"Qv{e}"])
            out[e] = (K2, V2)
        return out

    def drift(self) -> torch.Tensor:
        """Mean relative size of the per-document deltas (for logging / an optional L2 regularizer)."""
        itf, terms = self.itf, []
        if self.parts & {"dA", "dB"}:
            terms.append(itf.delta_reg())
        for e in sorted({int(k[2:]) for k in self.kv.keys() if k.startswith("Qk")}):
            for s in ("k", "v"):
                terms.append((self.kv[f"P{s}{e}"] @ self.kv[f"Q{s}{e}"]).pow(2).mean())
        if not terms:
            return torch.zeros((), device=itf.g.device)
        return torch.stack([t.to(itf.g.device) for t in terms]).mean()


__all__ = ["DocAdapter", "PARTS", "assert_no_test_questions", "drop_test_matches", "forbidden_strings", "leaks",
           "norm_q"]
