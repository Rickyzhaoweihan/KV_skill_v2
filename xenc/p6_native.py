"""P6: token injection in the READER'S NATIVE attention form (the "found the position but cannot read it out" fix).

P3 (xenc/p3_tok.py) wrote Σ p_j E[id_j] into the residual stream through a separate projection U; the oracle diagnostic
(outputs/P3_SUMMARY.md) showed that even with perfect addressing that form does not make the reader emit the gold
string. P6 puts the column's token into the reader's OWN attention value path instead:

    v_tok(l, j) = W_V^(l) · input_layernorm_l( h_src(id_j) )       the reader's own layer-l value projection
    o_tok       = per query head h:  Σ_j p_{h,j} v_tok(l, j)[kv(h)]   (GQA: query head h reads kv head h // G)
    attn_out_l += g_l · o_proj_l( o_tok ⊙ σ(gate_l(x)) )             the reader's own output gate and o_proj

with h_src(id) a CONTEXT-FREE function of the token id only (never a reader pass over the document):
    "emb"      E[id], the reader's input embedding row
    "ctxfree"  the reader's own layer-l input hidden state for the token processed ALONE (a one-token sequence; a
               per-vocabulary lookup table in principle, computed on demand and cached by id)
p is supplied by the caller (oracle one-hot on the next gold column for the go/no-go check; later the read's own
attention). g_l = 0 (or inactive) is exactly the parent. Layers can include the deep full-attention layers 23/27/31,
where the stock reader attends only over its own rows.
"""

from __future__ import annotations

import types

import torch


class NativeInjector:
    """Patches the reader's attention modules at `layers` (chained over any earlier patch, e.g. D13Reader). While
    self.cols is a list of (column ids) for the CURRENT forward's last row, the injection is added at that row:
    one-hot / uniform p over those columns for every query head (the oracle form). self.g[l] scales layer l."""

    def __init__(self, reader, layers, source: str = "emb"):
        from deltaskill.backbone_utils import decoder_layers
        assert source in ("emb", "ctxfree"), source
        self.reader, self.layers, self.source = reader, sorted(int(x) for x in layers), source
        L = decoder_layers(reader)
        self.L = L
        self.mods = {l: L[l].self_attn for l in self.layers}
        for l, m in self.mods.items():
            assert hasattr(m, "q_proj") and hasattr(m, "v_proj"), f"layer {l} is not a full-attention layer"
        self.g = {l: 0.0 for l in self.layers}
        self.ids: torch.Tensor | None = None       # operator column ids [N]
        self.cols: list | None = None              # columns to inject at the last row of the next forward
        self.rows: dict | None = None              # teacher-forced: {row index: [cols]} for a full-sequence forward
        self._cache: dict = {}                     # (source, id) -> {l: v [Hkv*hd]}
        self.ratios: dict | None = None            # set to {} to record ‖injection‖/‖attn out‖ per layer
        self._orig = {}
        for l, m in self.mods.items():
            self._orig[l] = m.forward
            m.forward = types.MethodType(_inj_forward(self, l), m)

    def detach(self):
        for l, f in self._orig.items():
            self.mods[l].forward = f
        self._orig.clear()

    @torch.no_grad()
    def values(self, tok_ids: list, compute: bool = True) -> dict:
        """{id: {l: v_tok(l, id) [Hkv*hd] fp32}} (cached). compute=False: every id must already be cached ("ctxfree"
        runs the reader, so it is precomputed OUTSIDE any read forward, with every read patch in mode "off")."""
        need = sorted({int(t) for t in tok_ids if (self.source, int(t)) not in self._cache})
        if need:
            assert compute or self.source == "emb", f"ctxfree values not precomputed for ids {need[:5]}"
            E = self.reader.get_input_embeddings().weight
            if self.source == "emb":
                h = {l: E[torch.tensor(need, device=E.device)] for l in self.layers}
            else:                                   # the token alone through the reader: layer-l input hidden states
                cap, hooks = {}, []
                for l in self.layers:
                    hooks.append(self.L[l].register_forward_pre_hook(
                        lambda mod, args, kw, l=l: cap.__setitem__(l, (args[0] if args else kw["hidden_states"])),
                        with_kwargs=True))
                saved = (self.cols, self.rows)
                self.cols, self.rows = None, None
                try:
                    self.reader(input_ids=torch.tensor(need, device=E.device).view(-1, 1), use_cache=False,
                                logits_to_keep=1)
                finally:
                    self.cols, self.rows = saved
                    for hk in hooks:
                        hk.remove()
                h = {l: cap[l][:, -1] for l in self.layers}
            for l in self.layers:
                lay = self.L[l]
                hl = h[l].to(lay.input_layernorm.weight.device)
                v = lay.self_attn.v_proj(lay.input_layernorm(hl)).float()           # [K, Hkv*hd]
                for k, t in enumerate(need):
                    self._cache.setdefault((self.source, t), {})[l] = v[k]
        return {int(t): self._cache[(self.source, int(t))] for t in tok_ids}

    def inject(self, mod, l: int, x: torch.Tensor, cols: list) -> torch.Tensor:
        """x [T_sel, d] normed attention inputs of the injected rows; cols: list (per row) of column lists.
        -> [T_sel, d] g_l · o_proj(o_tok ⊙ σ(gate))."""
        hd = mod.head_dim
        T = x.shape[0]
        q, gate = torch.chunk(mod.q_proj(x).view(T, -1, hd * 2), 2, dim=-1)
        Hq = q.shape[1]
        gate = gate.reshape(T, -1)
        vals = self.values([int(self.ids[c]) for cs in cols for c in cs], compute=False)
        rows = []
        for cs in cols:
            v = torch.stack([vals[int(self.ids[c])][l] for c in cs]).mean(0)       # uniform p over the columns
            vk = v.view(-1, hd)                                                    # [Hkv, hd]
            G = Hq // vk.shape[0]
            rows.append(vk.repeat_interleave(G, 0).reshape(-1))                    # query head h <- kv head h // G
        o = torch.stack(rows).to(x.dtype)
        return float(self.g[l]) * mod.o_proj(o * torch.sigmoid(gate))


def _inj_forward(P: NativeInjector, l: int):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        out, w = P._orig[l](hidden_states, position_embeddings, attention_mask,
                            past_key_values=past_key_values, **kwargs)
        if P.ids is None or float(P.g[l]) == 0.0:
            return out, w
        T = hidden_states.shape[1]
        sel = {}
        if P.rows:
            sel = {r: cs for r, cs in P.rows.items() if 0 <= r < T and cs}
        elif P.cols:
            sel = {T - 1: P.cols}
        if not sel:
            return out, w
        idx = sorted(sel)
        x = hidden_states[0, idx]
        add = P.inject(self, l, x, [sel[r] for r in idx]).to(out.dtype)
        if P.ratios is not None:                    # calibration: ‖injection‖ / ‖attention output‖ at the injected rows
            with torch.no_grad():
                P.ratios.setdefault(l, []).append(float(add.float().norm() / out[0, idx].float().norm().clamp_min(1e-9)))
        out = out.clone()
        out[0, idx] = out[0, idx] + add.to(out.device)
        return out, w

    return forward


__all__ = ["NativeInjector"]
