# Code plan: small ENCODER / large READER in the delta-skill + cross_model pipeline

Repo root `R = /nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill`. All paths below are relative to `R`.
New code goes in a new package (suggested `KV_skill_v2/xenc/`, currently an empty dir) that imports
`deltaskill` and `xskill` the way `xskill` imports `deltaskill` (subclass, never patch).

## 1. Where the code assumes encoder == reader

Every assumption is "the skill was derived on the model that reads it":

| # | File:line | Assumption |
|---|---|---|
| A1 | `delta-skill/deltaskill/skill.py:24-25` | `K/V: dict[int, Tensor]  # layer_idx -> [r, d_s]` — keyed by the **absolute layer index of the deriving model**. |
| A2 | `delta-skill/deltaskill/derive.py:27-32` | `injection_layers(model, depths)` = `floor(rho*L)` of whatever model is passed; `derive.py:59-66` hooks `blocks[li]` by that absolute index; the dict is keyed by it (`:95-101`). |
| A3 | `derive.py:92-93` | `d_s = hidden_size` of the deriving model; `derive.py:100-101` K and V are clones (tied). |
| A4 | `delta-skill/deltaskill/interface.py:21` | `ModelInterface(d_m, d_s, inject_layers)` *does* support `d_s != d_m` (A: d_m→d_s, B: d_s→d_m, `:30-36`), but **every call site passes `(d_m, d_m)`**: `scripts/register_benchmarks.py:140`, `scripts/train_unified.py:98`, `scripts/longhealth_kv.py:126`, `cross_model/scripts/eval_shared.py:98`. `_pos` (`interface.py:26`) and `gate_b/log_gain` (`:38-39`) are sized by the reader's `inject_layers`. |
| A5 | `delta-skill/deltaskill/injector.py:69-70` | hooks attach at `blocks[li] for li in itf.inject_layers` (reader indices); `:111-112` `K = self.skill.K[li]` — **looks up the skill by the reader's layer index** → `KeyError` for any encoder with a different `L`, silently *wrong* layers for an encoder with the same `L`. |
| A6 | `injector.py:113` | `einsum("btd,rd->btr", q, K)` requires `itf.d_s == skill.d_s` (no check, shape error at runtime). |
| A7 | `injector.py:60-61`, `skill.py:43-49` | `load()` casts the skill to `model.device` + fp32; no check of `d_s`. |
| A8 | `injector.py:43-46, 119-122` | `calib = {layer: {"scale":[d_s], "gain"}}` keyed by **layer index**, scale sized to the skill's `d_s`. |
| A9 | `injector.py:114-117` | `read_softmax` / `a/sqrt(N)` act per layer over that layer's `r` slots — fine, but the slot count must agree across the keyed layers. |
| A10 | `register_benchmarks.py:188, 211-221` | derives with `model` (the reader); the `--random_skill` control copies `proto.K[li].shape` (reader-keyed). |
| A11 | `cross_model/xskill/shared_interface.py:79` | `self.d_s = core.d_c`; `:110-118 project(slots)` pushes the skill through the **reader's** lens `E_m: Linear(d_m, d_c)` (`:55`) — i.e. the skill must be in the reader's `d_m`. |
| A12 | `cross_model/xskill/injector.py:32-33` | `self.itf.project(self.skill.K[li])` — same reader-index lookup as A5. |
| A13 | `cross_model/xskill/multi_backbone.py:62` | `self.layers = injection_layers(self.model, DEPTHS)`; `:65-66` the lens lives on the device of the reader's first injection layer. |
| A14 | `cross_model/scripts/train_shared.py:134`, `register_shared.py:102-105`, `eval_shared.py:186-188` | `derive_skill(b.model, b.tok, ...)` — the reader encodes its own skills (pristine norms). |
| A15 | `shared_interface.py:72-73` | `assert len(inject_layers) == core.n_slots` — slot count fixed at 4; a dynamic candidate set of 8 breaks it. |

The stated invariant in `delta-skill/CLAUDE.md` ("Normalized depths, not absolute layers") holds at the
*config* level (`DEPTHS`) but not in the artifact: the saved `KVSkill` stores absolute indices.

## 2. Minimal design

### (a) `derive_skill` keyed by normalized depth ρ

Keep `KVSkill` but let keys be ρ floats and add the reader-side remap (new module `xenc/depth_skill.py`;
`skill.py` itself is untouched — `dict[int, …]` is only a type hint).

```python
# xenc/depth_skill.py
from deltaskill.derive import derive_skill, injection_layers
from deltaskill.skill import KVSkill

def derive_by_depth(enc_model, enc_tok, text, depths, slots, long_mode="pool") -> KVSkill:
    layers = injection_layers(enc_model, depths)                 # encoder's floor(rho*L)
    assert len(layers) == len(depths), f"depth collision on L={enc_L}: {depths}->{layers}"
    s = derive_skill(enc_model, enc_tok, text, layers, slots, long_mode)   # keyed by enc layer
    rho_of = dict(zip(layers, sorted(depths)))
    K = {rho_of[li]: t.half() for li, t in s.K.items()}          # fp16, keyed by rho
    return KVSkill(K=K, V=K, d_s=s.d_s, text=text, model_name=s.model_name,
                   meta={**s.meta, "keyed_by": "rho", "depths": sorted(depths), "tied": True,
                         "encoder": s.model_name, "slots": slots, "long_mode": long_mode})

def for_reader(skill: KVSkill, reader_model, depths=None) -> KVSkill:
    """rho-keyed -> reader-layer-keyed view (no copy of the tensors)."""
    depths = depths or skill.meta["depths"]
    rl = injection_layers(reader_model, depths)
    assert len(rl) == len(depths)
    m = dict(zip(sorted(depths), rl))                            # rho -> reader layer
    return KVSkill(K={m[r]: skill.K[r] for r in depths}, V={m[r]: skill.V[r] for r in depths},
                   d_s=skill.d_s, text=skill.text, model_name=skill.model_name, meta=skill.meta)
```
V shares storage with K (`tied`), so `save()` (`skill.py:51-55`) should drop V when `meta["tied"]` — add a
small `save_tied/load_tied` pair in the same module rather than editing `skill.py`.

### (b) Reader-side ρ→layer mapping
`for_reader` above. Qwen3.5: 2B L=24 → {6,12,18,23}; 4B/9B L=32 → {8,16,24,31}; 27B L=64 → {16,32,48,63};
gpt-oss-20b L=24. Note Qwen3.5 interleaves DeltaNet and full-attention blocks (every 4th is attention),
so a ρ may land on different block *types* across sizes; the hook is on the residual stream, so this is
legal, but keep ρ fixed across sizes rather than snapping to attention layers (snapping would silently
change the experiment).

### (c) Interface with `d_m = d_reader, d_s = d_encoder`
Per-model path: `ModelInterface(d_reader, d_encoder, reader_layers, rank=…)` already works (`interface.py:30-36`);
only call sites change. Add a guard in a subclass of the injector:

```python
class XEncInjector(ExternalSkillInjector):        # xenc/injector.py
    def load(self, skill):
        if skill.d_s != self.itf.d_s:
            raise ValueError(f"skill d_s={skill.d_s} (encoder) != interface d_s={self.itf.d_s}")
        if skill.meta.get("keyed_by") == "rho":
            skill = for_reader(skill, self.model)
        missing = set(self.itf.inject_layers) - set(skill.K)
        if missing: raise KeyError(f"skill has no operator for reader layers {sorted(missing)}")
        super().load(skill)
```
Shared path (`cross_model`): give `SharedInterfaceView` a second lens for the skill side,
`EncoderLens(d_enc, d_c)` (orthogonal init like `ModelLens.E`), and route `project()` through it:

```python
class XSharedView(SharedInterfaceView):
    def __init__(self, core, lens, enc_lens, inject_layers, skill_norm="row"):
        super().__init__(core, lens, inject_layers, skill_norm); self.enc = enc_lens; self.d_s = enc_lens.d_in
    def project(self, slots):
        z = F.linear(slots.float(), self._on(self.enc.E.weight, slots))
        return _l2(z) if self.skill_norm == "row" else z
```
This makes the encoder lens a *per-encoder* object shared by all readers — exactly the H2 artifact
("encode once with 4B, read with 9B and 27B").

### (d) Storage + cache
Format: one `.pt` per (encoder, doc chunk): `{"K": {rho: fp16 [N, d_enc]}, "d_s", "meta"}` (tied ⇒ no V).
Size at 4 depths × 512 slots × d_enc=2048 (2B) is 8 MB; 2560 (4B) 10 MB. Cache key =
`sha1(text)[:12]_{enc_short}_s{slots}_{long_mode}_d{depths}`; directory on `/scratch` (not NFS) and
`HF_HUB_OFFLINE`-safe. Chunking of a long doc = fixed ≤512-token windows (tokenizer of the encoder),
each window prefilled **with the preceding windows as causal context** (one prefill of the whole doc,
`long_mode` applied per window), so slots keep document-level context as `derive.py:52` promises.

```python
def encode_doc(enc, tok, doc_id, text, cache_dir, depths, win=512) -> list[Path]:
    paths = [cache_dir / f"{doc_id}_{i:03d}.pt" ...]; if all exist: return paths
    ids = tok(text).input_ids; hs = prefill_hidden(enc, ids, layers)       # one forward, hooks as derive.py:59-66
    for i, (a, b) in enumerate(windows(len(ids), win)):
        K = {rho: l2(hs[rho][a:b]).half() ...}; save_tied(K, meta, paths[i])
```

### (e) Library of per-document operators with a per-query selector
Goes in `xenc/injector.py` as an override of `_branch` (same seam `LensSkillInjector` uses,
`cross_model/xskill/injector.py:29-45`). A `Library` is `C` chunks of one document (or several docs):

```python
class Library:                                     # built once per doc from cache files
    def __init__(self, chunks: list[KVSkill], device, dtype=torch.float32):
        self.K = {rho: torch.stack([c.K[rho] for c in chunks]).to(device, dtype)}   # [C, N, d_s]
        self.S = {rho: _l2(self.K[rho].mean(1))}                                    # summary key [C, d_s]
        self.d_s = chunks[0].d_s; self.meta = chunks[0].meta

class LibraryInjector(XEncInjector):
    def __init__(self, *a, route_temp=8.0, top_k=0, **kw): ...
    def _branch(self, h, li):
        rho = self.rho_of[li]
        hn = rms_norm(h.float()); q = self.itf.query(hn)                       # [B,T,d_s]
        K = self.lib.K[rho]; S = self.lib.S[rho]                                 # [C,N,d], [C,d]
        logit = torch.einsum("btd,cd->btc", q, S) * self.route_temp
        if self.top_k:                                                            # sparse routing
            v, i = logit.topk(self.top_k, -1); logit = torch.full_like(logit, -1e9).scatter(-1, i, v)
        w = torch.softmax(logit, -1)                                              # [B,T,C], sums to 1
        self.route_terms.append(w)                                                # for an aux loss / entropy
        a = torch.einsum("btd,cnd->btcn", q, K)
        a = torch.softmax(a * self.read_temp, -1) if self.read_softmax else a / K.shape[1] ** 0.5
        r = torch.einsum("btc,btcn,cnd->btd", w, a, K)                            # V == K (tied)
        g = self.itf.gate(hn, li)
        return self.gain_scale * self.itf.gain(li) * g * self.itf.write(r)
```
Memory: `a` is `[B,T,C,N]`; at T=600, C=16, N=512 that is 5 M floats — trivial. With `top_k` the
within-chunk read can be gathered to `[B,T,k,N]` for very large libraries. The existing per-skill `calib`
(`injector.py:119-122`) is dropped on this path (it is a per-skill artifact and the library is per-doc).

### (f) Dynamic layer selector (sparse top-k over ~8 candidate reader layers)
Candidate depths `CAND = [0.2,0.3,…,0.9]` (8); the encoder is derived at the same 8 ρ (one prefill, 2× storage).
The reader's hooks attach at all 8 layers; selection must be known at the *first* candidate layer because
later layers run after it, so the selector reads the residual at `l_0` and stores per-token weights for the
rest of the forward:

```python
class LayerSelector(nn.Module):                     # part of the trainable interface
    def __init__(self, d_m, n_cand, k=2, alpha=0.05):
        self.u = nn.Linear(d_m, n_cand, bias=True); self.k = k
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha)))   # target ||delta||/||h|| budget
    def forward(self, hn0):                          # [B,T,d_m] at the first candidate layer
        z = self.u(hn0); v, i = z.topk(self.k, -1)
        w = torch.zeros_like(z).scatter(-1, i, torch.softmax(v, -1))  # sparse, sums to 1 over layers
        return w                                      # [B,T,n_cand]; grads flow to the selected k

# in LibraryInjector.hook(li): if li == self.cand[0]: self._w = self.sel(rms_norm(h.float()))
# in _branch(h, li), after u = self.itf.write(r):
    rel = u.norm(-1, keepdim=True) / (h.float().norm(-1, keepdim=True) + 1e-6)
    u = u / (rel + 1e-6) * self.sel.log_alpha.exp()          # MATCHED update norm: ||delta||/||h|| == alpha
    return self._w[..., self.pos[li]].unsqueeze(-1) * g * u    # times the layer weight (0 for unselected)
```
Matched norm means a layer choice changes *where*, not *how much*, so the top-k decision is not confounded
with gain; the existing `log_gain` (`interface.py:39`) becomes redundant on this path and `mag_terms`
(`injector.py:81-82`) reduce to `alpha` by construction (keep the penalty on `alpha`). The generation
path (`model.generate`) sees T=1 steps: cache `_w` per step — it is recomputed at `l_0` every step, so no
state leaks between steps. For A15, build `SharedCore(d_c, n_slots=len(CAND))` and derive train/replay
skills at `CAND` too (they are 128 slots, cheap).

## 3. Registration loop changes

**Where the prompt is built.** `register_benchmarks.py:223-231` (`task_ids`) renders
`[system, user=task["prompt"]]`; the teacher (`:259`) is `hard_system(t, b["text"])` which inserts the
*skill text* into the **system** message (`:63-66`); the student (`:267`) is the same prompt without it.
For extractive tasks the *document* is not the skill: `deltaskill/tasks.py:135-137` bakes `## Context`
into the **user** prompt at load time (`:167`), so the student still sees the document. One skill per
task (`:188`). `cross_model/scripts/register_shared.py:110-112,139,146` is the same pattern.

**Yes, a document-as-skill mode exists:** `delta-skill/scripts/longhealth_kv.py`. `user_msg(item, record)`
(`:74-77`) puts `## Document\n{record}` in the user turn only when `record` is given; the teacher is
`ids_for(it, record=records[pid])` (`:199`), the student `ids_for(it, record=None)` with
`inj.skill = skills[pid]` (`:212-213`), skills derived per document once (`:149-150`), and it already uses
`logits_to_keep` + `use_cache=False` on the teacher (`:209-210`). It hardcodes `HUB` (`:45`),
`ModelInterface(d_m, d_m)` (`:126`), `enable_thinking=False` (`:158`), and derives on the reader. The new
`register_doc.py` is a port of this script's loop onto `Backbone` + `LibraryInjector`:

```python
# xenc/doc_tasks.py: split the context out of the prompt at load time
def load_doc_tasks(split_dir, split):   # items carry doc_id, context, question, answers
    return [{"system": SYS, "doc_id": it["doc_id"], "q_prompt": f"## Question\n{it['question']}",
             "ctx_prompt": f"## Context\n{it['context']}\n\n## Question\n{it['question']}", ...}]
# register_doc.py (teacher = READER with doc in context; student = READER, doc removed, library loaded)
tids, _, m = reader.with_target([sys, {"role":"user","content": t["ctx_prompt"]}], target)
with torch.no_grad(): reader.inj.unload()
    pt = softmax(reader.model(tids, logits_to_keep=m+1, use_cache=False).logits[0, -(m+1):-1].float())
sids, slab, _ = reader.with_target([sys, {"role":"user","content": t["q_prompt"]}], target)
reader.inj.load_library(libs[t["doc_id"]])      # from the encoder cache, never the reader
```

**Avoiding direct-answer collapse.** Today the KL (`register_benchmarks.py:265-271`) and CE (`:272`) are
computed only on the `m ≈ 6` tokens of `<answer>X</answer>` (`gold_target_text`, `tasks.py:309-319`), and the
target starts immediately after the assistant header — the student is trained to emit the answer first.
Variant: **trace-KL** — target = the teacher's *own* greedy trace with the document in context
(generated once per train item with `StopOnAnswerClose`, budget 512, cached as
`traces/<reader>/<task>/<item_id>.json`), then one teacher-forced pass on the trace gives teacher logits
for all trace positions; the student is teacher-forced on the same trace:

```python
loss = KL(p_T || p_S)[all trace tokens].mean()  +  0.5 * CE(gold)[answer span only]  +  mag/alpha penalty
```
The answer-span mask comes from the `<answer>` offsets in the trace; items whose teacher trace is wrong are
either dropped or keep only the KL term (flag `--wrong_trace drop|kl_only`). Position of the term: replace
`:263-272` in the new script (`--target trace|gold`); nothing changes in `deltaskill/`. Trace generation is
an offline step like encoding (27B greedy at 512 tokens ≈ 10–20 s/item on H100; 1k items ≈ 4 h, batched 8× faster).
A cheaper first pass: `--target gold` but prepend a fixed "Reasoning:" prefix target — do not expect it to
be enough given the settled finding in `cross_model/CLAUDE.md:164-168`.

## 4. Memory / throughput

- **Sharding exists.** `multi_backbone.py:36-50`: `devices="cuda:0+cuda:1"` → `device_map="auto"`, and
  `SharedInterfaceView._on` (`shared_interface.py:82-85`) moves core/lens weights to the activation's device;
  lens on the first injection layer's device (`:65-66`); `register_shared.py:151-153` already `.to()`s the KL
  and `mag_terms` across devices. **Bug for L40S:** `max_memory` is hardcoded `"76GiB"` per GPU (`:42`) — on
  44 GB L40S accelerate will overfill GPU0. Make it a parameter (`--max_mem 40GiB`).
- **27B on 2×L40S:** 54 GB weights leaves ~30 GB. Teacher forward at 4k tokens: `register_shared.py:145`
  takes **full logits** `[1,T,152k]` then `.float()` → ~2.5 GB bf16 + 5 GB fp32 + softmax copy; use
  `logits_to_keep=m+1, use_cache=False` as `longhealth_kv.py:209-210` does (≈0). Activations under
  `no_grad` are transient (one layer at a time, SDPA) — a few GB at T=4k. Student forward with grads is on
  the *short* prompt (doc removed, ~300 + trace ≤512 tokens): with `grad_ckpt` (`multi_backbone.py:85-87`,
  numerically identical) it is well under 10 GB on 64 layers. So 27B registration fits 2×L40S and one H100;
  gpt-oss-20b fits one H100 (its full-vocab prefill OOMs at batch 8, `cross_model/CLAUDE.md:145` — keep batch 1–2).
- **Student CE on a 512-token trace** must also avoid full logits: compute CE manually from
  `logits_to_keep=m+1` instead of passing `labels=` (`:149`), otherwise it is another 5 GB fp32 per step.
- **Encoder is never in the registration process**: libraries come from the disk cache, which removes the
  start-up derive on the reader (`register_shared.py:99-106`). Per-doc library on GPU: 8 chunks × 4 ρ × 512 ×
  2560 × 4 B (fp32 read) ≈ 170 MB; with 8 candidate depths 340 MB. Keep libraries fp16 on CPU, move the
  active one per step (same pattern as `register_benchmarks.py:175,251`); a 1k-doc corpus is ~10–40 GB on
  disk, so use an LRU of ~50 docs in RAM, not a full preload.
- Throughput: the teacher pass dominates (4k tokens through 27B ≈ 0.5–1 s on H100, ~2× on L40S); 300 steps
  × (1 + replay 2) ≈ 10–20 min per task, as now.

## 5. Tests

Existing: `cross_model/tests/test_shared_interface.py` (4: identity-lens equivalence to `ExternalSkillInjector`,
no-op at init, gradient routing core-vs-lens, low-rank scale through a linear lens) — a plain script, CPU,
`TinyLM` with `model.layers`; `delta-skill/tests/test_lowrank.py` (C1 scale identity, C2 full-rank
reconstruction, clipped rank, …; synthetic `KVSkill`). Reuse `TinyLM`/`make_skill` from the former.

New `KV_skill_v2/tests/test_xenc.py` (same style, no GPU):
1. `test_noop_at_init_cross_dim` — reader `TinyLM(d=40)`, ρ-keyed skill with `d_s=24`,
   `ModelInterface(40, 24, …)`; forward equals the bare model exactly (zero-init B).
2. `test_dim_mismatch_raises` — `XEncInjector.load` raises `ValueError` for `d_s=24` vs interface `d_s=32`,
   and `KeyError` when a ρ needed by the reader is absent.
3. `test_rho_mapping` — fake configs with `L=24/32/64`: `for_reader` yields {6,12,18,23}/{8,16,24,31}/{16,32,48,63};
   `L=3` with 4 depths raises the collision assert; round-trip derive(enc)→for_reader(enc) is the identity.
4. `test_library_routing_sums_to_one` — C=5 chunks: `w.sum(-1)==1` (±1e-6) with `top_k∈{0,2}`; `top_k=2`
   has exactly 2 nonzeros per token; a 1-chunk library reproduces `ExternalSkillInjector._branch` to 1e-5.
5. `test_selector_topk_norm_matching` — 8 candidates, k=2: exactly k selected per token; after matching,
   `‖Δ_l‖/‖h‖ == w_l·α` (±1e-5) for two skills whose raw `write(r)` norms differ by 100×.
6. `test_equivalence_encoder_eq_reader` — encoder == reader `TinyLM`, `d_s=d_m`, 4 depths, selector off,
   1-chunk library: hook outputs equal the old `ExternalSkillInjector` path to 1e-5 (extends
   `test_equivalence_identity_lens`).

## 6. Effort (engineer-days) and risks

| item | days |
|---|---|
| (a)(b)(d) ρ-keyed derive, `for_reader`, tied fp16 cache + `encode_docs.py` offline job | 1.5 |
| (c) `XEncInjector` guards, `(d_reader, d_enc)` call sites, `EncoderLens` in the shared view | 1 |
| (e) `Library` + `LibraryInjector` (+ optional aux routing loss) | 1.5 |
| (f) `LayerSelector`, matched-norm write, 8-candidate derive, `n_slots` plumbing | 2 |
| §3 `doc_tasks.py`, `register_doc.py` (teacher-with-doc / student-with-library), trace cache + trace-KL | 3 |
| §4 `max_memory` flag, `logits_to_keep` on teacher and student, manual CE | 0.5 |
| §5 tests | 1 |
| smoke: 2B→4B on one L40S, then 4B→27B on H100 (resume/Slurm plumbing reused) | 1.5 |
| **total** | **≈ 12** |

Riskiest unknowns: (1) the paper's own L/L negative on document QA (`cross_model/CLAUDE.md`, brief) — the
operator carried ~no content even with encoder == reader; S/L adds a d_enc→d_c projection on top of that,
so the library/softmax read is the whole bet and should be de-risked first with **L/L + library** before any
cross-size run. (2) Whether a 2B/4B residual stream at ρ is *readable* by a 27B interface at ρ — normalized
depth is a convention, not a known alignment (and Qwen3.5's DeltaNet/attention interleave differs by size).
(3) Routing gets only answer-level gradient; expect to need an auxiliary "which chunk holds the gold span"
loss (computable from the span offset). (4) Trace-KL may still collapse if the gate learns to shortcut;
monitor generated length per condition (the `mean_gen_tokens` field in `eval_shared.py:250`). (5) 27B on
2×L40S is tight only through the `max_memory` bug and the full-logit paths — fix before the first run.
(6) transformers 5.13 `logits_to_keep` + `labels` semantics are untested here — compute CE manually.
