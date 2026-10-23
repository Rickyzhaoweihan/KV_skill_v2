# xenc — package spec (binding contract for all build agents)

Repo: `/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2` (git repo, `main`). Sibling
`/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/delta-skill` provides `deltaskill` (NEVER modify it).
Env: `conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35` (torch 2.12.1, transformers
5.13.0, **no pytest, no `datasets`**). For scripts that need `datasets`:
`export PYTHONPATH=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/SoftSkill/.deps_qwen35:$PYTHONPATH`.
HF: `HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home HF_HUB_CACHE=$HF_HOME/hub HF_HUB_OFFLINE=1`
(compute nodes are offline; the login node is online). `~/.bashrc` points HF at a dead scratch cache —
always export these in scripts/launchers. fla kernels: add
`/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/fla_kernels` to `sys.path` (done in `xenc/__init__.py`).

Design background: `docs/PLAYGROUND_PROPOSAL.md` §7 and `docs/audits/CODE_PLAN_encoder_reader.md`.
Read both before coding. The code plan has file:line citations into deltaskill for every assumption.

## Layout (each agent owns ONLY its files; create nothing else outside your list)

```
KV_skill_v2/
  xenc/__init__.py          # (written by orchestrator) sys.path bootstrap for delta-skill + fla_kernels
  xenc/depth_skill.py       # [A] derive_by_depth, for_reader, save/load fp16 tied, DepthSkill cache keys
  xenc/library.py           # [A] Library (stack of chunk operators + summary keys)
  xenc/injector.py          # [A] XEncInjector(ExternalSkillInjector), LibraryInjector
  xenc/selector.py          # [B] LayerSelector (sparse top-k over candidate layers, matched norm)
  xenc/interface.py         # [B] make_interface(d_reader, d_encoder, reader_layers, rank=0) + EncoderLens
  xenc/doc_tasks.py         # [C] DocItem schema, load_doc_split, chunk_document
  xenc/backbone.py          # [D] load_reader / load_encoder (Qwen3.5 multimodal-wrapper aware), chat helpers
  xenc/losses.py            # [D] kl_teacher_student, answer_ce (manual, logits_to_keep), trace targets
  scripts/encode_docs.py    # [A] CLI: encode a split's documents with an encoder into the operator cache
  scripts/data/prepare_quality.py, prepare_longhealth.py, prepare_lastingbench.py, prepare_musique.py  # [C]
  scripts/register_doc.py   # [D] registration loop (teacher doc-in-context vs student question+library)
  scripts/eval_doc.py       # [D] conditions: closed_book, text, kv, wrong_doc, null, oracle_chunk
  scripts/ladder.py         # [D] text-in-context size ladder (closed_book + text) over readers
  slurm/*.sbatch            # [D] launchers (account drjieliu; partitions drjieliu-h100 / drjieliu-l40s)
  tests/_tiny.py            # (orchestrator) TinyLM + make_skill fixtures — import, do not copy
  tests/test_depth_skill.py tests/test_library.py        # [A]
  tests/test_selector.py tests/test_interface.py         # [B]
  tests/test_data.py                                     # [C]
  tests/test_losses.py tests/test_register_plumbing.py   # [D]
  data/<name>_split/{train,val,test}/items.json          # [C] output; data/docs/<name>/<doc_id>.md
  data/raw_downloads/                                    # already present (gitignored): raw dataset files
```

Tests: plain python files runnable as `python tests/test_x.py` (use `assert`, print `OK` at end) AND
importable by pytest (functions named `test_*`). CPU only, no checkpoints, < 60 s each.

## Shared conventions

- Depths: `DEPTHS = [0.25, 0.5, 0.75, 1.0]`; candidate set for the selector `CAND = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]`.
  Reader layer for depth ρ = `min(L-1, int(ρ*L))` (`deltaskill.derive.injection_layers`).
- A skill keyed by depth: `KVSkill` whose `K`/`V` dict keys are **depth floats** (ρ) and whose
  `meta["keyed_by"] == "rho"`, `meta["depths"] == sorted list`, `meta["tied"] == True`,
  `meta["encoder"] == model name`. (deltaskill's `dict[int, Tensor]` annotation is only a hint.)
- Slots per chunk ≤ 512 tokens; chunks come from ONE whole-document prefill (states keep causal context),
  cut on token boundaries into windows of `chunk_tokens` (default 512). Each slot L2-normalized (as
  `derive_skill` does). K = V (tied) → store K only, fp16, CPU.
- Cache key for an encoded chunk: `f"{sha1(doc_text)[:12]}_{encoder_short}_c{chunk_tokens}_d{'-'.join(depths)}"`
  under `--cache_dir` (default `$XENC_CACHE` or `/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache`).
  File per document: `{key}.pt` = `{"chunks": [ {"K": {rho: fp16 [N,d_s]}, "n_tokens": int, "start": int} ... ],
  "d_s": int, "doc_id": str, "encoder": str, "chunk_tokens": int, "depths": [...]}`.
- Interface: `deltaskill.interface.ModelInterface(d_m=d_reader, d_s=d_encoder, inject_layers=reader_layers, rank=0)`.
  `B` is zero-init so the branch is a no-op at init — every injector must preserve this.
- All chat prompting goes through `xenc.backbone.build_chat(tokenizer, hub, system, user)` which passes
  `enable_thinking=False` for Qwen/GLM and handles gpt-oss (`deltaskill.backbone_utils.chat_template_kwargs`).
- Teacher forward: `use_cache=False`, `logits_to_keep=m+1` (never full-vocab logits at long context).
  Student CE computed manually from the kept logits, never via `labels=`.

## [A] depth_skill / library / injector API

```python
# xenc/depth_skill.py
def derive_by_depth(model, tok, text: str, depths=DEPTHS, chunk_tokens=512, max_doc_tokens=0) -> dict
    # returns the cache dict described above (K per chunk per depth, fp16 CPU), using
    # deltaskill.derive.injection_layers(model, depths) + forward hooks on decoder_layers(model)
    # (reuse deltaskill.derive.derive_skill for the hook mechanics or re-implement; NO pooling).
def for_reader(skill_rho: KVSkill, reader_model, depths=None) -> KVSkill
    # ρ-keyed -> reader-layer-keyed VIEW (no copy); asserts len(set(layers)) == len(depths) (collision guard)
def chunk_to_kvskill(chunk: dict, d_s: int, doc_id: str, encoder: str) -> KVSkill   # ρ-keyed, K=V same tensor
def save_doc_cache(d: dict, path: str) / load_doc_cache(path: str) -> dict
def cache_key(doc_text: str, encoder_short: str, chunk_tokens: int, depths) -> str

# xenc/library.py
class Library:
    def __init__(self, chunks: list[KVSkill-like dicts or KVSkill], reader_layers: list[int], depths, device, dtype=torch.float32)
    # K[li]: [C, N_max, d_s] zero-padded, mask[li]: [C, N_max] bool, S[li]: [C, d_s] L2(mean of valid rows)
    # .n_chunks, .to(device), .subset(idx) (for oracle_chunk), .d_s
    @staticmethod
    def from_doc_cache(cache: dict, reader_model, depths) -> "Library"

# xenc/injector.py
class XEncInjector(ExternalSkillInjector):
    # load(skill): if skill.meta.get("keyed_by")=="rho": skill = for_reader(skill, self.model)
    #              if skill.d_s != self.itf.d_s: raise ValueError; missing reader layer -> KeyError
class LibraryInjector(XEncInjector):
    def __init__(self, model, interface, read_softmax=True, read_temp=12.0, route_temp=10.0, top_k=0, selector=None)
    def load_library(self, lib: Library) / unload()
    # _branch(h, li): hn=rms_norm(h.float()); q=itf.query(hn)  [B,T,d_s]
    #   route logits = q @ S[li].T * route_temp -> (top_k mask) -> softmax over chunks  w [B,T,C]
    #   a = einsum('btd,cnd->btcn', q, K[li]); masked -> softmax(a*read_temp) over n (or /sqrt(n) if not read_softmax)
    #   r = einsum('btc,btcn,cnd->btd', w, a, K[li])   (V == K)
    #   delta = gain_scale * itf.gain(li) * itf.gate(hn, li) * itf.write(r)
    #   if self.selector is not None: delta = self.selector.apply(delta, h, li)   # [B] contract below
    #   self.last_route = w.detach() (for logging/aux loss)
    # exact equivalence: a 1-chunk Library with read_softmax=False must equal ExternalSkillInjector._branch to 1e-5
```

## [B] selector / interface API

```python
# xenc/selector.py
class LayerSelector(nn.Module):
    def __init__(self, d_m: int, cand_layers: list[int], k: int = 2, alpha_init: float = 0.02)
    # u: Linear(d_m, n_cand, bias=True); log_alpha: scalar param (target ||delta||/||h||)
    def compute(self, hn0: Tensor[B,T,d_m]) -> None      # called ONCE per forward at the FIRST cand layer; stores self._w [B,T,n_cand]
    def apply(self, delta: Tensor[B,T,d], h: Tensor[B,T,d], li: int) -> Tensor
    #   if self._w is None: compute from rms_norm(h) (first layer); 
    #   rel = ||delta|| / (||h||+eps) per token; delta = delta / (rel+eps) * exp(log_alpha)   # matched norm
    #   return self._w[..., pos[li]].unsqueeze(-1) * delta
    def reset(self) -> None                               # clear _w (call per forward in the injector)
    def weights(self) -> Tensor | None
    # sparsity: top-k over logits, softmax over the kept entries, zeros elsewhere; sums to 1 per token.
    # Must be differentiable w.r.t. u on the kept entries.

# xenc/interface.py
def make_interface(d_reader: int, d_encoder: int, reader_layers: list[int], rank: int = 0,
                   gate_bias_init: float = -2.0, gain_init: float = 0.0) -> ModelInterface
class EncoderLens(nn.Module):   # optional per-encoder projection d_enc -> d_core for shared-core experiments
    def __init__(self, d_enc: int, d_core: int)   # orthogonal init; forward(K[...,d_enc]) -> [...,d_core]
def interface_param_groups(itf, lr: float, selector=None, selector_lr=None) -> list[dict]   # for AdamW
def save_interface(itf, path, selector=None, extra=None) / load_interface(path, d_reader, d_encoder, reader_layers) -> (itf, selector, extra)
```

## [C] data API and schema

Output `data/<name>_split/{train,val,test}/items.json` (JSON list) with items:
```json
{"key": "quality_val_0001", "doc_id": "quality_<sha12>", "question": "...",
 "choices": [{"label":"A","text":"..."}, ...] or null, "answers": ["gold string", ...],
 "correct_label": "B" or null, "task": "quality", "hop": null or int, "meta": {...}}
```
Documents are NOT inline: `data/docs/<name>/<doc_id>.md` (plain text). A `data/<name>_split/docs.json` maps
`doc_id -> {"path": relative, "n_tokens": int (Qwen3.5 tokenizer), "title": str}`.
Splits: QuALITY = HF `emozilla/quality` (train→train, validation→ split 50/50 by ARTICLE into val/test, seed 42;
all questions of an article stay together). LongHealth = `data/raw_downloads/longhealth/benchmark_v5.json`
(20 patients; 5-fold by patient written as `folds.json`; items.json train=all, val=all, test=all with `meta.fold`).
LastingBench = `data/raw_downloads/lastingbench/data/{hotpotqa,2wikimqa,musique,multifieldqa_en}{,_antifact}.jsonl`:
each original item is one document (its full context) with 1 question; counterfactual twin shares `doc_id`
prefix with suffix `_cf`; items carry `meta.counterfactual: bool`, `meta.rewritten: bool`; split 40/10/50 by
item, seed 42, twins kept in the same split. MuSiQue = `data/raw_downloads/musique/` dev: the DOCUMENT is the
concatenation of the item's paragraphs with `[P{i}] title\n text` markers; `meta.supporting_idx` lists gold
paragraph indices; `hop` from the id prefix; split 40/10/50 seed 42.
```python
# xenc/doc_tasks.py
@dataclass class DocItem: key, doc_id, question, choices, answers, correct_label, task, hop, meta
def load_doc_split(split_dir: str, split: str) -> list[DocItem]
def load_docs(split_dir: str) -> dict[str, str]            # doc_id -> text
def q_prompt(item) -> str                                   # question (+ choices) WITHOUT document
def ctx_prompt(item, doc_text: str, max_chars: int | None = None) -> str   # "## Document\n...\n\n## Question\n..."
def gold_text(item) -> str                                  # "<answer>X</answer>" style target (label for MCQ, first answer otherwise)
def score(item, generation: str) -> float                   # MCQ: exact label; span: token-F1 (port deltaskill/kvskill reward_bench F1)
def chunk_document(tok, text: str, chunk_tokens: int = 512) -> list[tuple[int,int,str]]  # (start_tok, end_tok, text)
```
`tests/test_data.py` validates every produced split: schema, doc files exist, doc_id reuse counts
(QuALITY ≥ 10 Qs/doc), counterfactual twins paired, token counts within expected ranges, `score()` sanity.

## [D] backbone / losses / scripts API

```python
# xenc/backbone.py
def load_reader(hub: str, dtype=torch.bfloat16, device_map="cuda", max_mem: str | None = None, grad_ckpt: bool = False)
   # Qwen3.5 → AutoModelForImageTextToText fallback (see deltaskill.backbone_utils.load_backbone); sets eval(), requires_grad False
def load_encoder(hub: str, **kw)   # same loader
def build_chat(tok, hub: str, system: str | None, user: str, add_generation_prompt=True) -> list[int]
def generate(model, tok, hub, ids, max_new_tokens, stop="</answer>") -> str   # greedy, returns decoded final text (final_channel for gpt-oss)

# xenc/losses.py
def teacher_logits(model, ids_ctx, m) -> Tensor[m+1, V]           # no_grad, use_cache=False, logits_to_keep=m+1
def student_logits(model, ids_q, m) -> Tensor[m+1, V]             # grad enabled through the interface only
def kl_teacher_student(p_t_logits, p_s_logits) -> Tensor          # KL(p_T || p_S) averaged over positions
def answer_ce(logits, target_ids) -> Tensor                        # manual CE on the answer span
def magnitude_penalty(injector) -> Tensor                          # mean of injector.mag_terms
def trace_target(teacher_generation: str) -> str                   # for --target trace (teacher's greedy trace up to </answer>)
```
`scripts/register_doc.py --reader Qwen/Qwen3.5-9B --encoder Qwen/Qwen3.5-2B --split_dir data/quality_split
  --cache_dir ... --out outputs/<run> --steps 300 --lr 2e-4 --target gold|trace --kl 1.0 --ce 0.5 --mag 0.3
  --route_aux 0.1 --chunk_tokens 512 --top_k 0 --selector 0|1 --max_docs N --seed 0 --batch 1 --max_mem 76GiB`
Per step: sample an item; teacher = reader on `ctx_prompt` (doc text); student = reader on `q_prompt` with the
item's Library loaded; loss = kl·KL + ce·CE + mag·mag + route_aux·(−log w[gold chunk]) when `meta.supporting_idx`
or a gold-span chunk is known. Saves `interface_best.pt` (selected on val score every `--eval_every`), `history.json`,
`args.json`, and prints `torch.cuda.max_memory_allocated()` at the end. Skips items whose doc is not in the cache
(warn). `--encoder same` means encoder == reader (L/L): encode with the reader itself.
`scripts/eval_doc.py --reader ... --split_dir ... --split test --interface outputs/<run>/interface_best.pt
  --conds closed_book,text,kv,wrong_doc,null,oracle_chunk --n 300 --budget 64 --batch 1`
writes `outputs/<run>/eval_<split>.json` with per-condition score, mean generated tokens, and per-item records.
`wrong_doc` = library of another document with the same task (deterministic shift by +7 in doc order);
`oracle_chunk` = library restricted to the gold chunk(s) (`meta.supporting_idx` or the chunk containing the first
gold answer string; if unknown, skip the item in that condition).
`scripts/ladder.py --readers Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B,Qwen/Qwen3.5-27B --split_dir ...
  --conds closed_book,text --n 200` → `outputs/ladder/<task>.json`.
Slurm: `slurm/register_doc.sbatch`, `slurm/eval_doc.sbatch`, `slurm/ladder.sbatch`, `slurm/smoke.sbatch`
(`#SBATCH --account=drjieliu --partition=drjieliu-h100 --gres=gpu:1 --mem=120G --time=4:00:00`; L40S variant
for 9B). Every launcher exports the HF env above and `PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels`.
`tests/test_register_plumbing.py`: with TinyLM + fake tokenizer, run 3 registration steps end-to-end on CPU
(monkeypatched `teacher_logits/student_logits` using TinyLM outputs projected to a 50-token vocab), assert loss
finite, interface grads nonzero, backbone grads None, and that `eval_doc`'s condition builders produce the right
libraries (wrong_doc ≠ own doc; oracle subset size).

## Acceptance (integration agent)
**NO GPU JOBS. Do not call `srun`, `sbatch`, or load any HF model checkpoint. The user will authorize the
first GPU run personally.** CPU verification only:
`python -m pytest tests -q` (install pytest with `pip install --target $REPO/.deps pytest` and add to PYTHONPATH)
and every `python tests/test_*.py` passes; `python scripts/data/prepare_*.py` produced all four splits on the
login node (CPU; tokenizer-only); every script supports `--dry_run` / `--tiny` that exercises its full control
flow with `tests/_tiny.TinyLM` and a fake tokenizer on CPU (`register_doc.py --tiny --steps 3`,
`eval_doc.py --tiny --n 4`, `encode_docs.py --tiny`, `ladder.py --tiny`) and exits 0. Slurm launchers are
written but not submitted.

## [D7] native document memory of the hybrid reader (`xenc/native_state.py`, `xenc/native_attn.py`)
Qwen3.5 (transformers 5.13) keeps two document memories after a prefill: full-attention K/V (8 of 32 layers in
the 9B, grows with T) and Gated-DeltaNet state per linear layer (24 layers in the 9B, 18 in the 2B):
`recurrent [1, H_v, d_k=128, d_v=128]` (S = Σ decayed β·v kᵀ — the paper's operator M = W Uᵀ, read linearly) +
`conv [1, conv_dim, kernel=4]`. 9B: H_v = 32; 2B: H_v = 16. Fixed size (9B ≈ 25 MB fp16 recurrent), independent of T.

`xenc.native_state` (D10 builds on this):
- `harvest_states(model, tok, doc_text=None, ids=None, max_tokens=0) -> {"recurrent": {li: T}, "conv": {li: T},
  "kv": {li: (K [1,n_kv,T,hd] post-RoPE, V)}, "n_doc": int, "layer_types": [...]}` — one prefill of the RAW
  document tokens (no chat template), tensors on the model's device/dtype, cloned out of a `DynamicCache`.
- `build_prefix_cache(model, st, keep_kv=True, keep_state=True, keep_conv=True) -> DynamicCache` — fresh cache
  (tensors cloned; generation mutates it). keep all == in-context prefill of [doc; prompt] (RoPE positions
  continue at n_doc); keep nothing == prompt alone. A linear layer is "has_previous_state" iff keep_state or
  keep_conv (the dropped part is zero; a missing conv entry is zero-filled). Without KV the prompt starts at
  position 0 (RoPE is relative).
- `greedy_continue(model, tok, cache, prompt_ids, max_new_tokens=64, stop="</answer>", hub="") -> (text, n)`;
  `continue_logits(model, cache, ids)`; `state_nbytes(st)`; `layer_types(model)`; `new_cache(model)`.
- `tile_states(src, src_types, dst_types, dst_heads, dst_conv_dim=None)` — zero-training cross-model map of a
  smaller encoder's recurrent states: reader linear layer j ← encoder linear layer at the nearest normalized
  depth j/(n−1); reader head h ← encoder head h·H_src//H_dst (d_k, d_v equal across sizes). Conv not mapped.
- Tests: `tests/test_native_state.py` (tiny real Qwen3_5ForCausalLM from `tests/_tiny.tiny_qwen35`: keep-all ==
  in-context logits within 1e-4; keep-none == closed book; partial caches; tiling shapes).
- Script: `scripts/eval_state.py` (conditions closed_book / text / all / kv_only / state_only / rec_only /
  wrong_kv / wrong_state, + tile_state / wrong_tile with `--encoder`), `slurm/D7_state.sbatch`.

`xenc.native_attn` (first D7 line; a negative result): `NativeAttnLibrary/NativeSpec` (raw attention K/V harvest,
`--harvest attn_raw`), `NativeAttnInterface` (per-layer null logit / gate / log-gain / log-temp, optional KV maps),
`NativeAttnInjector` (reader's own q_proj/q_norm/o_proj + output gate over the operator on a residual branch;
`rope_mode="doc"` = RoPE-consistent read, `head_mask`), `teacher_attention(model, ids, layers)` (real in-context
last-token attention). Flags: `register_doc/eval_doc --read native_attn`, `probe_read --read native_attn`.

### [D7] which layers carry the read (Qwen3.5-9B reader, training-free, LongHealth fold 0 n=80; QuALITY in outputs/D7_SUMMARY.md)
Full-attention layers of the 9B: 3, 7, 11, 15 ("early") and 19, 23, 27, 31 ("late"). Document memory = harvested
from one raw-document prefill (`harvest_states`), loaded via `build_prefix_cache` (+ `PrefixMask` to drop the
document from a subset of attention layers while keeping positions).
- all memory 0.81 (text 0.83, closed book 0.49). **Early KV + all DeltaNet states = 0.81 (= all); late KV + states
  = 0.24** (harmful, same as states alone). Late-layer KV is not needed.
- Early KV + conv states only (recurrent zero) = 0.76 (wrong doc 0.34); early KV + conv/recurrent only for linear
  layers < 16 = 0.80 (wrong doc 0.28). Recurrent states add ≈ +0.05 on top of KV + conv.
- DeltaNet states without any KV are harmful (0.24 vs closed book 0.49), independent of scale (×0.25 → 0.30).
- Single layers (with all states): kv@3 0.29, kv@7 0.34, kv@11 0.59, kv@15 0.38; pairs {3,7} 0.43, {11,15} 0.66 —
  the read is distributed over the four early layers; no single layer suffices.
- KV without conv (kv_only 0.575) loses ~0.19 vs KV + conv (0.76): the DeltaNet layers must be in "continuation"
  mode. A native reader prefill of the last 64 document tokens (its own conv/recurrent states, no harvested state)
  on top of the early KV gives 0.725 (wrong-doc body 0.44): the state requirement reduces to a short native tail
  prefill at a cost of ≈ −0.09 vs all; with the harvested body recurrent state it is back to 0.81.
⇒ Cross-model interfaces (D8/D10) should target the K/V of the EARLY full-attention layers (3, 7, 11, 15; ≈ 176 MiB
fp16 for an 11k-token document) and let the reader produce conv/recurrent states itself (tail prefill), optionally
adding the recurrent state of layers < 16.
