# KV_skill_v2 — `xenc` (small encoder / large reader document operators)

`xenc` is a playground package for **document-as-skill** reading: an ENCODER model (e.g. Qwen3.5-2B)
prefills a whole document once, its residual stream at normalized depths ρ is cut into ≤512-token
chunk operators (no pooling, L2-normalized slots, K = V tied, fp16, cached on disk keyed by ρ), and a
frozen READER (e.g. Qwen3.5-9B/27B) reads the per-document **Library** of chunk operators through a
small trainable interface (deltaskill's `ModelInterface` with `d_m = d_reader`, `d_s = d_encoder`) +
per-token routing over chunks (+ optional sparse `LayerSelector` over 8 candidate depths).
Registration distills the reader-with-document-in-context (teacher) into the reader-with-question-only
+ Library (student). Only the interface (+ selector) trains; both backbones stay frozen.

Binding contract: `docs/XENC_SPEC.md`. Design: `docs/PLAYGROUND_PROPOSAL.md`, `docs/audits/CODE_PLAN_encoder_reader.md`.
It imports `deltaskill` from `../delta-skill` (never modified) and fla kernels from `../fla_kernels`;
`xenc/__init__.py` bootstraps both on `sys.path`. No install step (no `pyproject.toml` on purpose).

```
xenc/depth_skill.py   derive_by_depth (ρ-keyed chunk cache), for_reader, cache_key, save/load
xenc/library.py       Library: K[li] [C,N,d_s], mask, summary keys S
xenc/injector.py      XEncInjector / LibraryInjector (routing, masked read, selector; sharding-aware)
xenc/selector.py      LayerSelector (top-k over candidate layers, matched update norm)
xenc/interface.py     make_interface / save_interface / load_interface / EncoderLens
xenc/doc_tasks.py     DocItem, load_doc_split/load_docs, q_prompt/ctx_prompt, score, chunk_document
xenc/backbone.py      load_reader / load_encoder, build_chat (enable_thinking=False), generate
xenc/losses.py        teacher/student kept-logits, KL, manual CE, magnitude, route aux, trace target
scripts/encode_docs.py  register_doc.py  eval_doc.py  ladder.py  data/prepare_*.py  run_tests.sh
slurm/{smoke,encode_docs,register_doc,eval_doc,ladder}.sbatch       (never submitted by agents)
```

## Environment

```bash
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
PY=/nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35/bin/python          # torch 2.12.1, transformers 5.13.0
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1                       # compute nodes are offline; ~/.bashrc points HF at a dead cache
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels
export XENC_CACHE=/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache     # operator cache (default)
cd $REPO
```
pytest lives in `$REPO/.deps` (add it to `PYTHONPATH` for tests). Every launcher in `slurm/` exports
the same HF / `PYTHONPATH` / `XENC_CACHE` variables itself.

> **No GPU without authorization.** Nothing in this repo has been run on a GPU. Do not `sbatch`/`srun`
> any launcher or load a model checkpoint until the user has personally authorized the first GPU run.
> Respect the shared lab queue: a few jobs pending at a time, submit stages incrementally.

## 1. Tests (CPU only)

```bash
bash scripts/run_tests.sh                                  # pytest + each tests/test_*.py + every --tiny/--dry_run path + bash -n
PYTHONPATH=.deps $PY -m pytest tests -q                    # pytest only
$PY tests/test_library.py                                  # any single file runs as a script (prints OK)
```

## 2. Data prep (login node, CPU, tokenizer only — already done; re-run to regenerate)

```bash
$PY scripts/data/prepare_quality.py       # data/quality_split     (+ data/docs/quality/*.md)
$PY scripts/data/prepare_longhealth.py    # data/longhealth_split  (folds.json, meta.fold)
$PY scripts/data/prepare_lastingbench.py  # data/lastingbench_split (counterfactual twins)
$PY scripts/data/prepare_musique.py       # data/musique_split     ([P{i}] markers, meta.supporting_idx)
```
Raw inputs live in `data/raw_downloads/` (gitignored). `register_doc.py --dry_run` reports cache coverage
for a split without loading any model.

## 3. GPU smoke (ONLY after user authorization)

```bash
sbatch slurm/smoke.sbatch        # READER=Qwen/Qwen3.5-4B ENCODER=Qwen/Qwen3.5-2B TASK=quality by default
```
Runs encode (2B, first 3 train + 3 val docs) → register (4B reader, 5 steps, val eval on 4 items) →
eval_doc (val, those docs, n=6, all six conditions); every stage prints `max_memory_allocated`.
Log: `slurm/logs/xsmoke_<jobid>.out`; outputs: `outputs/smoke_quality_Qwen3.5-4B_Qwen3.5-2B/`.

## 4. P0 content gate (Qwen3.5-9B; after the smoke passes)

L/L first (the reader encodes its own documents; `--encoder same` encodes lazily and caches under the
reader's short name, so pre-encoding with `encode_docs.py --encoder Qwen/Qwen3.5-9B` is equivalent):
```bash
READER=Qwen/Qwen3.5-9B ENCODER=same TASK=quality RUN=p0_quality_9b_LL \
  sbatch --partition=drjieliu-l40s slurm/register_doc.sbatch --steps 300 --max_docs 20
READER=Qwen/Qwen3.5-9B TASK=quality RUN=p0_quality_9b_LL \
  sbatch --partition=drjieliu-l40s slurm/eval_doc.sbatch --n 300
```
Then S/L with a 2B encoder (encode every split first, then register and eval):
```bash
ENCODER=Qwen/Qwen3.5-2B TASK=quality sbatch --partition=drjieliu-l40s slurm/encode_docs.sbatch --splits train,val,test
READER=Qwen/Qwen3.5-9B ENCODER=Qwen/Qwen3.5-2B TASK=quality RUN=p0_quality_9b_2b \
  sbatch --partition=drjieliu-l40s slurm/register_doc.sbatch --steps 300 --max_docs 20
READER=Qwen/Qwen3.5-9B TASK=quality RUN=p0_quality_9b_2b \
  sbatch --partition=drjieliu-l40s slurm/eval_doc.sbatch --n 300
```
The gate is the content margin `kv − wrong_doc` (printed and stored as
`summary.kv_minus_wrong_doc` in `outputs/<RUN>/eval_test.json`) against `closed_book`, `null` and `text`.
For the selector (`--selector 1`) encode at the CAND depths: `--depths 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9`.
`eval_doc.py` resumes from `outputs/<RUN>/eval_<split>.ckpt.jsonl` (pass `--no_resume` to start over).

## 5. Text-in-context size ladder

```bash
READERS=Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B TASK=quality \
  sbatch --partition=drjieliu-l40s slurm/ladder.sbatch --n 200
READERS=Qwen/Qwen3.5-27B TASK=quality sbatch --partition=drjieliu-h100 slurm/ladder.sbatch --n 200
```
Results merge into `outputs/ladder/<task>.json` keyed by reader, so readers can run in separate jobs.

## Measured footprints (docs/audits/BACKBONE_NOTES.md probes, 1×H100, sdpa, no causal-conv1d)

| reader | weights | teacher 16k ctx | student fwd+bwd Q=512 | registration (300 steps) |
|---|---|---|---|---|
| Qwen3.5-9B  | 16.7 GiB | 18.9 GiB | 20.6 GiB | ≈ 3.5 min (fits one L40S) |
| Qwen3.5-27B | 50.1 GiB | 53.3 GiB | 57.7 GiB | ≈ 8.5 min (one H100) |

The probes did not include the Library read (`[B,T,C,N]` fp32 per layer, ~0.1–0.5 GB for QuALITY-size
documents) or the periodic val generation; budget extra wall-clock for `--eval_every`.
