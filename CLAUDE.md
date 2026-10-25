# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

KV-Skill v2, "Small Encoder, Large Reader". A frozen Qwen3.5 **encoder** (4B; 9B for the 27B reader) prefills a
document once into an operator: post-RoPE K/V at its read layers, cached on disk. A frozen **reader** (Qwen3.5-9B,
layers 3/7/11/15/19, or 27B, layers 3..47) then reads that operator with its own query, through a small trained
interface (A, M, B, gates). **The reader never sees the document text at test time.** Only the interface trains. The
main method is unregistered **kv@0** from a shared base checkpoint. The only comparison that counts is kv against the
**same reader reading the text**.

This is a separate git repo (code only). `data/`, `outputs/` (~390 GB), `slurm/logs/`, `slurm/waiters/` and `*.pt` are
gitignored. Don't commit unless the user asks. The parent `../CLAUDE.md` covers the paper and the sibling projects.

**Where the state lives (read these before any research or scheduling work):** `docs/HANDOFF.md` (start at §00) is the
master state. Its §2 holds the binding user rules. `docs/HANDOFF_LOG.md` is append-only: one dated line per result or
job change, and `[COORD-DISCUSSION]` entries for user decisions. Only the HANDOFF agent edits `HANDOFF.md`. Everyone
else appends to the log. `docs/AGENT_BRIEF.md` is the authoritative rule file. Each research topic has its results in
`outputs/<TAG>_SUMMARY.md` and its job ids in `slurm/logs/<TAG>.jobids`. **Reply to the user in Chinese.**

## Environment

```bash
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
PY=/nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35/bin/python     # torch 2.12.1, transformers 5.13.0
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home HF_HUB_CACHE=$HF_HOME/hub HF_HUB_OFFLINE=1
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels:$REPO/scripts
export XENC_CACHE=/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache
```
- There is no install step and no `pyproject.toml`. `xenc/__init__.py` puts `../delta-skill` (deltaskill) and
  `../fla_kernels` on `sys.path`. Every `slurm/*.sbatch` exports all of the variables above itself.
- Chat templates always use `enable_thinking=False` (`xenc/backbone.py build_chat`).
- The login node has no GPU. Run only CPU work there: tests, tables, leakage checks, `--tiny` / `--dry_run` paths.
  Anything heavier goes to `slurm/cpu.sbatch`.
- Turbo is about 99% full. HF downloads go only to `hf_home`. Caches and other big files go under
  `/scratch/drjieliu_root/drjieliu/rickyhan/`.

## Tests (CPU)

```bash
bash scripts/run_tests.sh               # pytest tests/ + each tests/test_*.py as a script + --tiny/--dry_run paths + bash -n on sbatch → "ALL GREEN"
bash scripts/run_tests.sh -k recon      # pytest subset only
PYTHONPATH=.deps:. $PY tests/test_recon.py   # one file (pytest lives in .deps/)
```
Keep `run_tests.sh` green. **Don't change the default behaviour of shared scripts.** Add a default-off flag or a new
file instead. New read-form modules are zero-initialised, so step 0 equals the parent.

## Architecture (big picture)

- `xenc/backbone.py` loads the reader and encoder. `xenc/doc_tasks.py` holds `DocItem`, the split loaders, the
  prompts and the scorers (also used as the RL reward).
- **The main read is `xenc/d13_read.py`** (9B): `D13Interface` (A, M, B, g), `D13Reader`, `EncTeacher`, and
  save/load. `d12p_read.py` is its 27B twin, with a cached-encoder runtime. The older `interface.py` / `injector.py` /
  `library.py` / `depth_skill.py` are the P0–D9 era (`README.md` describes that era).
- Optional modules plug into `d13_read` / `register_d13.py` through default-off flags: `recon.py`, `cad.py` (contrastive
  decoding), `cmp_op.py` / `cmp_hyb.py` (compression; the HYB deep channel), `notes.py`, `p3_tok.py` / `p6_native.py`
  (token injection), `l1_read.py`, `r2_read.py`, `addr_*.py`. Baselines: `textbudget.py`, `lingua.py`, `kvcomp.py`.
- Training: `scripts/register_d13.py` (9B base training, about 170 flags) and `register_d12p.py` (27B). Evaluation:
  `scripts/eval_d13.py` and `eval_d12p.py`, with `--conds closed_book,closed_tail,text,kv,wrong_doc` and `--oom_skip 1`
  (exit 75 means resume; the launcher reruns it, and this is not an error). CPU table scripts: `ds_curve.py`,
  `ds_sweep_table.py`, `cad_table.py` / `cad_gate.py`, `cmp_table.py`, `bl_table.py`, `softconf_table.py`. Several of
  them write into `outputs/` by default, so pass `--out` when you only want to look. Leakage check:
  `scripts/ds_judged_leak.py`.
- `docs/HANDOFF.md` §4 has the full flag and launcher map, and §4.5 has copy-paste command lines.

## Submitting GPU work (strict)

- Submit only with **one plain command**, run with `run_in_background`, because it waits for a cap slot:
  `bash slurm/submit_when_free.sh auto ENV=val ... -- --job-name=D00<TOPIC>_<tag> <sbatch args> slurm/<file>.sbatch`.
  No loops, `nohup`, `bash -c` or `&&` chains. For CPU work:
  `... auto CMD="a ;; b" -- --job-name=D00cpu_<tag> slurm/cpu.sbatch`.
- Ranking comes from the `--job-name=` regex against `slurm/priority.txt`. The cap comes from `slurm/cap_override`
  (18 non-held + #held). `WAIT_JOB=` and `WAIT_FILE=` set preconditions. 9B jobs use `-p drjieliu-l40s,drjieliu-h100`;
  27B jobs use `-p drjieliu-h100 --gres=gpu:1`.
- **Never edit in place a script that a live waiter or job is executing** (bash re-reads by byte offset). Write a copy
  and `mv` it over atomically. Never kill another agent's waiter or job.
- Log manual `scontrol hold/release/update` actions in `slurm/logs/coordinator_cancel.log`.
- L40S and H100 greedy decodes differ by about .01–.03. Compare only rows from the same GPU type.

## Hard rules

- Never modify `../delta-skill`, `../cross_model` or `docs/report*.html`. Never write under
  `/nfs/turbo/umms-drjieliu/usr/xiangzhg/` (Hub Memory, a colleague's code: read-only data).
- Test docs and questions never enter train, val, registration or RL (`--exclude_sets`, `--exclude_docs`, the leakage
  assertion). `--select_min_step` keeps step 0 from ever being selected. A result identical to its parent is a bug.
- Subagents: Opus for research, Sonnet for routine work, **never Fable**. Agents end their turn after submitting.
  The coordinator's monitors wake them.
- Results tables show raw kv and text side by side, with the reader (and encoder) named, and **no kv−text columns**.
  Label a claim "preliminary" unless it has full n, paired SE, a 2nd seed, an independent recompute, sanity rows and a
  leakage check.
