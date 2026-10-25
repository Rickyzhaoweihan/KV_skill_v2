# KV-Skill v2 in Docker (for AWS EC2)

Goal: someone who gets the project on an EC2 GPU box can `docker build` (or pull), mount four directories, and run the same
inference / baseline / training commands with the same Python, torch, CUDA and transformers as on the UMich cluster.
Everything here was prepared and partly verified on the cluster on 2026-10-25 (section 9 says exactly what was and was not verified).

Contents of this directory: `Dockerfile`, `build.sh`, `prepare_context.sh` (snapshots code + checkpoints into a build context),
`requirements.lock`, `requirements.nodeps.lock`, `conda-env.lock`, `pip_freeze.full.txt`, `ENVIRONMENT.md` (versions, decisions),
`download_models.sh`, `package_data.sh`, `run_eval_example.sh`, `smoke_test.sh`.

## 1. What is in the image, what is mounted

| in the image (`/opt/kvskill`) | mounted at run time |
|---|---|
| python 3.11.15 + torch 2.12.1+cu130 + transformers 5.13.0 etc. (`/opt/venv`) | HF cache (Qwen3.5-4B/9B/27B, bge, llmlingua-2) at `/data/hf_home` |
| code snapshot: `KV_skill_v2/` (xenc, scripts, tests, slurm launchers, docs), `delta-skill/deltaskill`, `fla_kernels` | `data/` (splits + docs) at `/opt/kvskill/KV_skill_v2/data` |
| **two headline interface checkpoints** (fp32, unchanged) + their B maps in `/opt/kvskill/ckpt` (when `INCLUDE_CKPT=1`, default) | `outputs/` (set json files, results written here) at `/opt/kvskill/KV_skill_v2/outputs` |
| `docker_GIT_STATE.txt`, `docker_GIT_DIFF_vs_HEAD.patch`, `docker_MANIFEST.sha256` (provenance) | `XENC_CACHE` at `/data/xenc_cache` (operator caches; only the 27B path and training need it) |

The Qwen backbones are never baked in.

Baked-in checkpoints (`/opt/kvskill/ckpt`, listed with sha256 in `/opt/kvskill/ckpt/SHA256SUMS`; `SOURCE_SHA256SUMS` are the sums of the originals):

| file in image | original path (cluster) | reader | size | sha256 |
|---|---|---|---|---|
| `DS_max8k/d13_step6000.pt` | `outputs/DS_max8k/d13_step6000.pt` | 9B (encoder 4B) | 0.56 GB | `d271aaa6755c13642553d5e47e2f5dfc34d33e2d094e399ac9031f488a024e9f` |
| `D12b_max13k_s2/ckpt_5000.pt` | `outputs/D12b_max13k_s2/ckpt_5000.pt` | 27B (encoder 9B) | 2.47 GB | `ea25208a8e084e0f2a292414b25d13ec406f29ed4c92ca438522549a99f20358` |
| `maps/encB5_9b_4b_post_xf01.pt` | `outputs/D11_maps/encB5_9b_4b_post_xf01.pt` | B map for the 9B/4B pair | 0.21 GB | `5ad489e2ba2d96210384696a1000b13f90e8cab810ff33ded435cff676383fcf` |
| `maps/encB2_27b_9b_m.pt` | `$XENC_CACHE/D12_maps/encB2_27b_9b_m.pt` | B map for the 27B/9B pair | 0.67 GB | `3f9424af312d260eee11bd071f6a61f4cf404650b2199ed236c84feb8a2dacc9` |

Both interface files are already fp32 (checked: every tensor float32), copied byte for byte. The two B maps are not trainable
interfaces but the evals cannot run without them, hence they are in the image too.

**The image carries trained weights: push it to a PRIVATE registry only (private ECR repository), never Docker Hub public / ECR Public.**
Build the slim variant (`INCLUDE_CKPT=0`) if you want something shareable; ship the checkpoints separately then.

## 2. Pick the EC2 machine

CUDA: torch 2.12.1 is the **cu130** build, so the host NVIDIA driver must be **>= 580.65 (CUDA 13.0 capable)**. Check with
`nvidia-smi` ("CUDA Version: 13.x" in the header). cu130 supports Turing and newer (T4, A10G, A100, L40S, H100, ...); it does **not**
support V100. The container needs only the driver, via the NVIDIA container toolkit; the CUDA libraries are inside the image.

| job | instance | why |
|---|---|---|
| 9B reader + 4B encoder, one GPU (what the cluster L40S rows used) | **g6e.2xlarge** (1x L40S 48 GB, 64 GB RAM) | same GPU type as the cluster L40S rows; 4B (8 GB) + 9B (18 GB) + activations for 16k-token docs. g6e.xlarge has only 32 GB host RAM: too tight |
| 9B on H100 rows | p5.4xlarge (1x H100 80 GB) or p5.48xlarge | cluster H100 rows were produced on H100 |
| 27B reader (eval_d12p, training) | **p5.4xlarge** (1x H100 80 GB, 256 GB RAM) if available in your region/capacity, otherwise p5.48xlarge (8x H100) / p4de.24xlarge (8x A100 80 GB) | 27B bf16 = 54 GB; the 9B encoder must NOT sit on the same GPU, hence the cached-operator route (section 7) |
| g5 (A10G 24 GB) | g5.12xlarge (4x A10G) only for experiments | 9B reader alone is 18 GB: 16k-token docs will OOM; `DEVS="--enc_device cuda:0 --device cuda:1"` is the only layout that could work. **Not tested** |

Greedy decodes differ between GPU types by about .01-.03 (bf16 kernels). Compare only against rows from the same GPU type: L40S rows
live in `outputs/DSsw_*`, H100 rows are named in HANDOFF. A10G has no cluster reference rows at all, so expect differences and do not call them regressions.

AMI: an AWS **Deep Learning Base GPU AMI (Ubuntu 22.04/24.04)** of recent date (Docker and the NVIDIA container toolkit preinstalled). Check the
driver is >= 580.65 after launch; if the AMI ships an older driver pick a newer AMI or install `nvidia-driver-580-open` (then reboot).
(I could not look at the current AMI catalogue from the cluster, so the AMI name/driver pairing is for you to confirm.)

Disk (EBS gp3, root or an extra volume mounted at `/data`): image 10 GB on disk (6.05 GB slim; the registry/oci-archive form is 6.7 GB / ~3 GB) + HF models 98 GB
(4B 11 + 9B 22 + 27B 62 + bge 0.4 + llmlingua-2 2.4) + data 0.6 GB + results. **Provision >= 300 GB (HF 98 GB if all models, image 10 GB, work space) for the 9B path, >= 600 GB if you also run 27B
with operator caches** (section 7: +165 GB for the judge-set caches), several TB for training (the cluster's `XENC_CACHE` is 1.9 TB, `outputs/` 390 GB).
Use an instance-store NVMe (g6e/p5 have them) for `XENC_CACHE` when you can.

## 3. Get the image on the box

Option A, build on EC2 (reproducible, needs the project tree on the box: `KV_skill_v2/`, `delta-skill/deltaskill`, `fla_kernels/` as siblings,
plus the 4 checkpoint files, see section 4):
```bash
cd /data/project/KV_skill_v2
CKPT_SRC=/data/project/KV_skill_v2 MAP27=/data/project/KV_skill_v2/outputs/D12_maps/encB2_27b_9b_m.pt \
CTX=/data/build_ctx bash docker/build.sh              # -> kvskill-v2:cu130   (INCLUDE_CKPT=0 for the slim image, tag :cu130-slim)
```
`build.sh` calls `prepare_context.sh` (rsync of the code into `$CTX`, git state + sha256 manifest, checkpoint copy) and then `docker build`.
Pip/conda installs need internet; apt too. Nothing is pushed.

Option B, pull from your private ECR (after the user pushed it, section 8).

Option C, no Docker on the box (HPC style): `BUILDER=buildah SIF=1 bash docker/build.sh` produces a `.sif` (this is how it was proven on the cluster).

## 4. Move data and checkpoints (nothing is uploaded by these scripts; you run the `aws` commands)

On the cluster (or wherever the project lives):
```bash
bash docker/package_data.sh              # writes kvskill_data_eval.tar (~0.56 GB) + kvskill_ref_results.tar + SHA256SUMS under $OUT
```
Minimal set to reproduce the headline numbers:

| what | size | where from | needed for |
|---|---|---|---|
| `kvskill_data_eval.tar`: `data/*_split` (40 MB), `data/docs` (~0.5 GB packed, 1.1 GB on a block filesystem), `outputs/{D22_sets,MCQ_sets,OOD_sets}` (1.6 MB) | 0.56 GB | `package_data.sh` | every eval (items, docs, wrong-doc maps) |
| the 2 interface checkpoints + 2 B maps | 3.9 GB | inside the image (or the 4 files listed in section 1 if you build on EC2) | kv / wrong_doc rows |
| HF models Qwen3.5-4B, -9B (9B path); + -27B (27B path); bge, llmlingua-2 (baselines) | 33 GB / 98 GB total | `download_models.sh` from the HF Hub (or `aws s3 sync` of the cluster's `hub/` dirs) | all |
| `kvskill_ref_results.tar`: reference `eval_test.json` of `DSsw_m8k6000_*`, `D12b_max13k_c5000_*`, SUMMARY.md files | 5 MB | `package_data.sh` | to compare your rerun against the cluster numbers |
| 27B operator caches `XENC_CACHE/D12p/c9b_judge` (125 GB) + `c9b_mcq` (39 GB) | 164 GB | cluster `/scratch/.../xenc_cache/D12p/` | 27B eval without a second GPU (or regenerate, section 7) |
| NOT needed for inference: `outputs/` (390 GB, other ckpts), `XENC_CACHE` (1.9 TB), `data/raw*`, `data/D15_selfq` | | | training / archaeology only |

The 9B path needs **no** operator cache: `eval_d13.py` encodes the document on the fly (verified: it never touches `XENC_CACHE`).

S3 plan (commands for YOU to run; I did not run any of them):
```bash
# on the cluster, after package_data.sh   (needs AWS credentials that you hold)
aws s3 cp $OUT/kvskill_data_eval.tar   s3://<your-bucket>/kvskill/   && aws s3 cp $OUT/kvskill_ref_results.tar s3://<your-bucket>/kvskill/
aws s3 cp $OUT/SHA256SUMS              s3://<your-bucket>/kvskill/
# optional, 27B caches (164 GB):  aws s3 sync /scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache/D12p/c9b_judge s3://<your-bucket>/kvskill/xenc_cache/D12p/c9b_judge   (same for c9b_mcq)
# on EC2
mkdir -p /data/kvskill && cd /data/kvskill && aws s3 cp s3://<your-bucket>/kvskill/ . --recursive --exclude "*" --include "*.tar" --include SHA256SUMS
sha256sum -c SHA256SUMS && tar -xf kvskill_data_eval.tar && tar -xf kvskill_ref_results.tar   # -> /data/kvskill/{data,outputs}
```
Use an S3 gateway VPC endpoint (free) and a bucket in the instance's region; same-region S3 to EC2 transfer is free of egress charges.

## 5. Download the models (exact cluster revisions)

```bash
# inside the image (HF_HUB_OFFLINE is overridden by the script):
docker run --rm -v /data/hf_home:/data/hf_home kvskill-v2:cu130 bash docker/download_models.sh            # all 5 models, ~98 GB
docker run --rm -v /data/hf_home:/data/hf_home -e MODELS="4b 9b bge lingua2" kvskill-v2:cu130 bash docker/download_models.sh   # 9B path + baselines
```
Revisions are pinned (Qwen3.5-4B `851bf6e8...`, -9B `c2022362...`, -27B `fc05daec...`, bge-base-en-v1.5 `a5beb1e3...`,
llmlingua-2-xlm-roberta-large-meetingbank `ebaba9b0...`); the snapshot layout equals the cluster cache, so the `models--ORG--NAME/snapshots/<rev>` dirs can also be
copied from the cluster instead (`aws s3 sync`). Models used by code: `Qwen/Qwen3.5-{2B,4B,9B,27B}`, `BAAI/bge-base-en-v1.5`
(hybrid RAG), `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` (LLMLingua; LongLLMLingua scores with Qwen3.5-4B). Qwen3.5-2B is legacy (tests / old maps), not needed for the headline paths.
The HF datasets in the cluster cache (`datasets--*`) were only used to materialise `data/`; the tar replaces them.

## 6. Smoke test and one real eval

```bash
# CPU: imports + version assertions + checkpoint sha256 + scripts/run_tests.sh (pytest + every tests/test_*.py + --tiny/--dry_run + bash -n)
docker run --rm -v /data/hf_home:/data/hf_home:ro kvskill-v2:cu130 bash docker/smoke_test.sh
QUICK=1 docker run ... bash docker/smoke_test.sh           # imports/checksums only (no HF cache needed)

# GPU: eval_d13, 9B reader + 4B encoder, 2 QuALITY test items, conds closed_tail,text,kv,wrong_doc
DATA_ROOT=/data/kvskill HF_HOST=/data/hf_home bash docker/run_eval_example.sh            # MODE=27b for the 27B variant
```
`run_eval_example.sh` shows every mount (`-v` list) and env var and defaults to the baked-in checkpoint paths. Results land in
`$DATA_ROOT/outputs/DOCKER_example_9b/eval_test.json`. Keep the layout: scripts use paths relative to the repo (`data/...`, `outputs/...`).
For full runs copy the command lines out of `slurm/DS_sweep.sbatch` (9B, `--conds kv,wrong_doc`, `--oom_skip 1`, exit code 75 = resume: rerun the same command) and
`slurm/D12b_max8k.sbatch` (27B sweep section); replace `outputs/...` ckpt paths by `/opt/kvskill/ckpt/...`.

Other inference paths, same mounts (all use the same python env):
* `scripts/eval_d13.py` / `eval_d12p.py --conds closed_book,closed_tail,text,kv,wrong_doc`
* `scripts/eval_cad.py` (contrastive decoding), `scripts/eval_textbudget.py` (`--conds trunc_ht@2048,bm25@2048,hybrid@2048,lingua2@2048,llingua@2048`; needs bge + llmlingua-2 + Qwen3.5-4B in `HF_HOME`), `scripts/eval_kvcomp.py`.
* Training: `scripts/register_d13.py` / `register_d12p.py` (about 170 flags, see HANDOFF section 4). They need the big operator cache and teacher caches (`XENC_CACHE`, TBs) or must regenerate them; plan GPUs and disk accordingly. Not exercised in the image.

## 7. 27B on EC2 (cached-operator runtime)

`eval_d12p.py` is meant to run with `--enc_cache DIR` (operators precomputed by `scripts/cache_d12p.py`), so the 9B encoder never occupies the reader's GPU
(`--enc_device cpu --reader_mem 74GiB` in the cluster sweeps). Either (a) copy the 164 GB `c9b_judge`+`c9b_mcq` caches (section 4), or
(b) regenerate them on any GPU that fits the 9B encoder: `python scripts/cache_d12p.py --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B --layers 3,7,11,...,47 --out $XENC_CACHE/D12p/c9b_judge --evals_d22 outputs/D22_sets/qasper.json,... ` (see its docstring). Then pass `--enc_cache` (comma list) as in `slurm/D12b_max8k.sbatch`.

## 8. Pushing the image (instructions for the user; nothing here was run)

```bash
aws ecr create-repository --repository-name kvskill-v2 --image-scanning-configuration scanOnPush=true --region <region>   # PRIVATE ECR (default)
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <acct>.dkr.ecr.<region>.amazonaws.com
docker tag  kvskill-v2:cu130 <acct>.dkr.ecr.<region>.amazonaws.com/kvskill-v2:cu130
docker push <acct>.dkr.ecr.<region>.amazonaws.com/kvskill-v2:cu130
```
Do not push the `:cu130` (checkpoint-carrying) image to a public registry. Registry layer limit is fine; the checkpoint layer is ~3.9 GB.

## 9. What was verified on the cluster (2026-10-25), and what was not

The cluster has no Docker daemon, root or `/etc/subuid`. It has `buildah` 1.33, `podman` 4.9 (unusable rootless without subuid) and Apptainer/Singularity 4.4.1 (`/opt/singularity/4.4.1/bin/singularity`).
What worked, and how (all inside `/scratch/drjieliu_root/drjieliu/rickyhan/docker_build/`, nothing under turbo, nothing pushed):

* **The exact `Dockerfile` was built** with rootless `buildah` (`BUILDER=buildah bash docker/build.sh`): `--storage-driver vfs`, `ignore_chown_errors`, `--isolation rootless`;
  layer storage on the login node's local `/tmp` because NFS (`/scratch`, turbo) rejects vfs layers. Build time 4-9 min; both variants built: `kvskill-v2:cu130` (10 GB, with the 2 checkpoints) and `:cu130-slim` (6.05 GB; the `INCLUDE_CKPT=0` switch works, `/opt/kvskill/ckpt` holds only a README).
* **Converted to a `.sif`** (`SIF=1`): oci-archive 6.7 GB, `kvskill-v2_cu130.sif` 6.3 GB.
* **Inside the .sif** (versions, imports, checkpoint sha256 against `SHA256SUMS`, code-manifest) `docker/smoke_test.sh`: torch 2.12.1+cu130, CUDA 13.0, cuDNN 92000, transformers 5.13.0, triton 3.7.1 all as pinned; all four checkpoints `OK`.
* **Full CPU test in the .sif** on a compute node (`D00cpu_docker_smoke2`, with the HF cache, `data/` and `outputs/` mounted from the transfer tars): `scripts/run_tests.sh` -> pytest 378 tests passed, every `tests/test_*.py`, all `--tiny` / `--dry_run` paths, `bash -n` on the launchers -> **ALL GREEN**.
  (Without the `data/` mount two tests, `test_d12b` and `test_f1`, fail with FileNotFoundError: mount data.)
* The transfer path (`package_data.sh` -> tar -> extract -> used as mounts) was exercised end to end on the cluster.

NOT verified (no way to do it here): a real `docker build` / `docker run --gpus all` (same Dockerfile, but the daemon path and the NVIDIA container toolkit were not available), GPU numerics on A10G/A100/EC2 H100,
the 27B path inside the container (eval_d12p needs an 80 GB GPU and the operator caches), the baselines `eval_textbudget` (LLMLingua/bge) / `eval_kvcomp` / `eval_cad` on GPU in the container (their imports and unit tests pass on CPU), and training scripts.
The GPU reproduction check `docker/repro_check.sh` (5 items of the MCQ-hallu set decoded again with the baked 9B checkpoint on one GPU and compared token for token with the cluster's L40S rows) was submitted
as job `D00DK_smoke` through the cluster's cap queue; its result is in `slurm/logs/D00DK_smoke_<jobid>.out` (`REPRO_IDENTICAL` / `REPRO_DIFFERS`). Run the same script yourself on EC2 as the first GPU test:
`docker run --rm --gpus all -v ... kvskill-v2:cu130 bash docker/repro_check.sh` (mounts as in `run_eval_example.sh`, extracted `kvskill_ref_results.tar` under `outputs/`).

## 10. Risks (read before trusting a reproduced number)

* **Uncommitted code.** The snapshot is the working tree, not a commit: `git status` showed ~80 modified/untracked files (new baselines `cad`, `cmp_*`,
  `recon`, `textbudget`, `lingua`, `kvcomp`, new scripts and tests) on top of commit `5d86b5e`. `docker_GIT_STATE.txt` + `docker_GIT_DIFF_vs_HEAD.patch` +
  `docker_MANIFEST.sha256` record the exact state; commit it (user's decision) before treating the image as a release. `delta-skill` and `fla_kernels`
  are untracked in the parent repo and have no commit hash; only the sha256 manifest pins them. Other agents were editing the tree while the snapshot was taken.
* **Version drift.** Python/pip pins are exact, `micromamba` and the apt packages (gcc 13, Ubuntu 24.04, `nvidia/cuda:13.0.2-base-ubuntu24.04`) are not content-pinned
  (pin the base image by digest when you freeze a release: `docker inspect --format '{{index .RepoDigests 0}}'`). PyPI wheels can be yanked; keep the built image, not just the recipe.
  The cluster ran gcc 11.2 for triton's launcher compile; the image has gcc 13 (compiler only builds stubs; kernels are triton-compiled, not gcc).
* **GPU numerics.** L40S vs H100 differ by ~.01-.03 in greedy-decode scores; A10G and A100 are new GPU types with no cluster rows. cuDNN/cuBLAS versions are identical to the cluster
  (pip wheels), the driver is not. Re-run `text` on the new GPU type before judging kv vs text.
* **causal_conv1d deliberately absent** (ENVIRONMENT.md): do not add it.
* **Cluster paths.** `slurm/*.sbatch`, `run_tests.sh` and some docs hard-code `/nfs/turbo/...`; see ENVIRONMENT.md for how the container works around them. Any script that defaults to
  `outputs/...` needs the outputs volume mounted at the repo's `outputs/`.
* **Memory.** 27B + 9B encoder on one 80 GB GPU is not possible; use the cached-operator route.
* **Secrets.** None baked in. The checkpoint-carrying image is intellectual property; keep the registry private.
