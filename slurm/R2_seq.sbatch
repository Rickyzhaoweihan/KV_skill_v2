#!/bin/bash
#SBATCH --job-name=D00R2_seq
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=14:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# R2 (query-side read forms, xenc/r2_read.py): several register_d13 arms run ONE AFTER ANOTHER in one 2-GPU job
# (encoder on cuda:0, reader on cuda:1), so the arms share the GPU type (and node) and use one cap slot.
# Usage: CMDS='python scripts/register_d13.py ... --out A;;python scripts/register_d13.py ... --out B' sbatch slurm/R2_seq.sbatch
# Unlike D13_run.sbatch, a failed arm does NOT stop the next one (each arm writes its own --out dir); failures are listed.
# A step exiting 75 (eval_d13 --oom_skip after a CUDA OOM) is rerun in a fresh process and resumes (up to 60 times).
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels${PYTHONPATH:+:$PYTHONPATH}
export XENC_CACHE=${XENC_CACHE:-/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache}
export TMPDIR=/scratch/drjieliu_root/drjieliu/rickyhan/hf_caches/tmp
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35
set -uo pipefail
cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
hostname
FAILED=()
IFS=$'\n'; for c in $(echo "$CMDS" | sed 's/;;/\n/g'); do
  echo ">>> [$(date '+%F %T')] $c"
  # exit 75 = a step that hit a CUDA OOM and asks to resume in a FRESH process (eval_d13 --oom_skip): rerun it
  for try in $(seq 1 60); do ( eval "$c" ); rc=$?; [ $rc -eq 75 ] || break; echo "... exit 75 (OOM resume), rerun $try"; done
  [ $rc -eq 0 ] || { echo "!!! arm failed (rc $rc): $c"; FAILED+=("$c"); }
done
echo "failed arms: ${#FAILED[@]}"
echo R2_SEQ_DONE
