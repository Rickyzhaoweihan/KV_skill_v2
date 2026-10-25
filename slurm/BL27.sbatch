#!/bin/bash
#SBATCH --job-name=D12b_bl_run
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/BL_%x_%j.out
# BL27 (27B reader, H100): S2/S4 text baselines + same-job text row. New file; BL.sbatch untouched.
# Usage: CMDS='python scripts/a.py ...;;python scripts/b.py ...' sbatch slurm/BL.sbatch   (a failed step is logged, the next step runs)
# Job ids in slurm/logs/BL.jobids.
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
echo "$SLURM_JOB_ID $SLURM_JOB_NAME $(date +%F_%T)" >> slurm/logs/BL.jobids
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
IFS=$'\n'; for c in $(echo "$CMDS" | sed 's/;;/\n/g'); do echo ">>> $c"; eval "$c" || echo "!!! step failed: $c"; done
echo BL_DONE
