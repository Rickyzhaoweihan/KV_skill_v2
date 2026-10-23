#!/bin/bash
#SBATCH --job-name=D8_prune
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100,drjieliu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D8_prune_%j.out
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
# D8 phase 2: query-independent column pruning of the reader's native document KV (training-free).
# Usage: TASK=longhealth OUT=D8_prune_lh EXTRA="--fold 0" sbatch slurm/D8_prune.sbatch
#        TASK=quality OUT=D8_prune_q EXTRA="--n 200" sbatch slurm/D8_prune.sbatch
READER=${READER:-Qwen/Qwen3.5-9B}; TASK=${TASK:-longhealth}; OUT=${OUT:?set OUT}
python scripts/eval_prune.py --reader "$READER" --split_dir data/${TASK}_split --split test --out outputs/$OUT \
    --device cuda ${EXTRA:-}
echo D8_PRUNE_DONE
