#!/bin/bash
#SBATCH --job-name=xladder
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/xladder_%j.out
# Written by agent [D]; NOT submitted. 27B readers need drjieliu-h100 (one H100 80GB);
# readers <=9B may add the L40S partition: sbatch --partition=drjieliu-h100,drjieliu-l40s ...
# (on L40S pass MAX_MEM=40GiB if sharding with DEVICE=auto).
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
set -uo pipefail   # after activation: conda's activate scripts are not set -u clean
cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Usage: READERS=Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B,Qwen/Qwen3.5-27B TASK=quality \
#        sbatch slurm/ladder.sbatch [--n 200 --budget 256 ...]
# Results merge into outputs/ladder/<task>.json, so readers can also be split across jobs.
READERS=${READERS:-Qwen/Qwen3.5-2B,Qwen/Qwen3.5-4B,Qwen/Qwen3.5-9B,Qwen/Qwen3.5-27B}
TASK=${TASK:-quality}
python scripts/ladder.py --readers "$READERS" --split_dir data/${TASK}_split \
    --conds closed_book,text --device "${DEVICE:-cuda}" "$@"
