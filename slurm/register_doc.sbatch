#!/bin/bash
#SBATCH --job-name=xreg
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/xreg_%j.out
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

# Usage: READER=Qwen/Qwen3.5-9B ENCODER=Qwen/Qwen3.5-2B TASK=quality RUN=q_9b_2b \
#        sbatch slurm/register_doc.sbatch [extra register_doc.py args...]
READER=${READER:-Qwen/Qwen3.5-9B}
ENCODER=${ENCODER:-Qwen/Qwen3.5-2B}
TASK=${TASK:-quality}
RUN=${RUN:-reg_${TASK}_$(basename $READER)_$(basename $ENCODER)}
DEVICE=${DEVICE:-cuda}
MAX_MEM=${MAX_MEM:-76GiB}
python scripts/register_doc.py --reader "$READER" --encoder "$ENCODER" \
    --split_dir data/${TASK}_split --cache_dir "$XENC_CACHE" --out outputs/$RUN \
    --device "$DEVICE" --max_mem "$MAX_MEM" "$@"
