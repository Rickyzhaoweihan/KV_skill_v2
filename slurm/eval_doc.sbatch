#!/bin/bash
#SBATCH --job-name=xeval
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/xeval_%j.out
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

# Usage: READER=Qwen/Qwen3.5-9B TASK=quality RUN=q_9b_2b SPLIT=test \
#        sbatch slurm/eval_doc.sbatch [extra eval_doc.py args, e.g. --conds kv,wrong_doc --n 300]
READER=${READER:-Qwen/Qwen3.5-9B}
TASK=${TASK:-quality}
RUN=${RUN:?set RUN=<outputs/ dir holding interface_best.pt>}
SPLIT=${SPLIT:-test}
DEVICE=${DEVICE:-cuda}
MAX_MEM=${MAX_MEM:-76GiB}
python scripts/eval_doc.py --reader "$READER" --split_dir data/${TASK}_split --split "$SPLIT" \
    --interface outputs/$RUN/interface_best.pt --cache_dir "$XENC_CACHE" \
    --device "$DEVICE" --max_mem "$MAX_MEM" "$@"
