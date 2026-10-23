#!/bin/bash
#SBATCH --job-name=xenc
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/xenc_%j.out
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

# Usage: ENCODER=Qwen/Qwen3.5-2B TASK=quality sbatch --partition=drjieliu-l40s slurm/encode_docs.sbatch \
#            [--splits train,val,test --max_docs N --depths 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 ...]
# Default depths = xenc.DEPTHS (register_doc --selector 0). Existing cache files are skipped.
ENCODER=${ENCODER:-Qwen/Qwen3.5-2B}
TASK=${TASK:-quality}
python scripts/encode_docs.py --encoder "$ENCODER" --split_dir data/${TASK}_split --cache_dir "$XENC_CACHE" \
    --device "${DEVICE:-cuda}" "$@"
