#!/bin/bash
#SBATCH --job-name=xsmoke
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100,drjieliu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/xsmoke_%j.out
# NOT submitted by any agent — the user authorizes the first GPU run.
# GPU smoke of the whole S/L pipeline on a small pair (fits one L40S or one H100):
#   1) encode the first 3 train + first 3 val docs with the ENCODER (2B)
#   2) register a READER (4B) interface for 5 steps on those docs (eval on 4 val items at step 5)
#   3) eval_doc on 6 val items of those docs, all conditions
# Each stage prints torch.cuda.max_memory_allocated().
# Usage: sbatch slurm/smoke.sbatch            (defaults: READER=Qwen/Qwen3.5-4B ENCODER=Qwen/Qwen3.5-2B TASK=quality)
#        READER=... ENCODER=... TASK=musique sbatch slurm/smoke.sbatch
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
set -euo pipefail   # after activation: conda's activate scripts are not set -u clean
cd $REPO
mkdir -p "$XENC_CACHE" "$TMPDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

READER=${READER:-Qwen/Qwen3.5-4B}
ENCODER=${ENCODER:-Qwen/Qwen3.5-2B}
TASK=${TASK:-quality}
NDOCS=${NDOCS:-3}
OUT=${OUT:-outputs/smoke_${TASK}_$(basename $READER)_$(basename $ENCODER)}

echo "== [1/3] encode ($ENCODER, $NDOCS train + $NDOCS val docs)"
python scripts/encode_docs.py --encoder "$ENCODER" --split_dir data/${TASK}_split --splits train,val \
    --max_docs "$NDOCS" --cache_dir "$XENC_CACHE"
echo "== [2/3] register (reader $READER, encoder $ENCODER, 5 steps)"
python scripts/register_doc.py --reader "$READER" --encoder "$ENCODER" --split_dir data/${TASK}_split \
    --cache_dir "$XENC_CACHE" --out "$OUT" --steps 5 --eval_every 5 --n_val 4 --max_docs "$NDOCS"
echo "== [3/3] eval (val, $NDOCS docs, n=6)"
python scripts/eval_doc.py --reader "$READER" --split_dir data/${TASK}_split --split val \
    --interface "$OUT/interface_best.pt" --cache_dir "$XENC_CACHE" --max_docs "$NDOCS" --n 6 --budget 256 \
    --no_resume
echo "== smoke OK -> $OUT"
