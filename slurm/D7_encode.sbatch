#!/bin/bash
#SBATCH --job-name=D7_encode
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100,drjieliu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D7_encode_%j.out
# D7: raw (un-normalized) attention K/V harvests for the native-attention read.
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
FULL8=0.09375,0.21875,0.34375,0.46875,0.59375,0.71875,0.84375,0.96875
E="python scripts/encode_docs.py --cache_dir $XENC_CACHE --harvest attn_raw"
# 9B LongHealth, all 8 full-attention layers (L/L)
$E --encoder Qwen/Qwen3.5-9B --split_dir data/longhealth_split --depths $FULL8
# QuALITY standard depths: 2B train(40)/val/test, 9B train(40)/test
$E --encoder Qwen/Qwen3.5-2B --split_dir data/quality_split --splits train --max_docs 40
$E --encoder Qwen/Qwen3.5-2B --split_dir data/quality_split --splits val,test
$E --encoder Qwen/Qwen3.5-9B --split_dir data/quality_split --splits train --max_docs 40
$E --encoder Qwen/Qwen3.5-9B --split_dir data/quality_split --splits test
echo D7_ENCODE_DONE
