#!/bin/bash
#SBATCH --job-name=D26_run
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s,drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=140G
#SBATCH --time=16:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# D26 (cross-family readers: gpt-oss-20b, GLM-4-9B-0414, DeepSeek-V2-Lite-Chat; encoder Qwen3.5-4B). Same CMDS protocol as D13_run.sbatch.
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
# Usage: CMDS='python scripts/a.py ...;;python scripts/b.py ...' sbatch slurm/D26_run.sbatch
# runs each ';;'-separated command in order; stops at the first failure.
IFS=$'\n'; for c in $(echo "$CMDS" | sed 's/;;/\n/g'); do echo ">>> $c"; eval "$c" || exit 1; done
echo D26_DONE
