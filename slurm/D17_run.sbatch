#!/bin/bash
#SBATCH --job-name=D17_run
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s,drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=40:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D17_run_%j.out
# D17 (reward-learned paper-form reader-query read, GRPO on top of D13 inits). Job ids in slurm/logs/D17.jobids.
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
# Usage: CMDS='python scripts/a.py ...;;python scripts/b.py ...' sbatch slurm/D17_run.sbatch
# runs each ';;'-separated command in order; stops at the first failure.
IFS=$'\n'; for c in $(echo "$CMDS" | sed 's/;;/\n/g'); do echo ">>> $c"; eval "$c" || exit 1; done
echo D17_DONE
