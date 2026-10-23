#!/bin/bash
#SBATCH --job-name=D23_eff
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=110G
#SBATCH --time=06:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D23_%j.out
# D23 efficiency benchmark (scripts/bench_d23.py). Job ids in slurm/logs/D23.jobids.
# Usage: CMDS='python scripts/bench_d23.py ...;;python ...' sbatch slurm/D23_run.sbatch  (';;'-separated, stops at first failure)
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
echo "$SLURM_JOB_ID $SLURM_JOB_NAME $(date)" >> slurm/logs/D23.jobids
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit --format=csv,noheader || true
IFS=$'\n'; for c in $(echo "$CMDS" | sed 's/;;/\n/g'); do echo ">>> $c"; eval "$c" || exit 1; done
echo D23_DONE
