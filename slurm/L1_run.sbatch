#!/bin/bash
#SBATCH --job-name=L1_run
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# L1 (length-aware read forms; outputs/L1_SUMMARY.md). ONE job that sees ALL its GPUs (unlike lanes.sbatch, which pins one GPU
# per lane): 2x L40S with the reader on cuda:1 and the 4B encoder on cuda:0 (--device cuda:1 --enc_device cuda:0).
# CMDS = ';;'-separated commands, run in order; a failed command is reported and the next one still runs.
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels:$REPO/scripts${PYTHONPATH:+:$PYTHONPATH}
export XENC_CACHE=${XENC_CACHE:-/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache}
export TMPDIR=/scratch/drjieliu_root/drjieliu/rickyhan/hf_caches/tmp
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35
cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "$CMDS" | awk 'BEGIN{RS=";;"} {gsub(/^[ \n]+|[ \n]+$/,""); if(length) print}' | while IFS= read -r c; do
  echo ">>> [$(date +%H:%M)] $c"; bash -c "$c" || echo "!!! step failed: $c"; done
echo "${SLURM_JOB_NAME}_DONE"
