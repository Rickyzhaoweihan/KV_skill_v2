#!/bin/bash
#SBATCH --job-name=cpu
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-v100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# CPU-only jobs (analysis, recompute, tests, data prep, self-question post-processing) on the V100 nodes WITHOUT a GPU.
# No --gres, so it does not count toward the KV-Skill GPU cap. Pass the commands as CMD="a ;; b ;; c".
# usage: bash slurm/submit_when_free.sh auto CMD="python scripts/x.py ... ;; bash scripts/run_tests.sh" -- --job-name=D00cpu_tag slurm/cpu.sbatch
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home HF_HUB_CACHE=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels:$REPO/scripts${PYTHONPATH:+:$PYTHONPATH}
export XENC_CACHE=${XENC_CACHE:-/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache}
export TMPDIR=/scratch/drjieliu_root/drjieliu/rickyhan/hf_caches/tmp TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35
cd $REPO
echo "$CMD" | awk 'BEGIN{RS=";;"} {gsub(/^[ \n]+|[ \n]+$/,""); if(length) print}' | while IFS= read -r c; do
  echo ">>> [$(date +%H:%M)] $c"; bash -c "$c" || echo "!!! step failed: $c"; done
echo "${SLURM_JOB_NAME}_DONE"
