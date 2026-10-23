#!/bin/bash
#SBATCH --job-name=lanes
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s,drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# Generic multi-lane launcher: pack several independent runs into ONE job slot (user cap: 5 jobs running+pending),
# one lane per GPU. Pass LANE0..LANE7 as env vars; each is a sequence of shell commands separated by ";;" that run
# in order on that lane's GPU. Lane logs: slurm/logs/<jobname>_<jobid>_lane<i>.out. Override --gres/--job-name on
# the sbatch command line (e.g. via slurm/submit_when_free.sh 5 LANE0=... LANE1=... -- --gres=gpu:2 --job-name=D16_s2).
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
ngpu=$(nvidia-smi -L 2>/dev/null | wc -l)
i=0
for k in 0 1 2 3 4 5 6 7; do
  v="LANE$k"; cmds="${!v}"; [ -z "$cmds" ] && continue
  [ "$i" -ge "$ngpu" ] && { echo "lane $k skipped: only $ngpu GPUs"; continue; }
  log=slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_lane$k.out
  ( export CUDA_VISIBLE_DEVICES=$i
    echo "$cmds" | awk 'BEGIN{RS=";;"} {gsub(/^[ \n]+|[ \n]+$/,""); if(length) print}' | while IFS= read -r c; do
      echo ">>> [$(date +%H:%M)] $c"; bash -c "$c" || echo "!!! lane $k step failed: $c"; done
    echo "LANE${k}_DONE" ) > $log 2>&1 &
  i=$((i+1))
done
wait
grep -h "LANE.*_DONE\|step failed" slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_lane*.out
echo "${SLURM_JOB_NAME}_LANES_DONE"
