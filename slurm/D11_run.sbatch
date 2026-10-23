#!/bin/bash
#SBATCH --job-name=D11_run
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D11_run_%j.out
# D11 (statistical consolidation of the encoder-side read). ONE job, N lanes = N GPUs, each lane a sequential
# bash script (outputs/D11_code/lane<i>.sh) pinned to its own GPU. Log every job id in slurm/logs/D11.jobids.
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels${PYTHONPATH:+:$PYTHONPATH}
export PYTHONPATH=$PYTHONPATH:$REPO/scripts   # outputs/D11_code snapshots import register_doc etc.
export XENC_CACHE=${XENC_CACHE:-/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache}
export TMPDIR=/scratch/drjieliu_root/drjieliu/rickyhan/hf_caches/tmp
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35
cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
LANES=${LANES:-0,1}   # at most 2 GPUs per D11 job (coordinator rule); pass LANES=<a>,<b>
i=0
for l in $(echo $LANES | tr ',' ' '); do
  CUDA_VISIBLE_DEVICES=$i bash outputs/D11_code/lane$l.sh > slurm/logs/D11_lane${l}_${SLURM_JOB_ID}.out 2>&1 &
  i=$((i+1))
done
wait
echo D11_DONE
