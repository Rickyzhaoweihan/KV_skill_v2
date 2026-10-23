#!/bin/bash
#SBATCH --job-name=D13_run
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s,drjieliu-h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D13_run2_%j.out
# D13 (reader-query read distilled from the encoder-side read). Job ids in slurm/logs/D13.jobids.
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
# Usage: CMDS0='a;;b' CMDS1='c;;d' sbatch slurm/D13_run2.sbatch — two ';;'-separated chains, one per GPU, in parallel.
run_chain() { local g=$1; shift; IFS=$'\n'; for c in $(echo "$1" | sed 's/;;/\n/g'); do echo ">>> [gpu$g] $c"; CUDA_VISIBLE_DEVICES=$g bash -c "$c" >> slurm/logs/D13_run2_${SLURM_JOB_ID}_gpu$g.log 2>&1 || { echo "FAILED [gpu$g] $c"; return 1; }; done; echo "CHAIN$g DONE"; }
run_chain 0 "${CMDS0:-}" & P0=$!
run_chain 1 "${CMDS1:-}" & P1=$!
wait $P0; R0=$?; wait $P1; R1=$?
echo D13_DONE $R0 $R1
