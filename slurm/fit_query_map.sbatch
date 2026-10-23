#!/bin/bash
#SBATCH --job-name=D1_fitA
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100,drjieliu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D1_fitA_%j.out
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels${PYTHONPATH:+:$PYTHONPATH}
export XENC_CACHE=${XENC_CACHE:-/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache}
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35
set -uo pipefail
cd $REPO
python scripts/fit_query_map.py --reader_short Qwen3.5-9B --encoder_short Qwen3.5-2B --task quality \
  --center outputs/centers/center_Qwen3.5-2B.pt --out outputs/centers/A_9B_to_2B_quality.pt --max_docs 40
python scripts/fit_query_map.py --reader_short Qwen3.5-9B --encoder_short Qwen3.5-9B --task quality \
  --center outputs/centers/center_Qwen3.5-9B.pt --out outputs/centers/A_9B_to_9B_quality.pt --max_docs 40
echo FIT_DONE
