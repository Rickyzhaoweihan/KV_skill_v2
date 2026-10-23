#!/bin/bash
#SBATCH --job-name=D7_state2
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100,drjieliu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D7_state2_%j.out
# D7 phase 2: hybrid-reader document memory (KV vs DeltaNet state), training-free, 9B->9B.
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
# D7: minimal native operator (layer subsets) + "reader skims the tail" family, LH fold 0 then QuALITY.
R=${READER:-Qwen/Qwen3.5-9B}
for SET in ${SETS:-minimal tail}; do
  python scripts/eval_state.py --reader $R --split_dir data/longhealth_split --split test --fold 0 \
      --conds $SET --out outputs/D7_state_lh_$SET
done
for SET in ${SETS:-minimal tail}; do
  python scripts/eval_state.py --reader $R --split_dir data/quality_split --split test --max_docs 20 --n 200 \
      --conds $SET --out outputs/D7_state_q_$SET
done
echo D7_STATE2_DONE
