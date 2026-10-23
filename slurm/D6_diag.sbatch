#!/bin/bash
#SBATCH --job-name=D6_diag
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100,drjieliu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=3:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D6_diag_%j.out
# D6 diagnostics. STAGE=content : probe 1 (diag_content, 2B + 9B encoders) + probe 3 (diag_linear, 2B + 9B)
#                 STAGE=oracle  : probe 2 (diag_oracle_read on the p1 LongHealth 9B L/L centered interface,
#                                 + forced-gold registration of $REG_STEPS steps)
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels${PYTHONPATH:+:$PYTHONPATH}
export XENC_CACHE=${XENC_CACHE:-/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache}
export TMPDIR=/scratch/drjieliu_root/drjieliu/rickyhan/hf_caches/tmp
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35
set -uo pipefail
cd $REPO
STAGE=${STAGE:?set STAGE=content|oracle}; FOLD=${FOLD:-0}; REG_STEPS=${REG_STEPS:-100}
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ "$STAGE" = content ]; then
  for E in 2B 9B; do
    python scripts/diag_content.py --encoder Qwen/Qwen3.5-$E --center outputs/centers/center_Qwen3.5-$E.pt \
      --split_dir data/longhealth_split --fold $FOLD --cache_dir "$XENC_CACHE" --out outputs/D6_content/content_Qwen3.5-$E.json
  done
  for E in 2B 9B; do
    python scripts/diag_linear.py --encoder Qwen/Qwen3.5-$E --center outputs/centers/center_Qwen3.5-$E.pt \
      --split_dir data/longhealth_split --fold $FOLD --cache_dir "$XENC_CACHE" --out outputs/D6_content/linear_Qwen3.5-$E.json
  done
elif [ "$STAGE" = oracle ]; then
  python scripts/diag_oracle_read.py --reader Qwen/Qwen3.5-9B --split_dir data/longhealth_split --fold $FOLD \
    --interface outputs/p1_longhealth_9b_LL_f0_center/interface_best.pt --cache_dir "$XENC_CACHE" \
    --out_dir outputs/D6_oracle --register_steps $REG_STEPS "$@"
fi
