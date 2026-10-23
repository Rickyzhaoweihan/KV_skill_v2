#!/bin/bash
#SBATCH --job-name=D12h_reg
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=36:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12h_reg_%j.out
# D12h step 2: HEADROOM of per-document registration on the 27B reader, 4B-encoder base (outputs/D12p_4b_s2/d13_best.pt,
# reader layers 3..31). D20 recipe (dA+gate r16, lr 1e-3, shuffle 2, warmup 25, holdout .2), live d12p runtime
# (27B + 4B encoder both on the one H100). Arms: reader-free (self-questions by the 4B, --teacher none) on LH f0 and
# QuALITY 300; text-teacher (self-questions + targets by the 27B) on LH f0. kv@0 = base; report kv@200.
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home HF_HUB_CACHE=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels:$REPO/scripts${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=/scratch/drjieliu_root/drjieliu/rickyhan/hf_caches/tmp
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35
set -uo pipefail
cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
ARMS=${ARMS:-rf_lh,rf_q300,tt_lh}
has() { [[ ",$ARMS," == *",$1,"* ]]; }
RT="--runtime d12p --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-4B --B outputs/D10_maps/encB8_27b_4b_post.pt \
    --layers 3,7,11,15,19,23,27,31 --interface outputs/D12p_4b_s2/d13_best.pt --reader_mem 72GiB --enc_device cuda:0 \
    --itf_device cuda:0 --mem_safe 1"
REC="--parts dA,gate --rank 16 --lr 1e-3 --shuffle 2 --warmup 25 --holdout 0.2 --budgets 0,50,100,200 --split test"
SQ=outputs/D20_selfq
for t in longhealth quality; do
  if [ ! -s $SQ/${t}_by4b.jsonl ] && { has rf_lh || has rf_q300; }; then
    echo ">>> selfq $t by 4B"; python scripts/d20_selfq.py --split_dir data/${t}_split --docs test:0 --reader Qwen/Qwen3.5-4B \
        --out $SQ/${t}_by4b.jsonl || exit 1
  fi
done
if has rf_lh; then
  echo ">>> reader-free LH f0"; python scripts/register_d20.py $RT $REC --split_dir data/longhealth_split --fold 0 --n 0 \
      --selfq $SQ/longhealth_by4b.jsonl --teacher none --out outputs/D12h_reg_rf_lh || echo "!!! rf_lh failed"
fi
if has rf_q300; then
  echo ">>> reader-free QuALITY 300"; python scripts/register_d20.py $RT $REC --split_dir data/quality_split --n 300 \
      --selfq $SQ/quality_by4b.jsonl --teacher none --out outputs/D12h_reg_rf_q300 || echo "!!! rf_q300 failed"
fi
if has tt_lh; then
  [ -s $SQ/longhealth_by27b.jsonl ] || { echo ">>> selfq LH by 27B"; python scripts/d20_selfq.py --split_dir data/longhealth_split \
      --docs test:0 --reader Qwen/Qwen3.5-27B --out $SQ/longhealth_by27b.jsonl; }
  echo ">>> text-teacher LH f0"; python scripts/register_d20.py $RT $REC --split_dir data/longhealth_split --fold 0 --n 0 \
      --selfq $SQ/longhealth_by27b.jsonl --teacher text --out outputs/D12h_reg_tt_lh || echo "!!! tt_lh failed"
fi
echo D12h_REG_DONE
