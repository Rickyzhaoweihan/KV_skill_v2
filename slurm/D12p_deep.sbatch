#!/bin/bash
#SBATCH --job-name=D12p_deep
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=30:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12p_deep_%j.out
# D12p DEEP: the 27B paper-form read at reader layers 3..31 (base) + 35,39,43,47 (D25 context band, 9B enc 19/19/23/23).
#   A   cache_d12p.py: 9B encoder only -> operators/teacher records at enc layers 3,7,11,15,19,23 ($CACHE) incl. judge
#       docs (LongHealth f0, QuALITY 300, D25 conflict test sets) and their wrong-doc partners
#   s1  fit_d12p.py --init <best 27B base> : new layers = copy of L31 (gate 0), A re-fit by ridge + label-free attention
#       distillation of the NEW layers only (base layers frozen; gate-aware teacher forcing)
#   s2  register_d12p.py: paper KL(27B text teacher)+0.5 CE + distillation, all 12 layers, rank-128 deltas on A/B +
#       bias + gates (full-matrix Adam over 12 layers does not fit one H100), D13 data, online teacher where uncached
#   judge  base read (step 0, UNREGISTERED): deep vs current best base vs text on LH f0, QuALITY 300, D25 CQA-CF/FE-CF/NQ-Swap
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home HF_HUB_CACHE=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels${PYTHONPATH:+:$PYTHONPATH}
export XENC_CACHE=${XENC_CACHE:-/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache}
export TMPDIR=/scratch/drjieliu_root/drjieliu/rickyhan/hf_caches/tmp
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35
set -euo pipefail
cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
ENC=${ENC:-Qwen/Qwen3.5-9B}; TAG=${TAG:-9b}; STAGES=${STAGES:-A,s1,s2,judge}; STEPS=${STEPS:-5000}
BASE=${BASE:-outputs/D12p_9b_s2_d13r/d13_best.pt}
B=${B:-$XENC_CACHE/D12_maps/encB2_27b_${TAG}_m.pt}
LAYERS=3,7,11,15,19,23,27,31,35,39,43,47; NEW=35,39,43,47; BASE_LAYERS=3,7,11,15,19,23,27,31
CACHE=$XENC_CACHE/D12p/c${TAG}_deep
DATA=quality:150,musique:150,lastingbench:150,longhealth:0
# v2 overrides: S2_DATA / S2_VAL / TASK_P (stage-2 data incl. D25 conflict TRAIN docs), LRGD (deep-gate step size),
# MINSTEP (selection floor), EVERY, RUN (output suffix)
S2_DATA=${S2_DATA:-$DATA}; S2_VAL=${S2_VAL:-quality:10,musique:20,lastingbench:20,longhealth:0}
TASK_P=${TASK_P:-quality:0.4,longhealth:0.2,musique:0.2,lastingbench:0.2}; LRGD=${LRGD:-0}; MINSTEP=${MINSTEP:-0}
EVERY=${EVERY:-250}; RUN=${RUN:-}
S1=outputs/D12p_${TAG}_deep_s1; S2=outputs/D12p_${TAG}_deep${RUN}_s2; J=outputs/D12p_${TAG}_deep${RUN}
PB="--reader Qwen/Qwen3.5-27B --encoder $ENC --B $B --enc_device cpu --reader_mem 74GiB --itf_device cuda:0 --enc_cache $CACHE"
CONFLICT="d25_cqacf d25_fecf d25_nqswap"
has() { [[ ",$STAGES," == *",$1,"* ]]; }
if has A; then
  EV="data/longhealth_split:test:0:0;data/quality_split:test::300"
  for c in $CONFLICT; do EV="$EV;data/${c}_split:test::0"; done
  echo ">>> phase A"; python scripts/cache_d12p.py --reader Qwen/Qwen3.5-27B --encoder $ENC --layers $LAYERS --device cuda:0 \
      --s1_train $DATA --s2_train $S2_DATA --s2_val $S2_VAL --teacher_cache "" --evals "$EV" --out $CACHE
fi
if has s1; then
  echo ">>> stage 1 (new layers)"; python scripts/fit_d12p.py $PB --layers $LAYERS --init $BASE --train_layers $NEW \
      --extend_src 31 --train $DATA --variants mlp --epochs 3 --out $S1
fi
if has s2; then
  echo ">>> stage 2"; python scripts/register_d12p.py $PB --layers $LAYERS --init $S1/d13_mlp.pt --train $S2_DATA --val $S2_VAL --task_p $TASK_P \
      --teacher_cache $XENC_CACHE/D12_trcache_9b --teacher_only 0 --delta_rank ${RANK:-128} --reg 1.0 --lr_delta 5e-5 \
      --dist_w 1.0 --steps $STEPS --eval_every $EVERY --lr_g_deep $LRGD --select_min_step $MINSTEP --seed ${SEED:-0} --out $S2
fi
if has judge; then
  echo ">>> judge (base read, step 0, unregistered)"
  python scripts/eval_d12p.py $PB --layers $LAYERS --interface $S2/d13_best.pt --split_dir data/longhealth_split --fold 0 --n 0 \
      --conds closed_tail,text,kv,wrong_doc --out ${J}_lh
  python scripts/eval_d12p.py $PB --layers $LAYERS --interface $S2/d13_best.pt --split_dir data/quality_split --n 300 \
      --conds closed_tail,text,kv,wrong_doc --out ${J}_q300
  for c in $CONFLICT; do
    python scripts/eval_d12p.py $PB --layers $LAYERS --interface $S2/d13_best.pt --split_dir data/${c}_split --n 0 \
        --conds closed_tail,text,kv,wrong_doc --out ${J}_${c#d25_}
    [ -f outputs/D12p_${TAG}_d13r_${c#d25_}/eval_test.json ] || \
    python scripts/eval_d12p.py $PB --layers $BASE_LAYERS --interface $BASE --split_dir data/${c}_split --n 0 \
        --conds kv,wrong_doc --out outputs/D12p_${TAG}_d13r_${c#d25_}
  done
fi
echo D12p_DEEP_DONE
