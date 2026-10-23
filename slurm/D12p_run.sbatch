#!/bin/bash
#SBATCH --job-name=D12p
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100,drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12p_%j.out
# D12p: paper-form read into the 27B reader, TWO-PHASE (the encoder and the 27B are never on the GPUs together).
#   phase A  scripts/cache_d12p.py   encoder only (cuda:0): operators + encoder teacher records -> $CACHE
#   phase B  fit_d12p (stage 1, MLP-A distillation) -> register_d12p (stage 2, paper KL+CE+distill) -> eval_d12p
#            27B only, sharded over both GPUs; interface on the last GPU; encoder weights on CPU (projections only).
# env: ENC (Qwen/Qwen3.5-9B), TAG (9b), STEPS (3000), STAGES ("A,s1,s2,judge"). Job ids -> slurm/logs/D12.jobids.
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
set -euo pipefail
cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
ENC=${ENC:-Qwen/Qwen3.5-9B}; TAG=${TAG:-9b}; STEPS=${STEPS:-3000}; STAGES=${STAGES:-A,s1,s2,judge}
B=${B:-$XENC_CACHE/D12_maps/encB2_27b_${TAG}_m.pt}
LAYERS=3,7,11,15,19,23,27,31
CACHE=$XENC_CACHE/D12p/c$TAG
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
NG=$(nvidia-smi -L | wc -l)
if [ "$NG" -eq 1 ]; then RMEM=74GiB; ITF=cuda:0                    # one H100: 27B (~52 GiB) + interface on the same GPU
elif [ "$MEM" -gt 70000 ]; then RMEM=44GiB,40GiB; ITF=cuda:1; else RMEM=36GiB,26GiB; ITF=cuda:1; fi
echo "GPUs $NG x $MEM MiB -> reader budgets $RMEM, interface $ITF; encoder $ENC; B $B; cache $CACHE"
PB="--reader Qwen/Qwen3.5-27B --encoder $ENC --B $B --layers $LAYERS --enc_device cpu --reader_mem $RMEM --itf_device $ITF --enc_cache $CACHE"
has() { [[ ",$STAGES," == *",$1,"* ]]; }
# overridable (defaults = the first runs): S2_TRAIN (stage-2 docs), TEACHER_ONLY (1: only items with the cached 27B
# text teacher; 0: other items get the 27B teacher online), S2_INIT (default stage-1 MLP-A), RUN (output suffix), EVERY
S2_TRAIN=${S2_TRAIN:-quality:120,musique:60,lastingbench:60,longhealth:0}; TEACHER_ONLY=${TEACHER_ONLY:-1}
RUN=${RUN:-}; EVERY=${EVERY:-500}; TC=$XENC_CACHE/D12_trcache_9b
S1_TRAIN=${S1_TRAIN:-quality:120,musique:60,lastingbench:60,longhealth:0}
S1=outputs/D12p_${TAG}${S1RUN:-}_s1; S2=outputs/D12p_${TAG}_s2${RUN}; S2_INIT=${S2_INIT:-$S1/d13_mlp.pt}
if has A; then
  ATC=$TC; [ "$TEACHER_ONLY" = 1 ] || ATC=""
  echo ">>> phase A"; python scripts/cache_d12p.py --reader Qwen/Qwen3.5-27B --encoder $ENC --layers $LAYERS --device cuda:0 \
      --s1_train $S1_TRAIN --s2_train $S2_TRAIN --teacher_cache "$ATC" --out $CACHE
fi
if has s1; then
  echo ">>> stage 1"; python scripts/fit_d12p.py $PB --train $S1_TRAIN \
      --variants mlp --epochs 3 --out $S1
fi
if has s2; then
  echo ">>> stage 2"; python scripts/register_d12p.py $PB --init $S2_INIT --train $S2_TRAIN \
      --teacher_cache $TC --teacher_only $TEACHER_ONLY --dist_w 1.0 --steps $STEPS --eval_every $EVERY --seed ${SEED:-0} --out $S2
fi
if has judge; then
  echo ">>> judge"
  python scripts/eval_d12p.py $PB --interface $S2/d13_best.pt --split_dir data/longhealth_split --fold 0 --n 0 \
      --conds closed_tail,text,kv,wrong_doc --out outputs/D12p_${TAG}${RUN}_lh
  python scripts/eval_d12p.py $PB --interface $S2/d13_best.pt --split_dir data/quality_split --n 300 \
      --conds closed_tail,text,kv,wrong_doc --out outputs/D12p_${TAG}${RUN}_q300
  python scripts/eval_d12p.py $PB --interface $S2/d13_best.pt --split_dir data/quality_split --n 200 \
      --conds closed_tail,kv,wrong_doc --out outputs/D12p_${TAG}${RUN}_q200
fi
echo D12p_DONE
