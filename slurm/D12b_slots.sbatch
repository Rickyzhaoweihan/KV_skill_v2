#!/bin/bash
#SBATCH --job-name=D12b_slots
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12b_slots_%j.out
# D12b H-slots (xenc/d12b_slots.py): one forward of SlotH(9B operator) writes 64 per-document memory slots per read layer;
# the 27B read attends to them (separate softmax branch, Wv zero-init => step 0 == base). Base = lv3 (fuller-mix
# continuation of lever-full; beat lever-full on the judge, §5.12), FROZEN. lv3 mix + objective, 3000 steps, val every 500 with own-vs-swap + slot attention share, selection on the
# PER-TASK-BALANCED val (lever-full's 273 items + LooGLE val 8 docs + NarrativeQA last-8 train docs, 4 q/doc).
# ARM (env): s1 = plain; s2 = + anchored own-vs-swap contrast (w 1, p .5, margin .1, eps .05).
# Stages (STAGES env): A (s1 only: operators of the extra val docs -> c9b_val2, then a DONE marker), full, fjdg.
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home HF_HUB_CACHE=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels${PYTHONPATH:+:$PYTHONPATH}
export XENC_CACHE=${XENC_CACHE:-/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache}
export TMPDIR=/scratch/drjieliu_root/drjieliu/rickyhan/hf_caches/tmp
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35
set -uo pipefail
cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
ARM=${ARM:-s1}
STAGES=${STAGES:-A,full,fjdg}; has() { [[ ",$STAGES," == *",$1,"* ]]; }
L12=3,7,11,15,19,23,27,31,35,39,43,47
C=$XENC_CACHE/D12p; JC=$C/c9b_judge; LC=$C/c9b_lever; VC=$C/c9b_val2; FC=$C/c9b_full
INIT=outputs/D12b_lv3_s2/d13_best.pt; OUT=outputs/D12b_slots_$ARM
B="--B $XENC_CACHE/D12_maps/encB2_27b_9b_m.pt"
XS="--extra_splits musiquetr=data/musiquetr_split,narrativeqa=data/narrativeqa_split"
VX="--val_extra loogle:val:8,narrativeqa:train:-8 --val_extra_q 4"
PB="--reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B $B --layers $L12 --enc_device cpu --reader_mem 74GiB --itf_device cuda:0 \
    --enc_cache $JC,$LC,$C/c9b_deep,$FC,$VC --max_tokens 50000"
EX="outputs/D22_sets/qasper.json,outputs/D22_sets/contractnli.json,outputs/D22_sets/shakespeare_cf.json,outputs/D22_sets/loogle.json,outputs/D22_sets/loogle_long.json,outputs/D22_sets/clipper.json,outputs/D22_sets/musique.json,outputs/D22_sets/lastingbench.json"
FILT="--exclude_sets $EX --exclude_docs outputs/DS_leakage_phaseA.json --doc_len loogle:0:48000,clipper:0:48000,narrativeqa:0:48000"
VAL=quality:10,musique:20,lastingbench:20,longhealth:0,d25cqa:10,d25fecf:10,qasper:16,contractnli:16,shakespeare_cf:4,clipper:12
TRAIN=quality:0,musique:0,lastingbench:0,longhealth:0,qasper:0,contractnli:0,shakespeare_cf:0,loogle:0,clipper:0,d25cqa:200,d25fecf:150,musiquetr:1500,narrativeqa:160
declare -A SPEC=([lh]="--split_dir data/longhealth_split --fold 0 --n 0" [q300]="--split_dir data/quality_split --n 300" \
  [cqacf]="--split_dir data/d25_cqacf_split --n 0" [fecf]="--split_dir data/d25_fecf_split --n 0")
for s in qasper loogle shakespeare_cf lastingbench; do
  SPEC[$s]="--split_dir data/${s}_split --d22_set outputs/D22_sets/$s.json --wrong_map outputs/D22_sets/${s}_wrongmap.json"; done
CON=""; [ "$ARM" = s2 ] && CON="--h_swap_w 1.0 --h_swap_p 0.5 --h_swap_margin 0.1 --h_anchor_eps 0.05"
echo "ARM $ARM | init $INIT | $(ls -l $INIT)"
if has A; then
  if [ "$ARM" = s1 ]; then
    echo ">>> operators of the extra val docs"; python scripts/cache_d12p.py --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B \
        --layers $L12 --device cuda:0 --s1_train "" --s1_val "" --s2_train "" --s2_val "" --teacher_cache "" --evals "" \
        $VX $XS $FILT --skip_have $JC,$LC,$C/c9b_deep,$FC --max_tokens 50000 --out $VC || exit 1
    touch $VC/DONE
  else
    for i in $(seq 1 240); do [ -e $VC/DONE ] && break; sleep 60; done
    [ -e $VC/DONE ] || { echo "!!! c9b_val2 not ready"; exit 1; }
  fi
fi
if has full; then
  echo ">>> H-slots $ARM"; python scripts/register_d12p.py $PB $XS $VX --cached_only 2 $FILT --init $INIT --val $VAL \
      --train $TRAIN \
      --task_p quality:0.08,longhealth:0.05,musique:0.04,musiquetr:0.07,lastingbench:0.07,qasper:0.09,contractnli:0.07,shakespeare:0.04,loogle:0.09,narrativeqa:0.08,clipper:0.04,d25cqa:0.17,d25fecf:0.11 \
      --hyper 1 --h_kind slots --h_slots 64 --h_blocks 2 --h_lr ${HLR:-2e-4} --h_warmup 100 $CON \
      --delta_rank 0 --train_g 0 --train_B 0 --train_mlp 0 --dist_w 1.0 --dist_max_cols 6000000 \
      --select_min_step 500 --select_balanced 1 --save_items 1 --steps ${STEPS:-3000} --eval_every 500 --seed ${SEED:-0} \
      --teacher_cache $XENC_CACHE/D12_trcache_9b --teacher_only 0 --out $OUT || exit 1
fi
if has full && [ -s $OUT/h_best.pt ]; then
  echo ">>> H-slots cost"; python scripts/d12b_hcost.py --h $OUT/h_best.pt --ops $JC/op --n 20 --out $OUT/hcost.json || true
fi
if has fjdg; then
  for s in qasper loogle shakespeare_cf lastingbench fecf cqacf q300 lh; do
    echo ">>> slots $ARM $s"; python scripts/eval_d12p.py $PB --interface $INIT --hyper_ckpt $OUT/h_best.pt ${SPEC[$s]} \
        --conds kv,wrong_doc,kv_base,kv_Hswap --out outputs/D12b_slots_${ARM}_$s || echo "!!! slots $s failed"
  done
fi
echo D12B_SLOTS_DONE
