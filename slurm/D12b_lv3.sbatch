#!/bin/bash
#SBATCH --job-name=D12b_lv3
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12b_lv3_%j.out
# D12b exploration (coordinator 2026-10-22): the 27B lever-full base on a FULLER data mix (no control; vs the lever-full
# judge rows outputs/D12p_lvfull_*, H100). Data = ALL in-split train items (lever-full used <= 4 items/doc and subsets)
# under DS's strict leakage file + DS phase-B data (MuSiQue official train 1500 docs, NarrativeQA <= 48k tokens),
# docs <= 48k tokens (27B memory; operators front-truncated at 49k). ~23k items vs lever-full 8081.
#   A     operators for the new train docs only -> $XENC_CACHE/D12p/c9b_full (~0.3 TB; no new encoder records)
#   full  init lever-full, 3000 steps, val every 500 (the same 273 items), best val at step >= 500;
#         --cached_only 2: items with an old encoder record keep the distillation term, new items train on KL + CE
#         (27B text teacher online); conflict share .28 (d25cqa .17, d25fecf .11) for the FaithEval-CF regression
#   fjdg  8-set judge -> outputs/D12b_lv3_<set>
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
STAGES=${STAGES:-A,full,fjdg}; has() { [[ ",$STAGES," == *",$1,"* ]]; }
L12=3,7,11,15,19,23,27,31,35,39,43,47
C=$XENC_CACHE/D12p; JC=$C/c9b_judge; LC=$C/c9b_lever; FC=$C/c9b_full
INIT=outputs/D12p_lever_full_s2/d13_best.pt; OUT=outputs/D12b_lv3_s2
B="--B $XENC_CACHE/D12_maps/encB2_27b_9b_m.pt"
XS="--extra_splits musiquetr=data/musiquetr_split,narrativeqa=data/narrativeqa_split"
PB="--reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B $B --layers $L12 --enc_device cpu --reader_mem 74GiB --itf_device cuda:0 \
    --enc_cache $JC,$LC,$C/c9b_deep,$FC --max_tokens 50000"
EX="outputs/D22_sets/qasper.json,outputs/D22_sets/contractnli.json,outputs/D22_sets/shakespeare_cf.json,outputs/D22_sets/loogle.json,outputs/D22_sets/loogle_long.json,outputs/D22_sets/clipper.json,outputs/D22_sets/musique.json,outputs/D22_sets/lastingbench.json"
FILT="--exclude_sets $EX --exclude_docs outputs/DS_leakage_phaseA.json --doc_len loogle:0:48000,clipper:0:48000,narrativeqa:0:48000"
VAL=quality:10,musique:20,lastingbench:20,longhealth:0,d25cqa:10,d25fecf:10,qasper:16,contractnli:16,shakespeare_cf:4,clipper:12
TRAIN=quality:0,musique:0,lastingbench:0,longhealth:0,qasper:0,contractnli:0,shakespeare_cf:0,loogle:0,clipper:0,d25cqa:200,d25fecf:150,musiquetr:1500,narrativeqa:160
declare -A SPEC=([lh]="--split_dir data/longhealth_split --fold 0 --n 0" [q300]="--split_dir data/quality_split --n 300" \
  [cqacf]="--split_dir data/d25_cqacf_split --n 0" [fecf]="--split_dir data/d25_fecf_split --n 0")
for s in qasper loogle shakespeare_cf lastingbench; do
  SPEC[$s]="--split_dir data/${s}_split --d22_set outputs/D22_sets/$s.json --wrong_map outputs/D22_sets/${s}_wrongmap.json"; done
echo "init $INIT | $(ls -l $INIT)"
if has A; then
  echo ">>> operators for the new train docs"; python scripts/cache_d12p.py --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B \
      --layers $L12 --device cuda:0 --s1_train "" --s1_val "" --s2_train $TRAIN --s2_val "" --s2_ops_only 1 \
      --skip_have $JC,$LC,$C/c9b_deep --teacher_cache "" --evals "" $XS $FILT --max_tokens 50000 --out $FC || exit 1
  du -sh $FC
fi
if has full; then
  echo ">>> lever v3 (fuller mix)"; python scripts/register_d12p.py $PB $XS --cached_only 2 $FILT --init $INIT --val $VAL \
      --train $TRAIN \
      --task_p quality:0.08,longhealth:0.05,musique:0.04,musiquetr:0.07,lastingbench:0.07,qasper:0.09,contractnli:0.07,shakespeare:0.04,loogle:0.09,narrativeqa:0.08,clipper:0.04,d25cqa:0.17,d25fecf:0.11 \
      --delta_rank 128 --reg 1.0 --lr_delta 5e-5 --dist_w 1.0 --dist_max_cols 6000000 --lr_g_deep 1e-2 \
      --select_min_step 500 --save_items 1 --steps ${STEPS:-3000} --eval_every 500 --seed ${SEED:-0} \
      --teacher_cache $XENC_CACHE/D12_trcache_9b --teacher_only 0 --out $OUT || exit 1
fi
if has fjdg; then
  for s in fecf cqacf qasper loogle shakespeare_cf lastingbench q300 lh; do
    echo ">>> lv3 $s"; python scripts/eval_d12p.py $PB --interface $OUT/d13_best.pt ${SPEC[$s]} --conds kv,wrong_doc \
        --out outputs/D12b_lv3_$s || echo "!!! lv3 $s failed"
  done
fi
echo D12B_LV3_DONE
