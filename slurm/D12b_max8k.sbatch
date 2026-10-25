#!/bin/bash
#SBATCH --job-name=D12b_max8k
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12b_max8k_%j.out
# D12b max8k (user 2026-10-23: "直接弄8000步然后存checkpoint来测试…数据要堆上去"): the 27B lv4 base continued 8000 steps on
# the MAXIMAL mix = every in-split train item + DS phase-B (MuSiQue official train 4000, NarrativeQA <= 48k) + agent MCQ's
# comprehension splits (racemc, multirc, nqamc, looglemc) + conflict TRAIN, docs <= 48k tokens (27B memory); every
# MCQ / OOD judge set + D22 set excluded; DS strict judged-vs-training leak check = CPU job (dependency), its summary
# is printed below. Checkpoint every 1000 steps; sweep 2000/4000/6000/8000: unregistered kv@0 + wrong_doc on the 8 sets +
# mcq_leval / mcq_bamboo (text reused from existing rows).
#   A (new operators -> c9b_max, ~0.27 TB) | full (8000 steps) | sweep
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
STAGES=${STAGES:-A,full,sweep}; has() { [[ ",$STAGES," == *",$1,"* ]]; }
L12=3,7,11,15,19,23,27,31,35,39,43,47
C=$XENC_CACHE/D12p; JC=$C/c9b_judge; LC=$C/c9b_lever; FC=$C/c9b_full; RC=$C/c9b_max; MC=$C/c9b_mcq
INIT=${INIT:-outputs/D12b_lv4_s2/d13_best.pt}; OUT=${OUT:-outputs/D12b_max8k_s2}; SWP=${SWP:-D12b_max8k}  # OUT/SWP overridable (continuation runs)
B="--B $XENC_CACHE/D12_maps/encB2_27b_9b_m.pt"
XS="--extra_splits musiquetr=data/musiquetr_split,narrativeqa=data/narrativeqa_split,racemc=data/racemc_split,multirc=data/multirc_split,nqamc=data/nqamc_split,looglemc=data/looglemc_split"
PB="--reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B $B --layers $L12 --enc_device cpu --reader_mem 74GiB --itf_device cuda:0 \
    --enc_cache $JC,$LC,$C/c9b_deep,$FC,$RC,$MC --max_tokens 50000"
JS=$(ls outputs/MCQ_sets/mcq_*.json outputs/OOD_sets/ood_*.json | grep -v _wrongmap | paste -sd,)
EX="outputs/D22_sets/qasper.json,outputs/D22_sets/contractnli.json,outputs/D22_sets/shakespeare_cf.json,outputs/D22_sets/loogle.json,outputs/D22_sets/loogle_long.json,outputs/D22_sets/clipper.json,outputs/D22_sets/musique.json,outputs/D22_sets/lastingbench.json,$JS"
FILT="--exclude_sets $EX --exclude_docs outputs/DS_leakage_phaseA.json --doc_len loogle:0:48000,clipper:0:48000,narrativeqa:0:48000,nqamc:0:48000,looglemc:0:48000"
VAL=quality:10,musique:20,lastingbench:20,longhealth:0,d25cqa:10,d25fecf:10,qasper:16,contractnli:16,shakespeare_cf:4,clipper:12
TRAIN=quality:0,musique:0,lastingbench:0,longhealth:0,qasper:0,contractnli:0,shakespeare_cf:0,loogle:0,clipper:0,d25cqa:0,d25fecf:0,musiquetr:0,narrativeqa:0,racemc:0,multirc:0,nqamc:0,looglemc:0
declare -A SPEC=([lh]="--split_dir data/longhealth_split --fold 0 --n 0" [q300]="--split_dir data/quality_split --n 300" \
  [cqacf]="--split_dir data/d25_cqacf_split --n 0" [fecf]="--split_dir data/d25_fecf_split --n 0")
for s in qasper loogle shakespeare_cf lastingbench; do
  SPEC[$s]="--split_dir data/${s}_split --d22_set outputs/D22_sets/$s.json --wrong_map outputs/D22_sets/${s}_wrongmap.json"; done
echo "init $INIT | $(ls -l $INIT)"
if has A; then
  echo ">>> leak check (CPU job, dependency)"; grep -h -E "LEAKCHECK|flagged" outputs/D12b_max8k_leak.log 2>/dev/null | tail -14
  echo ">>> operators for the new train docs"; python scripts/cache_d12p.py --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B \
      --layers $L12 --device cuda:0 --s1_train "" --s1_val "" --s2_train racemc:0,multirc:0,musiquetr:0,narrativeqa:0 --s2_val "" \
      --s2_ops_only 1 --skip_have $JC,$LC,$C/c9b_deep,$FC --teacher_cache "" --evals "" $XS $FILT --max_tokens 50000 --out $RC || exit 1
  du -sh $RC
fi
if has full; then
  echo ">>> max8k (lv4 + maximal mix, 8000 steps)"; python scripts/register_d12p.py $PB $XS --op_alias nqamc=narrativeqa,looglemc=loogle --cached_only 2 $FILT --init $INIT --val $VAL \
      --train $TRAIN \
      --task_p quality:0.09,longhealth:0.03,contractnli:0.05,shakespeare:0.03,clipper:0.03,racemc:0.07,multirc:0.05,nqamc:0.08,looglemc:0.045,qasper:0.06,loogle:0.045,musique:0.02,musiquetr:0.03,lastingbench:0.05,narrativeqa:0.04,d25cqa:0.17,d25fecf:0.11 \
      --delta_rank 128 --reg 1.0 --lr_delta 5e-5 --dist_w 1.0 --dist_max_cols 6000000 --lr_g_deep 1e-2 \
      --select_min_step 1000 --save_items 1 --steps ${STEPS:-8000} --eval_every 1000 --save_every 1000 --seed ${SEED:-0} \
      --teacher_cache $XENC_CACHE/D12_trcache_9b --teacher_only 0 --out $OUT || exit 1
fi
if has sweep; then
  for s in mcq_leval mcq_bamboo; do
    SPEC[$s]="--split_dir $(python -I -c "import json;print(json.load(open('outputs/MCQ_sets/$s.json'))['split_dir'])") --d22_set outputs/MCQ_sets/$s.json --wrong_map outputs/MCQ_sets/${s}_wrongmap.json"; done
  SWEEP=${SWEEP:-2000,4000,6000,8000}; for k in ${SWEEP//,/ }; do
    CK=$OUT/ckpt_$k.pt; [ -s $CK ] || { echo "!!! no $CK"; continue; }
    for s in mcq_leval mcq_bamboo qasper loogle lastingbench shakespeare_cf fecf cqacf q300 lh; do
      echo ">>> max8k@$k $s"; python scripts/eval_d12p.py $PB --interface $CK ${SPEC[$s]} --conds kv,wrong_doc \
          --out outputs/${SWP}_c${k}_$s || echo "!!! max8k@$k $s failed"
    done
  done
fi
echo D12B_MAX8K_DONE
