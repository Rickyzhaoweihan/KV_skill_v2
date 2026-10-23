#!/bin/bash
#SBATCH --job-name=D12p_lvfull
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=40:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12p_lvfull_%j.out
# 27B lever-1 + deep. Stages (STAGES env, default all):
#   jA     cache the Shakespeare-CF + LastingBench D22 judge docs (c9b_judge)
#   jdg    SCF + LB test judge of the lever PILOT vs deep_v2 (same job, paired) -- where the pilot's val gain came from
#   A      extend the lever subset cache (c9b_lever): Qasper 250 -> 450 docs, LooGLE first 120 -> all <=48k (~+70 GB)
#   full   full lever run: init deep_v2, 3000 steps, val every 500 (same 273 val items), bar selection at 500,
#          --dist_max_cols 6e6 (long items train on KL+CE only: the pilot's 45 OOM skips were the distillation logits)
#   fjdg   judge of the full-run checkpoint on all 8 sets (LH f0, Q300, CQA-CF, FE-CF, Qasper, LooGLE, SCF, LB)
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
STAGES=${STAGES:-jA,jdg,A,full,fjdg}; has() { [[ ",$STAGES," == *",$1,"* ]]; }
L12=3,7,11,15,19,23,27,31,35,39,43,47
C=$XENC_CACHE/D12p; JC=$C/c9b_judge; LC=$C/c9b_lever
PILOT=outputs/D12p_lever_pilot_lever/d13_best.pt; DEEP=outputs/D12p_9b_deep_v2_s2/d13_best.pt; FULL=outputs/D12p_lever_full_s2
B="--B $XENC_CACHE/D12_maps/encB2_27b_9b_m.pt"
PB="--reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B $B --layers $L12 --enc_device cpu --reader_mem 74GiB --itf_device cuda:0 \
    --enc_cache $JC,$LC,$C/c9b_deep --max_tokens 50000"
EX="outputs/D22_sets/qasper.json,outputs/D22_sets/contractnli.json,outputs/D22_sets/shakespeare_cf.json,outputs/D22_sets/loogle.json,outputs/D22_sets/loogle_long.json,outputs/D22_sets/clipper.json,outputs/D22_sets/musique.json,outputs/D22_sets/lastingbench.json"
FILT="--exclude_sets $EX --exclude_docs outputs/D13_lever1_leakage.json --doc_len loogle:0:48000,clipper:0:48000"
VAL=quality:10,musique:20,lastingbench:20,longhealth:0,d25cqa:10,d25fecf:10,qasper:16,contractnli:16,shakespeare_cf:4,clipper:12
NEW=qasper:450,contractnli:200,shakespeare_cf:0,loogle:0,clipper:0
declare -A SPEC=([lh]="--split_dir data/longhealth_split --fold 0 --n 0" [q300]="--split_dir data/quality_split --n 300" \
  [cqacf]="--split_dir data/d25_cqacf_split --n 0" [fecf]="--split_dir data/d25_fecf_split --n 0")
for s in qasper loogle shakespeare_cf lastingbench; do
  SPEC[$s]="--split_dir data/${s}_split --d22_set outputs/D22_sets/$s.json --wrong_map outputs/D22_sets/${s}_wrongmap.json"; done
if has jA; then
  echo ">>> cache SCF + LB judge docs"; python scripts/cache_d12p.py --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B \
      --layers $L12 --device cuda:0 --s1_train "" --s1_val "" --s2_train "" --s2_val "" --evals "" \
      --evals_d22 outputs/D22_sets/shakespeare_cf.json,outputs/D22_sets/lastingbench.json --max_tokens 50000 --out $JC || exit 1
fi
if has jdg; then
  for s in shakespeare_cf lastingbench; do
    echo ">>> pilot $s"; python scripts/eval_d12p.py $PB --interface $PILOT ${SPEC[$s]} --conds closed_tail,text,kv,wrong_doc \
        --out outputs/D12p_lvp_lever_$s || echo "!!! pilot $s failed"
    echo ">>> deep_v2 $s"; python scripts/eval_d12p.py $PB --interface $DEEP ${SPEC[$s]} --conds kv,wrong_doc \
        --out outputs/D12p_lvp_deepv2_$s || echo "!!! deep_v2 $s failed"
  done
fi
if has A; then
  echo ">>> extend lever cache"; python scripts/cache_d12p.py --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B --layers $L12 \
      --device cuda:0 --s1_train "" --s1_val "" --s2_train $NEW --s2_q_per_doc 4 --s2_val $VAL --teacher_cache "" --evals "" \
      $FILT --max_tokens 50000 --out $LC || exit 1
  du -sh $LC
fi
if has full; then
  echo ">>> full lever run"; python scripts/register_d12p.py $PB --cached_only 1 $FILT --init $DEEP --val $VAL \
      --train quality:150,musique:150,lastingbench:150,longhealth:0,$NEW,d25cqa:200,d25fecf:150 \
      --task_p quality:0.16,longhealth:0.08,musique:0.06,lastingbench:0.06,qasper:0.10,contractnli:0.08,shakespeare:0.06,loogle:0.10,clipper:0.10,d25cqa:0.12,d25fecf:0.08 \
      --delta_rank 128 --reg 1.0 --lr_delta 5e-5 --dist_w 1.0 --dist_max_cols 6000000 --lr_g_deep 1e-2 \
      --select_min_step 500 --bar_select 1 --save_items 1 --steps ${STEPS:-3000} --eval_every 500 --seed ${SEED:-0} \
      --teacher_cache $XENC_CACHE/D12_trcache_9b --teacher_only 0 --out $FULL || exit 1
fi
if has fjdg; then
  for s in lh q300 cqacf fecf qasper loogle shakespeare_cf lastingbench; do
    echo ">>> full $s"; python scripts/eval_d12p.py $PB --interface $FULL/d13_best.pt ${SPEC[$s]} --conds kv,wrong_doc \
        --out outputs/D12p_lvfull_$s || echo "!!! full $s failed"
  done
fi
echo D12p_LVFULL_DONE
