#!/bin/bash
#SBATCH --job-name=D12b_lv2
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=30:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12b_lv2_%j.out
# D12b exploration (no dedicated control; compared with the lever-full judge rows outputs/D12p_lvfull_*, H100):
# continue the 27B lever-FULL base 1500 steps with a re-weighted mix:
#   conflict share back to .30 (d25cqa .18, d25fecf .12; lever-full .20) -> FaithEval-CF regression (back to the prior)
#   multi-passage share up (musique .10, lastingbench .10; lever-full .06/.06) -> 27B LastingBench/MuSiQue non-read
# same cached subset, val items, recipe as lever-full; selection = best val at step >= 500 (no bar); then the 8-set judge.
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
STAGES=${STAGES:-full,fjdg}; has() { [[ ",$STAGES," == *",$1,"* ]]; }
L12=3,7,11,15,19,23,27,31,35,39,43,47
C=$XENC_CACHE/D12p; JC=$C/c9b_judge; LC=$C/c9b_lever
INIT=outputs/D12p_lever_full_s2/d13_best.pt; OUT=outputs/D12b_lv2_s2
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
echo "init $INIT | $(ls -l $INIT)"
if has full; then
  echo ">>> lever v2 continuation"; python scripts/register_d12p.py $PB --cached_only 1 $FILT --init $INIT --val $VAL \
      --train quality:150,musique:150,lastingbench:150,longhealth:0,$NEW,d25cqa:200,d25fecf:150 \
      --task_p quality:0.12,longhealth:0.06,musique:0.10,lastingbench:0.10,qasper:0.08,contractnli:0.06,shakespeare:0.05,loogle:0.08,clipper:0.05,d25cqa:0.18,d25fecf:0.12 \
      --delta_rank 128 --reg 1.0 --lr_delta 5e-5 --dist_w 1.0 --dist_max_cols 6000000 --lr_g_deep 1e-2 \
      --select_min_step 500 --save_items 1 --steps ${STEPS:-1500} --eval_every 500 --seed ${SEED:-0} \
      --teacher_cache $XENC_CACHE/D12_trcache_9b --teacher_only 0 --out $OUT || exit 1
fi
if has fjdg; then
  for s in fecf cqacf lastingbench shakespeare_cf qasper loogle q300 lh; do
    echo ">>> lv2 $s"; python scripts/eval_d12p.py $PB --interface $OUT/d13_best.pt ${SPEC[$s]} --conds kv,wrong_doc \
        --out outputs/D12b_lv2_$s || echo "!!! lv2 $s failed"
  done
fi
echo D12B_LV2_DONE
