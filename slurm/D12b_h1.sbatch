#!/bin/bash
#SBATCH --job-name=D12b_h1
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=40:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12b_h1_%j.out
# D12b task 2: the hypernetwork H (xenc/d24_hyper.py v2, rank 64, bias 16) on the 27B reader. Base = lever-FULL
# (FROZEN, deltas merged). One forward of H(9B operator) -> per-document adapter; the 27B never reads the document.
# Trained on the lever-full mix and cached subset (KL to the 27B text teacher (cached top-k) + 0.5 CE gold + dist),
# 3000 steps, val every 500 with own-vs-swap; then the 8-set judge with kv (H) / kv_base / kv_Hswap / wrong_doc.
# Success: kv_H - kv_base > 2 SE AND kv_H - kv_Hswap > 2 SE.
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
INIT=outputs/D12p_lever_full_s2/d13_best.pt; OUT=outputs/D12b_h1
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
  echo ">>> H training"; python scripts/register_d12p.py $PB --cached_only 1 $FILT --init $INIT --val $VAL \
      --train quality:150,musique:150,lastingbench:150,longhealth:0,$NEW,d25cqa:200,d25fecf:150 \
      --task_p quality:0.16,longhealth:0.08,musique:0.06,lastingbench:0.06,qasper:0.10,contractnli:0.08,shakespeare:0.06,loogle:0.10,clipper:0.10,d25cqa:0.12,d25fecf:0.08 \
      --hyper 1 --h_rank 64 --h_bias_k 16 --h_lr ${HLR:-2e-4} --h_lr_basis ${HLR:-2e-4} --h_warmup 50 --h_stat_docs 64 \
      --delta_rank 0 --train_g 0 --train_B 0 --train_mlp 0 --dist_w 1.0 --dist_max_cols 6000000 \
      --select_min_step 500 --save_items 1 --steps ${STEPS:-3000} --eval_every 500 --seed ${SEED:-0} \
      --teacher_cache $XENC_CACHE/D12_trcache_9b --teacher_only 0 --out $OUT || exit 1
fi
if has fjdg; then
  for s in qasper loogle shakespeare_cf fecf cqacf lastingbench q300 lh; do
    echo ">>> H $s"; python scripts/eval_d12p.py $PB --interface $INIT --hyper_ckpt $OUT/h_best.pt ${SPEC[$s]} \
        --conds kv,wrong_doc,kv_base,kv_Hswap --out outputs/D12b_h1_$s || echo "!!! H $s failed"
  done
fi
echo D12B_H1_DONE
