#!/bin/bash
#SBATCH --job-name=D12b_sw8k_clip
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12b_sw8k_clip_%j.out
# DS follow-up to D12b_sw8k_x (27975797): its 27B TEXT pass OOMed on CLIPPER (88k-token doc, full-document prefill in
# greedy()). Here the long-doc sets run with
#   --text_max_tokens 50000  text reads the same last ~49k-token window the operator gets (27B --max_tokens 50000 regime)
#   --oom_skip 1             any OOM item is recorded, the process exits 75 and is rerun (resume); recorded items are
#                            dropped from EVERY cond, so text / kv / wrong_doc share one item set
# FIRST ckpt: text,kv,wrong_doc -> outputs/D12b_max8k_c<k>_<set>_t50k. Later ckpts: kv,wrong_doc, only when the main
# job did not already write outputs/D12b_max8k_c<k>_<set>/eval_test.json (else that row is reused).
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
SETS=${SETS:-clipper,loogle_long}
CKS=${CKS:-4000,8000}
L12=3,7,11,15,19,23,27,31,35,39,43,47
C=$XENC_CACHE/D12p; JC=$C/c9b_judge; LC=$C/c9b_lever; FC=$C/c9b_full; RC=$C/c9b_max; MC=$C/c9b_mcq; DC=$C/c9b_ds
OUT=outputs/D12b_max8k_s2
B="--B $XENC_CACHE/D12_maps/encB2_27b_9b_m.pt"
PB="--reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B $B --layers $L12 --enc_device cpu --reader_mem 74GiB --itf_device cuda:0 \
    --enc_cache $JC,$LC,$C/c9b_deep,$FC,$RC,$MC,$DC --max_tokens 50000 --oom_skip 1 --text_max_tokens 50000"
setf() { case $1 in mcq_*) echo outputs/MCQ_sets/$1.json;; *) echo outputs/D22_sets/$1.json;; esac; }
echo "sets $SETS | ckpts $CKS"
first=1
for k in ${CKS//,/ }; do
  CK=$OUT/ckpt_$k.pt; [ -s $CK ] || { echo "!!! no $CK"; continue; }
  echo "=== 27B max8k ckpt $k: $(ls -l $CK)"
  for s in ${SETS//,/ }; do
    F=$(setf $s); SD=$(python -I -c "import json;print(json.load(open('$F'))['split_dir'])"); WM=${F%.json}_wrongmap.json
    if [ $first = 1 ]; then CONDS=text,kv,wrong_doc; OD=outputs/D12b_max8k_c${k}_${s}_t50k
    else CONDS=kv,wrong_doc; OD=outputs/D12b_max8k_c${k}_$s
         [ -s $OD/eval_test.json ] && { echo "=== $OD exists (main job): reused"; continue; }; OD=${OD}_t50k; fi
    c="python scripts/eval_d12p.py $PB --interface $CK --split_dir $SD --split test --n 0 --d22_set $F --wrong_map $WM --conds $CONDS --out $OD"
    echo ">>> [$(date '+%F %T')] $c"
    for try in $(seq 1 60); do ( eval "$c" ); rc=$?; [ $rc -eq 75 ] || break; echo "... exit 75 (OOM resume), rerun $try"; done
    [ $rc -eq 0 ] || echo "!!! failed (rc $rc): $c"
  done
  first=0
done
echo D12B_SW8K_CLIP_DONE
