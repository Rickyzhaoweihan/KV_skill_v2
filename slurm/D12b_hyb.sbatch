#!/bin/bash
#SBATCH --job-name=D12b_hyb_s0
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# CMP hybrid on the 27B reader (agent CMP, 2026-10-25; user "在27B上也实现这个"). The 27B reads the 9B encoder's
# operator at its 12 layers 3..47 (D12p read, unchanged); HYB adds a compact per-token code of the 9B's deep K/V
# (9B layers 23/27/31) that the 27B's DEEPEST full-attention layers 51/55/59/63 read with their OWN attention
# (xenc/cmp_hyb.py; zcache mode: codes precomputed per document, the cached-operator runtime has no live encoder).
#   fit    scripts/cmp_fit_xm.py: 9B (phase 1) then 27B (phase 2) on the same TRAIN-doc tokens, one H100 -> outputs/CMP_fit27
#   zcache scripts/cmp_zcache.py: 9B codes (d = 2048 basis) of every own / wrong document of the judge sets
#   s0     eval_d12p kv@0 + wrong_doc on $CKPT: oracle (27B true deep K/V, g=1), closed-form HYB g=1 at d=2048 and
#          d=512 (first 512 components of the same basis) -> outputs/D12b_hyb_<arm>_<set>; base rows reused from
#          outputs/D12b_max13k_c5000_<set> (H100, same checkpoint, same args).
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home HF_HUB_CACHE=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels:$REPO/scripts${PYTHONPATH:+:$PYTHONPATH}
export XENC_CACHE=${XENC_CACHE:-/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache}
export TMPDIR=/scratch/drjieliu_root/drjieliu/rickyhan/hf_caches/tmp
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35
set -uo pipefail
cd $REPO
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1); echo "GPU: $GPU"; hostname
STAGES=${STAGES:-fit,zcache,s0}; has() { [[ ",$STAGES," == *",$1,"* ]]; }
CKPT=${CKPT:-outputs/D12b_max13k_s2/ckpt_5000.pt}; BASETAG=${BASETAG:-D12b_max13k_c5000}
SETS=${SETS:-qasper,loogle,lastingbench,q300,lh,shakespeare_cf,fecf,cqacf,mcq_leval,mcq_bamboo}
ARMS=${ARMS:-orc,h2048g1,h512g1}
FIT=outputs/CMP_fit27; SPEC="23+27+31+151+155+159+163|L23+L27+L31"
CMP=$FIT/cmp_23-27-31-151-155-159-163__L23-L27-L31_2048.pt; Z=$XENC_CACHE/CMP_z27
L12=3,7,11,15,19,23,27,31,35,39,43,47
C=$XENC_CACHE/D12p; EC=$C/c9b_judge,$C/c9b_lever,$C/c9b_deep,$C/c9b_full,$C/c9b_max,$C/c9b_mcq
PB="--reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B --B $XENC_CACHE/D12_maps/encB2_27b_9b_m.pt --layers $L12 --enc_device cpu --reader_mem 74GiB --itf_device cuda:0 --enc_cache $EC --max_tokens 50000"
declare -A SPECS=([lh]="--split_dir data/longhealth_split --fold 0 --n 0" [q300]="--split_dir data/quality_split --n 300" \
  [cqacf]="--split_dir data/d25_cqacf_split --n 0" [fecf]="--split_dir data/d25_fecf_split --n 0")
for s in qasper loogle shakespeare_cf lastingbench; do
  SPECS[$s]="--split_dir data/${s}_split --d22_set outputs/D22_sets/$s.json --wrong_map outputs/D22_sets/${s}_wrongmap.json"; done
for s in mcq_leval mcq_bamboo; do
  SPECS[$s]="--split_dir $(python -I -c "import json;print(json.load(open('outputs/MCQ_sets/$s.json'))['split_dir'])") --d22_set outputs/MCQ_sets/$s.json --wrong_map outputs/MCQ_sets/${s}_wrongmap.json"; done
FAILED=0
if has fit; then
  c="python scripts/cmp_fit_xm.py --out_dir $FIT --stats_path $XENC_CACHE/CMP_fit27_stats.pt --specs '$SPEC:512,$SPEC:2048' --device cuda:0"
  echo ">>> [$(date '+%F %T')] $c"; eval "$c" || { echo "!!! fit failed"; exit 1; }
fi
if has zcache; then
  c="python scripts/cmp_zcache.py --cmp $CMP --sets $SETS --enc_cache $EC --out $Z --device cuda:0"
  echo ">>> [$(date '+%F %T')] $c"; eval "$c" || { echo "!!! zcache failed"; exit 1; }
  du -sh $Z
fi
if has s0; then
  echo "=== checkpoint $CKPT ($(ls -l $CKPT | awk '{print $5, $6, $7, $8}')); base rows outputs/${BASETAG}_<set>"
  for s in ${SETS//,/ }; do
    [ -s outputs/${BASETAG}_$s/eval_test.json ] || echo "!!! no base row outputs/${BASETAG}_$s"
    for arm in ${ARMS//,/ }; do
      case $arm in
        orc)     X="--hyb_oracle 1 --hyb_g 1";;
        h2048g1) X="--hyb $CMP --hyb_zcache $Z --hyb_g 1";;
        h512g1)  X="--hyb $CMP --hyb_zcache $Z --hyb_zdim 512 --hyb_g 1";;
        *) echo "!!! unknown arm $arm"; continue;;
      esac
      OUTD=outputs/D12b_hyb_${arm}_$s
      [ -s $OUTD/eval_test.json ] && { echo "=== $OUTD exists"; continue; }
      c="python scripts/eval_d12p.py $PB --interface $CKPT ${SPECS[$s]} --conds kv,wrong_doc --oom_skip 1 $X --out $OUTD"
      echo ">>> [$(date '+%F %T')] $c"
      for try in $(seq 1 60); do ( eval "$c" ); rc=$?; [ $rc -eq 75 ] || break; echo "... exit 75 (OOM resume), rerun $try"; done
      [ $rc -eq 0 ] && echo "$GPU" > $OUTD/gpu.txt || { echo "!!! failed (rc $rc): $c"; FAILED=$((FAILED+1)); }
    done
  done
fi
echo "failed steps: $FAILED"
echo "${SLURM_JOB_NAME}_DONE"
