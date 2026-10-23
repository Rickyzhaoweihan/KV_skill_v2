#!/bin/bash
#SBATCH --job-name=D00DS_sweep
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=36:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# DS checkpoint sweep: UNREGISTERED kv@0 + wrong_doc (eval_d13, no registration) of one or more 9B base checkpoints on
# every 9B judge set + agent MCQ's comprehension sets, encoder on cuda:0 / reader on cuda:1 (2x L40S).
#   ITFS="tag=path[,tag=path...]"  checkpoints (each -> outputs/DSsw_<tag>_<set>)
#   SETS="..."                      default: all sets below
# --oom_skip 1: an OOM item is recorded and the process exits 75; this script reruns it (resume) up to 60 times.
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
hostname
SETS=${SETS:-musique,qasper,lastingbench,quality,longhealth,mcq_leval,mcq_bamboo,mcq_hallu,mcq_musr,contractnli,loogle,loogle_long,shakespeare_cf,clipper}
COMMON="--encoder Qwen/Qwen3.5-4B --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19 --enc_device cuda:0 --device cuda:1 --conds kv,wrong_doc --budget 32 --oom_skip 1"
FAILED=0
for pair in $(echo "$ITFS" | tr ',' ' '); do
  TAG=${pair%%=*}; ITF=${pair#*=}
  [ -s "$ITF" ] || { echo "!!! missing checkpoint $ITF"; FAILED=$((FAILED+1)); continue; }
  echo "=== checkpoint $TAG: $ITF ($(ls -l $ITF | awk '{print $5, $6, $7, $8}'))"
  for S in $(echo "$SETS" | tr ',' ' '); do
    case $S in
      quality)    ARGS="--split_dir data/quality_split --split test --n 0 --max_tokens 16000";;
      longhealth) ARGS="--split_dir data/longhealth_split --split test --fold 0 --n 0 --max_tokens 16000";;
      mcq_*)      F=outputs/MCQ_sets/$S.json
                  ARGS="--d22_set $F --wrong_map outputs/MCQ_sets/${S}_wrongmap.json --split test --n 0 --split_dir $(python -I -c "import json;print(json.load(open('$F'))['split_dir'])") --max_tokens $(python -I -c "import json;print(json.load(open('$F'))['max_tokens'])")";;
      *)          F=outputs/D22_sets/$S.json
                  ARGS="--d22_set $F --wrong_map outputs/D22_sets/${S}_wrongmap.json --split test --n 0 --split_dir $(python -I -c "import json;print(json.load(open('$F'))['split_dir'])") --max_tokens $(python -I -c "import json;print(json.load(open('$F'))['max_tokens'])")";;
    esac
    OUTD=outputs/DSsw_${TAG}_$S
    [ -s "$OUTD/eval_test.json" ] && { echo "=== $OUTD/eval_test.json exists: reused, not recomputed"; continue; }
    c="python scripts/eval_d13.py --interface $ITF $COMMON $ARGS --out $OUTD"
    echo ">>> [$(date '+%F %T')] $c"
    for try in $(seq 1 60); do ( eval "$c" ); rc=$?; [ $rc -eq 75 ] || break; echo "... exit 75 (OOM resume), rerun $try"; done
    [ $rc -eq 0 ] || { echo "!!! failed (rc $rc): $c"; FAILED=$((FAILED+1)); }
  done
done
echo "failed steps: $FAILED"
echo "${SLURM_JOB_NAME}_DONE"
