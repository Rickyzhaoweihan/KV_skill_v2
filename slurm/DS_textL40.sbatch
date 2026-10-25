#!/bin/bash
#SBATCH --job-name=D00DS_textL40
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
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
# D00DS_textL40: same-GPU (L40S) text-only rows (--conds text, no interface) for the 9B sets whose text row is H100:
# MuSiQue, LooGLE, LongHealth f0 (Qasper / LastingBench / QuALITY already have in-job L40S text in outputs/BL_9b_tb_*).
# Same items / prompts / max_tokens as the DSsw kv sweeps (--n 0, --split test). --oom_skip 1: exit 75 = resume.
SETS=${SETS:-musique,loogle,longhealth}
COMMON="--device cuda:0 --conds text --oom_skip 1"
FAILED=0
for S in $(echo "$SETS" | tr ',' ' '); do
  case $S in
    longhealth) ARGS="--split_dir data/longhealth_split --split test --fold 0 --n 0 --max_tokens 16000";;
    *) F=outputs/D22_sets/$S.json
       ARGS="--d22_set $F --wrong_map outputs/D22_sets/${S}_wrongmap.json --split test --n 0 --split_dir $(python -I -c "import json;print(json.load(open('$F'))['split_dir'])") --max_tokens $(python -I -c "import json;print(json.load(open('$F'))['max_tokens'])")";;
  esac
  OUTD=outputs/DStxtL40_$S
  [ -s "$OUTD/eval_test.json" ] && { echo "=== $OUTD exists, reused"; continue; }
  c="python scripts/eval_d13.py $COMMON $ARGS --out $OUTD"
  echo ">>> [$(date '+%F %T')] $c"
  for try in $(seq 1 60); do ( eval "$c" ); rc=$?; [ $rc -eq 75 ] || break; echo "... exit 75 (OOM resume), rerun $try"; done
  [ $rc -eq 0 ] || { echo "!!! failed (rc $rc)"; FAILED=$((FAILED+1)); }
done
echo "failed steps: $FAILED"
echo "${SLURM_JOB_NAME}_DONE"
