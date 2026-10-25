#!/bin/bash
#SBATCH --job-name=D00DS_cnli
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
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
# D00DS_cnli: same-GPU ContractNLI check, text + kv@0 (+wrong_doc) on the FULL 300 test items for max8k s6000 and s8000.
F=outputs/D22_sets/contractnli.json
ARGS="--d22_set $F --wrong_map outputs/D22_sets/contractnli_wrongmap.json --split test --n 0 --split_dir $(python -I -c "import json;print(json.load(open('$F'))['split_dir'])") --max_tokens $(python -I -c "import json;print(json.load(open('$F'))['max_tokens'])")"
COMMON="--encoder Qwen/Qwen3.5-4B --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19 --enc_device cuda:0 --device cuda:1 --conds text,kv,wrong_doc --budget 32 --oom_skip 1"
FAILED=0
for S in 6000 8000; do
  ITF=outputs/DS_max8k/d13_step$S.pt; OUTD=outputs/DScnli_m8k$S
  [ -s "$ITF" ] || { echo "!!! missing $ITF"; FAILED=$((FAILED+1)); continue; }
  [ -s "$OUTD/eval_test.json" ] && { echo "=== $OUTD exists, reused"; continue; }
  c="python scripts/eval_d13.py --interface $ITF $COMMON $ARGS --out $OUTD"
  echo ">>> [$(date '+%F %T')] $c"
  for try in $(seq 1 60); do ( eval "$c" ); rc=$?; [ $rc -eq 75 ] || break; echo "... exit 75 (OOM resume), rerun $try"; done
  [ $rc -eq 0 ] || { echo "!!! failed (rc $rc)"; FAILED=$((FAILED+1)); }
done
echo "failed steps: $FAILED"
echo "${SLURM_JOB_NAME}_DONE"
