#!/bin/bash
#SBATCH --job-name=D00NT_probe
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=36:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# NOTES probe (agent NOTES): unregistered kv@0 of a 9B base with / without encoder-written NOTE columns
# (scripts/eval_notes.py, xenc/notes.py). Training-free.
#   ITF   base checkpoint (default outputs/DS_AB_full/d13_best.pt)      TAG  output tag -> outputs/NT_<TAG>_<set>
#   SETS0 / SETS1  sets run as two 1-GPU lanes (encoder + reader on the same GPU), in parallel
#   SETS2          long sets run afterwards on both GPUs (encoder cuda:0, reader cuda:1)
#   EXTRA          extra eval_notes args (e.g. --note_types facts --note_lens 256)
# --oom_skip 1 + exit-75 rerun loop as slurm/DS_sweep.sbatch.
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
ITF=${ITF:-outputs/DS_AB_full/d13_best.pt}
TAG=${TAG:-dsab}
SETS0=${SETS0:-mcq_hallu,mcq_bamboo,mcq_leval,quality}
SETS1=${SETS1:-musique,qasper,lastingbench}
SETS2=${SETS2:-loogle}
EXTRA=${EXTRA:-}
[ -s "$ITF" ] || { echo "!!! missing checkpoint $ITF"; exit 1; }
echo "=== base $ITF ($(ls -l $ITF | awk '{print $5, $6, $7, $8}')) | XENC_CACHE $XENC_CACHE"
COMMON="--interface $ITF --encoder Qwen/Qwen3.5-4B --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19 --budget 32 --oom_skip 1 --cost_docs 4 $EXTRA"

set_args() {   # $1 = set name -> ARGS
  case $1 in
    quality)    ARGS="--split_dir data/quality_split --split test --n 300 --seed 0 --max_tokens 16000";;
    longhealth) ARGS="--split_dir data/longhealth_split --split test --fold 0 --n 0 --max_tokens 16000";;
    mcq_*)      F=outputs/MCQ_sets/$1.json
                ARGS="--d22_set $F --wrong_map outputs/MCQ_sets/${1}_wrongmap.json --split test --n 0 --split_dir $(python -I -c "import json;print(json.load(open('$F'))['split_dir'])") --max_tokens $(python -I -c "import json;print(json.load(open('$F'))['max_tokens'])")";;
    *)          F=outputs/D22_sets/$1.json
                ARGS="--d22_set $F --wrong_map outputs/D22_sets/${1}_wrongmap.json --split test --n 0 --split_dir $(python -I -c "import json;print(json.load(open('$F'))['split_dir'])") --max_tokens $(python -I -c "import json;print(json.load(open('$F'))['max_tokens'])")";;
  esac
}

run_sets() {   # $1 = sets, $2 = device args
  local FAILED=0
  for S in $(echo "$1" | tr ',' ' '); do
    set_args $S
    OUTD=outputs/NT_${TAG}_$S
    [ -s "$OUTD/eval_test.json" ] && { echo "=== $OUTD/eval_test.json exists: reused"; continue; }
    c="python scripts/eval_notes.py $COMMON $2 $ARGS --out $OUTD"
    echo ">>> [$(date '+%F %T')] $c"
    for try in $(seq 1 60); do ( eval "$c" ); rc=$?; [ $rc -eq 75 ] || break; echo "... exit 75 (OOM resume), rerun $try"; done
    [ $rc -eq 0 ] || { echo "!!! failed (rc $rc): $c"; FAILED=$((FAILED+1)); }
  done
  echo "failed steps: $FAILED"
}

# smoke (GPU 0): 3 QuALITY items, short notes; kv must reproduce eval_d13's kv generations exactly (note = None path)
if [ "${SMOKE:-1}" = "1" ]; then
  SM=outputs/NT_smoke_${SLURM_JOB_ID}
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_notes.py $COMMON --device cuda --split_dir data/quality_split --split test --n 3 \
      --seed 0 --max_tokens 16000 --note_lens 8,16 --max_new 16 --wrong_note_len 8 --cost_docs 1 --cost_len 16 \
      --notes_dir $XENC_CACHE/notes_smoke --out $SM || { echo "!!! smoke failed"; exit 1; }
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_d13.py --interface $ITF --encoder Qwen/Qwen3.5-4B \
      --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19 --budget 32 --device cuda --split_dir data/quality_split \
      --split test --n 3 --seed 0 --max_tokens 16000 --conds kv,wrong_doc --out ${SM}_d13 || { echo "!!! smoke d13 failed"; exit 1; }
  python -I -c "
import json,sys
a={r['key']:r for r in json.load(open('$SM/eval_test.json'))['items']}; b={r['key']:r for r in json.load(open('${SM}_d13/eval_test.json'))['items']}
assert a.keys()==b.keys(), 'item mismatch'
bad=[(k,c) for k in a for c in ('kv','wrong_doc') if a[k][c]['gen']!=b[k][c]['gen']]
print('smoke equality kv/wrong_doc vs eval_d13:', 'OK' if not bad else bad); sys.exit(1 if bad else 0)" || { echo "!!! smoke equality failed"; exit 1; }
fi

L0=slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_lane0.out
L1=slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_lane1.out
[ -n "$SETS0" ] && ( export CUDA_VISIBLE_DEVICES=0; run_sets "$SETS0" "--device cuda"; echo LANE0_DONE ) > $L0 2>&1 &
[ -n "$SETS1" ] && ( export CUDA_VISIBLE_DEVICES=1; run_sets "$SETS1" "--device cuda"; echo LANE1_DONE ) > $L1 2>&1 &
wait
grep -h "LANE.*_DONE\|failed" $L0 $L1 2>/dev/null
[ -n "$SETS2" ] && run_sets "$SETS2" "--enc_device cuda:0 --device cuda:1"
echo "${SLURM_JOB_NAME}_DONE"
