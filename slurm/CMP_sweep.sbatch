#!/bin/bash
#SBATCH --job-name=D00CMP_sweep
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=36:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# CMP compression design space: unregistered kv@0 (eval_d13, no registration) of ONE base checkpoint read through
# per-token operator bottlenecks (xenc/cmp_op.py), encoder cuda:0 / reader cuda:1 (2x L40S).
#   FIT=1                 stage 0: scripts/cmp_fit.py (stats on TRAINING docs + closed-form compressor files) -> $FITDIR
#   ITF=<d13 file>        base checkpoint (default outputs/DS_AB_full/d13_best.pt)
#   BASE=<tag>            also make the uncompressed rows outputs/DSsw_<tag>_<set> (kv,wrong_doc; reused if present,
#                         also from R2s_DSAB_ / ADDRsw_DSAB_ for ABbest) -- the d = full endpoint
#   CMPS="tag=file[@quant],..."   each -> outputs/CMPsw_<tag>_<set> (conds $CONDS, default kv); special values:
#                         tag=hyb:<file>[@g]   hybrid deep channel (xenc/cmp_hyb.py; compressor or hyb state file), [gates := g]
#                         tag=oracle@<g>       hybrid DIAGNOSTIC: memory = the reader's true deep K/V (reader prefill)
#   FITARGS="..."         extra cmp_fit.py args (e.g. --hyb_reader_layers 23,27,31); SETS=none: no evals
#   SETS="..."            default: the category-balanced CMP set
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
ITF=${ITF:-outputs/DS_AB_full/d13_best.pt}
FIT=${FIT:-0}; FITDIR=${FITDIR:-outputs/CMP_fit}; SPECS=${SPECS:-}
BASE=${BASE:-}; CMPS=${CMPS:-}; CONDS=${CONDS:-kv}
SETS=${SETS:-musique,lastingbench,qasper,loogle,mcq_leval,mcq_bamboo,quality,longhealth,mcq_hallu,shakespeare_cf,contractnli}
COMMON="--encoder Qwen/Qwen3.5-4B --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19 --enc_device cuda:0 --device cuda:1 --budget 32 --oom_skip 1"
FAILED=0
echo "=== base checkpoint $ITF ($(ls -l $ITF | awk '{print $5, $6, $7, $8}'))"
if [ "$FIT" = "1" ]; then
  c="python scripts/cmp_fit.py --out_dir $FITDIR --device cuda:0 ${SPECS:+--specs '$SPECS'} ${FITARGS:-}"
  echo ">>> [$(date '+%F %T')] $c"; eval "$c" || { echo "!!! cmp_fit failed"; FAILED=$((FAILED+1)); }
fi
setargs() {
  case $1 in
    quality)    ARGS="--split_dir data/quality_split --split test --n 0 --max_tokens 16000";;
    longhealth) ARGS="--split_dir data/longhealth_split --split test --fold 0 --n 0 --max_tokens 16000";;
    mcq_*)      F=outputs/MCQ_sets/$1.json
                ARGS="--d22_set $F --wrong_map outputs/MCQ_sets/${1}_wrongmap.json --split test --n 0 --split_dir $(python -I -c "import json;print(json.load(open('$F'))['split_dir'])") --max_tokens $(python -I -c "import json;print(json.load(open('$F'))['max_tokens'])")";;
    *)          F=outputs/D22_sets/$1.json
                ARGS="--d22_set $F --wrong_map outputs/D22_sets/${1}_wrongmap.json --split test --n 0 --split_dir $(python -I -c "import json;print(json.load(open('$F'))['split_dir'])") --max_tokens $(python -I -c "import json;print(json.load(open('$F'))['max_tokens'])")";;
  esac
}
run() {   # $1 out dir, $2 extra args
  local OUTD=$1 X=$2
  [ -s "$OUTD/eval_test.json" ] && { echo "=== $OUTD/eval_test.json exists: reused, not recomputed"; return; }
  c="python scripts/eval_d13.py --interface $ITF $COMMON $ARGS $X --out $OUTD"
  echo ">>> [$(date '+%F %T')] $c"
  for try in $(seq 1 60); do ( eval "$c" ); rc=$?; [ $rc -eq 75 ] || break; echo "... exit 75 (OOM resume), rerun $try"; done
  [ $rc -eq 0 ] || { echo "!!! failed (rc $rc): $c"; FAILED=$((FAILED+1)); }
}
[ "$SETS" = "none" ] && SETS=""
for S in $(echo "$SETS" | tr ',' ' '); do
  setargs $S
  if [ -n "$BASE" ]; then
    hit=""
    for o in outputs/DSsw_${BASE}_$S outputs/R2s_DSAB_$S outputs/ADDRsw_DSAB_$S; do
      [ "$BASE" != "ABbest" ] && [ "$o" != "outputs/DSsw_${BASE}_$S" ] && continue
      [ -s "$o/eval_test.json" ] && { hit=$o; break; }
    done
    if [ -n "$hit" ]; then echo "=== base row $S: $hit (reused)"; else run outputs/DSsw_${BASE}_$S "--conds kv,wrong_doc"; fi
  fi
  for pair in $(echo "$CMPS" | tr ',' ' '); do
    TAG=${pair%%=*}; F=${pair#*=}; Q=""
    case $F in
      oracle@*) run outputs/CMPsw_${TAG}_$S "--conds $CONDS --hyb_oracle 1 --hyb_g ${F#*@}"; continue;;
      hyb:*)    F=${F#hyb:}; G=""
                case $F in *@*) G="--hyb_g ${F#*@}"; F=${F%@*};; esac
                [ -s "$F" ] || { echo "!!! missing hyb file $F"; FAILED=$((FAILED+1)); continue; }
                run outputs/CMPsw_${TAG}_$S "--conds $CONDS --hyb $F $G"; continue;;
      *@*)      Q="--cmp_quant ${F#*@}"; F=${F%@*};;
    esac
    [ -s "$F" ] || { echo "!!! missing compressor $F"; FAILED=$((FAILED+1)); continue; }
    run outputs/CMPsw_${TAG}_$S "--conds $CONDS --cmp $F $Q"
  done
done
echo "failed steps: $FAILED"
echo "${SLURM_JOB_NAME}_DONE"
