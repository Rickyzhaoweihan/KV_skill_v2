#!/bin/bash
#SBATCH --job-name=D00CAD
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# Agent CAD (2026-10-23): context-aware decoding of the UNREGISTERED kv read (scripts/eval_cad.py), inference only.
#   ARCH=9b  : reader Qwen3.5-9B, encoder 4B live on cuda:0, reader on cuda:1 (2x L40S)        -> outputs/CAD_<TAG>_<set>
#   ARCH=27b : reader Qwen3.5-27B on one H100, cached 9B operators (--enc_cache); missing VAL operators are first
#              built by scripts/cache_d12p.py (with the MuSiQue judge operators) (encoder on the GPU, separate process) into $XENC_CACHE/D12p/c9b_cad
#   ITF=<d13 checkpoint>  TAG=<tag>  SETS=<comma list of the set names below>  [EXTRA="<more eval_cad args>"]
#   GPU1=1 (ARCH=9b): encoder AND reader on cuda:0 (--gres=gpu:1, -p drjieliu-l40s,drjieliu-h100); first a 5-item LongHealth f0
#          check that the 1-GPU kv decode equals existing same-GPU-type rows (L40S: DSsw_ABbest_longhealth, abort on mismatch;
#          H100: D13L1f_lh_f0 kv@0, reported). The GPU name is stored in every eval_cad json ('gpu').
# Val sets (alpha selection, no test item; FaithEval val deliberately NOT used: it is HM's selection set val_fecf):
#   v_cqacf  ConflictQA-CF val (210)        v_cqaor  ConflictQA ORIGINAL bundles val (210, do-no-harm)
#   v_nqswap NQ-Swap val (80)               v_quality QuALITY val, first 20 docs (the 9B's own ckpt-selection docs), n 150
#   v_qasper Qasper val, first 16 docs (span do-no-harm)
# Test sets: fecf cqacf nqswap scf quality mcq_leval mcq_hallu qasper musique
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
ARCH=${ARCH:-9b}; EXTRA=${EXTRA:-}
[ -s "$ITF" ] || { echo "!!! missing checkpoint $ITF"; exit 1; }
echo "=== CAD $ARCH tag $TAG: $ITF ($(ls -l $ITF | awk '{print $5, $6, $7, $8}')) sets $SETS"
C27=$XENC_CACHE/D12p
if [ "$ARCH" = 27b ]; then
  L27=3,7,11,15,19,23,27,31,35,39,43,47
  COMMON="--arch d12p --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B --B $XENC_CACHE/D12_maps/encB2_27b_9b_m.pt --layers $L27 --enc_device cpu --reader_mem 74GiB --itf_device cuda:0 --max_tokens 50000 --enc_cache $C27/c9b_judge,$C27/c9b_lever,$C27/c9b_deep,$C27/c9b_full,$C27/c9b_mcq,$C27/c9b_cad"
  # val operators + MuSiQue judge operators (absent from the existing 27B caches); skip_have keeps it incremental
  c="python scripts/cache_d12p.py --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B --layers $L27 --s1_train= --s1_val= --s2_train= --s2_val= --max_tokens 50000 --device cuda --evals 'data/d25_cqacf_split:val::0;data/d25_cqa_split:val::0;data/d25_nqswap_split:val::0;data/quality_split:val::0;data/qasper_split:val::0' --evals_d22 outputs/D22_sets/musique.json --skip_have $C27/c9b_judge,$C27/c9b_lever,$C27/c9b_deep,$C27/c9b_full,$C27/c9b_mcq --out $C27/c9b_cad"
  echo ">>> [$(date '+%F %T')] $c"; eval "$c" || { echo "!!! operator cache failed"; exit 1; }
  QN=300
else
  DEV="--enc_device cuda:0 --device cuda:1"; [ "${GPU1:-0}" = 1 ] && DEV="--enc_device cuda:0 --device cuda:0"
  COMMON="--arch d13 --encoder Qwen/Qwen3.5-4B --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19 $DEV"
  QN=0
  if [ "${GPU1:-0}" = 1 ]; then
    # same-GPU-type reference: L40S -> DS_AB best rows (eval_d13, 2x L40S; mismatch = abort);
    # H100 -> lever-1 full kv@0 rows (register_d20, H100; reported, not fatal: different code path)
    GN=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    case "$GN" in *H100*) CI=outputs/D13_lever1_full/d13_best.pt; CR=outputs/D13L1f_lh_f0/eval_test.json; CC="kv@0"; FATAL=0;;
                  *)      CI=outputs/DS_AB_full/d13_best.pt;     CR=outputs/DSsw_ABbest_longhealth/eval_test.json; CC=kv; FATAL=1;; esac
    CK=outputs/CAD_1gpu_check_${SLURM_JOB_ID:-x}
    c="python scripts/eval_d13.py --interface $CI --encoder Qwen/Qwen3.5-4B --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19 $DEV --split_dir data/longhealth_split --split test --fold 0 --n 5 --conds kv --budget 32 --max_tokens 16000 --out $CK"
    echo ">>> [$(date '+%F %T')] 1-GPU check on $GN vs $CR: $c"; eval "$c" || { echo "!!! 1-GPU check failed to run"; exit 1; }
    python -I -c "
import json,sys
a={r['key']:r['kv']['gen'] for r in json.load(open('$CK/eval_test.json'))['items']}
b={r['key']:r['$CC']['gen'] for r in json.load(open('$CR'))['items']}
m=sum(a[k]==b[k] for k in a); print(f'1-GPU check ($GN): {m}/{len(a)} kv decodes identical to $CR')
sys.exit(0 if a and m==len(a) else 1)" || { [ $FATAL = 1 ] && { echo "!!! 1-GPU decode differs from the 2-GPU L40S rows: abort"; exit 1; }; echo "(H100 check vs a register_d20 row: mismatch reported, not fatal)"; }
  fi

fi
mt() { [ "$ARCH" = 27b ] && echo "" || echo "--max_tokens $(python -I -c "import json;print(json.load(open('$1'))['max_tokens'])")"; }
FAILED=0
for S in $(echo "$SETS" | tr ',' ' '); do
  case $S in
    v_cqacf)   ARGS="--split_dir data/d25_cqacf_split --split val --n 0";;
    v_cqaor)   ARGS="--split_dir data/d25_cqa_split --split val --variant orig --n 0";;
    v_nqswap)  ARGS="--split_dir data/d25_nqswap_split --split val --n 0";;
    v_quality) ARGS="--split_dir data/quality_split --split val --doc_first 20 --n 150";;
    v_qasper)  ARGS="--split_dir data/qasper_split --split val --doc_first 16 --n 0";;
    fecf)      ARGS="--split_dir data/d25_fecf_split --split test --n 0";;
    cqacf)     ARGS="--split_dir data/d25_cqacf_split --split test --n 0";;
    nqswap)    ARGS="--split_dir data/d25_nqswap_split --split test --n 0";;
    quality)   ARGS="--split_dir data/quality_split --split test --n $QN";;
    mcq_*)     F=outputs/MCQ_sets/$S.json
               ARGS="--d22_set $F --wrong_map outputs/MCQ_sets/${S}_wrongmap.json --split test --n 0 --split_dir $(python -I -c "import json;print(json.load(open('$F'))['split_dir'])") $(mt $F)";;
    scf|qasper|musique) N=$S; [ $S = scf ] && N=shakespeare_cf; F=outputs/D22_sets/$N.json
               ARGS="--d22_set $F --wrong_map outputs/D22_sets/${N}_wrongmap.json --split test --n 0 --split_dir $(python -I -c "import json;print(json.load(open('$F'))['split_dir'])") $(mt $F)";;
    *) echo "!!! unknown set $S"; FAILED=$((FAILED+1)); continue;;
  esac
  OUTD=outputs/CAD_${TAG}_$S
  SPL=$(echo "$ARGS" | sed -n 's/.*--split \([a-z]*\).*/\1/p')
  [ -s "$OUTD/eval_$SPL.json" ] && { echo "=== $OUTD/eval_$SPL.json exists: reused"; continue; }
  c="python scripts/eval_cad.py --interface $ITF $COMMON $ARGS --oom_skip 1 $EXTRA --out $OUTD"
  echo ">>> [$(date '+%F %T')] $c"
  for try in $(seq 1 60); do ( eval "$c" ); rc=$?; [ $rc -eq 75 ] || break; echo "... exit 75 (OOM resume), rerun $try"; done
  [ $rc -eq 0 ] || { echo "!!! failed (rc $rc): $c"; FAILED=$((FAILED+1)); }
done
echo "failed steps: $FAILED"
echo "${SLURM_JOB_NAME}_DONE"
