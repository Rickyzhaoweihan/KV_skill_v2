#!/bin/bash
# OOD (agent OOD, 2026-10-22): run ONE step of the OOD judge for one set, INSIDE a job (lanes.sbatch / F1_run.sbatch).
#   bash scripts/ood_judge.sh <set> selfq [device]          4B-written self-questions for the set's reg docs
#   bash scripts/ood_judge.sh <set> l1f   [extra args...]    lever-1 full base: closed_tail,text,kv,wrong_doc
#   bash scripts/ood_judge.sh <set> old   [extra args...]    old base D13_4b_reg_w10:  kv,wrong_doc
# The judge is D22's exact recipe (register_d22.py: parts dA,gate, shuffle 2, warmup 25, holdout .2, lr 1e-3, early
# stop over budgets 0/50/100/200), reader Qwen3.5-9B, encoder 4B, B encB5_9b_4b_post_xf01, layers 3/7/11/15/19, under
# --oom_exit 1 with F1's restart loop (exit 75 = an OOM was recorded; the same command resumes in a fresh process).
# Sets / caps: outputs/OOD_sets/<set>.json (scripts/ood_build.py).
set -u
S=$1; MODE=$2; shift 2
SET=outputs/OOD_sets/$S.json
SD=$(python -I -c "import json;print(json.load(open('$SET'))['split_dir'])")
MT=$(python -I -c "import json;print(json.load(open('$SET'))['max_tokens'])")
SELFQ=outputs/OOD_selfq/$S.jsonl
case $S in
  ood_cuad|d25_nqswap) TM=span,mix,mcq;;
  ood_infbench)        TM=span,mcq,mix;;
  d25_fecf)            TM=mcq,mix,span;;
  *)                   TM=mcq,mix,nli;;
esac
if [ "$MODE" = selfq ]; then
  DEV=${1:-cuda}
  mkdir -p outputs/OOD_selfq
  exec python scripts/d22_selfq.py --reader Qwen/Qwen3.5-4B --split_dir $SD --docs file:$SET --templates $TM \
      --max_doc_tokens $MT --device $DEV --out $SELFQ
fi
[ -s "$SELFQ" ] || { echo "missing self-questions $SELFQ"; exit 2; }
case $MODE in
  l1f) ITF=outputs/D13_lever1_full/d13_best.pt; CONDS=closed_tail,text,kv,wrong_doc; OUTD=outputs/OOD_L1f_$S;;
  old) ITF=outputs/D13_4b_reg_w10/d13_best.pt;  CONDS=kv,wrong_doc;                  OUTD=outputs/OOD_old_$S;;
  *) echo "bad mode $MODE"; exit 2;;
esac
DC=""; [ "$MT" -gt 40000 ] && DC="--dist_chunk 64"
for i in $(seq 1 15); do
  echo "[ood_judge] $S $MODE attempt $i"
  python scripts/register_d22.py --d22_set $SET --wrong_map outputs/OOD_sets/${S}_wrongmap.json \
    --interface $ITF --encoder Qwen/Qwen3.5-4B --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19 \
    --mem_safe 1 --shuffle 2 --warmup 25 --holdout 0.2 --parts dA,gate --lr 1e-3 --budgets 0,50,100,200 \
    --conds $CONDS --split_dir $SD --split test --max_tokens $MT --selfq $SELFQ \
    --oom_exit 1 --oom_doc_max 8 $DC --seed 0 --out $OUTD "$@"
  rc=$?; [ $rc -ne 75 ] && break
done
echo "[ood_judge] $S $MODE exit $rc"
exit $rc
