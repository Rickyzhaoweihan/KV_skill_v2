#!/bin/bash
# OOD on the max8k s6000 base (REBASE agent, 2026-10-25): UNREGISTERED eval_d13 of ONE OOD set, conds
# closed_tail,text,kv,wrong_doc, all in one process (same GPU type for every cond). Run inside a job
# (lanes.sbatch / F1_run.sbatch).  bash scripts/ood_m8k_eval.sh <set> "<device args>"
#   <set>  ood_cuad | ood_infbench | ood_lbv2 | d25_nqswap | d25_fecf | d25_cqacf  (outputs/OOD_sets/<set>.json)
#   ITF    checkpoint (default outputs/DS_max8k/d13_step6000.pt)   -> outputs/OODm8k_<set>
# --oom_skip 1: an OOM item is recorded and the process exits 75; this script reruns it (resume), up to 40 times.
set -u
S=$1; DEV=${2:---device cuda:0}
ITF=${ITF:-outputs/DS_max8k/d13_step6000.pt}
F=outputs/OOD_sets/$S.json
SD=$(python -I -c "import json;print(json.load(open('$F'))['split_dir'])")
MT=$(python -I -c "import json;print(json.load(open('$F'))['max_tokens'])")
[ -s "$ITF" ] || { echo "!!! missing checkpoint $ITF"; exit 1; }
OUTD=outputs/OODm8k_$S
[ -s "$OUTD/eval_test.json" ] && { echo "=== $OUTD/eval_test.json exists: reused"; exit 0; }
for i in $(seq 1 40); do
  echo ">>> [$(date '+%F %T')] $S attempt $i  ($ITF, $DEV)"
  python scripts/eval_d13.py --interface $ITF --encoder Qwen/Qwen3.5-4B --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt \
    --layers 3,7,11,15,19 $DEV --d22_set $F --wrong_map outputs/OOD_sets/${S}_wrongmap.json --split test --n 0 \
    --split_dir $SD --max_tokens $MT --budget 32 --conds closed_tail,text,kv,wrong_doc --oom_skip 1 --out $OUTD
  rc=$?; [ $rc -eq 75 ] || break
  echo "... exit 75 (OOM resume)"
done
echo "[ood_m8k_eval] $S exit $rc"
exit $rc
