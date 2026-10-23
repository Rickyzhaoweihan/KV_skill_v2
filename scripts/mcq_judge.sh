#!/bin/bash
# MCQ (agent MCQ, 2026-10-22): ONE step of the comprehension-MCQ judge for one set, INSIDE a job (lanes.sbatch).
# Same recipe as scripts/ood_judge.sh (D22: register_d22.py, parts dA,gate, shuffle 2, warmup 25, holdout .2, lr 1e-3,
# per-doc early stop over budgets 0/50/100/200; reader Qwen3.5-9B, encoder 4B, B encB5_9b_4b_post_xf01, layers
# 3/7/11/15/19; --oom_exit 1 restart loop). kv@0 = budget 0 (UNREGISTERED read, the main row); kv_reg = es<=200.
#   bash scripts/mcq_judge.sh <set> selfq [device]              4B-written self-questions for the set's reg docs
#   bash scripts/mcq_judge.sh <set> l1f   [extra args...]        lever-1 full: closed_book,closed_tail,text,kv,wrong_doc
#   bash scripts/mcq_judge.sh <set> ckpt <d13_best.pt> <tag> [extra args...]   a trained base: kv,wrong_doc
#                                                                 -> outputs/MCQ_<tag>_<set>
# Sets: outputs/MCQ_sets/<set>.json (scripts/mcq_build.py). The reader never sees the doc in kv / wrong_doc rows.
set -u
S=$1; MODE=$2; shift 2
SET=outputs/MCQ_sets/$S.json
SD=$(python -I -c "import json;print(json.load(open('$SET'))['split_dir'])")
MT=$(python -I -c "import json;print(json.load(open('$SET'))['max_tokens'])")
SELFQ=outputs/MCQ_selfq/$S.jsonl
if [ "$MODE" = selfq ]; then
  DEV=${1:-cuda}
  mkdir -p outputs/MCQ_selfq
  exec python scripts/d22_selfq.py --reader Qwen/Qwen3.5-4B --split_dir $SD --docs file:$SET --templates mcq,mix,nli \
      --max_doc_tokens $MT --device $DEV --out $SELFQ
fi
[ -s "$SELFQ" ] || { echo "missing self-questions $SELFQ"; exit 2; }
case $MODE in
  l1f)  ITF=outputs/D13_lever1_full/d13_best.pt; CONDS=closed_book,closed_tail,text,kv,wrong_doc; OUTD=outputs/MCQ_L1f_$S;;
  ckpt) ITF=$1; TAG=$2; shift 2; CONDS=kv,wrong_doc; OUTD=outputs/MCQ_${TAG}_$S;;
  *) echo "bad mode $MODE"; exit 2;;
esac
[ -s "$ITF" ] || { echo "missing interface $ITF"; exit 2; }
echo "[mcq_judge] set $S mode $MODE interface $ITF ($(ls -l $ITF | awk '{print $5, $6, $7, $8}')) -> $OUTD"
DC=""; [ "$MT" -gt 40000 ] && DC="--dist_chunk 64"
for i in $(seq 1 15); do
  echo "[mcq_judge] $S $MODE attempt $i"
  python scripts/register_d22.py --d22_set $SET --wrong_map outputs/MCQ_sets/${S}_wrongmap.json \
    --interface $ITF --encoder Qwen/Qwen3.5-4B --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19 \
    --mem_safe 1 --shuffle 2 --warmup 25 --holdout 0.2 --parts dA,gate --lr 1e-3 --budgets 0,50,100,200 \
    --conds $CONDS --split_dir $SD --split test --max_tokens $MT --selfq $SELFQ \
    --oom_exit 1 --oom_doc_max 8 $DC --seed 0 --out $OUTD "$@"
  rc=$?; [ $rc -ne 75 ] && break
done
echo "[mcq_judge] $S $MODE exit $rc"
exit $rc
