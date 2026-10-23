#!/bin/bash
# L1: registered judge of an L1 read-form checkpoint on one D22 long set, INSIDE a 2x L40S job (slurm/L1_run.sbatch).
#   bash scripts/l1_judge.sh <set> <run_dir|base> [extra register_d22 args...]
#     set     loogle | loogle_long | clipper
#     run_dir outputs/L1F_<arm>: judges <run_dir>/d13_best.pt (d13_last.pt if no val point beat step 0) -> <run_dir>_<set>
#     base    lever-1 FULL (outputs/D13_lever1_full/d13_best.pt), kv@0 only (budget 0, no registration) on THIS GPU type
#             -> outputs/L1F_base0_<set> (the GPU-matched kv@0 reference; the D13L1f judge rows ran on H100)
# Recipe = the lever-1 full judge rows (outputs/D13L1f_loogle*/args_register_eval_s0.json, H2_L1f_flat_clipper):
# parts dA,gate r16, shuffle 2, warmup 25, holdout .2, lr 1e-3, budgets 0/25/50/100/200, max_items 48,
# max_tokens 128000, D22 self-questions; plus --oom_exit 1 with F1's restart loop (exit 75 -> resume in a fresh process)
# and --dist_chunk 64 (same loss, lower peak). The main row is kv@0 (the trained base read); kv es<=200 is reported too.
set -u
S=$1; RUN=$2; shift 2
case $S in
  loogle|loogle_long) SD=data/loogle_split; SQ=outputs/D22_selfq/loogle.jsonl;;
  clipper)            SD=data/clipper_split; SQ=outputs/D22_selfq/clipper.jsonl;;
  *) echo "bad set $S"; exit 2;;
esac
if [ "$RUN" = base ]; then
  ITF=outputs/D13_lever1_full/d13_best.pt; OUTD=outputs/L1F_base0_$S; BUD=0; CONDS=closed_tail,kv,wrong_doc
else
  ITF=$RUN/d13_best.pt; [ -e "$ITF" ] || ITF=$RUN/d13_last.pt
  OUTD=${RUN}_$S; BUD=0,25,50,100,200; CONDS=closed_tail,kv,wrong_doc
fi
[ -e "$ITF" ] || { echo "[l1_judge] missing interface $ITF"; exit 2; }
echo "[l1_judge] set $S interface $ITF -> $OUTD (budgets $BUD)"
for i in $(seq 1 20); do
  echo "[l1_judge] $S attempt $i"
  python scripts/register_d22.py --d22_set outputs/D22_sets/$S.json --wrong_map outputs/D22_sets/${S}_wrongmap.json \
    --interface $ITF --encoder Qwen/Qwen3.5-4B --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19 \
    --enc_device cuda:0 --device cuda:1 --mem_safe 1 --shuffle 2 --warmup 25 --holdout 0.2 --parts dA,gate --lr 1e-3 \
    --max_items 48 --budgets $BUD --conds $CONDS --split_dir $SD --split test --max_tokens 128000 --selfq $SQ \
    --oom_exit 1 --oom_doc_max 8 --dist_chunk 64 --seed 0 --out $OUTD "$@"
  rc=$?; [ $rc -ne 75 ] && break
done
echo "[l1_judge] $S exit $rc"
exit $rc
