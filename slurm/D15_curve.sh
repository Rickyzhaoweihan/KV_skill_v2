#!/bin/bash
# D15 learning curve driver (runs INSIDE a 2-GPU D15_run job): for each N, fit lin (GPU0) and mlp (GPU1) in parallel,
# then evaluate ridge/lin/mlp on LongHealth fold 0 (n=80) and QuALITY test (n=300) with closed_tail.
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
EXTRA=${EXTRA:-}
# text condition once per task (same items / seed as every eval below); kv - text is joined by key in d15_aggregate.py
if [ ! -e outputs/D15_ev_text_q/eval_test.json ]; then
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_d13.py --interface outputs/D13_maps/d13_lin.pt --split_dir data/longhealth_split --fold 0 --n 0 --conds text,closed_book --out outputs/D15_ev_text_lh > slurm/logs/D15_ev_text_lh.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python scripts/eval_d13.py --interface outputs/D13_maps/d13_lin.pt --split_dir data/quality_split --n 300 --conds text,closed_book --out outputs/D15_ev_text_q > slurm/logs/D15_ev_text_q.log 2>&1 &
  wait
  grep -hE "====|^.{0,30}  (text|closed_book) " slurm/logs/D15_ev_text_lh.log slurm/logs/D15_ev_text_q.log
fi
for N in "$@"; do
  O=outputs/D15_maps_n$N
  CUDA_VISIBLE_DEVICES=0 python scripts/fit_d15.py --n_docs $N --variants lin --out $O $EXTRA > slurm/logs/D15_fit_n${N}_lin.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python scripts/fit_d15.py --n_docs $N --variants mlp --out ${O}_mlp $EXTRA > slurm/logs/D15_fit_n${N}_mlp.log 2>&1 &
  wait
  grep -E "D15 stage 1|ridge A|\[ridge\]|best|early stop|done" slurm/logs/D15_fit_n${N}_lin.log slurm/logs/D15_fit_n${N}_mlp.log | cut -c1-330
  for V in lin mlp; do
    F=$O/d15_$V.pt; [ $V = mlp ] && F=${O}_mlp/d15_mlp.pt
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_d13.py --interface $F --split_dir data/longhealth_split --fold 0 --n 0 --out outputs/D15_ev_n${N}_${V}_lh > slurm/logs/D15_ev_n${N}_${V}_lh.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python scripts/eval_d13.py --interface $F --split_dir data/quality_split --n 300 --out outputs/D15_ev_n${N}_${V}_q > slurm/logs/D15_ev_n${N}_${V}_q.log 2>&1 &
    wait
    grep -hE "====|kv -|^.{0,30}  (kv|closed_tail|wrong_doc) " slurm/logs/D15_ev_n${N}_${V}_lh.log slurm/logs/D15_ev_n${N}_${V}_q.log
  done
done
