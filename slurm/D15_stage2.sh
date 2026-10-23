#!/bin/bash
# D15 stage 2 driver (runs INSIDE a 2-GPU D15_run job). INIT = D15 stage-1 file (largest-pool MLP-A).
#   GPU0  big : register_d15 with ALL labelled split-train docs (QuALITY 150, MuSiQue 967, LastingBench 600, LH folds 2-4)
#               + label-free pool items (pool_p, reader-with-text pseudo-labels, conf >= 0.8), STEPS steps
#   GPU1  ctrl: register_d15 with D13's exact stage-2 data (pool_p 0, defaults), 2500 steps -> isolates the stage-1 data
# then judge both best checkpoints on LongHealth fold 0 (n=80) and QuALITY test (n=300).
cd /nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
INIT=${INIT:?set INIT}
STEPS=${STEPS:-5000}
POOL_P=${POOL_P:-0.3}
CUDA_VISIBLE_DEVICES=0 python scripts/register_d15.py --smoke --init $INIT --pool_p 0.5 --pool_docs 20 \
    --out outputs/D15_smoke_reg > slurm/logs/D15_smoke_reg.log 2>&1 || { tail -30 slurm/logs/D15_smoke_reg.log; exit 1; }
grep -E "pool items|pool pseudo|done" slurm/logs/D15_smoke_reg.log
CUDA_VISIBLE_DEVICES=0 python scripts/register_d15.py --init $INIT --train quality:0,musique:0,lastingbench:0,longhealth:0 \
    --pool_p $POOL_P --pool_docs 3000 --steps $STEPS --eval_every 500 --out outputs/D15_reg_big > slurm/logs/D15_reg_big.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/register_d15.py --init $INIT --steps 2500 --out outputs/D15_reg_ctrl \
    > slurm/logs/D15_reg_ctrl.log 2>&1 &
wait
grep -hE "val @|pool pseudo|done" slurm/logs/D15_reg_big.log slurm/logs/D15_reg_ctrl.log | cut -c1-250
for R in big ctrl; do
  F=outputs/D15_reg_$R/d13_best.pt
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_d13.py --interface $F --split_dir data/longhealth_split --fold 0 --n 0 --out outputs/D15_ev_s2$R\_lh > slurm/logs/D15_ev_s2$R\_lh.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python scripts/eval_d13.py --interface $F --split_dir data/quality_split --n 300 --out outputs/D15_ev_s2$R\_q > slurm/logs/D15_ev_s2$R\_q.log 2>&1 &
  wait
done
python scripts/d15_aggregate.py outputs/D15_ev_s2* outputs/D13_s2w10_lh outputs/D13_s2w10_q
