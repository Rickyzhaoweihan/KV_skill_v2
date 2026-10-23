#!/bin/bash
#SBATCH --job-name=CAP_run
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=220G
#SBATCH --time=5-00:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# CAP (interface capacity on the lever-1 data): optional training, then the registered judge (D22 rule, es<=200) on
# up to 7 sets. Job ids in slurm/logs/CAP.jobids. Env:
#   TRAIN    ';;'-separated training commands (2 GPUs: --enc_device cuda:0 --device cuda:1); stops the job on failure
#   CK       checkpoint to judge; default $CAP_OUT/d13_best.pt, else $CAP_OUT/d13_last.pt (full length always judged)
#   JT       judge output prefix, e.g. outputs/CAPj_big   -> outputs/CAPj_big_<set>
#   ITF      interface flags for the judge (default: lever-1 5-layer B / layers)
#   CONDS    judge conditions (default closed_tail,kv,wrong_doc)
#   REGX     extra registration flags (e.g. "--rank 64"); BUD budgets (default 0,25,50,100,200)
#   SETS0 / SETS1   sets judged in parallel, one GPU each (lane 0 / lane 1); SETSP: sets judged after, on 2 GPUs;
#   then SETS0B / SETS1B / SETSPB the same way (quality_s0 / quality_s1 / quality_eval = QuALITY sharded on 2 GPUs)
#   sets: qasper musique lastingbench contractnli quality loogle loogle_long; "set@X" judges with the overrides in
#   CFG_X (e.g. CFG_r64='JT=outputs/CAPj_L1f_r64 REGX="--rank 64" BUD=0,25,50,100,200,400 CONDS=closed_tail,kv,wrong_doc')
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels:$REPO/scripts${PYTHONPATH:+:$PYTHONPATH}
export XENC_CACHE=${XENC_CACHE:-/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache}
export TMPDIR=/scratch/drjieliu_root/drjieliu/rickyhan/hf_caches/tmp
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35
set -uo pipefail
cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
TRAIN=${TRAIN:-}; CAP_OUT=${CAP_OUT:-}; JT=${JT:?JT}; CK=${CK:-}
ITF=${ITF:-"--B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19"}
CONDS=${CONDS:-closed_tail,kv,wrong_doc}; REGX=${REGX:-}; BUD=${BUD:-0,25,50,100,200}
SETS0=${SETS0:-}; SETS1=${SETS1:-}; SETSP=${SETSP:-}

IFS=$'\n'; for c in $(echo "$TRAIN" | sed 's/;;/\n/g'); do echo ">>> [$(date +%H:%M)] $c"; eval "$c" || { echo "TRAIN FAILED"; exit 1; }; done; unset IFS
if [ -z "$CK" ]; then CK=$CAP_OUT/d13_best.pt; [ -f "$CK" ] || CK=$CAP_OUT/d13_last.pt; fi
[ -f "$CK" ] || { echo "no checkpoint $CK"; exit 1; }
echo "judging $CK -> ${JT}_<set> | ITF $ITF | conds $CONDS | budgets $BUD | regx $REGX"

judge() {   # judge <set[@CFG]> <device flags>  -> prints the command(s); CFG_<name> = assignments (JT=.. REGX=.. ..)
  local s=${1%%@*} dev=$2
  if [ "$1" != "$s" ]; then local v="CFG_${1#*@}"; local JT=$JT CONDS=$CONDS REGX=$REGX BUD=$BUD; eval "${!v}"; fi
  local com="--interface $CK --encoder Qwen/Qwen3.5-4B $ITF --mem_safe 1 --shuffle 2 --warmup 25 --holdout 0.2 --parts dA,gate --lr 1e-3 --budgets $BUD $REGX $dev"
  case $s in
    qasper|musique) echo "python scripts/register_d22.py --d22_set outputs/D22_sets/$s.json --wrong_map outputs/D22_sets/${s}_wrongmap.json $com --conds $CONDS --split_dir data/${s}_split --split test --max_tokens 16000 --selfq outputs/D22_selfq/$s.jsonl --out ${JT}_$s";;
    lastingbench) echo "python scripts/register_d22.py --d22_set outputs/D22_sets/$s.json --wrong_map outputs/D22_sets/${s}_wrongmap.json $com --conds $CONDS --split_dir data/${s}_split --split test --max_tokens 18000 --selfq outputs/D22_selfq/$s.jsonl --out ${JT}_$s";;
    contractnli) echo "python scripts/register_d22.py --d22_set outputs/D22_sets/$s.json --wrong_map outputs/D22_sets/${s}_wrongmap.json $com --conds $CONDS --split_dir data/${s}_split --split test --max_tokens 16000 --selfq outputs/D20_selfq/$s.jsonl --oom_skip 1 --out ${JT}_$s";;
    loogle|loogle_long) echo "python scripts/register_d22.py --d22_set outputs/D22_sets/$s.json --wrong_map outputs/D22_sets/${s}_wrongmap.json $com --conds $CONDS --split_dir data/loogle_split --split test --max_tokens 128000 --selfq outputs/D22_selfq/loogle.jsonl --oom_skip 1 --out ${JT}_$s";;
    quality) echo "python scripts/register_d20.py $com --split_dir data/quality_split --split test --n 300 --reg_n 0 --phase register --selfq outputs/D20_selfq/quality.jsonl --out ${JT}_quality"
             echo "python scripts/register_d20.py $com --split_dir data/quality_split --split test --n 0 --phase eval --conds $CONDS --selfq outputs/D20_selfq/quality.jsonl --eval_tag full --out ${JT}_quality";;
    quality_s0|quality_s1) echo "python scripts/register_d20.py $com --split_dir data/quality_split --split test --n 300 --reg_n 0 --phase register --shard ${s#quality_s} --nshard 2 --selfq outputs/D20_selfq/quality.jsonl --out ${JT}_quality";;
    quality_eval) echo "python scripts/register_d20.py $com --split_dir data/quality_split --split test --n 0 --phase eval --conds $CONDS --selfq outputs/D20_selfq/quality.jsonl --eval_tag full --out ${JT}_quality";;
    *) echo "echo unknown set $s; false";;
  esac
}
export CK
run_lane() {  # run_lane <gpu> <sets...>
  local g=$1; shift
  local log=slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_lane$g${PH:-}.out
  ( export CUDA_VISIBLE_DEVICES=$g
    for s in "$@"; do judge $s "" | while IFS= read -r c; do echo ">>> [$(date +%H:%M)] $c"; bash -c "$c" || echo "!!! lane $g step failed: $c"; done; done
    echo "LANE${g}_DONE" ) > $log 2>&1 &
}
[ -n "$SETS0" ] && run_lane 0 $SETS0
[ -n "$SETS1" ] && run_lane 1 $SETS1
wait
for s in $SETSP; do judge $s "--enc_device cuda:0 --device cuda:1" | while IFS= read -r c; do echo ">>> [$(date +%H:%M)] $c"; bash -c "$c" || echo "!!! post step failed: $c"; done; done
# phase B (e.g. QuALITY registration sharded over both GPUs: SETS0B=quality_s0 SETS1B=quality_s1 SETSPB=quality_eval)
SETS0B=${SETS0B:-}; SETS1B=${SETS1B:-}; SETSPB=${SETSPB:-}; PH=B
[ -n "$SETS0B" ] && run_lane 0 $SETS0B
[ -n "$SETS1B" ] && run_lane 1 $SETS1B
wait
for s in $SETSPB; do judge $s "--enc_device cuda:0 --device cuda:1" | while IFS= read -r c; do echo ">>> [$(date +%H:%M)] $c"; bash -c "$c" || echo "!!! post step failed: $c"; done; done
grep -h "LANE.*_DONE\|step failed" slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_lane*.out 2>/dev/null
echo CAP_DONE
