#!/bin/bash
#SBATCH --job-name=ADDR_run
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=220G
#SBATCH --time=4-00:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# ADDR (agent ADDR: the reader picks the wrong span): optional diagnostics / training on 2 GPUs, then the registered judge
# (D22 rule: dA+gate, lr 1e-3, budgets 0..200, holdout .2 -> es<=200; kv@0 = budget 0 = the main unregistered read),
# then optional post commands. Job ids in slurm/logs/ADDR.jobids. Env:
#   PRE     ';;'-separated commands on 2 GPUs (--enc_device cuda:0 --device cuda:1); the job stops on the first failure
#   OUT     training output dir (CK defaults to $OUT/d13_best.pt, else $OUT/d13_last.pt)
#   CK      checkpoint to judge;  JT  judge output prefix (outputs/ADDRj_<arm> -> outputs/ADDRj_<arm>_<set>)
#   ITF     interface flags (default: lever-1 5-layer B / layers);  CONDS (default kv,wrong_doc);  REGX;  BUD
#   SETS0 / SETS1   sets judged in parallel, one GPU each;  SETSP   sets judged afterwards on 2 GPUs
#   sets: qasper musique lastingbench loogle loogle_long scf
#   POST    ';;'-separated commands on 2 GPUs after the judge (failures are logged, the job continues)
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
PRE=${PRE:-}; OUT=${OUT:-}; JT=${JT:-}; CK=${CK:-}; POST=${POST:-}
ITF=${ITF:-"--B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19"}
CONDS=${CONDS:-kv,wrong_doc}; REGX=${REGX:-}; BUD=${BUD:-0,25,50,100,200}
SETS0=${SETS0:-}; SETS1=${SETS1:-}; SETSP=${SETSP:-}

IFS=$'\n'; for c in $(echo "$PRE" | sed 's/;;/\n/g'); do echo ">>> [$(date +%H:%M)] $c"; eval "$c" || { echo "PRE FAILED: $c"; exit 1; }; done; unset IFS
if [ -n "$SETS0$SETS1$SETSP" ]; then
  if [ -z "$CK" ]; then CK=$OUT/d13_best.pt; [ -f "$CK" ] || CK=$OUT/d13_last.pt; fi
  [ -f "$CK" ] || { echo "no checkpoint $CK"; exit 1; }
  echo "judging $CK -> ${JT}_<set> | ITF $ITF | conds $CONDS | budgets $BUD | regx $REGX"
fi

judge() {   # judge <set> <device flags>  -> prints the command
  local s=$1 dev=$2
  local com="--interface $CK --encoder Qwen/Qwen3.5-4B $ITF --mem_safe 1 --shuffle 2 --warmup 25 --holdout 0.2 --parts dA,gate --lr 1e-3 --budgets $BUD $REGX $dev --conds $CONDS"
  case $s in
    qasper|musique) echo "python scripts/register_d22.py --d22_set outputs/D22_sets/$s.json --wrong_map outputs/D22_sets/${s}_wrongmap.json $com --split_dir data/${s}_split --split test --max_tokens 16000 --selfq outputs/D22_selfq/$s.jsonl --out ${JT}_$s";;
    lastingbench) echo "python scripts/register_d22.py --d22_set outputs/D22_sets/$s.json --wrong_map outputs/D22_sets/${s}_wrongmap.json $com --split_dir data/${s}_split --split test --max_tokens 18000 --selfq outputs/D22_selfq/$s.jsonl --out ${JT}_$s";;
    loogle|loogle_long) echo "python scripts/register_d22.py --d22_set outputs/D22_sets/$s.json --wrong_map outputs/D22_sets/${s}_wrongmap.json $com --split_dir data/loogle_split --split test --max_tokens 128000 --selfq outputs/D22_selfq/loogle.jsonl --oom_skip 1 --out ${JT}_$s";;
    scf) echo "python scripts/register_d22.py --d22_set outputs/D22_sets/shakespeare_cf.json --wrong_map outputs/D22_sets/shakespeare_cf_wrongmap.json $com --split_dir data/shakespeare_cf_split --split test --max_tokens 60000 --selfq outputs/D20_selfq/shakespeare_cf.jsonl --oom_skip 1 --dist_chunk 64 --out ${JT}_scf";;
    *) echo "echo unknown set $s; false";;
  esac
}
run_lane() {  # run_lane <gpu> <sets...>
  local g=$1; shift
  local log=slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_lane$g.out
  ( export CUDA_VISIBLE_DEVICES=$g
    nvidia-smi -i $g --query-gpu=name --format=csv,noheader
    for s in "$@"; do c=$(judge $s ""); echo ">>> [$(date +%H:%M)] $c"; bash -c "$c" || echo "!!! lane $g step failed: $c"; done
    echo "LANE${g}_DONE" ) > $log 2>&1 &
}
[ -n "$SETS0" ] && run_lane 0 $SETS0
[ -n "$SETS1" ] && run_lane 1 $SETS1
wait
for s in $SETSP; do c=$(judge $s "--enc_device cuda:0 --device cuda:1"); echo ">>> [$(date +%H:%M)] $c"; bash -c "$c" || echo "!!! post step failed: $c"; done
grep -h "LANE.*_DONE\|step failed" slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_lane*.out 2>/dev/null
# kv@0-only sweep (EFFICIENCY RULES 2026-10-23): eval_d13 (no registration), closed_tail + kv + wrong_doc (R2's args), encoder cuda:0 / reader cuda:1.
#   SWEEP="tag=path[@deep],..."  -> outputs/ADDRsw_<tag>_<set>; an existing eval_test.json is NOT recomputed
#   SWSETS (default qasper,musique,lastingbench,loogle,shakespeare_cf)
SWEEP=${SWEEP:-}; SWSETS=${SWSETS:-qasper,musique,lastingbench,loogle,shakespeare_cf}; SWCONDS=${SWCONDS:-closed_tail,kv,wrong_doc}
#   SWSETS may also name quality, longhealth (fold 0) and mcq_* (outputs/MCQ_sets), as in slurm/DS_sweep.sbatch
ITF5="--B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19"
ITF8="--B outputs/D13_deep/encB8_9b_4b_post.pt --layers 3,7,11,15,19,23,27,31"
for ent in $(echo "$SWEEP" | tr ',' ' '); do
  TAG=${ent%%=*}; P=${ent#*=}; SI=$ITF5
  case $P in *@deep) P=${P%@deep}; SI=$ITF8;; esac
  [ -s "$P" ] || { echo "!!! sweep: missing checkpoint $P"; continue; }
  for S in $(echo "$SWSETS" | tr ',' ' '); do
    OUTD=outputs/ADDRsw_${TAG}_$S
    # dedup (coordinator 2026-10-23): agent R2 writes item-identical eval_d13 rows (same args) to outputs/R2s_<tag>_<set>
    # (e.g. R2s_DSAB_qasper); an existing row under either prefix is reused, never recomputed
    [ -s $OUTD/eval_test.json ] && { echo "=== exists, skipped: $OUTD"; continue; }
    [ -s outputs/R2s_${TAG}_$S/eval_test.json ] && { echo "=== exists (R2), reused: outputs/R2s_${TAG}_$S"; continue; }
    case $S in
      quality)    SA="--split_dir data/quality_split --split test --n 0 --max_tokens 16000";;
      longhealth) SA="--split_dir data/longhealth_split --split test --fold 0 --n 0 --max_tokens 16000";;
      mcq_*)      F=outputs/MCQ_sets/$S.json
                  SA="--d22_set $F --wrong_map outputs/MCQ_sets/${S}_wrongmap.json --split test --n 0 --split_dir $(python -I -c "import json;print(json.load(open('$F'))['split_dir'])") --max_tokens $(python -I -c "import json;print(json.load(open('$F'))['max_tokens'])")";;
      *)          F=outputs/D22_sets/$S.json
                  SA="--d22_set $F --wrong_map outputs/D22_sets/${S}_wrongmap.json --split test --n 0 --split_dir $(python -I -c "import json;print(json.load(open('$F'))['split_dir'])") --max_tokens $(python -I -c "import json;print(json.load(open('$F'))['max_tokens'])")";;
    esac
    c="python scripts/eval_d13.py --interface $P --encoder Qwen/Qwen3.5-4B $SI --enc_device cuda:0 --device cuda:1 --conds $SWCONDS --budget 32 --oom_skip 1 $SA --out $OUTD"
    # dedup: reuse any existing item-identical L40S row of the same checkpoint (scripts/r2_reuse.py)
    RA=$(echo "$SA" | sed -E 's/--(wrong_map|n|fold) [^ ]+//g; s/--split [^ ]+//')
    python scripts/r2_reuse.py --target $OUTD --interface $P $RA --conds $SWCONDS && { echo "=== reused via r2_reuse: $OUTD"; continue; }
    echo ">>> [$(date +%H:%M)] $c"
    for try in $(seq 1 60); do ( eval "$c" ); rc=$?; [ $rc -eq 75 ] || break; echo "... exit 75 (OOM resume), rerun $try"; done
    [ $rc -eq 0 ] || echo "!!! sweep step failed (rc $rc): $c"
  done
done
IFS=$'\n'; for c in $(echo "$POST" | sed 's/;;/\n/g'); do echo ">>> [$(date +%H:%M)] $c"; eval "$c" || echo "!!! POST step failed: $c"; done; unset IFS
echo ADDR_DONE
