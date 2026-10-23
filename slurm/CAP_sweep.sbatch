#!/bin/bash
#SBATCH --job-name=CAP_sweep
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=220G
#SBATCH --time=4-00:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# CAP under the EFFICIENCY RULES (docs/AGENT_BRIEF.md): train (2 GPUs) -> optional diagnostics -> UNREGISTERED kv@0
# (+ wrong_doc) sweep with scripts/eval_d13.py over several checkpoints (incl. the parent, same GPU type). Text is
# never recomputed (reused from existing item-identical rows). Job ids in slurm/logs/CAP.jobids. Env:
#   TRAIN   ';;'-separated training commands; stops the job on failure
#   DIAG    ';;'-separated diagnostics after training (failures logged, job continues)
#   SWEEP   "tag=path=nl ..." checkpoints to judge; nl = 5 (lever-1 layers / B) or 8 (+ deep 23/27/31)
#   SW0 / SW1  sets swept in parallel on GPU 0 / GPU 1; SWP: sets swept after, on 2 GPUs (encoder cuda:0)
#   sets: qasper musique lastingbench contractnli quality loogle loogle_long  -> outputs/CAPe_<tag>_<set>
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
TRAIN=${TRAIN:-}; DIAG=${DIAG:-}; SWEEP=${SWEEP:?SWEEP}; SW0=${SW0:-}; SW1=${SW1:-}; SWP=${SWP:-}

IFS=$'\n'; for c in $(echo "$TRAIN" | sed 's/;;/\n/g'); do echo ">>> [$(date +%H:%M)] $c"; eval "$c" || { echo "TRAIN FAILED"; exit 1; }; done
for c in $(echo "$DIAG" | sed 's/;;/\n/g'); do echo ">>> [$(date +%H:%M)] $c"; eval "$c" || echo "!!! diag step failed: $c"; done; unset IFS

itf_flags() { [ "$1" = 8 ] && echo "--B outputs/D13_deep/encB8_9b_4b_post.pt --layers 3,7,11,15,19,23,27,31" \
                           || echo "--B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19"; }
sweep_cmds() {   # sweep_cmds <set> <device flags> -> one eval_d13 command per checkpoint (skips finished outputs)
  local s=$1 dev=$2 e tag path nl out set_args
  case $s in
    qasper|musique|contractnli) set_args="--split_dir data/${s}_split --n 0 --d22_set outputs/D22_sets/$s.json --wrong_map outputs/D22_sets/${s}_wrongmap.json --max_tokens 16000";;
    lastingbench) set_args="--split_dir data/${s}_split --n 0 --d22_set outputs/D22_sets/$s.json --wrong_map outputs/D22_sets/${s}_wrongmap.json --max_tokens 18000";;
    loogle|loogle_long) set_args="--split_dir data/loogle_split --n 0 --d22_set outputs/D22_sets/$s.json --wrong_map outputs/D22_sets/${s}_wrongmap.json --max_tokens 128000 --oom_skip 1";;
    quality) set_args="--split_dir data/quality_split --split test --n 0";;
    *) echo "echo unknown set $s; false"; return;;
  esac
  for e in $SWEEP; do
    tag=${e%%=*}; path=${e#*=}; nl=${path##*=}; path=${path%=*}
    out=outputs/CAPe_${tag}_$s
    [ -f "$out/eval_test.json" ] && { echo "echo skip existing $out"; continue; }
    [ -f "$path" ] || { echo "echo missing checkpoint $path; false"; continue; }
    echo "python scripts/eval_d13.py --encoder Qwen/Qwen3.5-4B $(itf_flags $nl) --interface $path $set_args --conds kv,wrong_doc $dev --out $out"
  done
}
run_lane() {  # run_lane <gpu> <sets...>
  local g=$1; shift
  ( export CUDA_VISIBLE_DEVICES=$g
    for s in "$@"; do sweep_cmds $s "" | while IFS= read -r c; do echo ">>> [$(date +%H:%M)] $c"; bash -c "$c" || echo "!!! lane $g step failed: $c"; done; done
    echo "LANE${g}_DONE" ) > slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_lane$g.out 2>&1 &
}
echo "sweep: $SWEEP | lanes [$SW0] [$SW1] | 2-GPU [$SWP]"
[ -n "$SW0" ] && run_lane 0 $SW0
[ -n "$SW1" ] && run_lane 1 $SW1
wait
for s in $SWP; do sweep_cmds $s "--enc_device cuda:0 --device cuda:1" | while IFS= read -r c; do echo ">>> [$(date +%H:%M)] $c"; bash -c "$c" || echo "!!! post step failed: $c"; done; done
grep -h "LANE.*_DONE\|step failed" slurm/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_lane*.out 2>/dev/null
echo CAP_DONE
