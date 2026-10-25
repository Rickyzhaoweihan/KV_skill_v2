#!/bin/bash
#SBATCH --job-name=D00SC
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# Agent SOFTCONF (2026-10-25): graded / subtle knowledge-conflict eval (outputs/SOFTCONF_SUMMARY.md). NOT SUBMITTED BY
# THE AGENT — the coordinator schedules it. Inference only; the reader never sees the document in any kv row.
#   MODE=prior   READER=Qwen/Qwen3.5-9B|27B  RTAG=9b|27b : closed-book prior over 3 option orders (scripts/softconf_prior.py)
#   MODE=seedgen : Qwen3.5-27B writes perspective seeds (scripts/softconf_seedgen.py), then the persp axis is rebuilt (CPU)
#   MODE=eval    ARCH=9b|27b ITF=<d13 ckpt> TAG=<tag> (or ARMS="ckpt:tag,ckpt:tag") AXES=num,hedge[,ent,persp] SPLITS=val,test
#                [SC_ARGS="--gain_scale 1.25 | --ad_mode priorjsd --ad_tau .9"] [EVARGS=<override of the default eval_cad
#                condition set>] [CLEAN=1: delete the 27B operator cache of an axis after its eval]
#     ARCH=9b : reader 9B + live 4B encoder, both on cuda:0 (1 GPU; 2 GPUs: GPU2=1 -> encoder cuda:0, reader cuda:1)
#     ARCH=27b: reader 27B on one H100; the 9B-encoder operators are first cached (scripts/cache_d12p.py) into
#               $XENC_CACHE/D12p/c9b_sc_<axis> (~55 GB num, ~13 GB hedge), encoder off the GPU during the eval.
# Outputs: outputs/SC_<TAG>_<axis>/eval_<split>.json ; tables: scripts/softconf_table.py (CPU).
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
SC=${SCROOT:-/scratch/drjieliu_root/drjieliu/rickyhan/softconf/data}
MODE=${MODE:-eval}
run() { echo ">>> [$(date '+%F %T')] $*"; eval "$*"; }

if [ "$MODE" = prior ]; then
  DIRS=$(ls -d $SC/softconf_*_split | grep -v pilot | paste -sd,)
  run "python scripts/softconf_prior.py --reader ${READER:-Qwen/Qwen3.5-27B} --split_dirs $DIRS --perms 3 --out outputs/SOFTCONF/prior_${RTAG:-27b}.json"
  echo "${SLURM_JOB_NAME}_DONE"; exit 0
fi
if [ "$MODE" = seedgen ]; then
  run "python scripts/softconf_seedgen.py --model Qwen/Qwen3.5-27B --per_domain 16 --out outputs/SOFTCONF/persp_seeds_gen.json" || exit 1
  run "python scripts/softconf_build.py build --axes persp --seeds outputs/SOFTCONF/persp_seeds_gen.json --out_root $SC"
  echo "${SLURM_JOB_NAME}_DONE"; exit 0
fi

ARCH=${ARCH:-27b}; SC_ARGS=${SC_ARGS:-}
# ARMS="<ckpt>:<tag>,<ckpt>:<tag>" evaluates several interfaces on the same (cached) operators; default "$ITF:$TAG"
ARMS=${ARMS:-"$ITF:$TAG"}
for A_ in $(echo "$ARMS" | tr ',' ' '); do [ -s "${A_%%:*}" ] || { echo "!!! missing checkpoint ${A_%%:*}"; exit 1; }; done
echo "=== SOFTCONF eval $ARCH arms $ARMS axes ${AXES:-num,hedge} splits ${SPLITS:-val,test} on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
# default condition set (per item): plain kv / closed_book / wrong_doc generations, text generation, per-step CAD of kv
# vs closed_book at a = .25 (the CAD_SUMMARY val-selected alpha) and 1 (over-amplification probe), forced-choice
# letter log-probs for kv / closed_tail / closed_book / text / wrong_doc (every FC rule is then a CPU recompute).
EVARGS=${EVARGS:-"--plain kv,closed_book,wrong_doc --cad_refs closed_book --cad_alphas 0.25,1 --text 1 --text_alphas= --fc 1 --budget 32"}
C27=$XENC_CACHE/D12p
FAILED=0
for AX in $(echo "${AXES:-num,hedge}" | tr ',' ' '); do
  SD=$SC/softconf_${AX}_split
  [ -d "$SD" ] || { echo "!!! missing $SD (run scripts/softconf_build.py)"; FAILED=$((FAILED+1)); continue; }
  if [ "$ARCH" = 27b ]; then
    L27=3,7,11,15,19,23,27,31,35,39,43,47
    EV=$(for S in $(echo "${SPLITS:-val,test}" | tr ',' ' '); do printf "%s:%s::0;" "$SD" "$S"; done)
    run "python scripts/cache_d12p.py --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B --layers $L27 --s1_train= --s1_val= --s2_train= --s2_val= --max_tokens 50000 --device cuda --evals '$EV' --out $C27/c9b_sc_$AX" || { echo "!!! operator cache failed ($AX)"; FAILED=$((FAILED+1)); continue; }
    COMMON="--arch d12p --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B --B $XENC_CACHE/D12_maps/encB2_27b_9b_m.pt --layers $L27 --enc_device cpu --reader_mem 74GiB --itf_device cuda:0 --max_tokens 50000 --enc_cache $C27/c9b_sc_$AX"
  else
    DEV="--enc_device cuda:0 --device cuda:0"; [ "${GPU2:-0}" = 1 ] && DEV="--enc_device cuda:0 --device cuda:1"
    COMMON="--arch d13 --encoder Qwen/Qwen3.5-4B --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19 $DEV --max_tokens 16000"
  fi
  for A_ in $(echo "$ARMS" | tr ',' ' '); do
   ITF_=${A_%%:*}; TAG_=${A_##*:}
   for S in $(echo "${SPLITS:-val,test}" | tr ',' ' '); do
    OUTD=outputs/SC_${TAG_}_$AX
    [ -s "$OUTD/eval_$S.json" ] && { echo "=== $OUTD/eval_$S.json exists: reused"; continue; }
    c="python scripts/softconf_eval.py $SC_ARGS -- --interface $ITF_ $COMMON --split_dir $SD --split $S --n 0 --wrong_map $SD/wrongmap.json $EVARGS --oom_skip 1 --out $OUTD"
    for try in $(seq 1 60); do run "$c"; rc=$?; [ $rc -eq 75 ] || break; echo "... exit 75 (OOM resume), rerun $try"; done
    [ $rc -eq 0 ] || { echo "!!! failed (rc $rc): $c"; FAILED=$((FAILED+1)); }
   done
  done
  [ "$ARCH" = 27b ] && [ "${CLEAN:-0}" = 1 ] && rm -rf "$C27/c9b_sc_$AX"
done
echo "failed steps: $FAILED"
echo "${SLURM_JOB_NAME}_DONE"
