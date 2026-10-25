#!/bin/bash
# Run ONE small eval inside the image: eval_d13 (9B reader + 4B encoder) on 2 QuALITY test items, conds closed_tail,text,kv,wrong_doc.
# Shows every volume mount and env var the image expects.  MODE=27b runs eval_d12p (27B reader) instead (needs one 80 GB GPU + the
# precomputed encoder cache, or two GPUs; see README section 7).
#
#   DATA_ROOT=/data/kvskill HF_HOST=/data/hf_home bash docker/run_eval_example.sh
#   (DOCKER_USER="$(id -u):$(id -g)" makes the result files belong to you instead of root; HOME/triton cache dirs are world-writable in the image)
#
# Host layout expected:
#   $DATA_ROOT/data/      <- from kvskill_data_eval.tar (data/*_split, data/docs)
#   $DATA_ROOT/outputs/   <- from kvskill_data_eval.tar (outputs/{D22,MCQ,OOD}_sets); results are written here
#   $HF_HOST              <- HF cache filled by docker/download_models.sh   (contains hub/models--Qwen--Qwen3.5-9B ...)
#   $XENC_HOST            <- operator-cache dir (empty is fine for the 9B path; filled during the run)
# The checkpoints are INSIDE the image: /opt/kvskill/ckpt/{DS_max8k/d13_step6000.pt, D12b_max13k_s2/ckpt_5000.pt, maps/*.pt}
set -euo pipefail
RUNTIME="${RUNTIME:-docker}"                       # docker | singularity (apptainer)
IMAGE="${IMAGE:-kvskill-v2:cu130}"                 # singularity: path to the .sif
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT (dir containing data/ and outputs/)}"
HF_HOST="${HF_HOST:?set HF_HOST (HF cache dir, i.e. what download_models.sh filled)}"
XENC_HOST="${XENC_HOST:-$DATA_ROOT/xenc_cache}"; mkdir -p "$XENC_HOST" "$DATA_ROOT/outputs"
MODE="${MODE:-9b}"
R=/opt/kvskill/KV_skill_v2
# one L40S/H100 (>=48 GB): both models on cuda:0.  Two 24 GB GPUs (A10G x2): DEVS="--enc_device cuda:0 --device cuda:1"
DEVS="${DEVS:---device cuda:0}"
OUTNAME="${OUTNAME:-DOCKER_example_${MODE}}"

if [ "$MODE" = 9b ]; then
  CMD="python scripts/eval_d13.py \
    --reader Qwen/Qwen3.5-9B --encoder Qwen/Qwen3.5-4B \
    --B /opt/kvskill/ckpt/maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19 \
    --interface /opt/kvskill/ckpt/DS_max8k/d13_step6000.pt $DEVS \
    --split_dir data/quality_split --split test --n 2 --max_tokens 16000 \
    --conds closed_tail,text,kv,wrong_doc --budget 32 --oom_skip 1 --out outputs/$OUTNAME"
else
  L12=3,7,11,15,19,23,27,31,35,39,43,47
  CMD="python scripts/eval_d12p.py \
    --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B \
    --B /opt/kvskill/ckpt/maps/encB2_27b_9b_m.pt --layers $L12 \
    --interface /opt/kvskill/ckpt/D12b_max13k_s2/ckpt_5000.pt \
    --device cuda:0 --enc_device ${ENC_DEVICE:-cpu} --itf_device cuda:0 --reader_mem ${READER_MEM:-74GiB} \
    --split_dir data/quality_split --split test --n 2 --max_tokens 50000 \
    --conds closed_tail,kv,wrong_doc --budget 32 --oom_skip 1 --out outputs/$OUTNAME"
fi
echo ">>> $CMD"

if [ "$RUNTIME" = docker ]; then
  docker run --rm --gpus all --ipc=host ${DOCKER_USER:+--user $DOCKER_USER} \
    -v "$HF_HOST":/data/hf_home:ro \
    -v "$DATA_ROOT/data":$R/data:ro \
    -v "$DATA_ROOT/outputs":$R/outputs \
    -v "$XENC_HOST":/data/xenc_cache \
    -e HF_HOME=/data/hf_home -e HF_HUB_CACHE=/data/hf_home/hub -e HF_HUB_OFFLINE=1 -e XENC_CACHE=/data/xenc_cache \
    "$IMAGE" bash -lc "cd $R && $CMD"
else   # apptainer / singularity (--nv exposes the host driver)
  ${SING:-singularity} exec --nv \
    --bind "$HF_HOST":/data/hf_home:ro --bind "$DATA_ROOT/data":$R/data:ro --bind "$DATA_ROOT/outputs":$R/outputs \
    --bind "$XENC_HOST":/data/xenc_cache \
    --env HF_HOME=/data/hf_home --env HF_HUB_CACHE=/data/hf_home/hub --env HF_HUB_OFFLINE=1 --env XENC_CACHE=/data/xenc_cache \
    "$IMAGE" bash -c "cd $R && $CMD"
fi
echo "results: $DATA_ROOT/outputs/$OUTNAME/eval_test.json"
