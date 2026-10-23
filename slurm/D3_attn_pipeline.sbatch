#!/bin/bash
#SBATCH --job-name=D3_attn
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D3_%j.out
# D3 pipeline: encode (harvest=attn) -> register -> eval -> probe, one job.
# readers <=9B may add the L40S partition: sbatch --partition=drjieliu-h100,drjieliu-l40s ...
# (on L40S pass MAX_MEM=40GiB if sharding with DEVICE=auto).
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels${PYTHONPATH:+:$PYTHONPATH}
export XENC_CACHE=${XENC_CACHE:-/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache}
export TMPDIR=/scratch/drjieliu_root/drjieliu/rickyhan/hf_caches/tmp
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35
set -uo pipefail   # after activation: conda's activate scripts are not set -u clean
cd $REPO

# Usage: ENCODER=Qwen/Qwen3.5-9B READER=Qwen/Qwen3.5-9B TASK=longhealth RUN=D3_lh_LL_attn \
#        sbatch --partition=drjieliu-h100,drjieliu-l40s slurm/D3_attn_pipeline.sbatch
# ENCODER=same is not used here: we pre-encode with the named encoder (same cache key as 'same' when
# ENCODER == READER). REG_ARGS / EVAL_ARGS / PROBE_ARGS add extra CLI args.
READER=${READER:-Qwen/Qwen3.5-9B}
ENCODER=${ENCODER:-Qwen/Qwen3.5-9B}
TASK=${TASK:-longhealth}
RUN=${RUN:?set RUN}
FOLD=${FOLD:-0}
REG_ENC=$ENCODER; [ "$ENCODER" = "$READER" ] && REG_ENC=same
set -e
python scripts/encode_docs.py --encoder "$ENCODER" --split_dir data/${TASK}_split --cache_dir "$XENC_CACHE" \
    --harvest attn --device cuda
python scripts/register_doc.py --reader "$READER" --encoder "$REG_ENC" --harvest attn \
    --split_dir data/${TASK}_split --cache_dir "$XENC_CACHE" --out outputs/$RUN --device cuda \
    --test_fold $FOLD --steps ${STEPS:-300} --read_temp ${READ_TEMP:-30} --n_val ${N_VAL:-60} ${REG_ARGS:-}
python scripts/eval_doc.py --reader "$READER" --split_dir data/${TASK}_split --split test \
    --interface outputs/$RUN/interface_best.pt --cache_dir "$XENC_CACHE" --device cuda \
    --fold $FOLD --n 0 ${EVAL_ARGS:-}
python scripts/probe_read.py --reader "$READER" --split_dir data/${TASK}_split --interface outputs/$RUN/interface_best.pt \
    --cache_dir "$XENC_CACHE" --out outputs/$RUN/probe_read.json --fold $FOLD --n 40 ${PROBE_ARGS:-}
echo D3_PIPELINE_DONE
