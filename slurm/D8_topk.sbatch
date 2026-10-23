#!/bin/bash
#SBATCH --job-name=D8_topk
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100,drjieliu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D8_topk_%j.out
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
set -uo pipefail
cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# D8 pipeline: (a) selection probe with the fresh reader queries (+ read gold mass through INIT interface),
# (b) register --read topk from INIT, eval all conditions, probe the trained read.
# Usage: RUN=D8_lh_LL_k64 TASK=longhealth FOLD=0 K=64 INIT=outputs/p1_longhealth_9b_LL_f0_center/interface_best.pt \
#        CENTER=outputs/centers/center_Qwen3.5-9B.pt sbatch slurm/D8_topk.sbatch
READER=${READER:-Qwen/Qwen3.5-9B}; ENCODER=${ENCODER:-same}; TASK=${TASK:-longhealth}; RUN=${RUN:?set RUN}
K=${K:-64}; RHO=${RHO:-0.25}; INIT=${INIT:?set INIT}; CENTER=${CENTER:-outputs/centers/center_Qwen3.5-9B.pt}
FOLD_ARGS=${FOLD_ARGS:---fold 0}; REG_FOLD=${REG_FOLD:---test_fold 0}
STAGES=${STAGES:-probe0,reg,eval,probe1}
mkdir -p outputs/$RUN
set -e
if [[ $STAGES == *probe0* ]]; then
python scripts/probe_read.py --reader "$READER" --split_dir data/${TASK}_split --split test --interface "$INIT" \
    --cache_dir "$XENC_CACHE" --out outputs/$RUN/probe_select_init.json $FOLD_ARGS --n ${PROBE_N:-80} \
    --read topk --topk $K --select_layer_rho $RHO ${SEL_ARGS:-} ${PROBE_ARGS:-}
fi
if [[ $STAGES == *reg* ]]; then
python scripts/register_doc.py --reader "$READER" --encoder "$ENCODER" --split_dir data/${TASK}_split \
    --cache_dir "$XENC_CACHE" --out outputs/$RUN --device cuda $REG_FOLD --steps ${STEPS:-300} \
    --center "$CENTER" --init "$INIT" --read topk --topk $K --select_layer_rho $RHO --route_aux 0 \
    --n_val ${N_VAL:-60} --budget 32 ${SEL_ARGS:-} ${REG_ARGS:-}
fi
if [[ $STAGES == *eval* ]]; then
python scripts/eval_doc.py --reader "$READER" --split_dir data/${TASK}_split --split test \
    --interface outputs/$RUN/interface_best.pt --cache_dir "$XENC_CACHE" --device cuda $FOLD_ARGS --n ${EVAL_N:-0} \
    --budget 64 ${EVAL_ARGS:-}
fi
if [[ $STAGES == *probe1* ]]; then
python scripts/probe_read.py --reader "$READER" --split_dir data/${TASK}_split --split test \
    --interface outputs/$RUN/interface_best.pt --cache_dir "$XENC_CACHE" --out outputs/$RUN/probe_topk.json \
    $FOLD_ARGS --n ${PROBE_N:-80} --read topk --topk $K --select_layer_rho $RHO ${SEL_ARGS:-} ${PROBE_ARGS:-}
fi
echo D8_PIPELINE_DONE
