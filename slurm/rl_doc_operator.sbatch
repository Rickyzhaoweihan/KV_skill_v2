#!/bin/bash
#SBATCH --job-name=D4_rl
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100,drjieliu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D4_rl_%j.out
# D4 "RL from document": per-document compact operators from task reward (scripts/rl_doc_operator.py),
# then (EVAL=1) eval_doc.py on the SAME docs' held-out questions for: RL operators, their SVD init
# (no-RL control) and the harvested Library baseline — each with kv / wrong_doc / null.
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Usage: RUN=D4_rl_quality_9b ITF=outputs/p1_quality_9b_LL_center/interface_best.pt MAX_DOCS=5 EVAL=1 \
#        sbatch slurm/rl_doc_operator.sbatch [extra rl_doc_operator.py args, e.g. --steps 100]
READER=${READER:-Qwen/Qwen3.5-9B}
TASK=${TASK:-quality}
RUN=${RUN:?set RUN=D4_<name>}
ITF=${ITF:-outputs/p1_quality_9b_LL_center/interface_best.pt}
MAX_DOCS=${MAX_DOCS:-5}
EVAL=${EVAL:-1}
HARVEST=${HARVEST:-1}          # 1 = also eval the harvested Library through $ITF
EVAL_EXTRA=${EVAL_EXTRA:-}     # extra eval_doc.py args (e.g. --fold filters)
python scripts/rl_doc_operator.py --reader "$READER" --split_dir data/${TASK}_split --split train \
    --q_split train --max_docs "$MAX_DOCS" --interface "$ITF" --cache_dir "$XENC_CACHE" \
    --out outputs/$RUN "$@" || exit 1
[ "$EVAL" = "1" ] || exit 0
EV="--reader $READER --split_dir data/${TASK}_split --split train --q_split heldout --max_docs $MAX_DOCS --n 0 --budget 32 --conds closed_book,kv,wrong_doc,null --cache_dir $XENC_CACHE --no_resume $EVAL_EXTRA"
python scripts/eval_doc.py $EV --interface outputs/$RUN/interface_ref.pt --operator_dir outputs/$RUN/operators --out outputs/$RUN/eval_rl
python scripts/eval_doc.py $EV --interface outputs/$RUN/interface_ref.pt --operator_dir outputs/$RUN/operators_init --out outputs/$RUN/eval_init
[ "$HARVEST" = "1" ] && python scripts/eval_doc.py $EV --interface "$ITF" --out outputs/$RUN/eval_harvested
echo D4_DONE
