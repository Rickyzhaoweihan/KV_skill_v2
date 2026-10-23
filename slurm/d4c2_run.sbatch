#!/bin/bash
#SBATCH --job-name=D4c2
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100,drjieliu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D4c2_%j.out
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

# D4c2: scaled RL (snapshots at 30/60, early stop on no signal, resumable) + snapshot evals.
# Usage: RUN=D4c2_SL_s0 ITF=outputs/D4c2_reg_q_SL/interface_best.pt TASK=quality MAX_DOCS=30 \
#        sbatch slurm/d4c2_run.sbatch [extra rl_doc_operator.py args]
READER=${READER:-Qwen/Qwen3.5-9B}
TASK=${TASK:-quality}
RUN=${RUN:?set RUN=D4c2_<name>}
ITF=${ITF:?set ITF=<registered interface>}
MAX_DOCS=${MAX_DOCS:-30}
EVAL_EXTRA=${EVAL_EXTRA:-}
python scripts/rl_doc_operator.py --reader "$READER" --split_dir data/${TASK}_split --split train \
    --q_split train --max_docs "$MAX_DOCS" --interface "$ITF" --cache_dir "$XENC_CACHE" --out outputs/$RUN \
    --steps 100 --prompts 24 --temp 1.5 --kl_warmup 20 --snap_steps 30,60 --stop_patience 2 --eval_every 0 \
    --resume "$@" || exit 1
EV="--reader $READER --split_dir data/${TASK}_split --split train --q_split heldout --max_docs $MAX_DOCS --n 0 --budget 32 --cache_dir $XENC_CACHE --no_resume $EVAL_EXTRA"
R=outputs/$RUN/interface_ref.pt
python scripts/eval_doc.py $EV --interface $R --operator_dir outputs/$RUN/operators --conds closed_book,kv,wrong_doc,null --out outputs/$RUN/eval_rl
python scripts/eval_doc.py $EV --interface $R --operator_dir outputs/$RUN/operators_init --conds kv,wrong_doc,null --out outputs/$RUN/eval_init
python scripts/eval_doc.py $EV --interface $R --operator_dir outputs/$RUN/operators_s30 --conds kv,wrong_doc --out outputs/$RUN/eval_s30
python scripts/eval_doc.py $EV --interface $R --operator_dir outputs/$RUN/operators_s60 --conds kv,wrong_doc --out outputs/$RUN/eval_s60
# harvested Library through the same (disjoint) interface, on the same docs' held-out questions
python scripts/eval_doc.py $EV --interface "$ITF" --conds kv,wrong_doc --out outputs/$RUN/eval_harvested
echo D4C2_DONE
