#!/bin/bash
#SBATCH --job-name=D00NT_train
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# NOTES stage 2 (agent NOTES): train the 9B read WITH encoder-written note columns present (register_d13 --notes),
# from the current best 9B base, on DS_AB_full's exact data recipe (DS Phase A+B; exclusions incl. MCQ / OOD sets),
# then an unregistered kv@0 sweep of every saved checkpoint with eval_notes (kv without notes, with own notes, with the
# wrong document's notes, wrong_doc) on the probe's sets. Notes of the judged docs are reused from the probe's store.
#   NOTES   TYPE:L (required)       INIT  init checkpoint (default outputs/DS_AB_full/d13_best.pt)
#   OUT     output dir (default outputs/NT_tr_<type><L>)   STEPS (3000)
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
NOTES=${NOTES:?set NOTES=TYPE:L}
NT=${NOTES%%:*}; NL=${NOTES##*:}
INIT=${INIT:-outputs/DS_AB_full/d13_best.pt}
OUT=${OUT:-outputs/NT_tr_${NT}${NL}}
STEPS=${STEPS:-3000}
TAGP=${TAGP:-tr${NT}${NL}}
[ -s "$INIT" ] || { echo "!!! missing init $INIT"; exit 1; }
echo "=== init $INIT ($(ls -l $INIT | awk '{print $5, $6, $7, $8}')) | notes $NT:$NL | out $OUT"
EXCL=outputs/D22_sets/qasper.json,outputs/D22_sets/contractnli.json,outputs/D22_sets/shakespeare_cf.json,outputs/D22_sets/loogle.json,outputs/D22_sets/loogle_long.json,outputs/D22_sets/clipper.json,outputs/D22_sets/musique.json,outputs/D22_sets/lastingbench.json,outputs/MCQ_sets/mcq_bamboo.json,outputs/MCQ_sets/mcq_hallu.json,outputs/MCQ_sets/mcq_leval.json,outputs/MCQ_sets/mcq_musr.json,outputs/OOD_sets/ood_cuad.json,outputs/OOD_sets/ood_infbench.json,outputs/OOD_sets/ood_lbv2.json
if [ ! -s "$OUT/d13_step${STEPS}.pt" ] && [ ! -s "$OUT/d13_last.pt" ]; then
  c="python scripts/register_d13.py --encoder Qwen/Qwen3.5-4B --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --init $INIT --enc_device cuda:0 --device cuda:1 --extra_splits musiquetr=data/musiquetr_split,narrativeqa=data/narrativeqa_split --train quality:150,musique:0,lastingbench:0,longhealth:0,qasper:0,contractnli:0,shakespeare_cf:0,loogle:0,clipper:0,musiquetr:0,narrativeqa:0 --train_from_val quality:20 --task_p quality:0.12,longhealth:0.06,musique:0.08,musiquetr:0.08,lastingbench:0.11,qasper:0.15,contractnli:0.08,shakespeare:0.03,loogle:0.12,narrativeqa:0.07,clipper:0.10 --val quality:20,musique:20,lastingbench:20,longhealth:0,qasper:16,contractnli:16,shakespeare_cf:4,loogle:0,clipper:12 --val_q_per_doc 4 --save_items 1 --exclude_sets $EXCL --exclude_docs outputs/DS_leakage_phaseA.json --doc_len loogle:32000:96000,clipper:32000:96000 --train_doc_len loogle:0:96000,clipper:0:96000,narrativeqa:0:96000 --max_tokens 100000 --lru 2 --mem_safe 1 --dist_w 1.0 --steps $STEPS --eval_every 1000 --select_min_step 1000 --save_every 1000 --seed 0 --notes $NT:$NL --out $OUT"
  echo ">>> [$(date '+%F %T')] $c"
  eval "$c" || { echo "!!! training failed"; exit 1; }
fi
# kv@0 sweep of the saved checkpoints (2 lanes of 1 GPU, then LooGLE on both GPUs), notes reused from the probe store
SWEEP_CK=${SWEEP_CK:-1000,2000,3000}
for S_ in $(echo "$SWEEP_CK" | tr ',' ' '); do
  CK=$OUT/d13_step${S_}.pt
  [ -s "$CK" ] || { echo "!!! missing $CK"; continue; }
  echo "=== sweep $CK"
  env ITF=$CK TAG=${TAGP}s${S_} SMOKE=0 EXTRA="--note_types $NT --note_lens $NL --prompt_only 0 --wrong_note_types $NT --wrong_note_len $NL --cost_docs 0" \
      SLURM_JOB_NAME=${SLURM_JOB_NAME}_s${S_} bash slurm/NT_probe.sbatch
done
echo "${SLURM_JOB_NAME}_DONE"
