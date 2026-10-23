#!/bin/bash
#SBATCH --job-name=D00MCQ_train
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# MCQ (agent MCQ, 2026-10-22): comprehension-heavy continuation of the lever-1 full base.
# = agent DS's Phase A+B command (outputs/DS_SUMMARY.md, D00DS_ABfull -> outputs/DS_AB_full: same init, same val n=321,
#   same exclusions, same length windows, same 4000 steps / eval 500 / select_min_step 500, same 2x L40S layout)
#   + 4 NEW comprehension MCQ / judgment train splits (scripts/mcq_build.py, strict leakage vs every val / test doc):
#     racemc (RACE-high MCQ), multirc (MultiRC answer judgment), nqamc (NarrativeQA long-story MCQ, same-doc
#     distractors), looglemc (LooGLE long-doc MCQ, same-doc distractors)
#   + a comprehension-heavy task mix (MCQ / judgment .66 vs span .34; DS's mix is span-up-weighted).
# DS_AB_full is therefore the matched comparison for "adding comprehension MCQ" (only the data / weights differ).
# The MCQ / OOD judge sets are added to --exclude_sets so register_d13's leakage assertion covers them in code.
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
OUT=${OUT:-outputs/MCQ_full}
INIT=outputs/D13_lever1_full/d13_best.pt
echo "init $INIT | $(ls -l $INIT)"
D22=outputs/D22_sets/qasper.json,outputs/D22_sets/contractnli.json,outputs/D22_sets/shakespeare_cf.json,outputs/D22_sets/loogle.json,outputs/D22_sets/loogle_long.json,outputs/D22_sets/clipper.json,outputs/D22_sets/musique.json,outputs/D22_sets/lastingbench.json
JS=$(ls outputs/MCQ_sets/mcq_*.json outputs/OOD_sets/ood_*.json | grep -v _wrongmap | paste -sd,)
grep -q "ALL GREEN" slurm/logs/D00cpu_MCQbuild_*.out || { echo "!!! MCQ build tests not green"; exit 1; }
python scripts/register_d13.py --encoder Qwen/Qwen3.5-4B --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --init $INIT \
  --enc_device cuda:0 --device cuda:1 \
  --extra_splits musiquetr=data/musiquetr_split,narrativeqa=data/narrativeqa_split,racemc=data/racemc_split,multirc=data/multirc_split,nqamc=data/nqamc_split,looglemc=data/looglemc_split \
  --train quality:150,musique:0,lastingbench:0,longhealth:0,qasper:0,contractnli:0,shakespeare_cf:0,loogle:0,clipper:0,musiquetr:0,narrativeqa:0,racemc:0,multirc:0,nqamc:0,looglemc:0 \
  --train_from_val quality:20 \
  --task_p quality:0.13,longhealth:0.05,contractnli:0.07,shakespeare:0.04,clipper:0.10,racemc:0.07,multirc:0.05,nqamc:0.09,looglemc:0.06,qasper:0.08,loogle:0.06,musique:0.03,musiquetr:0.04,lastingbench:0.07,narrativeqa:0.06 \
  --val quality:20,musique:20,lastingbench:20,longhealth:0,qasper:16,contractnli:16,shakespeare_cf:4,loogle:0,clipper:12 \
  --val_q_per_doc 4 --save_items 1 --exclude_sets $D22,$JS --exclude_docs outputs/DS_leakage_phaseA.json \
  --doc_len loogle:32000:96000,clipper:32000:96000 --train_doc_len loogle:0:96000,clipper:0:96000,narrativeqa:0:96000 \
  --max_tokens 100000 --lru 2 --mem_safe 1 --dist_w 1.0 --steps ${STEPS:-4000} --eval_every 500 --select_min_step 500 \
  --seed ${SEED:-0} --out $OUT
echo "${SLURM_JOB_NAME}_DONE"
