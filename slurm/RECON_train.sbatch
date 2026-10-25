#!/bin/bash
#SBATCH --job-name=D00RC_train
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=30:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# RECON (agent RECON, 2026-10-23): HM's verbatim-fidelity signal as an aux objective in 9B base training.
# = agent DS's D00DS_max8k command VERBATIM (init outputs/DS_AB_full/d13_best.pt, max8k mix 66876 items / 9884 docs,
#   same task_p, val n=321, exclude_sets D22+MCQ+OOD, exclude_docs, length windows, seed 0, eval / save every 1000)
#   but 3000 steps + RECON aux items (xenc/recon.py; register_d13 --recon_*; own RNG, so the MAIN item stream is the
#   same as DS_max8k's: step-1 main losses must equal DS_max8k's step-1 line, loss .4806 kl .0295 ce .0977 dist .4022).
#   DS_max8k's own step-1000/2000/3000 checkpoints are therefore the matched no-RECON control.
# Then: kv@0 + wrong_doc sweep (slurm/DS_sweep.sbatch, eval_d13, no registration) of d13_step{1000,2000,3000}.
#   ARM=<tag> (outputs/RECON_<ARM>, sweep tags RC<ARM>_s<step>)   RCFLAGS="--recon_w ... "
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
: "${ARM:?set ARM}"; : "${RCFLAGS:?set RCFLAGS}"
OUT=outputs/RECON_$ARM
INIT=outputs/DS_AB_full/d13_best.pt
echo "ARM $ARM | RCFLAGS $RCFLAGS | init $INIT | $(ls -l $INIT)"
EX=outputs/D22_sets/qasper.json,outputs/D22_sets/contractnli.json,outputs/D22_sets/shakespeare_cf.json,outputs/D22_sets/loogle.json,outputs/D22_sets/loogle_long.json,outputs/D22_sets/clipper.json,outputs/D22_sets/musique.json,outputs/D22_sets/lastingbench.json,outputs/MCQ_sets/mcq_bamboo.json,outputs/MCQ_sets/mcq_hallu.json,outputs/MCQ_sets/mcq_leval.json,outputs/MCQ_sets/mcq_musr.json,outputs/OOD_sets/ood_cuad.json,outputs/OOD_sets/ood_infbench.json,outputs/OOD_sets/ood_lbv2.json
if [ ! -s $OUT/d13_step3000.pt ]; then
  c="python scripts/register_d13.py --encoder Qwen/Qwen3.5-4B --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --init $INIT --enc_device cuda:0 --device cuda:1 --extra_splits musiquetr=data/musiquetr_split,narrativeqa=data/narrativeqa_split,racemc=data/racemc_split,multirc=data/multirc_split,nqamc=data/nqamc_split,looglemc=data/looglemc_split --train quality:150,musique:0,lastingbench:0,longhealth:0,qasper:0,contractnli:0,shakespeare_cf:0,loogle:0,clipper:0,musiquetr:0,narrativeqa:0,racemc:0,multirc:0,nqamc:0,looglemc:0 --train_from_val quality:20 --task_p quality:0.10,longhealth:0.04,contractnli:0.06,shakespeare:0.03,clipper:0.08,racemc:0.07,multirc:0.05,nqamc:0.08,looglemc:0.05,qasper:0.12,loogle:0.08,musique:0.04,musiquetr:0.06,lastingbench:0.07,narrativeqa:0.07 --val quality:20,musique:20,lastingbench:20,longhealth:0,qasper:16,contractnli:16,shakespeare_cf:4,loogle:0,clipper:12 --val_q_per_doc 4 --save_items 1 --exclude_sets $EX --exclude_docs outputs/DS_leakage_phaseA.json --doc_len loogle:32000:96000,clipper:32000:96000 --train_doc_len loogle:0:96000,clipper:0:96000,narrativeqa:0:96000 --max_tokens 100000 --lru 2 --mem_safe 1 --dist_w 1.0 --steps 3000 --eval_every 1000 --select_min_step 1000 --save_every 1000 --seed 0 $RCFLAGS --out $OUT"
  echo ">>> [$(date '+%F %T')] $c"
  eval "$c" || { echo "!!! training failed"; exit 1; }
fi
ITFS=""
for s in 1000 2000 3000; do ITFS="${ITFS:+$ITFS,}RC${ARM}_s$s=$OUT/d13_step$s.pt"; done
echo ">>> [$(date '+%F %T')] sweep $ITFS"
ITFS=$ITFS SETS=${SETS:-musique,lastingbench,qasper,loogle,mcq_hallu,mcq_leval,mcq_bamboo,quality,shakespeare_cf} bash slurm/DS_sweep.sbatch
echo "${SLURM_JOB_NAME}_DONE"
