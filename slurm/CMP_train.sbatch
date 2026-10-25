#!/bin/bash
#SBATCH --job-name=D00CMP_tr
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# CMP trained arm: D00DS_max8k's EXACT command (init outputs/DS_AB_full/d13_best.pt, max8k mix, seed 0, same val) plus
# a CMP operator bottleneck (--cmp $CMP, decoders trained at --lr_cmp) for $STEPS steps (ckpt every 1000), then the
# kv@0 sweep of every saved step (slurm/CMP_sweep.sbatch logic) -> outputs/CMPsw_<TAG>s<step>_<set>.
# The matched uncompressed control is DS_max8k itself (same command, same steps; its d13_step<N>.pt).
#   TAG=all128 CMP=outputs/CMP_fit/cmp_all_128.pt [LR_CMP=1e-4] [STEPS=2000] [EXTRA="..."] [SWEEP=1]
#   TAG=hyb512 HYB=outputs/CMP_fit/cmp_<hyb spec>.pt   (hybrid deep channel, gates 0 at init; xenc/cmp_hyb.py)
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
TAG=${TAG:?TAG}; CMP=${CMP:-}; HYB=${HYB:-}; LR_CMP=${LR_CMP:-1e-4}; STEPS=${STEPS:-2000}; EXTRA=${EXTRA:-}; SWEEP=${SWEEP:-1}
INIT=${INIT:-outputs/DS_AB_full/d13_best.pt}
OUT=outputs/CMP_tr_$TAG
ls -l $INIT ${CMP:-/dev/null}
EXS=outputs/D22_sets/qasper.json,outputs/D22_sets/contractnli.json,outputs/D22_sets/shakespeare_cf.json,outputs/D22_sets/loogle.json,outputs/D22_sets/loogle_long.json,outputs/D22_sets/clipper.json,outputs/D22_sets/musique.json,outputs/D22_sets/lastingbench.json,outputs/MCQ_sets/mcq_bamboo.json,outputs/MCQ_sets/mcq_hallu.json,outputs/MCQ_sets/mcq_leval.json,outputs/MCQ_sets/mcq_musr.json,outputs/OOD_sets/ood_cuad.json,outputs/OOD_sets/ood_infbench.json,outputs/OOD_sets/ood_lbv2.json
c="python scripts/register_d13.py --encoder Qwen/Qwen3.5-4B --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --init $INIT --enc_device cuda:0 --device cuda:1 --extra_splits musiquetr=data/musiquetr_split,narrativeqa=data/narrativeqa_split,racemc=data/racemc_split,multirc=data/multirc_split,nqamc=data/nqamc_split,looglemc=data/looglemc_split --train quality:150,musique:0,lastingbench:0,longhealth:0,qasper:0,contractnli:0,shakespeare_cf:0,loogle:0,clipper:0,musiquetr:0,narrativeqa:0,racemc:0,multirc:0,nqamc:0,looglemc:0 --train_from_val quality:20 --task_p quality:0.10,longhealth:0.04,contractnli:0.06,shakespeare:0.03,clipper:0.08,racemc:0.07,multirc:0.05,nqamc:0.08,looglemc:0.05,qasper:0.12,loogle:0.08,musique:0.04,musiquetr:0.06,lastingbench:0.07,narrativeqa:0.07 --val quality:20,musique:20,lastingbench:20,longhealth:0,qasper:16,contractnli:16,shakespeare_cf:4,loogle:0,clipper:12 --val_q_per_doc 4 --save_items 1 --exclude_sets $EXS --exclude_docs outputs/DS_leakage_phaseA.json --doc_len loogle:32000:96000,clipper:32000:96000 --train_doc_len loogle:0:96000,clipper:0:96000,narrativeqa:0:96000 --max_tokens 100000 --lru 2 --mem_safe 1 --dist_w 1.0 --steps $STEPS --eval_every 1000 --select_min_step 1000 --save_every 1000 --seed 0 ${CMP:+--cmp $CMP --lr_cmp $LR_CMP} ${HYB:+--hyb $HYB} $EXTRA --out $OUT"
if [ ! -s $OUT/d13_last.pt ]; then
  echo ">>> [$(date '+%F %T')] $c"; eval "$c" || echo "!!! training failed"
fi
[ "$SWEEP" = "1" ] || { echo "${SLURM_JOB_NAME}_DONE"; exit 0; }
for k in $(seq 1000 1000 $STEPS); do
  [ -s $OUT/d13_step$k.pt ] || { echo "!!! missing $OUT/d13_step$k.pt"; continue; }
  if [ -n "$HYB" ]; then
    ITF=$OUT/d13_step$k.pt CMPS="${TAG}s$k=hyb:$OUT/hyb_step$k.pt" CONDS=${CONDS:-kv} bash slurm/CMP_sweep.sbatch
  elif [ -n "$CMP" ]; then
    ITF=$OUT/d13_step$k.pt CMPS="${TAG}s$k=$OUT/cmp_step$k.pt" CONDS=${CONDS:-kv} bash slurm/CMP_sweep.sbatch
  else
    ITF=$OUT/d13_step$k.pt BASE=${TAG}s$k bash slurm/CMP_sweep.sbatch
  fi
done
echo "${SLURM_JOB_NAME}_DONE"
