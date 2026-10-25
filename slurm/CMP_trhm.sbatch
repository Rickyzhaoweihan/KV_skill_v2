#!/bin/bash
#SBATCH --job-name=D00CMP_trhm
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# HYB-HM trained arm (xenc/hyb_hm.py): D00DS_max8k's EXACT data command (max8k mix, exclusions, seed 0, same val) from
# INIT=outputs/DS_max8k/d13_step6000.pt plus an HM-style learned writer z=rmsnorm(W h_t) (d numbers/token, bf16) read by the
# reader's deep layers through learned K/V decoders (V zero-init, fixed scale 1: step 0 == the init). 2 GPUs (enc cuda:0,
# reader cuda:1), --hyb_ckpt 1. Then the kv + wrong_doc sweep of each saved step on the 14 standard 9B sets
# (slurm/CMP_sweep2.sbatch) -> outputs/CMPsw_<TAG>s<step>_<set>. Matched control: the DS_max8k step-6000 rows (DSsw_m8k6000_*).
#   TAG=hm512 HMD=512 [HML=23,27,31] [HMLAYER=23] [HMMLP=0] [LR_HM=3e-4] [STEPS=2000] [EXTRA="..."] [SWEEP=1]
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
TAG=${TAG:?TAG}; HMD=${HMD:?HMD}; HML=${HML:-23,27,31}; HMLAYER=${HMLAYER:-23}; HMMLP=${HMMLP:-0}; LR_HM=${LR_HM:-3e-4}
STEPS=${STEPS:-2000}; EXTRA=${EXTRA:-}; SWEEP=${SWEEP:-1}
INIT=${INIT:-outputs/DS_max8k/d13_step6000.pt}
CMP=
SETS14=musique,qasper,lastingbench,quality,longhealth,mcq_leval,mcq_bamboo,mcq_hallu,mcq_musr,contractnli,loogle,loogle_long,shakespeare_cf,clipper
OUT=outputs/CMP_tr_$TAG
ls -l $INIT ${CMP:-/dev/null}
EXS=outputs/D22_sets/qasper.json,outputs/D22_sets/contractnli.json,outputs/D22_sets/shakespeare_cf.json,outputs/D22_sets/loogle.json,outputs/D22_sets/loogle_long.json,outputs/D22_sets/clipper.json,outputs/D22_sets/musique.json,outputs/D22_sets/lastingbench.json,outputs/MCQ_sets/mcq_bamboo.json,outputs/MCQ_sets/mcq_hallu.json,outputs/MCQ_sets/mcq_leval.json,outputs/MCQ_sets/mcq_musr.json,outputs/OOD_sets/ood_cuad.json,outputs/OOD_sets/ood_infbench.json,outputs/OOD_sets/ood_lbv2.json
c="python scripts/register_d13.py --encoder Qwen/Qwen3.5-4B --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt --init $INIT --enc_device cuda:0 --device cuda:1 --extra_splits musiquetr=data/musiquetr_split,narrativeqa=data/narrativeqa_split,racemc=data/racemc_split,multirc=data/multirc_split,nqamc=data/nqamc_split,looglemc=data/looglemc_split --train quality:150,musique:0,lastingbench:0,longhealth:0,qasper:0,contractnli:0,shakespeare_cf:0,loogle:0,clipper:0,musiquetr:0,narrativeqa:0,racemc:0,multirc:0,nqamc:0,looglemc:0 --train_from_val quality:20 --task_p quality:0.10,longhealth:0.04,contractnli:0.06,shakespeare:0.03,clipper:0.08,racemc:0.07,multirc:0.05,nqamc:0.08,looglemc:0.05,qasper:0.12,loogle:0.08,musique:0.04,musiquetr:0.06,lastingbench:0.07,narrativeqa:0.07 --val quality:20,musique:20,lastingbench:20,longhealth:0,qasper:16,contractnli:16,shakespeare_cf:4,loogle:0,clipper:12 --val_q_per_doc 4 --save_items 1 --exclude_sets $EXS --exclude_docs outputs/DS_leakage_phaseA.json --doc_len loogle:32000:96000,clipper:32000:96000 --train_doc_len loogle:0:96000,clipper:0:96000,narrativeqa:0:96000 --max_tokens 100000 --lru 2 --mem_safe 1 --dist_w 1.0 --steps $STEPS --eval_every 1000 --select_min_step 1000 --save_every 1000 --seed 0 --hyb_hm $HMD --hm_layer $HMLAYER --hm_layers $HML --hm_mlp $HMMLP --lr_hm $LR_HM --hyb_ckpt 1 $EXTRA --out $OUT"
if [ ! -s $OUT/d13_last.pt ]; then
  echo ">>> [$(date '+%F %T')] $c"; eval "$c" || echo "!!! training failed"
fi
[ "$SWEEP" = "1" ] || { echo "${SLURM_JOB_NAME}_DONE"; exit 0; }
for k in $(seq 1000 1000 $STEPS); do
  [ -s $OUT/d13_step$k.pt ] || { echo "!!! missing $OUT/d13_step$k.pt"; continue; }
  ITF=$OUT/d13_step$k.pt CMPS="${TAG}s$k=hyb:$OUT/hyb_step$k.pt" CONDS=kv,wrong_doc SETS=${SETS:-$SETS14} bash slurm/CMP_sweep2.sbatch
done
echo "${SLURM_JOB_NAME}_DONE"
