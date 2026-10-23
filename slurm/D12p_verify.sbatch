#!/bin/bash
#SBATCH --job-name=D12p_verify
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=36:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12p_verify_%j.out
# Verification of D12p_deep_v2 (27B reader, 9B enc, deep layers): (1) INDEPENDENT RECOMPUTE of the judge for the
# selected checkpoint and of the d13r base kv on the same GPU / items (new output dirs, *_rc_*); (2) SECOND SEED of
# stage 2 (identical v2 recipe, --seed 1) + its judge (outputs/D12p_9b_deep_v2s1_*).
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home HF_HUB_CACHE=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels${PYTHONPATH:+:$PYTHONPATH}
export XENC_CACHE=${XENC_CACHE:-/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache}
export TMPDIR=/scratch/drjieliu_root/drjieliu/rickyhan/hf_caches/tmp
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35
set -uo pipefail
cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
CACHE=$XENC_CACHE/D12p/c9b_deep
PB="--reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B --B $XENC_CACHE/D12_maps/encB2_27b_9b_m.pt --enc_device cpu --reader_mem 74GiB --itf_device cuda:0 --enc_cache $CACHE"
DEEP=outputs/D12p_9b_deep_v2_s2/d13_best.pt; BASE=outputs/D12p_9b_s2_d13r/d13_best.pt
L12=3,7,11,15,19,23,27,31,35,39,43,47; L8=3,7,11,15,19,23,27,31
declare -A SPEC=([lh]="data/longhealth_split --fold 0 --n 0" [q300]="data/quality_split --n 300" \
  [cqacf]="data/d25_cqacf_split --n 0" [fecf]="data/d25_fecf_split --n 0" [nqswap]="data/d25_nqswap_split --n 0")
if [[ "${SKIP_RC:-0}" != 1 ]]; then
for s in lh q300 cqacf fecf nqswap; do
  echo ">>> recompute deep_v2 $s"; python scripts/eval_d12p.py $PB --layers $L12 --interface $DEEP --split_dir ${SPEC[$s]} \
      --conds closed_tail,text,kv,wrong_doc --out outputs/D12p_9b_deep_v2_rc_$s || echo "!!! failed"
  echo ">>> recompute d13r $s"; python scripts/eval_d12p.py $PB --layers $L8 --interface $BASE --split_dir ${SPEC[$s]} \
      --conds kv,wrong_doc --out outputs/D12p_9b_d13r_rc_$s || echo "!!! failed"
done
fi
echo ">>> seed 1 replication of v2"
STAGES=s2,judge RUN=_v2s1 SEED=1 ENC=Qwen/Qwen3.5-9B TAG=9b \
  S2_DATA=quality:150,musique:150,lastingbench:150,longhealth:0,d25cqa:200,d25fecf:150,d25nqs:300 \
  S2_VAL=quality:10,musique:20,lastingbench:20,longhealth:0,d25cqa:10,d25fecf:10,d25nqs:10 \
  TASK_P=quality:0.3,longhealth:0.15,musique:0.1,lastingbench:0.1,d25cqa:0.15,d25fecf:0.1,d25nqs:0.1 \
  LRGD=1e-2 MINSTEP=250 EVERY=500 STEPS=5000 bash slurm/D12p_deep.sbatch
echo D12p_VERIFY_DONE
