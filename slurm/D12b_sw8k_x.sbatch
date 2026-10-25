#!/bin/bash
#SBATCH --job-name=D12b_sw8k_x
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12b_sw8k_x_%j.out
# DS add-on to the 27B max8k sweep (coordinator 2026-10-23): sets the D12b_max8k sweep does not cover, at max8k ckpts
# 4000 + 8000 (submit with --dependency=afterany on the D12b_max8k job). No 27B text row exists for these sets, so the
# FIRST checkpoint also runs the text pass (checkpoint-independent). Same 27B read regime as D12b (9B encoder operators
# cached, --max_tokens 50000: LooGLE-long / CLIPPER docs are front-truncated to ~49k tokens, as in 27B training).
#   A    9B operators of the sets' judged + wrong-map docs -> $XENC_CACHE/D12p/c9b_ds (skips operators cached elsewhere)
#   jdg  eval_d12p kv,wrong_doc (+ text on the first ckpt) -> outputs/D12b_max8k_c<step>_<set>
# Scoring on the strict clean subsets is done on CPU afterwards (scripts/ds_sweep_table.py-style, DS_judged_leak_max8k).
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
SETS=${SETS:-musique,contractnli,mcq_musr,clipper,loogle_long}
CKS=${CKS:-4000,8000}
STAGES=${STAGES:-A,jdg}; has() { [[ ",$STAGES," == *",$1,"* ]]; }
L12=3,7,11,15,19,23,27,31,35,39,43,47
C=$XENC_CACHE/D12p; JC=$C/c9b_judge; LC=$C/c9b_lever; FC=$C/c9b_full; RC=$C/c9b_max; MC=$C/c9b_mcq; DC=$C/c9b_ds
OUT=outputs/D12b_max8k_s2
B="--B $XENC_CACHE/D12_maps/encB2_27b_9b_m.pt"
PB="--reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B $B --layers $L12 --enc_device cpu --reader_mem 74GiB --itf_device cuda:0 \
    --enc_cache $JC,$LC,$C/c9b_deep,$FC,$RC,$MC,$DC --max_tokens 50000"
setf() { case $1 in mcq_*) echo outputs/MCQ_sets/$1.json;; *) echo outputs/D22_sets/$1.json;; esac; }
SF=$(for s in ${SETS//,/ }; do setf $s; done | paste -sd,)
echo "sets $SETS | ckpts $CKS | set files $SF"
if has A; then
  echo ">>> 9B operators for $SF"; python scripts/cache_d12p.py --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B \
      --layers $L12 --device cuda:0 --s1_train "" --s1_val "" --s2_train "" --s2_val "" --teacher_cache "" --evals "" \
      --evals_d22 $SF --skip_have $JC,$LC,$C/c9b_deep,$FC,$RC,$MC --max_tokens 50000 --out $DC || exit 1
  du -sh $DC
fi
if has jdg; then
  first=1
  for k in ${CKS//,/ }; do
    CK=$OUT/ckpt_$k.pt; [ -s $CK ] || { echo "!!! no $CK"; continue; }
    echo "=== 27B max8k ckpt $k: $(ls -l $CK)"
    CONDS=kv,wrong_doc; [ $first = 1 ] && CONDS=text,kv,wrong_doc
    for s in ${SETS//,/ }; do
      F=$(setf $s); SD=$(python -I -c "import json;print(json.load(open('$F'))['split_dir'])")
      WM=${F%.json}_wrongmap.json
      echo ">>> 27B max8k@$k $s ($CONDS)"; python scripts/eval_d12p.py $PB --interface $CK --split_dir $SD --split test \
          --n 0 --d22_set $F --wrong_map $WM --conds $CONDS --out outputs/D12b_max8k_c${k}_$s || echo "!!! 27B max8k@$k $s failed"
    done
    first=0
  done
fi
echo D12B_SW8K_X_DONE
