#!/bin/bash
#SBATCH --job-name=D12b_mcqj
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12b_mcqj_%j.out
# D12b: MCQ-set judges (mcq_leval, mcq_bamboo; operators already in c9b_mcq) for several 27B checkpoints in ONE job,
# submitted with --dependency=afterany on the runs that produce them. JOBS (env) = "tag:interface[:hyper_ckpt];..."
# (a hyper ckpt adds kv_base / kv_Hswap). Missing checkpoints are skipped. Out: outputs/D12b_mcq_<tag>_<set>.
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
SETS=${SETS:-mcq_leval,mcq_bamboo}
STAGES=${STAGES:-jdg}; has() { [[ ",$STAGES," == *",$1,"* ]]; }
L12=3,7,11,15,19,23,27,31,35,39,43,47
C=$XENC_CACHE/D12p; JC=$C/c9b_judge; LC=$C/c9b_lever; FC=$C/c9b_full; MC=$C/c9b_mcq
ITF=${ITF:-outputs/D12b_lv3_s2/d13_best.pt}
B="--B $XENC_CACHE/D12_maps/encB2_27b_9b_m.pt"
PB="--reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B $B --layers $L12 --enc_device cpu --reader_mem 74GiB --itf_device cuda:0 \
    --enc_cache $JC,$LC,$C/c9b_deep,$FC,$MC --max_tokens 50000"
echo "interface $ITF | $(ls -l $ITF) | sets $SETS"
SF=$(echo $SETS | tr ',' '\n' | sed 's#^#outputs/MCQ_sets/#; s#$#.json#' | paste -sd,)
if has A; then
  echo ">>> 9B operators for $SF"; python scripts/cache_d12p.py --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B \
      --layers $L12 --device cuda:0 --s1_train "" --s1_val "" --s2_train "" --s2_val "" --teacher_cache "" --evals "" \
      --evals_d22 $SF --skip_have $JC,$LC,$C/c9b_deep,$FC --max_tokens 50000 --out $MC || exit 1
fi
if has jdg; then
  IFS=';' read -ra JL <<< "${JOBS:-}"
  for J in "${JL[@]}"; do
    IFS=':' read -r TAG ITF HC <<< "$J"
    [ -s "$ITF" ] || { echo "!!! $TAG: no interface $ITF"; continue; }
    HX=""; CONDS=${CONDS_PLAIN:-closed_book,text,kv,wrong_doc}
    if [ -n "${HC:-}" ]; then [ -s "$HC" ] || { echo "!!! $TAG: no H ckpt $HC"; continue; }; HX="--hyper_ckpt $HC"; CONDS=kv,wrong_doc,kv_base,kv_Hswap; fi
    for s in ${SETS//,/ }; do
      SD=$(python -I -c "import json;print(json.load(open('outputs/MCQ_sets/$s.json'))['split_dir'])")
      echo ">>> 27B $TAG $s"; python scripts/eval_d12p.py $PB --interface $ITF $HX --split_dir $SD --split test \
          --d22_set outputs/MCQ_sets/$s.json --wrong_map outputs/MCQ_sets/${s}_wrongmap.json \
          --conds $CONDS --out outputs/D12b_mcq_${TAG}_$s || echo "!!! 27B $TAG $s failed"
    done
  done
fi
echo D12B_MCQJ_DONE
