#!/bin/bash
#SBATCH --job-name=D12b_mcq
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12b_mcq_%j.out
# MCQ (agent MCQ, 2026-10-22) -> HANDED TO AGENT D12b (MCQ does not submit 27B jobs).
# 27B reader judge of the D12b lever-v3 base (outputs/D12b_lv3_s2/d13_best.pt: reader Qwen3.5-27B, encoder 9B, 12
# read layers, B encB2_27b_9b_m) on the comprehension MCQ / judgment sets built by scripts/mcq_build.py
# (outputs/MCQ_sets/<set>.json; leakage-checked vs every lever-1 / DS train split; D12b lv3's own mix is a subset of
# those sources + d25cqa / d25fecf train, which share no documents with these sets).
# Same form as D12b_lv3.sbatch's fjdg stage: eval_d12p.py, UNREGISTERED read (kv@0) + wrong_doc + text + closed_book
# (text / closed_book are 27B rows; no registration; the reader never sees the doc in kv / wrong_doc).
#   A    9B-encoder operators for the sets' judged + wrong-map docs -> $XENC_CACHE/D12p/c9b_mcq (cache_d12p --evals_d22)
#   jdg  eval_d12p per set -> outputs/D12b_mcq_<set>
# usage (D12b): bash slurm/submit_when_free.sh auto SETS=mcq_leval,mcq_hallu,mcq_bamboo -- -p drjieliu-h100 slurm/MCQ_27b.sbatch
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
SETS=${SETS:-mcq_leval,mcq_hallu,mcq_bamboo}
STAGES=${STAGES:-A,jdg}; has() { [[ ",$STAGES," == *",$1,"* ]]; }
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
  for s in ${SETS//,/ }; do
    SD=$(python -I -c "import json;print(json.load(open('outputs/MCQ_sets/$s.json'))['split_dir'])")
    echo ">>> 27B $s"; python scripts/eval_d12p.py $PB --interface $ITF --split_dir $SD --split test \
        --d22_set outputs/MCQ_sets/$s.json --wrong_map outputs/MCQ_sets/${s}_wrongmap.json \
        --conds closed_book,text,kv,wrong_doc --out outputs/D12b_mcq_$s || echo "!!! 27B $s failed"
  done
fi
echo D12B_MCQ_DONE
