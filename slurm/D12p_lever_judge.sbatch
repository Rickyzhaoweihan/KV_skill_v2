#!/bin/bash
#SBATCH --job-name=D12p_lvjudge
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=30:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12p_lvjudge_%j.out
# Judge of the 27B lever-pilot checkpoint vs deep_v2, same H100 job, same items (paired): LH f0, QuALITY 300,
# ConflictQA-CF, FaithEval-CF (do-no-harm), and the D22 judge sets of Qasper + LooGLE (new train tasks; leakage-filtered
# test items, D22 wrong maps). Base read, UNREGISTERED step 0. Phase A first caches the Qasper/LooGLE judge docs
# ($XENC_CACHE/D12p/c9b_judge; LooGLE operators front-truncated at 49k tokens = the --max_tokens 50000 regime of training).
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
L12=3,7,11,15,19,23,27,31,35,39,43,47
JC=$XENC_CACHE/D12p/c9b_judge
LEVER=${LEVER:-outputs/D12p_lever_pilot_lever/d13_best.pt}; DEEP=outputs/D12p_9b_deep_v2_s2/d13_best.pt; TAG=${TAG:-lvp}
PB="--reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B --B $XENC_CACHE/D12_maps/encB2_27b_9b_m.pt --layers $L12 --enc_device cpu \
    --reader_mem 74GiB --itf_device cuda:0 --enc_cache $JC,$XENC_CACHE/D12p/c9b_lever,$XENC_CACHE/D12p/c9b_deep --max_tokens 50000"
echo ">>> phase A (Qasper + LooGLE judge docs)"
python scripts/cache_d12p.py --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B --layers $L12 --device cuda:0 \
    --s1_train "" --s1_val "" --s2_train "" --s2_val "" --evals "" \
    --evals_d22 outputs/D22_sets/qasper.json,outputs/D22_sets/loogle.json --max_tokens 50000 --out $JC || exit 1
du -sh $JC
declare -A SPEC=([lh]="--split_dir data/longhealth_split --fold 0 --n 0" [q300]="--split_dir data/quality_split --n 300" \
  [cqacf]="--split_dir data/d25_cqacf_split --n 0" [fecf]="--split_dir data/d25_fecf_split --n 0" \
  [qasper]="--split_dir data/qasper_split --d22_set outputs/D22_sets/qasper.json --wrong_map outputs/D22_sets/qasper_wrongmap.json" \
  [loogle]="--split_dir data/loogle_split --d22_set outputs/D22_sets/loogle.json --wrong_map outputs/D22_sets/loogle_wrongmap.json")
for s in lh q300 cqacf fecf qasper loogle; do
  echo ">>> lever $s"; python scripts/eval_d12p.py $PB --interface $LEVER ${SPEC[$s]} --conds closed_tail,text,kv,wrong_doc \
      --out outputs/D12p_${TAG}_lever_$s || echo "!!! lever $s failed"
  echo ">>> deep_v2 $s (same job)"; python scripts/eval_d12p.py $PB --interface $DEEP ${SPEC[$s]} --conds kv,wrong_doc \
      --out outputs/D12p_${TAG}_deepv2_$s || echo "!!! deep_v2 $s failed"
done
echo D12p_LVJUDGE_DONE
