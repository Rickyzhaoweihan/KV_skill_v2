#!/bin/bash
#SBATCH --job-name=D12p_lever
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=30:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12p_lever_%j.out
# D12p "lever 1 + deep" PILOT on the 27B reader (9B encoder, L3-31 + L35-47), init = deep_v2 best.
# ONE H100. Phase A caches a SUBSET of the new lever-1 docs (~190 GB, $XENC_CACHE/D12p/c9b_lever; /scratch only):
#   qasper first 250 train docs, contractnli 200, shakespeare_cf all 25, loogle first 120, clipper all (each after the
#   exclusion + <=48k filters), <=4 items per doc, + the new tasks' val docs. Old tasks come from c9b_deep.
#   Both arms train with --cached_only 1 on the SAME cache (identical subset); all val items must be cached.
# Two arms, same GPU type / val items / seed / steps / recipe (rank-128 deltas, deep gates 1e-2, D13 bar selection):
#   ctrl   continuation on the v2 mix WITHOUT NQ-Swap (its released labels are the prior answers)
#   lever  + ALL benchmark train splits (Qasper, ContractNLI, Shakespeare-CF, LooGLE <=48k, CLIPPER <=48k)
# Both exclude every D22 judged doc (--exclude_sets) + near-duplicates (D13_lever1_leakage.json); paired val (--save_items).
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
ARMS=${ARMS:-A,ctrl,lever}; STEPS=${STEPS:-500}; EVERY=${EVERY:-250}; MINSTEP=${MINSTEP:-250}; RUN=${RUN:-pilot}
INIT=${INIT:-outputs/D12p_9b_deep_v2_s2/d13_best.pt}
L12=3,7,11,15,19,23,27,31,35,39,43,47
CACHE=$XENC_CACHE/D12p/c9b_lever
PB="--reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B --B $XENC_CACHE/D12_maps/encB2_27b_9b_m.pt --layers $L12 \
    --reader_mem 74GiB --itf_device cuda:0 --enc_device cpu --enc_cache $CACHE,$XENC_CACHE/D12p/c9b_deep --cached_only 1"
NEW=qasper:250,contractnli:200,shakespeare_cf:0,loogle:120,clipper:0
EX="outputs/D22_sets/qasper.json,outputs/D22_sets/contractnli.json,outputs/D22_sets/shakespeare_cf.json,outputs/D22_sets/loogle.json,outputs/D22_sets/loogle_long.json,outputs/D22_sets/clipper.json,outputs/D22_sets/musique.json,outputs/D22_sets/lastingbench.json"
FILT="--exclude_sets $EX --exclude_docs outputs/D13_lever1_leakage.json --doc_len loogle:0:48000,clipper:0:48000 --max_tokens 50000"
# one val spec, used verbatim by phase A and by both arms (select_items draws with ONE rng in spec order)
VAL=quality:10,musique:20,lastingbench:20,longhealth:0,d25cqa:10,d25fecf:10,qasper:16,contractnli:16,shakespeare_cf:4,clipper:12
COMMON="$PB $FILT --init $INIT --val $VAL --delta_rank 128 --reg 1.0 --lr_delta 5e-5 --dist_w 1.0 --lr_g_deep 1e-2 \
    --select_min_step $MINSTEP --bar_select 1 --save_items 1 --steps $STEPS --eval_every $EVERY --seed ${SEED:-0} \
    --teacher_cache $XENC_CACHE/D12_trcache_9b --teacher_only 0"
has() { [[ ",$ARMS," == *",$1,"* ]]; }
if has A; then
  echo ">>> phase A (lever subset, 9B encoder only)"; python scripts/cache_d12p.py --reader Qwen/Qwen3.5-27B \
      --encoder Qwen/Qwen3.5-9B --layers $L12 --device cuda:0 --s1_train "" --s1_val "" --s2_train $NEW --s2_q_per_doc 4 \
      --s2_val $VAL --teacher_cache "" --evals "" \
      --exclude_sets $EX --exclude_docs outputs/D13_lever1_leakage.json --doc_len loogle:0:48000,clipper:0:48000 \
      --max_tokens 50000 --out $CACHE || exit 1
  du -sh $CACHE
fi
if has ctrl; then
  echo ">>> ctrl (v2 mix without NQ-Swap)"; python scripts/register_d12p.py $COMMON \
      --train quality:150,musique:150,lastingbench:150,longhealth:0,d25cqa:200,d25fecf:150 \
      --task_p quality:0.35,longhealth:0.17,musique:0.12,lastingbench:0.12,d25cqa:0.14,d25fecf:0.10 \
      --out outputs/D12p_lever_${RUN}_ctrl || echo "!!! ctrl failed"
fi
if has lever; then
  echo ">>> lever 1 (all benchmark train splits)"; python scripts/register_d12p.py $COMMON \
      --train quality:150,musique:150,lastingbench:150,longhealth:0,$NEW,d25cqa:200,d25fecf:150 \
      --task_p quality:0.16,longhealth:0.08,musique:0.06,lastingbench:0.06,qasper:0.10,contractnli:0.08,shakespeare:0.06,loogle:0.10,clipper:0.10,d25cqa:0.12,d25fecf:0.08 \
      --out outputs/D12p_lever_${RUN}_lever || echo "!!! lever failed"
fi
echo D12p_LEVER_DONE
