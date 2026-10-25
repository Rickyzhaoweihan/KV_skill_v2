#!/bin/bash
#SBATCH --job-name=D17d_run
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-l40s
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=72:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/%x_%j.out
# D17d (agent RL): GRPO base training (scripts/rl_d17.py) from the current best 9B base, then a kv@0-only checkpoint
# sweep (scripts/eval_d13.py, no registration, conds kv,wrong_doc; text rows are reused from existing L40S jobs).
# Env:  TAG    run name -> outputs/D17d_<TAG>
#       RLARGS extra rl_d17.py args (e.g. "--group 16 --lr 1e-5 --lr_mlp 5e-5")
#       INIT   init checkpoint (default outputs/DS_AB_full/d13_best.pt; it is also swept as the step-0 base row)
#       SKIP_TRAIN=1 only runs the sweep (e.g. after a time-out)
#       SWEEP_BASE=0 skips the INIT rows (they go to the shared outputs/D17d_base_k0; one job computes them)
#       BASE_DIR   where the INIT (step-0) rows go (default outputs/D17d_base_k0; set it when INIT is not DS_AB_full)
# Sweep: INIT + d17_step{1000,2000,3000}.pt + d17_best.pt, each on the D22 sets (Qasper, MuSiQue, LastingBench,
# ContractNLI, Shakespeare-CF, LooGLE, LooGLE-long), LongHealth fold 0 and QuALITY test full -> outputs/D17d_<TAG>_k0/<ckpt>_<set>.
# Job ids: slurm/logs/D17d.jobids.
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels:$REPO/scripts${PYTHONPATH:+:$PYTHONPATH}
export XENC_CACHE=${XENC_CACHE:-/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache}
export TMPDIR=/scratch/drjieliu_root/drjieliu/rickyhan/hf_caches/tmp
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35
set -uo pipefail
cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
TAG=${TAG:?set TAG}
INIT=${INIT:-outputs/DS_AB_full/d13_best.pt}
OUT=outputs/D17d_$TAG
RLARGS=${RLARGS:-}
COMMON="--encoder Qwen/Qwen3.5-4B --B outputs/D11_maps/encB5_9b_4b_post_xf01.pt"
echo "D17d TAG=$TAG INIT=$INIT OUT=$OUT RLARGS=[$RLARGS]"

if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  python scripts/rl_d17.py $COMMON --device cuda:1 --enc_device cuda:0 --mem_safe 1 --row_chunk 2 --oom_skip 1 \
    --train quality:150,musique:0,lastingbench:0,longhealth:0,qasper:0,contractnli:0,shakespeare_cf:0,loogle:0,clipper:0 \
    --train_from_val quality:10 \
    --val quality:10,musique:30,lastingbench:30,longhealth:0,qasper:12,contractnli:8,shakespeare_cf:2,loogle:10 --val_q_per_doc 6 \
    --exclude_sets outputs/D22_sets/qasper.json,outputs/D22_sets/contractnli.json,outputs/D22_sets/shakespeare_cf.json,outputs/D22_sets/loogle.json,outputs/D22_sets/loogle_long.json,outputs/D22_sets/clipper.json,outputs/D22_sets/musique.json,outputs/D22_sets/lastingbench.json \
    --exclude_docs outputs/DS_leakage_phaseA.json --doc_len loogle:0:48000,clipper:0:48000 --max_tokens 48000 \
    --task_p quality:0.2,longhealth:0.1,musique:0.1,lastingbench:0.1,qasper:0.12,contractnli:0.1,shakespeare:0.06,loogle:0.1,clipper:0.12 \
    --no_repeat 1 --save_items 1 --prompts 8 --group 8 --mb 2 --lr 5e-6 --lr_mlp 2.5e-5 --beta 0.1 --kl_warmup 25 \
    --temp 1.2 --temp_max 1.5 --anchor_w 1 --dist_w 1 --patience 0 \
    --init $INIT --steps 3000 --eval_every 500 --select_from 500 --save_every 1000 --time_limit_h 44 \
    $RLARGS --out $OUT || echo "!!! training exited non-zero; sweeping whatever checkpoints exist"
fi

# ---- kv@0 sweep (unregistered read, kv + wrong_doc; text reused from existing L40S rows) ----
SW=${OUT}_k0; mkdir -p $SW
EV="python scripts/eval_d13.py $COMMON --enc_device cuda:0 --device cuda:1 --oom_skip 1 --conds kv,wrong_doc --budget 32"
ckpts=""
[ "${SWEEP_BASE:-1}" = "1" ] && ckpts="base:$INIT"
for s in 1000 2000 3000; do [ -f $OUT/d17_step$s.pt ] && ckpts="$ckpts s$s:$OUT/d17_step$s.pt"; done
[ -f $OUT/d17_best.pt ] && ckpts="$ckpts best:$OUT/d17_best.pt"
echo "sweep checkpoints: $ckpts"
for ck in $ckpts; do
  name=${ck%%:*}; path=${ck#*:}
  SW=${OUT}_k0; [ "$name" = base ] && SW=${BASE_DIR:-outputs/D17d_base_k0}; mkdir -p $SW
  for spec in qasper:16000 musique:16000 lastingbench:18000 contractnli:16000 shakespeare_cf:60000 loogle:128000 loogle_long:128000; do
    set_=${spec%%:*}; mt=${spec#*:}; o=$SW/${name}_$set_
    [ -f $o/eval_test.json ] && { echo "skip $o (exists)"; continue; }
    echo ">>> [$(date +%H:%M)] kv@0 $name $set_"
    $EV --interface $path --d22_set outputs/D22_sets/$set_.json --wrong_map outputs/D22_sets/${set_}_wrongmap.json \
      --split_dir data/${set_%_long}_split --split test --n 0 --max_tokens $mt --out $o || echo "!!! failed $o"
  done
  o=$SW/${name}_lh_f0
  [ -f $o/eval_test.json ] || { echo ">>> [$(date +%H:%M)] kv@0 $name lh_f0"; $EV --interface $path --split_dir data/longhealth_split --split test --fold 0 --n 0 --max_tokens 16000 --out $o || echo "!!! failed $o"; }
  o=$SW/${name}_quality
  [ -f $o/eval_test.json ] || { echo ">>> [$(date +%H:%M)] kv@0 $name quality"; $EV --interface $path --split_dir data/quality_split --split test --n 0 --max_tokens 16000 --out $o || echo "!!! failed $o"; }
done
echo D17d_DONE
