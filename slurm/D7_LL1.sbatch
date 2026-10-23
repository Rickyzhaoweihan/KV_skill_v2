#!/bin/bash
#SBATCH --job-name=D7_LL1
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100,drjieliu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=6:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D7_LL1_%j.out
# D7 step 1-2: L/L LongHealth fold 0, training-free native-attention read (gamma=g=1, null off).
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels${PYTHONPATH:+:$PYTHONPATH}
export XENC_CACHE=${XENC_CACHE:-/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache}
export TMPDIR=/scratch/drjieliu_root/drjieliu/rickyhan/hf_caches/tmp
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35
set -uo pipefail
cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
FULL8=0.09375,0.21875,0.34375,0.46875,0.59375,0.71875,0.84375,0.96875
R=Qwen/Qwen3.5-9B
COMMON="--reader $R --split_dir data/longhealth_split --cache_dir $XENC_CACHE --device cuda"
mkdir -p outputs/D7_lh_LL_free8 outputs/D7_lh_LL_free4
python scripts/check_native_harvest.py --model $R --split_dir data/longhealth_split --depths $FULL8 --n_tok 256 \
    --out outputs/D7_lh_LL_free8/harvest_check.json
python scripts/probe_read.py --read native_attn --interface fresh --encoder same --depths $FULL8 $COMMON \
    --split test --fold 0 --n 40 --out outputs/D7_lh_LL_free8/probe_read.json
for V in 8 4; do
  if [ $V = 8 ]; then NL=""; else NL="--native_layers 11,19,27,31"; fi
  python scripts/register_doc.py $COMMON --encoder same --harvest attn_raw --read native_attn --depths $FULL8 $NL \
      --test_fold 0 --steps 0 --out outputs/D7_lh_LL_free$V
  python scripts/eval_doc.py $COMMON --split test --fold 0 --n 0 --conds closed_book,kv,wrong_doc,null,oracle_chunk \
      --interface outputs/D7_lh_LL_free$V/interface_best.pt --out outputs/D7_lh_LL_free$V
done
echo D7_LL1_DONE
