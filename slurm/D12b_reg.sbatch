#!/bin/bash
#SBATCH --job-name=D12b_reg
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=40:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12b_reg_%j.out
# D12b task 1: per-document registration HEADROOM on the 27B lever-FULL base (outputs/D12p_lever_full_s2/d13_best.pt,
# 9B encoder, 12 read layers), judged on exactly the items of the base judge (outputs/D12p_lvfull_<set>).
# Recipe: D22 = D20 (parts dA,gate r16, lr 1e-3, shuffle 2, warmup 25, 20% held-out self-q, budgets 0/50/100/200,
# kv_reg = per-doc early stop es<=200 on held-out self-q loss), KL(27B text teacher)+0.5 CE on 9B-written
# self-questions; the encoder-attention distillation term is OFF (dist_w 0): the operators come from the phase-A cache
# (scripts/register_d12b.py), which holds no encoder records for self-question rows. kv@0 == the base judge's read.
# ARMS (env, comma list): fecf (+ its 9B self-questions first), qas, scf, loogle.
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
ARMS=${ARMS:-fecf,qas,scf,loogle}; has() { [[ ",$ARMS," == *",$1,"* ]]; }
C=$XENC_CACHE/D12p
BASE=outputs/D12p_lever_full_s2/d13_best.pt
RT="--runtime d12p --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-9B --B $XENC_CACHE/D12_maps/encB2_27b_9b_m.pt \
    --layers 3,7,11,15,19,23,27,31,35,39,43,47 --interface $BASE --reader_mem 74GiB --enc_device cpu --itf_device cuda:0 \
    --mem_safe 1 --oom_skip 1 --max_tokens 50000 --enc_cache $C/c9b_judge,$C/c9b_deep"
REC="--rank 16 --lr 1e-3 --shuffle 2 --warmup 25 --holdout 0.2 --budgets 0,50,100,200 --split test --parts dA,gate \
     --teacher text --dist_w 0 --conds kv,wrong_doc"
echo "base $BASE | $(ls -l $BASE)"
if has fecf; then
  [ -s outputs/D12b_selfq/fecf.jsonl ] || { echo ">>> FaithEval-CF self-questions by 9B"; mkdir -p outputs/D12b_selfq
    python scripts/d22_selfq.py --split_dir data/d25_fecf_split --docs test:0 --templates mcq,mix \
      --reader Qwen/Qwen3.5-9B --out outputs/D12b_selfq/fecf.jsonl || echo "!!! fecf selfq failed"; }
  echo ">>> FaithEval-CF registration"; python scripts/register_d12b.py $RT $REC --split_dir data/d25_fecf_split --n 0 \
      --selfq outputs/D12b_selfq/fecf.jsonl --out outputs/D12b_reg_fecf || echo "!!! fecf failed"
fi
for s in qasper loogle shakespeare_cf; do
  case $s in qasper) arm=qas; sq=outputs/D22_selfq/qasper.jsonl;; loogle) arm=loogle; sq=outputs/D22_selfq/loogle.jsonl;;
    shakespeare_cf) arm=scf; sq=outputs/D20_selfq/shakespeare_cf.jsonl;; esac
  has $arm || continue
  echo ">>> $s registration"; python scripts/register_d12b.py $RT $REC --split_dir data/${s}_split \
      --d22_set outputs/D22_sets/$s.json --wrong_map outputs/D22_sets/${s}_wrongmap.json --selfq $sq \
      --out outputs/D12b_reg_$s || echo "!!! $s failed"
done
echo D12B_REG_DONE
