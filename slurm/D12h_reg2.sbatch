#!/bin/bash
#SBATCH --job-name=D12h_reg2
#SBATCH --account=drjieliu
#SBATCH --partition=drjieliu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150G
#SBATCH --time=40:00:00
#SBATCH --chdir=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
#SBATCH --output=slurm/logs/D12h_reg2_%j.out
# D12h: 27B registration HEADROOM on the sets where 9B registration helped (D20/D22), base outputs/D12p_4b_s2/d13_best.pt
# (4B encoder, reader layers 3..31), live d12p runtime (27B + 4B on one H100, mem_safe). Arms (each matched to its 9B row):
#   scf_rf   Shakespeare-CF, reader-free (selfq by 4B), dA,dB,bias,gate, max_tokens 60000   (9B: D20_4b_scf_rf +.146)
#   qas_rf   Qasper D22 set, reader-free (span selfq by 4B), dA,gate                        (9B text-teacher: +.054)
#   clip_rf  CLIPPER D22 set, reader-free (selfq by 4B), dA,gate, max_tokens 128000, no text cond (27B 128k text = OOM risk)
#   qas_tt   Qasper D22 set, 27B text teacher (selfq D22_selfq/qasper.jsonl), dA,gate       (matches D22_qasper)
REPO=/nfs/turbo/umms-drjieliu/usr/rickyhan/KV_skill/KV_skill_v2
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home HF_HUB_CACHE=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home/hub
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export PYTHONPATH=$REPO:$REPO/../delta-skill:$REPO/../fla_kernels:$REPO/scripts${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=/scratch/drjieliu_root/drjieliu/rickyhan/hf_caches/tmp
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /sw/pkgs/arc/python3.9-anaconda/2021.11/etc/profile.d/conda.sh
conda activate /nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35
set -uo pipefail
cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
ARMS=${ARMS:-scf_rf,qas_rf,clip_rf,qas_tt}
has() { [[ ",$ARMS," == *",$1,"* ]]; }
RT="--runtime d12p --reader Qwen/Qwen3.5-27B --encoder Qwen/Qwen3.5-4B --B outputs/D10_maps/encB8_27b_4b_post.pt \
    --layers 3,7,11,15,19,23,27,31 --interface outputs/D12p_4b_s2/d13_best.pt --reader_mem 72GiB --enc_device cuda:0 \
    --itf_device cuda:0 --mem_safe 1"
REC="--rank 16 --lr 1e-3 --shuffle 2 --warmup 25 --holdout 0.2 --budgets 0,50,100,200 --split test"
SQ=outputs/D20_selfq; S22=outputs/D22_selfq
if has scf_rf; then
  echo ">>> SCF reader-free"; python scripts/register_d20.py $RT $REC --parts dA,dB,bias,gate --split_dir data/shakespeare_cf_split \
      --n 0 --max_tokens 60000 --selfq $SQ/shakespeare_cf_by4b.jsonl --teacher none --out outputs/D12h_reg_scf_rf || echo "!!! scf_rf failed"
fi
if has qas_rf; then
  [ -s $S22/qasper_by4b.jsonl ] || { echo ">>> qasper selfq by 4B"; python scripts/d22_selfq.py --split_dir data/qasper_split \
      --docs file:outputs/D22_sets/qasper.json --templates span,mix,mcq --max_doc_tokens 16000 --reader Qwen/Qwen3.5-4B \
      --out $S22/qasper_by4b.jsonl; }
  echo ">>> Qasper reader-free"; python scripts/register_d22.py --d22_set outputs/D22_sets/qasper.json \
      --wrong_map outputs/D22_sets/qasper_wrongmap.json $RT $REC --parts dA,gate --split_dir data/qasper_split --n 0 \
      --max_tokens 16000 --selfq $S22/qasper_by4b.jsonl --teacher none --out outputs/D12h_reg_qas_rf || echo "!!! qas_rf failed"
fi
if has clip_rf; then
  [ -s $S22/clipper_by4b.jsonl ] || { echo ">>> clipper selfq by 4B"; python scripts/d22_selfq.py --split_dir data/clipper_split \
      --docs file:outputs/D22_sets/clipper.json --templates nli,mcq,mix --max_doc_tokens 128000 --reader Qwen/Qwen3.5-4B \
      --out $S22/clipper_by4b.jsonl; }
  echo ">>> CLIPPER reader-free"; python scripts/register_d22.py --d22_set outputs/D22_sets/clipper.json \
      --wrong_map outputs/D22_sets/clipper_wrongmap.json $RT $REC --parts dA,gate --split_dir data/clipper_split --n 0 \
      --max_tokens 128000 --selfq $S22/clipper_by4b.jsonl --teacher none --conds closed_tail,kv,wrong_doc \
      --out outputs/D12h_reg_clip_rf || echo "!!! clip_rf failed"
fi
if has qas_tt; then
  echo ">>> Qasper 27B text teacher"; python scripts/register_d22.py --d22_set outputs/D22_sets/qasper.json \
      --wrong_map outputs/D22_sets/qasper_wrongmap.json $RT $REC --parts dA,gate --split_dir data/qasper_split --n 0 \
      --max_tokens 16000 --selfq $S22/qasper.jsonl --teacher text --out outputs/D12h_reg_qas_tt || echo "!!! qas_tt failed"
fi
echo D12h_REG2_DONE
