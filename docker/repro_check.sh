#!/bin/bash
# GPU reproduction check INSIDE the container: re-decode the first K items of a finished cluster row with the baked-in checkpoint and
# compare the greedy generations with the cluster's eval_test.json (needs kvskill_ref_results.tar + kvskill_data_eval.tar extracted
# under the mounted outputs/ and data/). Reference: 9B reader / 4B encoder, d13_step6000, 1 GPU, unregistered kv + wrong_doc, MCQ hallu set,
# produced on an L40S. Identical generations on an L40S = same environment, same numbers.  On another GPU type (H100, A10G, ...)
# differences of a few items are expected (bf16 kernels); the script then reports and exits 0.
#   bash docker/repro_check.sh            # K=5 items, ~3 min on an L40S
set -uo pipefail
cd "${REPO:-/opt/kvskill/KV_skill_v2}"
K="${K:-5}"; S="${SET:-mcq_hallu}"; REF="outputs/DSsw_m8k6000_$S/eval_test.json"; OUT="${OUT:-outputs/DOCKER_repro_$S}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || { echo "no GPU visible (docker: --gpus all, singularity: --nv)"; exit 2; }
python -c "import torch; assert torch.cuda.is_available(), 'torch sees no GPU'; print('torch cuda ok:', torch.cuda.get_device_name(0), torch.version.cuda)" || exit 2
[ -s "$REF" ] || { echo "missing reference $REF (extract kvskill_ref_results.tar into outputs/)"; exit 3; }
mkdir -p "$OUT"
python scripts/ds_verify_dev.py make --ref "$REF" --set outputs/MCQ_sets/$S.json --k "$K" --out "$OUT/set.json" || exit 3
SPLIT_DIR=$(python -I -c "import json;print(json.load(open('outputs/MCQ_sets/$S.json'))['split_dir'])")
MAXTOK=$(python -I -c "import json;print(json.load(open('outputs/MCQ_sets/$S.json'))['max_tokens'])")
for try in $(seq 1 10); do
  python scripts/eval_d13.py --reader Qwen/Qwen3.5-9B --encoder Qwen/Qwen3.5-4B \
     --B /opt/kvskill/ckpt/maps/encB5_9b_4b_post_xf01.pt --layers 3,7,11,15,19 --interface /opt/kvskill/ckpt/DS_max8k/d13_step6000.pt \
     --device cuda:0 --conds kv,wrong_doc --budget 32 --oom_skip 1 --d22_set "$OUT/set.json" --wrong_map outputs/MCQ_sets/${S}_wrongmap.json \
     --split test --n 0 --split_dir "$SPLIT_DIR" --max_tokens "$MAXTOK" --out "$OUT"
  rc=$?; [ $rc -eq 75 ] || break; echo "... exit 75 (OOM resume), rerun"
done
[ $rc -eq 0 ] || { echo "eval failed rc=$rc"; exit 4; }
python scripts/ds_verify_dev.py compare --ref "$REF" --new "$OUT/eval_test.json"; crc=$?
if [ $crc -eq 0 ]; then echo "REPRO_IDENTICAL: all $K items decode identically to the cluster rows"
else echo "REPRO_DIFFERS: some generations differ from the cluster rows (expected only across GPU types; compare the GPU name above with the cluster's L40S)"; fi
exit 0
