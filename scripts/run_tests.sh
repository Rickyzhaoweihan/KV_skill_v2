#!/bin/bash
# Run the whole CPU suite (pytest + every tests/test_*.py as a script + the --tiny / --dry_run script
# paths). CPU only: no GPU, no model checkpoint loads (tokenizer-only for test_data). Exit != 0 on failure.
#   bash scripts/run_tests.sh            # everything
#   bash scripts/run_tests.sh -k library # extra args go to pytest only
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-/nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35/bin/python}
export HF_HOME=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home HF_HUB_CACHE=/nfs/turbo/umms-drjieliu/usr/rickyhan/hf_home/hub
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=$REPO/.deps:$REPO:$REPO/../delta-skill:$REPO/../fla_kernels${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/xenc_tests.XXXXXX"); trap 'rm -rf "$TMP"' EXIT

echo "== pytest"; $PY -m pytest tests -q "$@"
[ $# -gt 0 ] && exit 0
for t in tests/test_*.py; do echo "== $t"; $PY "$t" > "$TMP/log" 2>&1 || { cat "$TMP/log"; exit 1; }; tail -1 "$TMP/log"; done
echo "== scripts (--tiny / --dry_run)"
$PY scripts/encode_docs.py --tiny --cache_dir "$TMP/cache" | tail -1
$PY scripts/register_doc.py --tiny --steps 3 --out "$TMP/reg" | tail -1
$PY scripts/eval_doc.py --tiny --n 4 --interface "$TMP/reg/interface_best.pt" | tail -3
$PY scripts/ladder.py --tiny --out_dir "$TMP/ladder" | tail -1
$PY scripts/rl_doc_operator.py --tiny --steps 2 --out "$TMP/d4rl" | tail -1     # D4
$PY scripts/eval_doc.py --tiny --n 4 --interface "$TMP/d4rl/interface_ref.pt" --operator_dir "$TMP/d4rl/operators" \
    --conds kv,wrong_doc,null --out "$TMP/d4ev" | tail -1
$PY scripts/fit_state_maps.py --tiny --device cpu --out "$TMP/d10maps/maps.pt" | tail -1          # D10
$PY scripts/register_state.py --tiny --steps 2 --eval_every 2 --budget 3 --branches attn,state --attn_distill 1.0 --init "$TMP/d10maps/maps.pt" --out "$TMP/d10reg" | tail -1
$PY scripts/eval_state_xm.py --tiny --n 2 --budget 3 --interface "$TMP/d10reg/interface_best.pt" --out "$TMP/d10ev" | tail -1
$PY scripts/eval_state_xm.py --tiny --n 2 --budget 3 --encoder same --identity --branches attn,state --kl_text --js --out "$TMP/d10ev2" \
    --conds closed_book,text,kv,wrong_doc,null,prefix_full,prefix_state,prefix_kv,kv_kvprefix,wrong_kvprefix | tail -1
$PY scripts/register_doc.py --dry_run --split_dir data/quality_split --reader Qwen/Qwen3.5-9B \
    --encoder Qwen/Qwen3.5-2B --cache_dir "$TMP/empty_cache"
for f in slurm/*.sbatch; do bash -n "$f"; done; echo "slurm launchers: bash -n ok"
echo "ALL GREEN"
