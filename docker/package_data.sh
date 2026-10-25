#!/bin/bash
# RUN ON THE CLUSTER (or wherever the project lives): bundle the small data needed to reproduce headline evals into tar files
# under $OUT. Reads the project, writes only $OUT. Uploading to S3 is a separate step the USER runs (see README, section 4).
#   bash docker/package_data.sh [OUT=/scratch/.../docker_build/transfer]
# Produces (sizes measured 2026-10-25):
#   kvskill_data_eval.tar   data/*_split + data/docs + outputs/{D22,MCQ,OOD}_sets     0.56 GB   <- REQUIRED for any eval
#   kvskill_ref_results.tar reference eval jsons/summaries of the two headline ckpts  5 MB       <- to compare numbers
# (the two checkpoints + B maps are inside the image; they are NOT in these tars)
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${OUT:-/scratch/drjieliu_root/drjieliu/rickyhan/docker_build/transfer}"
mkdir -p "$OUT"; cd "$REPO"
tar --exclude='data/raw' --exclude='data/raw_downloads' --exclude='data/D15_selfq' -cf "$OUT/kvskill_data_eval.tar" \
    data outputs/D22_sets outputs/MCQ_sets outputs/OOD_sets
# reference rows: every eval_test.json/summary of the 9B max8k s6000 sweep and the 27B max13k c5000 judge, + the summaries
ls -d outputs/DSsw_m8k6000* outputs/D12b_max13k_c5000_* 2>/dev/null > "$OUT/ref_dirs.txt"
tar -cf "$OUT/kvskill_ref_results.tar" --exclude='*.pt' $(cat "$OUT/ref_dirs.txt") outputs/*_SUMMARY.md docs/HANDOFF.md docs/HANDOFF_LOG.md 2>/dev/null || true
( cd "$OUT" && sha256sum kvskill_data_eval.tar kvskill_ref_results.tar > SHA256SUMS && ls -lh *.tar && cat SHA256SUMS )
