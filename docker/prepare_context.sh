#!/bin/bash
# Assemble the Docker build context (a snapshot of the code + the two headline checkpoints).
#   bash docker/prepare_context.sh                       # CTX defaults to $PWD/../.. docker_build/ctx (see below)
#   CTX=/some/dir INCLUDE_CKPT=0 bash docker/prepare_context.sh   # slim: no checkpoints
# Layout of the project (the same on the cluster and on EC2):  <root>/KV_skill_v2  <root>/delta-skill  <root>/fla_kernels
# Writes only into $CTX. Never modifies the source tree. Nothing is uploaded.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/.." && pwd)}"
ROOT="${ROOT:-$(cd "$REPO/.." && pwd)}"
CTX="${CTX:-/scratch/drjieliu_root/drjieliu/rickyhan/docker_build/ctx}"
INCLUDE_CKPT="${INCLUDE_CKPT:-1}"
# the checkpoints baked in (paths relative to $REPO); override CKPT_SRC if they live elsewhere (e.g. on EC2)
CKPT_SRC="${CKPT_SRC:-$REPO}"
MAP27="${MAP27:-/scratch/drjieliu_root/drjieliu/rickyhan/xenc_cache/D12_maps/encB2_27b_9b_m.pt}"
CK9="outputs/DS_max8k/d13_step6000.pt"; CK27="outputs/D12b_max13k_s2/ckpt_5000.pt"; MAP9="outputs/D11_maps/encB5_9b_4b_post_xf01.pt"

echo "REPO=$REPO ROOT=$ROOT CTX=$CTX INCLUDE_CKPT=$INCLUDE_CKPT"
mkdir -p "$CTX"/{repo,delta-skill,fla_kernels,ckpt}
RS=(rsync -a --delete --exclude=__pycache__ --exclude='*.pyc' --exclude='*.pt' --exclude=.pytest_cache)
# 1. the repo: code + docs + launchers. NOT data/ outputs/ .git .deps .claude slurm/logs slurm/waiters (mounted / not needed)
"${RS[@]}" --exclude='/data' --exclude='/outputs' --exclude='/.git' --exclude='/.deps' --exclude='/.claude' --exclude='/libpod' \
    --exclude='/slurm/logs' --exclude='/slurm/waiters' --exclude='/docker/.ctx' "$REPO"/ "$CTX/repo/"
# 2. sibling packages, read-only copies (the originals are never touched). delta-skill: only the importable package.
"${RS[@]}" "$ROOT/delta-skill/deltaskill" "$CTX/delta-skill/"
"${RS[@]}" "$ROOT/fla_kernels"/ "$CTX/fla_kernels/"
# 3. provenance: git state of the repo (uncommitted changes are part of the snapshot!) + sha256 of every file copied
{
  echo "snapshot_time_utc: $(date -u +%FT%TZ)"
  echo "host: $(hostname)"
  echo "== KV_skill_v2 HEAD"; git -C "$REPO" rev-parse HEAD 2>&1 || true
  echo "== KV_skill_v2 branch/remote"; git -C "$REPO" branch --show-current 2>&1 || true; git -C "$REPO" remote -v 2>&1 | head -2 || true
  echo "== KV_skill_v2 git status --short (M = modified tracked, ?? = untracked: NOT in any commit, only in this snapshot)"
  git -C "$REPO" status --short 2>&1 || true
  echo "== delta-skill / fla_kernels: untracked in the parent repo, no commit hash exists; see MANIFEST.sha256"
  echo "fla_kernels: $(ls -d "$ROOT"/fla_kernels/flash_linear_attention-*.dist-info 2>/dev/null | xargs -n1 basename)"
} > "$CTX/repo/docker_GIT_STATE.txt"
git -C "$REPO" diff > "$CTX/repo/docker_GIT_DIFF_vs_HEAD.patch" 2>/dev/null || true
( cd "$CTX" && find repo delta-skill fla_kernels -type f ! -name 'MANIFEST.sha256' -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )
# image-relative copy (checked in the container with: cd /opt/kvskill && sha256sum -c ...); repo/ -> KV_skill_v2/
sed -e 's#  repo/#  KV_skill_v2/#' "$CTX/MANIFEST.sha256" > "$CTX/repo/docker_MANIFEST.sha256"
# 4. checkpoints (unchanged copies; fp32 as trained) + sha256 manifest
rm -rf "$CTX/ckpt"; mkdir -p "$CTX/ckpt"
if [ "$INCLUDE_CKPT" = 1 ]; then
  mkdir -p "$CTX/ckpt/DS_max8k" "$CTX/ckpt/D12b_max13k_s2" "$CTX/ckpt/maps"
  cp -p "$CKPT_SRC/$CK9"  "$CTX/ckpt/DS_max8k/d13_step6000.pt"
  cp -p "$CKPT_SRC/$CK27" "$CTX/ckpt/D12b_max13k_s2/ckpt_5000.pt"
  # the B (encoder->reader) maps these two interfaces are used with (eval needs them; small)
  cp -p "$CKPT_SRC/$MAP9" "$CTX/ckpt/maps/encB5_9b_4b_post_xf01.pt"
  cp -p "$MAP27"           "$CTX/ckpt/maps/encB2_27b_9b_m.pt"
  ( cd "$CTX/ckpt" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS )
  # checksum of the SOURCE files, to prove the copy is unchanged
  ( cd "$CKPT_SRC" && sha256sum "$CK9" "$CK27" "$MAP9" ) > "$CTX/ckpt/SOURCE_SHA256SUMS"; sha256sum "$MAP27" >> "$CTX/ckpt/SOURCE_SHA256SUMS"
  cat "$CTX/ckpt/SHA256SUMS"
fi
echo "INCLUDE_CKPT=$INCLUDE_CKPT: no checkpoints in this context (slim image)" > "$CTX/ckpt/README.txt"
[ "$INCLUDE_CKPT" = 1 ] && echo "checkpoints baked in under /opt/kvskill/ckpt; SHA256SUMS lists them" > "$CTX/ckpt/README.txt"
du -sh "$CTX"/*; echo "context ready: $CTX"
