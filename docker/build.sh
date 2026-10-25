#!/bin/bash
# Build the KV-Skill v2 image.  Local, builds nothing remote, pushes nothing.
#   On EC2 (docker installed):   bash docker/build.sh                       # -> kvskill-v2:cu130  (with the 2 checkpoints)
#                                INCLUDE_CKPT=0 bash docker/build.sh        # slim image, no checkpoints (tag :cu130-slim)
#   On the UMich cluster (no docker daemon): BUILDER=buildah bash docker/build.sh   [SIF=1 to also make a .sif]
# Env: CTX (build context dir), TAG, INCLUDE_CKPT (1|0), BUILDER (docker|podman|buildah; auto-detected), SIF (0|1).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export INCLUDE_CKPT="${INCLUDE_CKPT:-1}"
export CTX="${CTX:-$HOME/kvskill_ctx}"                       # the cluster default is overridden by the caller (see README)
TAG="${TAG:-kvskill-v2:cu130$([ "$INCLUDE_CKPT" = 1 ] || echo -slim)}"
BUILDER="${BUILDER:-$(for b in docker podman buildah; do command -v $b >/dev/null && { echo $b; break; }; done)}"
[ -n "$BUILDER" ] || { echo "no docker/podman/buildah found"; exit 1; }
echo "builder=$BUILDER tag=$TAG ctx=$CTX INCLUDE_CKPT=$INCLUDE_CKPT"

bash "$HERE/prepare_context.sh"                            # snapshot code (+ checkpoints) into $CTX

case "$BUILDER" in
  docker|podman)
    DOCKER_BUILDKIT=1 $BUILDER build -f "$HERE/Dockerfile" --build-arg INCLUDE_CKPT="$INCLUDE_CKPT" -t "$TAG" "$CTX" ;;
  buildah)
    # rootless without /etc/subuid: vfs driver + ignore_chown_errors + --isolation rootless (chroot fails on setgroups); storage on a LOCAL disk (NFS rejects it).
    BROOT="${BROOT:-/tmp/$USER/buildah_storage}"; mkdir -p "$BROOT" "${BRUN:=/tmp/$USER/buildah_run}"
    export TMPDIR="${BTMP:-/tmp/$USER/buildah_tmp}"; mkdir -p "$TMPDIR"   # must be local disk, not NFS
    buildah --storage-driver vfs --storage-opt vfs.ignore_chown_errors=true --root "$BROOT" --runroot "$BRUN" \
        build --isolation "${BISO:-rootless}" --format docker -f "$HERE/Dockerfile" --build-arg INCLUDE_CKPT="$INCLUDE_CKPT" -t "$TAG" "$CTX"
    if [ "${SIF:-0}" = 1 ]; then
      OUT="${OUT:-$CTX/..}"; ARCH="$OUT/$(echo "$TAG" | tr ':/' '__').oci.tar"
      buildah --storage-driver vfs --storage-opt vfs.ignore_chown_errors=true --root "$BROOT" --runroot "$BRUN" push "$TAG" "oci-archive:$ARCH"
      SING=$(command -v singularity; ls /opt/singularity/*/bin/singularity 2>/dev/null | tail -1); SING=$(echo "$SING" | grep -v "^/usr/local/bin" | head -1)   # cluster: stub in /usr/local/bin, real binary under /opt/singularity
      "${SING:-singularity}" build --force "${ARCH%.oci.tar}.sif" "oci-archive:$ARCH"
      ls -lh "${ARCH%.oci.tar}.sif"
    fi ;;
esac
echo "built $TAG"
