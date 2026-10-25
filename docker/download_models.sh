#!/bin/bash
# Download exactly the HF models the code loads, at the SAME revisions as the cluster cache, into a mounted HF_HOME.
#   Inside the container:  docker run --rm -v /data/hf_home:/data/hf_home -e HF_HUB_OFFLINE=0 kvskill-v2:cu130 bash docker/download_models.sh
#   On the host (needs `pip install huggingface_hub`): HF_HOME=/data/hf_home bash docker/download_models.sh
#   Subset:  MODELS="4b 9b" bash docker/download_models.sh        (keys: 4b 9b 27b 2b bge lingua2)
# Needs outbound internet, nothing else. No token is needed for these public repos (export HF_TOKEN yourself if you hit rate limits;
# it is never written anywhere). Sizes are what the cluster cache holds on disk.
set -euo pipefail
export HF_HOME="${HF_HOME:-/data/hf_home}" HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}" HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
MODELS="${MODELS:-4b 9b 27b bge lingua2}"            # default = everything the headline + baseline paths need (2b only for legacy tests)
declare -A REPO=( [4b]=Qwen/Qwen3.5-4B [9b]=Qwen/Qwen3.5-9B [27b]=Qwen/Qwen3.5-27B [2b]=Qwen/Qwen3.5-2B
                  [bge]=BAAI/bge-base-en-v1.5 [lingua2]=microsoft/llmlingua-2-xlm-roberta-large-meetingbank )
declare -A REV=(  [4b]=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a [9b]=c202236235762e1c871ad0ccb60c8ee5ba337b9a
                  [27b]=fc05daec18b0a78c049392ed2e771dde82bdf654 [2b]=15852e8c16360a2fea060d615a32b45270f8a8fc
                  [bge]=a5beb1e3e68b9ab74eb54cfd186867f64f240e1a [lingua2]=ebaba9b0e874dadd3003ffcff828e4397e568089 )
declare -A SIZE=( [4b]=11G [9b]=22G [27b]=62G [2b]=5G [bge]=0.4G [lingua2]=2.4G )
echo "HF_HOME=$HF_HOME"; df -h "$HF_HOME" 2>/dev/null | tail -1 || true
PYBIN="$(command -v python || command -v python3)"
for k in $MODELS; do
  echo ">>> $k  ${REPO[$k]} @ ${REV[$k]}  (~${SIZE[$k]})"
  "$PYBIN" - "${REPO[$k]}" "${REV[$k]}" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev = sys.argv[1:3]
# *.msgpack / *.h5 / *.onnx etc. are not used by transformers here; keep safetensors + tokenizer/config files
p = snapshot_download(repo_id=repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"])
print("ok", p)
PY
done
echo "done. A fresh snapshot is named by the revision, so the on-disk layout equals the cluster cache (models--ORG--NAME/snapshots/<rev>)."
echo "Also verify: du -sh $HF_HUB_CACHE/models--*"
