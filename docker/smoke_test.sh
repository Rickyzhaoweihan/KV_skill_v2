#!/bin/bash
# CPU smoke test inside the image: imports + versions + checkpoint checksums + the repo's own scripts/run_tests.sh
# (pytest + every tests/test_*.py + the --tiny/--dry_run paths + bash -n on slurm launchers). No GPU, no model weights needed,
# except tokenizers (tests tokenize with Qwen3.5-2B/9B): mount the HF cache; and 2 tests (test_d12b, test_f1) read data/quality_split
# and data/contractnli_split: mount the data volume too (see run_eval_example.sh), otherwise they fail with FileNotFoundError.
#   docker run --rm -v /data/hf_home:/data/hf_home:ro -v /data/kvskill/data:/opt/kvskill/KV_skill_v2/data:ro \
#       -v /data/kvskill/outputs:/opt/kvskill/KV_skill_v2/outputs kvskill-v2:cu130 bash docker/smoke_test.sh   # full
#   QUICK=1 ... bash docker/smoke_test.sh                                                              # imports/checksums only
#   singularity exec --bind /data/hf_home:/data/hf_home:ro kvskill-v2_cu130.sif bash /opt/kvskill/KV_skill_v2/docker/smoke_test.sh
set -euo pipefail
REPO="${REPO:-/opt/kvskill/KV_skill_v2}"; cd "$REPO"
export CUDA_VISIBLE_DEVICES=""
echo "== versions"
python - <<'PY'
import sys, torch, transformers, triton, accelerate
print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(),
      "transformers", transformers.__version__, "triton", triton.__version__)
import xenc, deltaskill, fla
print("xenc", xenc.__file__); print("deltaskill", deltaskill.__file__); print("fla", fla.__file__)
assert torch.__version__.startswith("2.12.1") and transformers.__version__ == "5.13.0", "version drift"
from xenc import d13_read, d12p_read, doc_tasks, backbone, textbudget, lingua, kvcomp, cad  # noqa: F401
lingua._ensure_importable()    # llmlingua imports nltk/tiktoken at module top; xenc registers placeholders (as on the cluster)
import llmlingua; print('llmlingua', llmlingua.__file__)
print("imports ok: d13_read d12p_read doc_tasks backbone textbudget lingua kvcomp cad")
PY
echo "== checkpoints"
if [ -s "${KV_CKPT:-/opt/kvskill/ckpt}/SHA256SUMS" ]; then (cd "${KV_CKPT:-/opt/kvskill/ckpt}" && sha256sum -c SHA256SUMS); else echo "(slim image: no checkpoints)"; fi
echo "== code manifest (sha256 of every file copied into the image)"
(cd "$(dirname "$REPO")" && sha256sum -c --quiet "$REPO/docker_MANIFEST.sha256" 2>&1 | head -5) && echo "code manifest OK" || echo "WARNING: files differ from the snapshot manifest (edited inside the container?)"
[ "${QUICK:-0}" = 1 ] && { echo "QUICK smoke ok"; exit 0; }
echo "== scripts/run_tests.sh (container-path copy: the original hard-codes the cluster paths PY / HF_HOME / REPO env)"
T="$(mktemp -d "${TMPDIR:-/tmp}/kvsmoke.XXXXXX")"; trap 'rm -rf "$T"' EXIT
sed -e "s#^REPO=.*#REPO=$REPO#" \
    -e "s#^PY=.*#PY=\${PY:-python}#" \
    -e "s#^export HF_HOME=.*#export HF_HOME=\${HF_HOME:-/data/hf_home} HF_HUB_CACHE=\${HF_HUB_CACHE:-/data/hf_home/hub}#" \
    scripts/run_tests.sh > "$T/run_tests.sh"
bash "$T/run_tests.sh"
