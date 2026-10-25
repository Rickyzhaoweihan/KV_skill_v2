# Environment notes (what the image reproduces, and what it deliberately does not)

Source env: `/nfs/turbo/umms-drjieliu/usr/rickyhan/envs/qwen35` (conda-forge python + pip wheels), dumped 2026-10-25.

| item | value |
|---|---|
| Python | 3.11.15 (conda-forge `h7508c33_1_cpython`; exact specs in `conda-env.lock`) |
| torch | 2.12.1+cu130 (PyPI wheel, built with GCC 13.3, MKL-DNN 3.11.2) |
| CUDA runtime | 13.0 (`torch.version.cuda`), from pip wheels `nvidia-cuda-runtime 13.0.96`, `nvidia-cublas 13.1.1.3`, `nvidia-nccl-cu13 2.29.7`, ... `cuda-toolkit 13.0.2` |
| cuDNN | 9.20.0 (`torch.backends.cudnn.version() == 92000`, wheel `nvidia-cudnn-cu13 9.20.0.48`) |
| triton | 3.7.1 (needs a C compiler at run time: the image installs gcc) |
| transformers | 5.13.0, tokenizers 0.22.2, accelerate 1.14.0, safetensors 0.8.0, huggingface_hub 1.22.0 |
| numpy | 2.4.6 |
| llmlingua | 0.2.2, installed `--no-deps` (nltk / tiktoken absent on purpose; `xenc/lingua.py` registers placeholder modules) |
| flash-linear-attention | 0.5.2, **vendored** at `../fla_kernels` (pure python, plus its own einops 0.8.2); found through `PYTHONPATH` / `xenc/__init__.py` |
| deltaskill | `../delta-skill/deltaskill` (21 py files), copied read-only; untracked in git |
| pytest | 9.1.1 (cluster: vendored `.deps/`; image: pip, same version) |
| NOT installed (as on the cluster) | `causal_conv1d`, `flash-attn`, `vllm`, `pandas`, `pyarrow`, `nltk`, `tiktoken`, `rank_bm25` (BM25 is a 30-line in-repo class), `sentence-transformers` (bge runs through plain `AutoModel`) |

Files: `requirements.lock` (runtime pins, 65 lines), `requirements.nodeps.lock` (llmlingua + sha256), `conda-env.lock`,
`pip_freeze.full.txt` (the untouched `pip freeze`, 92 lines, for audit). The full freeze also lists the alfworld / textworld /
spacy / pydantic / omegaconf group and the editable parent `kvskill` package: those are another project's dependencies, nothing in
`xenc/`, `scripts/`, `tests/` imports them, so they are left out. `pip check` result in the image is printed at build time.

## causal_conv1d: NOT installed (decision)
Qwen3.5's Gated-DeltaNet layers use `causal_conv1d` (CUDA kernel) when importable and otherwise a torch `conv1d` fallback; the
warning "fast path is not available ... causal-conv1d" is benign. Every number in `outputs/` and in HANDOFF was produced WITHOUT it.
The CUDA kernel and the fallback differ at the bf16 rounding level, and greedy decoding is token-exact sensitive (H100 vs L40S already
moves scores by .01-.03), so installing it would silently change the numerics against all existing rows. It also needs nvcc and a
source build against torch 2.12. Keep it out; if you ever add it, treat every row as a new GPU/kernel type and re-run the matching text row.
The fla kernels (chunked delta rule) ARE used, exactly as on the cluster.

## Env vars the code reads (set by the Dockerfile; override with `-e`)
`HF_HOME=/data/hf_home`, `HF_HUB_CACHE=$HF_HOME/hub`, `HF_HUB_OFFLINE=1` (downloads only through `download_models.sh`, which sets it to 0),
`HF_HUB_DISABLE_XET=1`, `XENC_CACHE=/data/xenc_cache`, `PYTHONPATH=$REPO:/opt/kvskill/delta-skill:/opt/kvskill/fla_kernels:$REPO/scripts`,
`TOKENIZERS_PARALLELISM=false`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the cluster sbatch files all set it; it affects OOM behaviour),
`TRITON_CACHE_DIR=/var/tmp/triton_cache`. `xenc/__init__.py` only falls back to the cluster's turbo HF path if that directory exists
and the env var is unset, so the container never touches it.

## Things in the repo that are cluster-specific (left untouched, worked around)
* `scripts/run_tests.sh` hard-codes `PY=.../envs/qwen35/bin/python` and `HF_HOME=/nfs/turbo/...`. `docker/smoke_test.sh` runs a sed-patched copy
  in a temp dir (REPO, PY, HF_HOME substituted); the repo file is not changed.
* `slurm/*.sbatch` hard-code `/nfs/turbo/...`, `/scratch/...` and `conda activate`. They are for the cluster; inside the container run the
  `python scripts/...` command lines directly (see `run_eval_example.sh`). `bash -n` on them is part of run_tests.
* `slurm/submit_when_free.sh` and the priority/cap machinery do not apply on EC2.
