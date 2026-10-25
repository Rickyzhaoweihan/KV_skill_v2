# syntax=docker/dockerfile:1
# KV-Skill v2 inference/eval (+ training) image.   Build with docker/build.sh (it prepares the context first).
#   docker build -f docker/Dockerfile --build-arg INCLUDE_CKPT=1 -t kvskill-v2:cu130 "$CTX"     # CTX = output of prepare_context.sh
#
# Design: the image carries ONLY code, python env, and (optionally) the two headline interface checkpoints.
# Backbone weights (Qwen3.5-4B/9B/27B, bge, llmlingua-2), data/, outputs/ and the operator caches are MOUNTED.
#
# CUDA: torch 2.12.1 is the cu130 wheel; CUDA 13.0 runtime + cuDNN 9.20 libraries come from the pip nvidia-*-cu13 wheels
# inside the python env (exactly like the cluster env), so the base image only needs the NVIDIA driver from the host
# (driver >= 580.65, supplied through the NVIDIA container toolkit).
ARG BASE_IMAGE=nvidia/cuda:13.0.2-base-ubuntu24.04
ARG INCLUDE_CKPT=1

# ---- checkpoint payload stages (selected by INCLUDE_CKPT=1|0; the unused stage is not built) ----------------------------
FROM scratch AS ckpt_1
COPY ckpt/ /
FROM scratch AS ckpt_0
COPY ckpt/README.txt /README.txt
FROM ckpt_${INCLUDE_CKPT} AS ckpt_sel

# ---- main image -----------------------------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS main
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
# gcc: triton JIT-compiles small C launcher stubs at run time (the cluster used gcc 11 from a module; Ubuntu 24.04 has gcc 13).
# bzip2/curl/ca-certificates: fetch micromamba. rsync/git/less/vim-tiny/procps: working-tool conveniences, not used by the code.
RUN printf 'APT::Sandbox::User "root";\n' > /etc/apt/apt.conf.d/99nosandbox \
 && apt-get update \
 && apt-get install -y --no-install-recommends gcc libc6-dev bzip2 curl ca-certificates git rsync less procps \
 && rm -rf /var/lib/apt/lists/*

# python 3.11.15 + the conda-forge pieces of the cluster env, exact name=version=build (see conda-env.lock)
ARG MICROMAMBA_URL=https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.3.2-0.tar.bz2
COPY repo/docker/conda-env.lock /tmp/conda-env.lock
RUN mkdir -p /opt/mm && curl -fsSL "$MICROMAMBA_URL" | tar -xj -C /opt/mm bin/micromamba \
 && /opt/mm/bin/micromamba create -y -p /opt/venv -c conda-forge --override-channels --file /tmp/conda-env.lock \
 && /opt/mm/bin/micromamba clean -a -y && rm -rf /opt/mm /tmp/conda-env.lock
ENV PATH=/opt/venv/bin:$PATH

# python wheels: exact pins. torch's nvidia-* CUDA 13.0 wheels are in the lock. --no-deps keeps pip from "fixing" the lock.
COPY repo/docker/requirements.lock repo/docker/requirements.nodeps.lock /tmp/
RUN python -m pip install --no-cache-dir --no-deps -r /tmp/requirements.lock \
 && python -m pip install --no-cache-dir --no-deps --require-hashes -r /tmp/requirements.nodeps.lock \
 && python -m pip check || true

# code at a pinned snapshot (see /opt/kvskill/KV_skill_v2/docker_GIT_STATE.txt + docker_MANIFEST.sha256)
COPY repo/         /opt/kvskill/KV_skill_v2/
COPY delta-skill/  /opt/kvskill/delta-skill/
COPY fla_kernels/  /opt/kvskill/fla_kernels/
# headline interface checkpoints + B maps (unchanged fp32 copies) and their SHA256SUMS; empty README in the slim image
COPY --from=ckpt_sel / /opt/kvskill/ckpt/

# Runtime layout. Mount the big stuff here (see docker/run_eval_example.sh):
#   /data/hf_home            HF cache (Qwen3.5-4B/9B/27B, bge, llmlingua-2)      read-only is fine
#   .../KV_skill_v2/data     materialised splits + docs                          read-only is fine
#   .../KV_skill_v2/outputs  D22_sets/MCQ_sets/OOD_sets json + where results are written   read-write
#   /data/xenc_cache         operator caches (optional; recomputed when missing for the 9B path)
ENV KV_ROOT=/opt/kvskill \
    REPO=/opt/kvskill/KV_skill_v2 \
    KV_CKPT=/opt/kvskill/ckpt \
    PYTHONPATH=/opt/kvskill/KV_skill_v2:/opt/kvskill/delta-skill:/opt/kvskill/fla_kernels:/opt/kvskill/KV_skill_v2/scripts \
    HF_HOME=/data/hf_home HF_HUB_CACHE=/data/hf_home/hub HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
    XENC_CACHE=/data/xenc_cache \
    TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRITON_CACHE_DIR=/var/tmp/triton_cache TMPDIR=/var/tmp \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN mkdir -p /data/hf_home/hub /data/xenc_cache /var/tmp/triton_cache && chmod -R a+rwX /data /var/tmp \
 && python -c "import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)"
WORKDIR /opt/kvskill/KV_skill_v2
LABEL org.opencontainers.image.title="kvskill-v2" \
      org.opencontainers.image.description="KV-Skill v2 (small encoder, large reader): eval/baselines/training env, torch 2.12.1+cu130, transformers 5.13.0"
CMD ["bash"]
