# syntax=docker/dockerfile:1
###############################################################################
# Stable Audio 3 — Gradio demo
#
#   docker build --secret id=hf_token,env=HF_TOKEN -t stable-audio-3 .
#   docker run --gpus all -p 7860:7860 stable-audio-3
#
# Prefer the wrapper: ./deploy.sh {build|up|down|logs|status|doctor|clean}
###############################################################################

###############################################################################
# Stage 1 — model weights (~17 GB)
#
# A separate stage so edits to the dependency layer in stage 2 never
# invalidate the checkpoint download. All three SA3 repos are gated (license
# must be accepted on huggingface.co), so preloading needs a token:
# ./deploy.sh build feeds $HF_TOKEN (or ~/.cache/huggingface/token) in as a
# BuildKit secret, which never lands in the image layers. Build with
# --build-arg PRELOAD_MODELS=0 to skip and let the app download on first boot.
###############################################################################
FROM nvidia/cuda:13.1.2-runtime-ubuntu24.04 AS model-cache

ARG PRELOAD_MODELS=1
ARG PRELOAD_REPOS="stabilityai/stable-audio-3-medium stabilityai/stable-audio-3-small-music stabilityai/stable-audio-3-small-sfx"

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/preload-venv \
    && /opt/preload-venv/bin/pip install --no-cache-dir -q huggingface_hub

RUN --mount=type=secret,id=hf_token \
    mkdir -p /cache/huggingface \
    && if [ "$PRELOAD_MODELS" = "1" ]; then \
           export HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
           for repo in $PRELOAD_REPOS; do \
               echo "==> preloading $repo"; \
               HF_HOME=/cache/huggingface /opt/preload-venv/bin/python \
                   -c "from huggingface_hub import snapshot_download; snapshot_download('${repo}')" \
               || { echo "ERROR: failed to download ${repo}"; \
                    echo "  1. accept the license at https://huggingface.co/stabilityai (repos are gated), and"; \
                    echo "  2. pass a token:  HF_TOKEN=hf_... ./deploy.sh build"; \
                    exit 1; }; \
           done; \
       else \
           echo "PRELOAD_MODELS=0 — skipping model preload (weights download at first boot)"; \
       fi \
    && chown -R 1000:1000 /cache

###############################################################################
# Stage 2 — application
###############################################################################
# Pinned to CUDA 13.1.2 (not 13.2): the nvidia/cuda images bake in
# NVIDIA_REQUIRE_CUDA, which nvidia-container-cli checks against the host
# driver — a driver exposing CUDA 13.1 refuses to start a 13.2 image. The
# image's CUDA libs are irrelevant to the app anyway: torch brings its own
# CUDA runtime via pip wheels.
FROM nvidia/cuda:13.1.2-runtime-ubuntu24.04

# HF_HOME holds the baked model weights (see stage 1). Gradio must bind
# 0.0.0.0 to be reachable from outside the container.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_DISABLE_TELEMETRY=1

# Ubuntu 24.04 marks the system Python as externally managed (PEP 668), so
# everything goes into a venv. ffmpeg backs gradio's audio post-processing,
# libsndfile backs soundfile, build-essential is there in case a dep needs to
# compile from sdist.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip build-essential \
        ffmpeg libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && ln -sf /opt/venv/bin/python /usr/local/bin/python

WORKDIR /app

# --- dependencies (cached layer) -------------------------------------------
# The torch trio must share one CUDA version: torchaudio stops at 2.11.0 and
# was never built for CUDA 13.2, so letting requirements.txt's cu132 index
# resolve torch==2.14.1+cu132 makes torchaudio's strict _check_cuda_version()
# raise at import. Install the trio first as +cu130 builds (same CUDA as
# torchaudio's PyPI/default-index wheels; covers sm_86 and sm_120), then
# requirements.txt sees them already satisfied and leaves them alone.
# gradio + spaces ship with the HF Spaces SDK, not requirements.txt.
# stable-audio-tools is installed with --no-deps because it strict-pins
# torch==2.7.1 (no sm_120 kernels); its transitive deps are the ones listed
# in requirements.txt. 0.0.20 is required by app.py's inpainting/sampler
# kwargs but declares Requires-Python <3.11 (a proxy for its stale dep
# pins), hence --ignore-requires-python; the code itself runs fine on 3.12.
ARG GRADIO_VERSION=6.14.0
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl \
        torch==2.14.1+cu130 torchvision==0.29.1+cu130 torchaudio==2.11.0+cu130 \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir "gradio==${GRADIO_VERSION}" spaces \
    && python -m pip install --no-cache-dir --no-deps --ignore-requires-python \
        stable-audio-tools==0.0.20

# Triton JIT-compiles CUDA kernel launchers at inference time and #include <Python.h>; without the headers that compile fails mid-generation.
# Own layer on purpose: headers are cheap, the pip layer above is not.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# --- baked model weights ----------------------------------------------------
# Weights are already owned by uid 1000 (= the base image's 'ubuntu' user),
# so only /app needs chowning below.
COPY --from=model-cache /cache/huggingface /cache/huggingface

# With weights baked in, never phone home at startup. Note this also
# applies when PRELOAD_MODELS=0, where it would break first boot — override
# at runtime with -e HF_HUB_OFFLINE=0 in that case.
ARG PRELOAD_MODELS=1
ENV HF_HUB_OFFLINE=$PRELOAD_MODELS

# 12 GB cards run this app near the VRAM limit — avoid allocator
# fragmentation (torch recommends this itself when hitting OOM).
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Only preload the medium checkpoint (~5.2 GB on GPU) by default; all three
# together leave too little activation headroom on 12 GB cards. Override at
# runtime, e.g. -e SA3_VARIANTS=all or -e SA3_VARIANTS=small-music.
ENV SA3_VARIANTS=medium

# --- application ------------------------------------------------------------
COPY app.py ./
RUN mkdir -p /cache/huggingface && chown -R ubuntu:ubuntu /app && chown ubuntu:ubuntu /cache /cache/huggingface
USER ubuntu

EXPOSE 7860

# Weights are baked in, so startup only loads models from disk — still a
# few minutes for ~17 GB, hence the long start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20m --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${GRADIO_SERVER_PORT}/" >/dev/null || exit 1

CMD ["python", "app.py"]
