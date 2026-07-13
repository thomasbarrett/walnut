# syntax=docker/dockerfile:1
#
# Multi-stage build for the walnut inference server.
#
# Both stages share the same python base image so the virtualenv built in the
# builder stage keeps a valid interpreter symlink when copied into the runtime
# stage.
#
# NOTE: torch is pinned to the CPU wheel on Linux via `[tool.uv.sources]` in
# pyproject.toml, keeping this image small. For GPU inference, point torch at a
# CUDA index and base the runtime stage on an `nvidia/cuda` image instead.

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.10.10

# ---- Builder: resolve dependencies into /app/.venv ----------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

# Torch variant to install: "cpu" (default) or "cu130" (CUDA 13.0). The CUDA
# wheels bundle their own runtime libs, so both variants use this slim base;
# a CUDA image just needs the host's NVIDIA driver + container toolkit at runtime.
ARG TORCH_EXTRA=cpu

# Pin uv by copying it from its published image.
COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install third-party dependencies first so this layer is cached until the
# lockfile changes (the project source changes far more often).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --extra ${TORCH_EXTRA}

# Install the project itself as a non-editable wheel.
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra ${TORCH_EXTRA}

# ---- Runtime: minimal image with just the venv --------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

# Run as an unprivileged user with a fixed UID/GID so Kubernetes can verify
# `runAsNonRoot` / pin `runAsUser`.
RUN groupadd --system --gid 10001 walnut \
    && useradd --system --uid 10001 --gid walnut --home-dir /app --no-create-home walnut

WORKDIR /app
COPY --from=builder --chown=walnut:walnut /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    WALNUT_HOST=0.0.0.0 \
    WALNUT_PORT=8000

USER walnut

EXPOSE 8000

# `walnut` is the entry point; supply a command + model, e.g.
#   docker run -p 8000:8000 <image> serve meta-llama/Llama-3.1-8B-Instruct
ENTRYPOINT ["walnut"]
CMD ["--help"]
