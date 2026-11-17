# syntax=docker/dockerfile:1

# Stage 1: Build dependencies
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Create venv and install dependencies
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

COPY requirements.docker.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.docker.txt

# Stage 2: Runtime
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Copy venv from build stage
COPY --from=build /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# onnxruntime-gpu's CUDA provider must find this image's CUDA libraries. On a
# mismatch (e.g. a wheel built for CUDA 13) ONNX Runtime silently falls back
# to CPU at runtime, so fail the build instead. libcuda.so.1 comes from the
# host driver at runtime and is absent here, so it is ignored.
RUN missing=$(ldd /app/venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so \
        | grep "not found" | grep -v "libcuda.so.1" || true); \
    if [ -n "$missing" ]; then echo "onnxruntime CUDA provider cannot load:"; echo "$missing"; exit 1; fi

# Copy bundled model manifest
COPY api/models.json ./models.json

# Copy application code
COPY api/ ./

# Mount points. /root stays traversable so the non-root user can reach the
# InsightFace cache at its existing mount path (/root/.insightface).
RUN mkdir -p /data /media /root/.insightface && chmod 711 /root

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 755 /usr/local/bin/entrypoint.sh

# Environment defaults
ENV DATA_DIR=/data
ENV MEDIA_DIR=/media
ENV INSIGHTFACE_HOME=/root/.insightface
ENV PUID=99
ENV PGID=100
ENV UMASK=022
ENV PYTHONUNBUFFERED=1
ENV ENABLE_TATTOO_SIGNAL=auto

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

STOPSIGNAL SIGTERM

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
