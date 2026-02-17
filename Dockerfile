# syntax=docker/dockerfile:1

# Stage 1: Build dependencies
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS build

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
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

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

# Copy ONNX models (baked into image, ~220MB)
COPY api/models/ ./models/

# Copy application code
COPY api/ ./

# Create data directory mount point
RUN mkdir -p /data

# Environment defaults
ENV DATA_DIR=/data
ENV PYTHONUNBUFFERED=1
ENV ENABLE_TATTOO_SIGNAL=auto

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
