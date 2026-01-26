# Stage 1: Build dependencies
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Create venv and install dependencies
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install PyTorch with CUDA 12.4 (closest available to 12.8)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# Stage 2: Runtime
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Copy venv from builder
COPY --from=base /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Copy application code
COPY api/ ./

# Create data directory mount point
RUN mkdir -p /data

# Environment variables
ENV DATA_DIR=/data
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
