# Dockerize Stash Sense Sidecar - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Containerize the stash-sense sidecar API with GPU support and deploy to unRAID server at 10.0.0.4.

**Architecture:** Two-stage Docker build on nvidia/cuda:12.4.0-runtime-ubuntu22.04. ONNX models baked into image, data files volume-mounted from NVMe cache. Stripped PyTorch/TF/DeepFace (not needed at runtime). unRAID XML template for Docker UI management.

**Tech Stack:** Docker, nvidia-container-toolkit, FastAPI, ONNX Runtime GPU, Python 3.11

**Design doc:** `docs/plans/2026-02-13-dockerize-sidecar-for-unraid.md`

---

### Task 1: Change tattoo signal default to disabled

Since PyTorch is being removed from the Docker image, the tattoo detector must default to off.

**Files:**
- Modify: `api/config.py:97` — change `ENABLE_TATTOO_SIGNAL` default from `"true"` to `"false"`

**Step 1: Change default**

In `api/config.py`, change line 97 from:
```python
enable_tattoo=os.environ.get("ENABLE_TATTOO_SIGNAL", "true").lower() == "true",
```
to:
```python
enable_tattoo=os.environ.get("ENABLE_TATTOO_SIGNAL", "false").lower() == "true",
```

**Step 2: Verify sidecar starts without tattoo**

Run: `cd /home/carrot/code/stash-sense/api && /home/carrot/code/stash-sense/.venv/bin/python -c "from config import MultiSignalConfig; c = MultiSignalConfig.from_env(); print(f'tattoo={c.enable_tattoo}, body={c.enable_body}'); assert not c.enable_tattoo; assert c.enable_body; print('OK')"`
Expected: `tattoo=False, body=True` then `OK`

**Step 3: Commit**

```bash
git add api/config.py
git commit -m "config: default ENABLE_TATTOO_SIGNAL to false

Tattoo detection requires PyTorch which is being stripped from the
Docker image. Defaults to off; can be re-enabled via env var when
PyTorch is available."
```

---

### Task 2: Update requirements.docker.txt

Remove dependencies not needed at runtime, add missing ones.

**Files:**
- Modify: `requirements.docker.txt`

**Step 1: Rewrite requirements.docker.txt**

```
# Docker runtime requirements
# PyTorch: NOT included (tattoo detection disabled by default)
# TensorFlow: NOT included (only needed by convert_models_to_onnx.py)
# DeepFace: NOT included (only needed by convert_models_to_onnx.py)

# Face recognition
insightface>=0.7.0
onnxruntime-gpu>=1.15.0
voyager>=2.0.0

# Body proportion detection
mediapipe>=0.10.9

# API
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
httpx>=0.25.0

# Image processing
pillow>=10.0.0
numpy<2
opencv-python-headless>=4.8.0

# Utilities
pydantic>=2.0.0
python-dotenv>=1.0.0
tqdm>=4.65.0
```

**Step 2: Commit**

```bash
git add requirements.docker.txt
git commit -m "docker: strip tensorflow/deepface/torch from runtime deps

These are only needed by the offline model conversion script.
Added mediapipe for body proportion signal."
```

---

### Task 3: Create .dockerignore

Prevent leaking secrets and unnecessary files into the Docker build context.

**Files:**
- Create: `.dockerignore`

**Step 1: Create .dockerignore**

```
# Secrets
.env
*.env
api/.env
api/*.env

# Git
.git
.gitignore

# Data files (volume-mounted, not baked into image)
api/data/
data/

# Python
__pycache__/
*.py[cod]
.venv/
venv/

# Large files that should not be in context
*.voy
*.db
*.h5
*.pt

# Development
api/test_output/
api/validation/
api/profile_test/
api/benchmark_results/
api/benchmark/
api/tests/
docs/
.claude/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db

# Plugin (not needed in sidecar image)
plugin/
```

Note: ONNX models in `api/models/` are intentionally NOT excluded — they get baked into the image.

**Step 2: Commit**

```bash
git add .dockerignore
git commit -m "docker: add .dockerignore to exclude secrets and large files"
```

---

### Task 4: Update Dockerfile

The main change: strip PyTorch/TF, add ffmpeg+curl, bake ONNX models.

**Files:**
- Modify: `Dockerfile`

**Step 1: Rewrite Dockerfile**

```dockerfile
# syntax=docker/dockerfile:1

# Stage 1: Build dependencies
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
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
ENV ENABLE_TATTOO_SIGNAL=false

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
```

Key changes from original:
- Added `# syntax=docker/dockerfile:1` for BuildKit features
- Removed PyTorch install step (saves ~2GB)
- Added `--no-install-recommends` to apt-get
- Added `ffmpeg` and `curl` to runtime stage
- Added pip cache mount (`--mount=type=cache`)
- Bakes ONNX models via `COPY api/models/ ./models/`
- Sets `ENABLE_TATTOO_SIGNAL=false` explicitly
- Increased `--start-period` to 120s (loading 1.4GB of indices takes time)

**Step 2: Verify build succeeds**

Run: `cd /home/carrot/code/stash-sense && docker build -t carrotwaxr/stash-sense:latest .`
Expected: Build completes. Watch for pip install errors (the main risk is insightface dependency resolution without PyTorch).

**Step 3: Commit**

```bash
git add Dockerfile
git commit -m "docker: slim image - strip PyTorch/TF, add ffmpeg, bake ONNX models

- Remove PyTorch (~2GB) and tensorflow-cpu (~600MB) from image
- Add ffmpeg (needed for /identify/scene) and curl (HEALTHCHECK)
- Bake ONNX models into image for simpler deployment
- Use BuildKit cache mount for pip
- Increase start-period to 120s for index loading"
```

---

### Task 5: Update docker-compose.yml

Update for local development/testing. Not used on unRAID but useful for validating the image.

**Files:**
- Modify: `docker-compose.yml`

**Step 1: Rewrite docker-compose.yml**

```yaml
services:
  stash-sense:
    build: .
    image: carrotwaxr/stash-sense:latest
    container_name: stash-sense
    restart: unless-stopped
    ports:
      - "6960:5000"
    volumes:
      - ./api/data:/data
      - stash-sense-insightface:/root/.insightface
    environment:
      - STASH_URL=${STASH_URL:-http://10.0.0.4:6969}
      - STASH_API_KEY=${STASH_API_KEY}
      - DATA_DIR=/data
      - PYTHONUNBUFFERED=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

volumes:
  stash-sense-insightface:
```

Changes from original:
- Removed deprecated `version: "3.8"`
- Changed port from `5000:5000` to `6960:5000`
- Removed `stash-sense-models` volume (no more DeepFace cache)
- Added `STASH_URL` and `STASH_API_KEY` from host env
- Removed `:ro` from data volume (stash_sense.db needs write)
- Increased start_period to 120s

**Step 2: Commit**

```bash
git add docker-compose.yml
git commit -m "docker-compose: update port to 6960, remove deprecated version field"
```

---

### Task 6: Create unRAID XML template

unRAID uses XML templates instead of docker-compose. This defines the container in the unRAID Docker UI.

**Files:**
- Create: `unraid-template.xml`

**Step 1: Create template**

```xml
<?xml version="1.0"?>
<Container version="2">
  <Name>stash-sense</Name>
  <Repository>carrotwaxr/stash-sense:latest</Repository>
  <Registry>https://hub.docker.com/r/carrotwaxr/stash-sense</Registry>
  <Network>bridge</Network>
  <Privileged>false</Privileged>
  <Support>https://github.com/carrotwaxr/stash-sense/issues</Support>
  <Project>https://github.com/carrotwaxr/stash-sense</Project>
  <Overview>AI-powered performer identification using face recognition. Provides a REST API for identifying performers in images and scenes using FaceNet512 and ArcFace embeddings against a database of 100k+ performers from StashDB and other stash-box instances.</Overview>
  <Category>MediaApp:Other</Category>
  <WebUI>http://[IP]:[PORT:6960]/docs</WebUI>
  <Icon>https://raw.githubusercontent.com/carrotwaxr/stash-sense/main/icon.png</Icon>
  <ExtraParams>--runtime=nvidia --gpus all</ExtraParams>
  <PostArgs/>
  <DonateText/>
  <DonateLink/>
  <Requires>NVIDIA GPU with nvidia-container-toolkit (Nvidia-Driver plugin)</Requires>
  <Config Name="API Port" Target="5000" Default="6960" Mode="tcp" Description="Port for the Stash Sense API" Type="Port" Display="always" Required="true" Mask="false">6960</Config>
  <Config Name="Data Directory" Target="/data" Default="/mnt/nvme_cache/appdata/stash-sense/data" Mode="rw" Description="Face recognition database files (Voyager indices, performers.db, JSON metadata)" Type="Path" Display="always" Required="true" Mask="false">/mnt/nvme_cache/appdata/stash-sense/data</Config>
  <Config Name="InsightFace Cache" Target="/root/.insightface" Default="/mnt/nvme_cache/appdata/stash-sense/insightface-cache" Mode="rw" Description="InsightFace model cache (RetinaFace detector, downloaded on first run)" Type="Path" Display="advanced" Required="false" Mask="false">/mnt/nvme_cache/appdata/stash-sense/insightface-cache</Config>
  <Config Name="STASH_URL" Target="STASH_URL" Default="http://10.0.0.4:6969" Mode="" Description="URL of your Stash instance" Type="Variable" Display="always" Required="true" Mask="false">http://10.0.0.4:6969</Config>
  <Config Name="STASH_API_KEY" Target="STASH_API_KEY" Default="" Mode="" Description="Stash API key (JWT token from Stash Settings &gt; Security)" Type="Variable" Display="always" Required="true" Mask="true"></Config>
  <Config Name="DATA_DIR" Target="DATA_DIR" Default="/data" Mode="" Description="Container path for data files (should match Data Directory target)" Type="Variable" Display="advanced" Required="false" Mask="false">/data</Config>
  <Config Name="PYTHONUNBUFFERED" Target="PYTHONUNBUFFERED" Default="1" Mode="" Description="Force unbuffered Python output for Docker logs" Type="Variable" Display="advanced" Required="false" Mask="false">1</Config>
  <Config Name="ENABLE_BODY_SIGNAL" Target="ENABLE_BODY_SIGNAL" Default="true" Mode="" Description="Enable body proportion matching (requires mediapipe)" Type="Variable" Display="advanced" Required="false" Mask="false">true</Config>
</Container>
```

**Step 2: Commit**

```bash
git add unraid-template.xml
git commit -m "unraid: add Docker template for unRAID deployment"
```

---

### Task 7: Build and push Docker image

Build the image on dev machine and push to Docker Hub.

**Step 1: Build image**

Run: `cd /home/carrot/code/stash-sense && docker build -t carrotwaxr/stash-sense:latest .`
Expected: Build completes successfully. Watch the output for:
- pip install completes without errors
- ONNX model files are copied (COPY api/models/ should show ~220MB)

**Step 2: Verify image locally (quick smoke test)**

Run: `docker run --rm -e STASH_URL=http://10.0.0.4:6969 -e STASH_API_KEY=test carrotwaxr/stash-sense:latest python -c "import onnxruntime; print('ort:', onnxruntime.__version__); import insightface; print('insightface OK'); import mediapipe; print('mediapipe OK'); import fastapi; print('fastapi OK')"`
Expected: All imports succeed without errors. This validates the dependency chain is intact.

**Step 3: Verify ffmpeg is available**

Run: `docker run --rm carrotwaxr/stash-sense:latest ffmpeg -version`
Expected: ffmpeg version output

**Step 4: Push to Docker Hub**

Run: `docker push carrotwaxr/stash-sense:latest`
Expected: Image pushes successfully to `carrotwaxr/stash-sense`

**Step 5: Commit any build-related fixes**

If the build exposed any issues, fix and recommit before continuing.

---

### Task 8: Transfer data files to unRAID

Copy the essential face recognition data to the unRAID NVMe cache.

**Step 1: Create directory on unRAID**

Run: `ssh root@10.0.0.4 "mkdir -p /mnt/nvme_cache/appdata/stash-sense/data /mnt/nvme_cache/appdata/stash-sense/insightface-cache"`

**Step 2: Transfer data files**

Run:
```bash
rsync -avz --progress \
  /home/carrot/code/stash-sense/api/data/face_facenet.voy \
  /home/carrot/code/stash-sense/api/data/face_arcface.voy \
  /home/carrot/code/stash-sense/api/data/performers.db \
  /home/carrot/code/stash-sense/api/data/performers.json \
  /home/carrot/code/stash-sense/api/data/faces.json \
  /home/carrot/code/stash-sense/api/data/manifest.json \
  root@10.0.0.4:/mnt/nvme_cache/appdata/stash-sense/data/
```
Expected: ~1.4GB transferred. This will take a few minutes over LAN.

**Step 3: Verify files on server**

Run: `ssh root@10.0.0.4 "ls -lh /mnt/nvme_cache/appdata/stash-sense/data/"`
Expected: All 6 files present with correct sizes.

---

### Task 9: Deploy to unRAID and verify

**Step 1: Deploy template**

Run: `scp /home/carrot/code/stash-sense/unraid-template.xml root@10.0.0.4:/boot/config/plugins/dockerMan/templates-user/my-stash-sense.xml`

**Step 2: Pull image on unRAID**

Run: `ssh root@10.0.0.4 "docker pull carrotwaxr/stash-sense:latest"`

**Step 3: Start container via CLI (for initial testing)**

Run:
```bash
ssh root@10.0.0.4 "docker run -d \
  --name stash-sense \
  --restart unless-stopped \
  --runtime=nvidia --gpus all \
  -p 6960:5000 \
  -v /mnt/nvme_cache/appdata/stash-sense/data:/data \
  -v /mnt/nvme_cache/appdata/stash-sense/insightface-cache:/root/.insightface \
  -e STASH_URL=http://10.0.0.4:6969 \
  -e STASH_API_KEY=<your-key> \
  -e DATA_DIR=/data \
  -e PYTHONUNBUFFERED=1 \
  carrotwaxr/stash-sense:latest"
```

Note: Replace `<your-key>` with actual Stash API key. Get it from `grep STASH_API_KEY /home/carrot/code/stash-sense/api/.env | cut -d= -f2`.

**Step 4: Check startup logs**

Run: `ssh root@10.0.0.4 "docker logs -f stash-sense"`
Expected output (wait ~60-120s for full startup):
```
Loading face database from /data...
Face database loaded successfully!
Initializing body proportion extractor...
Initializing multi-signal matcher...
Initializing recommendations database at /data/stash_sense.db...
Recommendations database initialized!
Stash connection configured: http://10.0.0.4:6969
```

**Step 5: Verify health endpoint**

Run: `curl -s http://10.0.0.4:6960/health | python3 -m json.tool`
Expected: `{"status": "healthy", "database_loaded": true, "performer_count": 107759, "face_count": 277097}`

**Step 6: Verify GPU is available**

Run: `ssh root@10.0.0.4 "docker exec stash-sense python -c \"import onnxruntime as ort; print(ort.get_available_providers())\""`
Expected: Output includes `CUDAExecutionProvider`

**Step 7: Verify ffmpeg**

Run: `curl -s http://10.0.0.4:6960/health/ffmpeg | python3 -m json.tool`
Expected: `{"ffmpeg_available": true, "v2_endpoint_ready": true}`

**Step 8: Update plugin sidecar URL**

In Stash UI (http://10.0.0.4:6969):
1. Go to Settings > Plugins > Stash Sense
2. Change "Stash Sense API URL" from `http://10.0.0.5:5000` to `http://10.0.0.4:6960`
3. Save

**Step 9: End-to-end test**

Navigate to a scene in Stash, open the Stash Sense panel, and run identification. Verify faces are detected and performers are matched.

---

### Task 10: Final commit and update memory

**Step 1: Commit any remaining changes**

```bash
git add -A
git status  # Review what's being committed
git commit -m "docker: complete Dockerization for unRAID deployment"
```

**Step 2: Update CLAUDE.md with Docker deployment info**

Add a section to `CLAUDE.md` documenting the Docker deployment workflow for future reference.
