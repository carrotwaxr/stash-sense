# Dockerize Stash Sense Sidecar for unRAID

**Date:** 2026-02-13
**Status:** Proposed

## Goal

Move the stash-sense sidecar API from the development VM (10.0.0.5) to the unRAID server (10.0.0.4) as a Docker container with GPU support, running alongside Stash.

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| ONNX models | Baked into image | Simpler deployment, models rarely change |
| Data files | Volume-mounted from NVMe cache | Too large for image, may update independently |
| Host port | 6960 | Fits Stash family (6968-6971) |
| PyTorch | Removed | Tattoo detection not production-ready, saves ~2GB |
| TensorFlow | Removed | Only used by offline conversion script |
| DeepFace | Removed | Only used by offline conversion script |
| Data path | `/mnt/nvme_cache/appdata/stash-sense/` | Direct NVMe for low-latency I/O |

## Image Architecture

**Base:** `nvidia/cuda:12.4.0-runtime-ubuntu22.04`

**Two-stage build:**

```
Stage 1 (build):
  nvidia/cuda:12.4.0-runtime-ubuntu22.04
  + Python 3.11 + venv
  + pip install requirements.docker.txt (slimmed)
  + Copy ONNX models into /app/models/

Stage 2 (runtime):
  nvidia/cuda:12.4.0-runtime-ubuntu22.04
  + Python 3.11 (no venv tools)
  + ffmpeg, curl
  + Copy venv from stage 1
  + Copy app code + models
  ~2-2.5GB compressed
```

**Stripped dependencies** (not needed at runtime):
- `tensorflow-cpu` — only in `convert_models_to_onnx.py`
- `deepface` — only in `convert_models_to_onnx.py`
- `torch` / `torchvision` — only in `tattoo_detector.py` (disabled)

**Added dependencies:**
- `mediapipe>=0.10.9` — body proportion signal (was missing from docker reqs)

**System packages added to runtime stage:**
- `ffmpeg` — required by `/identify/scene` endpoint
- `curl` — required by HEALTHCHECK

## Runtime Configuration

### Environment Variables

| Variable | Default | Required | Description |
|---|---|---|---|
| `STASH_URL` | *(none)* | Yes | Stash GraphQL endpoint, e.g. `http://10.0.0.4:6969` |
| `STASH_API_KEY` | *(none)* | Yes | Stash API authentication JWT |
| `DATA_DIR` | `/data` | No | Container path for data files |
| `PYTHONUNBUFFERED` | `1` | No | Unbuffered logging |
| `ENABLE_BODY_SIGNAL` | `true` | No | Body proportion matching |
| `ENABLE_TATTOO_SIGNAL` | `false` | No | Disabled (no PyTorch in image) |
| `STASH_RATE_LIMIT` | `5.0` | No | Stash API rate limit (req/s) |

### Volume Mounts

| Container Path | Host Path (unRAID) | Mode | Contents |
|---|---|---|---|
| `/data` | `/mnt/nvme_cache/appdata/stash-sense/data/` | rw | Voyager indices, performers.db, JSON metadata, stash_sense.db |
| `/root/.insightface` | `/mnt/nvme_cache/appdata/stash-sense/insightface-cache/` | rw | InsightFace model cache (RetinaFace detector, downloaded on first run) |

### Networking

- Container port: 5000
- Host port: 6960
- Bridge networking (standard unRAID)
- Inter-container communication via `10.0.0.4:port` (not container names)

### GPU

ExtraParams: `--runtime=nvidia --gpus all`

Requires `nvidia-container-toolkit` on unRAID (Nvidia-Driver plugin).

## Data Files to Transfer

Essential files from `api/data/` to deploy to unRAID:

| File | Size | Notes |
|---|---|---|
| `face_facenet.voy` | 573 MB | FaceNet Voyager index |
| `face_arcface.voy` | 573 MB | ArcFace Voyager index |
| `performers.db` | 210 MB | Performer metadata SQLite |
| `performers.json` | 26 MB | Performer info JSON |
| `faces.json` | 14 MB | Face-to-performer mapping |
| `manifest.json` | <1 KB | Database version metadata |
| **Total** | **~1.4 GB** | |

`stash_sense.db` (recommendations database) will be created automatically on first startup.

Skip: backup files, progress files, old database versions.

## unRAID XML Template

New file: `unraid-template.xml` checked into repo root.

Template fields exposed in unRAID UI:
- Sidecar Port (6960)
- STASH_URL
- STASH_API_KEY
- Data path
- InsightFace cache path

## Plugin Configuration

The sidecar URL is already configurable in the plugin settings (`stash-sense.yml` → `sidecarUrl`). After deployment, update the setting in Stash UI:

- **Before:** `http://10.0.0.5:5000` (dev machine)
- **After:** `http://10.0.0.4:6960` (unRAID)

No code changes needed in the plugin.

## Build & Deploy Workflow

```bash
# Build on dev machine
docker build -t carrotwaxr/stash-sense:latest .
docker push carrotwaxr/stash-sense:latest

# Transfer data files (one-time)
ssh root@10.0.0.4 "mkdir -p /mnt/nvme_cache/appdata/stash-sense/data"
rsync -avz --progress \
  api/data/{face_facenet.voy,face_arcface.voy,performers.db,performers.json,faces.json,manifest.json} \
  root@10.0.0.4:/mnt/nvme_cache/appdata/stash-sense/data/

# Deploy template
scp unraid-template.xml root@10.0.0.4:/boot/config/plugins/dockerMan/templates-user/my-stash-sense.xml

# Pull and start on unRAID
ssh root@10.0.0.4 "docker pull carrotwaxr/stash-sense:latest"
# Then start via unRAID Docker UI or CLI
```

## Files to Create/Modify

| File | Action | Description |
|---|---|---|
| `Dockerfile` | Modify | Strip TF/PyTorch, add ffmpeg+curl, bake ONNX models, add .dockerignore |
| `requirements.docker.txt` | Modify | Remove tensorflow-cpu/deepface, add mediapipe |
| `.dockerignore` | Create | Exclude .env, data/, .git, models (handled in COPY) |
| `docker-compose.yml` | Modify | Update port, remove deprecated version field, drop deepface volume |
| `unraid-template.xml` | Create | unRAID Docker template with all config |
| `api/main.py` | Modify | Guard tattoo detector import behind ENABLE_TATTOO_SIGNAL check |
| `api/config.py` | Verify | Ensure ENABLE_TATTOO_SIGNAL defaults to false |
