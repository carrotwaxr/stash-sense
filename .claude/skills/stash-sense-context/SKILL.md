---
name: stash-sense-context
description: Use when starting work on stash-sense to load project context, architecture overview, and development patterns. Reference for any implementation decisions.
---

# Stash Sense Project Context

Quick-reference for architecture, conventions, and operational knowledge. See `docs/architecture.md` for the full system design.

## Architecture

**Two components:**
- **Sidecar API** (`api/`) — Python/FastAPI, face recognition, recommendations, upstream sync
- **Plugin** (`plugin/`) — JS/CSS/Python injected into Stash web UI, proxies to sidecar

**Two databases:**
- `performers.db` — Read-only, distributed via GitHub Releases. Face metadata, stash-box IDs, Voyager indices.
- `stash_sense.db` — Read-write, user-local. Recommendations, watermarks, upstream snapshots, fingerprints.

**Deployment:**
- Sidecar: Docker on unRAID (`10.0.0.4:6960`), image `carrotwaxr/stash-sense:latest`
- Plugin: SCP to `/mnt/nvme_cache/appdata/stash/config/plugins/stash-sense/`
- Dev sidecar: `http://localhost:5000` with `--reload`

## Key Systems

| System | Key Files | Pattern |
|--------|-----------|---------|
| Face Recognition | `embeddings.py`, `recognizer.py`, `face_config.py` | 3-phase batch: extract -> detect -> embed+match |
| Recommendations | `recommendations_router.py`, `recommendations_db.py`, `analyzers/` | BaseAnalyzer + incremental watermarking |
| Duplicate Detection | `analyzers/duplicate_scenes.py` | Candidate generation (SQL joins) -> sequential scoring |
| Upstream Sync | `upstream_field_mapper.py`, `stashbox_client.py` | 3-way diff (upstream vs local vs snapshot) |
| Multi-Signal | `multi_signal_matcher.py`, `body_proportions.py` | Face primary, body/tattoo multiplicative adjustment |
| Gallery ID | `recognizer.py` (`/identify/image`, `/identify/gallery`) | Independent images, aggregate by performer |
| DB Self-Update | `database_updater.py` | download -> verify -> swap -> reload, 503 gating |
| Plugin Proxy | `stash_sense_backend.py` | All sidecar calls go through plugin backend |

## Development Commands

```bash
# Start sidecar (dev)
cd api && source ../.venv/bin/activate && make sidecar

# Or without shell activation
/home/carrot/code/stash-sense/.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload

# Run tests
cd api && ../.venv/bin/python -m pytest tests/ -v

# Deploy plugin
scp plugin/* root@10.0.0.4:/mnt/nvme_cache/appdata/stash/config/plugins/stash-sense/

# Build and push Docker image
docker build -t carrotwaxr/stash-sense:latest . && docker push carrotwaxr/stash-sense:latest
```

## Conventions

- **Logging:** Default level is WARNING. Use `logger.warning()` for user-visible progress. `logger.info()` is not visible.
- **Rate limiting:** Shared 5 req/s for Stash and StashBox APIs. StashBox uses `Priority.LOW`.
- **Plugin defaults:** Plugin sends NO face recognition defaults; relies on sidecar `face_config.py`.
- **Background tasks:** Don't inherit shell activation. Use explicit venv python path for background processes.
- **Hot reload caveat:** Background analysis tasks block uvicorn `--reload` on file changes; must kill and restart.

## Field Mapping (Stash vs StashBox)

| Diff Engine | Stash Mutation | StashBox | Notes |
|---|---|---|---|
| `aliases` | `alias_list` | `aliases` | |
| `height` | `height_cm` (Int) | `height` (Int) | |
| `breast_type` | `fake_tits` | `breast_type` | |
| `career_start_year` | `career_length` (String) | `career_start_year` (Int) | Combined "YYYY-YYYY" |
| `career_end_year` | `career_length` (String) | `career_end_year` (Int) | Combined "YYYY-YYYY" |
| `cup_size` | `measurements` (String) | `cup_size` | Combined "38F-24-35" |
| `band_size` | `measurements` (String) | `band_size` (Int) | Combined "38F-24-35" |
| `waist_size` | `measurements` (String) | `waist_size` (Int) | Combined "38F-24-35" |
| `hip_size` | `measurements` (String) | `hip_size` (Int) | Combined "38F-24-35" |

Translation: `recommendations_router.py:update_performer_fields()`

## Face Recognition Tuned Defaults

| Parameter | Value | Notes |
|-----------|-------|-------|
| Fusion weights | 0.5/0.5 FaceNet/ArcFace | Equal weight since ArcFace normalization fixed |
| max_distance | 0.5 | Plateau at 0.5, was 0.7 |
| num_frames | 60 | +12.5% accuracy vs 40 |
| min_unique_frames | 2 | Higher values cost precision |
| Cluster threshold | 0.6 cosine distance | On concatenated 1024-dim embeddings |
| Tagged performer boost | +0.03 | Applied when performer already on scene |

## Docker Image

- Base: `nvidia/cuda:12.4.0-runtime-ubuntu22.04`
- ONNX models baked in: `facenet512.onnx` (90MB) + `arcface.onnx` (130MB)
- Stripped: PyTorch, TensorFlow, DeepFace (not needed at runtime)
- Port: 6960 (host) -> 5000 (container)
- GPU: `--runtime=nvidia --gpus all`
- Volumes: `/data` (Voyager indices, performers.db, stash_sense.db), `/root/.insightface` (detector cache)

## GitHub

- Repo: `carrotwaxr/stash-sense`
- Issues: Used for feature planning (full implementation plans in body)
- Trainer repo: `/home/carrot/code/stash-sense-trainer/` (private, builds performers.db)

## Related Skills

- `deploy-dev-plugin` — SCP plugin files to Unraid
- `db-import-export` — Copy face DB from trainer to sidecar
- `create-ticket` — Plan and create GitHub Issues
- `work-ticket` — Pick up and implement a GitHub Issue
- `git-preferences` — Branching, commits, PR conventions
- `python-fastapi` — FastAPI patterns used in this project
- `stash` / `stash-box` / `stash-plugin-dev` — Stash ecosystem references
