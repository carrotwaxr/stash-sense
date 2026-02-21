# Settings System

The sidecar includes a hardware-adaptive settings system that auto-detects your hardware at startup and picks optimal defaults. All runtime settings can be viewed and adjusted via the Settings API or the Plugin Settings UI.

## How It Works

### Hardware Detection

On startup, the sidecar probes:
- **GPU**: ONNX Runtime CUDA provider + pynvml for VRAM/model name
- **CPU**: Core count (respects Docker cgroup limits)
- **Memory**: Total and available RAM (respects Docker cgroup limits)
- **Storage**: Free disk space at the data directory

### Hardware Tiers

Based on detection results, the sidecar classifies your hardware into a tier:

| Tier | Criteria | Batch Size | Concurrency | Detection Res | Frames/Scene |
|------|----------|------------|-------------|---------------|--------------|
| `gpu-high` | CUDA + VRAM >= 4GB | 32 | 8 | 640px | 60 |
| `gpu-low` | CUDA + VRAM < 4GB | 16 | 6 | 640px | 60 |
| `cpu` | No CUDA | 4 | 2 | 320px | 30 |

### Setting Resolution

Each setting is resolved in priority order:

1. **User override** (stored in the database) — highest priority
2. **Tier default** — based on your detected hardware tier
3. **Hardcoded fallback** — always present as a baseline

Only user overrides are stored. Absence means "use the tier default."

## Settings Reference

### Performance

| Setting | Type | Range | Description |
|---------|------|-------|-------------|
| `embedding_batch_size` | int | 1-128 | Faces processed per GPU inference call |
| `frame_extraction_concurrency` | int | 1-16 | Parallel ffmpeg processes for frame extraction |
| `detection_size` | int | 160-1280 | Face detection input resolution (pixels) |

### Rate Limits

| Setting | Type | Range | Description |
|---------|------|-------|-------------|
| `stash_api_rate` | float | 0.5-50 | Max requests/second to local Stash instance |

### Recognition

| Setting | Type | Range | Description |
|---------|------|-------|-------------|
| `gpu_enabled` | bool | - | Use GPU for inference (disable to force CPU) |
| `num_frames` | int | 10-200 | Frames to sample per scene |
| `face_candidates` | int | 5-100 | Candidate matches retrieved from vector index per face |

### Signals

| Setting | Type | Description |
|---------|------|-------------|
| `body_signal_enabled` | bool | Use body proportion analysis |
| `tattoo_signal_enabled` | bool | Use tattoo detection (requires model) |

## Settings API

### Get All Settings

```
GET /settings
```

Returns all settings grouped by category with metadata for UI rendering:

```json
{
  "hardware_tier": "gpu-high",
  "categories": {
    "performance": {
      "label": "Performance",
      "settings": {
        "embedding_batch_size": {
          "value": 32,
          "default": 32,
          "is_override": false,
          "type": "int",
          "min": 1,
          "max": 128,
          "label": "Embedding Batch Size",
          "description": "Faces processed per GPU inference call"
        }
      }
    }
  }
}
```

### Get Single Setting

```
GET /settings/{key}
```

### Update Setting

```
PUT /settings/{key}
Content-Type: application/json

{"value": 64}
```

### Bulk Update

```
PUT /settings
Content-Type: application/json

{"settings": {"stash_api_rate": 10.0, "num_frames": 45}}
```

### Reset Setting to Default

```
DELETE /settings/{key}
```

Removes the user override, reverting to the tier default.

### System Info

```
GET /system/info
```

Returns hardware profile, version, and uptime:

```json
{
  "version": "0.1.0-beta.4",
  "uptime_seconds": 3600,
  "hardware": {
    "gpu_available": true,
    "gpu_name": "NVIDIA GeForce GTX 1080",
    "gpu_vram_mb": 8192,
    "cpu_cores": 8,
    "memory_total_mb": 32768,
    "tier": "gpu-high",
    "summary": "NVIDIA GeForce GTX 1080 (8192MB VRAM), 32768MB RAM, 8 cores, 500000MB free disk"
  }
}
```

## Environment Variable Migration

The following env vars have been replaced by settings. If detected at startup, they are automatically migrated to the settings database and a deprecation warning is logged.

| Old Env Var | New Setting | Notes |
|-------------|-------------|-------|
| `STASH_RATE_LIMIT` | `stash_api_rate` | |
| `ENABLE_BODY_SIGNAL` | `body_signal_enabled` | |
| `ENABLE_TATTOO_SIGNAL` | `tattoo_signal_enabled` | |
| `FACE_CANDIDATES` | `face_candidates` | |

**Env vars that remain** (connection strings and secrets):
- `STASH_URL`, `STASH_API_KEY`, `DATA_DIR`, `LOG_LEVEL`
- Per-endpoint `*_URL` and `*_API_KEY` vars

## Architecture

### Key Files

| File | Purpose |
|------|---------|
| `api/hardware.py` | Hardware detection, tier classification |
| `api/settings.py` | Setting definitions, tier defaults, resolution logic, env var migration |
| `api/settings_router.py` | FastAPI endpoints for settings and system info |

### Startup Sequence

1. `init_hardware(data_dir)` — probes hardware, classifies tier
2. `init_settings(db, tier)` — creates settings manager with tier defaults
3. `migrate_env_vars(mgr)` — one-time migration of deprecated env vars
4. Sidecar logs hardware summary and active tier

### Storage

Settings overrides are stored in the `user_settings` table of `stash_sense.db` with a `settings.` key prefix. The table uses a simple key-value structure with JSON-encoded values.
