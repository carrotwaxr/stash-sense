# Configuration

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STASH_URL` | Yes | — | URL to your Stash instance (e.g., `http://stash:9999`) |
| `STASH_API_KEY` | Yes | — | Stash API key (Settings > Security > API Key) |
| `DATA_DIR` | No | `/data` | Path to database files inside the container |

### Stash-Box API Keys

Stash-box endpoint API keys are **not** configured as environment variables. The sidecar auto-discovers all configured stash-box endpoints from your Stash instance's **Settings > Metadata Providers** at startup. Any endpoints you've added there (StashDB, FansDB, ThePornDB, etc.) are automatically available for upstream sync.

To add a new endpoint, configure it in Stash's Metadata Providers settings, then restart the sidecar (or use the **Refresh** button in the plugin's Settings tab).

---

## Volume Mounts

| Container Path | Purpose | Mode |
|----------------|---------|------|
| `/data` | Database files, models, and local data (`stash_sense.db`) | **Read-write** |
| `/root/.insightface` | Cached face detection model weights (~500 MB) | Read-write |

!!! warning
    The `/data` volume **must** be read-write. The sidecar writes `stash_sense.db` (recommendations, settings, analysis history) here, and in-app database updates download files to this directory.

### Database Files

The `/data` volume should contain the files from a [stash-sense-data release](https://github.com/carrotwaxr/stash-sense-data/releases):

```
/data/
├── performers.db        # Performer metadata (SQLite)
├── face_facenet.voy     # FaceNet512 embedding index
├── face_arcface.voy     # ArcFace embedding index
├── faces.json           # Face-to-performer mapping
├── performers.json      # Performer lookup data
├── manifest.json        # Version and checksums
├── models/              # (auto-created) ONNX models downloaded via Settings UI
│   ├── facenet512.onnx  # FaceNet512 embedding model (~90 MB)
│   └── arcface.onnx     # ArcFace embedding model (~130 MB)
└── stash_sense.db       # (auto-created) Your local data — do not delete
```

!!! note
    `stash_sense.db` and the `models/` directory are created automatically by the sidecar. They are **not** part of the database release and should not be deleted during updates. The database files and models can both be downloaded from the plugin's **Settings** tab after startup.

### Model Cache

The `/root/.insightface` volume caches the RetinaFace face detection model weights. Without this mount, the model is re-downloaded on every container start (~500 MB download).

---

## Ports

The sidecar listens on port **5000** inside the container. The recommended host port is **6960**:

```
-p 6960:5000
```

---

## Runtime Settings

Most performance and recognition settings are configured via the plugin's **Settings** tab rather than environment variables. The sidecar auto-detects your hardware and sets sensible defaults.

See [Settings Reference](settings-system.md) for the full list of adjustable options.

### Migrated Environment Variables

The following environment variables have been replaced by runtime settings. If detected at startup, they are automatically migrated to the settings database:

| Old Variable | New Setting |
|-------------|-------------|
| `STASH_RATE_LIMIT` | `stash_api_rate` |
| `ENABLE_BODY_SIGNAL` | `body_signal_enabled` |
| `ENABLE_TATTOO_SIGNAL` | `tattoo_signal_enabled` |
| `FACE_CANDIDATES` | `face_candidates` |
