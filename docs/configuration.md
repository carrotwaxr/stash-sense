# Configuration

## Essential Configuration

Two things must be configured for Stash Sense to work:

### 1. Sidecar Environment Variables

Set these when starting the Docker container:

| Variable | Required | Description |
|----------|----------|-------------|
| `STASH_URL` | Yes | URL to your Stash instance (e.g., `http://10.0.0.4:9999`) |
| `STASH_API_KEY` | Yes | Stash API key from **Settings > Security > API Key** |
| `STASH_SENSE_API_KEY` | No | Key that external tools such as stash-curator send in the `ApiKey` header to use the [driver API](driver-api.md). When it is empty, the driver API is open to your network. |

```bash
-e STASH_URL=http://your-stash-host:9999 \
-e STASH_API_KEY=your-api-key
```

### 2. Plugin Sidecar URL

In Stash, go to **Settings > Plugins > Stash Sense** and set the **Sidecar URL** to the address of your Stash Sense container (e.g., `http://10.0.0.4:6960`).

This tells the plugin backend where to forward requests. The right URL depends on your setup — see [Networking](#networking) below.

---

## Networking

The sidecar runs inside a Docker container, so `localhost` means different things depending on where you are. Here's what to use for `STASH_URL` (sidecar connecting to Stash) and the **Sidecar URL** plugin setting (Stash connecting to sidecar):

| Your Setup | `STASH_URL` | Sidecar URL |
|---|---|---|
| Same machine, both in Docker (shared network) | Container name (e.g., `http://stash:9999`) | Container name (e.g., `http://stash-sense:5000`) |
| Same machine, both in Docker (no shared network) | Host LAN IP (e.g., `http://10.0.0.4:9999`) | Host LAN IP (e.g., `http://10.0.0.4:6960`) |
| Same machine, Stash runs natively (not in Docker) | `http://host.docker.internal:9999` | `http://127.0.0.1:6960` |
| Different machines | Stash machine's LAN IP | Sidecar machine's LAN IP |

**Key points:**

- **Don't use `localhost` in `STASH_URL`** — inside the container, `localhost` refers to the container itself, not the host machine.
- **`host.docker.internal`** resolves to the host machine's IP from inside a container. This is how the sidecar reaches Stash when Stash runs natively on the host (e.g., as a Windows binary). It works automatically on Docker Desktop (Windows/Mac). On Linux, add `--add-host=host.docker.internal:host-gateway` to your `docker run` command.
- **Use `127.0.0.1` instead of `localhost`** for the Sidecar URL when Stash runs natively. On Windows, `localhost` can resolve to `::1` (IPv6) on recent Docker Desktop versions, causing "connection refused" errors. Using `127.0.0.1` explicitly avoids this.
- **When both are in Docker** on separate networks, use the host's LAN IP for both. Docker container names only resolve on shared Docker networks.

---

## Stash-Box Endpoints

Stash-Box endpoint credentials are **not** configured as environment variables. The sidecar auto-discovers all configured Stash-Box endpoints from your Stash instance's **Settings > Metadata Providers** at startup.

Any endpoints you've added there (StashDB, FansDB, ThePornDB, etc.) are automatically available for upstream sync and scene tagging.

To add a new endpoint, configure it in Stash's Metadata Providers settings, then restart the sidecar or use the **Refresh** button in the plugin's Settings tab.

---

## Runtime Settings

Most performance and recognition settings are configured via the plugin's **Settings** tab rather than environment variables. The sidecar auto-detects your hardware and sets sensible defaults.

Settings you can adjust include:

- **Frames per scene** — How many frames to sample for face recognition
- **Embedding batch size** — Faces processed per GPU call
- **Rate limits** — Requests/second to local Stash API
- **Signal toggles** — Enable/disable body proportion and tattoo detection signals

See [Settings Reference](settings-system.md) for the full list of adjustable options.

---

## Volume Mounts

| Container Path | Purpose | Mode |
|----------------|---------|------|
| `/data` | Database files, models, and local data (`stash_sense.db`) | **Read-write** |
| `/root/.insightface` | Cached face detection model weights (~500 MB) | Read-write |
| `/media` | Your media library, the same files Stash reads | Read-only (read-write only for features that write files) |

!!! warning
    The `/data` volume **must** be read-write. The sidecar writes `stash_sense.db` (recommendations, settings, analysis history) here, and in-app database updates download files to this directory.

### Media library

Features that read video files directly (audio fingerprints, frame sampling) and features that write next to them (subtitles) need the library mounted into the container. Mount it read-only unless you turn on a feature that writes files. Those features check the mount and refuse to run with a message naming the fix when it is read-only.

Stash and the sidecar usually see the library at different paths. `MEDIA_PATH_MAP` maps Stash's paths to the sidecar's:

| Stash sees | Sidecar mount | `MEDIA_PATH_MAP` |
|---|---|---|
| `/data/Studio/a.mp4` | `/media` | `/data=/media` |
| `/data` and `/archive` | `/media`, `/media2` | `/data=/media,/archive=/media2` |

Without a map, Stash's paths are used unchanged, which works when you mount the library at the same container path Stash uses. The plugin's **Settings** tab and `GET /health` report each mount and whether it is `rw`, `ro` or `missing`.

### File ownership (PUID / PGID)

The container starts as root, hands `/data` and the InsightFace cache to `PUID:PGID`, then drops to that user. Everything the sidecar creates (database, models, subtitles) is owned by `PUID:PGID`. The defaults, `99:100`, match unRAID's `nobody:users`. On other hosts use the owner of your library (`id -u`, `id -g`). `PUID=0` keeps the old run-as-root behaviour. `UMASK` (default `022`) sets the file mode mask.

The media library is never chowned. In read-write mode `PUID:PGID` needs write permission on the folders it writes into.

### CPU and GPU

CPU-heavy jobs run ffmpeg under `nice` with an optional thread cap, both set in **Settings > Performance** (`media_job_nice`, `media_job_threads`). To keep the whole container off some cores, add `--cpuset-cpus=2-7` (your core list) to the template's Extra Parameters or `cpuset: "2-7"` in compose.

The GPU tier comes from the card's VRAM, read through NVML (bundled in the image) or `nvidia-smi`. Both need the NVIDIA runtime (`--runtime=nvidia`, or the compose `deploy.resources` block). Without it, the sidecar reports **No GPU** and uses the CPU tier.

---

## Ports

The sidecar listens on port **5000** inside the container. The recommended host port is **6960**:

```
-p 6960:5000
```

---

## Database Files

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
    `stash_sense.db` and the `models/` directory are created automatically by the sidecar. They are **not** part of the database release and should not be deleted during updates. Both the database and models can be downloaded from the plugin's **Settings** tab after startup.

---

## Environment Variable Migration

The following environment variables have been replaced by runtime settings. If detected at startup, they are automatically migrated to the settings database:

| Old Variable | New Setting |
|-------------|-------------|
| `STASH_RATE_LIMIT` | `stash_api_rate` |
| `ENABLE_BODY_SIGNAL` | `body_signal_enabled` |
| `ENABLE_TATTOO_SIGNAL` | `tattoo_signal_enabled` |
| `FACE_CANDIDATES` | `face_candidates` |

Environment variables that remain (connection strings):

- `STASH_URL`, `STASH_API_KEY`, `DATA_DIR`, `LOG_LEVEL`
- `MEDIA_PATH_MAP`, `MEDIA_DIR` (default `/media`), `PUID`, `PGID`, `UMASK`

---

## Configuration Checklist

- [ ] `STASH_URL` and `STASH_API_KEY` set as container environment variables
- [ ] Sidecar URL configured in Stash plugin settings
- [ ] Stash-Box endpoints configured in Stash's **Settings > Metadata Providers** (for upstream sync and scene tagging)
- [ ] `/data` volume mounted read-write
- [ ] `/root/.insightface` volume mounted (prevents re-downloading detection model on restart)
- [ ] Media library mounted at `/media` (read-only) and `MEDIA_PATH_MAP` set if Stash uses a different path
- [ ] `PUID` / `PGID` match the owner of your library
