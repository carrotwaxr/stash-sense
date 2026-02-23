# Installation

## Step 1: Start the Container

For best performance, install Stash Sense on the same machine as Stash.

### GPU vs CPU

Stash Sense uses GPU acceleration for face detection and embedding inference. An NVIDIA GPU with 4GB+ VRAM is recommended but not required — CPU mode works, just slower (~2-3 seconds per frame vs ~200ms with GPU).

- **GPU mode** requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on the host
- **CPU mode** requires no additional setup — just omit the GPU flags

### Docker Run

**With GPU (recommended):**

```bash
docker run -d \
  --name stash-sense \
  --gpus all \
  -p 6960:5000 \
  -e STASH_URL=http://your-stash-host:9999 \
  -e STASH_API_KEY=your-api-key \
  -v /path/to/stash-sense-data:/data \
  -v stash-sense-insightface:/root/.insightface \
  carrotwaxr/stash-sense:latest
```

**CPU only** — remove `--gpus all`:

```bash
docker run -d \
  --name stash-sense \
  -p 6960:5000 \
  -e STASH_URL=http://your-stash-host:9999 \
  -e STASH_API_KEY=your-api-key \
  -v /path/to/stash-sense-data:/data \
  -v stash-sense-insightface:/root/.insightface \
  carrotwaxr/stash-sense:latest
```

!!! warning "The `/data` volume must be read-write"
    The sidecar writes `stash_sense.db` (your recommendations, settings, and analysis history) to `/data`, and in-app database updates also write here. Do **not** mount with `:ro`.

??? note "Docker Compose"
    Create a `docker-compose.yml`:

    ```yaml
    services:
      stash-sense:
        image: carrotwaxr/stash-sense:latest
        container_name: stash-sense
        restart: unless-stopped
        ports:
          - "6960:5000"
        volumes:
          - /path/to/stash-sense-data:/data
          - stash-sense-insightface:/root/.insightface
        environment:
          - STASH_URL=http://your-stash-host:9999
          - STASH_API_KEY=your-api-key
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: 1
                  capabilities: [gpu]

    volumes:
      stash-sense-insightface:
    ```

    For CPU only, remove the `deploy.resources` block.

    ```bash
    docker compose up -d
    ```

---

## Step 2: Verify

```bash
curl http://localhost:6960/health
```

Should return:

```json
{
  "status": "degraded",
  "database_loaded": false,
  "performer_count": 0,
  "face_count": 0
}
```

This is expected — the database and models haven't been downloaded yet. Once downloaded (Step 4), the status will change to `"healthy"`.

---

## Step 3: Install the Stash Plugin

Add the Stash Sense plugin source in Stash:

1. Go to **Settings > Plugins > Available Plugins**
2. Click **Add Source** and enter:
   - **Name**: `Stash Sense`
   - **URL**: `https://carrotwaxr.github.io/stash-sense/plugin/index.yml`
3. Click **Save**
4. Find **Stash Sense** in the available plugins list and click **Install**
5. Go to **Settings > Plugins > Stash Sense** and set the **Sidecar URL** to `http://your-stash-sense-host:6960`

!!! tip
    If Stash and Stash Sense run on the same machine, use the host IP (e.g., `http://10.0.0.4:6960`). Docker container names don't resolve unless both containers share a Docker network.

---

## Step 4: Download Database and Models

Navigate to **`/plugins/stash-sense`** in your Stash UI to open the Stash Sense dashboard, then go to the **Settings** tab.

### Download the Face Recognition Database

1. In the **Database** section, click **Update** to download the latest database (~1.5 GB)
2. Progress is shown in real-time (download, extract, verify, swap)
3. Once complete, the `/health` endpoint will report `"database_loaded": true`

### Download ONNX Models

1. In the **Models** section, click **Download All** to download the required face recognition models (~220 MB)
2. Two models are required: **FaceNet512** (~90 MB) and **ArcFace** (~130 MB)
3. Optional models (tattoo detection) can be downloaded separately if needed

!!! note "Face recognition won't work until both the database and models are downloaded"
    If you try to identify performers before downloading, you'll see a "Database not loaded" or model-not-found error.

??? note "Manual database download"
    If you prefer to pre-populate the data directory before starting the container:

    1. Download the latest release from [stash-sense-data](https://github.com/carrotwaxr/stash-sense-data/releases/latest)
    2. Extract the zip into your data directory (the path mounted to `/data`)
    3. The sidecar will detect the files on startup

---

## Platform-Specific Notes

### unRAID

See [unRAID Setup](unraid/setup.md) for template-based installation and GPU passthrough configuration.

### Windows / WSL

Stash Sense runs on Windows via Docker Desktop with the WSL 2 backend.

**Setup:**

- Ensure **WSL 2** is enabled (Docker Desktop > Settings > General > Use the WSL 2 based engine)
- For GPU support, install the [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/) driver and enable GPU support in Docker Desktop (Settings > Resources > WSL Integration)

**Multi-line commands:** PowerShell uses backtick (`` ` ``) instead of backslash (`\`) for line continuation:

```powershell
docker run -d `
  --name stash-sense `
  --gpus all `
  -p 6960:5000 `
  -e STASH_URL=http://your-stash-host:9999 `
  -e STASH_API_KEY=your-api-key `
  -v C:/Users/you/stash-sense-data:/data `
  -v stash-sense-insightface:/root/.insightface `
  carrotwaxr/stash-sense:latest
```

**Networking gotchas:**

- On recent Docker Desktop versions, `localhost` may resolve to `::1` (IPv6) instead of `127.0.0.1`. If `curl http://localhost:6960/health` fails with "connection refused," try `http://127.0.0.1:6960/health` instead.
- When setting `STASH_URL`, use your machine's LAN IP (e.g., `http://192.168.1.100:9999`) rather than `localhost`, since the container has its own network namespace.
- Similarly, set the plugin's **Sidecar URL** to the LAN IP (e.g., `http://192.168.1.100:6960`), not `localhost`.

**Volume performance:** Named volumes (e.g., `stash-sense-data:/data`) are faster than bind mounts to Windows paths. If you use bind mounts, paths on the WSL filesystem are faster than Windows-mounted paths (`/mnt/c/...` is slower due to WSL2's 9P filesystem bridge).

---

## Installation Checklist

- [ ] Sidecar container running (`docker ps` shows `stash-sense`)
- [ ] Health check returns a response (`curl http://localhost:6960/health`)
- [ ] Plugin installed in Stash (visible in Settings > Plugins)
- [ ] Sidecar URL configured in plugin settings
- [ ] Face recognition database downloaded (Settings tab > Database > Update)
- [ ] ONNX models downloaded (Settings tab > Models > Download All)

---

## Next Steps

- [Configuration](configuration.md) — environment variables, Stash-Box auto-discovery, runtime settings
- [Plugin Usage](plugin.md) — navigating the dashboard, identifying performers
- [Features](features/performer-identification.md) — start with performer identification
