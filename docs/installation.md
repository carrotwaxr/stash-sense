# Installation

## Prerequisites

Before installing Stash Sense:

1. **Stash** running with GraphQL API enabled
2. **Stash API Key** generated in Stash Settings > Security
3. **Docker** installed on your system
4. **NVIDIA GPU** with 4GB+ VRAM recommended (CPU fallback available)
5. **Scene sprite sheets** generated in Stash (Settings > Tasks > Generate > Sprites)

## Step 1: Start the Container

### Docker Run

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

!!! warning "The `/data` volume must be read-write"
    The sidecar writes `stash_sense.db` (your recommendations, settings, and analysis history) to `/data`, and in-app database updates also write here. Do **not** mount with `:ro`.

### Docker Compose

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

Then start it:

```bash
docker compose up -d
```

### CPU-Only Mode

If you don't have an NVIDIA GPU, remove the GPU configuration:

**Docker Run** — remove `--gpus all`:

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

**Docker Compose** — remove the `deploy.resources` block.

CPU mode works but is significantly slower for face detection (~2-3 seconds per frame vs ~200ms with GPU).

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

> **Tip**: If Stash and Stash Sense run on the same machine, use the host IP (e.g., `http://10.0.0.4:6960`). Docker container names don't resolve unless both containers share a Docker network.

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

### Alternative: Manual Database Download

If you prefer to pre-populate the data directory before starting the container:

1. Download the latest release from [stash-sense-data](https://github.com/carrotwaxr/stash-sense-data/releases/latest)
2. Extract the zip into your data directory (the path mounted to `/data`)
3. The sidecar will detect the files on startup

---

## unRAID Installation

See [unRAID Setup](unraid/setup.md) for template-based installation and GPU passthrough configuration.

---

## Next Steps

- [Configuration](configuration.md) — environment variables and volume mounts
- [Plugin Usage](plugin.md) — identifying performers and using the dashboard
- [Features](features.md) — full feature overview
- [Database & Updates](database.md) — where the data comes from and how to update
