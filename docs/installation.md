# Installation

## Prerequisites

Before installing Stash Sense:

1. **Stash** running with GraphQL API enabled
2. **Stash API Key** generated in Stash Settings > Security
3. **Docker** installed on your system
4. **NVIDIA GPU** with 4GB+ VRAM recommended (CPU fallback available)
5. **Scene sprite sheets** generated in Stash (Settings > Tasks > Generate > Sprites)

## Step 1: Download the Database

Stash Sense requires a pre-built face recognition database. Download the latest release from the [stash-sense-data](https://github.com/carrotwaxr/stash-sense-data/releases/latest) repository.

```bash
mkdir -p stash-sense-data
cd stash-sense-data
# Download the .zip file from the latest release (~1.5 GB)
unzip stash-sense-data-*.zip
cd ..
```

After extraction, you should have:

```
stash-sense-data/
├── performers.db        # Performer metadata
├── face_facenet.voy     # FaceNet512 embedding index
├── face_arcface.voy     # ArcFace embedding index
├── faces.json           # Face reference data
├── performers.json      # Performer lookup data
└── manifest.json        # Database version and checksums
```

---

## Step 2: Start the Container

### Docker Run

```bash
docker run -d \
  --name stash-sense \
  --gpus all \
  -p 6960:5000 \
  -e STASH_URL=http://your-stash-host:9999 \
  -e STASH_API_KEY=your-api-key \
  -v /path/to/stash-sense-data:/data:ro \
  -v stash-sense-insightface:/root/.insightface \
  carrotwaxr/stash-sense:latest
```

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
      - /path/to/stash-sense-data:/data:ro
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
  -v /path/to/stash-sense-data:/data:ro \
  -v stash-sense-insightface:/root/.insightface \
  carrotwaxr/stash-sense:latest
```

**Docker Compose** — remove the `deploy.resources` block.

CPU mode works but is significantly slower for face detection (~2-3 seconds per frame vs ~200ms with GPU).

---

## Step 3: Verify

```bash
curl http://localhost:6960/health
```

Should return:

```json
{
  "status": "healthy",
  "database_loaded": true,
  "performer_count": 108001,
  "face_count": 366794
}
```

---

## Step 4: Install the Stash Plugin

Add the Stash Sense plugin source in Stash:

1. Go to **Settings > Plugins > Available Plugins**
2. Click **Add Source** and enter:
   - **Name**: `Stash Sense`
   - **URL**: `https://carrotwaxr.github.io/stash-sense/plugin/index.yml`
3. Click **Save**
4. Find **Stash Sense** in the available plugins list and click **Install**
5. Go to **Settings > Plugins > Stash Sense** and set the **Sidecar URL** to `http://your-stash-sense-host:6960`

> **Tip**: If Stash and Stash Sense run on the same machine, use `http://stash-sense:5000` (Docker networking) or `http://localhost:6960` (host networking).

---

## unRAID Installation

See [unRAID Setup](unraid/setup.md) for template-based installation and GPU passthrough configuration.

---

## Next Steps

- [Configure settings](configuration.md) — environment variables and volume mounts
- [Plugin usage](plugin.md) — how to identify performers and use the dashboard
- [Features](features.md) — full feature overview
- [Database & Updates](database.md) — where the data comes from and how to update
