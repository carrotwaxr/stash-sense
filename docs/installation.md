# Installation

## Docker Compose (Recommended)

### 1. Clone the repository

```bash
git clone https://github.com/carrotwaxr/stash-sense.git
cd stash-sense
```

### 2. Download the database

Download the latest database release and extract to `./data/`:

```bash
mkdir -p data
# Download link will be provided when database is released
tar -xzf stash-sense-db-latest.tar.gz -C data/
```

### 3. Configure environment

Create a `.env` file or edit `docker-compose.yml`:

```bash
STASH_URL=http://your-stash-host:9999
STASH_API_KEY=your-api-key-here
```

Get your API key from Stash: **Settings → Security → API Key**

### 4. Start the container

```bash
docker compose up -d
```

### 5. Verify it's running

```bash
curl http://localhost:5000/health
```

Should return `{"status": "healthy", ...}`

### 6. Install the Stash plugin

See [Plugin Setup](plugin.md).

---

## Docker Run (Alternative)

```bash
docker run -d \
  --name stash-sense \
  --runtime=nvidia \
  --gpus all \
  -p 5000:5000 \
  -e STASH_URL=http://your-stash-host:9999 \
  -e STASH_API_KEY=your-api-key \
  -v /path/to/data:/data:ro \
  -v stash-sense-models:/root/.insightface \
  ghcr.io/carrotwaxr/stash-sense:latest
```

---

## CPU-Only Mode

If you don't have an NVIDIA GPU, remove the GPU flags:

```bash
docker compose up -d  # It will auto-detect and fall back to CPU
```

CPU mode works but is significantly slower (~2-3 seconds per image vs ~200ms with GPU).

---

## Next Steps

- [Configure the service](configuration.md)
- [Install the Stash plugin](plugin.md)
- Unraid users: see [Unraid Setup](unraid/setup.md)
