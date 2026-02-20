# Stash Sense

AI-powered performer identification and library curation for [Stash](https://github.com/stashapp/stash). Identifies performers in your scenes using face recognition, detects duplicate scenes, syncs upstream metadata changes, and surfaces actionable recommendations — all running locally on your hardware.

## What is Stash Sense?

Stash Sense is a sidecar service and Stash plugin that brings AI-powered analysis to your Stash library:

- **Face Recognition** — Identify performers in scenes using sprite sheets. Matches against a database of 108,000+ performers from StashDB, FansDB, ThePornDB, PMVStash, and JAVStash
- **Duplicate Scene Detection** — Find duplicate scenes using face fingerprints, stash-box IDs, and metadata overlap — catches duplicates that phash matching misses
- **Upstream Sync** — Detect metadata changes on stash-box endpoints and review per-field merge controls to keep your library current
- **Recommendations Dashboard** — A unified view of all suggestions: duplicates, unidentified scenes, missing stash-box links, and upstream updates
- **Self-Updating Database** — Check for and apply database updates from the Settings UI without restarting the container
- **Hardware-Adaptive** — Auto-detects your GPU and adjusts performance settings. Works with NVIDIA GPUs or CPU-only (slower)

## Quick Start

### Prerequisites

1. **Stash** running with scene sprite sheets generated
2. **Docker** installed on your system
3. **NVIDIA GPU** with 4GB+ VRAM recommended (CPU fallback available)

### 1. Download the database

Download the latest release from [stash-sense-data](https://github.com/carrotwaxr/stash-sense-data/releases/latest) and extract it:

```bash
mkdir -p stash-sense-data
cd stash-sense-data
# Download the latest .zip from the releases page
unzip stash-sense-data-*.zip
cd ..
```

### 2. Start the container

```bash
docker run -d \
  --name stash-sense \
  --gpus all \
  -p 6960:5000 \
  -e STASH_URL=http://your-stash-host:9999 \
  -e STASH_API_KEY=your-api-key \
  -v ./stash-sense-data:/data:ro \
  -v stash-sense-insightface:/root/.insightface \
  carrotwaxr/stash-sense:latest
```

> **No NVIDIA GPU?** Remove `--gpus all` — the sidecar auto-detects and falls back to CPU mode.

### 3. Verify it's running

```bash
curl http://localhost:6960/health
```

### 4. Install the Stash plugin

In Stash, go to **Settings > Plugins > Available Plugins** and add this source:

```
https://carrotwaxr.github.io/stash-sense/plugin/index.yml
```

Name it **Stash Sense**, then install the plugin and configure the sidecar URL (`http://your-host:6960`).

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STASH_URL` | Yes | — | URL to your Stash instance (e.g., `http://stash:9999`) |
| `STASH_API_KEY` | Yes | — | Stash API key (Settings > Security > API Key) |
| `DATA_DIR` | No | `/data` | Path to database files inside the container |
| `LOG_LEVEL` | No | `warning` | Logging verbosity: `debug`, `info`, `warning`, `error` |

Additional performance and recognition settings are configurable via the **Settings** tab in the plugin UI. The sidecar auto-tunes defaults based on your hardware.

## Updating

### Database Updates

Stash Sense checks for new database releases automatically. To update:

1. Open the **Settings** tab in the plugin
2. Check the **Database** section for available updates
3. Click **Update** — the sidecar downloads and hot-swaps the data without restarting

### Container Updates

```bash
docker stop stash-sense && docker rm stash-sense
docker pull carrotwaxr/stash-sense:latest
# Re-run the same docker run command from installation
```

**unRAID users:** Click **Force Update** in the Docker tab to pull the latest image.

Your recommendation history and settings are stored separately from the face database and persist across both types of updates.

## Documentation

Full documentation: **[https://carrotwaxr.github.io/stash-sense](https://carrotwaxr.github.io/stash-sense)**

- [Installation Guide](https://carrotwaxr.github.io/stash-sense/installation/)
- [Configuration](https://carrotwaxr.github.io/stash-sense/configuration/)
- [Features](https://carrotwaxr.github.io/stash-sense/features/)
- [Plugin Setup](https://carrotwaxr.github.io/stash-sense/plugin/)
- [Database & Updates](https://carrotwaxr.github.io/stash-sense/database/)
- [Settings](https://carrotwaxr.github.io/stash-sense/settings-system/)
- [Troubleshooting](https://carrotwaxr.github.io/stash-sense/troubleshooting/)

## Requirements

| Component | Requirement |
|-----------|-------------|
| Stash | v0.25+ with sprite sheets generated |
| Docker | With `nvidia-container-toolkit` (for GPU) |
| GPU | NVIDIA with 4GB+ VRAM (optional — CPU fallback available) |
| Disk | ~1.5 GB for the face recognition database |

## Support

- **Documentation**: [https://carrotwaxr.github.io/stash-sense](https://carrotwaxr.github.io/stash-sense)
- **Bug Reports**: [GitHub Issues](https://github.com/carrotwaxr/stash-sense/issues)
- **Community**: [Stash Discord](https://discord.gg/2TsNFKt) `#third-party-integrations`

## License

MIT License — See [LICENSE](LICENSE) for details.
