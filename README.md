# Stash Sense

AI-powered performer identification for Stash.

## What it does

- Identifies performers in scenes using face recognition against StashDB
- Works with your existing Stash sprite sheets - no video processing needed
- Runs locally on your GPU - no cloud dependencies

## Quick Start

```bash
docker compose up -d
```

Then install the plugin in Stash and point it at `http://<host>:5000`.

## Documentation

- [Installation](docs/installation.md)
- [Unraid Setup](docs/unraid/setup.md)
- [GPU Passthrough](docs/unraid/gpu-passthrough.md)
- [Configuration](docs/configuration.md)
- [Plugin Setup](docs/plugin.md)
- [Troubleshooting](docs/troubleshooting.md)

## Requirements

- NVIDIA GPU with 4GB+ VRAM (CPU fallback available but slower)
- Docker with nvidia-container-toolkit
- Stash with scene sprite sheets generated

## Status

Database build in progress (~100k performers from StashDB). Pre-built database coming soon.
