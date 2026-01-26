# Stash Sense

AI-powered performer identification for Stash.

## What it does

- **Identifies performers** in scenes using face recognition against StashDB
- **Works with sprite sheets** - no video processing needed, uses your existing Stash-generated sprites
- **Runs locally** on your GPU - no cloud dependencies, no data leaves your network

## How it works

1. You click "Identify Performers" on a scene in Stash
2. Stash Sense fetches the scene's sprite sheet
3. Faces are detected and matched against the StashDB performer database
4. Results show matched performers with confidence scores
5. One click to add identified performers to the scene

## Quick Start

```bash
docker compose up -d
```

Then [install the plugin](plugin.md) in Stash and point it at `http://<host>:5000`.

## Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA with 4GB+ VRAM (CPU fallback available) |
| Docker | With nvidia-container-toolkit |
| Stash | Scene sprite sheets generated |

## Current Status

Database build in progress (~100k performers from StashDB). Pre-built database release coming soon.
