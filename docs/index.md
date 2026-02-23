# Stash Sense

ML-powered performer identification and library curation for [Stash](https://github.com/stashapp/stash).

## What It Does

- **Identifies performers** in scenes and images using face recognition against a database of 108,000+ performers sourced from multiple Stash-Box endpoints
- **Tags scenes from Stash-Box** by searching untagged scenes across Stash-Box endpoints using file fingerprints
- **Detects duplicate scenes** using face fingerprints, Stash-Box IDs, and metadata overlap — catches duplicates that phash matching misses
- **Syncs upstream changes** from Stash-Box endpoints with per-field merge controls for performers, studios, and scenes
- **Runs locally** on your hardware — no cloud dependencies, no data leaves your network

## Getting Started

Stash Sense has three components that work together:

- **Stash** — Your media organizer (already running). Stash Sense reads your library data and writes back performer tags, metadata updates, etc.
- **Stash Sense Sidecar** — A Docker container running ML models and analysis engines. It connects to Stash via API key and handles all the heavy lifting (face recognition, upstream diffing, duplicate detection).
- **Stash Sense Plugin** — Installed inside Stash, provides the UI (dashboard, settings, scene-level controls). It talks to the sidecar through a backend proxy.

Stash-Box endpoints (StashDB, FansDB, ThePornDB, etc.) are configured in Stash's **Settings > Metadata Providers** — the sidecar reads credentials from there automatically.

**Next steps:**

- [Installation](installation.md) — Docker setup, plugin install, database and model download
- [Configuration](configuration.md) — environment variables, plugin settings, Stash-Box auto-discovery
- [Plugin Usage](plugin.md) — navigating the dashboard, identifying performers, understanding results

## Requirements

| Component | Requirement |
|-----------|-------------|
| Stash | v0.25+ |
| Docker | With `nvidia-container-toolkit` for GPU acceleration |
| GPU | NVIDIA with 4GB+ VRAM **(Recommended)** — not required, CPU fallback available |
| Disk | ~2.5 GB for face recognition data, models, and working space |
| Generated content | Some features require Stash-generated content — see each feature page for prerequisites |

!!! note "Generated content"
    **Performer identification** requires sprite sheets (Settings > Tasks > Generate > Sprites). **Scene tagger** requires perceptual hashes (Settings > Tasks > Generate > Perceptual Hashes). Details are on each feature's page under "Prerequisites."
