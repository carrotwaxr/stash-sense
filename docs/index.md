# Stash Sense

AI-powered performer identification and library curation for [Stash](https://github.com/stashapp/stash).

## What it does

- **Identifies performers** in scenes using face recognition against a database of 108,000+ performers from StashDB, FansDB, ThePornDB, PMVStash, and JAVStash
- **Detects duplicate scenes** using face fingerprints, stash-box IDs, and metadata overlap — catches duplicates that phash matching misses
- **Syncs upstream changes** from stash-box endpoints with per-field merge controls
- **Runs locally** on your GPU — no cloud dependencies, no data leaves your network

## How it works

1. You click **Identify Performers** on a scene in Stash
2. Stash Sense extracts frames from the scene's sprite sheet
3. Faces are detected, aligned, and matched against the performer database
4. Results show matched performers with confidence scores, grouped by person
5. One click to add identified performers to the scene

## Get Started

- [Installation](installation.md) — Docker setup, database download, plugin install
- [Plugin Usage](plugin.md) — identifying performers, understanding results
- [Features](features.md) — full feature overview
- [Database & Updates](database.md) — where the data comes from, how to update

## Requirements

| Component | Requirement |
|-----------|-------------|
| Stash | v0.25+ with sprite sheets generated |
| Docker | With `nvidia-container-toolkit` (for GPU) |
| GPU | NVIDIA with 4GB+ VRAM (optional — CPU fallback available) |
| Disk | ~1.5 GB for the face recognition database |
