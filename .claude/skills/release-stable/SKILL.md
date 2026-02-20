---
name: release-stable
description: Release a stable version (non-beta) - bumps versions in all version files, commits, tags, and pushes
---

# Release Stable Version

Use this skill to release a stable (non-beta) version after beta testing is complete.

## Version Convention

Stable versions follow semantic versioning: `X.Y.Z`

- Remove beta suffix when promoting: `0.1.0-beta.8` -> `0.1.0`
- Or increment version for new release: `0.1.0` -> `0.1.1` or `0.2.0`

## Pre-Flight Checks

1. Ensure you're on `main` branch and it's up to date
2. Check current version: look at `api/main.py` FastAPI `version=` field
3. Determine target version (remove beta suffix or increment)

## Release Steps

### Step 1: Update Versions

Edit ALL THREE files to the SAME new version:
- `api/main.py` - update `version="..."` in FastAPI app initialization
- `api/settings_router.py` - update `_version: str = "..."`
- `plugin/stash-sense.yml` - update `version:` field

**CRITICAL**: All three files must have identical version strings.

### Step 2: Commit

```bash
git add api/main.py api/settings_router.py plugin/stash-sense.yml
git commit -m "chore: bump version to X.Y.Z"
```

### Step 3: Push to Main

```bash
git push origin main
```

### Step 4: Create and Push Tag

The tag MUST match the version with a `v` prefix:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

**Example**: Version `0.1.0` -> Tag `v0.1.0`

## What Happens Next

GitHub Actions (`.github/workflows/docker-build.yml`) triggers on tag push:
1. Builds Docker image from `./Dockerfile`
2. Pushes to Docker Hub: `carrotwaxr/stash-sense` with tags `X.Y.Z`, `X.Y`, `latest`, `stable`
3. Creates GitHub Release with auto-generated notes

## Semantic Versioning Guide

- **Patch** (0.1.X): Bug fixes, minor tweaks, no new features
- **Minor** (0.X.0): New features, backward-compatible changes
- **Major** (X.0.0): Breaking changes (API overhauls, major architecture changes)

## Common Mistakes to Avoid

- Forgetting to update one of the three version files
- Tag doesn't match version (missing `v` prefix or typo)
- Pushing tag before pushing commit
- Creating tag on wrong branch
- Leaving beta suffix in version string
