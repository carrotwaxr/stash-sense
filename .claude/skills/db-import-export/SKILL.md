---
name: db-import-export
description: Use when copying face recognition database from stash-sense-trainer to stash-sense sidecar after enrichment runs, or when debugging data loading failures at sidecar startup
---

# DB Import/Export Between Trainer and Sidecar

## Overview

The trainer (`stash-sense-trainer`) produces face recognition data. The sidecar (`stash-sense/api`) consumes it read-only. After each enrichment run, up to 9 files must be copied and validated (6 required + 3 optional).

## File Manifest

| File | Source (trainer) | Destination (sidecar) | Purpose |
|------|-----|-----|---------|
| `performers.db` | `data/performers.db` | `api/data/performers.db` | SQLite performer metadata, faces, stashbox IDs, aliases |
| `face_facenet.voy` | `data/face_facenet.voy` | `api/data/face_facenet.voy` | Voyager index - FaceNet512 embeddings |
| `face_arcface.voy` | `data/face_arcface.voy` | `api/data/face_arcface.voy` | Voyager index - ArcFace embeddings |
| `face_adaface.voy` | `data/face_adaface.voy` | `api/data/face_adaface.voy` | Voyager index - AdaFace IR-101 embeddings (optional) |
| `tattoo_embeddings.voy` | `data/tattoo_embeddings.voy` | `api/data/tattoo_embeddings.voy` | Voyager index - tattoo embeddings (optional) |
| `performers.json` | `data/performers.json` | `api/data/performers.json` | universal_id -> name/country/image/body/tattoo lookup |
| `faces.json` | `data/faces.json` | `api/data/faces.json` | Voyager index -> universal_id mapping |
| `tattoo_embeddings.json` | `data/tattoo_embeddings.json` | `api/data/tattoo_embeddings.json` | Tattoo index -> universal_id mapping (optional) |
| `manifest.json` | `data/manifest.json` | `api/data/manifest.json` | Version, checksums, model info, counts |

## Import Procedure

**Stop the sidecar before copying.**

### 1. Verify Trainer Output

```bash
cd /home/carrot/code/stash-sense-trainer/data

# Checkpoint WAL if in WAL mode
sqlite3 performers.db "PRAGMA wal_checkpoint(TRUNCATE);"

# Integrity check
sqlite3 performers.db "PRAGMA integrity_check;"

# Verify checksums against manifest
sha256sum performers.db face_facenet.voy face_arcface.voy face_adaface.voy tattoo_embeddings.voy
# Compare with manifest.json checksums (performers.db may drift due to WAL - update manifest if so)
# face_adaface.voy and tattoo_embeddings.voy are optional - skip if not present
```

### 2. Cross-Validate Counts

```python
import json, voyager

with open('manifest.json') as f: manifest = json.load(f)
with open('faces.json') as f: faces = json.load(f)
with open('performers.json') as f: performers = json.load(f)
facenet = voyager.Index.load('face_facenet.voy')
arcface = voyager.Index.load('face_arcface.voy')

assert len(faces) == facenet.num_elements == arcface.num_elements == manifest['face_count']
assert len(performers) == manifest['performer_count']
```

### 3. Check Schema Compatibility

The sidecar's `database_reader.py` uses `SELECT *` with dataclass unpacking (`**dict(row)`). If the trainer adds columns to any table, the corresponding dataclass must be updated.

**Tables and their sidecar dataclasses:**

| Table | Dataclass | Query Pattern |
|-------|-----------|---------------|
| `performers` | `Performer` | `SELECT *` -> `**dict(row)` |
| `faces` | `Face` | `SELECT *` -> `**dict(row)` |
| `stashbox_ids` | `StashboxId` | `SELECT *` -> `**dict(row)` |
| `aliases` | `Alias` | `SELECT *` -> `**dict(row)` |
| `tattoos` | dict | `SELECT location, description` (safe) |
| `piercings` | dict | `SELECT location, description` (safe) |
| `body_proportions` | dict | explicit columns (safe) |
| `tattoo_detections` | dict | explicit columns (safe) |

**To check for mismatches:**
```bash
sqlite3 performers.db "PRAGMA table_info(faces);"
sqlite3 performers.db "PRAGMA table_info(stashbox_ids);"
sqlite3 performers.db "PRAGMA table_info(aliases);"
sqlite3 performers.db "PRAGMA table_info(performers);"
```
Compare column names against the dataclass fields in `api/database_reader.py`. New columns need to be added as `Optional[...] = None` fields.

### 4. Copy Files

```bash
SRC=/home/carrot/code/stash-sense-trainer/data
DST=/home/carrot/code/stash-sense/api/data

for f in performers.db face_facenet.voy face_arcface.voy performers.json faces.json manifest.json; do
  cp "$SRC/$f" "$DST/$f"
done

# Copy optional files if they exist
for f in face_adaface.voy tattoo_embeddings.voy tattoo_embeddings.json; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DST/$f"
done
```

### 5. Post-Copy Verification

```bash
cd /home/carrot/code/stash-sense/api
python -c "
from database_reader import PerformerDatabaseReader
db = PerformerDatabaseReader('data/performers.db')
stats = db.get_stats()
print(f'{stats[\"performer_count\"]} performers, {stats[\"total_faces\"]} faces')
# Test all dataclass mappings
db.get_faces(1); db.get_stashbox_ids(1); db.get_aliases(1)
print('All dataclass mappings OK')
"
```

### 6. Fingerprint Invalidation

Fingerprints in `stash_sense.db` track `db_version` from `manifest.json`. When the version changes, fingerprints are **automatically flagged for refresh** on next analysis run. No manual intervention needed.

The `SCHEMA_VERSION` in `recommendations_db.py` is for the `stash_sense.db` schema, NOT the face recognition data. Only bump it if you change `stash_sense.db` tables.

## manifest.json Format

```json
{
  "version": "2026.02.18",
  "created_at": "2026-02-18T01:09:34Z",
  "performer_count": 108001,
  "face_count": 366794,
  "sources": ["fansdb", "javstash", "pmvstash", "stashdb", "theporndb"],
  "models": {
    "detector": "retinaface_buffalo_sc",
    "alignment": "insightface_norm_crop_5point",
    "facenet_dim": 512,
    "arcface_dim": 512,
    "adaface_dim": 512,
    "tattoo_dim": 1280,
    "flip_averaging": true
  },
  "checksums": {
    "performers.db": "sha256:...",
    "face_facenet.voy": "sha256:...",
    "face_arcface.voy": "sha256:...",
    "face_adaface.voy": "sha256:...",
    "tattoo_embeddings.voy": "sha256:...",
    "tattoo_embeddings.json": "sha256:..."
  }
}
```

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Copying while sidecar is running | Stop sidecar first - it holds read-only locks on performers.db |
| performers.db checksum doesn't match manifest | WAL mode causes drift. Run `PRAGMA wal_checkpoint(TRUNCATE)` then recompute, or update manifest |
| `TypeError: __init__() got an unexpected keyword argument` | New column in trainer DB. Add field to dataclass in `database_reader.py` |
| Forgetting to copy manifest.json | db_version won't update, fingerprints won't invalidate |
| Bumping SCHEMA_VERSION after import | Wrong version - that's for stash_sense.db schema, not face data |

## Key Files

- **Trainer export**: `stash-sense-trainer/build_runner.py` (`_finalize_build`, `_create_snapshot`)
- **Sidecar import**: `stash-sense/api/main.py` (lifespan startup), `config.py` (DatabaseConfig paths)
- **Schema compat**: `stash-sense/api/database_reader.py` (dataclasses)
- **Fingerprint tracking**: `stash-sense/api/recommendations_db.py` (db_version in fingerprint tables)
