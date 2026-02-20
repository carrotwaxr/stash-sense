# Configuration

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STASH_URL` | Yes | — | URL to your Stash instance (e.g., `http://stash:9999`) |
| `STASH_API_KEY` | Yes | — | Stash API key (Settings > Security > API Key) |
| `DATA_DIR` | No | `/data` | Path to database files inside the container |
| `LOG_LEVEL` | No | `warning` | Logging verbosity: `debug`, `info`, `warning`, `error` |

### Stash-Box API Keys (Optional)

For upstream sync to detect metadata changes on stash-box endpoints, add API keys for each endpoint you want to monitor:

| Variable | Endpoint |
|----------|----------|
| `STASHDB_API_KEY` | [StashDB](https://stashdb.org) |
| `FANSDB_API_KEY` | [FansDB](https://fansdb.cc) |
| `THEPORNDB_API_KEY` | [ThePornDB](https://theporndb.net) |
| `PMVSTASH_API_KEY` | [PMVStash](https://pmvstash.org) |
| `JAVSTASH_API_KEY` | [JAVStash](https://javstash.org) |

These are only needed for upstream sync. Face recognition works without them.

> **Tip**: Get API keys from each stash-box site's user settings page. StashDB requires an account at [stashdb.org](https://stashdb.org).

---

## Volume Mounts

| Container Path | Purpose | Mode |
|----------------|---------|------|
| `/data` | Database files (Voyager indices, performers.db, manifest) | Read-only |
| `/root/.insightface` | Cached model weights (~500 MB) | Read-write |

### Database Files

The `/data` volume should contain the files from a [stash-sense-data release](https://github.com/carrotwaxr/stash-sense-data/releases):

```
/data/
├── performers.db        # Performer metadata (SQLite)
├── face_facenet.voy     # FaceNet512 embedding index
├── face_arcface.voy     # ArcFace embedding index
├── faces.json           # Face-to-performer mapping
├── performers.json      # Performer lookup data
├── manifest.json        # Version and checksums
└── stash_sense.db       # (auto-created) Your local data — do not delete
```

!!! note
    `stash_sense.db` is created automatically by the sidecar and stores your recommendations, settings, and analysis history. It is **not** part of the database release and should not be deleted during updates.

### Model Cache

The `/root/.insightface` volume caches the RetinaFace detection model weights. Without this mount, the model is re-downloaded on every container start (~500 MB download).

---

## Ports

The sidecar listens on port **5000** inside the container. The recommended host port is **6960**:

```
-p 6960:5000
```

---

## Runtime Settings

Most performance and recognition settings are configured via the plugin's **Settings** tab rather than environment variables. The sidecar auto-detects your hardware and sets sensible defaults.

See [Settings Reference](settings-system.md) for the full list of adjustable options.

### Migrated Environment Variables

The following environment variables have been replaced by runtime settings. If detected at startup, they are automatically migrated to the settings database:

| Old Variable | New Setting |
|-------------|-------------|
| `STASH_RATE_LIMIT` | `stash_api_rate` |
| `ENABLE_BODY_SIGNAL` | `body_signal_enabled` |
| `ENABLE_TATTOO_SIGNAL` | `tattoo_signal_enabled` |
| `FACE_CANDIDATES` | `face_candidates` |
