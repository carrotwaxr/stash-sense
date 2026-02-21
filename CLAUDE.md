# Stash Sense - Development Notes

## Starting the Sidecar API

```bash
cd api
source ../.venv/bin/activate
make sidecar
```

The venv activation is required before `make sidecar` to ensure dependencies and `.env` vars are loaded.

- Dev API runs at `http://localhost:5000` with `--reload` for hot-reloading
- API docs at `http://localhost:5000/docs`
- Requires `.env` file in `api/` directory with `STASH_API_KEY` and stash-box API keys

## Architecture

- **`api/`** - FastAPI sidecar (Python) - face recognition, recommendations engine, upstream sync
- **`plugin/`** - Stash plugin (JS/CSS/Python) - UI injected into Stash web interface
- Sidecar runs on dev machine (`http://10.0.0.5:5000`), Stash runs on Unraid (`http://10.0.0.4:6969`)
- Plugin backend (`stash_sense_backend.py`) proxies requests from Stash to sidecar to bypass CSP

## Deploying Plugin to Unraid

```bash
scp plugin/* root@10.0.0.4:/mnt/nvme_cache/appdata/stash/config/plugins/stash-sense/
```

Or deploy specific files only. Hard refresh Stash UI (Ctrl+Shift+R) after deploying JS/CSS changes.

## Testing

```bash
cd api
../.venv/bin/python -m pytest tests/ -v
```

## Field Name Mapping (Upstream Sync)

The diff engine uses intermediate field names. Stash's `PerformerUpdateInput` has different names:

| Diff Engine | Stash Mutation | Notes |
|---|---|---|
| `aliases` | `alias_list` | |
| `height` | `height_cm` | Integer |
| `breast_type` | `fake_tits` | |
| `career_start_year` + `career_end_year` | `career_length` | Combined text |
| `cup_size` + `band_size` + `waist_size` + `hip_size` | `measurements` | Combined text "38F-24-35" |

Translation is handled in `recommendations_router.py:update_performer_fields()`.

## Key Files

- `api/recommendations_router.py` - All recommendation API endpoints
- `api/recommendations_db.py` - SQLite database layer (schema version 9)
- `api/queue_router.py` - Operation queue API endpoints
- `api/queue_manager.py` - Queue execution engine with resource-aware scheduling
- `api/settings_router.py` - Settings and system info API endpoints
- `api/upstream_field_mapper.py` - Field mapping, parsing, and 3-way diff engine
- `api/analyzers/upstream_performer.py` - Upstream change detection analyzer
- `api/analyzers/upstream_scene.py` - Upstream scene change detection analyzer
- `api/analyzers/base_upstream.py` - Base class for upstream analyzers with logic versioning
- `api/stash_client_unified.py` - Stash GraphQL client
- `api/stashbox_client.py` - StashBox GraphQL client
- `plugin/stash-sense-recommendations.js` - Recommendations dashboard UI
- `plugin/stash-sense-settings.js` - Settings and model management UI
- `plugin/stash-sense-operations.js` - Operation queue UI
- `plugin/stash-sense.css` - All styles
- `plugin/stash_sense_backend.py` - Plugin backend proxy
