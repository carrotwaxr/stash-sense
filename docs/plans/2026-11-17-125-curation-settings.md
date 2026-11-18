# #125 Shared curation settings stored in Stash

## Contract

Five STRING settings on the `stash-sense` Stash plugin (manifest `settings`), comma-separated, empty by default:

| Key | Meaning |
| --- | --- |
| `excludeTagIds` | Tag ids; a scene with the tag or any descendant tag is excluded |
| `excludeGroupIds` | Group ids; sub-groups included |
| `excludeStudioIds` | Studio ids; child studios included |
| `disfavoredVideoCodecs` | Lowercase codec names as Stash reports `video_codec` (e.g. `hevc`) |
| `disfavoredContainers` | Lowercase container names as Stash reports file `format` (e.g. `wmv`) |

Any tool reads them from `configuration { plugins }["stash-sense"]`. Stash's `configurePlugin` replaces the whole map, so writers read, merge, then write.

## Sidecar

- `api/curation_settings.py`: parse, load from Stash, resolve the excluded scene-id set (three `findScenes` id queries, depth -1), a TTL cache (60 s) with `invalidate()`, `rec_scene_ids()` and the SQL fragment that mirrors it, `require_not_excluded()` raising `SceneExcluded`. Stash unreachable -> `ExclusionUnavailable` (fail closed).
- Stash client: `get_plugin_configuration(plugin_id)`, `find_scene_ids(scene_filter)`. FakeStash models tag/group/studio hierarchies and the overwrite semantics of `configurePlugin`.
- Hooks:
  - `BaseAnalyzer.create_recommendation` skips recs touching excluded scenes; analysis jobs resolve the set at start.
  - Face fingerprint generation skips excluded scenes.
  - `RecommendationsDB.get_recommendations/count_recommendations/get_recommendation_counts` take `exclude_scene_ids`; the router passes the current set.
  - Review: `ReviewService._build` and batch eligibility refuse excluded scenes.
  - Writes read the list fresh (max_age=0); listings may fall back to the last good list.
  - Review checks the scenes the plan writes, not only the ones the rec names.
  - `scene_fingerprint_match` and upstream scene skip excluded scenes before any stash-box call.
  - Analysis job: when scenes leave the list, that analyzer runs in full once (watermarks moved past them).
  - Merge-performers refused when a source performer is in an excluded scene; stashbox create/link-performer guarded.
  - Scene-naming endpoints refuse with 403: `/identify/scene`, `/recommendations/scene/{id}`, `actions/delete-scene-files`, `merge-scenes`, `delete-scene`, `update-scene`, `accept-fingerprint-match`, `accept-all-fingerprint-matches` (skips excluded).
- `api/curation_router.py`: `GET /curation/settings` (parsed settings, excluded scene count, error), `POST /curation/refresh`.
- `duplicate_scene_files` keeper score: drop the hardcoded codec bonus; disfavored codec/container is the tie-breaker.

## Plugin

- Manifest declares the five settings.
- Settings tab: "Curation" section with tag/group/studio pickers (search via Stash GraphQL) and two text inputs; saves with read-merge-write `configurePlugin`, then `curation_refresh`.
- Backend proxy modes `curation_settings`, `curation_refresh`.

## Out of scope

- Content fingerprint CLI (`content_fingerprint/stash_source.py`): #129 turns it into a job and honors the list there.
- Using disfavored formats in #130 keeper suggestions and the #133 quality flag.
