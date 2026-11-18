# Driver API v1

The driver API is the stable surface external tools use to drive stash-sense. Its first user is the stash-curator CLI (`sense` commands). Everything else the sidecar serves (`/queue`, `/review`, `/recommendations` and the rest) is internal to the plugin and changes without notice. Tools must only call the endpoints on this page.

- **Base path:** `/api/v1/driver`
- **Versioning:** shapes on this page don't change within v1. New optional request fields and new response keys may be added. Removing or renaming a field, changing its type or meaning, or tightening validation goes to `/api/v2`.
- **Contract tests:** `api/tests/test_driver_api.py` pins every request and response shape. A failing contract test means a breaking change.
- **Direction:** tools call stash-sense, and stash-sense never calls them.

## Auth

Set `STASH_SENSE_API_KEY` in the sidecar's environment. Every driver call must then send it in the `ApiKey` header, the same header Stash uses:

```
ApiKey: <STASH_SENSE_API_KEY>
```

A missing or wrong key gets `401 unauthorized`. When the variable is unset, the driver API is open; `GET /api/v1/driver` reports `auth_required: false`. The key protects only the driver routes. The plugin's internal routes stay open on the LAN, as before.

## Conventions

- Timestamps are ISO 8601 in UTC, e.g. `2026-11-17T10:00:00Z`.
- Ids are integers, except Stash ids (scenes, files, performers), which are strings.
- Every error response has the same body. `code` is stable and meant for branching; `message` is for people. Some codes add fields:

  ```json
  {"error": {"code": "stale", "message": "The library changed since this preview; review the updated plan", "preview": {...}}}
  ```

- A body that doesn't match the schema, including unknown fields, gets `422 invalid_request` with `fields` listing the problems.
- Recommendations of types marked unfinished are hidden unless the user turns on *Settings > Features > Show Unfinished Features*. Hidden types are left out of lists, and naming one directly gets `403 hidden_type`.
- Recommendations that name a scene on the user's exclusion list (*Settings > Curation*) are left out of lists. Naming one directly gets `403 scene_excluded` with `scene_ids`. When Stash can't be read to check the list, the call fails closed with `503 exclusion_unavailable`.

## Index

### `GET /api/v1/driver`

```json
{
  "api_version": 1,
  "sidecar_version": "0.1.3-beta.3",
  "auth_required": true,
  "job_types": [
    {"type": "duplicate_performer", "name": "Duplicate Performer Detection",
     "description": "Finds performers that may be duplicates", "default_priority": 50, "params": {}}
  ],
  "review_types": [
    {"type": "review_demo", "name": "Review Demo", "description": "..."}
  ]
}
```

`job_types` lists the jobs a caller may start. `review_types` lists the recommendation types that can be previewed and accepted. `params` describes the parameters a job type takes; no type takes any yet.

## Jobs

### `POST /jobs`

```json
{"type": "duplicate_performer", "priority": "high", "params": {}}
```

- `priority` (optional) is `"critical"`, `"high"`, `"normal"`, `"low"` or the matching number (0, 10, 50, 100; lower runs first). It defaults to the type's `default_priority`.
- `params` (optional) must be empty or omitted until a job type declares params.

Only one job per type is queued or running at a time.

- `201` returns `{"created": true, "job": <Job>}`.
- `200` returns `{"created": false, "job": <Job>}` when a job of that type is already queued or running; `job` is that existing job.
- Errors: `404 unknown_job_type`, `403 hidden_type`, `400 invalid_params`, `409 media_not_writable` (the job writes media and the media mount is read-only).

### `GET /jobs/{id}`

Returns `{"job": <Job>}`, or `404 not_found`.

```json
{
  "id": 12,
  "type": "duplicate_performer",
  "status": "completed",
  "priority": 10,
  "progress": {"processed": 11, "total": 11},
  "error": null,
  "result": {"analysis_run_id": 4, "items_processed": 11, "recommendations_created": 2, "excluded_skipped": 0},
  "created_at": "2026-11-17T10:00:00Z",
  "started_at": "2026-11-17T10:00:01Z",
  "completed_at": "2026-11-17T10:00:03Z"
}
```

`status` is one of `queued`, `running`, `stopping`, `completed`, `failed` or `cancelled`. `progress.total` is `null` until the job knows its total. `result` is `null` until the job finishes, and stays `null` for job types that don't report one. Analysis jobs report the shape above. `error` holds the failure message when `status` is `failed`.

**Typical loop:** `POST /jobs`, then poll `GET /jobs/{id}` every few seconds until `status` is `completed`, `failed` or `cancelled`. After that, read the new recommendations from the feed.

## Recommendations

### The Recommendation object

```json
{
  "id": 41,
  "type": "review_demo",
  "status": "pending",
  "target_type": "scene",
  "target_id": "1234",
  "confidence": 1.0,
  "evidence": {"tag": "stash-sense-review-demo"},
  "reviewable": true,
  "resolution": null,
  "created_at": "2026-11-17T10:00:03Z",
  "updated_at": "2026-11-17T10:00:03Z"
}
```

- `status` is `pending`, `dismissed` or `resolved`.
- `evidence` is the full payload the analyzer recorded. Its shape depends on the type.
- `reviewable` is true when the type can be previewed and accepted through this API (it is listed in `review_types`). Other types can be read and dismissed only.
- `resolution` is set when `status` is `resolved`: `{"action": "applied", "details": {"apply_log_id": 7, "option": {...}}, "resolved_at": "..."}`.

### `GET /recommendations?type=&status=&since=&cursor=&limit=`

Returns recommendations in the order they last changed (`updated_at`, then `id`), oldest first.

```json
{"recommendations": [<Recommendation>, ...], "next_cursor": "MjAyNi0x...", "has_more": true}
```

- `since`: an ISO 8601 time. Only recommendations updated at or after it are returned. A time without an offset is read as UTC.
- `cursor`: the `next_cursor` from an earlier call. Only rows after that one are returned.
- `limit`: 1 to 500, default 100.
- `has_more` is true when the page is full, so call again with `next_cursor`.

The list doubles as a change feed. Every status change (accept, dismiss, revert) and every re-analysis moves a recommendation's `updated_at` forward. Keep the last `next_cursor` and call again later to receive only what changed since. One recommendation may come back more than once across calls, once per change, so key on `id`.

Rows from the current second are held back until that second ends, so a change shows up within about a second. This keeps a cursor from pointing into a second that can still gain rows.

Errors: `400 invalid_since`, `400 invalid_cursor`.

### `GET /recommendations/{id}`

```json
{"recommendation": <Recommendation>, "preview": <Preview> | null}
```

For a pending, reviewable recommendation, `preview` is the plan accepting it would apply now, with the suggested option. Otherwise it is `null`. Errors: `404 not_found`, `403 hidden_type`, `403 scene_excluded`, and any preview error listed below.

### `POST /recommendations/{id}/preview`

```json
{"option": {"keep_file_id": "88"}}
```

Returns `{"preview": <Preview>}` for the chosen option. A partial option is filled in from the suggestion, and an empty body previews the suggestion. Errors: `409 not_reviewable`, `422 rejected` (the option is not one the type offers, e.g. a file that isn't on the scene), `404 not_found` (the target is gone from Stash).

### The Preview object

```json
{
  "revision": "4f2a9c0d1e3b5a7c9d0e1f2a",
  "summary": "rename, delete 1 file(s)",
  "option": {"keep_file_id": "88", "title": "clip"},
  "default_option": {"keep_file_id": "88", "title": "clip"},
  "is_default_option": true,
  "high_confidence": true,
  "choices": [
    {"field": "keep_file_id", "label": "File to keep", "kind": "pick",
     "values": [{"value": "87", "label": "clip-a.mp4"}, {"value": "88", "label": "clip.mp4"}]},
    {"field": "title", "label": "Title", "kind": "text"}
  ],
  "evidence": {"scene": {"id": "1234", "title": ""}, "files": [...]},
  "plan": {
    "field_changes": [
      {"entity_type": "scene", "entity_id": "1234", "field": "title", "before": "", "after": "clip"}
    ],
    "keep_files": [
      {"file_id": "88", "scene_id": "1234", "path": "/media/x/clip.mp4", "size": 200,
       "fingerprints": {"oshash": "...", "phash": "..."}}
    ],
    "delete_files": [
      {"file_id": "87", "scene_id": "1234", "path": "/media/x/clip-a.mp4", "size": 100, "fingerprints": {...}}
    ],
    "primary_file": {"scene_id": "1234", "before": "87", "after": "88"},
    "bytes_reclaimed": 100
  },
  "trash_path": "/data/.trash"
}
```

- `revision` fingerprints exactly what this plan would do: the option, every field's before and after, and every kept and deleted file. It belongs to this option. A different option, or any library change that alters the plan, gives a different revision.
- `plan` is the complete outcome of an accept. `primary_file` is set when the scene's primary file moves to a kept file before the others are deleted.
- `trash_path` is where deleted files go. An empty string means they are deleted permanently, because Stash has no trash configured.
- `choices` are the inputs a caller may set in `option`. A `pick` lists its `values`; a `text` takes any string.
- `high_confidence` means the type would batch-accept this suggestion without a person looking at it.

### `POST /recommendations/{id}/accept`

```json
{"option": {"keep_file_id": "88", "title": "clip"}, "revision": "4f2a9c0d1e3b5a7c9d0e1f2a", "confirm_delete": ["87"]}
```

The sidecar rebuilds the plan from Stash's current state with `option` and applies it only if its revision equals `revision`. The caller therefore applies exactly the plan it showed.

- `confirm_delete` must list exactly the `file_id`s in `plan.delete_files`. Leave it empty when nothing is deleted.
- `200` returns `{"already_applied": false, "entry": <ApplyLogEntry>, "warnings": []}`.
- **Idempotent:** accepting a recommendation that was already applied returns `200` with `already_applied: true` and the original entry, and writes nothing. That holds even when the second call carries an old revision.
- `warnings` lists bookkeeping that failed after the writes succeeded, e.g. marking the recommendation resolved. The writes still happened.

Errors:

| Status | `code` | Meaning | Extra fields |
| --- | --- | --- | --- |
| 409 | `stale` | The plan for this option no longer matches `revision`. Nothing was written. | `preview`: the fresh Preview |
| 409 | `confirm_delete` | `confirm_delete` doesn't name exactly the files the plan deletes. | `files` (`file_id`, `path`, `size`), `bytes` |
| 409 | `not_pending` | The recommendation is dismissed, or was resolved some other way. | |
| 409 | `not_reviewable` | The type can't be accepted through this API. | |
| 422 | `rejected` | The option is invalid, or the plan breaks a safety rule (e.g. it would delete every file of a scene). Nothing was written. | |
| 502 | `apply_failed` | A Stash write failed part way. | `entry`: the apply log entry recording which steps ran |
| 403 | `scene_excluded` | The recommendation or its plan touches an excluded scene. | `scene_ids` |

### `POST /recommendations/{id}/dismiss`

```json
{"reason": "not a duplicate", "permanent": false}
```

Both fields are optional. A dismissed target is not recommended again by later runs; with `permanent: true` that also holds after its data changes. Returns `{"recommendation": <Recommendation>}`. Dismissing an already dismissed recommendation is a no-op. A resolved one gets `409 not_pending`.

## Apply log

Every accept is logged before its first write, with the plan, the state it replaced, and each step that ran.

### `GET /apply-log?since=&type=&limit=`

```json
{"entries": [<ApplyLogEntry>, ...], "next_since": 9, "has_more": false}
```

Entries are returned oldest first. Pass the last id you saw as `since` to get only newer ones; `next_since` is the id to pass next time. `type` filters by recommendation type, and `limit` is 1 to 500 (default 100).

### `GET /apply-log/{id}`

Returns `{"entry": <ApplyLogEntry>}`, or `404 not_found`.

### The ApplyLogEntry object

```json
{
  "id": 9,
  "action": "apply",
  "status": "applied",
  "rec_id": 41,
  "rec_type": "review_demo",
  "target_type": "scene",
  "target_id": "1234",
  "batch_id": null,
  "reverts_id": null,
  "reverted_by": null,
  "option": {"keep_file_id": "88", "title": "clip"},
  "revision": "4f2a9c0d1e3b5a7c9d0e1f2a",
  "plan": {<same shape as Preview.plan>},
  "result": {"steps": [...], "bytes_reclaimed": 100, "files_deleted": 1, "fields_changed": 1},
  "error": null,
  "trash_path": "/data/.trash",
  "created_at": "2026-11-17T10:05:00Z",
  "completed_at": "2026-11-17T10:05:01Z"
}
```

- `action` is `apply`, or `revert` for an undo made in the Stash UI. On a revert, `reverts_id` names the entry it undid, and that entry's `reverted_by` points back.
- `status` is `applying`, `applied` or `failed`. An `applying` entry is still in flight; re-read it with `GET /apply-log/{id}`. Because the feed moves by id, it won't return that entry again.
- `batch_id` groups entries that a batch accept in the Stash UI applied together.

## Example: curator session

```bash
B=http://10.0.0.5:5000/api/v1/driver; H="ApiKey: $STASH_SENSE_API_KEY"

# Start a job and wait for it
curl -s -H "$H" -H 'Content-Type: application/json' -d '{"type":"duplicate_performer"}' $B/jobs
curl -s -H "$H" $B/jobs/12

# Read what's pending, then keep next_cursor for the next session
curl -s -H "$H" "$B/recommendations?status=pending&limit=50"

# Show one, then apply exactly what was shown
curl -s -H "$H" $B/recommendations/41
curl -s -H "$H" -H 'Content-Type: application/json' \
  -d '{"option":{"keep_file_id":"88","title":"clip"},"revision":"4f2a9c0d1e3b5a7c9d0e1f2a","confirm_delete":["87"]}' \
  $B/recommendations/41/accept

# What was applied since last time
curl -s -H "$H" "$B/apply-log?since=8"
```
