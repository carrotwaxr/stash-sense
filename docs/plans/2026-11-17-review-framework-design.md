# Review framework and dashboard entry point (#127)

## Goal

One review surface for every curation recommendation: the card is the preview, accept is the apply, every apply is logged, metadata applies can be reverted, and user corrections become labeled examples.

## Shape

**Types describe, the framework writes.** A review type (`api/review/types.py: ReviewType`) turns a recommendation and a chosen option into a `Plan`:

- `field_changes`: entity, id, Stash mutation field, `before`, `after` (both in mutation-input shape, so revert writes `before` back verbatim)
- `primary_file`: the scene's new primary file when a deletion would remove the current one
- `keep_files` / `delete_files`: id, path, size, Stash fingerprints

Types never call Stash mutations. One executor (`api/review/executor.py`) applies any plan, so every write lands in the apply log by construction.

**Plan token.** `Plan.token()` hashes the option, the field changes and the delete list. Accept carries the token from the preview the user saw. The sidecar rebuilds the plan from Stash's current state; a different token means the library changed since the preview, and the accept is refused with the fresh preview (HTTP 409). This is also the revision check #126 needs.

**Delete confirm.** Accept must carry `confirm_delete`: the exact set of file ids the plan deletes. Any mismatch is a 409 that lists each path and size. Batch items carry it too, so no request deletes a file it did not name.

**Batch accept.** Batch preview lists pending items the type marks high-confidence, each built with the type's default option, plus a summary (items, files, bytes, field changes). Batch accept refuses the whole request if any item is not high-confidence. Batch requests cannot carry an option, so a file the user picked by hand is never batch-deleted.

**Apply log** (`apply_log` table, schema v10). The row is written with `status='applying'` and the prior state before the first write, then completed as `applied` or `failed` with the steps that ran. Prior state holds field `before` values and each deleted file's path, size and fingerprints (oshash, md5, phash from Stash; no full-file hashing). It also records Stash's `deleteTrashPath`: Stash has had a delete trash since v0.30, so the log says whether a deleted file went to trash or was gone for good.

**Revert.** Allowed for an `applied` entry with field changes and no prior revert. If a field's current value no longer equals the applied `after`, revert is refused with the drifted fields unless `force`. A revert is its own log entry (`action='revert'`, `reverts_id`), and the recommendation reopens as pending. Deleted files are not restored; the response says so.

**Labels** (`review_labels` table). When the accepted option differs from the type's default, the framework stores suggested, chosen and the evidence snapshot. `GET /review/labels?type=` exports them as a regression set.

**Media.** `GET /review/media/still?scene_id=&t=` runs ffmpeg on the primary file through the media mount and returns a JPEG. The plugin gets it base64 through the backend proxy. Playback uses Stash's own scene stream with a `#t=` media fragment; no clip rendering.

**Entry point.** The plugin patches `MainNavBar.MenuItems` to add a "Stash Sense" link to `/plugins/stash-sense`, with Stash's own nav markup.

## Test type

`review_demo` (hidden unfinished type, analyzer + job): for each scene carrying the tag `stash-sense-review-demo`, propose a title from the primary file's basename and, when the scene has several files, keep the largest and delete the rest. High-confidence when the scene has one file, or every file shares an oshash or md5 with the keeper. Picking a different keeper records a label. Scoped to the tag so it never touches a library that didn't opt in.

## Out of scope

- Moving the existing ad-hoc `/recommendations/actions/*` writes onto the framework: their types are replaced by #130 and later features.
- The exclusion list (#125). The framework has one choke point for it: `ReviewType.scene_ids(rec)`; #125 plugs its refusal in there. The checklist box stays open until then.
