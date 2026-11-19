# Face recognition failure analysis (#139)

The 2026-02-12 re-evaluation left scene accuracy near 42% and could not say why performers are missed. This benchmark runs the production scene pipeline over about 200 of your own scenes and gives every missed performer one reason. That tells us whether detection, frame choice or matching loses the most, which decides whether #140 (pose-guided frame selection) is next.

Code: `api/benchmark/failure/`. Data and results live under `$DATA_DIR/benchmark/` (`/data/benchmark/` in the container), never in the repo, because the scene list is personal.

## Running it

Run it inside the sidecar container, which has the GPU, the current face DB and (with `MEDIA_PATH_MAP` set) the media mount.

```bash
# 1. Freeze a benchmark set (once). 200 scenes, seed 139, stratified by era x resolution.
docker exec stash-sense python -m benchmark.failure build-set

# 2. Run: writes raw/<scene>.json per scene, then report.md and summary.json.
docker exec stash-sense python -m benchmark.failure run

# Smoke test first if you like
docker exec stash-sense python -m benchmark.failure run --limit 3

# Resume an interrupted run
docker exec stash-sense python -m benchmark.failure run --resume /data/benchmark/runs/<run_id>

# Re-score a run with other thresholds (no GPU work)
docker exec stash-sense python -m benchmark.failure report /data/benchmark/runs/<run_id> --attribution 0.55 --suffix -strict
```

`docker exec` runs as root (it skips the entrypoint's drop to `PUID`), which keeps the InsightFace cache at `/root/.insightface` visible. Files it writes under `/data/benchmark` are handed back to `PUID:PGID` the next time the container starts. The run loads its own copy of the models, so it uses GPU memory alongside the sidecar. Pipeline settings (`num_frames`, `detection_size`, frame extraction workers) come from the sidecar's saved settings, the same values `/identify/scene` uses, and are recorded in `meta.json`.

## What a scene qualifies as

A StashDB-linked scene with a dated file, at least one performer, and every performer linked to StashDB. That last rule makes the expected set complete, so false positives are real. Eras: pre-2010, 2010-2014, 2015-2019, 2020+. Resolution by the file's short side: SD (< 720), 720p, 1080p (including 1440p), 4K. Each stratum gets an equal share of the 200, and shares a small stratum can't fill pass to the others.

## How a scene is scored

The run calls the same code as `POST /identify/scene` in frequency mode: evenly spaced frames, RetinaFace detection, size filter, batch embedding plus index match (`scene_matcher.recognize_detected_faces`), the upscaled screenshot, then `clustered_frequency_matching`. Two deliberate differences:

- No tagged-performer boost. The plugin sends the scene's tagged performers, which here are the answer key.
- No body or tattoo re-ranking. This measures the face pipeline.

A performer counts as **found** when a person with `frame_count > 1` has them as best match. That is the plugin's main group; single-frame persons sit in a collapsed section. A **false positive** is a multi-frame person matched to someone not in the scene. Identity is compared by performer, so a match keyed by any of their stash-box ids counts.

## Miss reasons

For each scene face, the run also computes its fused distance to every expected performer's own DB faces (`gallery.py`, the same weighted FaceNet plus ArcFace cosine production uses). A face is **attributed** to a performer when that distance is at most 0.65 and no other expected performer is closer. Attribution uses the answer key and only explains misses.

Checked in order, first hit wins:

| Reason | Meaning |
|---|---|
| `not_in_db` | No faces for the performer in the face DB |
| `no_face_detected` | No detected face is attributed to them |
| `dropped_by_aggregation` | They were top-1 (within max_distance) on at least one face production saw, but scene aggregation didn't report them as a multi-frame person |
| `small_or_profile` | Every attributed face is under 80 px on the short side or turned past 45 degrees yaw |
| `wrong_match` | A usable face of theirs was detected and embedded, and someone else won it (distance to the right answer and the winner are recorded) |

`dropped_by_aggregation` goes beyond the issue's four reasons. The re-eval found that `min_unique_frames` 2 to 1 recovered 9 of 28 misses, so per-face correct matches lost in aggregation are common, and folding them into `wrong_match` would hide a cheap fix.

## Frame choice vs detection: the keyframe probe

"No face detected in the sampled frames" can't tell a detector miss from a bad pick of frames on its own. For `no_face_detected` and `small_or_profile` misses, the run decodes the scene's keyframes (up to 400, evenly thinned, keyframes only so it's cheap) and looks for a face that is attributed to the performer, at least 80 px, frontal, and matched top-1 to them. The outcome maps each miss to a stage:

| Stage | Misses counted |
|---|---|
| `frame_choice` | No face or small/profile in the samples, but a keyframe elsewhere has a usable, correctly matched face |
| `detection_or_absent` | No face detected, and no keyframe has one either |
| `size_or_pose_throughout` | Small/profile in the samples, and never usable anywhere |
| `matching` | `wrong_match` + `dropped_by_aggregation` |
| `db_coverage` | `not_in_db` |
| `unprobed` | Probe didn't run (no local file and no `--probe-stream`, `--no-probe`, or an error) |

The probe reads the whole file, so by default it only runs on local files from the media mount. `--probe-stream` allows it over Stash's HTTP stream.

The verdict line names the stage with the most misses. `frame_choice` points at #140.

## Detection test set

This measures detector recall on SD footage independently of recognition, which is what `detection_or_absent` needs to be read correctly.

```bash
# Sample 300 SD frames from the benchmark set and pre-draw boxes with a high-recall detector
docker exec stash-sense python -m benchmark.failure detect-sample

# Label: copy /data/benchmark/detection to a machine with a browser and run the labeler there
# (stdlib only, no ML deps), then copy labels.json back.
cd api && python -m benchmark.failure label --dir <copy>    # http://localhost:8139

# Score the production detector against your labels
docker exec stash-sense python -m benchmark.failure detect-score
```

InsightFace's RetinaFace expects BGR input (it calls `blobFromImage(..., swapRB=True)`). Until #147 the sidecar and the trainer fed it RGB. On this set, RGB gave 72.2% recall and 99.2% precision, and BGR gave 75.3% and 97.8%. Both now feed it BGR and still align crops from the RGB image, so embeddings are unchanged. `detect-score` scores production (BGR) by default; `--channel-order rgb` reproduces the old input.

Labeling policy: box every human face where at least one eye is visible, any size, any angle. Accept a frame with no boxes if it has no faces. The score reports recall by face size (< 40, 40-79, 80+ px), precision, and "usable recall" (detected and large enough to pass the production size filter).

## Outputs

`/data/benchmark/runs/<run_id>/`:

- `meta.json`: DB version, resolved pipeline settings, set seed
- `raw/<scene_id>.json`: every face (box, yaw, detection score, top 5 matches, distance to each expected performer), the persons production returned, probe verdicts
- `summary.json`: totals, reason and stage counts, per-stratum tables, wrong-match distance stats, attribution sensitivity at 0.55, 0.65 and 0.75, and one outcome per expected performer
- `report.md`: the same, readable

Compare runs across changes by diffing `summary.json`, or re-score older raw data with `report`.
