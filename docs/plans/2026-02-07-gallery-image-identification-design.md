# Gallery & Image Identification Design

## Overview

Add performer identification (face recognition) to Stash Gallery and Image pages. Currently identification only works on Scene pages via frame extraction. Gallery images provide a more controlled environment for face recognition — higher quality, better framed photos compared to extracted video frames.

## Use Cases

- **Bulk tagging**: Identify and tag performers across an entire gallery at once
- **Verification**: Check existing performer tags and find missing performers
- **Testing**: Gallery images serve as a better-controlled test environment for face recognition than scene frame scraping

## API Endpoints

### `POST /identify/image`

Identify performers in a single Stash image by ID.

**Request:**
```json
{
  "image_id": "27839",
  "top_k": 5,
  "max_distance": 0.6,
  "min_face_confidence": 0.5
}
```

**Flow:**
1. Fetch image URL from Stash via GraphQL (`findImage` -> `paths.image`)
2. Download and run through existing `recognize_image()` pipeline
3. Store fingerprint in `image_fingerprints` / `image_fingerprint_faces`
4. Return same structure as existing `/identify` response

### `POST /identify/gallery`

Identify all performers across a gallery.

**Request:**
```json
{
  "gallery_id": "797",
  "top_k": 5,
  "max_distance": 0.6,
  "min_face_confidence": 0.5
}
```

**Flow:**
1. Fetch gallery metadata + all image IDs/URLs from Stash via GraphQL
2. Process each image through `recognize_image()`
3. Aggregate results per-performer across all images
4. Store per-image fingerprints
5. Return aggregated results

**Response:**
```json
{
  "gallery_id": "797",
  "total_images": 47,
  "images_processed": 47,
  "faces_detected": 62,
  "performers": [
    {
      "performer_id": "abc123",
      "name": "Performer Name",
      "best_distance": 0.32,
      "avg_distance": 0.41,
      "image_count": 23,
      "image_ids": ["101", "102", "105", ...],
      "all_matches": [...]
    }
  ]
}
```

## Data Model

### `image_fingerprints` table

| Column | Type | Notes |
|--------|------|-------|
| stash_image_id | TEXT PK | Stash image ID |
| gallery_id | TEXT nullable | Set when scanned as part of gallery, NULL for individual scans |
| faces_detected | INT | Number of faces found |
| db_version | INT | Face database version at scan time |
| created_at | TIMESTAMP | |
| updated_at | TIMESTAMP | |

### `image_fingerprint_faces` table

| Column | Type | Notes |
|--------|------|-------|
| stash_image_id | TEXT FK | References image_fingerprints |
| performer_id | TEXT | Universal performer ID from face DB |
| confidence | REAL | Match confidence |
| distance | REAL | Match distance |
| bbox_x, bbox_y, bbox_w, bbox_h | REAL | Normalized bounding box (0-1) |

No separate `gallery_fingerprints` table. A gallery's fingerprint is the set of `image_fingerprints` rows where `gallery_id` matches. If an image is scanned via gallery and later individually, it's the same row.

## Aggregation Strategy (Galleries)

Gallery aggregation is simpler than scene identification — no temporal clustering or cross-frame face grouping needed since gallery images are independent photos.

1. Process each image through `recognize_image()`
2. For each detected face, record best match and distance
3. Group by performer ID across all images
4. Per unique performer, compute:
   - **Best distance** (lowest across all images)
   - **Average distance** across appearances
   - **Image count** and **image IDs** (for per-image tagging)
5. Filter: include only performers with 2+ appearances OR single match with distance < 0.4
6. Sort by image count descending, then best distance ascending

## Plugin UI — Image Page

Route detection: add `image` type matching `/images/(\d+)/` in `getRoute()`.

- Inject "Identify Performers" button in image detail header
- On click: call backend proxy `mode: "identify_image"` -> sidecar `/identify/image`
- Show results in same modal style as scene identification
- Each face with top matches, accept/reject per match
- Accept calls Stash `imageUpdate` mutation to add performer

## Plugin UI — Gallery Page

Route detection: add `gallery` type matching `/galleries/(\d+)/` in `getRoute()`.

- Inject "Identify Performers" button in gallery detail header
- On click: show progress spinner ("Identifying performers in gallery...")
- Call backend proxy `mode: "identify_gallery"` -> sidecar `/identify/gallery`
- Results modal shows identified performers, each with:
  - Performer name + thumbnail
  - Confidence and image count (e.g. "Found in 23/47 images")
  - Toggle: **"Gallery only"** (default) vs **"Gallery + Images"**
  - Accept / Reject per performer
- **Accept All** button applies all results with current toggle settings
- Accept behavior:
  - Gallery only: `galleryUpdate` mutation adding performer
  - Gallery + Images: `galleryUpdate` + `imageUpdate` for each image where performer was found

## Backend Proxy

Two new modes in `stash_sense_backend.py`:

- `identify_image` — POST to `/identify/image`, timeout 30s
- `identify_gallery` — POST to `/identify/gallery`, timeout 300s

Tagging is done from plugin JS side via `stash.callGQL()`, consistent with scene identification.

## Stash GraphQL Queries

New queries in `stash_client_unified.py`:

- `get_image(image_id)` — fetch `paths { image }`, `performers { id, name }`
- `get_gallery(gallery_id)` — fetch `images { id, paths { image } }`, `performers { id, name }`, image count

## Progress & Timeouts

V1 uses synchronous processing with generous timeout (300s). No streaming progress for now. If galleries regularly exceed timeout, async processing + polling can be added in a follow-up.

## Files to Modify

- `api/main.py` — New `/identify/image` and `/identify/gallery` endpoints
- `api/recommendations_db.py` — New fingerprint tables, schema version bump
- `api/stash_client_unified.py` — New gallery/image GraphQL queries
- `plugin/stash-sense.js` — Image page identification UI
- `plugin/stash-sense.js` — Gallery page identification UI
- `plugin/stash-sense-core.js` — Route detection for image/gallery pages
- `plugin/stash_sense_backend.py` — New proxy modes
- `plugin/stash-sense.css` — Styles for gallery results (toggle, image count)
