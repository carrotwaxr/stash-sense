# Performer Identification

Stash Sense identifies performers in your scenes and images using face recognition against a database of 108,000+ performers sourced from multiple Stash-Box endpoints.

## Prerequisites

- **Sprite sheets** generated for scenes you want to identify (Stash: **Settings > Tasks > Generate > Sprites**)
- **Face recognition database** downloaded (plugin **Settings** tab > Database > Update)
- **ONNX models** downloaded (plugin **Settings** tab > Models > Download All)

Without sprite sheets, the sidecar has no frames to analyze. Without the database and models, there's nothing to match against.

---

## How It Works

1. Click **Identify Performers** on any scene page in Stash
2. The sidecar extracts frames from the scene's sprite sheet (no video decoding required)
3. Faces are detected using RetinaFace and aligned via 5-point similarity transform
4. Each face is embedded using two models (FaceNet512 + ArcFace) with flip-averaging for stability
5. Embeddings are searched against Voyager vector indices containing 366,000+ face references
6. Results are clustered by person — the same performer appearing across multiple frames is grouped together
7. Matched performers are shown with confidence scores and one-click tagging

---

## Matching Modes

**Clustered frequency matching** (default) — Groups detected faces by person using cosine distance, then frequency-matches within each cluster. Multi-frame appearances boost confidence — a performer appearing in 30 out of 60 frames is a much stronger signal than a single-frame detection.

**Tagged-performer boost** — Performers already tagged on the scene receive a small distance bonus (+0.03), reducing false negatives for known cast members.

---

## Gallery and Image Identification

Face recognition extends to gallery images, which are typically higher quality and better-framed than video frames.

- Single image or full gallery identification
- Results grouped by performer across images with best/average distance
- Fingerprint caching avoids re-processing on subsequent requests

---

## Additional Signals

Face recognition alone is the primary signal. Two optional signals can improve results when faces are unclear or absent:

- **Body proportions** — MediaPipe pose estimation extracts shoulder-hip ratio, leg-torso ratio, and arm-span-height ratio. Mismatches apply a penalty multiplier. Enabled via Settings.
- **Tattoo presence** — YOLO-based detection. If the query shows tattoos but a candidate has none, a penalty is applied. Matching tattoo locations give a small boost. Requires the optional tattoo detection models.

**Fusion:** `final_score = face_score × body_multiplier × tattoo_multiplier`. Missing signals are neutral (1.0 multiplier).

---

## Performance

On a GTX 1080 (8GB VRAM), a typical scene (60 frames) processes in ~5 seconds. The 3-phase batch pipeline (extract → detect → embed+match) processes all faces in bulk rather than one at a time.

CPU mode works but is significantly slower (~2-3 seconds per frame for face detection vs ~200ms with GPU).

---

## Understanding Results

Each detected person shows:

- **Face thumbnails** — Cropped faces from the scene frames
- **Best match** — Top performer match from the database
- **Distance score** — Lower is better (cosine distance, 0.0 = perfect match)
- **Appearances** — How many frames this person appeared in

Results are split into two groups:

- **Multi-frame detections** — Performers appearing in multiple frames (shown prominently)
- **Single-frame detections** — One-off detections (in a collapsible section, less reliable)

### Result Actions

| Button | Action |
|--------|--------|
| Add to Scene | Links the performer to the scene |
| Already tagged on scene | Shown instead of "Add to Scene" for performers already tagged |
| View on Stash-Box | Opens the performer's page on the source Stash-Box endpoint |
