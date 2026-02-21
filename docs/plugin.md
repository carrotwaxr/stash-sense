# Stash Plugin Setup

## Installation

### 1. Copy plugin files to Stash

Copy the `plugin/` directory contents to your Stash plugins folder:

```
~/.stash/plugins/stash-sense/
├── stash-sense.yml
├── stash-sense.js
├── stash-sense.css
└── stash_sense_backend.py
```

Or if using Docker:

```
/root/.stash/plugins/stash-sense/
```

### 2. Reload plugins

In Stash: **Settings → Plugins → Reload Plugins**

### 3. Configure the plugin

In Stash: **Settings → Plugins → Stash Sense**

| Setting | Value |
|---------|-------|
| Sidecar URL | `http://localhost:5000` or your Stash Sense host |

## Usage

### Identifying Performers in a Scene

1. Navigate to a scene page in Stash
2. Click the **"Identify Performers"** button
3. Wait for analysis (uses scene's sprite sheet)
4. Review results in the modal:
   - Detected face thumbnail
   - Best match from StashDB
   - Confidence score
5. Click **"Add to Scene"** to link the performer

### Understanding Results

Each detected face shows:

- **Thumbnail**: Cropped face from the scene
- **Match**: Best matching performer from StashDB
- **Confidence**: Lower is better (cosine distance)
- **Appearances**: How many frames this face appeared in

Results are grouped by person - the same performer appearing in multiple frames is clustered together.

### Result Actions

| Button | Action |
|--------|--------|
| Add to Scene | Links performer to scene (if in your library) |
| View on StashDB | Opens performer page on stashdb.org |
| Not in Library | Performer matched but not in your Stash |

## Requirements

- Scene must have a **sprite sheet** generated
- Stash Sense sidecar must be running and accessible

### Generating Sprite Sheets

If a scene doesn't have sprites:

1. Go to scene page
2. Click **Generate** → Enable **Sprite**
3. Or bulk generate: **Settings → Tasks → Generate → Sprites**

## Troubleshooting

### "Failed to connect to Stash Sense"

- Verify sidecar is running: `curl http://localhost:5000/health`
- Check sidecar URL in plugin settings
- If using Docker, ensure network connectivity

### "No faces detected"

- Scene may not have clear face shots
- Sprite sheet may be low quality
- Try a different scene to verify setup

### "Performer not in library"

The face matched a StashDB performer who isn't in your Stash library. Options:

1. Click "View on StashDB" to see the performer
2. Manually add the performer to your Stash
3. The plugin shows StashDB ID for easy lookup

### Plugin not appearing

1. Check plugin files are in correct location
2. Reload plugins in Stash settings
3. Check Stash logs for errors

---

## UI Components

### Recommendations Dashboard

The main dashboard (accessible from the Stash navigation) shows all recommendation types:

- **Upstream Changes** — Performer field updates detected from stash-box endpoints, with a 3-way diff view showing local, upstream, and original values
- **Duplicates** — Candidate duplicate scenes identified by face fingerprint matching
- **All Recommendations** — Combined view of face match suggestions, upstream sync proposals, and duplicate candidates with filtering and bulk actions

### Settings Tab

Found under **Settings > Plugins > Stash Sense**, this tab provides:

- **Sidecar URL** configuration
- **Model Management** — Download or remove optional ONNX models (e.g., AdaFace) directly from the UI
- **Hardware tier** and current setting overrides display

### Operation Queue

The operations panel shows:

- **Job Status** — Real-time status of running, queued, completed, and failed jobs with progress indicators
- **Scheduling** — Configure recurring analysis jobs (interval, enable/disable) that automatically re-queue on a timer

### Scene Upstream Sync Detail

When viewing a scene, the upstream sync detail view shows:

- Side-by-side comparison of local performer data vs. upstream stash-box data
- Field-level diffs with accept/reject controls for individual changes
- Links to the performer's stash-box page for manual review
