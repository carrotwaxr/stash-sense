# Stash Plugin Setup

## Installation

### 1. Add the plugin source

In Stash, go to **Settings > Plugins > Available Plugins** and click **Add Source**:

- **Name**: `Stash Sense`
- **URL**: `https://carrotwaxr.github.io/stash-sense/plugin/index.yml`

### 2. Install the plugin

Find **Stash Sense** in the available plugins list and click **Install**.

### 3. Configure the sidecar URL

Go to **Settings > Plugins > Stash Sense** and set the **Sidecar URL** to the address of your Stash Sense container (e.g., `http://10.0.0.4:6960`).

### 4. Open the dashboard

Navigate to **`/plugins/stash-sense`** in your Stash UI to access the Stash Sense dashboard. This is where you'll find the Settings tab, Recommendations dashboard, and Operation Queue.

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

- Verify sidecar is running: `curl http://localhost:6960/health`
- Check sidecar URL in plugin settings matches your container's host and port
- If using Docker, ensure network connectivity between Stash and the sidecar

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

The main dashboard is accessible at **`/plugins/stash-sense`** in your Stash UI. It shows all recommendation types:

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
