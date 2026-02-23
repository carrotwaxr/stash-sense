# Plugin Usage

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

Navigate to **`/plugins/stash-sense`** in your Stash UI to access the Stash Sense dashboard.

## Identifying Performers in a Scene

1. Navigate to a scene page in Stash
2. Click the **"Identify Performers"** button
3. Wait for analysis (uses the scene's sprite sheet)
4. Review results in the modal:
   - Detected face thumbnails grouped by person
   - Best match from the performer database
   - Distance score (lower is better)
5. Click **"Add to Scene"** to link the performer

See [Performer Identification](features/performer-identification.md) for details on how matching works and how to interpret results.

## Recommendations Dashboard

The main dashboard is accessible at **`/plugins/stash-sense`** in your Stash UI. It shows all actionable recommendations:

- **Upstream Changes** — Field updates detected from Stash-Box endpoints, with 3-way diff views
- **Duplicates** — Candidate duplicate scenes with confidence scores
- **All Recommendations** — Combined view with filtering and bulk actions

Each recommendation type runs as a background analyzer with incremental watermarking — only items modified since the last run are re-processed.

## Settings Tab

The Settings tab in the dashboard provides:

- **Database** — Current version, check for updates, download/update
- **Models** — Download or remove ONNX models
- **Stash-Box Endpoints** — View auto-discovered endpoints, enable/disable for upstream sync, refresh
- **Hardware** — Current tier and setting overrides
- **Runtime Settings** — Adjust performance, recognition, and signal settings

See [Settings Reference](settings-system.md) for the full list of adjustable options.

## Operation Queue

The Operations tab shows:

- **Job Status** — Real-time status of running, queued, completed, and failed jobs with progress indicators
- **Manual Triggers** — Start analysis runs, upstream sync, scene tagging, and other operations on demand
- **Scheduling** — Configure recurring analysis jobs with interval and enable/disable controls

Jobs run in the background with resource-aware scheduling — GPU-intensive work won't overlap with other GPU jobs, and network-bound operations respect rate limits.

## Requirements

- Scene must have a **sprite sheet** generated for performer identification
- Stash Sense sidecar must be running and accessible

### Generating Sprite Sheets

If a scene doesn't have sprites:

1. Go to the scene page
2. Click **Generate** > Enable **Sprite**
3. Or bulk generate: **Settings > Tasks > Generate > Sprites**

## Troubleshooting

### "Failed to connect to Stash Sense"

- Verify sidecar is running: `curl http://your-host:6960/health`
- Check Sidecar URL in plugin settings matches your container's host and port
- If using Docker, ensure network connectivity between Stash and the sidecar

### "No faces detected"

- Scene may not have clear face shots
- Sprite sheet may be low quality or missing
- Try a different scene to verify setup

### Plugin not appearing

1. Check plugin files are in the correct location
2. Reload plugins in Stash settings
3. Check Stash logs for errors
