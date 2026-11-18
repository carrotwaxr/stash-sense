# Content Index

> Unfinished feature: the job is hidden until **Settings > Features > Show Unfinished Features** is on. It becomes visible when [duplicate detection](duplicate-detection.md) (#130) ships. That job reads this index, so run Content Index first.

The content index answers one question for the whole library: does the content of one file appear inside another, and where? Stash's perceptual hash can't answer it. A trimmed intro, letterboxing or a watermark moves every frame it samples, and a clip inside a longer file never matches. [Duplicate detection](duplicate-detection.md), split detection and the quality flags build on this index.

## What it stores

For every file, once per content (the same bytes in two places count once):

- **An audio fingerprint**: Chromaprint over the whole soundtrack, about 8 values a second. It finds a clip inside a longer file and gives the offset to a fraction of a second.
- **A video signature**: a 64-bit hash of every keyframe, with its time. It catches what audio can't: the same picture with another soundtrack, such as a dubbed version or a re-scored edit.
- **Overlaps**: for each pair of files sharing audio, which stretch of one file is which stretch of the other, and how closely they match.

Everything lives in `content_fp.db` in the data directory (about 1.8 GB for 28,000 files), apart from `stash_sense.db`. The lookup index (`content_index/`, about 1.3 GB) sits beside it and is rebuilt when files are added.

## Running it

Run **Content Index** from **Operations**, or schedule it there. Each run:

1. Reads the exclusion list (see [Curation Settings](curation-settings.md)). A file is skipped when any of its scenes is excluded. If Stash can't be reached, the run stops.
2. Fingerprints every file that has no fingerprint yet. A run with up to 200 files to do takes the most recently added scenes first. A larger backlog, such as the first pass over a library, is taken in a scattered order so that parallel workers read from different disks.
3. Rebuilds the lookup index if files were added, then looks up each new file and records its overlaps.

The job is safe to stop or restart. A finished file is never fingerprinted again, so a stopped run carries on where it left off. It also gives way to higher-priority CPU work.

It reads files through the media mount (see [Configuration](../configuration.md)). If 50 files in a row are missing, the run stops with a message to check `MEDIA_PATH_MAP` and the `/media` mount, so a bad mount doesn't make it skip the whole library.

### CPU use

ffmpeg runs under the **Media Job Priority (nice)** and **Media Job Threads** settings in **Settings > Performance**. **Content Index Workers** (default 2) sets how many files are processed at once. Each file costs about one decode of its audio plus its keyframes, a few seconds of CPU for a 30-minute scene, so the first pass over a large library is limited by disk reads, not CPU: each ffmpeg uses about a tenth of one core.

On an array of spinning disks (unRAID), each worker keeps one disk busy while it reads a file. Playback from a disk the job is reading can stutter; other disks are unaffected. Two workers on the same disk seek against each other and read slower than one, which is why a large backlog is taken in scattered order. More workers finish the first pass sooner but keep more disks busy at once.

## Importing a store built elsewhere

A store built by the standalone worker (`python -m content_fingerprint.cli`) on another machine can be imported instead of fingerprinting those files again:

1. Stop the worker that built it, so its last writes are in the main file (or copy the `-wal` file alongside, renamed to match), then copy it into the data directory as `content_fp_import.db` (on unRAID: `/mnt/nvme_cache/appdata/stash-sense/data/`).
2. Run **Content Index**.

Before copying anything, the job fingerprints a sample of the imported files itself (a few of each file type) and compares the results byte for byte. Fingerprints depend on the ffmpeg build: ffmpeg 4.4 and 6.1 agree on AAC and MP3 audio but not on WMA, for example. If any sample differs, nothing is imported, the file is renamed to `content_fp_import.db.refused`, and the run's result says which file types differed and which ffmpeg builds were involved. The job then fingerprints those files itself. After a successful import, the file is renamed to `content_fp_import.db.imported` and can be deleted.

Only files still in the library are imported. Video signatures and overlaps are not imported; the job computes them.

## How matches are judged

Audio overlaps are kept when a stretch of at least 20 seconds matches with a bit error rate of 0.25 or less. Measured on a real library:

| Bit error rate | Meaning |
| --- | --- |
| 0.04-0.2 | The same content (a re-encode, a remux, a re-release) |
| 0.3-0.4 | Grey zone: often different scenes of one production sharing a music bed |
| 0.45 and up | Unrelated |

A pair of files sharing audio is classed by how much of each file the overlap covers: **duplicate** (most of both), **contained** (most of the shorter file sits inside the longer one, like a scene inside a DVD rip), **partial** (a long unbroken shared stretch) or **incidental** (short scraps, such as an intro sting or a music bed).

Video signatures are compared pair by pair, by the consumer that needs them. A match reports the time offset and the share of comparable keyframes that match. Only keyframes, not every frame, are decoded, which costs about 40 times less. Two encodes rarely put keyframes at the same moments, though, so only some keyframes can be compared. Measured on real files: duplicates at 0.25-1.0, different scenes from the same studio at 0.03 or less, and unrelated files share no matching keyframes at all.

## Limits

- A file without an audio track has no audio fingerprint; its video signature still works.
- One pair in sixteen real duplicates tested (an AVI against an MP4 re-render) gave a video offset 10 seconds away from the audio's. Prefer the audio offset when both exist.
- Overlaps already recorded stay in the store when a scene is later excluded; features reading them filter by the exclusion list.
