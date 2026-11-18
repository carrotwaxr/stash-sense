# Quality Flags

!!! note "Hidden as unfinished"
    Quality flags are hidden until they pass the production-ready checklist ([#133](https://github.com/carrotwaxr/stash-sense/issues/133)). To use them now, turn on **Settings > Features > Show Unfinished Features** and reload the page. Existing recommendations are kept and reappear when the setting is on.

Some files in a library are not what you meant to keep: a download that stopped halfway, a copy ripped from a tube site, a version far smaller than the studio's own release, a format you would rather not have. Stash Sense looks at each scene's primary file and raises up to four flags. Each flagged scene gets one review card where you answer yes or no per flag. Nothing changes in Stash until you accept a card, and by default even accepting changes nothing (see [report-only](#report-only-and-control-cards)).

This is stage 1. It uses what is already in the file and in your library. Reading text off the picture is stage 2 (see [what waits for stage 2](#what-waits-for-stage-2)).

## The four flags

| Flag | What it means | What to do |
| --- | --- | --- |
| **Broken** | The file is missing from disk, will not decode, has no video or no audio, has an index that is gone, or is shorter or longer than Stash says. | Open the file. If it really is damaged or cut short, replace it. A few files the sidecar can't see are flagged too, so check the mount first. |
| **Tube or leak rip** | The file name or its embedded tags point at a tube site or carry a link. | If it is a rip, look for a clean copy. |
| **Below the studio's release** | The file has much less detail than the studio's other scenes from the same years. | Look for the studio's own release of the scene. |
| **Disfavored format** | The file's codec or container is on one of your [disfavored lists](curation-settings.md#disfavored-formats). | Re-encode or replace it, if you care. |

A scene can carry several flags. Each one is a separate yes or no on the card.

## Broken: the tiered check

Checking every file in full would take days, so the check has two tiers.

**Cheap pass, every file.** Three steps, cached by the file's oshash and size so each file is only read again if it changes:

1. One `ffprobe`: are there a video and an audio stream, do their durations agree, and does the file's duration match Stash's (a difference over 2% or 5 seconds, whichever is more).
2. A container index check. It reads the first and last 4 MB, never the whole file, and finds a missing MP4 `moov` atom, an AVI with no index or an MKV with no cue points. A `moov` that is merely at the end of the file (no faststart) is only a note, not damage.
3. A decode of the last 30 seconds, where a cut-short download shows its errors.

The result is **ok**, **warn** (small oddities such as a few seconds of trailing audio, or an index that makes seeking slow) or **suspect** (a hard failure above). Only a suspect continues.

**Deep check, suspects only.** The same 24-window health check that [duplicate detection](duplicate-detection.md) uses: 24 one-second decodes spread across the file. A window that fails is decoded again from 20 seconds earlier, because a seek artifact vanishes on the second try and real damage repeats. The deep result is final, so a file the cheap pass doubts can still come out clean. Deep results are stored in the [Content Index](content-index.md) store and shared with duplicate detection, so a file checked by one is not checked again by the other. If you have not built a content index yet, the deep check still runs, but its result is not kept.

A file that is not on disk is flagged broken but is not cached, so it is checked again next run once the mount is back. A check that timed out or could not start is not cached either; the next run tries again. When more than 2% of the checked files (and more than 10) are missing in one run, the job treats it as a media mount problem: it raises no flag for those files, leaves their open cards as they are, and logs a warning.

### What it costs

The cheap pass reads a few megabytes and decodes 30 seconds per file, two files at a time. On a spinning disk the first run over a large library takes hours, and later runs mostly read the cache. The deep check adds 24 short decodes for each suspect, which is a small fraction of the library. The job runs at low priority and needs the media mount like the other jobs that read files. See [Configuration](../configuration.md).

## Below the studio's release

Each file's **effective detail** is the same number [duplicate detection](duplicate-detection.md) uses to pick a keeper: the file's pixels, capped by what its bitrate can carry in its codec. A 1080p file at a starved bitrate counts as less than 1080p.

A file is compared with its **peers**: other scenes' files from the same studio.

1. The same studio and **the same year** of release.
2. If that group has fewer than 8 other files, the same studio **within 2 years** either side.
3. If that still has fewer than 8, **the whole studio**.

The first group with at least 8 other files is used. A scene with no studio is never flagged, and neither is one whose studio has fewer than 8 other scenes.

A file is flagged only when both are true:

- its effective detail is **below the percentile** of its peers' that you set (default 10th), and
- it is **under 0.6 times the peers' median**.

The second test stops a studio whose files are all alike from flagging its own bottom 10%. The card shows the file's detail against the median, which peer group was used and how many peers it had, and the share of peers that are better.

### Blur and block sharpness

For a below-studio file, Stash Sense also decodes six 3-second windows (at 10%, 25%, 40%, 55%, 70% and 85% of the file) and measures **blur** and **blockiness** with ffmpeg. The card shows the medians. Lower is better for both: a low blur number is a sharp picture, and a high block number means visible compression squares. These numbers are for you to judge. They do not decide the flag. They are cached per file.

### The percentile setting

**Peer percentile** runs from 1 to 50 and defaults to 10. Raise it to see more cards, lower it to see fewer. The 0.6 times median test still applies at any value. Set it in **Stash Sense > Settings > Curation > Quality flags**.

## Tube or leak rips

Stage 1 looks only at clues in the file itself. Each clue has a weight, and the weights combine so that two weak clues can add up:

| Clue | Weight |
| --- | --- |
| A well-known tube site in a container tag (title, comment, encoder and so on) | 0.9 |
| A tube site in the file name | 0.6 |
| Any other link or web address in a container tag | 0.3 |
| The file is below the studio's release (above) | 0.2 |

The score is `1 - (1 - w1)(1 - w2)...`. A card is raised at **0.5 or more**. A tube site in the name alone is enough, and a tube site in a tag is certainly enough. A stray link by itself (0.3) is not, and neither is a link together with low detail (0.44). The card lists each clue as `signal: text`.

### What waits for stage 2

Stage 1 never reads text off the picture. These wait for stage 2, which needs tesseract and more calibration:

- an on-screen **watermark** that stays on for the whole video
- a scrolling **marquee** along the bottom
- a **domain** printed in the frames
- **pre-roll ads**: a short clip at the start that is the same across many files. A trial run found that most hits were shared intro music, not ads, so stage 2 will confirm with shared video frames and count networks rather than single studios.

Until then, a rip with a clean file name and no tags is not flagged.

## Disfavored format

A file whose video codec or container is on one of your [disfavored lists](curation-settings.md#disfavored-formats) raises the format flag, with the reason on the card. It shows **only when nothing else is raised** for the scene, so a broken or ripped file shows its more important problem first. With both lists empty the format flag never shows.

## Running it

Run **Quality Flags** from **Operations**, or schedule it there. It runs weekly by default. The job:

1. Reads the exclusion list. If Stash can't be reached, the run stops. Excluded scenes are skipped.
2. Takes every scene's primary file and runs the cheap pass, then the deep check for suspects.
3. Works out the peer groups and measures sharpness for below-studio files.
4. Writes one recommendation per flagged scene, updating it in place on later runs.

A card you dismissed is not created again for the same scene. A card you answered stays answered while the scene keeps the same file, even when its peer evidence shifts; it comes back for review only when the file is replaced. A card whose file no longer has any flag, because you replaced the file or changed a setting, is resolved as stale on the next run. A card you have not answered is kept until then. The dashboard tile is **Quality Flags**. See [Reviewing Recommendations](review.md) for how cards work.

Quality cards are never high confidence and are not part of the batch accept. Every flag needs your own yes or no.

## Report-only and control cards

**Report-only is on by default.** While it is on, accepting a card only records your answers. It adds nothing to any group and changes nothing in Stash. The apply log keeps the entry. This is how you check the flags before you trust them.

While report-only is on, the job also keeps up to **20 control cards** open. A control card is a random scene that had **no flag**, picked from scenes that never had a quality card. It looks like any other card but offers all four flags with the default **no**. Ask yourself whether anything is wrong with the file, and answer yes for what is.

### How your answers become calibration labels

Every accepted card stores your yes or no per flag. A dismissed card counts as no for every flag. `scripts/quality_calibration.py` reads them, and writes nothing:

```bash
python scripts/quality_calibration.py /path/to/stash_sense.db --unflagged 25000
```

- **Precision** per flag: yes answers divided by reviewed flagged cards. Of the flags raised, how many were right.
- **Recall estimate** per flag: from the control cards, the share of unflagged scenes you marked yes gives a miss rate. Combined with the number of unflagged scenes you pass in `--unflagged`, that estimates how many real problems the job did not flag, and so what share of all problems it did.

Use `--json` for the raw numbers. A few dozen answers per flag give a rough picture. Hundreds give a useful one.

## Work-queue groups

Once the flags are right, turn report-only off to have accepted cards act. Each flag has its own **group ID** in **Stash Sense > Settings > Curation > Quality flags**: broken, rip, below studio and format. Create the groups in Stash first and paste their IDs.

With report-only off, accepting a card adds the scene to the group of each flag you answered **yes**. A flag with **no** is left alone. A flag with no group set writes nothing, and a scene already in the group is not added again. The groups are your work queue: work through them with whatever tools you replace files with.

A flag answer other than yes or no is refused. If the scene is deleted, or its file is not the one the card was made for, the card can't be applied.

## Settings

All quality settings are stored in the `stash-sense` plugin's configuration in Stash, next to the [curation settings](curation-settings.md).

| Setting | Default | Meaning |
| --- | --- | --- |
| Report only | on | Accepting records answers and changes nothing. |
| Broken, rip, below studio and format group IDs | empty | One Stash group per flag. Used only when report-only is off. |
| Below-studio percentile | 10 | 1 to 50. A file below this percentile of its peers, and under 0.6 times their median, is flagged. |
