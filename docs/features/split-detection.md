# Split Detection

!!! note "Hidden as unfinished"
    Split detection is hidden until it passes the production-ready checklist ([#131](https://github.com/carrotwaxr/stash-sense/issues/131)). To use it now, turn on **Settings > Features > Show Unfinished Features** and reload the page. Existing recommendations are kept and reappear when the setting is on.

Some files in a library hold more than one scene: a DVD rip with four scenes back to back, a compilation, a long file with a menu loop at the end. Stash Sense finds the boundaries inside these files and proposes where to cut. For each file you see a review card with the cut times, a before and an after still for every cut, and a button that plays the file from just before the cut. Nothing changes until you accept, and even then Stash Sense writes nothing to Stash. It hands the plan to [stash-curator](#handing-the-plan-to-stash-curator), which makes the cuts.

## What it scans

The job considers two kinds of scenes:

- **Scenes over 60 minutes.**
- **Scenes whose file the [Content Index](content-index.md) shows holding another scene's whole file.** A scene that was already split out, but whose source file is still in the library, is the usual case.

It skips:

- scenes on the exclusion list (see [Curation Settings](curation-settings.md))
- the outputs of earlier splits, which are the scenes in the `[curator] Split Cuts` group and, for older runs, the scenes tagged `[curator] split`
- a file whose plan you have already accepted. An accepted plan is never proposed again for the same file.

A title spread over several scenes (one title, several parts) is out of scope.

## Running it

Run **Split Multi-Scene Files** from **Operations**, or schedule it there. It runs weekly by default and uses a lot of CPU. The job:

1. Reads the exclusion list. If Stash can't be reached, the run stops.
2. Finds the candidate scenes.
3. Reads each file's signals, extracting them first if they are not cached.
4. Plans every candidate and writes one recommendation per scene.

Then open the dashboard and look for the **Split Multi-Scene Files** tile. See [Reviewing Recommendations](review.md) for how cards work.

### What it costs

Each file is decoded once, in full, at 6 frames per second and 160 pixels wide, plus one pass over the audio. A 63-minute 360p test file took about 15 seconds. The signals are cached in `splits.db` in the data directory, keyed by the file's oshash and size, so a file is only decoded again if it changes or if a release changes what is extracted (the signals carry a version). Every run plans every candidate again from the cache, which is fast. When the planner improves, open plans are re-scored on the next run, and the job result says how many changed confidence or moved a cut by more than 2 seconds.

The job needs the media mount, like the other jobs that read files. See [Configuration](../configuration.md).

## How cuts are found

The planner works on the cached signals, not on the video again.

- **Blacks.** A stretch of near-black frames, usually with silence or a sharp drop in volume, is a candidate boundary. Many productions put a fade to black between scenes. The cut goes at the **start** of the black. The black and any title card after it belong to the next scene.
- **Static pairs.** Some sources separate scenes with a burst of static instead of a black. Pairs of these are treated as boundaries too.
- **Coverage.** Where the Content Index shows that a stretch of the file is held by another scene's file, the edges of that stretch are cuts. An edge moves to the start of a nearby black run when there is one.
- **Menu tails.** A DVD menu that loops at the end of a file is trimmed off, and the card shows it as `keep until`.
- **Series profiles.** Scenes from one series tend to be cut the same way: how long the black lasts, whether there is a static burst, how long a scene runs. The planner learns this per series from cuts you have accepted before, and uses it to confirm good candidates and to drop odd ones.

Chapter markers and generic scene-change detection are not used. They did not find the boundaries reliably.

## Confidence

Every card has a confidence. It decides whether the card can be taken in the batch accept.

| Confidence | What it means |
| --- | --- |
| **High** | The plan is confirmed by one of two things: a series profile where the scenes are mostly separated by blacks, or Content Index coverage of at least 90% of the file where a detected cut confirms every coverage edge. High plans can be accepted in a batch. |
| **Medium** | The planner found cuts and every resulting scene runs between 5 and 60 minutes. Look at the card before accepting. |
| **Low** | Anything else. A file with no cuts at all is suggested as **leave whole**. |

There is no high confidence for a file just because it looks clear: without a profile or coverage to confirm it, a plan is at most medium. The planner is tested against a regression set of labelled titles. See [the split regression](../research/split-regression.md) for the numbers and what changed after the first run.

## Reviewing a card

Each card shows the file, its length, the confidence and the reasons behind it. Below that is one row per cut with its time, what kind of boundary it is, the evidence, a **before** and an **after** still, and a **play from** button that plays the file from five seconds before the cut. Blacks that were found but not used are listed as unconfirmed blacks. The **Outcome** panel lists the ranges that will be handed to stash-curator.

You can adjust the plan with three fields:

- **Cut times.** The cut times as text, separated by commas. Use `h:mm:ss`, `mm:ss` or plain seconds, for example `22:40, 1:05:10`. Empty with no trims means **leave whole**.
- **Keep from** and **Keep until.** Trim the start or the end of the file, for example to drop a menu loop. Empty keeps the start or the end.

A time you type **snaps** to the start of a black run when one starts within 4 seconds before it to 2 seconds after it. This lets you type a rough time from a player and still cut where the scene starts. The Outcome panel says how many cuts snapped. A time with no black near it is used as typed. The Outcome panel updates to the new ranges as you preview.

### Leave whole

A file with no cuts is suggested as **leave whole**. Accepting it records that decision and the plan is not proposed again for that file.

### Redundant files

A file whose every stretch, outside its first and last seconds, is held by other scenes' files at equal or better quality gets a **redundant** card. It lists the covering scenes as links. Accepting only resolves the card. It deletes nothing and splits nothing. Deleting the redundant file belongs to [duplicate detection](duplicate-detection.md).

## What accept does

Accepting writes **nothing to Stash**. It marks the recommendation resolved and records the plan with it. Because nothing changed in Stash, a split-only accept can't be reverted from the [apply log](review.md#the-apply-log). The log keeps the entry for reference.

## Handing the plan to stash-curator

An accepted plan appears in the [driver API](../driver-api.md) at `resolution.details.split`:

```json
{"scene_id": "7", "file_id": "70", "oshash": "abc", "size": 1000, "duration": 3000.0,
 "ranges": [{"start": 0.0, "end": 1500.0}, {"start": 1500.0, "end": 3000.0}],
 "cuts": [1500.0]}
```

`ranges` has one entry per scene to create, in seconds of the file. A leave-whole decision has `split: null`, and there is nothing to execute.

stash-curator reads accepted plans with `split --from-sense` ([#111](https://github.com/carrotwaxr/stash-sense/issues/111)), cuts the file and creates the scenes. It then reports what it did with `POST /api/v1/driver/recommendations/{id}/split-result`. The result is stored with the recommendation and logged. See [the driver API](../driver-api.md#post-recommendationsidsplit-result) for the fields.

## Seeding series profiles

Profiles start empty and fill in as you accept plans. To start with more, place a `split_seed.json` file in the data directory. On the next run the job imports it once and renames it `split_seed.json.imported`. It holds labelled cuts for the series you have already split by hand. Build it with `scripts/build_split_seed.py`, which reads your existing labels and splits and writes the file. The seed names real series, so it is built locally and never belongs in a repository.

After the seed, every accepted plan with cuts adds a label on each run, and a re-run overwrites its own labels rather than adding duplicates.
