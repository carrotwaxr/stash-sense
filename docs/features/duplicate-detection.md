# Duplicate Detection

!!! note "Hidden as unfinished"
    Duplicate Scenes is hidden until it passes the production-ready checklist. It replaces the legacy **Duplicate Scenes (legacy)** and **Duplicate Scene Files (legacy)** types ([#130](https://github.com/carrotwaxr/stash-sense/issues/130)), which are hidden too. To use it now, turn on **Settings > Features > Show Unfinished Features** and reload the page. Existing recommendations are kept and reappear when the setting is on.

Stash Sense finds copies of the same scene that Stash's phash matching misses, and helps you merge them and keep the best file. For each set of duplicates you see a review card. It shows both scenes, the file Stash Sense would keep and why, and stills from the matching moments. Nothing changes until you accept.

## What it finds

Phash compares a few frames. A trimmed intro, black bars, a re-encode, an upscale or a watermark moves them, and a scene inside a longer file never matches. Duplicate detection compares the sound and the picture instead, so it finds:

- trims and different intros or outros
- letterboxed, cropped or watermarked copies
- re-encodes and upscales
- a scene that sits inside a longer file, such as a DVD rip
- the same scene with another soundtrack, such as a dub

## What it needs

- **The [Content Index](content-index.md) job**, for the best results. Run it first. It has already fingerprinted the audio and keyframes of your files, so this job reads them from the index instead of decoding the files again. Without the index, the job fingerprints the audio of each file it checks, which is slower.
- **Sprites in Stash.** Stash's generated sprite sheets give the job a way to compare pictures. Generate them for your scenes in Stash (**Settings > Tasks > Generate**). A scene without a sprite sheet is still checked by audio and keyframes.
- **The media mount**, so the job can read the files. See [Configuration](../configuration.md).

## Running it

Run **Duplicate Scenes** from **Operations**, or schedule it there. The job:

1. Reads the exclusion list (see [Curation Settings](curation-settings.md)). If Stash can't be reached, the run stops.
2. Finds candidate pairs.
3. Checks each pair against the evidence.
4. Groups the pairs that hold up into sets and writes one recommendation per set.
5. Health checks the file it would keep in each set that would delete anything.

Then open the dashboard and look for the **Duplicate Scenes** tile. Each card is one set. See [Reviewing Recommendations](review.md) for how cards work.

## How candidates are found and checked

Comparing every scene with every other would take too long. The job first collects pairs from cheap sources:

- the same Stash-Box ID
- the same studio and title
- the same studio, date and performers
- a close phash
- a shared URL
- two files in one scene
- an overlap the Content Index found in the audio

Each pair is then checked, in this order:

1. **Audio fingerprint.** The job compares the soundtracks. A low error rate means the same recording, and it also gives the offset between the two files.
2. **Keyframes.** The job compares the stored keyframe signatures of both files. This catches the same picture with a different soundtrack.
3. **Sprite tiles.** The job compares the tiles of the two sprite sheets. It only does this when audio and keyframes can't settle the pair. Tiles that recur across many scenes of a studio, such as logos and intro cards, are ignored.

A match on sprites is good evidence. A miss proves nothing. Sprites sample only 81 moments, and a trimmed copy lands on different ones. So a pair with no sprite match is never called different on that alone.

A pair that shows no evidence at all, because none of the three could be computed, is dropped. There is nothing for you to judge.

## The classes

Every card has a class. The class decides what the card offers and what accept does.

| Class | What it means | What the card offers |
| --- | --- | --- |
| **Duplicate** | The same content in both scenes. | Merge the scenes, keep the best file, delete the rest. You can pick another file, or a second file to keep. |
| **Contained** | One scene sits inside a longer file. | The longer file is never deleted. See [Contained cards](#contained-cards). |
| **Partial** | A long stretch is shared, but each side has more. | Nothing changes. Accept only marks it resolved. |
| **Language variant** | The same picture with another soundtrack. | Merge the scenes and keep every file. |
| **Mislabel** | The files match, but the metadata disagrees: different Stash-Box IDs, studios, or no performer in common. | Choose which scene has the right metadata. The scenes merge into it. |
| **Review** | The evidence is unclear. | Nothing changes. Look at the pair, then resolve it. |

A shared music bed or an intro sting is not enough for any class. Pairs that share only that are left out.

## How the keeper is chosen

The card suggests one file to keep and says why in a sentence. The rules run in this order, and the first one that separates the files decides:

1. A rip loses to a clean file. (Rips are not flagged yet, so this rule rarely fires.)
2. An AI upscale loses to its original.
3. The file with more real detail wins. This weighs the pixels against the bit rate, and counts hevc and av1 as more efficient than h264.
4. A codec or container you disfavored loses, but only when the detail is about equal.
5. More bits per pixel wins, since that is a cleaner picture with fewer blocks.
6. A clearly higher frame rate wins.
7. For duplicates, a file more than a minute longer wins, since it is more complete.
8. A better container (mp4, m4v, mkv) wins.

If nothing separates the files, the card says they are practically equal and keeps one. Either is fine.

A few cases keep two files:

- **An upscale pair.** If a set holds both an original and an upscale, Stash Sense keeps the best of each and deletes the other copies.
- **A longer file that is worse.** If the best file is much shorter than another copy, the longest file is kept too, because it may hold extra content, such as a supercut. You decide.
- **A contained scene that is better.** If the short file is strictly better quality than the longer cut that holds it, both stay. See [Contained cards](#contained-cards).

An AI upscale is detected from the file name, the folder, the encoder tag, or an av1 file much larger in pixels or size than its siblings. hevc and vp9 files are never flagged by size alone, since real HD releases use them.

The disfavored codecs and containers come from the exclusion and format settings in [Curation Settings](curation-settings.md). They only break near ties. They never override a clear difference in quality.

### Health check

Before a card offers to delete anything, the job probes the file it would keep and decodes a few short windows spread through it. A truncated or damaged file fails. A file that fails is never the keeper while a healthy one exists. If no file in the set passes, the card keeps every file and deletes nothing.

### How well it agrees with you

On the owner's library, the suggestion matched the recorded pick in 92.8% of 567 scored pairs. Practical ties and contained pairs deferred to splitting are left out. The pairs and every disagreement are listed in [the keeper regression](../research/dupes-keeper-regression.md).

## What accept does

Accept runs the outcome the card shows, in this order:

1. **Merge the scenes in Stash.** The scene with the most history stays. Scenes with a play count, an O count, a rating, the organized flag or groups rank first. Empty fields on it are filled from the others. Performers, tags, URLs, groups and galleries are unioned. Play and O history moves to the scene that stays.
2. **Delete the files not kept.** A confirm names each file by path and size first. If Stash has a trash folder (**Settings > System > Trash path**), the files go there. If not, they are **permanently deleted**.

If a scene changed after the card loaded, accept is refused and the card reloads. Every accept is logged, and metadata can be reverted from the [apply log](review.md#the-apply-log). Deleted files cannot be brought back.

### Contained cards

Quality decides, and the workflow is always merge first (to collect the metadata), then delete the lower-quality file. The longer file is never deleted.

- **The longer file holds several scenes** (it is at least 1.5 times as long as the shorter file). The card says to split the longer file first and changes nothing: no merge and no delete. This keeps a scene from being merged into an unsplit multi-scene file. [Split detection](split-detection.md) is meant to run before duplicate detection; until the longer file is split, these cards leave both files alone.
- **A re-cut where the shorter file is strictly better.** Both files stay and the scenes stay separate.
- **A re-cut at equal quality, or where the longer cut is better.** At equal quality the longer, more complete cut is kept. The shorter scene merges into the longer cut's scene, whatever history either has: the longer scene's values win, empty fields are filled from the shorter scene, and performers, tags, URLs, groups and galleries are unioned. Then the shorter file is deleted.

If you pick another option on the card, it is honored: choosing to keep only the longer file deletes the shorter file and merges its scene into the longer one. Contained cards are never in the batch accept.

### Mislabel cards

A mislabel card never unions metadata. The scene you pick as having the right metadata keeps its fields exactly as they are, and nothing is copied in from the other scene.

### Your pick is remembered

Files are picked by file hash, not by position on the card. If you keep a different file than the suggestion, the choice and the evidence are saved as a labeled example, and it means the same file however the card is ordered.

## Accept all high-confidence

Batch accept is limited to the plain case. A card qualifies only when all of these hold:

- the class is **duplicate**
- you have not changed the suggestion
- exactly one file is kept, and at least one is deleted
- the keeper raised no flag
- every deleted file is identical to the kept file (same md5 or oshash) or is the same encode (same codec and frame size, duration within a second)

Everything else is left for you to review one card at a time. See [Accept all high-confidence](review.md#accept-all-high-confidence).

## Exclusions

The exclusion list is honored. A scene on it is never a candidate, and an existing card that names an excluded scene is hidden. If Stash can't be reached to read the list, the run stops instead of guessing.

## Cards update in place

Each set has one recommendation. When you run the job again:

- A set whose evidence or keeper suggestion changed is rewritten in place.
- A set that gains a scene stays one card.
- A set that is no longer found is marked stale, but only when the run finished.
- A card you resolved returns if its details change.
- A card you dismissed stays dismissed.

Tools that read the [driver API](../driver-api.md) follow these changes. A card is never deleted, so a tool never misses that a set went away.
