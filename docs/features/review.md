# Reviewing Recommendations

Curation features (duplicates, split points, metadata fixes) end with a decision only you should make. Stash Sense shows each one as a review card inside Stash. Nothing changes until you accept it, and every accepted change is logged.

## Opening the dashboard

Click **Stash Sense** in Stash's top navigation bar, next to Scenes and Performers. It opens the dashboard at `/plugins/stash-sense`.

## The review card

Each card shows two things side by side:

- **Evidence**: why the recommendation exists, such as the scene's files with size, codec, resolution and duration, stills taken from each file, and a **Play here** button that plays the scene from Stash's own stream at the relevant moment.
- **Outcome**: exactly what accepting does:
    - each field that changes, with its value before and after
    - the files kept and the files deleted, by path and size
    - the space reclaimed

When a card offers choices (for example which file to keep, or an edited title), changing one rebuilds the outcome. The card always shows what **Accept** will do.

## Accepting

**Accept** applies the outcome shown on the card.

- **Deleting files needs a second confirm.** The confirm lists every file by path and size. It also says where the files go: to Stash's trash folder if one is set (**Settings > System > Trash path**, Stash v0.30 and later), or **permanently deleted** if not.
- **If the library changed after the card loaded**, for example someone edited the scene, accept is refused. Nothing is written, and the card reloads with the updated outcome for you to check again.
- **Accepting twice is harmless.** A second accept of the same recommendation returns the first result and changes nothing.

## Accept all high-confidence

Each recommendation type marks the items it is confident about. **Accept all high-confidence…** on a type's page:

1. lists how many items qualify, the field changes, every file that would be deleted and the total space reclaimed
2. leaves every other item for individual review
3. applies only after you confirm the summary

Batch accept always applies the type's suggestion. An item where you picked something different, such as another file to keep, is never batch-accepted. If any item in a batch is no longer high-confidence when it runs, the whole batch is refused and nothing is written.

## The apply log

**Apply log** on the dashboard lists every accepted change, newest first. Expand an entry to see the state before the change:

- the old value of every field it changed
- every deleted file's path, size and Stash fingerprints (oshash, md5, phash), and whether it went to the trash

**Revert metadata** writes the old field values back and reopens the recommendation. If a field was edited again after the apply, revert shows the edits and asks before it overwrites them. Revert cannot bring back deleted files; recover those from Stash's trash folder if one was set.

## Your corrections improve the suggestions

When you accept something different from the suggestion, for example keeping another file or editing a proposed value, Stash Sense saves the suggestion, your choice and the evidence as a labeled example. These examples become the test set for that feature, so later versions are checked against your real decisions. The sidecar exports them at `GET /review/labels?type=<type>`.

## Unfinished types

A recommendation type that has not passed the production checklist stays hidden. To see it, turn on **Settings > Features > Show Unfinished Features**. The **Review Demo** type is one of these: it only acts on scenes you tag `stash-sense-review-demo`, and it exists to try the review flow on a test library.
