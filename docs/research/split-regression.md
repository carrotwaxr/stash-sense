# Split regression (#131)

This is the split planner (`api/splits/planner.py`) run on every title of the regression set and scored against the owner's labelled cuts. The set is `api/tests/fixtures/split_regression/`: `index.jsonl` holds one record per title (anonymized id, series label, duration, the labelled cuts, whether the owner left it whole, and the labelled `keep_until`), and `<id>.npz` the extracted video and audio signals. The report holds only fixture ids, series labels and numbers.

Each title is planned leave-one-out: its profile is built (`profiles.build_profile`) from the other titles of its series that have labelled cuts, using `features_at` on their labelled cuts, with each title's total taken as its labelled `keep_until` when it has one and its duration otherwise. A series with no other labelled title, or a profile with fewer than 3 boundaries, gives no profile (the planner then gives at most `medium`, unless coverage confirms the plan).

Decision 2026-11-18: `high` needs a fade-type series profile (black rate >= 0.75) or confirmed coverage. The relaxed fallback and the no-profile rule were removed after the first run showed 9 of 12 high plans missing labels.

Terms:

- **exact**: the plan has as many cuts as the label and every cut is within 2 s of its label.
- **recall**: the share of labelled cuts that have a plan cut within 2 s (or 30 s). Median distance is from each labelled cut to the nearest plan cut, over the labelled cuts of the plans in that row that have one.
- **at a black start**: a dark run (`detect.dark_runs`) starts within 0.5 s of the labelled cut. Some labels were never snapped to a black start: titles with static transitions (owner scrub times, which land a few seconds after the first static burst) and a few labels from before the cut-at-black-start rule (they sit mid-black). In the tables a label that is not at a black start is marked `*`. This column is for reading only; the pass/fail rule does not use it.

Titles: 118 (109 labelled, 9 left whole). Labelled cuts: 558, of which 271/558 (48.6%) are at a black start.

## Totals

Labelled titles, by plan confidence.

| Confidence | Plans | Exact | Recall 2 s | Recall 30 s | Median distance (s) | With profile |
|---|---|---|---|---|---|---|
| high | 1 | 1 | 3/3 (100.0%) | 3/3 (100.0%) | 0.02 | 1 |
| medium | 26 | 9 | 107/159 (67.3%) | 112/159 (70.4%) | 0.14 | 9 |
| low | 82 | 1 | 141/396 (35.6%) | 177/396 (44.7%) | 6.15 | 36 |
| all | 109 | 11 | 251/558 (45.0%) | 292/558 (52.3%) | 1.07 | 46 |

Labelled cuts by whether they sit at a black start (all confidences).

| Labels | Cuts | Recall 2 s | Recall 30 s |
|---|---|---|---|
| at a black start | 271 | 223/271 (82.3%) | 234/271 (86.3%) |
| not at a black start | 287 | 28/287 (9.8%) | 58/287 (20.2%) |

Menu tail (`keep_until`): 6/8 (75.0%) labelled `keep_until` values are found within 30 s; 0 titles without a label got one.

Titles left whole, by plan confidence.

| Confidence | Plans | Plans with cuts | Cuts planned |
|---|---|---|---|
| high | 0 | 0 | 0 |
| medium | 0 | 0 | 0 |
| low | 9 | 2 | 2 |

## High plans that miss (0 of 1)

Times are seconds; `*` marks a label that is not at a black start. "Misses not at a black start" says whether the labels with no plan cut within 2 s are all, partly or not at all of that kind.

| Title | Series | Label cuts | Plan cuts | Profile | Misses not at a black start | Reasons |
|---|---|---|---|---|---|---|

## Left-whole titles with a high plan (0)

| Title | Series | Plan cuts | Profile | Reasons |
|---|---|---|---|---|

## Medium and low plans that miss

Counts only.

| Confidence | Plans that miss | Of those, every missed label is not at a black start |
|---|---|---|
| medium | 17 | 10 |
| low | 81 | 44 |
