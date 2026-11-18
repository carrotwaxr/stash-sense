# Split regression (#131)

This is the split planner (`api/splits/planner.py`) run on every title of the regression set and scored against the owner's labelled cuts. The set is `api/tests/fixtures/split_regression/`: `index.jsonl` holds one record per title (anonymized id, series label, duration, the labelled cuts, whether the owner left it whole, and the labelled `keep_until`), and `<id>.npz` the extracted video and audio signals. The report holds only fixture ids, series labels and numbers.

Each title is planned leave-one-out: its profile is built (`profiles.build_profile`) from the other titles of its series that have labelled cuts, using `features_at` on their labelled cuts, with each title's total taken as its labelled `keep_until` when it has one and its duration otherwise. A series with no other labelled title, or a profile with fewer than 3 boundaries, gives no profile (the planner then needs every cut to be unmistakable for `high`).

Terms:

- **exact**: the plan has as many cuts as the label and every cut is within 2 s of its label.
- **recall**: the share of labelled cuts that have a plan cut within 2 s (or 30 s). Median distance is from each labelled cut to the nearest plan cut, over the labelled cuts of the plans in that row that have one.
- **at a black start**: a dark run (`detect.dark_runs`) starts within 0.5 s of the labelled cut. Some labels were never snapped to a black start: titles with static transitions (owner scrub times, which land a few seconds after the first static burst) and a few labels from before the cut-at-black-start rule (they sit mid-black). In the tables a label that is not at a black start is marked `*`. This column is for reading only; the pass/fail rule does not use it.
- **relaxed**: the plan used the fallback that accepts shorter blacks that carry silence (`SplitPlan.relaxed`).

Titles: 118 (109 labelled, 9 left whole). Labelled cuts: 558, of which 271/558 (48.6%) are at a black start.

## Totals

Labelled titles, by plan confidence.

| Confidence | Plans | Exact | Recall 2 s | Recall 30 s | Median distance (s) | With profile | Relaxed |
|---|---|---|---|---|---|---|---|
| high | 12 | 3 | 41/64 (64.1%) | 43/64 (67.2%) | 0.50 | 10 | 2 |
| medium | 21 | 8 | 88/131 (67.2%) | 92/131 (70.2%) | 0.13 | 6 | 0 |
| low | 76 | 0 | 119/363 (32.8%) | 153/363 (42.1%) | 11.20 | 30 | 0 |
| all | 109 | 11 | 248/558 (44.4%) | 288/558 (51.6%) | 1.30 | 46 | 2 |

Labelled cuts by whether they sit at a black start (all confidences).

| Labels | Cuts | Recall 2 s | Recall 30 s |
|---|---|---|---|
| at a black start | 271 | 221/271 (81.5%) | 232/271 (85.6%) |
| not at a black start | 287 | 27/287 (9.4%) | 56/287 (19.5%) |

Menu tail (`keep_until`): 6/8 (75.0%) labelled `keep_until` values are found within 30 s; 0 titles without a label got one.

Titles left whole, by plan confidence.

| Confidence | Plans | Plans with cuts | Cuts planned |
|---|---|---|---|
| high | 0 | 0 | 0 |
| medium | 0 | 0 | 0 |
| low | 9 | 2 | 2 |

## High plans that miss (9 of 12)

Times are seconds; `*` marks a label that is not at a black start. "Misses not at a black start" says whether the labels with no plan cut within 2 s are all, partly or not at all of that kind.

| Title | Series | Label cuts | Plan cuts | Profile | Relaxed | Misses not at a black start | Reasons |
|---|---|---|---|---|---|---|---|
| 00d19a8594f2 | s040 | 492.5, 1552.7, 3389.5, 4960.6, 6379.8, 7449.3 | 393.8, 1552.7, 3389.5, 4960.5, 6379.8, 7449.2 | yes | yes | no: the missed label is at a black start | blacks shorter than the series' (>= 0.3 s with silence) accepted: every scene fits its range; the series profile confirms the scene count and lengths |
| 0d8374c11ae3 | s058 | 1107.1*, 2006.4*, 2471.2*, 3118.9, 3550.0, 3919.2*, 4572.0* | 1106.2, 2005.8, 2470.5, 3118.7, 3550.0, 3918.7 | yes | no | yes: the missed label is not at a black start | the series profile confirms the scene count and lengths |
| 2c7109c7248f | s058 | 1077.7, 2169.0*, 2332.0*, 2770.9*, 3110.6*, 3812.0*, 4237.0 | 1077.7, 2068.3, 2769.8, 3110.0, 4237.2 | yes | no | yes: all 3 missed labels are not at a black start | the series profile confirms the scene count and lengths |
| 3b2d60861948 | s058 | 1255.1, 2140.3, 2692.8, 3201.7, 3843.0*, 4322.0*, 4840.5 | 226.8, 1255.0, 2140.3, 2692.7, 3201.7, 4320.8 | yes | no | partly: 1 of 2 missed labels are not at a black start | DVD menu loop from ~5346 s: the rest of the file is dropped; the series profile confirms the scene count and lengths |
| 3e1dde199980 | s058 | 817.8, 1492.6, 2015.8*, 3210.1, 3794.2*, 4542.6*, 4781.0* | 1728.5, 3234.0, 4013.3, 5081.0 | yes | yes | partly: 4 of 7 missed labels are not at a black start | blacks shorter than the series' (>= 0.3 s with silence) accepted: every scene fits its range; the series profile confirms the scene count and lengths |
| 6020de3a4997 | s064 | 1290.9, 2949.0, 3896.2 | 1290.8, 1888.8, 2948.7, 3896.2 | no | no | no label missed (the plan has a different cut count) | every cut is an unmistakable black (>= 2.5 s, silence >= 1 s, deep dip) and every scene is 5-60 min |
| 945173e45afb | s058 | 1168.3, 1612.1, 2589.6, 2994.9*, 4040.9, 4477.7 | 159.7, 501.2, 1167.8, 1611.8, 2589.3, 2994.2, 4040.7, 4477.3 | yes | no | no label missed (the plan has a different cut count) | the series profile confirms the scene count and lengths |
| aca4cc48bae5 | s058 | 1862.0*, 2984.8, 3332.5, 3698.0*, 4184.0*, 5075.0* | 193.5, 2984.8, 3332.0 | yes | no | yes: all 4 missed labels are not at a black start | the series profile confirms the scene count and lengths |
| d51ec0cd155f | s048 | 288.0*, 1020.0*, 1906.0*, 3032.0*, 3653.3, 4521.0* | 1900.7, 3653.3 | yes | no | yes: all 5 missed labels are not at a black start | the series profile confirms the scene count and lengths |

## Left-whole titles with a high plan (0)

| Title | Series | Plan cuts | Profile | Relaxed | Reasons |
|---|---|---|---|---|---|

## Medium and low plans that miss

Counts only.

| Confidence | Plans that miss | Of those, every missed label is not at a black start |
|---|---|---|
| medium | 13 | 7 |
| low | 76 | 43 |
