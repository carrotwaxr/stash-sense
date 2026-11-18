"""Dark-run split-point detection (#131): a port of the October v3 planner.

Input is the signal dict of `signals.to_dict`: `fr` = [[t, Y, SAT, YDIF, scd]] at 6 fps, `au` = [[t, rms_db]]
every 0.25 s, `sil` = [[start, end]] from silencedetect.

A scene boundary in a DVD-style compilation is a stretch of black frames (luma Y <= 32), usually with
silence, and the cut sits at the START of that black run. Owner rule (2026-10-24): the black and the title
card after it belong to the next scene.
"""

import bisect
import statistics

BLACK_Y = 32  # a frame at or below this luma is dark
MIN_BLACK_S = 0.17  # shortest dark run that can be a candidate
MERGE_S = 15  # candidates closer than this collapse to the longest
MIN_SEG_S = 90  # no scene is shorter: two blacks nearer than this are one transition
INTRO_GUARD_S = 150  # the first 150 s are the studio intro bumper, not a scene boundary
DEEP_DIP = (
    0.35  # a dip to this share of the surrounding level (or lower) is a fade to black
)
DARK_SCENE_MIN_Y = 20  # 24566: a stretch whose darkest frame is above this and ...
DARK_SCENE_RATIO = (
    0.6  # ... barely dips below its surroundings is a dark scene, not a black
)
SHORT_BLACK_S = 0.5  # 24261, 27188: shorter blacks need silence AND a deep dip
MIN_SIL_S = 0.3
UNMISTAKABLE = (
    2.5,
    1.0,
    0.35,
)  # (black length s, silence s, dip ratio) all met: no doubt

_RUN_GAP_S = 0.4  # gaps shorter than this inside a dark run are bridged
_FRAME_S = 1 / 6
_DIGITAL_SILENCE_DB = -60


def _med(a):
    return statistics.median(a) if a else None


def dark_runs(fr, ymax=BLACK_Y, gap=_RUN_GAP_S):
    """Contiguous runs of frames with Y <= ymax (gaps shorter than `gap` s bridged): [t0, t1, len_s, min_y, mean_y]."""
    runs = []
    cur = None
    for f in fr:
        if f[1] <= ymax:
            if cur and f[0] - cur[1] <= gap:
                cur[1] = f[0]
                cur[2].append(f[1])
            else:
                if cur:
                    runs.append(cur)
                cur = [f[0], f[0], [f[1]]]
        elif cur and f[0] - cur[1] > gap:
            runs.append(cur)
            cur = None
    if cur:
        runs.append(cur)
    return [
        [
            r[0],
            r[1],
            round(r[1] - r[0] + _FRAME_S, 2),
            min(r[2]),
            round(sum(r[2]) / len(r[2]), 1),
        ]
        for r in runs
    ]


def silences(d, minlen=0.0):
    """ffmpeg silencedetect intervals merged with digital-silence (rms <= -60 dB) runs."""
    out = [list(x) for x in d.get("sil", [])]
    run = None
    for t, v in d.get("au", []) + [[1e9, 0]]:
        if v <= _DIGITAL_SILENCE_DB and run is None:
            run = [t, t + 0.25]
        elif v <= _DIGITAL_SILENCE_DB:
            run[1] = t + 0.25
        elif run is not None:
            if run[1] - run[0] >= 0.25:
                out.append(run)
            run = None
    return [x for x in out if x[1] - x[0] >= minlen]


def ref_level(fr, t, w0=2.0, w1=9.0, ts=None):
    """The luma level around t: the lower of the medians 2-9 s before and 2-9 s after; None near the file edges."""
    ts = ts or [f[0] for f in fr]
    before = [
        f[1]
        for f in fr[bisect.bisect_left(ts, t - w1) : bisect.bisect_left(ts, t - w0)]
    ]
    after = [
        f[1]
        for f in fr[bisect.bisect_right(ts, t + w0) : bisect.bisect_right(ts, t + w1)]
    ]
    if len(after) < 6 or len(before) < 6:
        return None
    return min(_med(after), _med(before))


def features_at(d, t, win=4.0, runs=None, sils=None, ts=None):
    """What the transition at t looks like.

    blk: longest dark run touching [t-win, t+win]; sil: longest silence touching it; dip_ratio: the lowest
    luma in the window relative to the level 2-9 s either side (catches fades that never reach Y <= 32);
    scd: max scene-change score within 1.5 s; ydif: max frame difference.
    """
    fr = d["fr"]
    runs = runs if runs is not None else dark_runs(fr)
    sils = sils if sils is not None else silences(d)
    near = [r for r in runs if r[1] >= t - win and r[0] <= t + win]
    blk = max(near, key=lambda r: r[2]) if near else None
    sl = [s for s in sils if s[1] >= t - win and s[0] <= t + win]
    sm = max(sl, key=lambda s: s[1] - s[0]) if sl else None
    ts = ts or [f[0] for f in fr]
    w = fr[bisect.bisect_left(ts, t - win) : bisect.bisect_right(ts, t + win)]
    ref = ref_level(fr, t, ts=ts)
    miny = min((f[1] for f in w), default=None)
    return {
        "blk_len": blk[2] if blk else 0.0,
        "blk_min": blk[3] if blk else None,
        "blk_t": round((blk[0] + blk[1]) / 2, 2) if blk else None,
        "sil_len": round(sm[1] - sm[0], 2) if sm else 0.0,
        "sil_t": round((sm[0] + sm[1]) / 2, 2) if sm else None,
        "min_y": miny,
        "ref_y": None if ref is None else round(ref),
        "dip_ratio": (
            None if (ref is None or miny is None or ref <= 0) else round(miny / ref, 2)
        ),
        "scd": max((f[4] for f in w if abs(f[0] - t) <= 1.5), default=0),
        "ydif": max((f[3] for f in w if abs(f[0] - t) <= 1.5), default=0),
    }


def candidates(d, min_len=MIN_BLACK_S, min_gap=MERGE_S, win=4.0, min_t=0, total=None):
    """All dark runs of at least `min_len` s over the whole file with their silence and scd features.

    Candidates closer than `min_gap` s merge (the longest wins). The cut position `t` is the run START
    (owner rule 2026-10-24: the black and the title card after it belong to the NEXT scene); `mid` is the
    run's middle. Candidates before `min_t`, or within `min_t` of `total`, are dropped.
    """
    runs = dark_runs(d["fr"])
    sils = silences(d)
    ts = [f[0] for f in d["fr"]]
    found = []
    for r in runs:
        if r[2] < min_len:
            continue
        f = features_at(d, (r[0] + r[1]) / 2, win=win, runs=runs, sils=sils, ts=ts)
        found.append(
            {
                "kind": "black",
                "t": round(r[0], 1),
                "mid": round((r[0] + r[1]) / 2, 1),
                "len": r[2],
                "min_y": r[3],
                "sil": f["sil_len"],
                "scd": f["scd"],
            }
        )
    merged = []
    for c in found:
        if merged and c["t"] - merged[-1]["t"] < min_gap:
            if c["len"] > merged[-1]["len"]:
                merged[-1] = c
        else:
            merged.append(c)
    return [
        c
        for c in merged
        if c["t"] >= min_t and (total is None or c["t"] <= total - min_t)
    ]


def refine(d, cands, win=9.0):
    """Add `ref` (the level before the run start and after its end: the lower of the two 9 s medians, 1 s guard)
    and `dip_ratio` = min_y / ref to each candidate. None when either side has under 6 frames."""
    fr = d["fr"]
    ts = [f[0] for f in fr]
    for c in cands:
        a = c["t"]
        b = c["t"] + c["len"]
        before = [
            f[1]
            for f in fr[bisect.bisect_left(ts, a - win) : bisect.bisect_left(ts, a - 1)]
        ]
        after = [
            f[1]
            for f in fr[
                bisect.bisect_right(ts, b + 1) : bisect.bisect_right(ts, b + win)
            ]
        ]
        ref = (
            min(_med(before), _med(after))
            if len(before) >= 6 and len(after) >= 6
            else None
        )
        c["ref"] = None if ref is None else round(ref)
        c["dip_ratio"] = None if not ref else round(c["min_y"] / ref, 2)
    return cands


def is_strong(c, profile=None):
    """The candidate satisfies the transition signature.

    - A dark scene is not a black: min Y >= 20 with the dip barely below its surroundings (ratio >= 0.6) was
      taken for a 7.5 s black in 24566 and beat the real 5.8 s black (Y 0, title card on it) 29 s earlier.
    - Short blacks (< 0.5 s; 24261, 27188) need BOTH silence >= 0.3 s and a deep dip (<= 0.35); longer ones
      need either.
    - No per-series "silence required" rule (9140: 3 of 4 true cuts have no silence).

    `profile` is the per-series profile; the generic rules do not read it.
    """
    if c["len"] < MIN_BLACK_S:
        return False
    dr = c.get("dip_ratio")
    deep = dr is not None and dr <= DEEP_DIP
    if dr is not None and c["min_y"] >= DARK_SCENE_MIN_Y and dr >= DARK_SCENE_RATIO:
        return False
    if c["len"] < SHORT_BLACK_S:
        return c["sil"] >= MIN_SIL_S and deep
    return c["sil"] >= MIN_SIL_S or deep


def unmistakable(c):
    """A long black, long silence and deep dip together: nothing else looks like this."""
    min_len, min_sil, max_dip = UNMISTAKABLE
    return (
        c["len"] >= min_len
        and c["sil"] >= min_sil
        and c.get("dip_ratio") is not None
        and c["dip_ratio"] <= max_dip
    )


def plan_cuts(cands, profile, total, min_seg_s=MIN_SEG_S):
    """Pick the cuts for a file: every strong candidate, dropping any closer than `min_seg_s` to a stronger one.

    Owner 2026-10-24: short scenes (5-7 minutes) exist, so there is no minimum scene length beyond the 90 s
    needed to tell two separate blacks apart. Nothing is cut in the first 150 s or within `min_seg_s` of the
    end. Returns (cuts, leftovers): cuts are candidates (position = run START), leftovers the non-strong
    candidates of at least 0.3 s (notes only).
    """
    strong = sorted((c for c in cands if is_strong(c, profile)), key=lambda c: c["t"])
    out = []
    for c in strong:
        if c["t"] < max(min_seg_s, INTRO_GUARD_S) or total - c["t"] < min_seg_s:
            continue
        if out and c["t"] - out[-1]["t"] < min_seg_s:
            if (c["len"], c["sil"]) > (out[-1]["len"], out[-1]["sil"]):
                out[-1] = c
            continue
        out.append(c)
    left = [c for c in cands if not is_strong(c, profile) and c["len"] >= 0.3]
    return out, left


# --- DVD menu tails ---

MENU_MIN_TAIL_S = (
    180  # a trailing static picture must run at least this long to count as a menu loop
)
MENU_YDIF_MAX = 1.0  # "quiet" frame: frame difference below this ...
MENU_SCD_MAX = 3.0  # ... and scene-change score below this
MENU_BREAK_TOL = 0.01  # share of the tail that may be non-quiet frames
MENU_SHORT_TAIL_S = 45  # shortest tail accepted through the black-run anchor
MENU_BLACK_S = 2.0  # the black that opens the menu is at least this long
MENU_DARK_Y = 80  # median luma of the first 60 s of the menu below this: a dark menu
MENU_BRIGHT_Y = 60  # first frame above this after the black is the menu's first picture
MENU_HARD_CUT_SCD = 50  # a scene change this strong into the menu is a hard cut
MENU_HARD_CUT_LEAD_S = (
    3.0  # the content ends this long before a hard cut into a bright menu
)


def menu_tail(
    fr,
    min_len_s=MENU_MIN_TAIL_S,
    ydif_max=MENU_YDIF_MAX,
    scd_max=MENU_SCD_MAX,
    tol=MENU_BREAK_TOL,
):
    """Start (s) of a DVD title-menu loop that ends the file, else None.

    Owner 2026-10-24, scene 24566 (a multi-title DVD file): the picture is static and repeats, and
    20+ min of it was discarded. The tail is the longest trailing run where YDIF < `ydif_max` and
    scd < `scd_max`; up to `tol` of its frames may break the run.
    """
    n = len(fr)
    if n < 10:
        return None
    quiet = [f[3] < ydif_max and f[4] < scd_max for f in fr]
    bad = 0
    start = n
    for i in range(n - 1, -1, -1):
        if not quiet[i]:
            bad += 1
            if bad > max(1, tol * (n - i)):
                break
        else:
            start = i
    # tighten: first frame of the trailing quiet run
    i = start
    while i < n and not quiet[i]:
        i += 1
    t0 = fr[i][0] if i < n else None
    if t0 is not None and fr[-1][0] - t0 >= min_len_s:
        return round(t0, 1)
    return None


def menu_keep_until(fr, min_tail_s=MENU_SHORT_TAIL_S):
    """End of the real content when a DVD menu loop closes the file.

    Returns {"keep_until", "kind", "black_start", "menu_start"} or None.

    Anchor: the file ends in a quiet tail (`menu_tail`: frame difference ~0, no scene changes)
    that opens with a black run of 2 s or more (the quiet tail starts inside or right after the
    black). For a short tail (`min_tail_s` or more) the anchor is the last black run of 2 s or
    more after which every frame stays quiet (YDIF < 6 over 5 s blocks, no scene change of 10 or
    more after the opening cut). Scene 26702's quiet tail is a bright still with no black before
    it: not matched.

    keep_until: a dark menu (median Y < 80 in its first 60 s, scene 24566) cuts at the black
    START (owner: 8268.23). A bright menu (scenes 27170, 27175) cuts at the first bright frame
    (Y > 60) after the black, minus 3 s when it is a hard cut with scene change >= 50 (27170:
    4244.8 - 3 = 4241.8 vs owner 4241.83; 27175: 5213.3 vs 5212.9).
    """
    if not fr:
        return None
    end = fr[-1][0]
    runs = dark_runs(fr)
    anchor = None
    t0 = menu_tail(fr)
    if t0 is not None:
        found = [
            r
            for r in runs
            if r[2] >= MENU_BLACK_S
            and (r[0] - 3 <= t0 <= r[1] + 3 or 0 <= t0 - r[1] <= 5)
        ]
        anchor = found[0] if found else None
    if anchor is None:
        # Only the last long-enough black run is tried.
        for r in reversed(runs):
            if r[2] < MENU_BLACK_S or end - r[1] < min_tail_s:
                continue
            after = [f for f in fr if f[0] > r[1] + 0.5]
            late = [f for f in after if f[0] > r[1] + 15]
            if len(after) < 30 or any(f[4] >= 10 for f in late):
                continue
            blocks = [
                statistics.mean(f[3] for f in after if b <= f[0] < b + 5)
                for b in range(int(r[1]) + 15, int(end) - 5, 5)
                if any(b <= f[0] < b + 5 for f in after)
            ]
            if blocks and max(blocks) < 6:
                anchor = r
            break
    if anchor is None:
        return None
    after = [f for f in fr if f[0] > anchor[1] + 0.5]
    first = next((f for f in after if f[1] > MENU_BRIGHT_Y), None)
    if first is None:
        return None
    menu_y = statistics.median(f[1] for f in after if f[0] <= anchor[1] + 60)
    if menu_y < MENU_DARK_Y:
        return {
            "keep_until": round(anchor[0], 2),
            "kind": "dark menu: black start",
            "black_start": round(anchor[0], 2),
            "menu_start": round(first[0], 1),
        }
    lead = MENU_HARD_CUT_LEAD_S if first[4] >= MENU_HARD_CUT_SCD else 0.0
    return {
        "keep_until": round(first[0] - lead, 1),
        "kind": "bright menu",
        "black_start": round(anchor[0], 2),
        "menu_start": round(first[0], 1),
    }
