"""Static-snow scene transitions (#131).

Some masters change scene through analog TV "snow" instead of a fade to black, so the dark-run planner
sees nothing. Snow is near-grayscale (low mean saturation) with a frame difference that stays high for a
burst of frames, while a real cut spikes the difference for one frame and motion stays colourful.

The bursts come in pairs 5-32 s apart that bracket a title card; the scene opens at the START of the first
burst of the pair. A pair before the intro guard, after the outro guard, or a lone burst is not a cut.
A pair with silence near its first burst is strong; without it the pair is reported but not strong.
"""

from . import detect

STATIC_SAT = 4.0  # mean saturation below this is near-grayscale
STATIC_YDIF = 20.0  # frame difference above this is uncorrelated noise
STATIC_MIN_FRAMES = 5  # a burst is at least this many consecutive snow frames
PAIR_GAP_S = (5.0, 32.0)  # first burst starts this far before the second
PAIR_SILENCE_S = 10.0  # a silence whose middle is within this of the first burst confirms the pair


def bursts(d) -> list[list[float]]:
    """Runs of at least STATIC_MIN_FRAMES snow frames as [start, length] (seconds, rounded to 0.1)."""
    out = []
    run = []
    for f in d["fr"] + [[1e9, 0, 99, 0, 0]]:
        if f[2] < STATIC_SAT and f[3] > STATIC_YDIF:
            run.append(f)
            continue
        if len(run) >= STATIC_MIN_FRAMES:
            out.append([round(run[0][0], 1), round(run[-1][0] - run[0][0], 1)])
        run = []
    return out


def static_candidates(d, total, guard_s=150.0) -> list[dict]:
    """One candidate per burst pair: {t, kind "static", len, sil, strong, pair}.

    `t` is the first burst's start, `len` the pair's span, `sil` the longest silence within PAIR_SILENCE_S
    of the first burst (0.0 if none), `strong` whether there is one. Lone bursts and pairs starting before
    `guard_s` or after `total - guard_s` are dropped.
    """
    st = sorted(bursts(d))
    sils = detect.silences(d)
    used = set()
    out = []
    for i, (t, _) in enumerate(st):
        if i in used:
            continue
        nxt = [j for j in range(i + 1, len(st)) if PAIR_GAP_S[0] <= st[j][0] - t <= PAIR_GAP_S[1]]
        if not nxt:
            continue
        j = nxt[0]
        used.update([i, j])
        if t < guard_s or t > total - guard_s:
            continue
        near = [b - a for a, b in sils if abs((a + b) / 2 - t) <= PAIR_SILENCE_S]
        sil = round(max(near), 2) if near else 0.0
        out.append(
            {
                "t": t,
                "kind": "static",
                "len": round(st[j][0] - t, 1),
                "sil": sil,
                "strong": bool(near),
                "pair": [t, st[j][0]],
            }
        )
    return out
