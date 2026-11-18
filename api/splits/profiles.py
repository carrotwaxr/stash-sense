"""Per-series transition profiles (#131): a port of the October `series_key` and `series_profiles` code.

A series' known boundaries (owner-labelled cuts) describe how its scenes change: how often there is a black,
how long it is, how deep the dip, and how many scenes of what length a title holds. The planner uses the
profile to confirm a plan (`judge`) and to fall back to shorter silent blacks when the strict plan fails.

Aliases name real titles, so they are never in the repo: callers pass them in (from the data dir).
"""

import re
import statistics


def series_key(title, aliases=()):
    """Series key from a title: volume numbers, '#', Vol., years, disc/part tokens and resolution tags stripped,
    lower-cased. `aliases` = [(regex, name)]: the first regex that matches the key replaces it with `name`."""
    k = _series_key(title)
    for pat, name in aliases:
        if re.search(pat, k):
            return name
    return k


def _series_key(title):
    t = (title or "").strip()
    t = re.sub(r",\s*the\b", "", t, flags=re.I)
    t = re.sub(r"\.(mp4|mkv|avi|wmv)$", "", t, flags=re.I)
    t = re.sub(r"\((?:19|20)\d\d\)|\(2D Version\)|\(Decensored\)|\(Upscaled\)", "", t, flags=re.I)
    t = re.sub(r"\b(?:480|720|1080)p\b", "", t, flags=re.I)
    t = re.sub(r"\s*[-–:]\s*(?:disc|dvd|cd|part)\s*\d+.*$", "", t, flags=re.I)
    t = re.sub(r"\s*[-–]\s*part\s*\d+.*$", "", t, flags=re.I)
    t = re.sub(r"\b(?:disc|dvd|cd|part)\s*\d+\b", "", t, flags=re.I)
    t = re.sub(r"(?:vol(?:ume)?\.?\s*)?#?\s*\d+(?:-\d+)?\s*(?:[:\-].*)?$", "", t, flags=re.I)
    t = re.sub(r"^the\s+", "", t, flags=re.I)
    t = re.sub(r"[,]?\s*the$", "", t.strip(), flags=re.I)
    t = re.sub(r"[#\-–:\s]+$", "", t)
    return re.sub(r"\s+", " ", t).strip().lower()


def _med(a):
    return statistics.median(a) if a else None


def _pct(a, p):
    a = sorted(a)
    return a[min(len(a) - 1, int(p * len(a)))] if a else None


def summarize(rows):
    """`detect.features_at` dicts at the known boundaries of one series -> the distribution of each feature.
    `black_rate` = share with a dark run, `sil_rate` = share with silence >= 0.3 s."""
    n = len(rows)
    blacks = [r["blk_len"] for r in rows if r["blk_len"] > 0]
    return {
        "n_boundaries": n,
        "black_rate": round(len(blacks) / n, 2) if n else 0,
        "black_len": {
            "min": min(blacks),
            "p10": _pct(blacks, 0.1),
            "med": _med(blacks),
            "p90": _pct(blacks, 0.9),
            "max": max(blacks),
        }
        if blacks
        else None,
        "dip_ratio_med": _med([r["dip_ratio"] for r in rows if r["dip_ratio"] is not None]),
        "sil_rate": round(sum(1 for r in rows if r["sil_len"] >= 0.3) / n, 2) if n else 0,
        "sil_len_med": _med([r["sil_len"] for r in rows if r["sil_len"] > 0]),
        "scd_med": _med([r["scd"] for r in rows]),
    }


def build_profile(titles):
    """titles: [{"key", "total", "cuts": [boundary seconds], "feat": [features_at dict per cut, or None]}] of one
    series -> profile. `transition` is black (>= 75% of boundaries have a black), mixed (>= 30%) or hardcut;
    `scene_count` and `scene_len_min` are the observed ranges; `detect` is what the planner reads."""
    feats = [f for t in titles for f in t["feat"] if f]
    s = summarize(feats) if feats else {}
    segs = []
    counts = []
    for t in titles:
        cs = sorted(t["cuts"])
        counts.append(len(cs) + 1)
        edges = [0, *cs, t["total"]]
        segs += [(b - a) / 60 for a, b in zip(edges, edges[1:])]
    black_rate = s.get("black_rate", 0)
    bl = s.get("black_len")
    kind = "black" if black_rate >= 0.75 else ("mixed" if black_rate >= 0.3 else "hardcut")
    det = {"kind": kind}
    if bl:
        det["min_black_s"] = max(0.17, round(0.7 * bl["p10"], 2))
        det["need_sil"] = s.get("sil_rate", 0) >= 0.99
    return {
        "n_titles": len(titles),
        "n_boundaries": len(feats),
        "titles": [t["key"] for t in titles],
        "transition": kind,
        "summary": s,
        "scene_count": {"min": min(counts), "max": max(counts), "med": _med(counts)} if counts else None,
        "scene_len_min": {
            "min": round(min(segs), 1),
            "p10": round(_pct(segs, 0.1), 1),
            "med": round(_med(segs), 1),
            "p90": round(_pct(segs, 0.9), 1),
            "max": round(max(segs), 1),
        }
        if segs
        else None,
        "detect": det,
    }


def judge(cuts, profile, total):
    """Profile-confirmed test: (ok, reasons). ok needs a profile with >= 3 known boundaries that is not hard-cut,
    a scene count within one of the series range (when the profile has >= 2 titles), and every scene length
    within the series' min * 0.8 .. max * 1.25 minutes."""
    r = []
    if not profile or profile["n_boundaries"] < 3:
        return False, ["no profile with >=3 known boundaries"]
    if profile["transition"] == "hardcut":
        return False, ["series profile is hard-cut: black cuts are not part of its signature"]
    n = len(cuts) + 1
    sc = profile["scene_count"]
    if profile["n_titles"] >= 2 and not (sc["min"] - 1 <= n <= sc["max"] + 1):
        r.append(f"{n} scenes outside the series range {sc['min']}-{sc['max']}")
    ts = [0, *(c["t"] for c in cuts), total]
    lengths = [(b - a) / 60 for a, b in zip(ts, ts[1:])]
    sl = profile["scene_len_min"]
    bad = [round(x, 1) for x in lengths if not (sl["min"] * 0.8 <= x <= sl["max"] * 1.25)]
    if bad:
        r.append(f"scene lengths {bad} min outside the series range {sl['min']}-{sl['max']}")
    return (not r), r
