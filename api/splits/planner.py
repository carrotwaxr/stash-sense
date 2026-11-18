"""The full split planner (#131): one file's signals -> a SplitPlan with cuts, confidence and reasons.

A port of the October `rescore3.run` without the catalog-listing branches (no listing source until #135):

1. A DVD menu loop at the end trims the file at `keep_until`.
2. Black candidates (`detect.candidates` + `refine`) and static-snow pairs (`static.static_candidates`) go
   through one `detect.plan_cuts`, with the series profile when it has at least 3 known boundaries.
3. `profiles.judge` confirms the plan against the series.
4. Vetoes: a scene longer than any in the series that holds an unconfirmed black, or a 1-frame black in a
   series whose blacks are longer, cancel the profile's confirmation.
5. Coverage edges (stretches of this file that other scenes' files hold) become cuts, snapped to a black
   run start within 10 s; a detector cut within 15 s of an edge is replaced by it.

Confidence (decided 2026-11-18 after the regression set showed 9 of 12 high plans missing labels): `high` needs
a judged plan from a series profile whose transition is `black` (black rate >= 0.75), or a coverage plan in which
a detector cut confirms every edge and the file is at least 90% covered. A judged plan from a `mixed` series
stays `medium`, since a fade-only plan can miss its hard-cut boundaries. Without a profile or coverage the plan
is `medium` when every scene is 5-60 min (the `plan_new.py` rule), else `low`. October's relaxed fallback is not
ported (it never ran there) and neither is the invented no-profile `high` rule: both gave wrong high plans.
"""

from dataclasses import dataclass

from . import detect, profiles, static

PLANNER_VERSION = 1  # bump when a rule or number below changes; the job then re-plans and reports what moved

PROFILE_MIN_BOUNDARIES = 3  # a series profile is used with at least this many labelled boundaries
SCENE_MIN_MIN, SCENE_MAX_MIN = 5, 60  # plausible scene lengths (minutes) for medium
LONG_SCENE_BLACK_S = 0.8  # an unconfirmed black this long inside an over-long scene: probably a missed cut
ONE_FRAME_S = 0.3  # a cut black shorter than this is a single frame ...
ONE_FRAME_SERIES_P10 = 0.35  # ... suspicious in a series whose 10th-percentile black is longer
COVERAGE_SNAP_S = 10.0  # a coverage edge moves to a black run start within this
COVERAGE_MATCH_S = 15.0  # a detector cut within this of a coverage edge is the same boundary
COVERAGE_HIGH_SHARE = 0.9  # covered share of the file for a detector-confirmed coverage plan to be high
EDGE_END_GUARD_S = 1.0  # coverage edges this close to the file's start or end are not cuts
MAX_LEFTOVERS = 12


@dataclass
class SplitPlan:
    cuts: list[dict]  # {t, kind: black|static|coverage, len, sil, dip_ratio, min_y, evidence: str}
    keep_until: float | None
    confidence: str  # high | medium | low
    reasons: list[str]  # why this confidence, one short sentence each
    leftovers: list[dict]  # strongest non-strong candidates (max 12), shown as hints on the card
    profile_used: bool

    @property
    def leave_whole_suggested(self) -> bool:
        return not self.cuts


def _num(x):
    return f"{x:g}" if isinstance(x, (int, float)) else str(x)


def _s(x):
    return f"{x:.1f}"


def _evidence(c) -> str:
    if c.get("kind") == "static":
        a, b = c.get("pair") or (c["t"], c["t"] + c["len"])
        return f"static pair {_s(c['len'])} s (bursts at {_s(a)} and {_s(b)} s), silence {_s(c['sil'])} s"
    ref = c.get("ref")
    src = f", from ~{ref}" if ref is not None else ""
    return f"black {_s(c['len'])} s (min Y {_num(c['min_y'])}{src}), silence {_s(c['sil'])} s"


def _shape(c, kind=None) -> dict:
    out = {
        "t": c["t"],
        "kind": kind or c.get("kind", "black"),
        "len": c.get("len", 0.0),
        "sil": c.get("sil", 0.0),
        "dip_ratio": c.get("dip_ratio"),
        "min_y": c.get("min_y"),
        "evidence": _evidence(c),
    }
    if c.get("pair"):
        out["pair"] = c["pair"]
    return out


def _lengths_min(cuts, total):
    ts = [0, *(c["t"] for c in cuts), total]
    return [round((b - a) / 60, 1) for a, b in zip(ts, ts[1:])]


def _covered_share(coverage, total):
    spans = sorted((max(0.0, a), min(total, b)) for a, b in coverage if b > a)
    covered = 0.0
    end = 0.0
    for a, b in spans:
        a = max(a, end)
        if b > a:
            covered += b - a
            end = b
    return covered / total if total > 0 else 0.0


def _coverage_cuts(d, coverage, total, detector_cuts):
    """(cuts, all edges confirmed): one cut per coverage edge, snapped to a black run start within 10 s.
    A detector cut within 15 s of the edge is replaced by the snapped edge and lends it its black features."""
    edges = []
    for e in sorted(x for r in coverage for x in r):
        if EDGE_END_GUARD_S < e < total - EDGE_END_GUARD_S and all(abs(e - x) > COVERAGE_MATCH_S for x in edges):
            edges.append(e)
    runs = [r for r in detect.dark_runs(d["fr"]) if r[2] >= detect.MIN_BLACK_S and r[0] < total]
    out = []
    replaced = set()
    confirmed = bool(edges)
    for e in edges:
        near = [r for r in runs if abs(r[0] - e) <= COVERAGE_SNAP_S]
        run = min(near, key=lambda r: abs(r[0] - e)) if near else None
        t = round(run[0], 1) if run else e
        match = [i for i, c in enumerate(detector_cuts) if abs(c["t"] - e) <= COVERAGE_MATCH_S]
        if not match:
            confirmed = False
        replaced.update(match)
        if match:
            src = dict(detector_cuts[min(match, key=lambda i: abs(detector_cuts[i]["t"] - e))])
        elif run:
            sil = detect.features_at(d, (run[0] + run[1]) / 2)["sil_len"]
            src = detect.refine(d, [{"t": t, "len": run[2], "min_y": run[3], "sil": sil}])[0]
        else:
            src = None
        if src is None:
            cut = {"t": t, "kind": "coverage", "len": 0.0, "sil": 0.0, "dip_ratio": None, "min_y": None}
            how = "no black within 10 s"
        else:
            cut = _shape(src, kind="coverage")
            how = f"snapped to {cut['evidence']}" if src.get("kind") != "static" else cut["evidence"]
            cut["t"] = t
        cut["edge"] = e
        cut["evidence"] = f"coverage edge {_s(e)} s (another scene's file), {how}"
        out.append(cut)
    kept = [c for i, c in enumerate(detector_cuts) if i not in replaced]
    return sorted(kept + out, key=lambda c: c["t"]), confirmed


def plan_file(
    d: dict,
    total: float,
    *,
    profile: dict | None = None,
    coverage: list[tuple[float, float]] | None = None,
) -> SplitPlan:
    """Plan the cuts of one file from its signal dict (`signals.to_dict`) and duration in seconds."""
    reasons = []
    mk = detect.menu_keep_until(d["fr"])
    keep = mk["keep_until"] if mk else None
    eff = keep or total
    cands = detect.refine(d, detect.candidates(d)) + static.static_candidates(d, eff)
    cands.sort(key=lambda c: c["t"])
    if keep:
        cands = [c for c in cands if c["t"] < keep]
        reasons.append(f"DVD menu loop from ~{keep:.0f} s: the rest of the file is dropped")

    pf = profile if profile and profile["n_boundaries"] >= PROFILE_MIN_BOUNDARIES else None
    cuts, left = detect.plan_cuts(cands, pf, eff)
    ok, why = profiles.judge(cuts, pf, eff)
    why = list(why)

    if pf and ok and pf.get("scene_len_min"):
        mx = pf["scene_len_min"]["max"] * 60
        ts = [0, *(c["t"] for c in cuts), eff]
        for a, b in zip(ts, ts[1:]):
            if b - a > mx and any(a < x["t"] < b and x["len"] >= LONG_SCENE_BLACK_S for x in left):
                ok = False
                why.append(f"scene {round((b - a) / 60, 1)} min is longer than any in the series and holds an unconfirmed black")
    p10 = ((pf or {}).get("summary", {}).get("black_len") or {}).get("p10") or 9
    if ok and pf and any(c["len"] < ONE_FRAME_S for c in cuts) and p10 > ONE_FRAME_SERIES_P10:
        ok = False
        why.append("a 1-frame black (< 0.3 s) in a series whose blacks are longer")

    detector_cuts = [_shape(c) for c in cuts]
    cov_high = False
    if coverage:
        final, confirmed = _coverage_cuts(d, coverage, eff, detector_cuts)
        share = _covered_share(coverage, eff)
        cov_high = confirmed and share >= COVERAGE_HIGH_SHARE
        if pf and final != detector_cuts:
            ok3, why3 = profiles.judge(final, pf, eff)
            ok = ok and ok3
            why += [w for w in why3 if w not in why]
        detector_cuts = final
        reasons.append(f"{share:.0%} of the file is held by other scenes' files; their edges are cuts")
    cuts = detector_cuts

    lengths = _lengths_min(cuts, eff)
    plausible = bool(cuts) and all(SCENE_MIN_MIN <= x <= SCENE_MAX_MIN for x in lengths)
    mixed_ok = bool(cuts and ok and pf and pf["transition"] != "black")
    if cuts and ok and pf and not mixed_ok:
        confidence = "high"
        reasons.append("the series profile confirms the scene count and lengths")
    elif cuts and cov_high:
        confidence = "high"
        reasons.append("the file is at least 90% covered and a detector cut confirms every coverage edge")
    elif plausible:
        confidence = "medium"
        reasons.append("strong transitions and every scene is 5-60 min, but nothing confirms the plan")
    elif cuts:
        confidence = "low"
        reasons.append(f"a scene length is outside 5-60 min ({lengths})")
    else:
        confidence = "low"
        reasons.append("no transition found: leave whole unless the owner marks cuts")
    if mixed_ok and confidence != "high":
        reasons.append("the series mixes fades and hard cuts, so a fade-only plan can miss boundaries")
    if pf:
        reasons += why

    strongest = sorted(left, key=lambda c: (c["len"], c["sil"]), reverse=True)[:MAX_LEFTOVERS]
    leftovers = [_shape(c) for c in sorted(strongest, key=lambda c: c["t"])]
    return SplitPlan(
        cuts=cuts,
        keep_until=keep,
        confidence=confidence,
        reasons=reasons,
        leftovers=leftovers,
        profile_used=pf is not None,
    )
