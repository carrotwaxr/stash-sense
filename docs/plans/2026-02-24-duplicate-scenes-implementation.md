# Duplicate Scenes Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the broken duplicate scenes scoring (22K junk recs), add phash-based candidate generation, and build the missing plugin UI with merge/delete/dismiss actions.

**Architecture:** Three-layer change: (1) Rewrite scoring engine to use phash + metadata + face corroboration tiers, (2) Replace face candidate generation with phash Hamming distance, (3) Build list card + detail view in plugin with merge/delete/dismiss via Stash's `sceneMerge`/`sceneDestroy` mutations.

**Tech Stack:** Python (FastAPI), SQLite, JavaScript (Stash plugin), Stash GraphQL API

**Design Doc:** `docs/plans/2026-02-24-duplicate-scenes-redesign.md`

---

### Task 1: Fix Face Scoring — Require Shared Performers

**Files:**
- Modify: `api/duplicate_detection/scoring.py:118-154`
- Modify: `api/duplicate_detection/models.py:85-94`
- Test: `api/tests/test_duplicate_scoring.py`

**Step 1: Update SignalBreakdown model to support phash**

Add `phash_distance` field to `SignalBreakdown` in `api/duplicate_detection/models.py:85-94`:

```python
@dataclass
class SignalBreakdown:
    """Breakdown of signals contributing to duplicate confidence."""
    stashbox_match: bool
    stashbox_endpoint: Optional[str]
    phash_distance: Optional[int]  # Hamming distance, None if no phash
    face_score: float  # 0-75
    face_reasoning: str
    metadata_score: float  # 0-60
    metadata_reasoning: str
```

**Step 2: Write failing tests for the new face scoring**

Add to `api/tests/test_duplicate_scoring.py` — replace `TestFaceSignatureSimilarity` class (lines 152-226):

```python
class TestFaceSignatureSimilarity:
    """Tests for the fixed face_signature_similarity function."""

    def test_zero_shared_performers_returns_zero(self):
        """Two scenes with completely different performers score 0."""
        fp_a = SceneFingerprint(
            stash_scene_id=1, total_faces_detected=10, frames_analyzed=60,
            faces={"performer_a": FaceAppearance("performer_a", 10, 0.9, 1.0)},
        )
        fp_b = SceneFingerprint(
            stash_scene_id=2, total_faces_detected=10, frames_analyzed=60,
            faces={"performer_b": FaceAppearance("performer_b", 10, 0.9, 1.0)},
        )
        score, reason = face_signature_similarity(fp_a, fp_b)
        assert score == 0.0
        assert "No shared performers" in reason

    def test_identical_cast_scores_high(self):
        """Two scenes with identical cast and proportions score high."""
        faces = {
            "p1": FaceAppearance("p1", 5, 0.9, 0.5),
            "p2": FaceAppearance("p2", 5, 0.9, 0.5),
        }
        fp_a = SceneFingerprint(stash_scene_id=1, total_faces_detected=10, frames_analyzed=60, faces=faces)
        fp_b = SceneFingerprint(stash_scene_id=2, total_faces_detected=10, frames_analyzed=60, faces=dict(faces))
        score, reason = face_signature_similarity(fp_a, fp_b)
        assert score >= 70.0
        assert "2 shared" in reason

    def test_partial_cast_overlap_scores_moderately(self):
        """Scenes sharing some but not all performers score based on Jaccard."""
        fp_a = SceneFingerprint(
            stash_scene_id=1, total_faces_detected=10, frames_analyzed=60,
            faces={
                "p1": FaceAppearance("p1", 5, 0.9, 0.5),
                "p2": FaceAppearance("p2", 5, 0.9, 0.5),
            },
        )
        fp_b = SceneFingerprint(
            stash_scene_id=2, total_faces_detected=10, frames_analyzed=60,
            faces={
                "p1": FaceAppearance("p1", 5, 0.9, 0.5),
                "p3": FaceAppearance("p3", 5, 0.9, 0.5),
            },
        )
        score, reason = face_signature_similarity(fp_a, fp_b)
        # Jaccard = 1/3 ≈ 0.33, so base ≈ 20, plus proportion bonus
        assert 15.0 <= score <= 40.0

    def test_unknown_performers_excluded(self):
        """The 'unknown' performer ID is excluded from calculations."""
        fp_a = SceneFingerprint(
            stash_scene_id=1, total_faces_detected=10, frames_analyzed=60,
            faces={"unknown": FaceAppearance("unknown", 10, 0.5, 1.0)},
        )
        fp_b = SceneFingerprint(
            stash_scene_id=2, total_faces_detected=10, frames_analyzed=60,
            faces={"unknown": FaceAppearance("unknown", 10, 0.5, 1.0)},
        )
        score, _ = face_signature_similarity(fp_a, fp_b)
        assert score == 0.0

    def test_cap_at_75(self):
        """Face score is capped at 75, not 85."""
        faces = {"p1": FaceAppearance("p1", 10, 0.95, 1.0)}
        fp_a = SceneFingerprint(stash_scene_id=1, total_faces_detected=10, frames_analyzed=60, faces=faces)
        fp_b = SceneFingerprint(stash_scene_id=2, total_faces_detected=10, frames_analyzed=60, faces=dict(faces))
        score, _ = face_signature_similarity(fp_a, fp_b)
        assert score <= 75.0
```

**Step 3: Run tests to verify they fail**

Run: `cd api && ../.venv/bin/python -m pytest tests/test_duplicate_scoring.py::TestFaceSignatureSimilarity -v`
Expected: Multiple failures (old function gives non-zero scores for zero shared performers)

**Step 4: Implement the fixed face scoring**

Replace `face_signature_similarity` in `api/duplicate_detection/scoring.py:118-154`:

```python
def face_signature_similarity(
    fp_a: SceneFingerprint, fp_b: SceneFingerprint
) -> tuple[float, str]:
    """
    Calculate face signature similarity score (0-75).

    Requires at least one shared performer. Score based on Jaccard
    similarity of performer sets, with proportion bonus for shared performers.
    """
    all_a = set(fp_a.faces.keys()) - {"unknown"}
    all_b = set(fp_b.faces.keys()) - {"unknown"}
    shared = all_a & all_b

    if not shared:
        return 0.0, "No shared performers"

    # Jaccard similarity of performer sets
    union = all_a | all_b
    jaccard = len(shared) / len(union) if union else 0.0

    # Proportion similarity for shared performers only
    proportion_diffs = []
    for pid in shared:
        diff = abs(fp_a.faces[pid].proportion - fp_b.faces[pid].proportion)
        proportion_diffs.append(diff)
    avg_diff = sum(proportion_diffs) / len(proportion_diffs)

    # Score: Jaccard drives the base, proportion similarity is a bonus
    base = jaccard * 60.0
    proportion_bonus = max(0.0, 15.0 * (1.0 - avg_diff * 4.0))
    score = min(base + proportion_bonus, 75.0)

    reason = f"{len(shared)} shared performers (Jaccard {jaccard:.0%}), {avg_diff:.1%} avg proportion diff"
    return score, reason
```

**Step 5: Run tests to verify they pass**

Run: `cd api && ../.venv/bin/python -m pytest tests/test_duplicate_scoring.py::TestFaceSignatureSimilarity -v`
Expected: All PASS

**Step 6: Commit**

```
git add api/duplicate_detection/scoring.py api/duplicate_detection/models.py api/tests/test_duplicate_scoring.py
git commit -m "fix: rewrite face scoring to require shared performers

Face score now returns 0 when no performers are actually shared.
Score based on Jaccard similarity of performer sets (not proportion
distribution). Capped at 75 instead of 85. Add phash_distance field
to SignalBreakdown model."
```

---

### Task 2: Rewrite Metadata Scoring

**Files:**
- Modify: `api/duplicate_detection/scoring.py:64-115`
- Test: `api/tests/test_duplicate_scoring.py`

**Step 1: Write failing tests for revised metadata scoring**

Replace `TestMetadataScore` class in `api/tests/test_duplicate_scoring.py`:

```python
class TestMetadataScore:
    """Tests for revised metadata scoring with date as strongest signal."""

    def _scene(self, **kwargs):
        defaults = dict(scene_id="1", studio_id=None, performer_ids=set(), date=None, duration_seconds=None)
        defaults.update(kwargs)
        return SceneMetadata(**defaults)

    def test_same_date_exact_performers_highest_score(self):
        """Same date + exact performer match = 45 points (strongest combo)."""
        a = self._scene(scene_id="1", date="2024-01-15", performer_ids={"p1", "p2"}, studio_id="s1")
        b = self._scene(scene_id="2", date="2024-01-15", performer_ids={"p1", "p2"}, studio_id="s1")
        score, reason = metadata_score(a, b)
        assert score >= 55.0  # 45 (date+perf) + 10 (studio)
        assert "Same date" in reason

    def test_same_date_partial_performers(self):
        """Same date + partial overlap = 35 points."""
        a = self._scene(scene_id="1", date="2024-01-15", performer_ids={"p1", "p2"})
        b = self._scene(scene_id="2", date="2024-01-15", performer_ids={"p1", "p3"})
        score, reason = metadata_score(a, b)
        assert 35.0 <= score <= 50.0

    def test_exact_performers_no_date(self):
        """Exact performer match without date = 25 points."""
        a = self._scene(scene_id="1", performer_ids={"p1", "p2"})
        b = self._scene(scene_id="2", performer_ids={"p1", "p2"})
        score, reason = metadata_score(a, b)
        assert 25.0 <= score <= 40.0

    def test_studio_only_mild_boost(self):
        """Same studio alone = 10 points."""
        a = self._scene(scene_id="1", studio_id="s1")
        b = self._scene(scene_id="2", studio_id="s1")
        score, reason = metadata_score(a, b)
        assert score == 10.0

    def test_no_metadata_returns_zero(self):
        """No useful metadata = 0."""
        a = self._scene(scene_id="1")
        b = self._scene(scene_id="2")
        score, _ = metadata_score(a, b)
        assert score == 0.0

    def test_duration_penalty_extreme(self):
        """Extreme duration mismatch (3min vs 45min) applies penalty."""
        a = self._scene(scene_id="1", date="2024-01-15", performer_ids={"p1"}, duration_seconds=180)
        b = self._scene(scene_id="2", date="2024-01-15", performer_ids={"p1"}, duration_seconds=2700)
        score, reason = metadata_score(a, b)
        # Ratio = 180/2700 = 0.067, penalty -20
        score_no_dur = self._scene(scene_id="1", date="2024-01-15", performer_ids={"p1"})
        base_score, _ = metadata_score(score_no_dur, self._scene(scene_id="2", date="2024-01-15", performer_ids={"p1"}))
        assert score < base_score

    def test_duration_no_penalty_web_vs_dvd(self):
        """50m web vs 25m DVD (ratio 0.5) = no penalty."""
        a = self._scene(scene_id="1", date="2024-01-15", performer_ids={"p1"}, duration_seconds=3000)
        b = self._scene(scene_id="2", date="2024-01-15", performer_ids={"p1"}, duration_seconds=1500)
        score, reason = metadata_score(a, b)
        # Ratio 0.5 → no penalty
        assert "penalty" not in reason.lower() or "no penalty" in reason.lower()

    def test_cap_at_60(self):
        """Metadata score capped at 60."""
        a = self._scene(scene_id="1", date="2024-01-15", performer_ids={"p1", "p2"}, studio_id="s1")
        b = self._scene(scene_id="2", date="2024-01-15", performer_ids={"p1", "p2"}, studio_id="s1")
        score, _ = metadata_score(a, b)
        assert score <= 60.0
```

**Step 2: Run tests to verify they fail**

Run: `cd api && ../.venv/bin/python -m pytest tests/test_duplicate_scoring.py::TestMetadataScore -v`

**Step 3: Implement revised metadata scoring**

Replace `metadata_score` in `api/duplicate_detection/scoring.py:64-115`:

```python
def metadata_score(scene_a: SceneMetadata, scene_b: SceneMetadata) -> tuple[float, str]:
    """
    Calculate metadata similarity score (0-60).

    Date + performers is the strongest signal. Studio is a mild boost.
    Duration only penalizes at extremes, never adds confidence.
    """
    score = 0.0
    reasons = []

    # Date comparison
    days = _days_between(scene_a.date, scene_b.date)
    same_date = days is not None and days == 0
    close_date = days is not None and days <= 7

    # Performer comparison
    performer_overlap = _jaccard_similarity(scene_a.performer_ids, scene_b.performer_ids)
    exact_performers = performer_overlap == 1.0 and bool(scene_a.performer_ids)
    partial_performers = performer_overlap >= 0.5

    # Primary scoring: date + performers is strongest
    if same_date and exact_performers:
        score += 45.0
        reasons.append("Same date + exact performer match")
    elif same_date and partial_performers:
        score += 35.0
        reasons.append(f"Same date + performers overlap ({performer_overlap:.0%})")
    elif close_date and exact_performers:
        score += 35.0
        reasons.append("Release dates within 7 days + exact performer match")
    elif exact_performers:
        score += 25.0
        reasons.append("Exact performer match")
    elif same_date and (scene_a.performer_ids or scene_b.performer_ids):
        score += 20.0
        reasons.append("Same date")
    elif partial_performers:
        score += 15.0
        reasons.append(f"Performers overlap ({performer_overlap:.0%})")

    # Studio boost (mild, not decisive)
    if scene_a.studio_id and scene_b.studio_id and scene_a.studio_id == scene_b.studio_id:
        score += 10.0
        reasons.append("Same studio")

    # Duration penalty (only subtracts, never adds)
    if scene_a.duration_seconds and scene_b.duration_seconds:
        shorter = min(scene_a.duration_seconds, scene_b.duration_seconds)
        longer = max(scene_a.duration_seconds, scene_b.duration_seconds)
        ratio = shorter / longer if longer > 0 else 1.0
        if ratio < 0.15:
            score -= 20.0
            reasons.append("Duration ratio < 15% (penalty -20)")
        elif ratio < 0.30:
            score -= 10.0
            reasons.append("Duration ratio < 30% (penalty -10)")

    if not reasons:
        return 0.0, "No metadata signals"

    return min(max(score, 0.0), 60.0), " + ".join(reasons)
```

**Step 4: Run tests to verify they pass**

Run: `cd api && ../.venv/bin/python -m pytest tests/test_duplicate_scoring.py::TestMetadataScore -v`
Expected: All PASS

**Step 5: Commit**

```
git add api/duplicate_detection/scoring.py api/tests/test_duplicate_scoring.py
git commit -m "feat: rewrite metadata scoring with date as strongest signal

Date + performers is the strongest combo (45 pts). Studio is a mild
boost (10 pts). Duration never adds confidence, only penalizes at
extreme ratios (< 0.15 = -20, < 0.30 = -10). Handles 50m web vs
25m DVD case without penalty."
```

---

### Task 3: Add Phash Scoring and Rewrite Combined Confidence

**Files:**
- Modify: `api/duplicate_detection/scoring.py:157-246`
- Modify: `api/duplicate_detection/__init__.py`
- Test: `api/tests/test_duplicate_scoring.py`

**Step 1: Write failing tests for phash scoring and combined confidence**

Add new test classes to `api/tests/test_duplicate_scoring.py`:

```python
class TestPhashScore:
    """Tests for phash Hamming distance scoring."""

    def test_identical_phash(self):
        score, reason = phash_score(0)
        assert score >= 85.0
        assert "identical" in reason.lower() or "distance 0" in reason.lower()

    def test_very_close_phash(self):
        """Hamming distance 3 = strong match."""
        score, _ = phash_score(3)
        assert 80.0 <= score <= 90.0

    def test_moderate_phash(self):
        """Hamming distance 8 = moderate match."""
        score, _ = phash_score(8)
        assert 40.0 <= score <= 65.0

    def test_distant_phash(self):
        """Hamming distance > 10 = no score."""
        score, _ = phash_score(11)
        assert score == 0.0

    def test_none_returns_zero(self):
        score, _ = phash_score(None)
        assert score == 0.0


class TestHammingDistance:
    """Tests for phash Hamming distance computation."""

    def test_identical(self):
        assert hamming_distance("eb716d2e0149f2d1", "eb716d2e0149f2d1") == 0

    def test_one_bit_diff(self):
        # Flip one bit: last hex char d1 (1101 0001) -> d0 (1101 0000)
        assert hamming_distance("eb716d2e0149f2d1", "eb716d2e0149f2d0") == 1

    def test_completely_different(self):
        assert hamming_distance("0000000000000000", "ffffffffffffffff") == 64

    def test_none_handling(self):
        assert hamming_distance(None, "eb716d2e0149f2d1") is None
        assert hamming_distance("eb716d2e0149f2d1", None) is None


class TestCombinedConfidenceRedesign:
    """Tests for the redesigned calculate_duplicate_confidence."""

    def _scene(self, **kwargs):
        defaults = dict(scene_id="1", studio_id=None, performer_ids=set(), date=None, duration_seconds=None, stash_ids=[])
        defaults.update(kwargs)
        return SceneMetadata(**defaults)

    def test_stashbox_match_still_100(self):
        a = self._scene(scene_id="1", stash_ids=[StashID("https://stashdb.org", "abc")])
        b = self._scene(scene_id="2", stash_ids=[StashID("https://stashdb.org", "abc")])
        match = calculate_duplicate_confidence(a, b)
        assert match is not None
        assert match.confidence == 100.0

    def test_strong_phash_scores_high(self):
        """Close phash (distance 2) = 85-95% even without metadata."""
        a = self._scene(scene_id="1")
        b = self._scene(scene_id="2")
        match = calculate_duplicate_confidence(a, b, phash_distance=2)
        assert match is not None
        assert 85.0 <= match.confidence <= 95.0

    def test_moderate_phash_needs_corroboration(self):
        """Moderate phash (distance 8) alone scores lower than threshold."""
        a = self._scene(scene_id="1")
        b = self._scene(scene_id="2")
        match = calculate_duplicate_confidence(a, b, phash_distance=8)
        # Without corroboration, should be below useful threshold
        assert match is None or match.confidence < 50.0

    def test_moderate_phash_with_metadata_scores_well(self):
        """Moderate phash + metadata = good confidence."""
        a = self._scene(scene_id="1", studio_id="s1", performer_ids={"p1"}, date="2024-01-15")
        b = self._scene(scene_id="2", studio_id="s1", performer_ids={"p1"}, date="2024-01-15")
        match = calculate_duplicate_confidence(a, b, phash_distance=8)
        assert match is not None
        assert match.confidence >= 60.0

    def test_metadata_only_catches_different_intros(self):
        """Strong metadata with no phash still creates recommendation."""
        a = self._scene(scene_id="1", studio_id="s1", performer_ids={"p1", "p2"}, date="2024-01-15")
        b = self._scene(scene_id="2", studio_id="s1", performer_ids={"p1", "p2"}, date="2024-01-15")
        match = calculate_duplicate_confidence(a, b)
        assert match is not None
        assert match.confidence >= 50.0

    def test_no_signals_returns_none(self):
        a = self._scene(scene_id="1")
        b = self._scene(scene_id="2")
        match = calculate_duplicate_confidence(a, b)
        assert match is None

    def test_same_scene_returns_none(self):
        a = self._scene(scene_id="1")
        match = calculate_duplicate_confidence(a, a)
        assert match is None

    def test_signal_breakdown_includes_phash(self):
        a = self._scene(scene_id="1")
        b = self._scene(scene_id="2")
        match = calculate_duplicate_confidence(a, b, phash_distance=2)
        assert match is not None
        assert match.signal_breakdown.phash_distance == 2
```

**Step 2: Run tests to verify they fail**

Run: `cd api && ../.venv/bin/python -m pytest tests/test_duplicate_scoring.py::TestPhashScore tests/test_duplicate_scoring.py::TestHammingDistance tests/test_duplicate_scoring.py::TestCombinedConfidenceRedesign -v`
Expected: ImportError (phash_score, hamming_distance not defined)

**Step 3: Implement phash scoring and rewrite calculate_duplicate_confidence**

Add `phash_score` and `hamming_distance` to `api/duplicate_detection/scoring.py`, and rewrite `calculate_duplicate_confidence`:

```python
def hamming_distance(phash_a: Optional[str], phash_b: Optional[str]) -> Optional[int]:
    """Calculate Hamming distance between two hex-encoded 64-bit phashes."""
    if phash_a is None or phash_b is None:
        return None
    try:
        xor = int(phash_a, 16) ^ int(phash_b, 16)
        return bin(xor).count("1")
    except (ValueError, TypeError):
        return None


def phash_score(distance: Optional[int]) -> tuple[float, str]:
    """
    Calculate phash similarity score (0-85) from Hamming distance.

    Distance 0 = identical content (85 pts)
    Distance 1-4 = very similar (70-85 pts)
    Distance 5-7 = moderately similar (40-65 pts)
    Distance 8-10 = weak similarity (20-35 pts)
    Distance >10 = no signal (0 pts)
    """
    if distance is None:
        return 0.0, "No phash available"
    if distance > 10:
        return 0.0, f"Phash distance {distance} (too far)"

    # Linear interpolation: distance 0 → 85, distance 10 → 20
    score = 85.0 - (distance * 6.5)
    reason = f"Phash distance {distance}"
    if distance == 0:
        reason = "Identical phash"
    elif distance <= 4:
        reason = f"Very similar phash (distance {distance})"
    elif distance <= 7:
        reason = f"Moderately similar phash (distance {distance})"
    else:
        reason = f"Weak phash similarity (distance {distance})"

    return score, reason


def calculate_duplicate_confidence(
    scene_a: SceneMetadata,
    scene_b: SceneMetadata,
    fp_a: Optional[SceneFingerprint] = None,
    fp_b: Optional[SceneFingerprint] = None,
    phash_distance: Optional[int] = None,
) -> Optional[DuplicateMatch]:
    """
    Calculate overall duplicate confidence combining all signals.

    Tiered approach:
      Tier 1: Stash-box ID match → 100%
      Tier 2: Strong phash (distance ≤ 4) → 85-95%
      Tier 3: Moderate phash + corroboration → 60-84%
      Tier 4: Metadata + face only → 50-70%
    """
    if scene_a.scene_id == scene_b.scene_id:
        return None

    # Tier 1: Stash-box ID match
    stashbox = check_stashbox_match(scene_a, scene_b)
    if stashbox.matched:
        return DuplicateMatch(
            scene_a_id=int(scene_a.scene_id),
            scene_b_id=int(scene_b.scene_id),
            confidence=100.0,
            reasoning=[f"Identical stash-box ID: {stashbox.stash_id}"],
            signal_breakdown=SignalBreakdown(
                stashbox_match=True,
                stashbox_endpoint=stashbox.endpoint,
                phash_distance=None,
                face_score=0.0,
                face_reasoning="",
                metadata_score=0.0,
                metadata_reasoning="",
            ),
        )

    # Compute individual signals
    p_score, p_reasoning = phash_score(phash_distance)

    face_sc = 0.0
    face_reasoning = "No fingerprint available"
    if fp_a and fp_b:
        if fp_a.total_faces_detected == 0 and fp_b.total_faces_detected == 0:
            face_reasoning = "No faces detected in either scene"
        elif fp_a.total_faces_detected == 0 or fp_b.total_faces_detected == 0:
            face_reasoning = "Asymmetric face detection"
        else:
            face_sc, face_reasoning = face_signature_similarity(fp_a, fp_b)

    meta_sc, meta_reasoning = metadata_score(scene_a, scene_b)

    # No signals = no match
    if p_score == 0 and face_sc == 0 and meta_sc == 0:
        return None

    # Calculate combined confidence
    if p_score >= 70.0:
        # Tier 2: Strong phash — phash drives confidence, others are bonuses
        confidence = p_score + min(meta_sc * 0.15, 5.0) + min(face_sc * 0.1, 5.0)
    elif p_score > 0:
        # Tier 3: Moderate phash — needs corroboration
        corroboration = max(meta_sc, face_sc)
        if corroboration > 0:
            confidence = p_score + corroboration * 0.4
        else:
            confidence = p_score * 0.6  # Unconfirmed moderate phash
    else:
        # Tier 4: No phash — metadata + face only
        primary = max(meta_sc, face_sc)
        secondary = min(meta_sc, face_sc)
        confidence = primary + secondary * 0.3

    # Cap at 95% without stash-box confirmation
    confidence = min(confidence, 95.0)

    # Build reasoning
    reasoning = []
    if p_score > 0:
        reasoning.append(p_reasoning)
    if face_sc > 0:
        reasoning.append(f"Face analysis: {face_reasoning}")
    if meta_sc > 0:
        reasoning.append(f"Metadata: {meta_reasoning}")

    if confidence >= 80:
        reasoning.insert(0, "High confidence duplicate")
    elif confidence >= 50:
        reasoning.insert(0, "Likely duplicate")
    else:
        reasoning.insert(0, "Possible duplicate")

    return DuplicateMatch(
        scene_a_id=int(scene_a.scene_id),
        scene_b_id=int(scene_b.scene_id),
        confidence=round(confidence, 1),
        reasoning=reasoning,
        signal_breakdown=SignalBreakdown(
            stashbox_match=False,
            stashbox_endpoint=None,
            phash_distance=phash_distance,
            face_score=face_sc,
            face_reasoning=face_reasoning,
            metadata_score=meta_sc,
            metadata_reasoning=meta_reasoning,
        ),
    )
```

Update `api/duplicate_detection/__init__.py` to export `phash_score` and `hamming_distance`:

```python
from .scoring import (
    check_stashbox_match,
    metadata_score,
    face_signature_similarity,
    calculate_duplicate_confidence,
    phash_score,
    hamming_distance,
    StashboxMatchResult,
)

__all__ = [
    ...
    "phash_score",
    "hamming_distance",
]
```

**Step 4: Run all scoring tests**

Run: `cd api && ../.venv/bin/python -m pytest tests/test_duplicate_scoring.py -v`
Expected: All PASS

**Step 5: Commit**

```
git add api/duplicate_detection/scoring.py api/duplicate_detection/__init__.py api/tests/test_duplicate_scoring.py
git commit -m "feat: add phash scoring and rewrite combined confidence

Tiered scoring: stashbox (100%), strong phash (85-95%), moderate
phash + corroboration (60-84%), metadata + face only (50-70%).
Add hamming_distance() and phash_score() functions."
```

---

### Task 4: Rewrite Analyzer — Phash Candidates Replace Face Candidates

**Files:**
- Modify: `api/analyzers/duplicate_scenes.py`
- Modify: `api/recommendations_db.py` (add phash candidate generation method)
- Test: `api/tests/test_duplicate_scenes_analyzer.py`

**Step 1: Add phash-related DB methods**

Add to `api/recommendations_db.py` near the other duplicate candidate methods (after `generate_face_candidates` at line 1707):

```python
def store_scene_phashes(self, phashes: list[tuple[int, str]]) -> int:
    """
    Store scene phashes for duplicate candidate generation.
    Input: list of (stash_scene_id, phash_hex) tuples.
    Creates/replaces temp table for the current analysis run.
    Returns count stored.
    """
    with self._connection() as conn:
        conn.execute("DROP TABLE IF EXISTS _tmp_scene_phashes")
        conn.execute("""
            CREATE TEMP TABLE _tmp_scene_phashes (
                scene_id INTEGER PRIMARY KEY,
                phash INTEGER NOT NULL
            )
        """)
        rows = []
        for scene_id, phash_hex in phashes:
            try:
                rows.append((scene_id, int(phash_hex, 16)))
            except (ValueError, TypeError):
                continue
        conn.executemany(
            "INSERT OR IGNORE INTO _tmp_scene_phashes (scene_id, phash) VALUES (?, ?)",
            rows,
        )
        return len(rows)

def generate_phash_candidates(self, max_distance: int = 10) -> list[tuple[int, int, int]]:
    """
    Find all scene pairs with phash Hamming distance ≤ max_distance.
    Computes all-pairs comparison using the temp phash table.
    Returns list of (scene_a_id, scene_b_id, hamming_distance) in canonical order.
    """
    with self._connection() as conn:
        rows = conn.execute("SELECT scene_id, phash FROM _tmp_scene_phashes").fetchall()

    if not rows:
        return []

    # All-pairs Hamming distance comparison
    candidates = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            xor = rows[i][1] ^ rows[j][1]
            dist = bin(xor).count("1")
            if dist <= max_distance:
                a, b = rows[i][0], rows[j][0]
                if a > b:
                    a, b = b, a
                candidates.append((a, b, dist))

    return candidates
```

**Step 2: Update the analyzer to use phash candidates**

Rewrite `_generate_candidates` in `api/analyzers/duplicate_scenes.py:115-194`. The key changes:
- Query phashes from Stash using `get_scenes_with_fingerprints()`
- Store phashes in temp table
- Generate phash candidates instead of face candidates
- Store phash distances in a lookup dict for scoring phase
- Remove the `generate_face_candidates()` call

Also update `_score_candidates` to pass `phash_distance` to the scoring function.

Replace the full `_generate_candidates` method:

```python
async def _generate_candidates(self, run_id: int) -> int:
    """Phase 1: Generate candidate pairs from stash-box IDs, phash, and metadata."""
    orphans = self.rec_db.clear_orphaned_candidates()
    if orphans:
        logger.warning(f"Cleaned up {orphans} orphaned candidates (NULL run_id)")
    self.rec_db.clear_candidates(run_id)

    stashbox_index: dict[tuple[str, str], list[int]] = {}
    combined_index: dict[tuple[str, str], set[int]] = {}
    phash_data: list[tuple[int, str]] = []

    # Single pass: build indices from scenes with fingerprints (has phash)
    offset = 0
    total_scenes = 0
    while True:
        scenes, total = await self.stash.get_scenes_with_fingerprints(
            limit=self.batch_size, offset=offset,
        )
        total_scenes = total

        for scene in scenes:
            sid = int(scene["id"])

            # Stash-box index
            for stash_id_entry in scene.get("stash_ids", []):
                key = (stash_id_entry["endpoint"], stash_id_entry["stash_id"])
                stashbox_index.setdefault(key, []).append(sid)

            # Phash extraction from file fingerprints
            for f in scene.get("files", []):
                for fp in f.get("fingerprints", []):
                    if fp["type"] == "phash" and fp.get("value"):
                        phash_data.append((sid, fp["value"]))
                        break  # One phash per scene

        if len(scenes) == 0 or offset + len(scenes) >= total:
            break
        offset += self.batch_size
        await asyncio.sleep(0)

    logger.warning(f"Loaded {total_scenes} scenes ({len(phash_data)} with phash). Building candidates...")

    # Now fetch metadata for studio/performer index (separate query has these fields)
    offset = 0
    while True:
        scenes, total = await self.stash.get_scenes_for_fingerprinting(
            limit=self.batch_size, offset=offset,
        )

        for scene in scenes:
            sid = int(scene["id"])
            studio = scene.get("studio")
            if studio and studio.get("id"):
                studio_id = studio["id"]
                for performer in scene.get("performers", []):
                    key = (studio_id, performer["id"])
                    combined_index.setdefault(key, set()).add(sid)

        if len(scenes) == 0 or offset + len(scenes) >= total:
            break
        offset += self.batch_size
        await asyncio.sleep(0)

    # Source A: Stash-box candidates
    stashbox_pairs = []
    for key, scene_ids in stashbox_index.items():
        if len(scene_ids) > 1:
            for i, a in enumerate(scene_ids):
                for b in scene_ids[i + 1:]:
                    stashbox_pairs.append((a, b, "stashbox"))
    if stashbox_pairs:
        self.rec_db.insert_candidates_batch(stashbox_pairs, run_id)
        logger.warning(f"  Stash-box ID: {len(stashbox_pairs)} candidates")

    # Source B: Phash candidates (replaces face candidates)
    self._phash_distances = {}  # Store for scoring phase
    if phash_data:
        self.rec_db.store_scene_phashes(phash_data)
        phash_pairs = self.rec_db.generate_phash_candidates(max_distance=10)
        if phash_pairs:
            batch = []
            for a, b, dist in phash_pairs:
                batch.append((a, b, "phash"))
                self._phash_distances[(a, b)] = dist
            self.rec_db.insert_candidates_batch(batch, run_id)
            logger.warning(f"  Phash: {len(phash_pairs)} candidates ({self.rec_db.count_candidates(run_id)} after dedup)")

    # Source C: Metadata candidates
    metadata_pairs = []
    for key, scene_ids in combined_index.items():
        if len(scene_ids) > 1:
            sorted_ids = sorted(scene_ids)
            for i, a in enumerate(sorted_ids):
                for b in sorted_ids[i + 1:]:
                    metadata_pairs.append((a, b, "metadata"))
    if metadata_pairs:
        self.rec_db.insert_candidates_batch(metadata_pairs, run_id)
        logger.warning(f"  Metadata: {len(metadata_pairs)} candidates ({self.rec_db.count_candidates(run_id)} after dedup)")

    return self.rec_db.count_candidates(run_id)
```

Update `_score_candidates` to pass phash_distance:

In the scoring loop (around line 260), change the call to `calculate_duplicate_confidence`:

```python
# Look up phash distance for this pair
pair_key = (min(candidate["scene_a_id"], candidate["scene_b_id"]),
            max(candidate["scene_a_id"], candidate["scene_b_id"]))
phash_dist = self._phash_distances.get(pair_key)

match = calculate_duplicate_confidence(scene_a, scene_b, fp_a, fp_b, phash_distance=phash_dist)
```

**Step 3: Update tests**

Update `api/tests/test_duplicate_scenes_analyzer.py` — the key changes:
- `TestCandidateGeneration.test_generates_face_candidates` → rename/replace with phash test
- Mock `get_scenes_with_fingerprints` to return phash data
- Remove references to `generate_face_candidates()` in the analyzer tests

The face candidate generation DB method (`generate_face_candidates`) stays in `recommendations_db.py` for now (not deleted, just no longer called by the analyzer). The DB tests for it can remain.

Add a new test to `TestCandidateGeneration`:

```python
async def test_generates_phash_candidates(self):
    """Scenes with similar phashes become candidates."""
    # Two scenes with Hamming distance 2
    self.mock_stash.get_scenes_with_fingerprints = AsyncMock(return_value=([
        {"id": "1", "stash_ids": [], "files": [{"duration": 1800, "fingerprints": [{"type": "phash", "value": "eb716d2e0149f2d1"}]}]},
        {"id": "2", "stash_ids": [], "files": [{"duration": 1800, "fingerprints": [{"type": "phash", "value": "eb716d2e0149f2d0"}]}]},
        {"id": "3", "stash_ids": [], "files": [{"duration": 1800, "fingerprints": [{"type": "phash", "value": "0000000000000000"}]}]},
    ], 3))
    self.mock_stash.get_scenes_for_fingerprinting = AsyncMock(return_value=([], 0))

    analyzer = DuplicateScenesAnalyzer(self.mock_stash, self.db)
    count = await analyzer._generate_candidates(run_id=1)
    # Scenes 1 and 2 are close (distance 1), scene 3 is far from both
    assert count >= 1
    candidates = self.db.get_candidates_batch(run_id=1)
    scene_pairs = {(c["scene_a_id"], c["scene_b_id"]) for c in candidates}
    assert (1, 2) in scene_pairs
```

**Step 4: Run tests**

Run: `cd api && ../.venv/bin/python -m pytest tests/test_duplicate_scenes_analyzer.py -v`

**Step 5: Commit**

```
git add api/analyzers/duplicate_scenes.py api/recommendations_db.py api/tests/test_duplicate_scenes_analyzer.py
git commit -m "feat: replace face candidates with phash-based candidate generation

Query phash from Stash's file fingerprints, find scenes with Hamming
distance ≤ 10. Removes the 4M+ face candidate explosion. Face scoring
remains as a corroboration signal during the scoring phase."
```

---

### Task 5: Add Scene Merge and Delete to Stash Client

**Files:**
- Modify: `api/stash_client_unified.py`
- Modify: `api/recommendations_router.py`
- Test: `api/tests/test_duplicate_scenes_analyzer.py` (or a new test file)

**Step 1: Add `merge_scenes` and `destroy_scene` methods**

Add to `api/stash_client_unified.py` near `get_scene_by_id` (after line 1008):

```python
async def merge_scenes(
    self, source_ids: list[str], destination_id: str,
) -> dict:
    """
    Merge source scenes into destination scene via Stash's sceneMerge mutation.
    Consolidates files, tags, performers, stash IDs. Preserves play/o history.
    """
    query = """
    mutation SceneMerge($input: SceneMergeInput!) {
      sceneMerge(input: $input) {
        count
      }
    }
    """
    variables = {
        "input": {
            "source": source_ids,
            "destination": destination_id,
            "play_history": True,
            "o_history": True,
        }
    }
    data = await self._execute(query, variables)
    return data.get("sceneMerge", {})

async def destroy_scene(
    self, scene_id: str, delete_file: bool = False, delete_generated: bool = True,
) -> bool:
    """Delete a scene. Optionally deletes the underlying file."""
    query = """
    mutation SceneDestroy($input: SceneDestroyInput!) {
      sceneDestroy(input: $input)
    }
    """
    variables = {
        "input": {
            "id": scene_id,
            "delete_file": delete_file,
            "delete_generated": delete_generated,
        }
    }
    data = await self._execute(query, variables)
    return data.get("sceneDestroy", False)
```

Also extend `get_scene_by_id` to include more fields needed for the detail view — add `paths { screenshot }` and file details:

Replace the query in `get_scene_by_id` (lines 982-1005):

```python
async def get_scene_by_id(self, scene_id: str) -> Optional[dict]:
    """Get a scene by ID with full metadata for detail views."""
    query = """
    query GetScene($id: ID!) {
      findScene(id: $id) {
        id
        title
        date
        updated_at
        paths {
          screenshot
        }
        studio {
          id
          name
        }
        performers {
          id
          name
        }
        files {
          id
          path
          size
          duration
          video_codec
          width
          height
          fingerprints {
            type
            value
          }
        }
        stash_ids {
          endpoint
          stash_id
        }
      }
    }
    """
    data = await self._execute(query, {"id": scene_id})
    return data.get("findScene")
```

**Step 2: Add router endpoints for merge and delete**

Add to `api/recommendations_router.py` after the existing actions (after line 570):

```python
class MergeScenesRequest(BaseModel):
    """Request to merge duplicate scenes."""
    destination_id: str
    source_ids: list[str]


@router.post("/actions/merge-scenes")
async def merge_scenes(request: MergeScenesRequest):
    """Execute a scene merge via Stash's sceneMerge mutation."""
    stash = get_stash_client()
    try:
        result = await stash.merge_scenes(request.source_ids, request.destination_id)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DeleteSceneRequest(BaseModel):
    """Request to delete a scene."""
    scene_id: str
    delete_file: bool = False


@router.post("/actions/delete-scene")
async def delete_scene(request: DeleteSceneRequest):
    """Delete a scene from Stash."""
    stash = get_stash_client()
    try:
        result = await stash.destroy_scene(request.scene_id, delete_file=request.delete_file)
        return {"success": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 3: Commit**

```
git add api/stash_client_unified.py api/recommendations_router.py
git commit -m "feat: add scene merge and delete mutations

Add merge_scenes() using Stash's sceneMerge mutation (consolidates
files, tags, performers). Add destroy_scene() for scene deletion.
Add router endpoints for both actions. Extend get_scene_by_id with
file details and screenshot path for detail view."
```

---

### Task 6: Add Plugin Backend Proxy Functions

**Files:**
- Modify: `plugin/stash_sense_backend.py`

**Step 1: Add proxy functions and dispatch cases**

Add proxy functions near the existing `rec_merge_performers` (around line 431):

```python
def rec_merge_scenes(sidecar_url, destination_id, source_ids):
    """Execute scene merge."""
    data = {
        "destination_id": destination_id,
        "source_ids": source_ids,
    }
    return sidecar_post(sidecar_url, "/recommendations/actions/merge-scenes", data, timeout=120)


def rec_delete_scene(sidecar_url, scene_id, delete_file=False):
    """Delete a scene."""
    data = {
        "scene_id": scene_id,
        "delete_file": delete_file,
    }
    return sidecar_post(sidecar_url, "/recommendations/actions/delete-scene", data, timeout=60)


def rec_get_scene(sidecar_url, scene_id):
    """Get scene details for display."""
    return sidecar_get(sidecar_url, f"/recommendations/scene/{scene_id}")
```

Add dispatch cases in the main dispatch function (near the existing `rec_merge_performers` dispatch around line 541):

```python
elif mode == "rec_merge_scenes":
    destination_id = args.get("destination_id")
    source_ids = args.get("source_ids", [])
    if not destination_id or not source_ids:
        return {"error": "destination_id and source_ids required"}
    return rec_merge_scenes(sidecar_url, destination_id, source_ids)

elif mode == "rec_delete_scene":
    scene_id = args.get("scene_id")
    delete_file = args.get("delete_file", False)
    if not scene_id:
        return {"error": "scene_id required"}
    return rec_delete_scene(sidecar_url, scene_id, delete_file)

elif mode == "rec_get_scene":
    scene_id = args.get("scene_id")
    if not scene_id:
        return {"error": "scene_id required"}
    return rec_get_scene(sidecar_url, scene_id)
```

Also add a scene detail endpoint to the router (`api/recommendations_router.py`):

```python
@router.get("/scene/{scene_id}")
async def get_scene_detail(scene_id: str):
    """Get scene details for the duplicate scenes detail view."""
    stash = get_stash_client()
    try:
        scene = await stash.get_scene_by_id(scene_id)
        if not scene:
            raise HTTPException(status_code=404, detail="Scene not found")
        return scene
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 2: Commit**

```
git add plugin/stash_sense_backend.py api/recommendations_router.py
git commit -m "feat: add plugin backend proxy for scene merge/delete/detail

Wire up rec_merge_scenes, rec_delete_scene, and rec_get_scene
operations through the plugin backend proxy to sidecar API."
```

---

### Task 7: Plugin UI — List Card Renderer

**Files:**
- Modify: `plugin/stash-sense-recommendations.js`

**Step 1: Add `duplicate_scenes` card to `renderRecommendationCard`**

Add a new `if` block for `duplicate_scenes` after the `duplicate_scene_files` handler (after line 1085):

```javascript
if (rec.type === 'duplicate_scenes') {
  const d = details;
  const conf = d.confidence || (rec.confidence * 100);
  const confColor = conf >= 80 ? '#28a745' : conf >= 60 ? '#ffc107' : '#6c757d';
  const sb = d.signal_breakdown || {};
  const primarySignal = sb.stashbox_match ? 'Stash-box match'
    : sb.phash_distance != null && sb.phash_distance <= 10 ? `Phash (dist ${sb.phash_distance})`
    : sb.metadata_score > 0 ? 'Metadata'
    : 'Face analysis';

  return SS.createElement('div', {
    className: 'ss-rec-card ss-rec-dup-scenes',
    innerHTML: `
      <div class="ss-rec-card-header">
        <div class="ss-rec-tag-icon">
          <svg viewBox="0 0 24 24" width="32" height="32" fill="currentColor"><path d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6zm16-4H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H8V4h12v12z"/></svg>
        </div>
        <div class="ss-rec-card-info">
          <div class="ss-rec-card-title">Scene ${rec.target_id} &harr; Scene ${d.scene_b_id}</div>
          <div class="ss-rec-card-subtitle">
            <span style="color: ${confColor}">${Math.round(conf)}% confidence</span>
            &middot; ${primarySignal}
          </div>
        </div>
      </div>
    `,
  });
}
```

**Step 2: Add API methods for scene operations**

Add to the `RecommendationsAPI` object (around line 117):

```javascript
async mergeScenes(destinationId, sourceIds) {
  return apiCall('rec_merge_scenes', {
    destination_id: destinationId,
    source_ids: sourceIds,
  });
},

async deleteScene(sceneId, deleteFile = false) {
  return apiCall('rec_delete_scene', {
    scene_id: sceneId,
    delete_file: deleteFile,
  });
},

async getSceneDetail(sceneId) {
  return apiCall('rec_get_scene', { scene_id: sceneId });
},
```

**Step 3: Commit**

```
git add plugin/stash-sense-recommendations.js
git commit -m "feat: add duplicate scenes list card and API methods

Show scene IDs, confidence percentage, and primary signal type in
the recommendation list card. Add mergeScenes, deleteScene, and
getSceneDetail API methods."
```

---

### Task 8: Plugin UI — Detail View Renderer

**Files:**
- Modify: `plugin/stash-sense-recommendations.js`
- Modify: `plugin/stash-sense.css`

**Step 1: Add detail view dispatch**

In `renderDetail()` (around line 1278), add a case before the fallback:

```javascript
} else if (rec.type === 'duplicate_scenes') {
  await renderDuplicateScenesDetail(content, rec);
} else {
```

**Step 2: Implement `renderDuplicateScenesDetail`**

Add the function after `renderDuplicateSceneFilesDetail` (after its closing brace):

```javascript
async function renderDuplicateScenesDetail(container, rec) {
  const details = rec.details;
  const sceneAId = rec.target_id;
  const sceneBId = String(details.scene_b_id);

  // Show loading state
  container.innerHTML = '<div class="ss-loading">Loading scene details...</div>';

  // Fetch both scenes in parallel
  let sceneA, sceneB;
  try {
    [sceneA, sceneB] = await Promise.all([
      RecommendationsAPI.getSceneDetail(sceneAId),
      RecommendationsAPI.getSceneDetail(sceneBId),
    ]);
  } catch (e) {
    container.innerHTML = `<div class="ss-error-state"><p>Failed to load scenes: ${e.message}</p></div>`;
    return;
  }

  const conf = details.confidence || (rec.confidence * 100);
  const confColor = conf >= 80 ? '#28a745' : conf >= 60 ? '#ffc107' : '#6c757d';
  const reasoning = details.reasoning || [];
  const sb = details.signal_breakdown || {};

  function formatDuration(seconds) {
    if (!seconds) return 'N/A';
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
  }

  function formatFileSize(bytes) {
    if (!bytes) return 'N/A';
    const gb = bytes / (1024 * 1024 * 1024);
    if (gb >= 1) return `${gb.toFixed(1)} GB`;
    return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
  }

  function renderSceneCard(scene, id, label) {
    const file = scene?.files?.[0];
    const resolution = file ? `${file.width}x${file.height}` : 'N/A';
    const screenshotUrl = scene?.paths?.screenshot;

    return `
      <div class="ss-dup-scene-card" data-id="${id}">
        <div class="ss-dup-scene-thumb">
          ${screenshotUrl ? `<img src="${screenshotUrl}" alt="Scene ${id}" loading="lazy" onerror="this.style.display='none'" />` : '<div class="ss-no-image">No Screenshot</div>'}
        </div>
        <h4><a href="/scenes/${id}" target="_blank">${escapeHtml(scene?.title || 'Unknown Scene')}</a></h4>
        <ul class="ss-dup-scene-meta">
          ${scene?.studio?.name ? `<li><strong>Studio:</strong> ${escapeHtml(scene.studio.name)}</li>` : ''}
          ${scene?.performers?.length ? `<li><strong>Performers:</strong> ${scene.performers.map(p => escapeHtml(p.name)).join(', ')}</li>` : ''}
          ${scene?.date ? `<li><strong>Date:</strong> ${scene.date}</li>` : ''}
          <li><strong>Duration:</strong> ${formatDuration(file?.duration)}</li>
          ${file ? `<li><strong>File:</strong> ${resolution} · ${file.video_codec || 'N/A'} · ${formatFileSize(file.size)}</li>` : ''}
        </ul>
        <label class="ss-radio-label">
          <input type="radio" name="keeper" value="${id}" />
          Keep this scene
        </label>
      </div>
    `;
  }

  container.innerHTML = `
    <div class="ss-detail-dup-scenes">
      <h2>Duplicate Scenes</h2>
      <div class="ss-dup-confidence" style="color: ${confColor}">
        ${Math.round(conf)}% confidence &mdash; ${reasoning[0] || ''}
      </div>
      <div class="ss-dup-signals">
        ${reasoning.slice(1).map(r => `<span class="ss-signal-badge">${escapeHtml(r)}</span>`).join('')}
      </div>

      <div class="ss-dup-scenes-grid">
        ${renderSceneCard(sceneA, sceneAId, 'Scene A')}
        <div class="ss-dup-vs">VS</div>
        ${renderSceneCard(sceneB, sceneBId, 'Scene B')}
      </div>

      <div class="ss-detail-actions">
        <button class="ss-btn ss-btn-primary" id="ss-merge-btn">Merge Scenes</button>
        <button class="ss-btn ss-btn-danger" id="ss-delete-a-btn">Delete Scene ${sceneAId}</button>
        <button class="ss-btn ss-btn-danger" id="ss-delete-b-btn">Delete Scene ${sceneBId}</button>
        <button class="ss-btn ss-btn-secondary" id="ss-dismiss-btn">Dismiss</button>
      </div>
    </div>
  `;

  // Default selection: first scene as keeper
  const firstRadio = container.querySelector('input[name="keeper"]');
  if (firstRadio) firstRadio.checked = true;

  // Click card to select radio
  container.querySelectorAll('.ss-dup-scene-card').forEach(card => {
    card.style.cursor = 'pointer';
    card.addEventListener('click', (e) => {
      if (e.target.closest('a')) return;
      const radio = card.querySelector('input[type="radio"]');
      if (radio) radio.checked = true;
    });
  });

  // Merge action
  container.querySelector('#ss-merge-btn').addEventListener('click', async () => {
    const keeperId = container.querySelector('input[name="keeper"]:checked')?.value;
    if (!keeperId) return;

    const sourceId = keeperId === sceneAId ? sceneBId : sceneAId;
    const btn = container.querySelector('#ss-merge-btn');

    showConfirmModal(
      `Merge scene ${sourceId} into scene ${keeperId}? Files, tags, and performers will be consolidated.`,
      async () => {
        try {
          btn.disabled = true;
          btn.textContent = 'Merging...';
          await RecommendationsAPI.mergeScenes(keeperId, [sourceId]);
          await RecommendationsAPI.resolve(rec.id, 'merged', { keeper_id: keeperId, source_id: sourceId });
          showSuccessAndReturn(btn, 'Merged!');
        } catch (e) {
          btn.textContent = `Failed: ${e.message}`;
          btn.classList.add('ss-btn-error');
          btn.disabled = false;
        }
      }
    );
  });

  // Delete A action
  container.querySelector('#ss-delete-a-btn').addEventListener('click', async () => {
    const btn = container.querySelector('#ss-delete-a-btn');
    showConfirmModal(
      `Delete scene ${sceneAId} ("${escapeHtml(sceneA?.title || '')}")? This cannot be undone.`,
      async () => {
        try {
          btn.disabled = true;
          btn.textContent = 'Deleting...';
          await RecommendationsAPI.deleteScene(sceneAId, true);
          await RecommendationsAPI.resolve(rec.id, 'deleted', { deleted_scene_id: sceneAId, kept_scene_id: sceneBId });
          showSuccessAndReturn(btn, 'Deleted!');
        } catch (e) {
          btn.textContent = `Failed: ${e.message}`;
          btn.classList.add('ss-btn-error');
          btn.disabled = false;
        }
      },
      { showDontAsk: true, storageKey: 'delete-dup-scene' }
    );
  });

  // Delete B action
  container.querySelector('#ss-delete-b-btn').addEventListener('click', async () => {
    const btn = container.querySelector('#ss-delete-b-btn');
    showConfirmModal(
      `Delete scene ${sceneBId} ("${escapeHtml(sceneB?.title || '')}")? This cannot be undone.`,
      async () => {
        try {
          btn.disabled = true;
          btn.textContent = 'Deleting...';
          await RecommendationsAPI.deleteScene(sceneBId, true);
          await RecommendationsAPI.resolve(rec.id, 'deleted', { deleted_scene_id: sceneBId, kept_scene_id: sceneAId });
          showSuccessAndReturn(btn, 'Deleted!');
        } catch (e) {
          btn.textContent = `Failed: ${e.message}`;
          btn.classList.add('ss-btn-error');
          btn.disabled = false;
        }
      },
      { showDontAsk: true, storageKey: 'delete-dup-scene' }
    );
  });

  // Dismiss action
  container.querySelector('#ss-dismiss-btn').addEventListener('click', async () => {
    const btn = container.querySelector('#ss-dismiss-btn');
    try {
      btn.disabled = true;
      btn.textContent = 'Dismissing...';
      await RecommendationsAPI.dismiss(rec.id, 'User dismissed');
      currentState.view = 'list';
      currentState.selectedRec = null;
      renderCurrentView(document.getElementById('ss-recommendations'));
    } catch (e) {
      btn.textContent = `Failed: ${e.message}`;
      btn.disabled = false;
    }
  });
}
```

**Step 3: Add CSS styles**

Add to `plugin/stash-sense.css` after the existing `.ss-file-path` styles (around line 1530):

```css
/* Duplicate Scenes */
.ss-dup-scenes-grid {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: 16px;
  align-items: start;
  margin: 20px 0;
}

.ss-dup-vs {
  align-self: center;
  font-size: 18px;
  font-weight: bold;
  color: var(--bs-secondary-color);
  padding: 0 8px;
}

.ss-dup-scene-card {
  background: var(--bs-tertiary-bg);
  border-radius: 8px;
  padding: 16px;
  border: 2px solid transparent;
  transition: border-color 0.2s;
}

.ss-dup-scene-card:hover {
  border-color: var(--bs-secondary-color);
}

.ss-dup-scene-thumb {
  width: 100%;
  aspect-ratio: 16/9;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 12px;
  background: var(--bs-body-bg);
}

.ss-dup-scene-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.ss-dup-scene-card h4 {
  margin: 0 0 8px 0;
  font-size: 14px;
}

.ss-dup-scene-card h4 a {
  color: var(--bs-body-color);
  text-decoration: none;
}

.ss-dup-scene-card h4 a:hover {
  color: var(--bs-primary);
}

.ss-dup-scene-meta {
  list-style: none;
  padding: 0;
  margin: 0 0 12px 0;
  font-size: 13px;
  color: var(--bs-secondary-color);
}

.ss-dup-scene-meta li {
  margin: 4px 0;
}

.ss-dup-confidence {
  font-size: 18px;
  font-weight: bold;
  margin: 8px 0;
}

.ss-dup-signals {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 12px;
}

.ss-signal-badge {
  font-size: 12px;
  padding: 2px 8px;
  border-radius: 12px;
  background: var(--bs-tertiary-bg);
  color: var(--bs-secondary-color);
}
```

**Step 4: Commit**

```
git add plugin/stash-sense-recommendations.js plugin/stash-sense.css
git commit -m "feat: add duplicate scenes detail view with merge/delete/dismiss

Side-by-side scene comparison with screenshots, metadata, and file
info. Keeper selection with merge (via Stash sceneMerge), delete
either scene, dismiss, and open-in-Stash links. Confidence badge
and signal breakdown display."
```

---

### Task 9: Verification and Deploy

**Step 1: Run all tests**

Run: `cd api && ../.venv/bin/python -m pytest tests/ -v`
Expected: All PASS

**Step 2: Start sidecar and verify API**

Run: `cd api && source ../.venv/bin/activate && make sidecar`

Test the duplicate scenes analyzer endpoint manually:
```
curl -X POST http://localhost:5000/recommendations/analysis/run \
  -H "Content-Type: application/json" \
  -d '{"analysis_type": "duplicate_scenes"}'
```

**Step 3: Deploy plugin to Stash**

```
scp plugin/* root@10.0.0.4:/mnt/nvme_cache/appdata/stash/config/plugins/stash-sense/
```

Hard refresh Stash UI (Ctrl+Shift+R). Navigate to Stash Sense → Recommendations → Duplicate Scenes. Verify:
- List shows scene IDs with confidence badges (not raw `DUPLICATE_SCENES` text)
- Clicking a recommendation shows the side-by-side detail view
- Merge, delete, dismiss buttons work

**Step 4: Final commit**

```
git add -A
git commit -m "chore: verification pass — all tests pass, UI renders correctly"
```
