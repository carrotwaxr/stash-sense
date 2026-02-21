# Scene Fingerprint Matching - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automated background job that matches local Stash scenes to stash-box entries via fingerprint (MD5/OSHASH/PHASH) lookups, with quality scoring, pair-based dismissals, and batch accept.

**Architecture:** New `SceneFingerprintMatchAnalyzer` extends `BaseAnalyzer` (not `BaseUpstreamAnalyzer`, since we're finding unlinked scenes rather than diffing linked ones). Quality scoring is a standalone module. Composite `target_id` encodes `{scene_id}|{endpoint}|{stashbox_scene_id}` so the existing dismissal system handles pair-based logic without DB changes.

**Tech Stack:** Python/FastAPI (sidecar), vanilla JS (plugin), SQLite (recommendations DB), GraphQL (Stash + StashBox APIs)

**Design doc:** `docs/plans/2026-02-20-scene-fingerprint-matching-design.md`

---

### Task 1: Quality Scoring Module

Pure functions with no external dependencies. TDD-first.

**Files:**
- Create: `api/scene_fingerprint_scoring.py`
- Create: `api/tests/test_scene_fingerprint_scoring.py`

**Step 1: Write the failing tests**

```python
# api/tests/test_scene_fingerprint_scoring.py
"""Tests for scene fingerprint match quality scoring."""

import pytest
from scene_fingerprint_scoring import score_match, is_high_confidence


class TestScoreMatch:
    """Test quality score computation for a candidate match."""

    def test_single_md5_match(self):
        """MD5 exact match should score high."""
        result = score_match(
            matching_fingerprints=[
                {"algorithm": "MD5", "hash": "abc123", "duration": 1834.5, "submissions": 3}
            ],
            total_local_fingerprints=3,
            local_duration=1835.0,
        )
        assert result["match_count"] == 1
        assert result["match_percentage"] == pytest.approx(33.3, abs=0.1)
        assert result["has_exact_hash"] is True

    def test_multiple_fingerprints_all_match(self):
        """All fingerprints matching = 100% match percentage."""
        result = score_match(
            matching_fingerprints=[
                {"algorithm": "MD5", "hash": "abc", "duration": 1800.0, "submissions": 5},
                {"algorithm": "OSHASH", "hash": "def", "duration": 1800.0, "submissions": 3},
                {"algorithm": "PHASH", "hash": "ghi", "duration": 1800.0, "submissions": 1},
            ],
            total_local_fingerprints=3,
            local_duration=1800.0,
        )
        assert result["match_count"] == 3
        assert result["match_percentage"] == pytest.approx(100.0)
        assert result["has_exact_hash"] is True

    def test_phash_only_match(self):
        """PHASH-only match should flag has_exact_hash=False."""
        result = score_match(
            matching_fingerprints=[
                {"algorithm": "PHASH", "hash": "abc", "duration": 1800.0, "submissions": 1}
            ],
            total_local_fingerprints=2,
            local_duration=1800.0,
        )
        assert result["has_exact_hash"] is False

    def test_duration_agreement(self):
        """Close duration = good agreement, large gap = bad."""
        close = score_match(
            matching_fingerprints=[
                {"algorithm": "MD5", "hash": "a", "duration": 1834.0, "submissions": 1}
            ],
            total_local_fingerprints=1,
            local_duration=1835.0,
        )
        assert close["duration_agreement"] is True

        far = score_match(
            matching_fingerprints=[
                {"algorithm": "MD5", "hash": "a", "duration": 1700.0, "submissions": 1}
            ],
            total_local_fingerprints=1,
            local_duration=1835.0,
        )
        assert far["duration_agreement"] is False

    def test_empty_fingerprints(self):
        """No matching fingerprints should return zero counts."""
        result = score_match(
            matching_fingerprints=[],
            total_local_fingerprints=3,
            local_duration=1800.0,
        )
        assert result["match_count"] == 0
        assert result["match_percentage"] == 0.0


class TestIsHighConfidence:
    """Test high-confidence classification."""

    def test_meets_both_thresholds(self):
        assert is_high_confidence(match_count=2, match_percentage=66.7) is True

    def test_below_count_threshold(self):
        assert is_high_confidence(match_count=1, match_percentage=100.0) is False

    def test_below_percentage_threshold(self):
        assert is_high_confidence(match_count=2, match_percentage=50.0) is False

    def test_custom_thresholds(self):
        assert is_high_confidence(
            match_count=3, match_percentage=80.0,
            min_count=3, min_percentage=80
        ) is True

    def test_custom_thresholds_fail(self):
        assert is_high_confidence(
            match_count=2, match_percentage=80.0,
            min_count=3, min_percentage=80
        ) is False
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -m pytest tests/test_scene_fingerprint_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scene_fingerprint_scoring'`

**Step 3: Write minimal implementation**

```python
# api/scene_fingerprint_scoring.py
"""Quality scoring for scene fingerprint matches.

Scores a candidate match between a local scene and a stash-box scene
based on fingerprint overlap, type, duration agreement, and community votes.
"""

# Duration tolerance in seconds for "agreement"
DURATION_TOLERANCE_SECONDS = 5.0

# Algorithms considered "exact hash" (not perceptual)
EXACT_HASH_ALGORITHMS = {"MD5", "OSHASH"}

# Default high-confidence thresholds
DEFAULT_MIN_COUNT = 2
DEFAULT_MIN_PERCENTAGE = 66


def score_match(
    matching_fingerprints: list[dict],
    total_local_fingerprints: int,
    local_duration: float,
) -> dict:
    """Score a candidate fingerprint match.

    Args:
        matching_fingerprints: List of dicts with keys:
            algorithm (str), hash (str), duration (float), submissions (int)
        total_local_fingerprints: How many fingerprints the local scene has
        local_duration: Local scene file duration in seconds

    Returns:
        Dict with: match_count, match_percentage, has_exact_hash,
        duration_agreement, duration_diff, total_submissions
    """
    match_count = len(matching_fingerprints)

    if total_local_fingerprints > 0:
        match_percentage = (match_count / total_local_fingerprints) * 100
    else:
        match_percentage = 0.0

    has_exact_hash = any(
        fp["algorithm"] in EXACT_HASH_ALGORITHMS
        for fp in matching_fingerprints
    )

    # Duration agreement: compare local vs remote fingerprint durations
    duration_diffs = []
    for fp in matching_fingerprints:
        remote_dur = fp.get("duration")
        if remote_dur is not None and local_duration is not None:
            duration_diffs.append(abs(local_duration - remote_dur))

    if duration_diffs:
        avg_diff = sum(duration_diffs) / len(duration_diffs)
        duration_agreement = avg_diff <= DURATION_TOLERANCE_SECONDS
        duration_diff = round(avg_diff, 2)
    else:
        duration_agreement = True  # No data to disagree
        duration_diff = None

    total_submissions = sum(fp.get("submissions", 0) for fp in matching_fingerprints)

    return {
        "match_count": match_count,
        "match_percentage": round(match_percentage, 1),
        "has_exact_hash": has_exact_hash,
        "duration_agreement": duration_agreement,
        "duration_diff": duration_diff,
        "total_submissions": total_submissions,
    }


def is_high_confidence(
    match_count: int,
    match_percentage: float,
    min_count: int = DEFAULT_MIN_COUNT,
    min_percentage: int = DEFAULT_MIN_PERCENTAGE,
) -> bool:
    """Check if a match qualifies as high-confidence for batch accept.

    Both thresholds must be met.
    """
    return match_count >= min_count and match_percentage >= min_percentage
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -m pytest tests/test_scene_fingerprint_scoring.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add api/scene_fingerprint_scoring.py api/tests/test_scene_fingerprint_scoring.py
git commit -m "feat: add scene fingerprint match quality scoring module (#48)"
```

---

### Task 2: Stash Client - Scene Fingerprint Query

Add a method to query Stash for scenes with their file fingerprint hashes.

**Files:**
- Modify: `api/stash_client_unified.py` (add method after `get_scenes_for_fingerprinting` ~line 815)
- Create: `api/tests/test_scene_fingerprint_query.py`

**Step 1: Write the failing test**

```python
# api/tests/test_scene_fingerprint_query.py
"""Tests for scene fingerprint query in Stash client."""

import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_get_scenes_with_fingerprints_returns_expected_shape():
    """Verify the query method returns scenes with fingerprint data."""
    from stash_client_unified import StashClientUnified

    mock_response = {
        "findScenes": {
            "count": 1,
            "scenes": [
                {
                    "id": "42",
                    "title": "Test Scene",
                    "updated_at": "2026-01-15T00:00:00Z",
                    "files": [
                        {
                            "id": "f1",
                            "duration": 1835.5,
                            "fingerprints": [
                                {"type": "md5", "value": "abc123"},
                                {"type": "oshash", "value": "def456"},
                                {"type": "phash", "value": "0011223344556677"},
                            ],
                        }
                    ],
                    "stash_ids": [
                        {"endpoint": "https://stashdb.org/graphql", "stash_id": "uuid-1"}
                    ],
                }
            ],
        }
    }

    client = StashClientUnified.__new__(StashClientUnified)
    client._execute = AsyncMock(return_value=mock_response)

    scenes, total = await client.get_scenes_with_fingerprints()

    assert total == 1
    assert len(scenes) == 1
    scene = scenes[0]
    assert scene["id"] == "42"
    assert len(scene["files"]) == 1
    assert len(scene["files"][0]["fingerprints"]) == 3
    assert scene["files"][0]["fingerprints"][0]["type"] == "md5"


@pytest.mark.asyncio
async def test_get_scenes_with_fingerprints_incremental():
    """Verify updated_after filter is passed to query."""
    from stash_client_unified import StashClientUnified

    mock_response = {"findScenes": {"count": 0, "scenes": []}}

    client = StashClientUnified.__new__(StashClientUnified)
    client._execute = AsyncMock(return_value=mock_response)

    await client.get_scenes_with_fingerprints(updated_after="2026-01-01T00:00:00Z")

    call_args = client._execute.call_args
    variables = call_args[0][1]
    assert "scene_filter" in variables
    assert "updated_at" in variables["scene_filter"]
    assert variables["scene_filter"]["updated_at"]["value"] == "2026-01-01T00:00:00Z"
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -m pytest tests/test_scene_fingerprint_query.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'get_scenes_with_fingerprints'`

**Step 3: Write minimal implementation**

Add this method to `StashClientUnified` in `api/stash_client_unified.py`, after `get_scenes_for_fingerprinting` (~line 815):

```python
    async def get_scenes_with_fingerprints(
        self,
        updated_after: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """
        Get scenes with file fingerprint hashes for stash-box matching.
        Returns (scenes, total_count).

        Each scene includes files[].fingerprints with type (md5/oshash/phash)
        and value, plus stash_ids showing which endpoints are already linked.
        """
        query = """
        query ScenesWithFingerprints($filter: FindFilterType, $scene_filter: SceneFilterType) {
          findScenes(filter: $filter, scene_filter: $scene_filter) {
            count
            scenes {
              id
              title
              updated_at
              files {
                id
                duration
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
        }
        """
        filter_input = {"per_page": limit, "page": (offset // limit) + 1}
        scene_filter = {}

        if updated_after:
            scene_filter["updated_at"] = {"value": updated_after, "modifier": "GREATER_THAN"}

        data = await self._execute(query, {"filter": filter_input, "scene_filter": scene_filter})
        return data["findScenes"]["scenes"], data["findScenes"]["count"]
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -m pytest tests/test_scene_fingerprint_query.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add api/stash_client_unified.py api/tests/test_scene_fingerprint_query.py
git commit -m "feat: add get_scenes_with_fingerprints query to Stash client (#48)"
```

---

### Task 3: StashBox Client - Batch Fingerprint Lookup

Add a method to query StashBox's `findScenesBySceneFingerprints` batch API.

**Files:**
- Modify: `api/stashbox_client.py` (add method)
- Create: `api/tests/test_stashbox_fingerprint_query.py`

**Step 1: Write the failing test**

```python
# api/tests/test_stashbox_fingerprint_query.py
"""Tests for stash-box batch fingerprint lookup."""

import pytest
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_find_scenes_by_fingerprints_returns_matched_scenes():
    """Verify batch lookup returns per-scene match lists."""
    from stashbox_client import StashBoxClient

    # StashBox returns one list per input fingerprint set
    mock_response = {
        "findScenesBySceneFingerprints": [
            # Matches for scene 1's fingerprints
            [
                {
                    "id": "sb-uuid-1",
                    "title": "Matched Scene",
                    "date": "2024-01-15",
                    "studio": {"id": "st-1", "name": "Studio A"},
                    "performers": [{"performer": {"id": "p-1", "name": "Actor A"}, "as": None}],
                    "urls": [{"url": "https://example.com", "site": {"name": "Example"}}],
                    "fingerprints": [
                        {"hash": "abc123", "algorithm": "MD5", "duration": 1834, "submissions": 5, "created": "2024-01-01", "updated": "2024-06-01"},
                        {"hash": "def456", "algorithm": "OSHASH", "duration": 1834, "submissions": 3, "created": "2024-01-01", "updated": "2024-06-01"},
                    ],
                    "duration": 1834,
                }
            ],
            # No matches for scene 2's fingerprints
            [],
        ]
    }

    client = StashBoxClient.__new__(StashBoxClient)
    client._execute = AsyncMock(return_value=mock_response)

    fingerprint_sets = [
        [{"hash": "abc123", "algorithm": "MD5"}, {"hash": "def456", "algorithm": "OSHASH"}],
        [{"hash": "xyz789", "algorithm": "MD5"}],
    ]

    results = await client.find_scenes_by_fingerprints(fingerprint_sets)

    assert len(results) == 2
    assert len(results[0]) == 1  # One match for scene 1
    assert results[0][0]["id"] == "sb-uuid-1"
    assert len(results[0][0]["fingerprints"]) == 2
    assert len(results[1]) == 0  # No matches for scene 2


@pytest.mark.asyncio
async def test_find_scenes_by_fingerprints_empty_input():
    """Empty input should return empty output without API call."""
    from stashbox_client import StashBoxClient

    client = StashBoxClient.__new__(StashBoxClient)
    client._execute = AsyncMock()

    results = await client.find_scenes_by_fingerprints([])

    assert results == []
    client._execute.assert_not_called()
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -m pytest tests/test_stashbox_fingerprint_query.py -v`
Expected: FAIL with `AttributeError: ... has no attribute 'find_scenes_by_fingerprints'`

**Step 3: Write minimal implementation**

Add to `StashBoxClient` in `api/stashbox_client.py`:

```python
    async def find_scenes_by_fingerprints(
        self, fingerprint_sets: list[list[dict]]
    ) -> list[list[dict]]:
        """Batch lookup scenes by fingerprint sets.

        Uses stash-box's findScenesBySceneFingerprints query which accepts
        multiple fingerprint sets (one per local scene) and returns matched
        stash-box scenes for each.

        Args:
            fingerprint_sets: List of fingerprint lists. Each inner list has
                dicts with keys: hash (str), algorithm (str: MD5/OSHASH/PHASH)

        Returns:
            List of match-lists, one per input fingerprint set. Each match
            includes scene metadata and matched fingerprints.
        """
        if not fingerprint_sets:
            return []

        query = """
        query FindScenesByFingerprints($fingerprints: [[FingerprintQueryInput!]!]!) {
          findScenesBySceneFingerprints(fingerprints: $fingerprints) {
            id
            title
            date
            duration
            studio { id name }
            performers { performer { id name } as }
            urls { url site { name } }
            images { id url }
            fingerprints {
              hash
              algorithm
              duration
              submissions
              created
              updated
            }
          }
        }
        """

        data = await self._execute(
            query, variables={"fingerprints": fingerprint_sets}
        )
        return data.get("findScenesBySceneFingerprints", [])
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -m pytest tests/test_stashbox_fingerprint_query.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add api/stashbox_client.py api/tests/test_stashbox_fingerprint_query.py
git commit -m "feat: add batch fingerprint scene lookup to StashBox client (#48)"
```

---

### Task 4: SceneFingerprintMatchAnalyzer

The core analyzer that orchestrates: query scenes → batch lookup → score → recommend.

**Files:**
- Create: `api/analyzers/scene_fingerprint_match.py`
- Create: `api/tests/test_scene_fingerprint_match_analyzer.py`

**Step 1: Write the failing tests**

```python
# api/tests/test_scene_fingerprint_match_analyzer.py
"""Tests for SceneFingerprintMatchAnalyzer."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from analyzers.scene_fingerprint_match import SceneFingerprintMatchAnalyzer


def make_mock_db():
    """Create a mock RecommendationsDB."""
    db = MagicMock()
    db.is_dismissed.return_value = False
    db.create_recommendation.return_value = 1
    db.get_user_setting.return_value = None
    db.get_watermark.return_value = None
    db.set_watermark.return_value = None
    return db


def make_mock_stash():
    """Create a mock StashClientUnified."""
    stash = AsyncMock()
    stash.get_stashbox_connections.return_value = [
        {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB"}
    ]
    return stash


def make_scene(scene_id, title, fingerprints, stash_ids=None, duration=1800.0, updated_at="2026-01-15T00:00:00Z"):
    """Helper to build a scene dict."""
    return {
        "id": scene_id,
        "title": title,
        "updated_at": updated_at,
        "files": [
            {
                "id": f"file-{scene_id}",
                "duration": duration,
                "fingerprints": fingerprints,
            }
        ],
        "stash_ids": stash_ids or [],
    }


def make_stashbox_match(scene_id, title, fingerprints, studio=None, performers=None, date=None, duration=None):
    """Helper to build a stash-box match result."""
    return {
        "id": scene_id,
        "title": title,
        "date": date,
        "duration": duration,
        "studio": studio,
        "performers": performers or [],
        "urls": [],
        "images": [],
        "fingerprints": fingerprints,
    }


class TestAnalyzerRun:
    """Test the full analyzer run pipeline."""

    @pytest.mark.asyncio
    async def test_creates_recommendation_for_match(self):
        """A scene with matching fingerprints should produce a recommendation."""
        db = make_mock_db()
        stash = make_mock_stash()

        # Local scene with 2 fingerprints, no stash_ids for this endpoint
        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene", [
                {"type": "md5", "value": "abc123"},
                {"type": "oshash", "value": "def456"},
            ])],
            1,
        )

        stashbox_match = make_stashbox_match(
            "sb-uuid-1", "Matched Scene",
            [
                {"hash": "abc123", "algorithm": "MD5", "duration": 1800, "submissions": 5, "created": "2024-01-01", "updated": "2024-06-01"},
                {"hash": "def456", "algorithm": "OSHASH", "duration": 1800, "submissions": 3, "created": "2024-01-01", "updated": "2024-06-01"},
            ],
            studio={"id": "s1", "name": "Studio A"},
            performers=[{"performer": {"id": "p1", "name": "Actor A"}, "as": None}],
            date="2024-01-15",
        )

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC:
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[stashbox_match]]
            MockSBC.return_value = mock_sbc

            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            result = await analyzer.run(incremental=False)

        assert result.recommendations_created == 1
        db.create_recommendation.assert_called_once()
        call_kwargs = db.create_recommendation.call_args[1]
        assert call_kwargs["type"] == "scene_fingerprint_match"
        assert call_kwargs["target_type"] == "scene"
        # Composite target_id: scene_id|endpoint|stashbox_scene_id
        assert "42|" in call_kwargs["target_id"]
        assert "sb-uuid-1" in call_kwargs["target_id"]

    @pytest.mark.asyncio
    async def test_skips_scene_already_linked_to_endpoint(self):
        """Scene with existing stash_id for this endpoint should be skipped."""
        db = make_mock_db()
        stash = make_mock_stash()

        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene",
                [{"type": "md5", "value": "abc123"}],
                stash_ids=[{"endpoint": "https://stashdb.org/graphql", "stash_id": "existing-uuid"}],
            )],
            1,
        )

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC:
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[]]
            MockSBC.return_value = mock_sbc

            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            result = await analyzer.run(incremental=False)

        assert result.recommendations_created == 0
        # Should not even call fingerprint lookup for already-linked scenes
        mock_sbc.find_scenes_by_fingerprints.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_dismissed_pair(self):
        """Dismissed (scene, stashbox_scene) pair should be skipped."""
        db = make_mock_db()
        # This specific pair is dismissed
        db.is_dismissed.return_value = True
        stash = make_mock_stash()

        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene", [{"type": "md5", "value": "abc123"}])],
            1,
        )

        stashbox_match = make_stashbox_match(
            "sb-uuid-1", "Matched Scene",
            [{"hash": "abc123", "algorithm": "MD5", "duration": 1800, "submissions": 5, "created": "2024-01-01", "updated": "2024-06-01"}],
        )

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC:
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[stashbox_match]]
            MockSBC.return_value = mock_sbc

            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            result = await analyzer.run(incremental=False)

        assert result.recommendations_created == 0
        db.create_recommendation.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_stashbox_connections_returns_zero(self):
        """If no stash-box endpoints configured, analyzer exits cleanly."""
        db = make_mock_db()
        stash = make_mock_stash()
        stash.get_stashbox_connections.return_value = []

        analyzer = SceneFingerprintMatchAnalyzer(stash, db)
        result = await analyzer.run()

        assert result.items_processed == 0
        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_ambiguous_matches_not_high_confidence(self):
        """Multiple matches for same scene from same endpoint = not high confidence."""
        db = make_mock_db()
        stash = make_mock_stash()

        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene", [
                {"type": "md5", "value": "abc123"},
                {"type": "oshash", "value": "def456"},
            ])],
            1,
        )

        match_a = make_stashbox_match("sb-1", "Match A", [
            {"hash": "abc123", "algorithm": "MD5", "duration": 1800, "submissions": 3, "created": "2024-01-01", "updated": "2024-06-01"},
            {"hash": "def456", "algorithm": "OSHASH", "duration": 1800, "submissions": 2, "created": "2024-01-01", "updated": "2024-06-01"},
        ])
        match_b = make_stashbox_match("sb-2", "Match B", [
            {"hash": "abc123", "algorithm": "MD5", "duration": 1800, "submissions": 1, "created": "2024-01-01", "updated": "2024-06-01"},
        ])

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC:
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[match_a, match_b]]
            MockSBC.return_value = mock_sbc

            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            result = await analyzer.run(incremental=False)

        assert result.recommendations_created == 2
        # Both recs created, but check high_confidence is False for all
        for call in db.create_recommendation.call_args_list:
            details = call[1]["details"]
            assert details["high_confidence"] is False
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -m pytest tests/test_scene_fingerprint_match_analyzer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'analyzers.scene_fingerprint_match'`

**Step 3: Write the implementation**

```python
# api/analyzers/scene_fingerprint_match.py
"""Analyzer: match local scenes to stash-box entries via fingerprints.

Extends BaseAnalyzer (not BaseUpstreamAnalyzer) because this finds
unlinked scenes rather than diffing already-linked entities.
"""

import logging
from typing import Optional

from .base import BaseAnalyzer, AnalysisResult
from scene_fingerprint_scoring import score_match, is_high_confidence
from stashbox_client import StashBoxClient

logger = logging.getLogger(__name__)

# Max scenes per stash-box batch query (matches Stash tagger batching)
BATCH_SIZE = 40


class SceneFingerprintMatchAnalyzer(BaseAnalyzer):
    type = "scene_fingerprint_match"

    async def run(self, incremental: bool = True) -> AnalysisResult:
        connections = await self.stash.get_stashbox_connections()
        if not connections:
            return AnalysisResult(items_processed=0, recommendations_created=0)

        # Load user-configurable thresholds
        min_count = self._get_setting("scene_fp_min_count", 2)
        min_percentage = self._get_setting("scene_fp_min_percentage", 66)

        total_processed = 0
        total_created = 0

        for conn in connections:
            endpoint = conn["endpoint"]
            api_key = conn.get("api_key", "")
            endpoint_name = conn.get("name", endpoint)

            processed, created = await self._process_endpoint(
                endpoint, api_key, endpoint_name,
                incremental, min_count, min_percentage,
            )
            total_processed += processed
            total_created += created

        return AnalysisResult(
            items_processed=total_processed,
            recommendations_created=total_created,
        )

    async def _process_endpoint(
        self,
        endpoint: str,
        api_key: str,
        endpoint_name: str,
        incremental: bool,
        min_count: int,
        min_percentage: int,
    ) -> tuple[int, int]:
        """Process one stash-box endpoint. Returns (processed, created)."""
        watermark_key = f"scene_fp_match_{endpoint}"
        watermark = self.rec_db.get_watermark(watermark_key) if incremental else None

        # Fetch all local scenes with fingerprint data
        scenes_needing_match = []
        offset = 0
        latest_updated = watermark

        while True:
            scenes, total = await self.stash.get_scenes_with_fingerprints(
                updated_after=watermark, limit=100, offset=offset,
            )
            if not scenes:
                break

            for scene in scenes:
                # Track latest updated_at for watermark
                updated_at = scene.get("updated_at")
                if updated_at and (latest_updated is None or updated_at > latest_updated):
                    latest_updated = updated_at

                # Skip scenes already linked to this endpoint
                linked_endpoints = {
                    sid["endpoint"] for sid in (scene.get("stash_ids") or [])
                }
                if endpoint in linked_endpoints:
                    continue

                # Collect fingerprints from all files
                fingerprints = []
                duration = None
                for f in scene.get("files") or []:
                    if duration is None and f.get("duration"):
                        duration = f["duration"]
                    for fp in f.get("fingerprints") or []:
                        fingerprints.append({
                            "hash": fp["value"],
                            "algorithm": fp["type"].upper(),
                        })

                if fingerprints:
                    scenes_needing_match.append({
                        "scene": scene,
                        "fingerprints": fingerprints,
                        "duration": duration,
                    })

            offset += len(scenes)
            if offset >= total:
                break

        if not scenes_needing_match:
            if latest_updated:
                self.rec_db.set_watermark(watermark_key, latest_updated)
            return 0, 0

        # Batch query stash-box
        stashbox = StashBoxClient(endpoint, api_key)
        created = 0

        for batch_start in range(0, len(scenes_needing_match), BATCH_SIZE):
            batch = scenes_needing_match[batch_start:batch_start + BATCH_SIZE]
            fp_sets = [item["fingerprints"] for item in batch]

            results = await stashbox.find_scenes_by_fingerprints(fp_sets)

            for i, matches in enumerate(results):
                item = batch[i]
                scene = item["scene"]
                local_fps = item["fingerprints"]
                local_duration = item["duration"]
                is_ambiguous = len(matches) > 1

                for match in matches:
                    # Build composite target_id for pair-based dismissal
                    target_id = f"{scene['id']}|{endpoint}|{match['id']}"

                    if self.is_dismissed("scene", target_id):
                        continue

                    # Find which local fingerprints matched this stash-box scene
                    local_hashes = {fp["hash"] for fp in local_fps}
                    matching_fps = [
                        fp for fp in match.get("fingerprints", [])
                        if fp["hash"] in local_hashes
                    ]

                    score_result = score_match(
                        matching_fingerprints=matching_fps,
                        total_local_fingerprints=len(local_fps),
                        local_duration=local_duration or 0,
                    )

                    high_conf = (
                        not is_ambiguous
                        and is_high_confidence(
                            score_result["match_count"],
                            score_result["match_percentage"],
                            min_count=min_count,
                            min_percentage=min_percentage,
                        )
                    )

                    performers = [
                        p["performer"]["name"]
                        for p in (match.get("performers") or [])
                        if p.get("performer")
                    ]
                    studio = match.get("studio")
                    images = match.get("images") or []

                    details = {
                        "local_scene_id": scene["id"],
                        "local_scene_title": scene.get("title") or f"Scene {scene['id']}",
                        "endpoint": endpoint,
                        "endpoint_name": endpoint_name,
                        "stashbox_scene_id": match["id"],
                        "stashbox_scene_title": match.get("title"),
                        "stashbox_studio": studio.get("name") if studio else None,
                        "stashbox_performers": performers,
                        "stashbox_date": match.get("date"),
                        "stashbox_cover_url": images[0]["url"] if images else None,
                        "matching_fingerprints": matching_fps,
                        "total_local_fingerprints": len(local_fps),
                        "match_count": score_result["match_count"],
                        "match_percentage": score_result["match_percentage"],
                        "has_exact_hash": score_result["has_exact_hash"],
                        "duration_local": local_duration,
                        "duration_remote": match.get("duration"),
                        "duration_agreement": score_result["duration_agreement"],
                        "duration_diff": score_result["duration_diff"],
                        "total_submissions": score_result["total_submissions"],
                        "high_confidence": high_conf,
                    }

                    confidence = score_result["match_percentage"] / 100.0

                    rec_id = self.create_recommendation(
                        target_type="scene",
                        target_id=target_id,
                        details=details,
                        confidence=confidence,
                    )
                    if rec_id:
                        created += 1

        processed = len(scenes_needing_match)
        self.update_progress(processed, created)

        if latest_updated:
            self.rec_db.set_watermark(watermark_key, latest_updated)

        return processed, created

    def _get_setting(self, key: str, default):
        """Read a user setting with fallback."""
        val = self.rec_db.get_user_setting(key)
        if val is None:
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -m pytest tests/test_scene_fingerprint_match_analyzer.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add api/analyzers/scene_fingerprint_match.py api/tests/test_scene_fingerprint_match_analyzer.py
git commit -m "feat: implement SceneFingerprintMatchAnalyzer (#48)"
```

---

### Task 5: Job Registration and Analyzer Registry

Wire the analyzer into the job queue and analyzer registry.

**Files:**
- Modify: `api/job_models.py` (add registration)
- Modify: `api/recommendations_router.py` (add to ANALYZERS dict)

**Step 1: Add job registration**

Add to `api/job_models.py` after the existing `_register` calls (after the `upstream_scene_changes` block):

```python
_register(
    "scene_fingerprint_match",
    "Scene Fingerprint Matching",
    "Match local scenes to stash-box entries via fingerprints",
    ResourceType.NETWORK,
    JobPriority.NORMAL,
    supports_incremental=True,
    schedulable=True,
    default_interval_hours=168,
    allowed_intervals=(
        (24, "Daily"),
        (72, "Every 3 days"),
        (168, "Weekly"),
        (336, "Every 2 weeks"),
    ),
)
```

**Step 2: Add to analyzer registry**

In `api/recommendations_router.py`, add the import at the top with the other analyzer imports:

```python
from analyzers.scene_fingerprint_match import SceneFingerprintMatchAnalyzer
```

Add to the `ANALYZERS` dict:

```python
"scene_fingerprint_match": SceneFingerprintMatchAnalyzer,
```

**Step 3: Verify sidecar starts**

Run: `cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -c "from job_models import JOB_REGISTRY; assert 'scene_fingerprint_match' in JOB_REGISTRY; print('OK')"`
Expected: `OK`

Run: `cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -c "from recommendations_router import ANALYZERS; assert 'scene_fingerprint_match' in ANALYZERS; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add api/job_models.py api/recommendations_router.py
git commit -m "feat: register scene fingerprint match job and analyzer (#48)"
```

---

### Task 6: Router - Accept Fingerprint Match Endpoint

Add API endpoint to accept a single fingerprint match (adds stash_id to local scene).

**Files:**
- Modify: `api/recommendations_router.py` (add endpoint + request model)
- Add test to: `api/tests/test_scene_fingerprint_match_analyzer.py`

**Step 1: Write the failing test**

Add to `api/tests/test_scene_fingerprint_match_analyzer.py`:

```python
class TestAcceptAction:
    """Test the accept fingerprint match action logic."""

    @pytest.mark.asyncio
    async def test_accept_adds_stash_id_to_scene(self):
        """Accepting a match should add the stash_id to the local scene."""
        from recommendations_router import _accept_fingerprint_match

        mock_stash = AsyncMock()
        mock_stash.get_scene_by_id.return_value = {
            "id": "42",
            "stash_ids": [
                {"endpoint": "https://other.org/graphql", "stash_id": "other-uuid"},
            ],
        }
        mock_stash.update_scene.return_value = {"id": "42"}

        mock_db = MagicMock()
        mock_db.get_recommendation.return_value = MagicMock(
            id=1, type="scene_fingerprint_match", status="pending"
        )
        mock_db.resolve_recommendation.return_value = True

        await _accept_fingerprint_match(
            stash=mock_stash,
            db=mock_db,
            rec_id=1,
            scene_id="42",
            endpoint="https://stashdb.org/graphql",
            stash_id="sb-uuid-1",
        )

        # Verify stash_ids includes both old and new
        call_kwargs = mock_stash.update_scene.call_args[1]
        stash_ids = call_kwargs["stash_ids"]
        assert len(stash_ids) == 2
        assert {"endpoint": "https://stashdb.org/graphql", "stash_id": "sb-uuid-1"} in stash_ids

        # Verify recommendation was resolved
        mock_db.resolve_recommendation.assert_called_once_with(1, action="accepted")
```

**Step 2: Run test to verify it fails**

Run: `cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -m pytest tests/test_scene_fingerprint_match_analyzer.py::TestAcceptAction -v`
Expected: FAIL with `ImportError: cannot import name '_accept_fingerprint_match'`

**Step 3: Write the implementation**

Add to `api/recommendations_router.py`:

```python
class AcceptFingerprintMatchRequest(BaseModel):
    recommendation_id: int
    scene_id: str
    endpoint: str
    stash_id: str


async def _accept_fingerprint_match(
    stash, db, rec_id: int, scene_id: str, endpoint: str, stash_id: str,
):
    """Accept a fingerprint match: add stash_id to local scene, resolve rec."""
    # Get current scene stash_ids
    scene = await stash.get_scene_by_id(scene_id)
    existing_stash_ids = scene.get("stash_ids") or []

    # Append new stash_id (avoid duplicates)
    new_entry = {"endpoint": endpoint, "stash_id": stash_id}
    already_linked = any(
        s["endpoint"] == endpoint and s["stash_id"] == stash_id
        for s in existing_stash_ids
    )
    if not already_linked:
        updated_stash_ids = existing_stash_ids + [new_entry]
        await stash.update_scene(scene_id, stash_ids=updated_stash_ids)

    # Resolve the recommendation
    db.resolve_recommendation(rec_id, action="accepted")


@router.post("/actions/accept-fingerprint-match")
async def accept_fingerprint_match(request: AcceptFingerprintMatchRequest):
    stash = get_stash_client()
    db = get_rec_db()
    await _accept_fingerprint_match(
        stash, db,
        rec_id=request.recommendation_id,
        scene_id=request.scene_id,
        endpoint=request.endpoint,
        stash_id=request.stash_id,
    )
    return {"success": True}
```

**Step 4: Run test to verify it passes**

Run: `cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -m pytest tests/test_scene_fingerprint_match_analyzer.py::TestAcceptAction -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/recommendations_router.py api/tests/test_scene_fingerprint_match_analyzer.py
git commit -m "feat: add accept-fingerprint-match API endpoint (#48)"
```

---

### Task 7: Router - Accept All Fingerprint Matches Endpoint

Batch accept all high-confidence matches, optionally filtered by endpoint.

**Files:**
- Modify: `api/recommendations_router.py` (add endpoint)
- Add test to: `api/tests/test_scene_fingerprint_match_analyzer.py`

**Step 1: Write the failing test**

Add to `api/tests/test_scene_fingerprint_match_analyzer.py`:

```python
class TestAcceptAllAction:
    """Test the accept-all fingerprint matches action."""

    @pytest.mark.asyncio
    async def test_accepts_only_high_confidence(self):
        """Only high_confidence recommendations should be accepted."""
        from recommendations_router import _accept_all_fingerprint_matches
        from recommendations_db import Recommendation

        high_conf_rec = Recommendation(
            id=1, type="scene_fingerprint_match", status="pending",
            target_type="scene", target_id="42|https://stashdb.org/graphql|sb-1",
            details={
                "local_scene_id": "42",
                "endpoint": "https://stashdb.org/graphql",
                "stashbox_scene_id": "sb-1",
                "high_confidence": True,
            },
            resolution_action=None, resolution_details=None, resolved_at=None,
            confidence=0.67, source_analysis_id=None,
            created_at="2026-01-01", updated_at="2026-01-01",
        )
        low_conf_rec = Recommendation(
            id=2, type="scene_fingerprint_match", status="pending",
            target_type="scene", target_id="43|https://stashdb.org/graphql|sb-2",
            details={
                "local_scene_id": "43",
                "endpoint": "https://stashdb.org/graphql",
                "stashbox_scene_id": "sb-2",
                "high_confidence": False,
            },
            resolution_action=None, resolution_details=None, resolved_at=None,
            confidence=0.33, source_analysis_id=None,
            created_at="2026-01-01", updated_at="2026-01-01",
        )

        mock_db = MagicMock()
        mock_db.get_recommendations.return_value = [high_conf_rec, low_conf_rec]
        mock_db.resolve_recommendation.return_value = True

        mock_stash = AsyncMock()
        mock_stash.get_scene_by_id.return_value = {"id": "42", "stash_ids": []}
        mock_stash.update_scene.return_value = {"id": "42"}

        accepted = await _accept_all_fingerprint_matches(mock_stash, mock_db)

        assert accepted == 1
        mock_db.resolve_recommendation.assert_called_once_with(1, action="accepted")
```

**Step 2: Run test to verify it fails**

Run: `cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -m pytest tests/test_scene_fingerprint_match_analyzer.py::TestAcceptAllAction -v`
Expected: FAIL with `ImportError: cannot import name '_accept_all_fingerprint_matches'`

**Step 3: Write the implementation**

Add to `api/recommendations_router.py`:

```python
class AcceptAllFingerprintMatchesRequest(BaseModel):
    endpoint: Optional[str] = None


async def _accept_all_fingerprint_matches(
    stash, db, endpoint: Optional[str] = None,
) -> int:
    """Accept all high-confidence fingerprint matches. Returns count accepted."""
    recs = db.get_recommendations(
        status="pending", type="scene_fingerprint_match", limit=10000,
    )

    accepted = 0
    for rec in recs:
        details = rec.details or {}
        if not details.get("high_confidence"):
            continue
        if endpoint and details.get("endpoint") != endpoint:
            continue

        try:
            await _accept_fingerprint_match(
                stash, db,
                rec_id=rec.id,
                scene_id=details["local_scene_id"],
                endpoint=details["endpoint"],
                stash_id=details["stashbox_scene_id"],
            )
            accepted += 1
        except Exception as e:
            logger.warning("Failed to accept rec %s: %s", rec.id, e)

    return accepted


@router.post("/actions/accept-all-fingerprint-matches")
async def accept_all_fingerprint_matches(
    request: AcceptAllFingerprintMatchesRequest = AcceptAllFingerprintMatchesRequest(),
):
    stash = get_stash_client()
    db = get_rec_db()
    accepted = await _accept_all_fingerprint_matches(stash, db, request.endpoint)
    return {"success": True, "accepted_count": accepted}
```

**Step 4: Run test to verify it passes**

Run: `cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -m pytest tests/test_scene_fingerprint_match_analyzer.py::TestAcceptAllAction -v`
Expected: PASS

**Step 5: Run all tests**

Run: `cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -m pytest tests/test_scene_fingerprint_match_analyzer.py tests/test_scene_fingerprint_scoring.py tests/test_scene_fingerprint_query.py tests/test_stashbox_fingerprint_query.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add api/recommendations_router.py api/tests/test_scene_fingerprint_match_analyzer.py
git commit -m "feat: add accept-all-fingerprint-matches API endpoint (#48)"
```

---

### Task 8: Plugin Backend Proxy Modes

Add proxy modes so the plugin JS can call the new sidecar endpoints.

**Files:**
- Modify: `plugin/stash_sense_backend.py`

**Step 1: Add proxy handlers**

In `plugin/stash_sense_backend.py`, add handlers within the recommendations section (where other `rec_*` modes are handled). Find the pattern for existing `rec_` modes and add:

```python
    elif mode == "rec_accept_fingerprint_match":
        scene_id = input.get("scene_id")
        recommendation_id = input.get("recommendation_id")
        endpoint = input.get("endpoint")
        stash_id = input.get("stash_id")
        return sidecar_post(
            base_url,
            "/recommendations/actions/accept-fingerprint-match",
            json={
                "recommendation_id": recommendation_id,
                "scene_id": scene_id,
                "endpoint": endpoint,
                "stash_id": stash_id,
            },
        )

    elif mode == "rec_accept_all_fingerprint_matches":
        endpoint = input.get("endpoint")
        payload = {}
        if endpoint:
            payload["endpoint"] = endpoint
        return sidecar_post(
            base_url,
            "/recommendations/actions/accept-all-fingerprint-matches",
            json=payload,
        )
```

**Step 2: Test manually**

Deploy plugin to Stash and verify modes are reachable (or defer to integration testing after UI is built).

**Step 3: Commit**

```bash
git add plugin/stash_sense_backend.py
git commit -m "feat: add plugin proxy modes for fingerprint match actions (#48)"
```

---

### Task 9: Plugin UI - Dashboard Card

Add the fingerprint match card to the recommendations dashboard.

**Files:**
- Modify: `plugin/stash-sense-recommendations.js` (API methods + dashboard card)
- Modify: `plugin/stash-sense.css` (fingerprint card styling)

**Step 1: Add API methods**

Add to the `RecommendationsAPI` object in `stash-sense-recommendations.js`:

```javascript
acceptFingerprintMatch: async function(recommendationId, sceneId, endpoint, stashId) {
    return apiCall('rec_accept_fingerprint_match', {
        recommendation_id: recommendationId,
        scene_id: sceneId,
        endpoint: endpoint,
        stash_id: stashId,
    });
},

acceptAllFingerprintMatches: async function(endpoint) {
    return apiCall('rec_accept_all_fingerprint_matches', {
        endpoint: endpoint || null,
    });
},
```

**Step 2: Add dashboard card rendering**

Find the dashboard card rendering section (where other recommendation types are rendered as cards). Add a case for `scene_fingerprint_match`:

```javascript
// Inside the dashboard card rendering logic, add:
if (counts.scene_fingerprint_match) {
    const fpCount = counts.scene_fingerprint_match;
    const pending = fpCount.pending || 0;

    // Count high-confidence from the pending recommendations
    const card = document.createElement('div');
    card.className = 'ss-rec-card ss-rec-card-fingerprint';
    card.innerHTML = `
        <div class="ss-rec-card-header">
            <span class="ss-rec-card-icon">&#128279;</span>
            <span class="ss-rec-card-title">Scene Fingerprint Matches</span>
        </div>
        <div class="ss-rec-card-body">
            <span class="ss-rec-card-count">${pending}</span>
            <span class="ss-rec-card-label">pending matches</span>
        </div>
    `;
    card.addEventListener('click', () => {
        currentState.view = 'list';
        currentState.type = 'scene_fingerprint_match';
        currentState.status = 'pending';
        currentState.page = 1;
        renderCurrentView(container);
    });
    cardsContainer.appendChild(card);
}
```

**Step 3: Commit**

```bash
git add plugin/stash-sense-recommendations.js plugin/stash-sense.css
git commit -m "feat: add fingerprint match dashboard card and API methods (#48)"
```

---

### Task 10: Plugin UI - List View

Add card rendering for fingerprint match recommendations in the list view.

**Files:**
- Modify: `plugin/stash-sense-recommendations.js`

**Step 1: Add list card renderer**

Find `renderRecommendationCard` and add a case for `scene_fingerprint_match`:

```javascript
// Inside renderRecommendationCard, add case:
if (rec.type === 'scene_fingerprint_match') {
    const d = rec.details;
    const matchColor = d.match_percentage >= 100 ? '#28a745' :
                       d.match_percentage >= 66 ? '#ffc107' : '#dc3545';

    card.innerHTML = `
        <div class="ss-rec-card-header">
            <div class="ss-rec-card-title-row">
                <span class="ss-rec-card-title">${escapeHtml(d.local_scene_title)}</span>
                ${d.high_confidence ? '<span class="ss-badge ss-badge-success">High Confidence</span>' : ''}
            </div>
            <span class="ss-rec-card-subtitle">${escapeHtml(d.endpoint_name || d.endpoint)}</span>
        </div>
        <div class="ss-fp-match-summary">
            <div class="ss-fp-match-arrow">
                <span class="ss-fp-local">${escapeHtml(d.local_scene_title)}</span>
                <span class="ss-fp-arrow">&rarr;</span>
                <span class="ss-fp-remote">${escapeHtml(d.stashbox_scene_title || 'Unknown')}</span>
            </div>
            <div class="ss-fp-match-meta">
                ${d.stashbox_studio ? `<span class="ss-fp-studio">${escapeHtml(d.stashbox_studio)}</span>` : ''}
                ${d.stashbox_performers?.length ? `<span class="ss-fp-performers">${d.stashbox_performers.map(escapeHtml).join(', ')}</span>` : ''}
                ${d.stashbox_date ? `<span class="ss-fp-date">${d.stashbox_date}</span>` : ''}
            </div>
        </div>
        <div class="ss-fp-match-badge" style="color: ${matchColor}">
            ${d.match_count}/${d.total_local_fingerprints} fingerprints
            ${d.has_exact_hash ? '(exact)' : '(phash)'}
        </div>
    `;
}
```

**Step 2: Add Accept All button for fingerprint matches**

In the list view header area (where "Accept All" exists for upstream types), add support for `scene_fingerprint_match`:

```javascript
// In the list view header, if type is scene_fingerprint_match and status is pending:
if (currentState.type === 'scene_fingerprint_match' && currentState.status === 'pending') {
    const acceptAllBtn = document.createElement('button');
    acceptAllBtn.className = 'ss-btn ss-btn-primary ss-btn-sm';
    acceptAllBtn.textContent = 'Accept All High-Confidence';
    acceptAllBtn.addEventListener('click', async () => {
        acceptAllBtn.disabled = true;
        acceptAllBtn.textContent = 'Accepting...';
        try {
            const result = await RecommendationsAPI.acceptAllFingerprintMatches();
            acceptAllBtn.textContent = `Accepted ${result.accepted_count}!`;
            acceptAllBtn.classList.add('ss-btn-success');
            setTimeout(() => renderCurrentView(container), 1500);
        } catch (e) {
            acceptAllBtn.textContent = `Failed: ${e.message}`;
            acceptAllBtn.classList.add('ss-btn-error');
            acceptAllBtn.disabled = false;
        }
    });
    headerRight.appendChild(acceptAllBtn);
}
```

**Step 3: Commit**

```bash
git add plugin/stash-sense-recommendations.js
git commit -m "feat: add fingerprint match list view with Accept All (#48)"
```

---

### Task 11: Plugin UI - Detail View

Add detailed view for a single fingerprint match recommendation.

**Files:**
- Modify: `plugin/stash-sense-recommendations.js`
- Modify: `plugin/stash-sense.css`

**Step 1: Add detail renderer**

Find the detail view type dispatch (where `renderDuplicatePerformerDetail`, etc. are called) and add:

```javascript
} else if (rec.type === 'scene_fingerprint_match') {
    renderFingerprintMatchDetail(content, rec);
}
```

Then add the renderer function:

```javascript
function renderFingerprintMatchDetail(content, rec) {
    const d = rec.details;
    const isPending = rec.status === 'pending';

    content.innerHTML = `
        <div class="ss-fp-detail">
            <div class="ss-fp-detail-header">
                <h3>Scene Fingerprint Match</h3>
                <span class="ss-badge ${d.high_confidence ? 'ss-badge-success' : 'ss-badge-warning'}">
                    ${d.high_confidence ? 'High Confidence' : 'Review Recommended'}
                </span>
            </div>

            <div class="ss-fp-detail-comparison">
                <div class="ss-fp-detail-side">
                    <h4>Local Scene</h4>
                    <div class="ss-fp-field"><strong>Title:</strong> ${escapeHtml(d.local_scene_title)}</div>
                    <div class="ss-fp-field"><strong>Duration:</strong> ${d.duration_local ? formatDuration(d.duration_local) : 'N/A'}</div>
                    <div class="ss-fp-field"><strong>Fingerprints:</strong> ${d.total_local_fingerprints}</div>
                </div>
                <div class="ss-fp-detail-divider"></div>
                <div class="ss-fp-detail-side">
                    <h4>Stash-Box Match</h4>
                    <div class="ss-fp-field"><strong>Title:</strong> ${escapeHtml(d.stashbox_scene_title || 'Unknown')}</div>
                    ${d.stashbox_studio ? `<div class="ss-fp-field"><strong>Studio:</strong> ${escapeHtml(d.stashbox_studio)}</div>` : ''}
                    ${d.stashbox_performers?.length ? `<div class="ss-fp-field"><strong>Performers:</strong> ${d.stashbox_performers.map(escapeHtml).join(', ')}</div>` : ''}
                    ${d.stashbox_date ? `<div class="ss-fp-field"><strong>Date:</strong> ${d.stashbox_date}</div>` : ''}
                    <div class="ss-fp-field"><strong>Duration:</strong> ${d.duration_remote ? formatDuration(d.duration_remote) : 'N/A'}</div>
                    <div class="ss-fp-field"><strong>Endpoint:</strong> ${escapeHtml(d.endpoint_name || d.endpoint)}</div>
                </div>
            </div>

            <div class="ss-fp-detail-fingerprints">
                <h4>Fingerprint Comparison</h4>
                <div class="ss-fp-table-wrap">
                    <table class="ss-fp-table">
                        <thead>
                            <tr>
                                <th>Algorithm</th>
                                <th>Hash</th>
                                <th>Duration</th>
                                <th>Submissions</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${(d.matching_fingerprints || []).map(fp => `
                                <tr>
                                    <td><span class="ss-badge ss-badge-${fp.algorithm === 'PHASH' ? 'warning' : 'success'}">${fp.algorithm}</span></td>
                                    <td class="ss-fp-hash">${fp.hash}</td>
                                    <td>${fp.duration ? formatDuration(fp.duration) : 'N/A'}</td>
                                    <td>${fp.submissions || 0}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
                <div class="ss-fp-summary-line">
                    <strong>${d.match_count}/${d.total_local_fingerprints}</strong> fingerprints match
                    (${d.match_percentage}%)
                    ${d.duration_agreement ? '&mdash; duration agrees' : '&mdash; <span class="ss-text-warning">duration mismatch</span>'}
                </div>
            </div>

            ${isPending ? `
                <div class="ss-fp-detail-actions">
                    <button id="ss-fp-accept-btn" class="ss-btn ss-btn-primary">Accept Match</button>
                    <button id="ss-fp-dismiss-btn" class="ss-btn ss-btn-secondary">Dismiss</button>
                </div>
            ` : `
                <div class="ss-fp-detail-status">
                    Status: <strong>${rec.status}</strong>
                    ${rec.resolution_action ? ` (${rec.resolution_action})` : ''}
                </div>
            `}
        </div>
    `;

    if (!isPending) return;

    // Accept button
    const acceptBtn = content.querySelector('#ss-fp-accept-btn');
    acceptBtn.addEventListener('click', async () => {
        acceptBtn.disabled = true;
        acceptBtn.textContent = 'Accepting...';
        try {
            await RecommendationsAPI.acceptFingerprintMatch(
                rec.id, d.local_scene_id, d.endpoint, d.stashbox_scene_id
            );
            acceptBtn.textContent = 'Accepted!';
            acceptBtn.classList.add('ss-btn-success');
            setTimeout(() => {
                currentState.view = 'list';
                renderCurrentView(document.getElementById('ss-recommendations'));
            }, 1500);
        } catch (e) {
            acceptBtn.textContent = `Failed: ${e.message}`;
            acceptBtn.classList.add('ss-btn-error');
            acceptBtn.disabled = false;
        }
    });

    // Dismiss button
    const dismissBtn = content.querySelector('#ss-fp-dismiss-btn');
    dismissBtn.addEventListener('click', async () => {
        dismissBtn.disabled = true;
        dismissBtn.textContent = 'Dismissing...';
        try {
            await RecommendationsAPI.dismiss(rec.id);
            dismissBtn.textContent = 'Dismissed!';
            dismissBtn.classList.add('ss-btn-success');
            setTimeout(() => {
                currentState.view = 'list';
                renderCurrentView(document.getElementById('ss-recommendations'));
            }, 1500);
        } catch (e) {
            dismissBtn.textContent = `Failed: ${e.message}`;
            dismissBtn.classList.add('ss-btn-error');
            dismissBtn.disabled = false;
        }
    });
}

function formatDuration(seconds) {
    if (!seconds && seconds !== 0) return 'N/A';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}
```

**Step 2: Add CSS**

Add to `plugin/stash-sense.css`:

```css
/* Scene Fingerprint Match Detail */
.ss-fp-detail-comparison {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}

.ss-fp-detail-side {
    flex: 1;
    padding: 1rem;
    background: var(--bs-body-bg);
    border-radius: 0.5rem;
}

.ss-fp-detail-divider {
    width: 2px;
    background: var(--bs-border-color);
}

.ss-fp-field {
    margin: 0.4rem 0;
    font-size: 0.9rem;
}

.ss-fp-table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.5rem 0;
}

.ss-fp-table th,
.ss-fp-table td {
    padding: 0.5rem;
    text-align: left;
    border-bottom: 1px solid var(--bs-border-color);
}

.ss-fp-hash {
    font-family: monospace;
    font-size: 0.8rem;
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
}

.ss-fp-summary-line {
    margin-top: 0.5rem;
    font-size: 0.9rem;
}

.ss-fp-detail-actions {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--bs-border-color);
}

.ss-fp-detail-status {
    margin-top: 1rem;
    padding: 0.5rem;
    background: var(--bs-body-bg);
    border-radius: 0.25rem;
}

/* List view */
.ss-fp-match-summary {
    padding: 0.5rem 0;
}

.ss-fp-match-arrow {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
}

.ss-fp-arrow {
    color: var(--bs-secondary-color);
}

.ss-fp-match-meta {
    display: flex;
    gap: 0.75rem;
    font-size: 0.8rem;
    color: var(--bs-secondary-color);
    margin-top: 0.25rem;
}

.ss-fp-match-badge {
    font-weight: 600;
    font-size: 0.85rem;
}

.ss-text-warning {
    color: #ffc107;
}
```

**Step 3: Deploy and verify visually**

```bash
scp plugin/stash-sense-recommendations.js plugin/stash-sense.css root@10.0.0.4:/mnt/nvme_cache/appdata/stash/config/plugins/stash-sense/
```

Hard refresh Stash UI (Ctrl+Shift+R) and verify the card, list, and detail views render correctly.

**Step 4: Commit**

```bash
git add plugin/stash-sense-recommendations.js plugin/stash-sense.css
git commit -m "feat: add fingerprint match detail view with accept/dismiss (#48)"
```

---

### Task 12: Integration Test and Final Verification

End-to-end verification of the full pipeline.

**Files:**
- All previously modified files

**Step 1: Run the full test suite**

```bash
cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -m pytest tests/ -v
```
Expected: All tests pass, no regressions

**Step 2: Start sidecar and verify import**

```bash
cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -c "
from job_models import JOB_REGISTRY
from recommendations_router import ANALYZERS
from analyzers.scene_fingerprint_match import SceneFingerprintMatchAnalyzer
from scene_fingerprint_scoring import score_match, is_high_confidence

assert 'scene_fingerprint_match' in JOB_REGISTRY
assert 'scene_fingerprint_match' in ANALYZERS
print('All imports OK')
print(f'Job: {JOB_REGISTRY[\"scene_fingerprint_match\"].display_name}')
print(f'Analyzer: {ANALYZERS[\"scene_fingerprint_match\"].__name__}')
"
```
Expected: All imports OK with correct names

**Step 3: Start sidecar and test endpoints**

```bash
cd /home/carrot/code/stash-sense/api
source ../.venv/bin/activate
make sidecar
```

In a separate terminal, verify the new endpoints exist:

```bash
curl -s http://localhost:5000/docs | grep -o 'fingerprint[^"]*' | head -5
```

**Step 4: Commit any fixes, then final commit**

```bash
git add -A
git status  # Review what's staged
git commit -m "feat: scene fingerprint matching - integration verification (#48)"
```

---

## Implementation Checklist

| # | Component | File(s) | Tests |
|---|-----------|---------|-------|
| 1 | Quality scoring | `api/scene_fingerprint_scoring.py` | `test_scene_fingerprint_scoring.py` |
| 2 | Stash client query | `api/stash_client_unified.py` | `test_scene_fingerprint_query.py` |
| 3 | StashBox batch lookup | `api/stashbox_client.py` | `test_stashbox_fingerprint_query.py` |
| 4 | Analyzer | `api/analyzers/scene_fingerprint_match.py` | `test_scene_fingerprint_match_analyzer.py` |
| 5 | Job registration | `api/job_models.py`, `api/recommendations_router.py` | Import verification |
| 6 | Accept endpoint | `api/recommendations_router.py` | `test_scene_fingerprint_match_analyzer.py` |
| 7 | Accept All endpoint | `api/recommendations_router.py` | `test_scene_fingerprint_match_analyzer.py` |
| 8 | Plugin proxy | `plugin/stash_sense_backend.py` | Manual |
| 9 | Dashboard card | `plugin/stash-sense-recommendations.js` | Visual |
| 10 | List view | `plugin/stash-sense-recommendations.js` | Visual |
| 11 | Detail view | `plugin/stash-sense-recommendations.js`, `plugin/stash-sense.css` | Visual |
| 12 | Integration | All | Full suite |

## Key Design Decisions

- **Composite `target_id`**: `{scene_id}|{endpoint}|{stashbox_scene_id}` enables pair-based dismissals using the existing DB schema with zero changes to `RecommendationsDB`
- **Not using `BaseUpstreamAnalyzer`**: This analyzer finds unlinked scenes, not diffs on linked ones
- **Ambiguity rule**: Multiple matches for one scene from same endpoint → none are `high_confidence`
- **Incremental watermark**: Keyed per-endpoint as `scene_fp_match_{endpoint}` in watermark table
- **Batch size 40**: Matches Stash tagger's own batching to stash-box API
