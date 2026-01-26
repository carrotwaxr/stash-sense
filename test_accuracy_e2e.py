#!/usr/bin/env python3
"""End-to-end accuracy test for face recognition.

Finds performers in the local Stash that exist in our database,
then tests if the face recognition API correctly identifies them in their scenes.
"""
import json
import os
import random
import sys
from dataclasses import dataclass

import httpx

# Configuration
STASH_URL = os.environ.get("STASH_URL", "http://10.0.0.4:6969")
STASH_API_KEY = os.environ.get("STASH_API_KEY", "")
SIDECAR_URL = os.environ.get("SIDECAR_URL", "http://localhost:5000")
DATABASE_PATH = os.environ.get("DATABASE_PATH", "./api/data/performers.json")


@dataclass
class TestResult:
    scene_id: str
    scene_title: str
    expected_performers: list[str]  # StashDB IDs
    detected_performers: list[str]  # StashDB IDs from top match
    all_matches: list[dict]  # Full match data
    success: bool
    error: str = ""


def graphql_query(query: str, variables: dict = None) -> dict:
    """Execute GraphQL query against Stash."""
    headers = {"Content-Type": "application/json"}
    if STASH_API_KEY:
        headers["ApiKey"] = STASH_API_KEY

    response = httpx.post(
        f"{STASH_URL}/graphql",
        json={"query": query, "variables": variables or {}},
        headers=headers,
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def load_database_stashdb_ids() -> set[str]:
    """Load all StashDB IDs from our database."""
    with open(DATABASE_PATH) as f:
        performers = json.load(f)

    # Extract just the UUID part from "stashdb.org:uuid"
    ids = set()
    for key in performers.keys():
        if key.startswith("stashdb.org:"):
            ids.add(key.split(":", 1)[1])
    return ids


def find_matching_performers_in_stash(db_ids: set[str]) -> list[dict]:
    """Find performers in local Stash that have StashDB IDs in our database."""
    # Query performers with StashDB links
    query = """
    query FindPerformersWithStashDB($page: Int!, $per_page: Int!) {
        findPerformers(
            filter: { page: $page, per_page: $per_page }
            performer_filter: {
                stash_id_endpoint: {
                    endpoint: "https://stashdb.org/graphql"
                    modifier: NOT_NULL
                }
            }
        ) {
            count
            performers {
                id
                name
                stash_ids {
                    endpoint
                    stash_id
                }
            }
        }
    }
    """

    matching = []
    page = 1
    per_page = 100

    while True:
        result = graphql_query(query, {"page": page, "per_page": per_page})
        performers = result["data"]["findPerformers"]["performers"]

        if not performers:
            break

        for p in performers:
            for stash_id in p["stash_ids"]:
                if stash_id["endpoint"] == "https://stashdb.org/graphql":
                    if stash_id["stash_id"] in db_ids:
                        matching.append({
                            "local_id": p["id"],
                            "name": p["name"],
                            "stashdb_id": stash_id["stash_id"],
                        })
                    break

        page += 1
        if len(performers) < per_page:
            break

    return matching


def find_scenes_for_performer(performer_id: str, limit: int = 5) -> list[dict]:
    """Find scenes that include a specific performer."""
    query = """
    query FindScenes($performer_id: ID!, $limit: Int!) {
        findScenes(
            scene_filter: { performers: { value: [$performer_id], modifier: INCLUDES } }
            filter: { per_page: $limit, sort: "random" }
        ) {
            scenes {
                id
                title
                performers {
                    id
                    name
                    stash_ids {
                        endpoint
                        stash_id
                    }
                }
                paths {
                    sprite
                }
            }
        }
    }
    """

    result = graphql_query(query, {"performer_id": performer_id, "limit": limit})
    scenes = result["data"]["findScenes"]["scenes"]

    # Filter to scenes that have sprites
    return [s for s in scenes if s.get("paths", {}).get("sprite")]


def identify_scene(scene_id: str) -> dict:
    """Call the face recognition API to identify performers in a scene."""
    response = httpx.post(
        f"{SIDECAR_URL}/identify/scene",
        json={
            "scene_id": scene_id,
            "max_frames": 20,
            "top_k": 5,
            "max_distance": 0.7,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def run_test(performer: dict, scene: dict) -> TestResult:
    """Run identification on a scene and check results."""
    scene_id = scene["id"]
    scene_title = scene.get("title", f"Scene {scene_id}")

    # Get expected StashDB IDs for all performers in this scene
    expected_ids = []
    for p in scene["performers"]:
        for stash_id in p.get("stash_ids", []):
            if stash_id["endpoint"] == "https://stashdb.org/graphql":
                expected_ids.append(stash_id["stash_id"])
                break

    try:
        result = identify_scene(scene_id)

        # Extract detected StashDB IDs from best matches
        detected_ids = []
        all_matches = []

        for person in result.get("persons", []):
            best_match = person.get("best_match")
            if best_match:
                detected_ids.append(best_match["stashdb_id"])
                all_matches.append({
                    "name": best_match["name"],
                    "stashdb_id": best_match["stashdb_id"],
                    "confidence": best_match["confidence"],
                    "frame_count": person["frame_count"],
                })

        # Check if our target performer was detected
        target_id = performer["stashdb_id"]
        success = target_id in detected_ids

        return TestResult(
            scene_id=scene_id,
            scene_title=scene_title,
            expected_performers=expected_ids,
            detected_performers=detected_ids,
            all_matches=all_matches,
            success=success,
        )

    except Exception as e:
        return TestResult(
            scene_id=scene_id,
            scene_title=scene_title,
            expected_performers=expected_ids,
            detected_performers=[],
            all_matches=[],
            success=False,
            error=str(e),
        )


def main():
    print("=" * 60)
    print("End-to-End Face Recognition Accuracy Test")
    print("=" * 60)
    print()

    # Configuration
    print(f"Stash URL: {STASH_URL}")
    print(f"Sidecar URL: {SIDECAR_URL}")
    print(f"Database: {DATABASE_PATH}")
    print()

    # Load database
    print("Loading database...")
    db_ids = load_database_stashdb_ids()
    print(f"  Database contains {len(db_ids)} StashDB performers")
    print()

    # Find matching performers in Stash
    print("Finding performers in your Stash that match our database...")
    matching_performers = find_matching_performers_in_stash(db_ids)
    print(f"  Found {len(matching_performers)} matching performers")

    if not matching_performers:
        print("\nNo matching performers found. Cannot run test.")
        return

    # Sample performers for testing
    max_performers = min(20, len(matching_performers))
    test_performers = random.sample(matching_performers, max_performers)
    print(f"  Testing {max_performers} performers")
    print()

    # Run tests
    results: list[TestResult] = []
    scenes_tested = 0
    max_scenes_per_performer = 2

    for i, performer in enumerate(test_performers):
        print(f"[{i+1}/{max_performers}] Testing: {performer['name']}")

        # Find scenes with this performer
        scenes = find_scenes_for_performer(
            performer["local_id"],
            limit=max_scenes_per_performer
        )

        if not scenes:
            print(f"  No scenes with sprites found, skipping")
            continue

        for scene in scenes:
            print(f"  Scene: {scene.get('title', scene['id'])[:50]}...")
            result = run_test(performer, scene)
            results.append(result)
            scenes_tested += 1

            if result.error:
                print(f"    ERROR: {result.error}")
            elif result.success:
                match = next((m for m in result.all_matches
                            if m["stashdb_id"] == performer["stashdb_id"]), None)
                if match:
                    print(f"    ✓ FOUND: {match['name']} ({match['confidence']*100:.0f}% confidence)")
            else:
                print(f"    ✗ NOT FOUND in results")
                if result.all_matches:
                    print(f"      Detected instead: {[m['name'] for m in result.all_matches]}")

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print()

    successful = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success and not r.error)
    errors = sum(1 for r in results if r.error)

    print(f"Scenes tested: {len(results)}")
    print(f"  ✓ Correct identification: {successful} ({successful/len(results)*100:.1f}%)")
    print(f"  ✗ Wrong/no identification: {failed} ({failed/len(results)*100:.1f}%)")
    print(f"  ⚠ Errors: {errors}")
    print()

    # Show some failure details
    failures = [r for r in results if not r.success and not r.error]
    if failures:
        print("Sample failures:")
        for r in failures[:5]:
            print(f"  Scene {r.scene_id}: {r.scene_title[:40]}...")
            print(f"    Expected: {r.expected_performers[:3]}")
            print(f"    Detected: {[m['name'] for m in r.all_matches[:3]]}")

    # Calculate per-performer accuracy
    print()
    print("Per-performer breakdown:")
    by_performer = {}
    for r in results:
        for exp_id in r.expected_performers:
            if exp_id not in by_performer:
                by_performer[exp_id] = {"total": 0, "found": 0}
            by_performer[exp_id]["total"] += 1
            if exp_id in r.detected_performers:
                by_performer[exp_id]["found"] += 1

    for perf_id, stats in sorted(by_performer.items(), key=lambda x: x[1]["found"]/max(x[1]["total"],1), reverse=True)[:10]:
        rate = stats["found"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {perf_id[:20]}... {stats['found']}/{stats['total']} ({rate:.0f}%)")


if __name__ == "__main__":
    main()
