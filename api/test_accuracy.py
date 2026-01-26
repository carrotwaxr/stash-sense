"""Comprehensive accuracy test for face recognition.

Tests the full pipeline:
1. Build database from StashDB performers that exist in local Stash
2. Test recognition using local Stash images
3. Compare results to ground truth (known StashDB links)
4. Generate detailed accuracy report
"""
import json
import os
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import requests

load_dotenv()

from config import DatabaseConfig, BuilderConfig, StashDBConfig, StashConfig
from stash_client import StashClient, Performer
from stashdb_client import StashDBClient
from database_builder import DatabaseBuilder
from recognizer import FaceRecognizer
from embeddings import load_image


@dataclass
class TestCase:
    """A single test case."""
    performer_name: str
    local_stash_id: str
    expected_stashdb_id: str
    image_source: str  # "profile" or "scene"
    image_url: str


@dataclass
class TestResult:
    """Result of a single test case."""
    test_case: TestCase
    face_detected: bool
    face_confidence: float = 0.0
    top_match_stashdb_id: Optional[str] = None
    top_match_name: Optional[str] = None
    top_match_score: float = 0.0
    correct_in_top_1: bool = False
    correct_in_top_3: bool = False
    correct_in_top_5: bool = False
    all_matches: list = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class AccuracyReport:
    """Overall accuracy report."""
    total_tests: int
    faces_detected: int
    top_1_accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float
    avg_correct_score: float
    avg_incorrect_score: float
    detection_rate: float
    results: list[TestResult]


def build_test_database(
    stash_client: StashClient,
    stashdb_client: StashDBClient,
    output_dir: Path,
    max_performers: int = 100,
    max_images: int = 3,
) -> tuple[list[Performer], DatabaseConfig]:
    """
    Build a test database from performers in local Stash that have StashDB IDs.

    Returns:
        Tuple of (list of local performers used, database config)
    """
    print("=" * 60)
    print("PHASE 1: Building Test Database")
    print("=" * 60)

    # Get performers from local Stash that have StashDB IDs
    print("\nQuerying local Stash for performers with StashDB links...")
    count, all_performers = stash_client.get_performers(
        per_page=500,
        with_stashdb_id=True,
    )
    print(f"Found {count} performers with StashDB links")

    # Filter to those with images or scenes
    performers_with_content = [
        p for p in all_performers
        if p.image_url or p.scene_count > 0
    ]
    print(f"  {len(performers_with_content)} have profile images or scenes")

    # Sample if we have too many
    if len(performers_with_content) > max_performers:
        performers_to_use = random.sample(performers_with_content, max_performers)
        print(f"  Sampled {max_performers} for testing")
    else:
        performers_to_use = performers_with_content

    # Get their StashDB IDs
    stashdb_ids = [p.stashdb_id for p in performers_to_use if p.stashdb_id]
    print(f"\nBuilding database from {len(stashdb_ids)} StashDB performers...")

    # Build database
    db_config = DatabaseConfig(data_dir=output_dir)
    builder_config = BuilderConfig(
        max_images_per_performer=max_images,
        min_face_confidence=0.8,
    )

    builder = DatabaseBuilder(db_config, builder_config, stashdb_client)
    builder.build_from_stashdb(performer_ids=stashdb_ids)
    builder.save()

    return performers_to_use, db_config


def create_test_cases(
    stash_client: StashClient,
    performers: list[Performer],
    max_per_performer: int = 2,
) -> list[TestCase]:
    """
    Create test cases from local Stash images.

    Uses profile images and scene screenshots as test inputs.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Creating Test Cases")
    print("=" * 60)

    test_cases = []

    for performer in tqdm(performers, desc="Creating test cases"):
        cases_for_performer = 0

        # Use profile image if available
        if performer.image_url and cases_for_performer < max_per_performer:
            test_cases.append(TestCase(
                performer_name=performer.name,
                local_stash_id=performer.id,
                expected_stashdb_id=performer.stashdb_id,
                image_source="profile",
                image_url=performer.image_url,
            ))
            cases_for_performer += 1

        # Use scene screenshots
        if performer.scene_count > 0 and cases_for_performer < max_per_performer:
            try:
                scenes = stash_client.get_performer_scenes(performer.id, limit=3)
                for scene in scenes:
                    if cases_for_performer >= max_per_performer:
                        break
                    screenshot_url = scene.get("paths", {}).get("screenshot")
                    if screenshot_url:
                        test_cases.append(TestCase(
                            performer_name=performer.name,
                            local_stash_id=performer.id,
                            expected_stashdb_id=performer.stashdb_id,
                            image_source="scene",
                            image_url=screenshot_url,
                        ))
                        cases_for_performer += 1
            except Exception as e:
                print(f"  Error getting scenes for {performer.name}: {e}")

    print(f"\nCreated {len(test_cases)} test cases")
    profile_cases = sum(1 for tc in test_cases if tc.image_source == "profile")
    scene_cases = sum(1 for tc in test_cases if tc.image_source == "scene")
    print(f"  Profile images: {profile_cases}")
    print(f"  Scene screenshots: {scene_cases}")

    return test_cases


def run_tests(
    recognizer: FaceRecognizer,
    test_cases: list[TestCase],
    stash_api_key: str,
) -> list[TestResult]:
    """Run recognition tests on all test cases."""
    print("\n" + "=" * 60)
    print("PHASE 3: Running Recognition Tests")
    print("=" * 60)

    results = []

    for test_case in tqdm(test_cases, desc="Testing"):
        result = TestResult(test_case=test_case, face_detected=False)

        try:
            # Download image from local Stash
            response = requests.get(
                test_case.image_url,
                headers={"ApiKey": stash_api_key},
                timeout=30,
            )
            response.raise_for_status()
            image = load_image(response.content)

            # Run recognition
            recognition_results = recognizer.recognize_image(
                image,
                top_k=5,
                max_distance=1.5,  # Be lenient for testing
            )

            if not recognition_results:
                result.face_detected = False
            else:
                # Use the first (most confident) face
                rec_result = recognition_results[0]
                result.face_detected = True
                result.face_confidence = rec_result.face.confidence

                if rec_result.matches:
                    # Store all matches for analysis
                    result.all_matches = [
                        {
                            "stashdb_id": m.stashdb_id,
                            "name": m.name,
                            "score": m.combined_score,
                        }
                        for m in rec_result.matches
                    ]

                    # Check top match
                    top = rec_result.matches[0]
                    result.top_match_stashdb_id = top.stashdb_id
                    result.top_match_name = top.name
                    result.top_match_score = top.combined_score

                    # Check accuracy at different k values
                    matched_ids = [m.stashdb_id for m in rec_result.matches]
                    result.correct_in_top_1 = test_case.expected_stashdb_id in matched_ids[:1]
                    result.correct_in_top_3 = test_case.expected_stashdb_id in matched_ids[:3]
                    result.correct_in_top_5 = test_case.expected_stashdb_id in matched_ids[:5]

        except Exception as e:
            result.error = str(e)

        results.append(result)

    return results


def calculate_accuracy(results: list[TestResult]) -> AccuracyReport:
    """Calculate accuracy metrics from test results."""
    total = len(results)
    detected = sum(1 for r in results if r.face_detected)
    top1_correct = sum(1 for r in results if r.correct_in_top_1)
    top3_correct = sum(1 for r in results if r.correct_in_top_3)
    top5_correct = sum(1 for r in results if r.correct_in_top_5)

    # Average scores for correct vs incorrect matches
    correct_scores = [r.top_match_score for r in results if r.correct_in_top_1 and r.top_match_score > 0]
    incorrect_scores = [r.top_match_score for r in results if not r.correct_in_top_1 and r.top_match_score > 0]

    return AccuracyReport(
        total_tests=total,
        faces_detected=detected,
        top_1_accuracy=top1_correct / total if total > 0 else 0,
        top_3_accuracy=top3_correct / total if total > 0 else 0,
        top_5_accuracy=top5_correct / total if total > 0 else 0,
        avg_correct_score=np.mean(correct_scores) if correct_scores else 0,
        avg_incorrect_score=np.mean(incorrect_scores) if incorrect_scores else 0,
        detection_rate=detected / total if total > 0 else 0,
        results=results,
    )


def generate_html_report(report: AccuracyReport, output_path: Path):
    """Generate detailed HTML report."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Face Recognition Accuracy Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }}
        h1, h2 {{ color: #fff; }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric {{
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 36px;
            font-weight: bold;
            color: #51cf66;
        }}
        .metric-value.warning {{ color: #ffd43b; }}
        .metric-value.danger {{ color: #ff6b6b; }}
        .metric-label {{ font-size: 14px; color: #888; margin-top: 5px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: #2a2a2a;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #3a3a3a; }}
        th {{ background: #3a3a3a; }}
        tr.correct {{ background: rgba(81, 207, 102, 0.1); }}
        tr.incorrect {{ background: rgba(255, 107, 107, 0.1); }}
        tr.no-face {{ background: rgba(255, 212, 59, 0.1); }}
        .score {{ font-family: monospace; }}
        .match-list {{ font-size: 12px; color: #888; }}
    </style>
</head>
<body>
    <h1>Face Recognition Accuracy Report</h1>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <div class="metrics">
        <div class="metric">
            <div class="metric-value{' warning' if report.top_1_accuracy < 0.8 else ''}{' danger' if report.top_1_accuracy < 0.5 else ''}">{report.top_1_accuracy*100:.1f}%</div>
            <div class="metric-label">Top-1 Accuracy</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report.top_3_accuracy*100:.1f}%</div>
            <div class="metric-label">Top-3 Accuracy</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report.top_5_accuracy*100:.1f}%</div>
            <div class="metric-label">Top-5 Accuracy</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report.detection_rate*100:.1f}%</div>
            <div class="metric-label">Face Detection Rate</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report.total_tests}</div>
            <div class="metric-label">Total Tests</div>
        </div>
        <div class="metric">
            <div class="metric-value">{report.faces_detected}</div>
            <div class="metric-label">Faces Detected</div>
        </div>
        <div class="metric">
            <div class="metric-value score">{report.avg_correct_score:.3f}</div>
            <div class="metric-label">Avg Correct Score</div>
        </div>
        <div class="metric">
            <div class="metric-value score">{report.avg_incorrect_score:.3f}</div>
            <div class="metric-label">Avg Incorrect Score</div>
        </div>
    </div>

    <h2>Detailed Results</h2>
    <table>
        <tr>
            <th>Performer</th>
            <th>Source</th>
            <th>Expected</th>
            <th>Top Match</th>
            <th>Score</th>
            <th>Top-1</th>
            <th>Top-3</th>
            <th>Top-5</th>
        </tr>
"""

    for result in report.results:
        tc = result.test_case

        if not result.face_detected:
            row_class = "no-face"
            status = "No face"
        elif result.correct_in_top_1:
            row_class = "correct"
            status = "✓"
        else:
            row_class = "incorrect"
            status = "✗"

        top_match = result.top_match_name or "-"
        score = f"{result.top_match_score:.3f}" if result.top_match_score else "-"

        html += f"""
        <tr class="{row_class}">
            <td>{tc.performer_name}</td>
            <td>{tc.image_source}</td>
            <td title="{tc.expected_stashdb_id}">{tc.performer_name}</td>
            <td title="{result.top_match_stashdb_id or ''}">{top_match}</td>
            <td class="score">{score}</td>
            <td>{"✓" if result.correct_in_top_1 else ("?" if not result.face_detected else "✗")}</td>
            <td>{"✓" if result.correct_in_top_3 else ("?" if not result.face_detected else "✗")}</td>
            <td>{"✓" if result.correct_in_top_5 else ("?" if not result.face_detected else "✗")}</td>
        </tr>
"""

    html += """
    </table>

    <h2>Score Distribution Analysis</h2>
    <p>Lower scores are better (cosine distance). A good threshold separates correct from incorrect matches.</p>
    <ul>
        <li><strong>Average correct match score:</strong> """ + f"{report.avg_correct_score:.3f}" + """</li>
        <li><strong>Average incorrect match score:</strong> """ + f"{report.avg_incorrect_score:.3f}" + """</li>
        <li><strong>Suggested threshold:</strong> """ + f"{(report.avg_correct_score + report.avg_incorrect_score) / 2:.3f}" + """ (midpoint)</li>
    </ul>

</body>
</html>
"""

    output_path.write_text(html)
    print(f"\nHTML report saved to {output_path}")


def main():
    """Run the full accuracy test."""
    import argparse

    parser = argparse.ArgumentParser(description="Test face recognition accuracy")
    parser.add_argument("--max-performers", type=int, default=50,
                        help="Maximum performers to test")
    parser.add_argument("--max-images-db", type=int, default=3,
                        help="Max images per performer for database building")
    parser.add_argument("--max-tests-per-performer", type=int, default=2,
                        help="Max test cases per performer")
    parser.add_argument("--output", type=str, default="./test_output",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize clients
    stash_config = StashConfig.from_env()
    stashdb_config = StashDBConfig.from_env()

    stash_client = StashClient(stash_config.url, stash_config.api_key)
    stashdb_client = StashDBClient(
        stashdb_config.url,
        stashdb_config.api_key,
        rate_limit_delay=stashdb_config.rate_limit_delay,
    )

    # Test connection
    print("Testing connections...")
    try:
        stats = stash_client.get_stats()
        print(f"  Local Stash: {stats['performer_count']} performers, {stats['scene_count']} scenes")
    except Exception as e:
        print(f"  ERROR: Could not connect to local Stash: {e}")
        return

    # Phase 1: Build test database
    performers, db_config = build_test_database(
        stash_client,
        stashdb_client,
        output_dir / "db",
        max_performers=args.max_performers,
        max_images=args.max_images_db,
    )

    # Phase 2: Create test cases
    test_cases = create_test_cases(
        stash_client,
        performers,
        max_per_performer=args.max_tests_per_performer,
    )

    if not test_cases:
        print("ERROR: No test cases created!")
        return

    # Phase 3: Run tests
    recognizer = FaceRecognizer(db_config)
    results = run_tests(recognizer, test_cases, stash_config.api_key)

    # Phase 4: Calculate and report accuracy
    print("\n" + "=" * 60)
    print("PHASE 4: Accuracy Results")
    print("=" * 60)

    report = calculate_accuracy(results)

    print(f"\n{'='*40}")
    print("ACCURACY SUMMARY")
    print(f"{'='*40}")
    print(f"Total tests:        {report.total_tests}")
    print(f"Faces detected:     {report.faces_detected} ({report.detection_rate*100:.1f}%)")
    print(f"Top-1 accuracy:     {report.top_1_accuracy*100:.1f}%")
    print(f"Top-3 accuracy:     {report.top_3_accuracy*100:.1f}%")
    print(f"Top-5 accuracy:     {report.top_5_accuracy*100:.1f}%")
    print(f"Avg correct score:  {report.avg_correct_score:.3f}")
    print(f"Avg incorrect score: {report.avg_incorrect_score:.3f}")
    print(f"{'='*40}")

    # Generate HTML report
    report_path = output_dir / f"accuracy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    generate_html_report(report, report_path)

    # Save raw results as JSON
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump({
            "metrics": {
                "total_tests": report.total_tests,
                "faces_detected": report.faces_detected,
                "top_1_accuracy": report.top_1_accuracy,
                "top_3_accuracy": report.top_3_accuracy,
                "top_5_accuracy": report.top_5_accuracy,
                "detection_rate": report.detection_rate,
                "avg_correct_score": report.avg_correct_score,
                "avg_incorrect_score": report.avg_incorrect_score,
            },
            "results": [
                {
                    "performer": r.test_case.performer_name,
                    "expected_id": r.test_case.expected_stashdb_id,
                    "source": r.test_case.image_source,
                    "face_detected": r.face_detected,
                    "top_match_id": r.top_match_stashdb_id,
                    "top_match_name": r.top_match_name,
                    "score": r.top_match_score,
                    "correct_top_1": r.correct_in_top_1,
                    "correct_top_3": r.correct_in_top_3,
                    "correct_top_5": r.correct_in_top_5,
                }
                for r in report.results
            ]
        }, f, indent=2)
    print(f"JSON results saved to {json_path}")


if __name__ == "__main__":
    main()
