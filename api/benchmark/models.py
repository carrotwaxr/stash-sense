"""Data models for the benchmark framework.

These dataclasses define the structure for test scenes, expected performers,
benchmark parameters, results, and checkpointing state.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ExpectedPerformer:
    """A performer expected in a test scene.

    Contains identifying information and metadata about the performer's
    coverage in the face recognition database.
    """
    stashdb_id: str
    name: str
    faces_in_db: int
    has_body_data: bool
    has_tattoo_data: bool


@dataclass
class TestScene:
    """A scene with known ground truth for benchmarking.

    Contains scene metadata and the list of performers expected to be
    identified in the scene.
    """
    scene_id: str
    stashdb_id: str
    title: str
    resolution: str  # "480p", "720p", "1080p", "4k"
    width: int
    height: int
    duration_sec: float
    expected_performers: list[ExpectedPerformer]
    db_coverage_tier: str  # "well-covered", "sparse"

    def is_well_covered(self) -> bool:
        """Return True if scene has well-covered database coverage."""
        return self.db_coverage_tier == "well-covered"


@dataclass
class BenchmarkParams:
    """Parameters for a benchmark run.

    These parameters control the face detection, matching, and clustering
    behavior during benchmark execution.
    """
    matching_mode: str = "frequency"
    max_distance: float = 0.5
    min_face_size: int = 40
    use_multi_signal: bool = True
    num_frames: int = 60
    start_offset_pct: float = 0.05
    end_offset_pct: float = 0.95
    min_face_confidence: float = 0.5
    top_k: int = 5
    cluster_threshold: float = 0.6
    facenet_weight: float = 0.5
    arcface_weight: float = 0.5
    min_appearances: int = 2
    min_unique_frames: int = 2
    min_confidence: float = 0.35

    def to_dict(self) -> dict:
        """Convert parameters to dictionary."""
        return asdict(self)


@dataclass
class SceneResult:
    """Results from running identification on a scene.

    Contains both the accuracy metrics and diagnostic information about
    the identification process.
    """
    scene_id: str
    params: dict
    true_positives: int
    false_negatives: int
    false_positives: int
    expected_in_top_1: int
    expected_in_top_3: int
    correct_match_scores: list[float]
    incorrect_match_scores: list[float]
    score_gap: float
    faces_detected: int
    faces_after_filter: int
    persons_clustered: int
    elapsed_sec: float

    @property
    def accuracy(self) -> float:
        """Calculate accuracy as TP / (TP + FN).

        Returns 0.0 if there are no expected performers (division by zero).
        """
        total = self.true_positives + self.false_negatives
        if total == 0:
            return 0.0
        return self.true_positives / total


@dataclass
class PerformerResult:
    """Result for a single expected performer.

    Tracks whether the performer was found and diagnostic information
    about why they may have been missed.
    """
    stashdb_id: str
    name: str
    faces_in_db: int
    has_body_data: bool
    has_tattoo_data: bool
    was_found: bool
    rank_if_found: Optional[int]
    confidence_if_found: Optional[float]
    distance_if_found: Optional[float]
    who_beat_them: list[tuple[str, float]]
    best_match_for_missed: Optional[str]


@dataclass
class AggregateMetrics:
    """Aggregate metrics across multiple scenes.

    Provides summary statistics and breakdowns by scene characteristics.
    """
    total_scenes: int
    total_expected: int
    total_true_positives: int
    total_false_positives: int
    total_false_negatives: int
    accuracy: float
    precision: float
    recall: float
    accuracy_by_resolution: dict[str, float] = field(default_factory=dict)
    accuracy_by_coverage: dict[str, float] = field(default_factory=dict)
    accuracy_by_face_count: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_results(cls, results: list[SceneResult]) -> "AggregateMetrics":
        """Compute aggregate metrics from a list of scene results.

        Args:
            results: List of SceneResult objects to aggregate.

        Returns:
            AggregateMetrics with computed summary statistics.
        """
        if not results:
            return cls(
                total_scenes=0,
                total_expected=0,
                total_true_positives=0,
                total_false_positives=0,
                total_false_negatives=0,
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
            )

        total_scenes = len(results)
        total_tp = sum(r.true_positives for r in results)
        total_fp = sum(r.false_positives for r in results)
        total_fn = sum(r.false_negatives for r in results)
        total_expected = total_tp + total_fn

        # Calculate accuracy: TP / (TP + FN)
        accuracy = total_tp / total_expected if total_expected > 0 else 0.0

        # Calculate precision: TP / (TP + FP)
        precision_denom = total_tp + total_fp
        precision = total_tp / precision_denom if precision_denom > 0 else 0.0

        # Calculate recall: TP / (TP + FN) - same as accuracy for this metric
        recall = accuracy

        return cls(
            total_scenes=total_scenes,
            total_expected=total_expected,
            total_true_positives=total_tp,
            total_false_positives=total_fp,
            total_false_negatives=total_fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
        )


@dataclass
class BenchmarkState:
    """Checkpoint state for iterative benchmark.

    Enables saving and restoring benchmark progress across runs.
    """
    round_num: int
    scenes: list[TestScene]
    results_by_round: dict[int, list[SceneResult]]
    current_best_params: dict
    current_best_accuracy: float
    parameters_eliminated: list[str]

    def to_json(self) -> str:
        """Serialize state to JSON string.

        Returns:
            JSON string representation of the state.
        """
        data = {
            "round_num": self.round_num,
            "scenes": [
                {
                    "scene_id": s.scene_id,
                    "stashdb_id": s.stashdb_id,
                    "title": s.title,
                    "resolution": s.resolution,
                    "width": s.width,
                    "height": s.height,
                    "duration_sec": s.duration_sec,
                    "expected_performers": [
                        {
                            "stashdb_id": p.stashdb_id,
                            "name": p.name,
                            "faces_in_db": p.faces_in_db,
                            "has_body_data": p.has_body_data,
                            "has_tattoo_data": p.has_tattoo_data,
                        }
                        for p in s.expected_performers
                    ],
                    "db_coverage_tier": s.db_coverage_tier,
                }
                for s in self.scenes
            ],
            "results_by_round": {
                str(round_num): [
                    {
                        "scene_id": r.scene_id,
                        "params": r.params,
                        "true_positives": r.true_positives,
                        "false_negatives": r.false_negatives,
                        "false_positives": r.false_positives,
                        "expected_in_top_1": r.expected_in_top_1,
                        "expected_in_top_3": r.expected_in_top_3,
                        "correct_match_scores": r.correct_match_scores,
                        "incorrect_match_scores": r.incorrect_match_scores,
                        "score_gap": r.score_gap,
                        "faces_detected": r.faces_detected,
                        "faces_after_filter": r.faces_after_filter,
                        "persons_clustered": r.persons_clustered,
                        "elapsed_sec": r.elapsed_sec,
                    }
                    for r in results
                ]
                for round_num, results in self.results_by_round.items()
            },
            "current_best_params": self.current_best_params,
            "current_best_accuracy": self.current_best_accuracy,
            "parameters_eliminated": self.parameters_eliminated,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "BenchmarkState":
        """Deserialize state from JSON string.

        Args:
            json_str: JSON string representation of the state.

        Returns:
            BenchmarkState reconstructed from the JSON.
        """
        data = json.loads(json_str)

        # Reconstruct scenes with expected performers
        scenes = []
        for s_data in data["scenes"]:
            performers = [
                ExpectedPerformer(
                    stashdb_id=p["stashdb_id"],
                    name=p["name"],
                    faces_in_db=p["faces_in_db"],
                    has_body_data=p["has_body_data"],
                    has_tattoo_data=p["has_tattoo_data"],
                )
                for p in s_data["expected_performers"]
            ]
            scene = TestScene(
                scene_id=s_data["scene_id"],
                stashdb_id=s_data["stashdb_id"],
                title=s_data["title"],
                resolution=s_data["resolution"],
                width=s_data["width"],
                height=s_data["height"],
                duration_sec=s_data["duration_sec"],
                expected_performers=performers,
                db_coverage_tier=s_data["db_coverage_tier"],
            )
            scenes.append(scene)

        # Reconstruct results by round
        results_by_round = {}
        for round_str, results_data in data["results_by_round"].items():
            round_num = int(round_str)
            results = [
                SceneResult(
                    scene_id=r["scene_id"],
                    params=r["params"],
                    true_positives=r["true_positives"],
                    false_negatives=r["false_negatives"],
                    false_positives=r["false_positives"],
                    expected_in_top_1=r["expected_in_top_1"],
                    expected_in_top_3=r["expected_in_top_3"],
                    correct_match_scores=r["correct_match_scores"],
                    incorrect_match_scores=r["incorrect_match_scores"],
                    score_gap=r["score_gap"],
                    faces_detected=r["faces_detected"],
                    faces_after_filter=r["faces_after_filter"],
                    persons_clustered=r["persons_clustered"],
                    elapsed_sec=r["elapsed_sec"],
                )
                for r in results_data
            ]
            results_by_round[round_num] = results

        return cls(
            round_num=data["round_num"],
            scenes=scenes,
            results_by_round=results_by_round,
            current_best_params=data["current_best_params"],
            current_best_accuracy=data["current_best_accuracy"],
            parameters_eliminated=data["parameters_eliminated"],
        )
