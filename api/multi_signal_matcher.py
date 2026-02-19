"""
Multi-signal performer identification matcher.

Combines face recognition with body proportion and tattoo signals to improve
performer identification accuracy through re-ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from body_proportions import BodyProportions
from signal_scoring import body_ratio_penalty, tattoo_adjustment
from tattoo_detector import TattooResult

if TYPE_CHECKING:
    from body_proportions import BodyProportionExtractor
    from database_reader import PerformerDatabaseReader
    from recognizer import FaceRecognizer
    from tattoo_detector import TattooDetector
    from tattoo_matcher import TattooMatcher


@dataclass
class MultiSignalMatch:
    """Result of multi-signal matching for a single detected face."""

    face: Any  # DetectedFace from recognizer
    matches: list[Any]  # list[PerformerMatch]
    body_ratios: Optional[BodyProportions] = None
    tattoo_result: Optional[TattooResult] = None
    signals_used: list[str] = field(default_factory=list)


class MultiSignalMatcher:
    """
    Combines face recognition with body and tattoo signals for identification.

    Wraps the existing FaceRecognizer and adds body proportion and tattoo
    signal re-ranking to improve identification accuracy.

    Usage:
        matcher = MultiSignalMatcher(
            face_recognizer=recognizer,
            db_reader=db_reader,
            body_extractor=body_extractor,  # optional
            tattoo_detector=tattoo_detector,  # optional
            tattoo_matcher=tattoo_matcher,  # optional — embedding-based matching
        )

        results = matcher.identify(image, top_k=5, use_body=True, use_tattoo=True)
    """

    def __init__(
        self,
        face_recognizer: FaceRecognizer,
        db_reader: PerformerDatabaseReader,
        body_extractor: Optional[BodyProportionExtractor] = None,
        tattoo_detector: Optional[TattooDetector] = None,
        tattoo_matcher: Optional[TattooMatcher] = None,
    ):
        """
        Initialize the multi-signal matcher.

        Args:
            face_recognizer: FaceRecognizer instance for face matching
            db_reader: PerformerDatabaseReader for loading body data
            body_extractor: Optional BodyProportionExtractor for body analysis
            tattoo_detector: Optional TattooDetector for YOLO tattoo detection
            tattoo_matcher: Optional TattooMatcher for embedding-based tattoo matching
        """
        self.face_recognizer = face_recognizer
        self.db_reader = db_reader
        self.body_extractor = body_extractor
        self.tattoo_detector = tattoo_detector
        self.tattoo_matcher = tattoo_matcher

        # Preload body data from database
        self.body_data = db_reader.get_all_body_proportions()

        # Build set of performer IDs that have tattoo embeddings in the index
        self.performers_with_tattoo_embeddings: set[str] = set()
        if tattoo_matcher and tattoo_matcher.tattoo_mapping:
            for entry in tattoo_matcher.tattoo_mapping:
                if entry is not None:
                    self.performers_with_tattoo_embeddings.add(entry["universal_id"])

    def identify(
        self,
        image: np.ndarray,
        top_k: int = 5,
        use_body: bool = True,
        use_tattoo: bool = True,
        face_candidates: int = 20,
    ) -> list[MultiSignalMatch]:
        """
        Identify performers in an image using multiple signals.

        Args:
            image: RGB image as numpy array (H, W, 3)
            top_k: Number of top matches to return per face
            use_body: Whether to use body proportions for re-ranking
            use_tattoo: Whether to use tattoo detection for re-ranking
            face_candidates: Number of face candidates to retrieve for re-ranking

        Returns:
            List of MultiSignalMatch objects, one per detected face
        """
        # Step 1: Get face candidates
        face_results = self.face_recognizer.recognize_image(image, top_k=face_candidates)

        # Step 2: Extract body ratios if enabled and extractor available
        body_ratios = None
        if use_body and self.body_extractor is not None:
            body_ratios = self.body_extractor.extract(image)

        # Step 3: Detect tattoos and run embedding matching if enabled
        tattoo_result = None
        tattoo_scores = None
        if use_tattoo and self.tattoo_detector is not None:
            tattoo_result = self.tattoo_detector.detect(image)

            # If we have a tattoo matcher and detected tattoos, run embedding matching
            if (tattoo_result and tattoo_result.has_tattoos
                    and self.tattoo_matcher is not None):
                tattoo_scores = self.tattoo_matcher.match(
                    image, tattoo_result.detections
                )

        # Step 4: Build signals_used list
        signals_used = ["face"]
        if use_body and self.body_extractor is not None and body_ratios is not None:
            signals_used.append("body")
        if use_tattoo and self.tattoo_detector is not None and tattoo_result is not None:
            signals_used.append("tattoo")

        # Step 5: Process each face result
        results = []
        for face_result in face_results:
            # Re-rank candidates using additional signals
            reranked_matches = self._rerank_candidates(
                candidates=face_result.matches,
                body_ratios=body_ratios,
                tattoo_result=tattoo_result,
                tattoo_scores=tattoo_scores,
                top_k=top_k,
            )

            results.append(
                MultiSignalMatch(
                    face=face_result.face,
                    matches=reranked_matches,
                    body_ratios=body_ratios,
                    tattoo_result=tattoo_result,
                    signals_used=signals_used.copy(),
                )
            )

        return results

    def _rerank_candidates(
        self,
        candidates: list[Any],
        body_ratios: Optional[BodyProportions],
        tattoo_result: Optional[TattooResult],
        tattoo_scores: Optional[dict[str, float]],
        top_k: int,
    ) -> list[Any]:
        """
        Re-rank candidates using body and tattoo signals.

        Args:
            candidates: List of PerformerMatch from face recognition
            body_ratios: Body proportions from query image (may be None)
            tattoo_result: Tattoo detection result from query image (may be None)
            tattoo_scores: Tattoo embedding similarity scores from TattooMatcher
            top_k: Number of top results to return

        Returns:
            Re-ranked list of PerformerMatch, sorted by final score descending
        """
        scored_candidates = []

        for candidate in candidates:
            # Step 1: Convert distance to similarity
            # combined_score is a distance (lower is better)
            # We convert to similarity (higher is better)
            base_score = 1.0 / (1.0 + candidate.combined_score)

            # Step 2: Apply body ratio penalty
            candidate_body = self._get_candidate_body_ratios(candidate.universal_id)
            body_mult = body_ratio_penalty(body_ratios, candidate_body)

            # Step 3: Apply tattoo adjustment (embedding-based)
            has_embeddings = candidate.universal_id in self.performers_with_tattoo_embeddings
            tattoo_mult = tattoo_adjustment(
                tattoo_result, candidate.universal_id, tattoo_scores, has_embeddings
            )

            # Step 4: Calculate final score
            final_score = base_score * body_mult * tattoo_mult

            scored_candidates.append((candidate, final_score))

        # Step 5: Sort by final score descending (highest first)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Return top_k candidates
        return [candidate for candidate, score in scored_candidates[:top_k]]

    def _get_candidate_body_ratios(
        self, universal_id: str
    ) -> Optional[BodyProportions]:
        """
        Get body proportions for a candidate from preloaded data.

        Args:
            universal_id: Candidate's universal ID (e.g., "stashdb.org:uuid")

        Returns:
            BodyProportions if available, None otherwise
        """
        data = self.body_data.get(universal_id)
        if data is None:
            return None

        return BodyProportions(
            shoulder_hip_ratio=data["shoulder_hip_ratio"],
            leg_torso_ratio=data["leg_torso_ratio"],
            arm_span_height_ratio=data["arm_span_height_ratio"],
            confidence=data["confidence"],
        )
