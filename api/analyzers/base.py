"""
Base Analyzer Class

All analyzers inherit from this class and implement the run() method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..stash_client_unified import StashClientUnified
    from ..recommendations_db import RecommendationsDB


@dataclass
class AnalysisResult:
    """Result of an analysis run."""
    items_processed: int
    recommendations_created: int
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class BaseAnalyzer(ABC):
    """
    Base class for all analyzers.

    Each analyzer:
    - Has a `type` that matches the recommendation type it produces
    - Implements `run()` to analyze the library and create recommendations
    - Uses the stash client to query the user's Stash instance
    - Uses the recommendations DB to create/check recommendations
    """

    type: str  # Recommendation type this analyzer produces

    def __init__(
        self,
        stash: "StashClientUnified",
        rec_db: "RecommendationsDB",
        run_id: Optional[int] = None,
    ):
        self.stash = stash
        self.rec_db = rec_db
        self.run_id = run_id

    def set_items_total(self, total: int):
        """Report total items to process for this run."""
        if self.run_id is not None:
            self.rec_db.update_analysis_items_total(self.run_id, total)

    def update_progress(self, items_processed: int, recommendations_created: int):
        """Report progress for this run."""
        if self.run_id is not None:
            self.rec_db.update_analysis_progress(self.run_id, items_processed, recommendations_created)

    @abstractmethod
    async def run(self, incremental: bool = True) -> AnalysisResult:
        """
        Run the analysis.

        Args:
            incremental: If True, only analyze items changed since last run.
                        If False, analyze everything.

        Returns:
            AnalysisResult with counts and any errors.
        """
        raise NotImplementedError

    def is_dismissed(self, target_type: str, target_id: str) -> bool:
        """Check if this target has been dismissed for this recommendation type."""
        return self.rec_db.is_dismissed(self.type, target_type, target_id)

    def create_recommendation(
        self,
        target_type: str,
        target_id: str,
        details: dict,
        confidence: float = None,
        source_analysis_id: int = None,
    ) -> int | None:
        """Create a recommendation if not already dismissed."""
        if self.is_dismissed(target_type, target_id):
            return None

        return self.rec_db.create_recommendation(
            type=self.type,
            target_type=target_type,
            target_id=target_id,
            details=details,
            confidence=confidence,
            source_analysis_id=source_analysis_id,
        )
