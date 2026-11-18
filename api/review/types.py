"""Review types: how a recommendation becomes a Plan.

A type reads Stash but never writes it. Register a type with
@register_review_type; its type_id must match the recommendation type
and an entry in job_models.JOB_REGISTRY (which carries production_ready).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from .plan import Preview

if TYPE_CHECKING:
    from recommendations_db import Recommendation


class TargetMissing(LookupError):
    """The recommendation's target is gone from Stash."""


class InvalidOption(ValueError):
    """The chosen option is not one the card offers."""


class ReviewType(ABC):
    type_id: str
    display_name: str
    description: str = ""

    @abstractmethod
    async def preview(self, rec: Recommendation, option: dict | None, stash) -> Preview:
        """Build the card for rec from Stash's current state.

        option holds the user's choices (possibly partial, or None); the type
        fills the rest from its suggestion and reports that suggestion as
        Preview.default_option. Raise TargetMissing or InvalidOption.
        """

    def scene_ids(self, rec: Recommendation) -> set[str]:
        """Scenes the recommendation touches (the exclusion list from #125 checks these)."""
        return {rec.target_id} if rec.target_type == "scene" else set()

    def label(self, rec: Recommendation, preview: Preview) -> dict | None:
        """A labeled example when the user chose something other than the suggestion."""
        if preview.is_default_option:
            return None
        return {"suggested": preview.default_option, "chosen": preview.plan.option}


REVIEW_TYPES: dict[str, ReviewType] = {}


def register_review_type(cls: type[ReviewType]) -> type[ReviewType]:
    REVIEW_TYPES[cls.type_id] = cls()
    return cls


def get_review_type(type_id: str) -> ReviewType | None:
    return REVIEW_TYPES.get(type_id)
