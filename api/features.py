"""Visibility of unfinished features.

A job type, recommendation type or plugin page that has not passed the
production-ready checklist (#123) stays hidden unless the
show_unfinished_features setting is on. The API rejects hidden job types,
and the plugin asks /system/features what to show.
"""

from __future__ import annotations

from job_models import JOB_REGISTRY

# Plugin page ids (the ?tab= values) that are not production-ready yet.
UNFINISHED_PAGES: frozenset[str] = frozenset()


class HiddenFeatureError(ValueError):
    """A request named a feature hidden behind show_unfinished_features."""


def show_unfinished() -> bool:
    from settings import get_setting
    try:
        return bool(get_setting("show_unfinished_features"))
    except RuntimeError:  # settings not initialized (tests, scripts)
        return False


def hidden_job_types() -> set[str]:
    if show_unfinished():
        return set()
    return {t for t, d in JOB_REGISTRY.items() if not d.production_ready}


def hidden_recommendation_types() -> set[str]:
    # Analysis jobs and the recommendations they produce share a type id.
    return hidden_job_types()


def hidden_pages() -> set[str]:
    return set() if show_unfinished() else set(UNFINISHED_PAGES)


def is_job_type_visible(type_id: str) -> bool:
    return type_id in JOB_REGISTRY and type_id not in hidden_job_types()


def require_visible_job_type(type_id: str) -> None:
    if type_id in hidden_job_types():
        raise HiddenFeatureError(
            f"Job type '{type_id}' is not production-ready yet; "
            "turn on Settings > Features > Show Unfinished Features to use it"
        )


def feature_state() -> dict:
    return {
        "show_unfinished": show_unfinished(),
        "hidden": {
            "job_types": sorted(hidden_job_types()),
            "recommendation_types": sorted(hidden_recommendation_types()),
            "pages": sorted(hidden_pages()),
        },
    }
