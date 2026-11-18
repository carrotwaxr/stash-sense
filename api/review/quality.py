"""scene_quality: review the quality flags #133 raised on a scene's primary file.

A recommendation (analyzers/scene_quality.py) holds the flags raised on one scene's file, with their evidence.
The card asks yes or no per raised flag (a control-sample card has no flags and asks about all four, default
no), so the owner's answers are labeled examples for calibrating the detectors.

Accept writes nothing unless report-only is off (``qualityReportOnly`` in the plugin settings, default on).
When it is off, each "yes" flag whose work-queue group is set adds the scene to that group; a scene already in
the group, or a flag with no group, changes nothing. The review only ever touches the scene's ``groups``.
"""

from __future__ import annotations

import curation_settings

from .plan import FieldChange, Plan, Preview, scene_field_values
from .types import InvalidOption, ReviewType, TargetMissing, register_review_type

TYPE_ID = "scene_quality"
FLAG_ORDER = ("broken", "rip", "below_studio", "format")
FLAG_LABELS = {
    "broken": "Broken",
    "rip": "Tube or leak rip",
    "below_studio": "Below the studio's release",
    "format": "Disfavored format",
}
CONTROL_SUMMARY = "Control sample: is anything wrong with this file?"
YES, NO = "yes", "no"


def _oshash(f: dict) -> str | None:
    return next((fp["value"] for fp in f.get("fingerprints") or [] if fp.get("type") == "oshash"), None)


@register_review_type
class QualityReview(ReviewType):
    type_id = TYPE_ID
    display_name = "Quality flags"
    description = "Confirm or reject each quality flag raised on a scene's file"

    async def preview(self, rec, option: dict | None, stash) -> Preview:
        details = rec.details or {}
        planned_file = details.get("file") or {}
        sid = str(rec.target_id)

        # 1. the scene and its primary file, as they are now
        scene = await stash.get_scene_for_review(sid)
        if not scene:
            raise TargetMissing(f"Scene {sid} no longer exists")
        files = scene.get("files") or []
        if not files:
            raise TargetMissing(f"Scene {sid} has no file left")
        oshash = _oshash(files[0])
        if oshash != planned_file.get("oshash") or (
                oshash is None and str(files[0]["id"]) != str(planned_file.get("id"))):
            raise TargetMissing(f"The file of scene {sid} changed since this card was made; "
                                "run the scene_quality job again")

        # 2. what the card asks: each raised flag, or all four on a control card
        raised = details.get("flags") or {}
        control = bool(details.get("control"))
        asked = [f for f in FLAG_ORDER if f in raised] if raised else (list(FLAG_ORDER) if control else [])
        default = {f: (YES if f in raised else NO) for f in asked}
        chosen = dict(default)
        for key, value in (option or {}).items():
            if key not in default:
                raise InvalidOption(f"{key!r} is not a flag this card asks about")
            if value not in (YES, NO):
                raise InvalidOption(f"{key}: choose yes or no, not {value!r}")
            chosen[key] = value

        # 3. the plan: add the scene to the group of each confirmed flag, only when report-only is off
        settings = (await curation_settings.current(stash)).settings
        groups = settings.quality_groups
        plan = Plan(rec_id=rec.id, rec_type=self.type_id, option=chosen)
        if not settings.quality_report_only:
            before = scene_field_values(scene)["groups"]
            after = [dict(g) for g in before]
            have = {str(g["group_id"]) for g in after}
            for flag in asked:
                gid = groups.get(flag)
                if chosen[flag] == YES and gid and str(gid) not in have:
                    after.append({"group_id": gid, "scene_index": None})
                    have.add(str(gid))
            if after != before:
                plan.field_changes.append(FieldChange("scene", sid, "groups", before, after))

        evidence = {
            "quality": raised,
            "scores": details.get("scores") or {},
            "control": control,
            "scene": {"id": sid, "title": scene.get("title") or ""},
            "file": planned_file,
            "report_only": settings.quality_report_only,
            "groups": groups,
        }
        summary = "; ".join(FLAG_LABELS[f] for f in FLAG_ORDER if f in raised) if raised else CONTROL_SUMMARY
        choices = [{"field": f, "label": FLAG_LABELS[f], "kind": "pick", "values": [YES, NO]} for f in asked]
        return Preview(plan=plan, evidence=evidence, choices=choices, default_option=default,
                       high_confidence=False, summary=summary)
