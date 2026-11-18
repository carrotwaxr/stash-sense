"""
Duplicate Scene Files Analyzer

Detects scenes that have multiple files attached, allowing the user
to choose which file(s) to keep and delete the rest.

Ported from: stash-plugins/scripts/scene-file-deduper/
"""

import curation_settings

from .base import BaseAnalyzer, AnalysisResult


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_duration(seconds: float | None) -> str:
    """Format duration in seconds as HH:MM:SS."""
    if seconds is None:
        return "Unknown"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def get_file_quality_score(file: dict) -> int:
    """
    Calculate a quality score for a file.

    Higher is better. Considers resolution and bitrate. Codec preference is
    taste, not fact (#125): the user's disfavored formats break ties instead.
    """
    score = 0

    # Resolution score (favor higher)
    width = file.get("width") or 0
    height = file.get("height") or 0
    pixels = width * height
    score += pixels // 10000  # Points per 10k pixels

    # Bitrate bonus
    bitrate = file.get("bit_rate") or 0
    score += bitrate // 1000000  # Points per Mbps

    return score


class DuplicateSceneFilesAnalyzer(BaseAnalyzer):
    """
    Finds scenes that have multiple files attached.

    This typically indicates duplicates or different quality versions
    of the same scene that the user should clean up.
    """

    type = "duplicate_scene_files"

    async def run(self, incremental: bool = True) -> AnalysisResult:
        """
        Run duplicate scene files analysis.

        Note: Incremental mode isn't well-supported here since we need
        to check all multi-file scenes. Always runs full scan.
        """
        # Fetch all scenes with more than one file
        scenes = await self.stash.get_multi_file_scenes()
        prefs = self.exclusion.settings if self.exclusion else curation_settings.CurationSettings()

        created = 0
        for scene in scenes:
            files = scene.get("files", [])
            if len(files) < 2:
                continue  # Shouldn't happen with our filter, but be safe

            # Score each file
            scored_files = []
            for f in files:
                score = get_file_quality_score(f)
                scored_files.append({
                    "id": f["id"],
                    "path": f["path"],
                    "basename": f.get("basename", f["path"].split("/")[-1]),
                    "size": f["size"],
                    "size_formatted": format_size(f["size"]),
                    "duration": f.get("duration"),
                    "duration_formatted": format_duration(f.get("duration")),
                    "video_codec": f.get("video_codec"),
                    "audio_codec": f.get("audio_codec"),
                    "format": f.get("format"),
                    "disfavored": prefs.is_disfavored(f),
                    "width": f.get("width"),
                    "height": f.get("height"),
                    "resolution": f"{f.get('width', '?')}x{f.get('height', '?')}",
                    "frame_rate": f.get("frame_rate"),
                    "bit_rate": f.get("bit_rate"),
                    "quality_score": score,
                })

            # Best first; on a tie, a file in a disfavored format goes after one that isn't.
            scored_files.sort(key=lambda x: (-x["quality_score"], x["disfavored"]))

            # Mark suggested keeper
            for i, f in enumerate(scored_files):
                f["is_suggested_keeper"] = i == 0

            rec_id = self.create_recommendation(
                target_type="scene",
                target_id=scene["id"],
                details={
                    "scene_title": scene.get("title") or f"Scene {scene['id']}",
                    "files": scored_files,
                    "suggested_keeper_id": scored_files[0]["id"],
                    "total_size": sum(f["size"] for f in scored_files),
                    "total_size_formatted": format_size(sum(f["size"] for f in scored_files)),
                    "potential_savings": sum(f["size"] for f in scored_files[1:]),
                    "potential_savings_formatted": format_size(sum(f["size"] for f in scored_files[1:])),
                    "performers": [
                        {"id": p["id"], "name": p["name"]}
                        for p in scene.get("performers", [])
                    ],
                    "studio": scene.get("studio"),
                },
                confidence=1.0,  # Deterministic - multiple files is certain
            )

            if rec_id:
                created += 1

        return AnalysisResult(
            items_processed=len(scenes),
            recommendations_created=created,
        )
