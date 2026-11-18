"""#125: shared curation settings stored in Stash (exclusion list, disfavored formats).

Runs against FakeStash, which models Stash's tag/group/studio hierarchies
and configurePlugin's replace-the-whole-map behaviour.
"""

from __future__ import annotations

import inspect

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import curation_router
import curation_settings
import features
import recommendations_router as rec_mod
from analyzers.duplicate_scene_files import DuplicateSceneFilesAnalyzer
from curation_settings import (
    PLUGIN_ID,
    CurationSettings,
    ExclusionCache,
    ExclusionUnavailable,
    SceneExcluded,
    rec_scene_ids,
)
from recommendations_db import RecommendationsDB
from review.demo import DEMO_TAG, DemoReviewAnalyzer
from review.service import Excluded, ReviewService
from review.store import ReviewStore
from stash_client_unified import StashClientUnified
from tests.fake_stash import FakeStash, FakeStashError

GB = 1024 ** 3


@pytest.fixture
def stash():
    return FakeStash()


@pytest.fixture
def real_cache(monkeypatch):
    """Undo conftest's empty list: resolve the exclusion list from the fake Stash."""
    cache = ExclusionCache()
    monkeypatch.setattr(curation_settings, "_cache", cache)
    return cache


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "stash_sense.db")


def set_settings(stash, **values):
    """Write settings the way the plugin does: read, merge, write the whole map."""
    merged = dict(stash.plugin_config.get(PLUGIN_ID) or {})
    merged.update(values)
    stash.plugin_config[PLUGIN_ID] = merged
    curation_settings.invalidate()


@pytest.fixture
def library(stash):
    """Scenes under a tag tree, a group tree and a studio tree, plus one untouched scene."""
    lib = {}
    lib["private"] = stash.add_tag("Private")
    lib["private_child"] = stash.add_tag("Private / Family", parent=lib["private"])
    lib["private_grandchild"] = stash.add_tag("Private / Family / 2019", parent=lib["private_child"])
    lib["series"] = stash.add_group("Home videos")
    lib["series_part"] = stash.add_group("Home videos, part 2", parent=lib["series"])
    lib["network"] = stash.add_studio("Network")
    lib["network_sub"] = stash.add_studio("Network Sub", parent=lib["network"])

    def scene(**kw):
        return stash.add_scene(stash.add_file(f"/m/{len(stash.scenes)}.mp4", GB), **kw)

    lib["s_tag"] = scene(tag_ids=(lib["private_grandchild"],))
    lib["s_group"] = scene(group_ids=[lib["series_part"]])
    lib["s_studio"] = scene(studio_id=lib["network_sub"])
    lib["s_free"] = scene(title="free")
    return lib


# ---------------- fake fidelity ----------------

@pytest.mark.parametrize("name", ["get_plugin_configuration", "find_scene_ids"])
def test_fake_matches_real_client_signature(name):
    real, fake = getattr(StashClientUnified, name), getattr(FakeStash, name)
    assert inspect.iscoroutinefunction(real) and inspect.iscoroutinefunction(fake)
    assert list(inspect.signature(real).parameters)[1:] == list(inspect.signature(fake).parameters)[1:]


async def test_fake_configure_plugin_replaces_whole_map_like_stash(stash):
    await stash.configure_plugin(PLUGIN_ID, {"sidecarUrl": "http://x", "excludeTagIds": "1"})
    await stash.configure_plugin(PLUGIN_ID, {"excludeTagIds": "2"})
    assert await stash.get_plugin_configuration(PLUGIN_ID) == {"excludeTagIds": "2"}


# ---------------- parsing ----------------

def test_parse_comma_strings_trims_dedupes_and_lowercases_formats():
    s = CurationSettings.from_plugin_config({
        "excludeTagIds": " 3, 4,,3 ", "excludeGroupIds": "", "excludeStudioIds": None,
        "disfavoredVideoCodecs": "HEVC, av1", "disfavoredContainers": "WMV",
        "sidecarUrl": "http://ignored",
    })
    assert s.exclude_tag_ids == ("3", "4")
    assert s.exclude_group_ids == () and s.exclude_studio_ids == ()
    assert s.disfavored_video_codecs == ("hevc", "av1")
    assert s.disfavored_containers == ("wmv",)


def test_parse_accepts_lists_from_other_writers():
    s = CurationSettings.from_plugin_config({"excludeTagIds": [5, "6"]})
    assert s.exclude_tag_ids == ("5", "6")


def test_missing_settings_mean_nothing_excluded_and_no_preference():
    s = CurationSettings.from_plugin_config(None)
    assert not s.has_exclusions
    assert not s.is_disfavored({"video_codec": "hevc", "format": "mkv"})


def test_is_disfavored_by_codec_or_container():
    s = CurationSettings(disfavored_video_codecs=("hevc",), disfavored_containers=("wmv",))
    assert s.is_disfavored({"video_codec": "HEVC", "format": "mp4"})
    assert s.is_disfavored({"video_codec": "h264", "format": "wmv"})
    assert not s.is_disfavored({"video_codec": "h264", "format": "mp4"})
    assert not s.is_disfavored({"video_codec": None, "format": None})


# ---------------- quality flags (#133) ----------------

def test_quality_defaults():
    s = CurationSettings.from_plugin_config({})
    assert s.quality_report_only is True
    assert s.quality_groups == {}
    assert s.quality_peer_percentile == 10.0


def test_quality_values_parsed():
    s = CurationSettings.from_plugin_config({
        "qualityReportOnly": False, "qualityGroupBroken": "12",
        "qualityGroupRip": " 13 ", "qualityPeerPercentile": "25",
    })
    assert s.quality_report_only is False
    assert s.quality_groups == {"broken": "12", "rip": "13"}
    assert s.quality_peer_percentile == 25.0
    for off in ("false", "0"):
        assert CurationSettings.from_plugin_config({"qualityReportOnly": off}).quality_report_only is False
    assert CurationSettings.from_plugin_config({"qualityReportOnly": True}).quality_report_only is True


def test_percentile_is_clamped():
    def pct(v):
        return CurationSettings.from_plugin_config({"qualityPeerPercentile": v}).quality_peer_percentile

    assert pct("0") == 1.0
    assert pct("90") == 50.0
    assert pct("x") == 10.0


def test_to_dict_includes_quality():
    d = CurationSettings.from_plugin_config(
        {"excludeTagIds": "3", "qualityGroupFormat": "7", "qualityReportOnly": "false"}
    ).to_dict()
    assert d["exclude_tag_ids"] == ["3"]
    assert d["quality_report_only"] is False
    assert d["quality_peer_percentile"] == 10.0
    assert d["quality_group_ids"] == [["format", "7"]]


# ---------------- resolution ----------------

async def test_descendants_of_excluded_tag_group_and_studio_are_excluded(stash, library, real_cache):
    set_settings(stash, excludeTagIds=library["private"], excludeGroupIds=library["series"],
                 excludeStudioIds=library["network"])
    ex = await curation_settings.current(stash)
    assert ex.scene_ids == {library["s_tag"], library["s_group"], library["s_studio"]}
    assert not ex.is_excluded(library["s_free"])


async def test_empty_list_excludes_nothing_and_asks_stash_for_no_scenes(stash, library, real_cache):
    stash.fail["find_scene_ids"] = AssertionError("should not query scenes")
    ex = await curation_settings.current(stash)
    assert ex.scene_ids == frozenset()


async def test_unreadable_stash_fails_closed_and_is_not_cached(stash, library, real_cache):
    stash.fail["get_plugin_configuration"] = FakeStashError("connection refused")
    with pytest.raises(ExclusionUnavailable):
        await curation_settings.current(stash)
    set_settings(stash, excludeTagIds=library["private"])
    assert library["s_tag"] in (await curation_settings.current(stash)).scene_ids


async def test_cache_serves_until_ttl_or_invalidate(stash, library):
    now = [0.0]
    cache = ExclusionCache(ttl=60, clock=lambda: now[0])
    assert (await cache.get(stash)).scene_ids == frozenset()
    stash.plugin_config[PLUGIN_ID] = {"excludeTagIds": library["private"]}
    now[0] = 30
    assert (await cache.get(stash)).scene_ids == frozenset()  # still cached
    assert library["s_tag"] in (await cache.get(stash, max_age=0)).scene_ids  # a job's fresh read
    stash.plugin_config[PLUGIN_ID] = {}
    cache.invalidate()
    assert (await cache.get(stash)).scene_ids == frozenset()


def test_scene_excluded_message_names_scenes():
    e = SceneExcluded({"12", "3"})
    assert e.scene_ids == ["3", "12"]
    assert "3, 12" in str(e) and "exclusion list" in str(e)


# ---------------- recommendations: Python and SQL agree ----------------

REC_SHAPES = [
    # (type, target_type, target_id, details)
    ("duplicate_scene_files", "scene", "10", {}),
    ("duplicate_scenes", "scene", "11:12", {"scene_a_id": "11", "scene_b_id": "12"}),
    ("duplicate_scenes", "scene", "13:14", {"scene_a_id": 13, "scene_b_id": 14}),  # ints in JSON
    ("scene_fingerprint_match", "scene", "15|https://stashdb.org/graphql|uuid",
     {"local_scene_id": "15", "stashbox_scene_id": "uuid"}),
    ("upstream_scene_changes", "scene", "16", {"scene_id": "16"}),
    ("duplicate_performer", "performer", "10", {"performers": []}),  # performer 10, not scene 10
]


@pytest.mark.parametrize("excluded", [set(), {"10"}, {"12"}, {"14"}, {"15"}, {"16"}, {"10", "11", "16"}])
def test_sql_filter_matches_rec_scene_ids(db, excluded):
    for t, tt, tid, d in REC_SHAPES:
        db.create_recommendation(type=t, target_type=tt, target_id=tid, details=d)
    expected = {
        (t, tid) for t, tt, tid, d in REC_SHAPES if not rec_scene_ids(tt, tid, d) & excluded
    }
    got = {(r.type, r.target_id) for r in db.get_recommendations(exclude_scene_ids=excluded)}
    assert got == expected
    assert db.count_recommendations(exclude_scene_ids=excluded) == len(expected)
    counts = db.get_recommendation_counts(exclude_scene_ids=excluded)
    assert sum(c.get("pending", 0) for c in counts.values()) == len(expected)


def test_rec_scene_ids_shapes():
    assert rec_scene_ids("scene", "11:12", {"scene_a_id": 11, "scene_b_id": "12"}) >= {"11", "12"}
    assert rec_scene_ids("performer", "10", {}) == set()


def test_scene_ids_list_key():
    """#130: a scene_set rec names its members in details.scene_ids; Python and SQL agree."""
    import json
    import sqlite3

    from curation_settings import scene_exclusion_sql

    recs = [
        (1, "scene_set", "set-a", {"scene_ids": ["5", "9"]}),
        (2, "scene_set", "set-b", {"scene_ids": ["5", "6"]}),
        (3, "scene_set", "set-c", {"scene_ids": [7, 9]}),  # ints in JSON
        (4, "scene_set", "set-d", {}),
        (5, "scene_set", "set-e", {"scene_ids": None}),
        (6, "scene_set", "set-f", {"scene_ids": "9"}),  # not a list
    ]
    excluded = {"9"}
    expected = {2, 4, 5, 6}

    assert {r[0] for r in recs if not rec_scene_ids(r[1], r[2], r[3]) & excluded} == expected
    assert rec_scene_ids("scene_set", "x", {"scene_ids": ["5", 9]}) == {"5", "9"}

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE recommendations (id INTEGER, target_type TEXT, target_id TEXT, details TEXT)")
    conn.executemany(
        "INSERT INTO recommendations VALUES (?, ?, ?, ?)",
        [(i, tt, tid, json.dumps(d)) for i, tt, tid, d in recs],
    )
    conn.execute("INSERT INTO recommendations VALUES (7, 'scene_set', 'set-g', NULL)")
    cond, params = scene_exclusion_sql(excluded)
    got = {r[0] for r in conn.execute(f"SELECT id FROM recommendations WHERE {cond}", params)}
    assert got == expected | {7}


# ---------------- analyzers ----------------

async def test_analyzer_skips_scenes_with_excluded_child_tag(stash, db, library, real_cache):
    demo_tag = stash.add_tag(DEMO_TAG)
    for key in ("s_tag", "s_free"):
        stash.scenes[library[key]]["tag_ids"].append(demo_tag)
        stash.scenes[library[key]]["file_ids"].append(stash.add_file(f"/m/{key}.copy.mp4", 2 * GB))
    set_settings(stash, excludeTagIds=library["private"])

    analyzer = DemoReviewAnalyzer(stash, db)
    analyzer.exclusion = await curation_settings.current(stash, max_age=0)
    await analyzer.run()

    targets = {r.target_id for r in db.get_recommendations()}
    assert targets == {library["s_free"]}
    assert analyzer.excluded_skipped == 1


async def test_analyzer_without_a_loaded_list_refuses_scene_recs(stash, db, monkeypatch):
    from analyzers.base import BaseAnalyzer
    monkeypatch.setattr(BaseAnalyzer, "exclusion", None)
    analyzer = DemoReviewAnalyzer(stash, db)
    with pytest.raises(RuntimeError, match="exclusion list"):
        analyzer.create_recommendation("scene", "1", {})
    # Recommendations that name no scene don't need it.
    assert analyzer.create_recommendation("performer", "1", {}) is not None


async def test_analysis_job_loads_the_list_fresh(stash, db, library, real_cache, monkeypatch):
    from base_job import JobContext
    from jobs.analysis_jobs import AnalysisJob

    monkeypatch.setattr(rec_mod, "rec_db", db)
    monkeypatch.setattr(rec_mod, "stash_client", stash)
    demo_tag = stash.add_tag(DEMO_TAG)
    sid = library["s_tag"]
    stash.scenes[sid]["tag_ids"].append(demo_tag)
    stash.scenes[sid]["file_ids"].append(stash.add_file("/m/copy.mp4", 2 * GB))
    await curation_settings.current(stash)  # cache an empty list...
    stash.plugin_config[PLUGIN_ID] = {"excludeTagIds": library["private"]}  # ...then exclude, no invalidate

    ctx = JobContext(job_id=1, db=db, queue_manager=None)
    await AnalysisJob("review_demo").run(ctx)
    assert db.get_recommendations() == []


# ---------------- keeper suggestion (duplicate_scene_files) ----------------

def _two_file_scene(stash, *, codecs=("hevc", "h264"), formats=("mp4", "mp4")):
    files = []
    for i, (codec, fmt) in enumerate(zip(codecs, formats)):
        fid = stash.add_file(f"/m/k{i}.{fmt}", GB, codec=codec)
        stash.files[fid]["format"] = fmt
        files.append(fid)
    return stash.add_scene(*files), files


class _MultiFileStash(FakeStash):
    async def get_multi_file_scenes(self, exclude_tag_ids=None):
        return [self._render(s) for s in self.scenes.values() if len(s["file_ids"]) > 1]


async def _keeper(stash, db, settings: CurationSettings):
    a = DuplicateSceneFilesAnalyzer(stash, db)
    a.exclusion = curation_settings.Exclusion(settings, frozenset(), 0.0)
    await a.run()
    [rec] = db.get_recommendations()
    return rec.details


async def test_no_codec_preference_without_settings(db):
    stash = _MultiFileStash()
    _, (hevc, _h264) = _two_file_scene(stash)
    details = await _keeper(stash, db, CurationSettings())
    # Same resolution and bitrate: a tie, so the first file stays first; HEVC earns nothing.
    assert details["suggested_keeper_id"] == hevc
    assert {f["quality_score"] for f in details["files"]} == {details["files"][0]["quality_score"]}
    assert not any(f["disfavored"] for f in details["files"])


async def test_disfavored_codec_breaks_a_tie(db):
    stash = _MultiFileStash()
    _, (_hevc, h264) = _two_file_scene(stash)
    details = await _keeper(stash, db, CurationSettings(disfavored_video_codecs=("hevc",)))
    assert details["suggested_keeper_id"] == h264
    assert [f["disfavored"] for f in details["files"]] == [False, True]


async def test_disfavored_container_breaks_a_tie_but_not_a_quality_gap(db):
    stash = _MultiFileStash()
    _, (wmv, _mp4) = _two_file_scene(stash, codecs=("h264", "h264"), formats=("wmv", "mp4"))
    stash.files[wmv]["width"], stash.files[wmv]["height"] = 3840, 2160  # the wmv is clearly better
    details = await _keeper(stash, db, CurationSettings(disfavored_containers=("wmv",)))
    assert details["suggested_keeper_id"] == wmv


# ---------------- review framework ----------------

@pytest.fixture
def svc(db, stash, tmp_path):
    return ReviewService(db, ReviewStore(db.db_path), stash)


async def _demo_rec(stash, db, library):
    demo_tag = stash.add_tag(DEMO_TAG)
    sid = library["s_tag"]
    stash.scenes[sid]["tag_ids"].append(demo_tag)
    stash.scenes[sid]["file_ids"].append(stash.add_file("/m/copy.mp4", 2 * GB))
    await DemoReviewAnalyzer(stash, db).run()  # conftest: empty list while the rec is created
    return db.get_recommendation_by_target("review_demo", "scene", sid)


async def test_review_refuses_preview_accept_and_batch_for_excluded_scene(
    stash, db, svc, library, real_cache, monkeypatch,
):
    monkeypatch.setattr(features, "show_unfinished", lambda: True)
    rec = await _demo_rec(stash, db, library)
    p = await svc.preview(rec.id)
    set_settings(stash, excludeTagIds=library["private"])

    with pytest.raises(Excluded) as e:
        await svc.preview(rec.id)
    assert e.value.status == 403 and e.value.payload["scene_ids"] == [library["s_tag"]]
    with pytest.raises(Excluded):
        await svc.accept(rec.id, option=None, plan_token=p["plan_token"], confirm_delete=[])
    assert (await svc.batch_preview("review_demo"))["items"] == []
    assert stash.writes == 0


async def test_review_fails_closed_when_stash_unreadable(stash, db, svc, library, real_cache, monkeypatch):
    monkeypatch.setattr(features, "show_unfinished", lambda: True)
    rec = await _demo_rec(stash, db, library)
    stash.fail["get_plugin_configuration"] = FakeStashError("down")
    with pytest.raises(ExclusionUnavailable):
        await svc.preview(rec.id)


async def test_revert_of_a_now_excluded_scene_is_refused(stash, db, svc, library, real_cache, monkeypatch):
    monkeypatch.setattr(features, "show_unfinished", lambda: True)
    rec = await _demo_rec(stash, db, library)
    p = await svc.preview(rec.id)
    out = await svc.accept(rec.id, option=None, plan_token=p["plan_token"],
                           confirm_delete=[f["file_id"] for f in p["plan"]["delete_files"]])
    set_settings(stash, excludeTagIds=library["private"])
    writes = stash.writes
    with pytest.raises(Excluded):
        await svc.revert(out["entry"]["id"])
    assert stash.writes == writes


# ---------------- API ----------------

@pytest.fixture
def client(db, stash, real_cache, monkeypatch):
    monkeypatch.setattr(rec_mod, "rec_db", db)
    monkeypatch.setattr(rec_mod, "stash_client", stash)
    app = FastAPI()
    app.include_router(rec_mod.router)
    curation_router.install(app)
    return TestClient(app)


def test_listing_and_counts_hide_recs_on_excluded_scenes(client, db, stash, library, monkeypatch):
    monkeypatch.setattr(features, "show_unfinished", lambda: True)
    db.create_recommendation(type="duplicate_scene_files", target_type="scene", target_id=library["s_tag"],
                             details={})
    db.create_recommendation(type="duplicate_scenes", target_type="scene",
                             target_id=f"{library['s_free']}:{library['s_group']}",
                             details={"scene_a_id": library["s_free"], "scene_b_id": library["s_group"]})
    keep = db.create_recommendation(type="duplicate_scene_files", target_type="scene",
                                    target_id=library["s_free"], details={})
    set_settings(stash, excludeTagIds=library["private"], excludeGroupIds=library["series"])

    listing = client.get("/recommendations").json()
    assert [r["id"] for r in listing["recommendations"]] == [keep] and listing["total"] == 1
    assert client.get("/recommendations/counts").json()["total_pending"] == 1


def test_endpoints_naming_an_excluded_scene_are_refused(client, stash, library):
    set_settings(stash, excludeStudioIds=library["network"])
    sid = library["s_studio"]
    calls = [
        ("get", f"/recommendations/scene/{sid}", None),
        ("post", "/recommendations/actions/delete-scene", {"scene_id": sid}),
        ("post", "/recommendations/actions/merge-scenes",
         {"destination_id": library["s_free"], "source_ids": [sid]}),
        ("post", "/recommendations/actions/delete-scene-files",
         {"scene_id": sid, "file_ids_to_delete": ["1"], "keep_file_id": "2", "all_file_ids": ["1", "2"]}),
        ("post", "/recommendations/actions/update-scene", {"scene_id": sid, "fields": {"title": "x"}}),
    ]
    for method, path, body in calls:
        r = client.request(method.upper(), path, json=body)
        assert r.status_code == 403, (path, r.status_code, r.text)
        assert r.json()["error"] == "scene_excluded" and r.json()["scene_ids"] == [sid]
    assert stash.writes == 0


def test_stash_unreachable_is_503_not_an_unfiltered_list(client, stash):
    stash.fail["get_plugin_configuration"] = FakeStashError("down")
    r = client.get("/recommendations")
    assert r.status_code == 503 and r.json()["error"] == "exclusion_unavailable"


def test_settings_round_trip_through_stash(client, stash, library):
    set_settings(stash, sidecarUrl="http://sidecar:5000")
    assert client.get("/curation/settings").json()["excluded_scene_count"] == 0

    # The plugin saves with read-merge-write, then asks the sidecar to refresh.
    merged = dict(stash.plugin_config[PLUGIN_ID], excludeTagIds=library["private"],
                  disfavoredVideoCodecs="hevc")
    stash.plugin_config[PLUGIN_ID] = merged
    body = client.post("/curation/refresh").json()
    assert body["error"] is None and body["excluded_scene_count"] == 1
    assert body["settings"]["exclude_tag_ids"] == [library["private"]]
    assert body["settings"]["disfavored_video_codecs"] == ["hevc"]
    assert stash.plugin_config[PLUGIN_ID]["sidecarUrl"] == "http://sidecar:5000"


def test_curation_settings_reports_stash_error(client, stash):
    stash.fail["get_plugin_configuration"] = FakeStashError("down")
    body = client.get("/curation/settings").json()
    assert body["settings"] is None and "down" in body["error"]


# ---------------- review follow-ups ----------------

def test_non_numeric_ids_are_dropped():
    # Stash interpolates hierarchical filter values into SQL; a non-numeric id fails the whole query.
    s = CurationSettings.from_plugin_config({"excludeTagIds": "5, abc, 7x, 9", "excludeStudioIds": "-1"})
    assert s.exclude_tag_ids == ("5", "9")
    assert s.exclude_studio_ids == ()


async def test_invalidate_during_a_read_does_not_cache_the_old_list(stash, library):
    cache = ExclusionCache()
    real = stash.get_plugin_configuration

    async def slow_read(plugin_id):
        out = await real(plugin_id)
        # The user saves a new list while this read is in flight.
        stash.plugin_config[PLUGIN_ID] = {"excludeTagIds": library["private"]}
        cache.invalidate()
        return out

    stash.get_plugin_configuration = slow_read
    assert (await cache.get(stash)).scene_ids == frozenset()  # the in-flight caller sees its read
    stash.get_plugin_configuration = real
    assert library["s_tag"] in (await cache.get(stash)).scene_ids  # but it wasn't cached


async def test_listings_may_use_the_last_good_list_but_writes_may_not(stash, library):
    cache = ExclusionCache()
    stash.plugin_config[PLUGIN_ID] = {"excludeTagIds": library["private"]}
    await cache.get(stash)
    stash.fail["get_plugin_configuration"] = FakeStashError("down")
    stale = await cache.get(stash, max_age=0, allow_stale=True)
    assert library["s_tag"] in stale.scene_ids
    stash.fail["get_plugin_configuration"] = FakeStashError("down")
    with pytest.raises(ExclusionUnavailable):
        await cache.get(stash, max_age=0)


def test_write_guard_reads_the_list_fresh(client, stash, library):
    assert client.get("/recommendations").status_code == 200  # caches an empty list
    stash.plugin_config[PLUGIN_ID] = {"excludeTagIds": library["private"]}  # no refresh call
    r = client.post("/recommendations/actions/delete-scene", json={"scene_id": library["s_tag"]})
    assert r.status_code == 403


def test_merge_performers_refused_when_a_source_is_in_an_excluded_scene(client, stash, library):
    stash.scenes[library["s_tag"]]["performer_ids"] = ["900"]
    set_settings(stash, excludeTagIds=library["private"])
    r = client.post("/recommendations/actions/merge-performers",
                    json={"destination_id": "901", "source_ids": ["900"]})
    assert r.status_code == 403 and r.json()["scene_ids"] == [library["s_tag"]]


async def test_review_refuses_a_plan_that_writes_an_excluded_scene_the_rec_does_not_name(
    stash, db, svc, library, real_cache, monkeypatch,
):
    from review import types as review_types
    from review.plan import FieldChange, Plan, Preview

    class _WritesOtherScene(review_types.ReviewType):
        type_id = "review_demo"  # registered, production_ready handled by show_unfinished
        display_name = "x"

        async def preview(self, rec, option, stash_):
            plan = Plan(rec_id=rec.id, rec_type=rec.type, option={})
            plan.field_changes = [FieldChange("scene", library["s_tag"], "title", "a", "b")]
            return Preview(plan=plan, summary="", evidence={}, choices={}, default_option={},
                           high_confidence=True)

    monkeypatch.setattr(features, "show_unfinished", lambda: True)
    monkeypatch.setitem(review_types.REVIEW_TYPES, "review_demo", _WritesOtherScene())
    rec_id = db.create_recommendation(type="review_demo", target_type="scene",
                                      target_id=library["s_free"], details={})
    set_settings(stash, excludeTagIds=library["private"])
    with pytest.raises(Excluded) as e:
        await svc.preview(rec_id)
    assert e.value.payload["scene_ids"] == [library["s_tag"]]


async def test_fingerprint_match_never_sends_excluded_scenes_to_stashbox():
    from unittest.mock import AsyncMock, patch

    from analyzers.scene_fingerprint_match import SceneFingerprintMatchAnalyzer
    from tests.test_scene_fingerprint_match_analyzer import (
        make_mock_db,
        make_mock_stash,
        make_scene,
    )

    stash = make_mock_stash()
    stash.get_scenes_with_fingerprints.return_value = (
        [make_scene("42", "excluded", [{"type": "md5", "value": "secret"}]),
         make_scene("43", "free", [{"type": "md5", "value": "fine"}])],
        2,
    )
    with patch("analyzers.scene_fingerprint_match.StashBoxClient") as sbc:
        sbc.return_value = AsyncMock()
        sbc.return_value.find_scenes_by_fingerprints.return_value = [[]]
        a = SceneFingerprintMatchAnalyzer(stash, make_mock_db())
        a.exclusion = curation_settings.Exclusion(CurationSettings(), frozenset({"42"}), 0.0)
        await a.run(incremental=False)
    sent = [fp["hash"] for call in sbc.return_value.find_scenes_by_fingerprints.call_args_list
            for fps in call.args[0] for fp in fps]
    assert "secret" not in sent and "fine" in sent
    assert a.excluded_skipped >= 1


async def test_scenes_leaving_the_list_trigger_one_full_run(stash, db, library, real_cache, monkeypatch):
    from analyzers.base import AnalysisResult, BaseAnalyzer
    from base_job import JobContext
    from jobs.analysis_jobs import AnalysisJob

    runs = []

    class _Recorder(BaseAnalyzer):
        type = "recorder"

        async def run(self, incremental=True):
            runs.append(incremental)
            return AnalysisResult(items_processed=0, recommendations_created=0)

    monkeypatch.setattr(rec_mod, "rec_db", db)
    monkeypatch.setattr(rec_mod, "stash_client", stash)
    monkeypatch.setitem(rec_mod.ANALYZERS, "recorder", _Recorder)
    job = AnalysisJob("recorder")
    ctx = JobContext(job_id=1, db=db, queue_manager=None)

    set_settings(stash, excludeTagIds=library["private"])
    await job.run(ctx)                      # first run: nothing released
    await job.run(ctx)                      # same list
    set_settings(stash, excludeTagIds="")   # s_tag released
    await job.run(ctx)
    await job.run(ctx)                      # back to incremental
    assert runs == [True, True, False, True]
