"""Unfinished features stay hidden, and media-writing jobs refuse a read-only mount."""

import asyncio
import dataclasses

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import media_library
import recommendations_router as rec_mod
import settings as settings_mod
from hardware import HardwareProfile
from job_models import JOB_REGISTRY, JobDefinition, JobPriority, ResourceType
from recommendations_db import RecommendationsDB

UNFINISHED = "duplicate_scenes"  # an analyzer type, so recommendations are covered too


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(str(tmp_path / "test.db"))


@pytest.fixture
def settings(db):
    original = settings_mod._settings_manager
    mgr = settings_mod.init_settings(db, "gpu-high")
    yield mgr
    settings_mod._settings_manager = original


@pytest.fixture
def unfinished(monkeypatch):
    """Mark one existing job type as not production-ready and add a media-writing type."""
    monkeypatch.setitem(
        JOB_REGISTRY, UNFINISHED,
        dataclasses.replace(JOB_REGISTRY[UNFINISHED], production_ready=False),
    )
    monkeypatch.setitem(JOB_REGISTRY, "test_write_subs", JobDefinition(
        type_id="test_write_subs", display_name="Subtitles", description="test",
        resource=ResourceType.CPU_HEAVY, default_priority=JobPriority.NORMAL,
        supports_incremental=False, schedulable=True, default_interval_hours=24,
        writes_media=True,
    ))


@pytest.fixture
def media(tmp_path, monkeypatch):
    """Point the media library at a writable temp dir; tests may swap it."""
    root = tmp_path / "media"
    root.mkdir()
    lib = media_library.MediaLibrary([], media_dir=str(root))
    monkeypatch.setattr(media_library, "_library", lib)
    return lib


@pytest.fixture
def client(db, settings, unfinished, media, monkeypatch):
    import hardware
    from database_health_router import router as health_router
    from queue_manager import QueueManager
    from queue_router import init_queue_router
    from queue_router import router as queue_router
    from settings_router import init_settings_router
    from settings_router import router as settings_router

    monkeypatch.setattr(hardware, "_hardware_profile", HardwareProfile(
        gpu_available=False, gpu_name=None, gpu_vram_mb=None, cpu_cores=4,
        memory_total_mb=8192, memory_available_mb=4096, storage_free_mb=1000, tier="cpu",
    ))
    monkeypatch.setattr(rec_mod, "rec_db", db)
    from tests.fake_stash import FakeStash
    monkeypatch.setattr(rec_mod, "stash_client", FakeStash())  # listings read the exclusion list
    mgr = QueueManager(db)
    mgr.seed_default_schedules()
    init_queue_router(mgr)
    init_settings_router()

    import queue_router as queue_mod
    app = FastAPI()
    for r in (queue_router, rec_mod.router, settings_router, health_router):
        app.include_router(r)
    yield TestClient(app)
    queue_mod._queue_manager = None


def _type_ids(client):
    return {t["type_id"] for t in client.get("/queue/types").json()["types"]}


class TestHiddenByDefault:
    def test_setting_defaults_off(self, settings):
        assert settings.get("show_unfinished_features") is False

    def test_job_type_absent_from_operations_types(self, client):
        ids = _type_ids(client)
        assert UNFINISHED not in ids
        assert "duplicate_performer" in ids

    def test_submit_rejected(self, client):
        resp = client.post("/queue", json={"type": UNFINISHED})
        assert resp.status_code == 400
        assert "Show Unfinished Features" in resp.json()["detail"]

    def test_schedule_hidden_and_not_editable(self, client):
        types = {s["type"] for s in client.get("/queue/schedules").json()["schedules"]}
        assert UNFINISHED not in types
        resp = client.put(f"/queue/schedules/{UNFINISHED}", json={"enabled": True, "interval_hours": 24})
        assert resp.status_code == 404

    def test_due_schedule_not_submitted(self, client, db):
        db.upsert_job_schedule(type=UNFINISHED, enabled=True, interval_hours=1, priority=50)
        import queue_router as queue_mod
        queue_mod._queue_manager._check_schedules()
        assert db.get_jobs(type=UNFINISHED) == []

    def test_analysis_run_rejected(self, client):
        resp = client.post(f"/recommendations/analysis/{UNFINISHED}/run")
        assert resp.status_code == 400
        analysis_types = {t["type"] for t in client.get("/recommendations/analysis/types").json()["types"]}
        assert UNFINISHED not in analysis_types

    def test_recommendations_hidden(self, client, db):
        hidden_id = db.create_recommendation(UNFINISHED, "scene", "1", {"x": 1})
        db.create_recommendation("duplicate_performer", "performer", "2", {"x": 2})

        listing = client.get("/recommendations").json()
        assert [r["type"] for r in listing["recommendations"]] == ["duplicate_performer"]
        assert listing["total"] == 1
        assert client.get(f"/recommendations?type={UNFINISHED}").json()["total"] == 0
        assert UNFINISHED not in client.get("/recommendations/counts").json()["counts"]
        assert client.get(f"/recommendations/{hidden_id}").status_code == 404

    def test_features_endpoint(self, client):
        state = client.get("/system/features").json()
        assert state["show_unfinished"] is False
        assert UNFINISHED in state["hidden"]["job_types"]
        assert UNFINISHED in state["hidden"]["recommendation_types"]


class TestShownWhenSettingOn:
    def test_everything_visible(self, client, settings, db):
        settings.set("show_unfinished_features", True)
        assert UNFINISHED in _type_ids(client)
        assert client.post("/queue", json={"type": UNFINISHED}).status_code == 200
        db.create_recommendation(UNFINISHED, "scene", "1", {"x": 1})
        assert client.get("/recommendations").json()["total"] == 1
        assert client.get("/system/features").json()["hidden"]["job_types"] == []


class TestMediaWrites:
    def test_writes_media_flag_exposed(self, client):
        types = {t["type_id"]: t for t in client.get("/queue/types").json()["types"]}
        assert types["test_write_subs"]["writes_media"] is True
        assert types["duplicate_performer"]["writes_media"] is False

    def test_submit_ok_when_writable(self, client):
        assert client.post("/queue", json={"type": "test_write_subs"}).status_code == 200

    def test_submit_refused_when_mount_read_only(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(media_library, "_library",
                            media_library.MediaLibrary([], media_dir=str(tmp_path / "absent")))
        resp = client.post("/queue", json={"type": "test_write_subs"})
        assert resp.status_code == 400
        assert resp.json()["detail"].startswith("Subtitles writes files into the media library")

    def test_analysis_run_on_read_only_mount_is_400(self, client, monkeypatch, tmp_path):
        monkeypatch.setitem(
            JOB_REGISTRY, "duplicate_performer",
            dataclasses.replace(JOB_REGISTRY["duplicate_performer"], writes_media=True),
        )
        monkeypatch.setattr(media_library, "_library",
                            media_library.MediaLibrary([], media_dir=str(tmp_path / "absent")))
        resp = client.post("/recommendations/analysis/duplicate_performer/run")
        assert resp.status_code == 400
        assert "writes files into the media library" in resp.json()["detail"]

    def test_read_jobs_unaffected_by_read_only_mount(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(media_library, "_library",
                            media_library.MediaLibrary([], media_dir=str(tmp_path / "absent")))
        assert client.post("/queue", json={"type": "duplicate_performer"}).status_code == 200

    def test_job_fails_cleanly_if_mount_goes_read_only_before_run(self, db, unfinished, monkeypatch, tmp_path):
        from queue_manager import QueueManager

        ran = []

        class Job:
            async def run(self, ctx, cursor=None):
                ran.append(True)

        monkeypatch.setattr(media_library, "_library",
                            media_library.MediaLibrary([], media_dir=str(tmp_path)))
        mgr = QueueManager(db)
        monkeypatch.setattr(mgr, "_create_job_instance", lambda t: Job())
        job_id = mgr.submit("test_write_subs", triggered_by="test")

        monkeypatch.setattr(media_library, "_library",
                            media_library.MediaLibrary([], media_dir=str(tmp_path / "absent")))

        async def run():
            await mgr._dispatch_once()
            await asyncio.gather(*mgr._running_tasks.values())

        asyncio.run(run())
        job = db.get_job(job_id)
        assert ran == []
        assert job["status"] == "failed"
        assert "writes files into the media library" in job["error_message"]


class TestMountReporting:
    def test_health_reports_mount_and_mode(self, client, media):
        body = client.get("/health").json()
        assert body["media"]["mounts"] == [{
            "path": media.media_dir, "mode": "rw", "readable": True, "writable": True, "reason": None,
        }]

    def test_system_info_reports_mount_and_process_ids(self, client):
        import os
        info = client.get("/system/info").json()
        assert info["media"]["writable"] is True
        assert info["process"] == {"uid": os.getuid(), "gid": os.getgid()}


class TestRegistryMarks:
    """The existing types marked unfinished in #124, and the ones kept visible."""

    def test_unfinished_types(self):
        unfinished = {t for t, d in JOB_REGISTRY.items() if not d.production_ready}
        assert unfinished == {"duplicate_scenes", "duplicate_scene_files", "fingerprint_generation", "review_demo",
                              "content_fingerprint", "scene_duplicates"}

    def test_fingerprint_generate_endpoint_rejected(self, client):
        resp = client.post("/recommendations/fingerprints/generate", json={})
        assert resp.status_code == 400


class TestQueuedBeforeHidden:
    def test_hidden_queued_job_waits_until_shown(self, db, settings, unfinished, monkeypatch):
        from queue_manager import QueueManager

        ran = []

        class Job:
            async def run(self, ctx, cursor=None):
                ran.append(True)

        mgr = QueueManager(db)
        monkeypatch.setattr(mgr, "_create_job_instance", lambda t: Job())
        settings.set("show_unfinished_features", True)
        job_id = mgr.submit(UNFINISHED, triggered_by="schedule")
        settings.set("show_unfinished_features", False)

        async def dispatch():
            await mgr._dispatch_once()
            await asyncio.gather(*mgr._running_tasks.values())

        asyncio.run(dispatch())
        assert (ran, db.get_job(job_id)["status"]) == ([], "queued")

        settings.set("show_unfinished_features", True)
        asyncio.run(dispatch())
        assert (ran, db.get_job(job_id)["status"]) == ([True], "completed")
