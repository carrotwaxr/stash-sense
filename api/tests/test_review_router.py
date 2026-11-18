"""Review API over HTTP: status codes and payloads the plugin relies on."""

import shutil
import subprocess

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import features
import media_library
import review_router
from recommendations_db import RecommendationsDB
from review.demo import DEMO_TAG, DemoReviewAnalyzer
from review.service import ReviewService
from review.store import ReviewStore
from tests.fake_stash import FakeStash


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "show_unfinished", lambda: True)
    db = RecommendationsDB(tmp_path / "s.db")
    stash = FakeStash()
    review_router.set_review_service(ReviewService(db, ReviewStore(db.db_path), stash))
    app = FastAPI()
    app.include_router(review_router.router)
    yield TestClient(app), db, stash
    review_router.set_review_service(None)


async def _seed(stash, db, *, identical=True):
    tag = stash.add_tag(DEMO_TAG)
    a = stash.add_file("/media/x/a.mp4", 100, oshash="o")
    b = stash.add_file("/media/x/b.mp4", 200, oshash="o" if identical else "p")
    sid = stash.add_scene(a, b, title="t", tag_ids=(tag,))
    await DemoReviewAnalyzer(stash, db).run()
    return db.get_recommendation_by_target("review_demo", "scene", sid), a, b


def test_types_hide_unfinished(world, monkeypatch):
    client, *_ = world
    assert [t["type_id"] for t in client.get("/review/types").json()["types"]] == ["review_demo"]
    monkeypatch.setattr(features, "show_unfinished", lambda: False)
    assert client.get("/review/types").json()["types"] == []


async def test_accept_flow_status_codes(world):
    client, db, stash = world
    rec, a, b = await _seed(stash, db)
    p = client.post(f"/review/{rec.id}/preview", json={}).json()

    r = client.post(f"/review/{rec.id}/accept", json={"option": p["option"], "plan_token": p["plan_token"]})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "confirm_delete"
    assert r.json()["detail"]["files"][0]["path"] == "/media/x/a.mp4"

    r = client.post(f"/review/{rec.id}/accept", json={"option": p["option"], "plan_token": "nope",
                                                     "confirm_delete": [a]})
    assert r.status_code == 409 and r.json()["detail"]["code"] == "stale"
    assert r.json()["detail"]["preview"]["plan_token"] == p["plan_token"]

    r = client.post(f"/review/{rec.id}/accept", json={"option": p["option"], "plan_token": p["plan_token"],
                                                     "confirm_delete": [a]})
    assert r.status_code == 200
    entry_id = r.json()["entry"]["id"]

    log = client.get("/review/apply-log").json()
    assert log["total"] == 1 and log["entries"][0]["id"] == entry_id
    r = client.post(f"/review/apply-log/{entry_id}/revert", json={})
    assert r.status_code == 200 and stash.scenes[rec.target_id]["title"] == "t"
    assert client.post(f"/review/apply-log/{entry_id}/revert").json()["detail"]["code"] == "not_revertible"


async def test_bad_option_and_missing_rec(world):
    client, db, stash = world
    rec, a, b = await _seed(stash, db)
    r = client.post(f"/review/{rec.id}/preview", json={"option": {"keep_file_id": "999"}})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "rejected"
    assert client.post("/review/424242/preview", json={}).status_code == 404


async def test_batch_ineligible_is_400(world):
    client, db, stash = world
    rec, a, b = await _seed(stash, db, identical=False)
    p = client.post(f"/review/{rec.id}/preview", json={}).json()
    r = client.post("/review/types/review_demo/batch/accept",
                    json={"items": [{"rec_id": rec.id, "plan_token": p["plan_token"], "confirm_delete": [a]}]})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "not_batch_eligible"


async def test_labels_export(world):
    client, db, stash = world
    rec, a, b = await _seed(stash, db, identical=False)
    p = client.post(f"/review/{rec.id}/preview", json={"option": {"keep_file_id": a}}).json()
    client.post(f"/review/{rec.id}/accept", json={"option": p["option"], "plan_token": p["plan_token"],
                                                 "confirm_delete": [b]})
    [label] = client.get("/review/labels?type=review_demo").json()["labels"]
    assert label["chosen"]["keep_file_id"] == a and label["suggested"]["keep_file_id"] == b


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_still_from_real_file(world, tmp_path, monkeypatch):
    client, db, stash = world
    video = tmp_path / "media" / "clip.mp4"
    video.parent.mkdir()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
                    "-t", "3", "-pix_fmt", "yuv420p", str(video)], check=True)
    monkeypatch.setattr(media_library, "_library", media_library.MediaLibrary.from_env(
        {"MEDIA_PATH_MAP": f"/stash/media={video.parent}"}))
    fid = stash.add_file("/stash/media/clip.mp4", video.stat().st_size, duration=3)
    sid = stash.add_scene(fid)
    other = stash.add_scene(stash.add_file("/stash/media/other.mp4", 1))

    r = client.get(f"/review/media/still?scene_id={sid}&file_id={fid}&t=1&width=160")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8"
    # A file must belong to the named scene: no reading arbitrary library files by id.
    assert client.get(f"/review/media/still?scene_id={other}&file_id={fid}").status_code == 404
