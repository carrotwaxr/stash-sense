"""The scene_duplicates analyzer (#130): verified sets upserted in place, so the driver feed never loses a rec."""

import json
import sqlite3

import numpy as np
import pytest

import curation_settings
import media_library
from analyzers.scene_duplicates import SceneDuplicatesAnalyzer
from content_fingerprint import AUDIO_ALGO, AUDIO_ALGO_VERSION
from content_fingerprint.store import ContentFpStore
from curation_settings import CurationSettings, Exclusion, ExclusionUnavailable
from duplicates import audio, health, sprites
from recommendations_db import RecommendationsDB
from tests.fake_stash import FakeStash

TYPE = "scene_duplicates"
SHARED = [{"endpoint": "e1", "stash_id": "x"}]
OLD = "2000-01-01 00:00:00"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv("DATA_DIR", str(d))
    return d


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "stash_sense.db")


@pytest.fixture
def stash():
    return FakeStash()


@pytest.fixture
def health_calls(monkeypatch):
    calls = []

    async def fake_health(local_path, *a, **k):
        calls.append(local_path)
        return {"status": "ok", "reasons": [], "params": {"points": 24}}

    async def no_sprite(*a, **k):
        return None

    async def no_audio(*a, **k):
        return None

    class Lib:
        def to_local(self, p):
            return p

    monkeypatch.setattr(health, "check_health", fake_health)
    monkeypatch.setattr(sprites, "fetch_sprite", no_sprite)
    monkeypatch.setattr(audio, "fingerprint_head", no_audio)
    monkeypatch.setattr(media_library, "get_media_library", lambda: Lib())
    return calls


@pytest.fixture(autouse=True)
def exclusion(monkeypatch):
    state = {"ids": frozenset(), "error": None}

    async def current(stash, **kw):
        if state["error"]:
            raise state["error"]
        return Exclusion(CurationSettings(), state["ids"], 0.0)

    monkeypatch.setattr(curation_settings, "current", current)
    return state


def fp(seed):
    return np.random.default_rng(seed).integers(0, 2 ** 32, 400, dtype=np.uint64).astype(np.uint32)


class World:
    def __init__(self, stash, data_dir):
        self.stash = stash
        self.path = str(data_dir / "content_fp.db")

    def scene(self, name, seed, *, stash_ids=SHARED, **file_fields):
        """A scene with one file whose stored print comes from ``seed`` (the same seed means the same audio; None
        stores no print). Extra fields go to the scene, e.g. studio_id."""
        fid = self.stash.add_file(f"/lib/{name}.mp4", 1000 + len(name), oshash=f"h-{name}")
        sid = self.stash.add_scene(fid, title=name, stash_ids=list(stash_ids), **file_fields)
        if seed is not None:
            self.print(name, seed)
        return sid, fid

    def print(self, name, seed):
        store = ContentFpStore(self.path)
        try:
            cid = store.upsert_file(f"h-{name}", 1000 + len(name))
            store.put_audio(cid, AUDIO_ALGO, AUDIO_ALGO_VERSION, {}, "ok", fp(seed))
        finally:
            store.close()

    def health(self, name):
        store = ContentFpStore(self.path)
        try:
            cid = store.upsert_file(f"h-{name}", 1000 + len(name))
            return store.get_health(cid)
        finally:
            store.close()


@pytest.fixture
def world(stash, data_dir):
    return World(stash, data_dir)


async def run(stash, db, stop=False):
    analyzer = SceneDuplicatesAnalyzer(stash, db)
    if stop:
        analyzer.request_stop()
    result = await analyzer.run()
    return analyzer, result


def recs(db, status=None):
    return db.get_recommendations(type=TYPE, status=status, limit=1000)


def age(db):
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("UPDATE recommendations SET updated_at = ?", (OLD,))


# ---------------- db ----------------

def test_upsert_recommendation_states(db):
    rid, how = db.upsert_recommendation("t", "scene_set", "k", {"a": 1, "b": 2}, 0.9, None)
    assert how == "created" and rid
    age(db)
    # same details in another key order and the same confidence change nothing
    assert db.upsert_recommendation("t", "scene_set", "k", {"b": 2, "a": 1}, 0.9, None) == (rid, "unchanged")
    assert db.get_recommendation(rid).updated_at == OLD
    # changed details or confidence update in place
    assert db.upsert_recommendation("t", "scene_set", "k", {"a": 1, "b": 3}, 0.9, None) == (rid, "updated")
    rec = db.get_recommendation(rid)
    assert rec.details == {"a": 1, "b": 3} and rec.updated_at > OLD
    age(db)
    assert db.upsert_recommendation("t", "scene_set", "k", {"a": 1, "b": 3}, 0.5, None) == (rid, "updated")
    assert db.get_recommendation(rid).confidence == 0.5
    # a resolved row that comes back unchanged is left alone
    db.resolve_recommendation(rid, "stale", {"why": "gone"})
    age(db)
    assert db.upsert_recommendation("t", "scene_set", "k", {"a": 1, "b": 3}, 0.5, None) == (rid, "skipped")
    rec = db.get_recommendation(rid)
    assert rec.status == "resolved" and rec.updated_at == OLD
    # one that comes back different is reopened under the same id, resolution cleared
    assert db.upsert_recommendation("t", "scene_set", "k", {"z": 1}, 0.1, None) == (rid, "reopened")
    rec = db.get_recommendation(rid)
    assert rec.status == "pending" and rec.details == {"z": 1} and rec.confidence == 0.1 and rec.updated_at > OLD
    assert rec.resolution_action is None and rec.resolution_details is None and rec.resolved_at is None
    # a dismissed row is never touched
    db.dismiss_recommendation(rid)
    age(db)
    assert db.upsert_recommendation("t", "scene_set", "k", {"y": 1}, 0.2, None) == (rid, "skipped")
    rec = db.get_recommendation(rid)
    assert rec.status == "dismissed" and rec.details == {"z": 1} and rec.updated_at == OLD


# ---------------- analyzer ----------------

async def test_creates_one_rec_per_item(stash, db, world, health_calls):
    a, _ = world.scene("a", 1)
    b, _ = world.scene("b", 1)
    analyzer, result = await run(stash, db)
    [rec] = recs(db)
    assert (rec.type, rec.target_type, rec.target_id, rec.status) == (TYPE, "scene_set", f"duplicate:{a}", "pending")
    assert rec.details["cls"] == "duplicate" and rec.details["scene_ids"] == [a, b]
    assert rec.details["pairs"][0]["evidence"]["ber"] == 0.0
    sug = rec.details["suggestion"]
    assert set(sug) == {"keep", "delete", "reason", "flags", "held"}
    assert len(sug["keep"]) == 1 and len(sug["delete"]) == 1 and sug["held"] is False
    assert set(sug["keep"] + sug["delete"]) == {"h-a", "h-b"}
    assert rec.confidence == 0.9
    assert result.items_processed == 1 and result.recommendations_created == 1


async def test_second_run_changes_nothing(stash, db, world, health_calls):
    world.scene("a", 1)
    world.scene("b", 1)
    await run(stash, db)
    age(db)
    before = [(r.id, r.updated_at, r.details) for r in recs(db)]
    analyzer, result = await run(stash, db)
    assert [(r.id, r.updated_at, r.details) for r in recs(db)] == before
    assert result.recommendations_created == 0
    assert analyzer.stats["unchanged"] == 1 and analyzer.stats["updated"] == 0


async def test_changed_suggestion_updates_in_place(stash, db, world, health_calls):
    world.scene("a", 1)
    _, fb = world.scene("b", 1)
    await run(stash, db)
    [first] = recs(db)
    age(db)
    stash.files[fb]["bit_rate"] = 20_000_000       # b becomes the better file
    analyzer, result = await run(stash, db)
    [second] = recs(db)
    assert second.id == first.id and second.status == "pending"
    assert second.updated_at > OLD
    assert second.details["suggestion"]["keep"] == ["h-b"]
    assert first.details["suggestion"]["keep"] == ["h-a"]
    assert result.recommendations_created == 0 and analyzer.stats["updated"] == 1


async def test_superseded_key_is_resolved(stash, db, world, health_calls):
    a, _ = world.scene("a", 7, stash_ids=[{"endpoint": "e1", "stash_id": "other"}])
    b, _ = world.scene("b", 1)
    c, _ = world.scene("c", 1)
    await run(stash, db)
    [old] = recs(db, "pending")
    assert old.target_id == f"duplicate:{b}"
    # the lowest scene turns out to hold the same content
    stash.scenes[a]["stash_ids"] = SHARED
    world.print("a", 1)
    analyzer, _ = await run(stash, db)
    by_key = {r.target_id: r for r in recs(db)}
    assert set(by_key) == {f"duplicate:{a}", f"duplicate:{b}"}
    assert by_key[f"duplicate:{a}"].status == "pending"
    assert by_key[f"duplicate:{a}"].details["scene_ids"] == [a, b, c]
    gone = by_key[f"duplicate:{b}"]
    assert gone.id == old.id and gone.status == "resolved" and gone.resolution_action == "superseded"
    assert gone.resolution_details == {"superseded_by": f"duplicate:{a}"}
    assert analyzer.stats["superseded"] == 1


async def test_vanished_item_is_resolved_stale_only_when_finished(stash, db, world, health_calls):
    a, _ = world.scene("a", 1)
    b, _ = world.scene("b", 1)
    await run(stash, db)
    stash.scenes[a]["stash_ids"] = []
    stash.scenes[b]["stash_ids"] = []
    await run(stash, db, stop=True)
    assert [r.status for r in recs(db)] == ["pending"]
    analyzer, result = await run(stash, db)
    [rec] = recs(db)
    assert rec.status == "resolved" and rec.resolution_action == "stale"
    assert analyzer.stats["stale"] == 1 and result.recommendations_created == 0


async def test_dismissed_rec_is_left_alone(stash, db, world, health_calls):
    world.scene("a", 1)
    _, fb = world.scene("b", 1)
    await run(stash, db)
    [rec] = recs(db)
    db.dismiss_recommendation(rec.id)
    age(db)
    stash.files[fb]["bit_rate"] = 20_000_000
    await run(stash, db)
    [after] = recs(db)
    assert after.id == rec.id and after.status == "dismissed" and after.updated_at == OLD
    assert after.details == rec.details


async def test_excluded_scenes_make_no_recs(stash, db, world, health_calls, exclusion):
    a, _ = world.scene("a", 1)
    world.scene("b", 1)
    c, _ = world.scene("c", 2, stash_ids=[{"endpoint": "e2", "stash_id": "y"}])
    d, _ = world.scene("d", 2, stash_ids=[{"endpoint": "e2", "stash_id": "y"}])
    exclusion["ids"] = frozenset({a})
    await run(stash, db)
    assert [r.target_id for r in recs(db)] == [f"duplicate:{c}"]


async def test_fails_closed_without_the_exclusion_list(stash, db, world, health_calls, exclusion):
    world.scene("a", 1)
    world.scene("b", 1)
    exclusion["error"] = ExclusionUnavailable("stash is down")
    with pytest.raises(ExclusionUnavailable):
        await run(stash, db)
    assert recs(db) == []


async def test_keeper_health_is_checked_once(stash, db, world, health_calls):
    world.scene("a", 1)
    world.scene("b", 1)
    await run(stash, db)
    assert len(health_calls) == 1
    keeper = recs(db)[0].details["suggestion"]["keep"][0]
    assert world.health(keeper[2:])["status"] == "ok"
    await run(stash, db)
    assert len(health_calls) == 1


async def test_different_pair_makes_no_rec(stash, db, world, health_calls):
    world.scene("a", 1)
    world.scene("b", 2)                 # the shared stash id makes them candidates; the audio says otherwise
    analyzer, result = await run(stash, db)
    assert recs(db) == []
    assert result.items_processed == 1 and result.recommendations_created == 0


async def test_pair_with_no_evidence_makes_no_rec_and_is_counted(stash, db, world, health_calls):
    world.scene("a", None)
    world.scene("b", None)              # candidates by shared stash id, but nothing readable to compare
    analyzer, result = await run(stash, db)
    assert recs(db) == []
    assert analyzer.stats["unverified"] == 1
    assert result.items_processed == 1 and result.recommendations_created == 0


async def test_unhealthy_keeper_is_replaced_and_held_when_none_is_healthy(stash, db, world, monkeypatch, health_calls):
    world.scene("a", 1)
    world.scene("b", 1)

    async def bad(local_path, *a, **k):
        health_calls.append(local_path)
        return {"status": "suspect", "reasons": ["x"], "params": {"points": 24}}

    monkeypatch.setattr(health, "check_health", bad)
    await run(stash, db)
    [rec] = recs(db)
    sug = rec.details["suggestion"]
    assert sug["held"] is True and sug["delete"] == [] and len(health_calls) == 2


async def test_returning_set_is_reopened_but_dismissed_stays(stash, db, world, health_calls):
    a, _ = world.scene("a", 1)
    _, fb = world.scene("b", 1)
    c, _ = world.scene("c", 2, stash_ids=[{"endpoint": "e2", "stash_id": "y"}])
    d, fd = world.scene("d", 2, stash_ids=[{"endpoint": "e2", "stash_id": "y"}])
    await run(stash, db)
    by_key = {r.target_id: r for r in recs(db)}
    db.dismiss_recommendation(by_key[f"duplicate:{c}"].id)
    ids = {a: stash.scenes[a]["stash_ids"], c: stash.scenes[c]["stash_ids"]}
    for sid in (a, c):
        stash.scenes[sid]["stash_ids"] = []
    await run(stash, db)
    assert {r.target_id: r.status for r in recs(db)} == {f"duplicate:{a}": "resolved", f"duplicate:{c}": "dismissed"}
    for sid in (a, c):
        stash.scenes[sid]["stash_ids"] = ids[sid]
    stash.files[fb]["bit_rate"] = 20_000_000
    stash.files[fd]["bit_rate"] = 20_000_000
    await run(stash, db)
    after = {r.target_id: r for r in recs(db)}
    back = after[f"duplicate:{a}"]
    assert back.id == by_key[f"duplicate:{a}"].id and back.status == "pending" and back.resolution_action is None
    assert back.details["suggestion"]["keep"] == ["h-b"]
    assert after[f"duplicate:{c}"].status == "dismissed"


async def test_sprites_are_fetched_only_when_audio_is_inconclusive(stash, db, world, health_calls, monkeypatch):
    fetched = []

    async def fetch(url, key, scene_id):
        fetched.append(scene_id)

    monkeypatch.setattr(sprites, "fetch_sprite", fetch)
    world.scene("a", 1)
    world.scene("b", 1)                  # strong stored audio: the picture cannot change the class
    analyzer, _ = await run(stash, db)
    assert fetched == [] and analyzer.stats["sprite_fetches"] == 0

    c, _ = world.scene("c", None, stash_ids=[{"endpoint": "e2", "stash_id": "y"}])
    d, _ = world.scene("d", None, stash_ids=[{"endpoint": "e2", "stash_id": "y"}])
    analyzer, _ = await run(stash, db)    # no print for either file and no local file
    assert sorted(fetched) == sorted([c, d]) and analyzer.stats["sprite_fetches"] == 2


async def test_stop_tiles_come_from_other_scenes_of_the_studio(stash, db, world, health_calls, monkeypatch):
    studio = stash.add_studio("S")
    ids = [world.scene(n, None, studio_id=studio)[0] for n in "abcdef"]
    odd, _ = world.scene("z", None, studio_id=stash.add_studio("T"))
    fetched, stop_calls = [], []

    async def fetch(url, key, scene_id):
        fetched.append(scene_id)
        return b"", ""

    rng = np.random.default_rng(5)

    def signature(img, vtt):
        full = rng.normal(size=(10, 432)).astype(np.float32)
        return {"times": np.arange(10, dtype=np.float32) * 10, "full": full / np.linalg.norm(full, axis=1)[:, None]}

    real = sprites.stop_tiles_for

    def spy(sig, others, min_scenes=4, thr=0.85):
        stop_calls.append(len(others))
        return real(sig, others, min_scenes, thr)

    monkeypatch.setattr(sprites, "fetch_sprite", fetch)
    monkeypatch.setattr(sprites, "sprite_signature", signature)
    monkeypatch.setattr(sprites, "stop_tiles_for", spy)
    monkeypatch.setattr("analyzers.scene_duplicates.STOP_SCENES", 3)
    analyzer, _ = await run(stash, db)
    # seven scenes share one stash id. The 15 pairs inside studio S are each compared (both sides) against at most 3
    # other scenes of S; the 6 pairs with the scene of studio T get no stop tiles.
    assert set(stop_calls) == {3} and len(stop_calls) == 2 * 15
    assert analyzer.stats["sprite_fetches"] == len(fetched) == len(ids) + 1 and odd in fetched


async def test_pair_in_different_studios_gets_no_stop_tiles(stash, db, world, health_calls, monkeypatch):
    world.scene("a", None, studio_id=stash.add_studio("S"))
    world.scene("b", None, studio_id=stash.add_studio("T"))
    calls = []

    async def fetch(url, key, scene_id):
        return b"", ""

    def signature(img, vtt):
        full = np.eye(10, 432, dtype=np.float32)
        return {"times": np.arange(10, dtype=np.float32) * 10, "full": full}

    monkeypatch.setattr(sprites, "fetch_sprite", fetch)
    monkeypatch.setattr(sprites, "sprite_signature", signature)
    monkeypatch.setattr(sprites, "stop_tiles_for", lambda *a, **k: calls.append(a) or set())
    await run(stash, db)
    assert calls == []
