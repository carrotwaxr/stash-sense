"""Candidate scene pairs for duplicate detection (#130)."""
import pytest

from content_fingerprint import AUDIO_ALGO_VERSION
from content_fingerprint.store import ContentFpStore
from duplicates import candidates
from tests.fake_stash import FakeStash


@pytest.fixture
def stash():
    return FakeStash()


def _scene(stash, name, *, oshash=None, size=1000, **fields):
    """A scene with one file; returns (scene id, file id)."""
    fid = stash.add_file(f"/lib/{name}.mp4", size, oshash=oshash or f"h-{name}")
    return stash.add_scene(fid, title=fields.pop("title", name), **fields), fid


def _only(cands):
    assert len(cands) == 1, cands
    return cands[0]


async def test_shared_stash_id(stash):
    a, _ = _scene(stash, "a", stash_ids=[{"endpoint": "e1", "stash_id": "x"}])
    b, _ = _scene(stash, "b", stash_ids=[{"endpoint": "e1", "stash_id": "x"}])
    _scene(stash, "c", stash_ids=[{"endpoint": "e2", "stash_id": "x"}])   # other endpoint
    c = _only(await candidates.generate(stash, None, set()))
    assert (c.scene_a, c.scene_b) == (a, b)
    assert c.sources == {"stash_id"}
    assert not c.weak
    assert c.file_a is None and c.file_b is None and c.overlap is None


async def test_studio_and_normalized_title(stash):
    s = stash.add_studio("S")
    a, _ = _scene(stash, "a", title="Hello, World!", studio_id=s)
    b, _ = _scene(stash, "b", title="  hello   WORLD ", studio_id=s)
    _scene(stash, "c", title="hello world")                 # no studio
    _scene(stash, "d", title="", studio_id=s)               # no title
    _scene(stash, "e", title="", studio_id=s)
    c = _only(await candidates.generate(stash, None, set()))
    assert (c.scene_a, c.scene_b) == (a, b)
    assert c.sources == {"studio_title"}


async def test_studio_date_performers(stash):
    s = stash.add_studio("S")
    a, _ = _scene(stash, "a", studio_id=s, date="2020-01-01", performer_ids=["p1", "p2"])
    b, _ = _scene(stash, "b", studio_id=s, date="2020-01-01", performer_ids=["p2", "p1"])
    _scene(stash, "c", studio_id=s, date="2020-01-01", performer_ids=["p1"])
    _scene(stash, "d", studio_id=s, date="2020-02-01", performer_ids=["p1", "p2"])
    _scene(stash, "e", studio_id=s, date="2021-01-01")
    _scene(stash, "f", studio_id=s, date="2021-01-01")      # empty performer sets never match
    c = _only(await candidates.generate(stash, None, set()))
    assert (c.scene_a, c.scene_b) == (a, b)
    assert c.sources == {"studio_date_performers"}


async def test_phash_groups(stash):
    a, _ = _scene(stash, "a")
    b, _ = _scene(stash, "b")
    c3, _ = _scene(stash, "c")
    stash.phash_groups = [[c3, a], [b, c3]]
    cands = await candidates.generate(stash, None, set())
    assert [(c.scene_a, c.scene_b) for c in cands] == [(a, c3), (b, c3)]
    assert all(c.sources == {"phash"} and not c.weak for c in cands)


async def test_url_is_weak_alone_but_not_with_a_strong_source(stash):
    a, _ = _scene(stash, "a", urls=["https://x/1"])
    b, _ = _scene(stash, "b", urls=["https://x/1"])
    c3, _ = _scene(stash, "c", urls=["https://x/2"])
    d, _ = _scene(stash, "d", urls=["https://x/2"])
    stash.phash_groups = [[c3, d]]
    cands = {(c.scene_a, c.scene_b): c for c in await candidates.generate(stash, None, set())}
    assert cands[(a, b)].sources == {"url"} and cands[(a, b)].weak
    assert cands[(c3, d)].sources == {"url", "phash"} and not cands[(c3, d)].weak


async def test_multi_file_scene(stash):
    f1 = stash.add_file("/lib/p1.mp4", 100, oshash="o1")
    f2 = stash.add_file("/lib/p2.mp4", 200, oshash="o2")
    s = stash.add_scene(f1, f2, title="two parts")
    c = _only(await candidates.generate(stash, None, set()))
    assert (c.scene_a, c.scene_b) == (s, s)
    assert {c.file_a, c.file_b} == {"o1", "o2"}
    assert c.sources == {"multi_file"}


def _index(tmp_path, stash, pairs):
    """A store with every Stash file and overlaps ``pairs`` [(file id, file id, a0, a1, b0, b1)]."""
    store = ContentFpStore(str(tmp_path / "fp.db"))
    ids = {}
    for f in stash.files.values():
        osh = next(x["value"] for x in f["fingerprints"] if x["type"] == "oshash")
        ids[f["id"]] = store.upsert_file(osh, f["size"], duration_s=f["duration"], path=f["path"])
    store.put_overlaps(AUDIO_ALGO_VERSION, [(ids[a], ids[b], a0, a1, b0, b1, 0.05) for a, b, a0, a1, b0, b1 in pairs])
    return store, ids


async def test_index_pair_kinds(stash, tmp_path):
    a, fa = _scene(stash, "a")
    b, fb = _scene(stash, "b")
    c3, fc = _scene(stash, "c")
    d, fd = _scene(stash, "d")
    store, _ = _index(tmp_path, stash, [
        (fa, fb, 0, 600, 0, 600),      # duplicate
        (fa, fc, 0, 100, 0, 100),      # incidental: dropped
        (fc, fd, 100, 400, 0, 300),    # partial
    ])
    cands = {(c.scene_a, c.scene_b): c for c in await candidates.generate(stash, store, set())}
    assert set(cands) == {(a, b), (c3, d)}
    dup = cands[(a, b)]
    assert dup.sources == {"index"} and not dup.weak
    assert (dup.file_a, dup.file_b) == ("h-a", "h-b")
    assert dup.overlap["kind"] == "duplicate"
    assert cands[(c3, d)].overlap["kind"] == "partial"


async def test_index_contained(stash, tmp_path):
    a, fa = _scene(stash, "a")
    f = stash.add_file("/lib/long.mp4", 5000, duration=6000.0, oshash="h-long")
    b = stash.add_scene(f, title="long")
    store, _ = _index(tmp_path, stash, [(fa, f, 0, 600, 1000, 1600)])
    c = _only(await candidates.generate(stash, store, set()))
    assert c.overlap["kind"] == "contained"


async def test_index_pair_orientation_follows_scene_order(stash, tmp_path):
    f_short = stash.add_file("/lib/s.mp4", 100, duration=600.0, oshash="h-s")
    f_long = stash.add_file("/lib/l.mp4", 200, duration=6000.0, oshash="h-l")
    # the long file's scene has the lower id, but the store orders by its own file ids
    s_long = stash.add_scene(f_long, title="l")
    s_short = stash.add_scene(f_short, title="s")
    store, _ = _index(tmp_path, stash, [(f_short, f_long, 0, 600, 0, 600)])
    c = _only(await candidates.generate(stash, store, set()))
    assert (c.scene_a, c.scene_b) == (s_long, s_short)
    assert (c.file_a, c.file_b) == ("h-l", "h-s")
    assert c.overlap["cover_a"] == 0.1 and c.overlap["cover_b"] == 1.0


async def test_index_pair_within_one_scene_is_a_file_pair(stash, tmp_path):
    f1 = stash.add_file("/lib/p1.mp4", 100, oshash="o1")
    f2 = stash.add_file("/lib/p2.mp4", 200, oshash="o2")
    s = stash.add_scene(f1, f2, title="two copies")
    store, _ = _index(tmp_path, stash, [(f1, f2, 0, 600, 0, 600)])
    c = _only(await candidates.generate(stash, store, set()))
    assert (c.scene_a, c.scene_b) == (s, s)
    assert {c.file_a, c.file_b} == {"o1", "o2"}
    assert c.sources == {"index", "multi_file"}
    assert c.overlap["kind"] == "duplicate"


async def test_index_pair_with_unknown_file_is_skipped(stash, tmp_path):
    a, fa = _scene(stash, "a")
    b, fb = _scene(stash, "b")
    store, _ = _index(tmp_path, stash, [(fa, fb, 0, 600, 0, 600)])
    del stash.files[fb]
    stash.scenes[b]["file_ids"] = []     # the file is gone from Stash, still in the index
    assert await candidates.generate(stash, store, set()) == []


async def test_scene_pair_and_file_pair_merge_and_sources_union(stash, tmp_path):
    s = stash.add_studio("S")
    a, fa = _scene(stash, "a", title="Same", studio_id=s)
    b, fb = _scene(stash, "b", title="same", studio_id=s)
    stash.phash_groups = [[a, b]]
    store, _ = _index(tmp_path, stash, [(fa, fb, 0, 600, 0, 600)])
    c = _only(await candidates.generate(stash, store, set()))
    assert c.sources == {"studio_title", "phash", "index"}
    assert (c.file_a, c.file_b) == ("h-a", "h-b")
    assert c.overlap["kind"] == "duplicate"


async def test_excluded_scenes_produce_nothing(stash, tmp_path):
    a, fa = _scene(stash, "a", stash_ids=[{"endpoint": "e", "stash_id": "x"}])
    b, fb = _scene(stash, "b", stash_ids=[{"endpoint": "e", "stash_id": "x"}])
    f1 = stash.add_file("/lib/p1.mp4", 100, oshash="o1")
    f2 = stash.add_file("/lib/p2.mp4", 200, oshash="o2")
    m = stash.add_scene(f1, f2, title="multi")
    stash.phash_groups = [[a, b]]
    store, _ = _index(tmp_path, stash, [(fa, fb, 0, 600, 0, 600), (f1, f2, 0, 600, 0, 600)])
    assert await candidates.generate(stash, store, {a, m}) == []
    assert len(await candidates.generate(stash, store, set())) == 2


async def test_pages_through_all_scenes(stash, monkeypatch):
    monkeypatch.setattr(candidates, "PAGE_SIZE", 2)
    ids = [_scene(stash, f"s{i}", stash_ids=[{"endpoint": "e", "stash_id": "x"}])[0] for i in range(5)]
    cands = await candidates.generate(stash, None, set())
    assert len(cands) == 10
    assert [(c.scene_a, c.scene_b) for c in cands] == sorted(
        [(x, y) for i, x in enumerate(ids) for y in ids[i + 1:]], key=lambda p: (int(p[0]), int(p[1])))
