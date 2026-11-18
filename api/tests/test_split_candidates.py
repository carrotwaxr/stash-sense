"""Split candidates: long scenes, files holding other scenes, and what the index says they cover (#131)."""
import pytest

from content_fingerprint import AUDIO_ALGO_VERSION
from content_fingerprint.store import ContentFpStore
from splits import candidates
from tests.fake_stash import FakeStash

MIN = 60


@pytest.fixture
def stash():
    return FakeStash()


def _scene(stash, name, minutes, *, size=1000, tag_ids=(), group_ids=(), **file_fields):
    """A scene with one file of ``minutes``; returns (scene id, file id)."""
    fid = stash.add_file(f"/lib/{name}.mp4", size, duration=minutes * MIN, oshash=f"h-{name}", **file_fields)
    return stash.add_scene(fid, title=name, tag_ids=tag_ids, group_ids=list(group_ids)), fid


def _index(tmp_path, stash, pairs):
    """A store with every Stash file and overlaps ``pairs`` [(file id, file id, a0, a1, b0, b1)]."""
    store = ContentFpStore(str(tmp_path / "fp.db"))
    ids = {}
    for f in stash.files.values():
        osh = next(x["value"] for x in f["fingerprints"] if x["type"] == "oshash")
        ids[f["id"]] = store.upsert_file(osh, f["size"], duration_s=f["duration"], path=f["path"])
    store.put_overlaps(AUDIO_ALGO_VERSION, [(ids[a], ids[b], a0, a1, b0, b1, 0.05) for a, b, a0, a1, b0, b1 in pairs])
    return store


async def test_long_scenes(stash):
    _scene(stash, "short", 59)
    long_id, long_fid = _scene(stash, "long", 61)
    cands = await candidates.generate(stash, None, set())
    assert [c.scene_id for c in cands] == [long_id]
    c = cands[0]
    assert c.file["id"] == long_fid
    assert (c.coverage, c.covering, c.redundant) == ([], [], False)


async def test_skips_split_outputs_and_excluded(stash):
    group = stash.add_group("[curator] Split Cuts")
    tag = stash.add_tag("[curator] split")
    other_tag = stash.add_tag("something else")
    _scene(stash, "in-group", 90, group_ids=[group])
    _scene(stash, "tagged", 90, tag_ids=(tag,))
    excluded, _ = _scene(stash, "excluded", 90)
    kept, _ = _scene(stash, "kept", 90, tag_ids=(other_tag,))
    cands = await candidates.generate(stash, None, {excluded})
    assert [c.scene_id for c in cands] == [kept]


async def test_missing_group_and_tag_skip_nothing(stash):
    kept, _ = _scene(stash, "kept", 90)
    assert [c.scene_id for c in await candidates.generate(stash, None, set())] == [kept]


def _holder(stash, tmp_path, *, short_fields=None, gaps=False):
    """A 40 min file holding two 15 min scenes' files, at 0-15 and 20-35 min of the long file."""
    long_id, lf = _scene(stash, "holder", 40)
    a_id, af = _scene(stash, "a", 15, **(short_fields or {}))
    b_id, bf = _scene(stash, "b", 15)
    a_end = 15 * MIN - (200 if gaps else 0)
    store = _index(tmp_path, stash, [
        (lf, af, 0, a_end, 0, a_end),
        (lf, bf, 20 * MIN, 35 * MIN, 0, 15 * MIN),
    ])
    return store, long_id, a_id, b_id, af, bf


async def test_contained_files_add_candidates_and_ranges(stash, tmp_path):
    store, long_id, a_id, b_id, af, bf = _holder(stash, tmp_path)
    _scene(stash, "other", 10)     # no overlap: not a candidate
    cands = await candidates.generate(stash, store, set())
    assert [c.scene_id for c in cands] == [long_id]
    c = cands[0]
    assert c.coverage == [(0, 900), (1200, 2100)]
    assert c.covering == [
        {"scene_id": a_id, "file_id": af, "start": 0, "end": 900},
        {"scene_id": b_id, "file_id": bf, "start": 1200, "end": 2100},
    ]
    assert not c.redundant        # 900-1200 and 2100-2400 are uncovered


async def test_partial_overlap_is_not_covering(stash, tmp_path):
    long_id, lf = _scene(stash, "holder", 40)
    _, af = _scene(stash, "a", 15)
    store = _index(tmp_path, stash, [(lf, af, 0, 600, 0, 600)])    # 2/3 of the short file
    assert await candidates.generate(stash, store, set()) == []


async def test_a_duplicate_is_not_covering(stash, tmp_path):
    _, lf = _scene(stash, "holder", 40)
    _, af = _scene(stash, "twin", 30)       # the long file is under 1.5x the other
    store = _index(tmp_path, stash, [(lf, af, 0, 1800, 0, 1800)])
    assert await candidates.generate(stash, store, set()) == []


async def test_excluded_covering_scene_is_ignored(stash, tmp_path):
    store, long_id, a_id, b_id, af, bf = _holder(stash, tmp_path)
    (c,) = await candidates.generate(stash, store, {a_id})
    assert [x["scene_id"] for x in c.covering] == [b_id]


def _whole(stash, tmp_path, *, short_fields=None):
    """A 40 min file whose 150 s guards are the only uncovered part, held by three scenes."""
    long_id, lf = _scene(stash, "holder", 40)
    ids = []
    pairs = []
    for name, (s, e) in (("a", (0, 900)), ("b", (900, 1700)), ("c", (1700, 2400))):
        sid, fid = _scene(stash, name, (e - s) / MIN, **((short_fields or {}) if name == "b" else {}))
        ids.append(sid)
        pairs.append((lf, fid, s, e, 0, e - s))
    return _index(tmp_path, stash, pairs), long_id


async def test_redundant_when_fully_covered_at_equal_quality(stash, tmp_path):
    store, long_id = _whole(stash, tmp_path)
    (c,) = await candidates.generate(stash, store, set())
    assert c.scene_id == long_id and c.redundant
    assert c.coverage == [(0, 2400)]


async def test_one_worse_covering_file_makes_it_not_redundant(stash, tmp_path):
    store, _ = _whole(stash, tmp_path, short_fields={"width": 640, "height": 360})
    (c,) = await candidates.generate(stash, store, set())
    assert not c.redundant
    assert c.coverage == [(0, 2400)]


async def test_guards_may_stay_uncovered(stash, tmp_path):
    long_id, lf = _scene(stash, "holder", 60)      # 3600 s, guards 0-150 and 3450-3600
    pairs = []
    for name, (s, e) in (("a", (150, 1700)), ("b", (1700, 2600)), ("c", (2600, 3450))):
        _, fid = _scene(stash, name, (e - s) / MIN)
        pairs.append((lf, fid, s, e, 0, e - s))
    store = _index(tmp_path, stash, pairs)
    (c,) = await candidates.generate(stash, store, set())
    assert c.redundant
    assert c.coverage == [(150, 3450)]


async def test_a_gap_outside_the_guards_is_not_redundant(stash, tmp_path):
    long_id, lf = _scene(stash, "holder", 60)
    pairs = []
    for name, (s, e) in (("a", (150, 1700)), ("b", (1800, 2600)), ("c", (2600, 3450))):
        _, fid = _scene(stash, name, (e - s) / MIN)
        pairs.append((lf, fid, s, e, 0, e - s))
    store = _index(tmp_path, stash, pairs)
    (c,) = await candidates.generate(stash, store, set())
    assert not c.redundant
