"""Quality store: cached per-file cheap-pass and sharpness results."""

from quality.store import QualityStore


def make(tmp_path):
    return QualityStore(str(tmp_path / "quality.db"))


def test_cheap_roundtrip(tmp_path):
    store = make(tmp_path)
    result = {"tier": "ok", "duration": 1234.5, "flags": ["a", "b"]}
    store.put_cheap("abc", 10, 1, result)
    assert store.get_cheap("abc", 10, min_version=1) == result
    store.close()


def test_cheap_missing_is_none(tmp_path):
    store = make(tmp_path)
    assert store.get_cheap("nope", 10, min_version=1) is None
    store.close()


def test_cheap_old_version_is_none(tmp_path):
    store = make(tmp_path)
    store.put_cheap("abc", 10, 1, {"x": 1})
    assert store.get_cheap("abc", 10, min_version=2) is None
    store.close()


def test_cheap_upsert_replaces(tmp_path):
    store = make(tmp_path)
    store.put_cheap("abc", 10, 1, {"x": 1})
    store.put_cheap("abc", 10, 2, {"x": 2})
    assert store.get_cheap("abc", 10, min_version=2) == {"x": 2}
    store.close()


def test_sharpness_roundtrip(tmp_path):
    store = make(tmp_path)
    result = {"blur": 0.4, "block": 0.1}
    store.put_sharpness("abc", 10, 3, result)
    assert store.get_sharpness("abc", 10, min_version=3) == result
    assert store.get_sharpness("abc", 10, min_version=4) is None
    store.close()


def test_tables_are_separate(tmp_path):
    store = make(tmp_path)
    store.put_cheap("abc", 10, 1, {"x": 1})
    assert store.get_sharpness("abc", 10, min_version=1) is None
    store.close()


def test_reopen_keeps_rows(tmp_path):
    store = make(tmp_path)
    store.put_cheap("abc", 10, 1, {"x": 1})
    store.put_sharpness("abc", 10, 1, {"y": 2})
    store.close()
    again = make(tmp_path)
    assert again.get_cheap("abc", 10, min_version=1) == {"x": 1}
    assert again.get_sharpness("abc", 10, min_version=1) == {"y": 2}
    again.close()


def test_size_is_part_of_key(tmp_path):
    store = make(tmp_path)
    store.put_cheap("abc", 10, 1, {"x": 1})
    assert store.get_cheap("abc", 11, min_version=1) is None
    store.close()
