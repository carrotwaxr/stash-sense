"""Audio bit error rate for duplicate verification (#130)."""
import numpy as np

from content_fingerprint import AUDIO_ALGO, AUDIO_ALGO_VERSION
from content_fingerprint import audio as index_audio
from content_fingerprint.store import ContentFpStore
from duplicates import audio


def _rand(seed, n):
    return np.random.default_rng(seed).integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32)


def test_ber_same_and_shifted():
    a = _rand(1, 3000)
    assert audio.audio_ber(a, a) == (0.0, 0.0)
    k = int(30 / audio.ITEM_SEC)
    b = a[k:]                       # b starts 30 s into a
    ber, off = audio.audio_ber(a, b)
    assert ber == 0.0
    assert off == 30.0
    ber, off = audio.audio_ber(b, a)   # a starts 30 s before b
    assert ber == 0.0
    assert off == -30.0


def test_ber_noise():
    ber, _ = audio.audio_ber(_rand(1, 3000), _rand(2, 3000))
    assert 0.45 < ber < 0.55


def test_low_distinct_returns_none():
    flat = np.full(3000, 12345, dtype=np.uint32)
    assert audio.audio_ber(flat, _rand(1, 3000)) is None
    assert audio.audio_ber(_rand(1, 3000), flat) is None


def test_too_short_returns_none():
    assert audio.audio_ber(_rand(1, 100), _rand(2, 3000)) is None


def test_first_seconds_of_stored_fp():
    fp = _rand(3, 5000)
    head = audio.head_items(fp, 300)
    assert len(head) == int(300 / audio.ITEM_SEC)
    assert np.array_equal(head, fp[:len(head)])
    assert len(audio.head_items(fp[:100], 300)) == 100


def test_stored_fp_reads_store(tmp_path):
    store = ContentFpStore(str(tmp_path / "fp.db"))
    fp = _rand(4, 500)
    fid = store.upsert_file("abcd1234abcd1234", 1000)
    store.put_audio(fid, AUDIO_ALGO, AUDIO_ALGO_VERSION, {}, "ok", fp)
    got = audio.stored_fp(store, "abcd1234abcd1234", 1000)
    assert np.array_equal(got, fp)
    assert audio.stored_fp(store, "ffffffffffffffff", 1000) is None
    assert audio.stored_fp(store, "abcd1234abcd1234", 999) is None
    store.close()


def test_fast_path_command_matches_index_algorithm():
    cmd = audio.fast_command("/m/a.mp4")
    index_cmd = index_audio.command("/m/a.mp4")
    assert cmd[cmd.index("-algorithm") + 1] == index_cmd[index_cmd.index("-algorithm") + 1]
    assert cmd[cmd.index("-t") + 1] == "300"
    assert cmd[cmd.index("-i") + 1] == "/m/a.mp4"
    assert cmd.index("-t") < cmd.index("-i")
    assert cmd[-1] == "-"
