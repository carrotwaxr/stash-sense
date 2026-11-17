"""Tests for the media library mount: path mapping, mount mode, write guard."""

import os

import pytest

from media_library import (
    MediaLibrary,
    MediaNotWritableError,
    MediaPathError,
    PathMapping,
    parse_path_map,
    probe_mount,
)


class TestParsePathMap:
    def test_single_mapping(self):
        assert parse_path_map("/data=/media") == [PathMapping("/data", "/media")]

    def test_multiple_mappings_longest_prefix_first(self):
        mappings = parse_path_map("/data=/media, /data/archive=/archive;/x=/y")
        assert [m.stash_prefix for m in mappings] == ["/data/archive", "/data", "/x"]

    def test_trailing_slashes_dropped(self):
        assert parse_path_map("/data/=/media/") == [PathMapping("/data", "/media")]

    def test_empty_is_no_mapping(self):
        assert parse_path_map("") == []
        assert parse_path_map(" , ") == []

    @pytest.mark.parametrize("raw", ["/data", "data=/media", "/data=media", "=/media"])
    def test_malformed_entry_raises(self, raw):
        with pytest.raises(MediaPathError, match="MEDIA_PATH_MAP"):
            parse_path_map(raw)


class TestToLocal:
    def test_maps_prefix(self):
        lib = MediaLibrary(parse_path_map("/data=/media"))
        assert lib.to_local("/data/Studio/scene.mp4") == "/media/Studio/scene.mp4"

    def test_longest_prefix_wins(self):
        lib = MediaLibrary(parse_path_map("/data=/media,/data/archive=/archive"))
        assert lib.to_local("/data/archive/a.mp4") == "/archive/a.mp4"
        assert lib.to_local("/data/new/b.mp4") == "/media/new/b.mp4"

    def test_prefix_matches_whole_components_only(self):
        lib = MediaLibrary(parse_path_map("/data=/media"))
        with pytest.raises(MediaPathError, match="outside every MEDIA_PATH_MAP entry"):
            lib.to_local("/database/a.mp4")

    def test_root_prefix(self):
        lib = MediaLibrary(parse_path_map("/=/host"))
        assert lib.to_local("/data/a.mp4") == "/host/data/a.mp4"

    def test_no_map_uses_stash_paths_as_is(self):
        assert MediaLibrary([]).to_local("/data/a.mp4") == "/data/a.mp4"

    def test_config_error_blocks_mapping(self):
        lib = MediaLibrary.from_env({"MEDIA_PATH_MAP": "nonsense"})
        with pytest.raises(MediaPathError, match="Bad MEDIA_PATH_MAP entry"):
            lib.to_local("/data/a.mp4")


class TestProbeMount:
    def test_missing(self, tmp_path):
        status = probe_mount(str(tmp_path / "absent"))
        assert (status.mode, status.writable, status.reason) == ("missing", False, "not mounted")

    def test_writable_dir(self, tmp_path):
        status = probe_mount(str(tmp_path))
        assert (status.mode, status.readable, status.writable) == ("rw", True, True)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permission bits")
    def test_dir_without_write_permission_is_ro(self, tmp_path):
        tmp_path.chmod(0o555)
        try:
            status = probe_mount(str(tmp_path))
        finally:
            tmp_path.chmod(0o755)
        assert status.mode == "ro"
        assert "PUID/PGID" in status.reason

    def test_read_only_filesystem(self, tmp_path, monkeypatch):
        # A :ro bind mount sets ST_RDONLY even though the directory's mode bits allow writes.
        real_statvfs = os.statvfs

        class RoStat:
            def __init__(self, st):
                self._st = st
                self.f_flag = st.f_flag | os.ST_RDONLY

        monkeypatch.setattr(os, "statvfs", lambda p: RoStat(real_statvfs(p)))
        status = probe_mount(str(tmp_path))
        assert (status.mode, status.writable, status.reason) == ("ro", False, "mounted read-only")


class TestStatusAndGuard:
    def test_roots_from_map_or_media_dir(self, tmp_path):
        assert MediaLibrary(parse_path_map("/a=/x,/b=/x,/c=/y")).roots == ["/x", "/y"]
        assert MediaLibrary([], media_dir=str(tmp_path)).roots == [str(tmp_path)]

    def test_status_reports_mode(self, tmp_path):
        lib = MediaLibrary.from_env({"MEDIA_PATH_MAP": f"/data={tmp_path}"})
        status = lib.status()
        assert status["path_map"] == [{"stash_prefix": "/data", "local_prefix": str(tmp_path)}]
        assert status["mounts"][0]["mode"] == "rw"
        assert status["writable"] is True

    def test_status_not_writable_when_any_mount_missing(self, tmp_path):
        lib = MediaLibrary(parse_path_map(f"/a={tmp_path},/b={tmp_path / 'gone'}"))
        status = lib.status()
        assert [m["mode"] for m in status["mounts"]] == ["rw", "missing"]
        assert status["writable"] is False

    def test_require_writable_passes_on_rw(self, tmp_path):
        MediaLibrary([], media_dir=str(tmp_path)).require_writable("Subtitles")

    def test_require_writable_names_the_fix(self, tmp_path):
        lib = MediaLibrary([], media_dir=str(tmp_path / "absent"))
        with pytest.raises(MediaNotWritableError) as exc:
            lib.require_writable("Subtitles")
        msg = str(exc.value)
        assert "Subtitles writes files into the media library" in msg
        assert "not mounted" in msg
        assert "Add a Media Library path" in msg

    def test_require_writable_on_config_error(self):
        lib = MediaLibrary.from_env({"MEDIA_PATH_MAP": "oops"})
        with pytest.raises(MediaNotWritableError, match="Bad MEDIA_PATH_MAP entry"):
            lib.require_writable("Subtitles")
