"""Media library mount: Stash-to-sidecar path mapping and access mode.

Stash and the sidecar usually see the library at different container paths
(Stash at /data, the sidecar at /media). MEDIA_PATH_MAP maps one to the other:

    MEDIA_PATH_MAP=/data=/media
    MEDIA_PATH_MAP=/data=/media,/archive=/media2

Without a map, Stash paths are used as-is, which works when the library is
mounted at the same container path in both containers. MEDIA_DIR names the
mount to report when no map is set (default /media).

The library is read-only by default. Features that write files (subtitles,
later splits) call require_writable() and get a message that says what to
change in the template.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Mapping, Optional

logger = logging.getLogger(__name__)

DEFAULT_MEDIA_DIR = "/media"


class MediaPathError(ValueError):
    """A path cannot be mapped into the sidecar's media mount."""


class MediaNotWritableError(RuntimeError):
    """A write feature ran while the media mount is not writable."""


@dataclass(frozen=True)
class PathMapping:
    stash_prefix: str
    local_prefix: str


def _clean_prefix(path: str) -> str:
    path = path.strip()
    if len(path) > 1:
        path = path.rstrip("/")
    return path


def parse_path_map(raw: str) -> list[PathMapping]:
    """Parse "stash=local[,stash=local...]" into mappings, longest prefix first.

    Raises MediaPathError on a malformed entry.
    """
    mappings = []
    for entry in raw.replace(";", ",").split(","):
        if not entry.strip():
            continue
        stash, sep, local = entry.partition("=")
        stash, local = _clean_prefix(stash), _clean_prefix(local)
        if not sep or not stash.startswith("/") or not local.startswith("/"):
            raise MediaPathError(
                f"Bad MEDIA_PATH_MAP entry {entry.strip()!r}: expected /stash/path=/sidecar/path"
            )
        mappings.append(PathMapping(stash, local))
    return sorted(mappings, key=lambda m: len(m.stash_prefix), reverse=True)


def _under(path: str, prefix: str) -> bool:
    return prefix == "/" or path == prefix or path.startswith(prefix + "/")


@dataclass(frozen=True)
class MountStatus:
    path: str
    mode: str  # "rw", "ro" or "missing"
    readable: bool
    writable: bool
    reason: Optional[str] = None  # why it is not writable, when it isn't

    def to_dict(self) -> dict:
        return asdict(self)


def probe_mount(path: str) -> MountStatus:
    """Report whether a mount exists and whether this process can write to it."""
    if not os.path.isdir(path):
        return MountStatus(path, "missing", False, False, "not mounted")
    readable = os.access(path, os.R_OK | os.X_OK)
    try:
        read_only_fs = bool(os.statvfs(path).f_flag & os.ST_RDONLY)
    except OSError:
        read_only_fs = False
    if read_only_fs:
        return MountStatus(path, "ro", readable, False, "mounted read-only")
    if not os.access(path, os.W_OK):
        return MountStatus(
            path, "ro", readable, False,
            f"not writable by uid {os.getuid()} gid {os.getgid()} (check PUID/PGID)",
        )
    return MountStatus(path, "rw", readable, True)


class MediaLibrary:
    """The sidecar's view of the media library."""

    def __init__(
        self,
        mappings: list[PathMapping],
        media_dir: str = DEFAULT_MEDIA_DIR,
        config_error: Optional[str] = None,
    ):
        self.mappings = mappings
        self.media_dir = media_dir
        self.config_error = config_error

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "MediaLibrary":
        media_dir = env.get("MEDIA_DIR", DEFAULT_MEDIA_DIR)
        try:
            mappings = parse_path_map(env.get("MEDIA_PATH_MAP", ""))
        except MediaPathError as e:
            return cls([], media_dir, config_error=str(e))
        return cls(mappings, media_dir)

    @property
    def roots(self) -> list[str]:
        """Sidecar-side mount points, one per mapping (or MEDIA_DIR without a map)."""
        if self.mappings:
            return list(dict.fromkeys(m.local_prefix for m in self.mappings))
        return [self.media_dir]

    def to_local(self, stash_path: str) -> str:
        """Translate a file path as Stash reports it into the sidecar's path."""
        if self.config_error:
            raise MediaPathError(self.config_error)
        if not stash_path.startswith("/"):
            raise MediaPathError(f"{stash_path} is not an absolute path")
        # Collapse "..", "." and doubled slashes so a path can't climb out of
        # its mapping (matters once features write files).
        stash_path = os.path.normpath(stash_path)
        if not self.mappings:
            return stash_path
        for m in self.mappings:
            if _under(stash_path, m.stash_prefix):
                rest = stash_path[len(m.stash_prefix):] if m.stash_prefix != "/" else stash_path
                rest = rest.strip("/")
                local = m.local_prefix if m.local_prefix == "/" else m.local_prefix.rstrip("/")
                if not rest:
                    return local
                return local.rstrip("/") + "/" + rest
        raise MediaPathError(
            f"{stash_path} is outside every MEDIA_PATH_MAP entry "
            f"({', '.join(m.stash_prefix for m in self.mappings)})"
        )

    def mounts(self) -> list[MountStatus]:
        return [probe_mount(root) for root in self.roots]

    def status(self) -> dict:
        mounts = self.mounts()
        return {
            "path_map": [asdict(m) for m in self.mappings],
            "config_error": self.config_error,
            "mounts": [m.to_dict() for m in mounts],
            "readable": bool(mounts) and all(m.readable for m in mounts) and not self.config_error,
            "writable": bool(mounts) and all(m.writable for m in mounts) and not self.config_error,
        }

    def require_writable(self, feature: str) -> None:
        """Raise MediaNotWritableError with a fix-it message unless every mount is writable."""
        if self.config_error:
            raise MediaNotWritableError(f"{feature} cannot run: {self.config_error}")
        for m in self.mounts():
            if not m.writable:
                hint = (
                    "Set the Media Library path's Access Mode to Read/Write in the container template"
                    if m.mode != "missing"
                    else "Add a Media Library path to the container template"
                )
                raise MediaNotWritableError(
                    f"{feature} writes files into the media library, but {m.path} is {m.reason}. "
                    f"{hint}, or leave {feature} off."
                )


_library: Optional[MediaLibrary] = None


def init_media_library(env: Mapping[str, str] = os.environ) -> MediaLibrary:
    """Build the library from the environment and log its state. Called once at startup."""
    global _library
    _library = MediaLibrary.from_env(env)
    if _library.config_error:
        logger.error("Media library: %s", _library.config_error)
    for m in _library.mounts():
        detail = f" ({m.reason})" if m.reason else ""
        logger.warning("Media library: %s mode=%s%s", m.path, m.mode, detail)
    return _library


def get_media_library() -> MediaLibrary:
    """The startup instance, or one built from the environment if startup has not run."""
    return _library if _library is not None else MediaLibrary.from_env()
