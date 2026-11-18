"""Container index check: is the file seekable and complete (#133).

Ported from the byte parsers of the stash-curator playback check. The kind is
sniffed by magic bytes, never by extension. Reads at most the first and last
4 MB plus atom headers; never the whole file.
"""

import os
import struct

_CHUNK = 4 * 1024 * 1024
_MKV_CUES = b"\x1c\x53\xbb\x6b"
_MP4_MAGIC = (b"ftyp", b"free", b"mdat", b"moov", b"wide", b"skip")

# The only index flag that means damage; mp4_moov_last is a missing faststart.
HARD_INDEX_FLAGS = frozenset({"mp4_no_moov"})


def _mp4_atoms(f, size: int) -> list[str]:
    pos, out = 0, []
    while pos < size and len(out) < 50:
        f.seek(pos)
        h = f.read(16)
        if len(h) < 8:
            break
        sz, typ = struct.unpack(">I4s", h[:8])
        typ = typ.decode("latin1")
        if sz == 1:
            if len(h) < 16:
                break
            sz = struct.unpack(">Q", h[8:16])[0]
        elif sz == 0:
            sz = size - pos
        out.append(typ)
        if sz < 8:
            break
        pos += sz
    return out


def _avi_index(f, size: int) -> str:
    f.seek(0)
    h = f.read(12)
    if h[:4] != b"RIFF" or h[8:12] != b"AVI ":
        return "notavi"
    head = f.read(256 * 1024)
    opendml = b"indx" in head or b"ix00" in head or b"ix01" in head
    pos, has_idx1, n = 12, False, 0
    while pos + 8 <= size and n < 200:
        f.seek(pos)
        c = f.read(8)
        if len(c) < 8:
            break
        tag, sz = c[:4], struct.unpack("<I", c[4:])[0]
        if tag == b"idx1":
            has_idx1 = True
            break
        if tag == b"RIFF":
            pos += 12
            n += 1
            continue
        pos += 8 + sz + (sz & 1)
        n += 1
    if not has_idx1:  # truncated size fields: look for idx1 in the tail
        f.seek(max(0, size - _CHUNK))
        has_idx1 = b"idx1" in f.read(_CHUNK)
    return "ok" if has_idx1 else ("opendml" if opendml else "none")


def _mkv_cues(f, size: int) -> bool:
    f.seek(0)
    if _MKV_CUES in f.read(1024 * 1024):
        return True
    f.seek(max(0, size - _CHUNK))
    return _MKV_CUES in f.read(_CHUNK)


def index_status(path: str) -> dict:
    """Return {"kind": "mp4"|"avi"|"mkv"|"other", "flags": [...]}.

    Flags: mp4_no_moov, mp4_moov_last, avi_no_index, mkv_no_cues.
    Raises FileNotFoundError when the path does not exist.
    """
    size = os.path.getsize(path)
    flags: list[str] = []
    with open(path, "rb") as f:
        magic = f.read(12)
        if magic[4:8] in _MP4_MAGIC:
            kind = "mp4"
            atoms = _mp4_atoms(f, size)
            if "moov" not in atoms:
                flags.append("mp4_no_moov")
            elif "mdat" in atoms and atoms.index("moov") > atoms.index("mdat"):
                flags.append("mp4_moov_last")
        elif magic[:4] == b"RIFF" and magic[8:12] == b"AVI ":
            kind = "avi"
            if _avi_index(f, size) == "none":
                flags.append("avi_no_index")
        elif magic[:4] == b"\x1a\x45\xdf\xa3":
            kind = "mkv"
            if not _mkv_cues(f, size):
                flags.append("mkv_no_cues")
        else:
            kind = "other"
    return {"kind": kind, "flags": flags}
