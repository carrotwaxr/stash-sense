"""Video signatures: a 64-bit dHash of every keyframe, with its time, and a matcher for two signatures.

Only keyframes are decoded (``-skip_frame nokey``): for 5 min of 1080p AV1 that is 1.3 s of CPU against 48 s for a
full decode at 1 fps, about what the audio fingerprint costs. Encoders place keyframes on shot cuts, so two encodes of
the same video share many keyframe moments even when their GOP lengths differ.

Each keyframe is cropped 12% per side (logos, timestamps, letterbox edges), scaled to 9x8 grey and hashed: bit i is
set when pixel (r, c) is brighter than (r, c + 1). Near-flat frames (black, fades) carry no information and are dropped.

The signature answers the question audio can't: is this the same picture with a different soundtrack (a language
variant, a re-scored edit)?
"""

import re

import numpy as np

CROP = 0.12
MIN_RANGE = 10          # grey levels between the darkest and brightest of the 72 pixels; below this, a flat frame
_PTS = re.compile(rb"Parsed_showinfo.*? pts_time:\s*(-?[\d.]+)")
_NO_VIDEO = ("does not contain any stream", "matches no streams", "Output file is empty")


def params():
    return {"crop": CROP, "grid": "9x8", "min_range": MIN_RANGE, "frames": "keyframes"}


def command(path, ffmpeg="ffmpeg"):
    """Needs ffmpeg 5.1 or later (-fps_mode)."""
    keep = 1 - 2 * CROP
    vf = f"crop=iw*{keep}:ih*{keep},scale=9:8:flags=area,format=gray,showinfo"
    return [ffmpeg, "-hide_banner", "-nostats", "-v", "info", "-nostdin", "-skip_frame", "nokey", "-i", path,
            "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-"]


def dhash(frames):
    """uint64 hash per 8x9 grey frame (array of shape [n, 8, 9])."""
    bits = (frames[:, :, :8] > frames[:, :, 1:]).reshape(len(frames), 64)
    return np.packbits(bits, axis=1, bitorder="little").view("<u8").ravel()


def parse(returncode, stdout, stderr):
    """(status, times in ms, hashes, error) from a finished ``command``; status ok / no_video / failed."""
    err = stderr if isinstance(stderr, bytes) else (stderr or "").encode()
    tail = err.decode(errors="replace").strip()[-500:]
    if any(s.encode() in err for s in _NO_VIDEO):
        return "no_video", None, None, tail or None
    if returncode != 0:
        return "failed", None, None, tail or f"ffmpeg exit {returncode}"
    n = len(stdout) // 72
    pts = [float(m) for m in _PTS.findall(err)]
    if len(stdout) % 72 or len(pts) != n:
        return "failed", None, None, f"{len(stdout)} bytes of frames against {len(pts)} frame times"
    if n == 0:
        return "no_video", None, None, tail or "no keyframes decoded"
    frames = np.frombuffer(stdout, dtype=np.uint8).reshape(n, 8, 9).astype(np.int16)
    keep = (frames.max(axis=(1, 2)) - frames.min(axis=(1, 2))) >= MIN_RANGE
    times = np.round(np.maximum(np.asarray(pts), 0) * 1000).astype(np.uint32)[keep]
    return "ok", times, dhash(frames[keep]), None


_POP = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def _hamming(a, b):
    """Bit differences between every hash in ``a`` and every hash in ``b``: [len(a), len(b)]."""
    x = (a[:, None] ^ b[None, :]).view(np.uint8).reshape(len(a), len(b), 8)
    return _POP[x].sum(axis=2, dtype=np.uint16)


def match(sig_a, sig_b, vote_bits=4, verify_bits=10, tolerance_s=1.5, max_cells=2_000_000):
    """How much of signature a appears in b at one time offset.

    ``sig_*`` are (times in ms, hashes). Keyframe pairs within ``vote_bits`` vote for the offset (b time - a time,
    whole seconds); at the winning offset, a keyframe of a counts as matched when b has a keyframe within
    ``tolerance_s`` of the shifted time and ``verify_bits`` of its hash. Returns None when nothing votes, else:

      offset_s    b time - a time, whole seconds
      considered  a's keyframes that fall inside b at that offset
      comparable  those with a b keyframe within the tolerance (sparse GOPs leave many without one)
      matched     comparable keyframes whose hash matches too
      fraction    matched / comparable: how alike the pictures are where they can be compared
      coverage    matched / considered
    """
    ta, ha = np.asarray(sig_a[0], dtype=np.int64), np.asarray(sig_a[1], dtype=np.uint64)
    tb, hb = np.asarray(sig_b[0], dtype=np.int64), np.asarray(sig_b[1], dtype=np.uint64)
    if not len(ha) or not len(hb):
        return None
    votes = {}
    chunk = max(1, max_cells // len(hb))      # bounds the [chunk, len(b), 8] byte array to ~16 MB
    for i in range(0, len(ha), chunk):
        d = _hamming(ha[i:i + chunk], hb)
        ia, ib = np.nonzero(d <= vote_bits)
        for off in np.round((tb[ib] - ta[i + ia]) / 1000).astype(np.int64):
            votes[int(off)] = votes.get(int(off), 0) + 1
    if not votes:
        return None
    # a true offset in whole seconds can straddle two buckets
    offset = max(votes, key=lambda o: votes[o] + max(votes.get(o - 1, 0), votes.get(o + 1, 0)))
    shifted = ta + offset * 1000
    inside = (shifted >= tb.min() - tolerance_s * 1000) & (shifted <= tb.max() + tolerance_s * 1000)
    considered = int(inside.sum())
    matched = comparable = 0
    order = np.argsort(tb)
    tbs, hbs = tb[order], hb[order]
    tol = int(tolerance_s * 1000)
    for t, h in zip(shifted[inside], ha[inside]):
        lo, hi = np.searchsorted(tbs, t - tol), np.searchsorted(tbs, t + tol, "right")
        if hi > lo:
            comparable += 1
            if _hamming(np.array([h], dtype=np.uint64), hbs[lo:hi]).min() <= verify_bits:
                matched += 1
    return {"offset_s": offset, "considered": considered, "comparable": comparable, "matched": matched,
            "fraction": round(matched / comparable, 3) if comparable else 0.0,
            "coverage": round(matched / considered, 3) if considered else 0.0}
