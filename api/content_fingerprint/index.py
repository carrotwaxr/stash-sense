"""Finding where one file's audio appears in others: an inverted index over fingerprint values, offset voting, and
window-by-window verification.

The index is derived data, kept as numpy arrays beside the store rather than in SQLite (hundreds of millions of
postings for a full library). Only every ``stride``-th position of each file is indexed; a query looks up every one of
its own positions, so any alignment is still found.
"""

import os

import numpy as np

# Chromaprint (ffmpeg muxer): 11025 Hz, a frame every 4096/3 samples
ITEM_SECONDS = 4096 / 3 / 11025


class Index:
    """values (sorted uint32), file ids (uint32) and positions (uint32), one entry per indexed item."""

    def __init__(self, values, file_ids, positions):
        self.values, self.file_ids, self.positions = values, file_ids, positions

    def __len__(self):
        return len(self.values)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for name in ("values", "file_ids", "positions"):
            np.save(os.path.join(path, f"{name}.npy"), getattr(self, name))

    @classmethod
    def load(cls, path, mmap=True):
        mode = "r" if mmap else None
        return cls(*(np.load(os.path.join(path, f"{n}.npy"), mmap_mode=mode) for n in ("values", "file_ids", "positions")))


def build(entries, stride=4, max_files_per_value=50):
    """An Index over ``entries`` [(file_id, uint32 fingerprint)].

    A value found in more than ``max_files_per_value`` files (silence, common intros and stings) is dropped: it says
    nothing about which file a clip came from, and it would dominate every query.
    """
    vals, fids, poss = [], [], []
    for fid, fp in entries:
        pos = np.arange(0, len(fp), stride, dtype=np.uint32)
        vals.append(np.asarray(fp, dtype=np.uint32)[pos])
        fids.append(np.full(len(pos), fid, dtype=np.uint32))
        poss.append(pos)
    if not vals:
        return Index(*(np.zeros(0, dtype=np.uint32) for _ in range(3)))
    v, f, p = np.concatenate(vals), np.concatenate(fids), np.concatenate(poss)
    order = np.lexsort((f, v))
    v, f, p = v[order], f[order], p[order]
    # files per value: count distinct (value, file) pairs
    first_of_pair = np.ones(len(v), dtype=bool)
    first_of_pair[1:] = (v[1:] != v[:-1]) | (f[1:] != f[:-1])
    uv, inv = np.unique(v, return_inverse=True)
    files_per_value = np.bincount(inv, weights=first_of_pair, minlength=len(uv))
    keep = files_per_value[inv] <= max_files_per_value
    return Index(v[keep], f[keep], p[keep])


def candidates(index, fp, min_votes=10, bucket=2, exclude=None, max_hits=50_000_000):
    """[(file_id, offset, votes)] best first: files whose audio lines up with ``fp`` at ``offset``
    (``fp[i]`` ~ ``other[i + offset]``). Offsets are voted in buckets of ``bucket`` items."""
    fp = np.asarray(fp, dtype=np.uint32)
    lo = np.searchsorted(index.values, fp, "left")
    hi = np.searchsorted(index.values, fp, "right")
    n = hi - lo
    sel = np.nonzero(n)[0]
    counts = n[sel]
    total = int(counts.sum())
    if total == 0:
        return []
    if total > max_hits:
        raise ValueError(f"{total} index hits: raise max_hits or lower max_files_per_value")
    starts = np.repeat(lo[sel], counts)
    within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    post = starts + within
    qpos = np.repeat(sel, counts).astype(np.int64)
    files = np.asarray(index.file_ids[post], dtype=np.int64)
    offs = np.asarray(index.positions[post], dtype=np.int64) - qpos
    keys = (files << 32) | ((offs // bucket) + (1 << 31))
    uk, votes = np.unique(keys, return_counts=True)
    out = {}
    for k, c in zip(uk[votes >= max(1, min_votes // 2)], votes[votes >= max(1, min_votes // 2)]):
        fid = int(k >> 32)
        if exclude is not None and fid in exclude:
            continue
        b = int(k & 0xFFFFFFFF) - (1 << 31)
        out[(fid, b)] = int(c)
    merged = []
    for (fid, b), c in out.items():   # a true offset can straddle two buckets
        total_votes = c + out.get((fid, b + 1), 0)
        if total_votes >= min_votes and out.get((fid, b - 1), 0) <= c:
            merged.append((fid, b * bucket, total_votes))
    best = {}
    for fid, off, v in sorted(merged, key=lambda x: -x[2]):
        best.setdefault(fid, []).append((off, v))
    # several offsets per file are legitimate (a scene repeated, or present twice); keep those far apart
    result = []
    for fid, offs_votes in best.items():
        kept = []
        for off, v in offs_votes:
            if all(abs(off - k) > 40 for k, _ in kept):
                kept.append((off, v))
        result += [(fid, off, v) for off, v in kept]
    return sorted(result, key=lambda x: -x[2])


def _bit_errors(a, b):
    return np.unpackbits((a ^ b).view(np.uint8)).reshape(-1, 32).sum(axis=1)


def best_offset(a, b, offset, search=4):
    """The offset within ``search`` of ``offset`` with the lowest bit error rate over the overlap."""
    best, best_ber = offset, 1.0
    for off in range(offset - search, offset + search + 1):
        i0, i1 = max(0, -off), min(len(a), len(b) - off)
        if i1 - i0 < 16:
            continue
        ber = _bit_errors(a[i0:i1], b[i0 + off:i1 + off]).mean() / 32
        if ber < best_ber:
            best, best_ber = off, ber
    return best, best_ber


def matched_runs(a, b, offset, window=32, max_ber=0.25, min_windows=3, gap=1):
    """Stretches where ``a[i]`` matches ``b[i + offset]``: [(a_start, a_end, b_start, b_end, ber)] in items.

    Compared in windows of ``window`` items (~4 s); a run is consecutive windows at or under ``max_ber``, allowing
    ``gap`` bad windows inside it (a glitch, a dropped frame), and at least ``min_windows`` long.
    """
    a = np.asarray(a, dtype=np.uint32)
    b = np.asarray(b, dtype=np.uint32)
    i0, i1 = max(0, -offset), min(len(a), len(b) - offset)
    nwin = (i1 - i0) // window
    if nwin < min_windows:
        return []
    errs = _bit_errors(a[i0:i0 + nwin * window], b[i0 + offset:i0 + offset + nwin * window])
    wber = errs.reshape(nwin, window).sum(axis=1) / (32 * window)
    good = wber <= max_ber
    runs, start, bad = [], None, 0
    for w in range(nwin + 1):
        ok = w < nwin and good[w]
        if ok:
            if start is None:
                start = w
            bad, last_good = 0, w
        elif start is not None:
            bad += 1
            if bad > gap or w == nwin:
                end = last_good + 1
                if end - start >= min_windows:
                    s, e = i0 + start * window, i0 + end * window
                    runs.append((s, e, s + offset, e + offset, float(wber[start:end].mean())))
                start, bad = None, 0
    return runs
