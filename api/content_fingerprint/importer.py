"""Importing a store built elsewhere (the standalone CLI on another machine) into this one.

A fingerprint is only worth importing when this machine's ffmpeg would have produced the same bytes: the overlaps
are found by exact value lookups, so a store mixing two decoders matches worse. Before copying anything, a sample of
the source's files is fingerprinted here and compared byte for byte. The sample takes a few files of every file
extension, because decoders differ per codec (ffmpeg 4.4 and 6.1 agree on AAC and MP3 but not on WMA).

Video signatures are not imported: no store built elsewhere has them yet.
"""

import json
import os
import random
import sqlite3

from . import AUDIO_ALGO, AUDIO_ALGO_VERSION, schema


class ImportRefused(RuntimeError):
    """The source store doesn't match this machine's fingerprints; nothing was copied."""


def _ext(path):
    return os.path.splitext(path or "")[1].lower() or "(none)"


def open_source(path):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT version FROM content_fp_schema_version").fetchone()
    except sqlite3.DatabaseError as e:
        conn.close()
        raise ImportRefused(f"{os.path.basename(path)} is not a content fingerprint store: {e}") from e
    if row is None or row[0] > schema.CONTENT_FP_SCHEMA_VERSION:
        conn.close()
        raise ImportRefused(f"{path} is content store schema v{row and row[0]}, this code reads up to "
                            f"v{schema.CONTENT_FP_SCHEMA_VERSION}")
    return conn


def source_keys(conn):
    """{(oshash, size): params JSON} for the source's ok fingerprints of the current version (no blobs: a store for
    a whole library holds gigabytes of them)."""
    return {(o, s): p for o, s, p in conn.execute(
        """SELECT f.oshash, f.size_bytes, a.params FROM content_files f
           JOIN audio_fingerprints a ON a.file_id = f.id
           WHERE a.algo = ? AND a.algo_version = ? AND a.status = 'ok'""", (AUDIO_ALGO, AUDIO_ALGO_VERSION))}


def source_fingerprint(conn, key):
    row = conn.execute(
        """SELECT a.fp FROM content_files f JOIN audio_fingerprints a ON a.file_id = f.id
           WHERE f.oshash = ? AND f.size_bytes = ? AND a.algo = ? AND a.algo_version = ? AND a.status = 'ok'""",
        (*key, AUDIO_ALGO, AUDIO_ALGO_VERSION)).fetchone()
    return row[0]


def pick_sample(paths_by_key, per_ext=3, total=24, seed=0):
    """Up to ``per_ext`` keys of every extension, deterministic. Taken round by round (one of each extension, then a
    second of each, ...) up to ``total``, and never fewer than one of each extension."""
    by_ext = {}
    for key, path in sorted(paths_by_key.items()):
        by_ext.setdefault(_ext(path), []).append(key)
    rnd = random.Random(seed)
    shuffled = {ext: rnd.sample(keys, min(per_ext, len(keys))) for ext, keys in sorted(by_ext.items())}
    limit = max(total, len(by_ext))
    picked = []
    for round_ in range(per_ext):
        for ext in sorted(shuffled):
            if round_ < len(shuffled[ext]) and len(picked) < limit:
                picked.append(shuffled[ext][round_])
    return picked


async def check_and_import(src_path, store, local_paths, fingerprint, local_ffmpeg, min_checked=3):
    """Import ``src_path`` into ``store`` after the build-equality check.

    local_paths: {(oshash, size): path this machine opens} for the library's current files. Only those files are
    imported (the source may hold files deleted since). fingerprint: async path -> (status, fp, error).

    Returns {"imported", "in_library", "checked", "source_ffmpeg"}; raises ImportRefused, copying nothing, when a
    sampled file differs or too few could be checked.
    """
    conn = open_source(src_path)
    try:
        src = source_keys(conn)
        common = {k: local_paths[k] for k in src if k in local_paths}
        if not common:
            raise ImportRefused(f"none of the {len(src)} fingerprints in {os.path.basename(src_path)} belong to a "
                                "file in this library")
        source_ffmpeg = sorted({json.loads(p).get("ffmpeg", "unknown") for p in src.values()})
        checked, differ, unreadable = 0, [], 0
        for key in pick_sample(common):
            status, fp, _err = await fingerprint(common[key])
            if status != "ok":
                unreadable += 1
                continue
            checked += 1
            if fp.tobytes() != source_fingerprint(conn, key):
                differ.append(_ext(common[key]))
        if differ:
            raise ImportRefused(
                f"{len(differ)} of {checked} sampled files fingerprint differently here (extensions: "
                f"{', '.join(sorted(set(differ)))}); the store was built with ffmpeg {', '.join(source_ffmpeg)}, "
                f"this container has {local_ffmpeg}. Nothing imported; the job fingerprints these files itself.")
        if checked < min(min_checked, len(common)):
            raise ImportRefused(f"only {checked} sampled files could be read here ({unreadable} unreadable); "
                                "check the media mount and retry")
        imported = store.copy_audio_from(conn, list(common), AUDIO_ALGO, AUDIO_ALGO_VERSION)
        return {"imported": imported, "in_library": len(common), "source_files": len(src), "checked": checked,
                "source_ffmpeg": source_ffmpeg}
    finally:
        conn.close()
