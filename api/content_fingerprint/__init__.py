"""Content fingerprints: what a file sounds and looks like, for finding the same content across files.

Audio fingerprints (Chromaprint raw, whole file) find a clip inside a longer file and its offset: a split scene inside
a DVD file, re-cuts, trimmed re-releases, multi-part pieces. Video signatures (one hash per second) catch what audio
can't, e.g. the same video with another language's audio.

This package uses only the standard library and numpy, and never imports the rest of stash-sense, so it also runs as a
standalone batch worker (``python -m content_fingerprint.cli``) on another machine. Rows are keyed by file content
(oshash + size), so a store built elsewhere imports into stash-sense as a plain row copy.
"""

AUDIO_ALGO = "chromaprint_raw"
AUDIO_ALGO_VERSION = 1
