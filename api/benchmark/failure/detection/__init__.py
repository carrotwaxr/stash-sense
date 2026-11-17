"""Detection test set: measure the face detector's recall on its own.

Three steps, all working in one directory (default
``$DATA_DIR/benchmark/detection``):

1. ``sample.sample_frames`` pulls frames from benchmark scenes and pre-draws
   proposal boxes with a permissive detector (``frames/``, ``proposals.json``).
2. ``labeler.serve`` is a small local web page for accepting, deleting and
   drawing face boxes (``labels.json``).
3. ``score.score`` runs the production detector on the reviewed frames and
   reports recall and precision by face size (``scores/<timestamp>/``).

Nothing here imports ML modules at import time, so labeling works on a
machine without them.
"""

import json
from pathlib import Path

FORMAT_VERSION = 1
FRAMES_DIR = "frames"
PROPOSALS_FILE = "proposals.json"
LABELS_FILE = "labels.json"
SCORES_DIR = "scores"


def load_proposals(out_dir: Path) -> dict:
    path = Path(out_dir) / PROPOSALS_FILE
    data = json.loads(path.read_text())
    if data.get("version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported proposals version {data.get('version')}")
    return data
