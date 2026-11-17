"""Ground-truth distances: how close is a scene face to the right performer?

Production only sees the index's top-k neighbours, so when the right
performer loses we don't learn by how much. This module reads the expected
performer's own DB faces straight from the Voyager indices and computes the
same fused distance production uses (weighted FaceNet + ArcFace cosine).

Identity is compared by internal performer id: a performer can carry ids on
several stash-boxes, and a match keyed by any of them is the right person.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import face_config


@dataclass
class PerformerGallery:
    """One expected performer's DB entry: ids and face vectors."""
    stashdb_id: str
    performer_id: Optional[int]
    name: Optional[str] = None
    stashbox_ids: set[str] = field(default_factory=set)  # bare uuids, any endpoint
    facenet: Optional[np.ndarray] = None  # (n, 512), L2-normalised rows
    arcface: Optional[np.ndarray] = None

    @property
    def face_count(self) -> int:
        return 0 if self.facenet is None else len(self.facenet)

    @property
    def in_db(self) -> bool:
        return self.performer_id is not None and self.face_count > 0

    def is_match(self, match_stashdb_id: str) -> bool:
        """True when a production match (bare uuid) names this performer."""
        return match_stashdb_id == self.stashdb_id or match_stashdb_id in self.stashbox_ids


def _normalise(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def fused_distances(
    facenet: np.ndarray,
    arcface: np.ndarray,
    gallery: PerformerGallery,
    facenet_weight: float = face_config.FACENET_WEIGHT,
    arcface_weight: float = face_config.ARCFACE_WEIGHT,
) -> np.ndarray:
    """Fused cosine distance from one face to each of the gallery's faces."""
    fn = _normalise(facenet)[0]
    af = _normalise(arcface)[0]
    fn_d = 1.0 - gallery.facenet @ fn
    af_d = 1.0 - gallery.arcface @ af
    return facenet_weight * fn_d + arcface_weight * af_d


def distance_to(
    facenet: np.ndarray,
    arcface: np.ndarray,
    gallery: PerformerGallery,
    facenet_weight: float = face_config.FACENET_WEIGHT,
    arcface_weight: float = face_config.ARCFACE_WEIGHT,
) -> Optional[float]:
    """Best (smallest) fused distance to any of the performer's faces."""
    if not gallery.in_db:
        return None
    return float(fused_distances(facenet, arcface, gallery, facenet_weight, arcface_weight).min())


class GalleryStore:
    """Loads and caches galleries from performers.db plus the two indices."""

    def __init__(self, db_reader, facenet_index, arcface_index, faces_mapping=None):
        self.db = db_reader
        self.facenet_index = facenet_index
        self.arcface_index = arcface_index
        # index -> universal_id (faces.json). Slots that are None are invisible
        # to production matching, so they are left out of the gallery too.
        self.faces_mapping = faces_mapping
        self._cache: dict[str, PerformerGallery] = {}

    def get(self, stashdb_id: str) -> PerformerGallery:
        if stashdb_id not in self._cache:
            self._cache[stashdb_id] = self._load(stashdb_id)
        return self._cache[stashdb_id]

    def _load(self, stashdb_id: str) -> PerformerGallery:
        performer = self.db.get_performer_by_stashbox_id("stashdb", stashdb_id)
        if performer is None:
            return PerformerGallery(stashdb_id=stashdb_id, performer_id=None)
        ids = {s.stashbox_performer_id for s in self.db.get_stashbox_ids(performer.id)}
        faces = [f for f in self.db.get_faces(performer.id) if self._visible(f.facenet_index)]
        gallery = PerformerGallery(
            stashdb_id=stashdb_id,
            performer_id=performer.id,
            name=performer.canonical_name,
            stashbox_ids=ids,
        )
        if faces:
            fn = np.asarray(self.facenet_index.get_vectors([f.facenet_index for f in faces]))
            af = np.asarray(self.arcface_index.get_vectors([f.arcface_index for f in faces]))
            gallery.facenet = _normalise(fn)
            gallery.arcface = _normalise(af)
        return gallery

    def _visible(self, index: int) -> bool:
        if self.faces_mapping is None:
            return True
        return 0 <= index < len(self.faces_mapping) and self.faces_mapping[index] is not None
