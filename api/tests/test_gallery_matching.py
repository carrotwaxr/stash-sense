"""Gallery matching (#148): score face clusters against whole performer galleries."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from recognizer import FaceRecognizer, PerformerMatch, RecognitionResult
from scene_matcher import clustered_frequency_matching, gallery_cluster_matching

DIM = 8


def _vec(**coords) -> np.ndarray:
    """Unit vector from named axes, e.g. _vec(e0=0.65, e1=0.55, e2=0.524)."""
    v = np.zeros(DIM, dtype=np.float32)
    for k, val in coords.items():
        v[int(k[1:])] = val
    return v / np.linalg.norm(v)


class _Index:
    def __init__(self, vectors: list[np.ndarray]):
        self.vectors = vectors

    def get_vectors(self, ids):
        return [self.vectors[i] for i in ids]


class _Recognizer:
    """Just what gallery matching touches: indices, faces mapping, performer metadata."""

    def __init__(self, galleries: dict[str, list[np.ndarray]]):
        vectors, faces = [], []
        for uid, vs in galleries.items():
            for v in vs:
                vectors.append(v)
                faces.append(uid)
        self.facenet_index = _Index(vectors)
        self.arcface_index = _Index(vectors)
        self.faces = faces
        self.performers = {uid: {"name": uid.split(":")[1].upper(), "country": None, "image_url": None}
                           for uid in galleries}
        self._gallery_cache = OrderedDict()
        self._face_slots = None

    GALLERY_CACHE_SIZE = FaceRecognizer.GALLERY_CACHE_SIZE
    gallery_vectors = FaceRecognizer.gallery_vectors


def _match(uid: str, distance: float) -> PerformerMatch:
    return PerformerMatch(universal_id=uid, stashdb_id=uid.split(":")[1], name=uid, country=None,
                          image_url=None, facenet_distance=distance, arcface_distance=distance,
                          combined_score=distance)


def _result(vec: np.ndarray, matches: list[tuple[str, float]]) -> RecognitionResult:
    face = SimpleNamespace(bbox={"x": 0, "y": 0, "w": 100, "h": 100}, confidence=0.9, yaw=0.0, image=None)
    return RecognitionResult(face=face, matches=[_match(u, d) for u, d in matches],
                             embedding=SimpleNamespace(facenet=vec, arcface=vec))


A, B, C = "stashdb.org:aaa", "stashdb.org:bbb", "stashdb.org:ccc"
A_VEC, B_VEC, C_VEC = _vec(e0=1), _vec(e1=1), _vec(e3=1)
# fused distance (same vector in both models) is 1 - cosine
MILD = _vec(e0=0.65, e1=0.55, e2=np.sqrt(1 - 0.65**2 - 0.55**2))  # A 0.35, B 0.45
LUCKY = _vec(e0=0.6, e1=0.8)                                       # A 0.40, B 0.20


def _mild_cluster(list_b_on_mild: bool = True):
    """Nine faces mildly closer to A, one lucky face much closer to B."""
    mild_matches = [(A, 0.35), (B, 0.45)] if list_b_on_mild else [(A, 0.35)]
    results = [(i, _result(MILD, mild_matches)) for i in range(9)]
    results.append((9, _result(LUCKY, [(B, 0.20), (A, 0.40)])))
    return results


def test_many_mild_faces_beat_one_lucky_face():
    rec = _Recognizer({A: [A_VEC], B: [B_VEC]})
    results = _mild_cluster()

    old = clustered_frequency_matching(results, rec, max_distance=0.5)
    assert old[0].best_match.stashdb_id == "bbb"  # the problem #148 fixes

    new = gallery_cluster_matching(results, rec, max_distance=0.5, min_margin=0.02, keep_fraction=0.5)
    assert len(new) == 1
    person = new[0]
    assert person.best_match.stashdb_id == "aaa"
    assert person.best_match.distance == pytest.approx(0.35, abs=1e-4)  # mean of the best 5 faces
    assert person.frame_count == 10
    assert [m.stashdb_id for m in person.all_matches] == ["aaa", "bbb"]
    assert person.all_matches[1].distance == pytest.approx(0.40, abs=1e-4)  # (0.20 + 4 x 0.45) / 5


def test_candidate_is_scored_on_faces_that_did_not_list_it():
    """B only appears in one face's list; the other nine still count via its gallery."""
    rec = _Recognizer({A: [A_VEC], B: [B_VEC]})
    person = gallery_cluster_matching(_mild_cluster(list_b_on_mild=False), rec, keep_fraction=1.0)[0]
    by_id = {m.stashdb_id: m.distance for m in person.all_matches}
    assert by_id["bbb"] == pytest.approx((0.20 + 9 * 0.45) / 10, abs=1e-4)
    assert by_id["aaa"] == pytest.approx((0.40 + 9 * 0.35) / 10, abs=1e-4)


def test_no_match_without_margin_over_runner_up():
    rec = _Recognizer({A: [A_VEC], B: [_vec(e0=1, e7=0.01)]})  # B is A's near twin
    results = [(i, _result(MILD, [(A, 0.35), (B, 0.35)])) for i in range(4)]

    persons = gallery_cluster_matching(results, rec, max_distance=0.5, min_margin=0.02)
    assert persons[0].best_match is None
    assert {m.stashdb_id for m in persons[0].all_matches} == {"aaa", "bbb"}  # still shown as candidates


def test_no_match_beyond_max_distance():
    rec = _Recognizer({A: [A_VEC]})
    results = [(i, _result(MILD, [(A, 0.35)])) for i in range(3)]
    assert gallery_cluster_matching(results, rec, max_distance=0.30)[0].best_match is None
    assert gallery_cluster_matching(results, rec, max_distance=0.40)[0].best_match.stashdb_id == "aaa"


def test_best_face_in_gallery_counts():
    """Each face is scored against its closest gallery face, not the gallery mean."""
    rec = _Recognizer({A: [A_VEC, C_VEC, C_VEC]})
    results = [(i, _result(A_VEC, [(A, 0.0)])) for i in range(3)]
    assert gallery_cluster_matching(results, rec)[0].best_match.distance == pytest.approx(0.0, abs=1e-5)


def test_a_performer_is_assigned_to_one_cluster_only():
    c_vec = _vec(e4=0.261, e5=np.sqrt(1 - 0.261**2))
    rec = _Recognizer({A: [A_VEC], C: [c_vec]})
    near_a = [(i, _result(A_VEC, [(A, 0.0)])) for i in range(5)]
    # A second person, too far from the first to share a cluster, whose
    # gallery scores put A (0.713) ahead of C (0.75). C leads the per-face
    # list so the merge step keeps the two clusters apart.
    other = _vec(e4=1, e0=0.3)
    far = [(10 + i, _result(other, [(C, 0.75), (A, 0.713)])) for i in range(2)]

    persons = gallery_cluster_matching(near_a + far, rec, max_distance=1.0, min_margin=0.0)
    assert len(persons) == 2
    assert persons[0].best_match.stashdb_id == "aaa" and persons[0].frame_count == 5
    assert persons[1].best_match.stashdb_id == "ccc"
    assert persons[1].best_match.distance == pytest.approx(0.75, abs=1e-3)


def test_tagged_performer_gets_the_boost():
    rec = _Recognizer({A: [A_VEC], B: [_vec(e0=1, e7=0.2)]})
    results = [(i, _result(A_VEC, [(A, 0.0), (B, 0.02)])) for i in range(3)]
    untagged = gallery_cluster_matching(results, rec, min_margin=0.0)
    tagged = gallery_cluster_matching(results, rec, min_margin=0.0, scene_performer_stashdb_ids=["bbb"])
    assert untagged[0].best_match.stashdb_id == "aaa"
    assert tagged[0].best_match.stashdb_id == "bbb"
    assert tagged[0].best_match.already_tagged


def test_empty_input():
    assert gallery_cluster_matching([], _Recognizer({A: [A_VEC]})) == []


def test_gallery_vectors_skip_invisible_slots_normalise_and_cache():
    rec = _Recognizer({A: [A_VEC * 3, B_VEC], B: [C_VEC]})
    rec.faces[1] = None  # an invisible slot is never part of a gallery
    fn, _af = rec.gallery_vectors(A)
    assert fn.shape == (1, DIM) and np.allclose(np.linalg.norm(fn, axis=1), 1.0)
    assert rec.gallery_vectors(A)[0] is fn
    assert rec.gallery_vectors("stashdb.org:missing") is None
