"""Verdict and pair class for #130 duplicate evidence."""
import pytest

from duplicates.verdict import PairEvidence, metadata_conflict, pair_class, verdict

STRONG = (9, 5)
WEAK = (3, 1)
SDB = "https://stashdb.org/graphql"


def scene(studio=None, performers=(), ids=()):
    return {
        "studio": {"id": studio} if studio else None,
        "performers": [{"id": p} for p in performers],
        "stash_ids": [{"endpoint": e, "stash_id": s} for e, s in ids],
        "title": "t",
    }


A = scene("1", ["p1"], [(SDB, "x")])
B = scene("1", ["p1"], [(SDB, "x")])


@pytest.mark.parametrize("ev, want_verdict, want_class", [
    (PairEvidence(ber=0.1), "SAME", "duplicate"),
    (PairEvidence(ber=0.25, sprite=WEAK), "SAME", "duplicate"),
    (PairEvidence(ber=0.25), "REVIEW", "review"),
    (PairEvidence(ber=0.5, sprite=STRONG), "REVIEW", "language_variant"),
    (PairEvidence(ber=0.5, keyframe_fraction=0.3), "REVIEW", "language_variant"),
    (PairEvidence(ber=0.35), "REVIEW", "different"),
    (PairEvidence(ber=0.35, sprite=(0, 0), keyframe_fraction=0.02), "REVIEW", "different"),
    (PairEvidence(ber=0.35, keyframe_fraction=0.3), "SAME", "duplicate"),
    (PairEvidence(ber=0.35, sprite=(1, 2)), "REVIEW", "review"),
    (PairEvidence(ber=0.5), "DIFFERENT", "different"),
    (PairEvidence(sprite=STRONG), "SAME", "duplicate"),
    (PairEvidence(), "REVIEW", "review"),
])
def test_table(ev, want_verdict, want_class):
    assert verdict(ev) == want_verdict
    assert pair_class(ev, A, B) == want_class


def test_keyframe_thresholds_match_sprite_tiers():
    assert verdict(PairEvidence(ber=0.25, keyframe_fraction=0.05)) == "SAME"
    assert verdict(PairEvidence(ber=0.25, keyframe_fraction=0.04)) == "REVIEW"
    assert verdict(PairEvidence(ber=0.5, keyframe_fraction=0.15)) == "REVIEW"
    assert verdict(PairEvidence(ber=0.5, keyframe_fraction=0.14)) == "DIFFERENT"


def test_overlap_kinds():
    for kind in ("contained", "partial", "duplicate"):
        ev = PairEvidence(overlap={"kind": kind, "shared_s": 900.0, "cover_a": 1.0, "cover_b": 0.3})
        assert pair_class(ev, A, B) == kind


def test_incidental_overlap_is_different_without_video_evidence():
    ev = PairEvidence(overlap={"kind": "incidental", "shared_s": 20.0, "cover_a": 0.1, "cover_b": 0.1})
    assert pair_class(ev, A, B) == "different"
    ev.keyframe_fraction = 0.3
    assert pair_class(ev, A, B) == "duplicate"


def test_mislabel_on_stash_ids_and_studio():
    same = PairEvidence(ber=0.1)
    c = scene("1", ["p1"], [(SDB, "y")])
    assert pair_class(same, A, c) == "mislabel"
    d = scene("2", ["p1"], [(SDB, "x")])
    assert pair_class(same, A, d) == "mislabel"
    # a conflict does not matter when the pair is not the same file
    assert pair_class(PairEvidence(ber=0.6), A, d) == "different"


def test_metadata_conflict_reasons():
    assert metadata_conflict(A, B) is None
    assert metadata_conflict(A, scene("1", ["p1"], [(SDB, "y")])) == "different StashDB ids"
    assert metadata_conflict(A, scene("2", ["p1"], [(SDB, "x")])) == "different studios"
    assert metadata_conflict(A, scene("1", ["p9"], [(SDB, "x")])) == "no performer in common"
    # missing data is not a conflict
    assert metadata_conflict(A, scene()) is None
    # other endpoints, or one shared id among several, are not conflicts
    assert metadata_conflict(A, scene("1", ["p1"], [("https://pmv.example/graphql", "z")])) is None
    assert metadata_conflict(scene(ids=[(SDB, "x"), (SDB, "y")]), scene(ids=[(SDB, "y")])) is None


def test_to_dict_round_trips_fields():
    ev = PairEvidence(ber=0.1, ber_offset_s=2.5, sprite=(9, 5))
    d = ev.to_dict()
    assert d["ber"] == 0.1 and d["ber_offset_s"] == 2.5 and d["sprite"] == (9, 5) or d["sprite"] == [9, 5]
    assert d["keyframe_fraction"] is None and d["overlap"] is None
