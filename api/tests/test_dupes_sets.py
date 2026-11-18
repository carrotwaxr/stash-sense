"""Grouping verified duplicate pairs into sets with stable keys (#130)."""
import json

from duplicates.candidates import Candidate
from duplicates.sets import build_items
from duplicates.verdict import PairEvidence


def v(a, b, cls, ev=None, fa=None, fb=None, sources=("stash_id",)):
    lo, hi = sorted([a, b], key=int)
    cand = Candidate(lo, hi, fa, fb, frozenset(sources), False)
    return cand, ev or PairEvidence(ber=0.1), cls


def by_key(items):
    return {i.key: i for i in items}


def test_chain_forms_one_set_with_lowest_numeric_anchor():
    items, sup = build_items([v("10", "9", "duplicate"), v("9", "100", "duplicate")])
    assert [i.key for i in items] == ["duplicate:9"]
    assert items[0].scene_ids == ["9", "10", "100"]
    assert items[0].cls == "duplicate"
    assert len(items[0].pairs) == 2
    assert sup == []


def test_three_pairwise():
    items, _ = build_items([v("1", "2", "duplicate"), v("1", "3", "duplicate"), v("2", "3", "duplicate")])
    assert len(items) == 1 and items[0].key == "duplicate:1" and len(items[0].pairs) == 3


def test_pair_classes_stay_pairs():
    items, _ = build_items([v("12", "3", c) for c in ["contained"]]
                           + [v("4", "5", "partial"), v("6", "7", "language_variant"), v("8", "9", "mislabel")])
    assert [i.key for i in items] == ["contained:3-12", "language_variant:6-7", "mislabel:8-9", "partial:4-5"]
    assert items[0].scene_ids == ["3", "12"]


def test_review_pairs():
    items, _ = build_items([v("20", "4", "review")])
    assert items[0].key == "review:4-20" and items[0].cls == "review"


def test_different_produces_nothing():
    assert build_items([v("1", "2", "different")]) == ([], [])


def test_growing_set_new_anchor_supersedes_old_key():
    items, sup = build_items([v("3", "5", "duplicate"), v("5", "9", "duplicate")],
                             previous_keys={"duplicate:5": {"5", "9"}})
    assert items[0].key == "duplicate:3"
    assert sup == ["duplicate:5"]


def test_same_anchor_keeps_key():
    items, sup = build_items([v("5", "9", "duplicate"), v("9", "12", "duplicate")],
                             previous_keys={"duplicate:5": {"5", "9"}})
    assert items[0].key == "duplicate:5"
    assert sup == []


def test_two_old_sets_join():
    items, sup = build_items([v("3", "5", "duplicate"), v("5", "9", "duplicate"), v("9", "11", "duplicate")],
                             previous_keys={"duplicate:3": {"3", "5"}, "duplicate:9": {"9", "11"},
                                            "duplicate:50": {"50", "51"}, "contained:1-2": {"1", "2"}})
    assert items[0].key == "duplicate:3"
    assert sup == ["duplicate:9"]


def test_within_scene_duplicate():
    items, _ = build_items([v("7", "7", "duplicate", fa="aa", fb="bb", sources=("multi_file",))])
    assert items[0].key == "duplicate:7" and items[0].scene_ids == ["7"]
    assert items[0].pairs[0]["file_a"] == "aa"


def test_within_scene_joins_cross_scene_set():
    items, _ = build_items([v("7", "7", "duplicate"), v("7", "9", "duplicate")])
    assert len(items) == 1 and items[0].key == "duplicate:7"
    assert items[0].scene_ids == ["7", "9"] and len(items[0].pairs) == 2


def test_within_scene_other_class_produces_nothing():
    cases = [v("7", "7", c) for c in ["contained", "partial", "review", "mislabel", "language_variant"]]
    assert build_items(cases) == ([], [])


def test_pair_payload_and_moment():
    ev = PairEvidence(ber=0.1, ber_offset_s=12.5, keyframe_offset_s=3.0)
    items, _ = build_items([v("1", "2", "duplicate", ev, fa="x", fb="y", sources=("phash", "stash_id"))])
    p = items[0].pairs[0]
    assert p["scene_a"] == "1" and p["scene_b"] == "2" and p["file_a"] == "x" and p["file_b"] == "y"
    assert p["sources"] == ["phash", "stash_id"]
    assert p["cls"] == "duplicate"
    assert p["evidence"] == ev.to_dict()
    assert p["moment"] == {"t_a": 60, "t_b": 72.5}


def test_moment_keyframe_zero_and_clamp():
    items, _ = build_items([
        v("1", "2", "duplicate", PairEvidence(keyframe_offset_s=-100.0)),
        v("3", "4", "duplicate", PairEvidence()),
    ])
    d = by_key(items)
    assert d["duplicate:1"].pairs[0]["moment"] == {"t_a": 60, "t_b": 0}
    assert d["duplicate:3"].pairs[0]["moment"] == {"t_a": 60, "t_b": 60}


def test_to_details_json_safe():
    items, _ = build_items([v("1", "2", "duplicate", PairEvidence(ber=0.1, sprite=(5, 6)))])
    d = items[0].to_details()
    assert set(d) == {"cls", "scene_ids", "pairs"}
    json.dumps(d)
