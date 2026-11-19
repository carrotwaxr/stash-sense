"""Detection test set: sampler, labeler server and scorer (issue #139).

Everything here runs on fakes: no models, no ffmpeg, no network.
"""

import http.client
import json
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from benchmark.failure.detection import labeler, sample, score
from benchmark.failure.scene_set import BenchmarkScene, BenchmarkSet
from benchmark.failure.source import VideoSource


def box(x, y, w, h, **extra):
    return {"x": x, "y": y, "w": w, "h": h, **extra}


# ---------------------------------------------------------------- IoU / match

class TestIou:
    def test_identical_is_one(self):
        assert score.iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) == pytest.approx(1.0)

    def test_disjoint_is_zero(self):
        assert score.iou(box(0, 0, 10, 10), box(20, 20, 5, 5)) == 0.0

    def test_touching_edges_is_zero(self):
        assert score.iou(box(0, 0, 10, 10), box(10, 0, 10, 10)) == 0.0

    def test_zero_area_is_zero(self):
        assert score.iou(box(0, 0, 0, 0), box(0, 0, 0, 0)) == 0.0
        assert score.iou(box(0, 0, 10, 0), box(0, 0, 10, 10)) == 0.0

    def test_partial_overlap(self):
        # 5x10 overlap, union 150
        assert score.iou(box(0, 0, 10, 10), box(5, 0, 10, 10)) == pytest.approx(50 / 150)

    def test_contained(self):
        assert score.iou(box(0, 0, 10, 10), box(0, 0, 10, 5)) == pytest.approx(0.5)


class TestGreedyMatch:
    def test_empty_inputs(self):
        assert score.greedy_match([], [box(0, 0, 5, 5)], 0.5) == []
        assert score.greedy_match([box(0, 0, 5, 5)], [], 0.5) == []

    def test_threshold_is_inclusive(self):
        pairs = score.greedy_match([box(0, 0, 10, 10)], [box(0, 0, 10, 5)], 0.5)
        assert [(li, pi) for li, pi, _ in pairs] == [(0, 0)]

    def test_below_threshold_unmatched(self):
        assert score.greedy_match([box(0, 0, 10, 10)], [box(5, 0, 10, 10)], 0.5) == []

    def test_highest_iou_first_not_label_order(self):
        # One prediction overlaps both labels; it goes to label 1 (IoU .82)
        # rather than label 0 (IoU .67), even though label 0 comes first.
        labels = [box(0, 0, 10, 10), box(3, 0, 10, 10)]
        pairs = score.greedy_match(labels, [box(2, 0, 10, 10)], 0.5)
        assert [(li, pi) for li, pi, _ in pairs] == [(1, 0)]

    def test_one_to_one(self):
        # Two predictions on one label: only the better one matches.
        labels = [box(0, 0, 10, 10)]
        preds = [box(1, 0, 10, 10), box(0, 0, 10, 10)]
        pairs = score.greedy_match(labels, preds, 0.5)
        assert [(li, pi) for li, pi, _ in pairs] == [(0, 1)]

    def test_two_labels_two_preds(self):
        labels = [box(0, 0, 10, 10), box(5, 0, 10, 10)]
        preds = [box(0, 0, 10, 10), box(4, 0, 10, 10)]
        pairs = score.greedy_match(labels, preds, 0.3)
        assert sorted((li, pi) for li, pi, _ in pairs) == [(0, 0), (1, 1)]


class TestSizeBucket:
    @pytest.mark.parametrize("w,h,bucket", [
        (10, 10, "<40"),
        (39, 200, "<40"),
        (200, 39, "<40"),
        (40, 40, "40-79"),
        (79, 300, "40-79"),
        (80, 80, "80+"),
        (300, 500, "80+"),
    ])
    def test_short_side(self, w, h, bucket):
        assert score.size_bucket(box(0, 0, w, h)) == bucket


# ---------------------------------------------------------------- labels I/O

class TestLabelsIo:
    def test_missing_file_is_empty(self, tmp_path):
        assert labeler.load_labels(tmp_path) == {"version": 1, "frames": {}}

    def test_round_trip(self, tmp_path):
        data = {"version": 1, "frames": {"a.jpg": {
            "reviewed": True, "boxes": [box(1, 2, 3, 4)], "updated_at": "t"}}}
        labeler.save_labels(tmp_path, data)
        assert labeler.load_labels(tmp_path) == data

    def test_update_label_merges(self, tmp_path):
        labeler.update_label(tmp_path, "a.jpg", [box(1, 2, 3, 4)], True)
        labeler.update_label(tmp_path, "b.jpg", [], True)
        frames = labeler.load_labels(tmp_path)["frames"]
        assert set(frames) == {"a.jpg", "b.jpg"}
        assert frames["a.jpg"]["boxes"] == [box(1, 2, 3, 4)]
        assert frames["a.jpg"]["reviewed"] is True
        assert frames["a.jpg"]["updated_at"]

    def test_failed_write_keeps_old_file(self, tmp_path, monkeypatch):
        labeler.update_label(tmp_path, "a.jpg", [box(1, 2, 3, 4)], True)
        before = (tmp_path / "labels.json").read_text()

        def boom(*a, **k):
            raise OSError("disk full")
        monkeypatch.setattr(labeler.os, "replace", boom)
        with pytest.raises(OSError):
            labeler.update_label(tmp_path, "b.jpg", [], True)
        assert (tmp_path / "labels.json").read_text() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ["labels.json"]


# ---------------------------------------------------------------- sampler

def make_scene(scene_id, resolution="SD", duration=100.0):
    return BenchmarkScene(
        scene_id=scene_id, title="", date="2012-01-01", era="2010-2014",
        resolution=resolution, width=640, height=480, duration=duration,
        path=f"/media/{scene_id}.mp4",
    )


def make_set(*scenes):
    return BenchmarkSet(seed=1, created_at="now", scenes=list(scenes))


class FakeVideo:
    def __init__(self, fail_scenes=()):
        self.fail_inputs = {f"vid-{s}" for s in fail_scenes}
        self.calls = []

    def resolve(self, stash_path, scene_id, stash_url, api_key):
        return VideoSource(input=f"vid-{scene_id}", api_key=None, local=True)

    def extract(self, input, timestamp, api_key):
        self.calls.append((input, timestamp))
        if input in self.fail_inputs:
            return None
        return np.full((48, 64, 3), 100, dtype=np.uint8)


def fake_propose(image):
    return [box(1, 2, 10, 12, score=0.9), box(30, 20, 8, 8, score=0.25)]


def run_sample(out_dir, bench, video=None, **kw):
    video = video or FakeVideo()
    return sample.sample_frames(
        bench, out_dir, extract_frame=video.extract, propose=fake_propose,
        resolve_source=video.resolve, **kw)


class TestSampleFrames:
    def test_spreads_over_matching_scenes_only(self, tmp_path):
        bench = make_set(make_scene("1"), make_scene("2"), make_scene("3", "720p"))
        path = run_sample(tmp_path, bench, count=5)
        data = json.loads(path.read_text())
        assert path == tmp_path / "proposals.json"
        assert data["version"] == 1
        assert data["seed"] == 139
        assert data["proposal_detector"] == {"det_thresh": 0.2, "det_size": 1280}
        frames = data["frames"]
        assert [f["scene_id"] for f in frames] == ["1", "1", "1", "2", "2"]
        for f in frames:
            assert 5.0 <= f["timestamp"] <= 95.0
            assert (f["width"], f["height"]) == (64, 48)
            assert f["resolution"] == "SD"
            assert f["proposals"] == fake_propose(None)
            assert f["file"] == f"{f['scene_id']}_{round(f['timestamp'] * 1000)}.jpg"
            assert (tmp_path / "frames" / f["file"]).is_file()

    def test_deterministic_for_seed(self, tmp_path):
        bench = make_set(make_scene("1"), make_scene("2"))
        a = json.loads(run_sample(tmp_path / "a", bench, count=6).read_text())
        b = json.loads(run_sample(tmp_path / "b", bench, count=6).read_text())
        c = json.loads(run_sample(tmp_path / "c", bench, count=6, seed=7).read_text())
        def files(d):
            return [f["file"] for f in d["frames"]]
        assert files(a) == files(b)
        assert files(a) != files(c)

    def test_skips_failed_extraction(self, tmp_path):
        bench = make_set(make_scene("1"), make_scene("2"), make_scene("3"))
        video = FakeVideo(fail_scenes={"2"})
        data = json.loads(run_sample(tmp_path, bench, video=video, count=3).read_text())
        assert [f["scene_id"] for f in data["frames"]] == ["1", "3"]
        assert any(inp == "vid-2" for inp, _ in video.calls)

    def test_resolutions_filter(self, tmp_path):
        bench = make_set(make_scene("1"), make_scene("2", "720p"))
        data = json.loads(run_sample(tmp_path, bench, count=2, resolutions=("720p",)).read_text())
        assert [f["scene_id"] for f in data["frames"]] == ["2", "2"]

    def test_refuses_existing_without_force_and_keeps_labels(self, tmp_path):
        bench = make_set(make_scene("1"))
        run_sample(tmp_path, bench, count=2)
        labels = tmp_path / "labels.json"
        labels.write_text('{"version": 1, "frames": {"x.jpg": {}}}')
        with pytest.raises(FileExistsError):
            run_sample(tmp_path, bench, count=2)
        run_sample(tmp_path, bench, count=3, force=True)
        assert labels.read_text() == '{"version": 1, "frames": {"x.jpg": {}}}'
        assert len(json.loads((tmp_path / "proposals.json").read_text())["frames"]) == 3


# ---------------------------------------------------------------- fixtures

def write_detection_dir(out_dir: Path, sizes, labels=None):
    """Proposals plus one JPEG per (width, height); frames named f0.jpg..."""
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True)
    frames = []
    for i, (w, h) in enumerate(sizes):
        name = f"f{i}.jpg"
        Image.fromarray(np.full((h, w, 3), 90, dtype=np.uint8)).save(frames_dir / name, quality=95)
        frames.append({"file": name, "scene_id": str(i), "timestamp": 1.0, "width": w,
                       "height": h, "resolution": "SD",
                       "proposals": [box(2, 2, 10, 10, score=0.9), box(20, 20, 8, 8, score=0.1)]})
    (out_dir / "proposals.json").write_text(json.dumps({
        "version": 1, "created_at": "now", "seed": 139,
        "proposal_detector": {"det_thresh": 0.2, "det_size": 1280}, "frames": frames}))
    if labels is not None:
        (out_dir / "labels.json").write_text(json.dumps({"version": 1, "frames": labels}))


# ---------------------------------------------------------------- scorer

class TestScore:
    def test_end_to_end(self, tmp_path):
        labels = {
            "f0.jpg": {"reviewed": True, "updated_at": "t",
                       "boxes": [box(10, 10, 20, 20), box(50, 10, 45, 45)]},
            "f1.jpg": {"reviewed": True, "updated_at": "t", "boxes": [box(10, 5, 85, 80)]},
            "f2.jpg": {"reviewed": False, "updated_at": "t", "boxes": [box(0, 0, 50, 50)]},
        }
        write_detection_dir(tmp_path, [(100, 80), (120, 90), (140, 100)], labels)
        preds_by_width = {
            100: [box(11, 11, 20, 20, score=0.9),        # matches tiny label, not usable
                  box(0, 60, 10, 10, score=0.8),         # false positive
                  box(50, 10, 45, 45, score=0.3)],       # below min_confidence
            120: [box(12, 6, 84, 80, score=0.95)],       # matches, usable
        }
        seen = []

        def detect(image):
            seen.append(image.shape[1])
            return preds_by_width[image.shape[1]]

        report = score.score(tmp_path, detect=detect, det_size=960)
        assert sorted(seen) == [100, 120]
        assert report.name == "report.md"
        assert report.parent.parent == tmp_path / "scores"

        results = json.loads((report.parent / "results.json").read_text())
        assert results["det_size"] == 960
        assert results["min_confidence"] == 0.5
        assert results["iou_threshold"] == 0.5
        assert results["min_face_size"] == 40
        assert results["reviewed_frames"] == 2
        assert results["labels"] == 3
        assert results["predictions"] == 3
        assert results["matched"] == 2
        assert results["recall"] == pytest.approx(2 / 3)
        assert results["precision"] == pytest.approx(2 / 3)
        assert results["usable_recall"] == pytest.approx(1 / 3)
        by = results["recall_by_size"]
        assert (by["<40"]["labels"], by["<40"]["matched"]) == (1, 1)
        assert (by["40-79"]["labels"], by["40-79"]["matched"]) == (1, 0)
        assert (by["80+"]["labels"], by["80+"]["matched"]) == (1, 1)
        assert results["missed"] == [{"file": "f0.jpg", "box": box(50, 10, 45, 45), "size": "40-79"}]

        md = report.read_text()
        for needle in ("960", "0.5", "Reviewed frames", "f0.jpg", "Usable recall"):
            assert needle in md

    def test_missed_list_capped_in_markdown_only(self, tmp_path):
        labels = {"f0.jpg": {"reviewed": True, "updated_at": "t",
                             "boxes": [box(i, 0, 1, 1) for i in range(60)]}}
        write_detection_dir(tmp_path, [(100, 80)], labels)
        report = score.score(tmp_path, detect=lambda image: [])
        results = json.loads((report.parent / "results.json").read_text())
        assert len(results["missed"]) == 60
        md = report.read_text()
        assert md.count("| f0.jpg |") == 50
        assert "10 more" in md

    def test_no_reviewed_frames_raises(self, tmp_path):
        write_detection_dir(tmp_path, [(100, 80)], {})
        with pytest.raises(ValueError):
            score.score(tmp_path, detect=lambda image: [])


# ---------------------------------------------------------------- HTTP server

@pytest.fixture
def server(tmp_path):
    write_detection_dir(tmp_path, [(100, 80), (120, 90)])
    (tmp_path / "frames" / "secret.jpg").write_bytes(b"not listed")
    srv = labeler.make_server(tmp_path, "127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield tmp_path, srv.server_address[1]
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=5)


def request(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Content-Type": "application/json"} if body is not None else {}
    conn.request(method, path, body=body, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    return resp.status, resp.getheader("Content-Type"), data


class TestLabelerServer:
    def test_index(self, server):
        _, port = server
        status, ctype, data = request(port, "GET", "/")
        assert status == 200
        assert ctype.startswith("text/html")
        assert b"at least one eye is visible" in data

    def test_state(self, server):
        _, port = server
        status, _, data = request(port, "GET", "/api/state")
        state = json.loads(data)
        assert status == 200
        assert [f["file"] for f in state["frames"]] == ["f0.jpg", "f1.jpg"]
        assert state["frames"][0]["reviewed"] is False
        assert state["frames"][0]["boxes"] == []
        assert len(state["frames"][0]["proposals"]) == 2
        assert (state["reviewed"], state["total"]) == (0, 2)

    def test_post_label_persists(self, server):
        out_dir, port = server
        body = json.dumps({"file": "f1.jpg", "boxes": [box(1, 2, 30, 40)], "reviewed": True})
        status, _, _ = request(port, "POST", "/api/label", body)
        assert status == 200
        saved = json.loads((out_dir / "labels.json").read_text())
        assert saved["frames"]["f1.jpg"]["boxes"] == [box(1, 2, 30, 40)]
        state = json.loads(request(port, "GET", "/api/state")[2])
        assert state["frames"][1]["reviewed"] is True
        assert state["frames"][1]["boxes"] == [box(1, 2, 30, 40)]
        assert state["reviewed"] == 1

    @pytest.mark.parametrize("payload", [
        "not json",
        json.dumps({"file": "nope.jpg", "boxes": [], "reviewed": True}),
        json.dumps({"file": "../proposals.json", "boxes": [], "reviewed": True}),
        json.dumps({"file": "f0.jpg", "boxes": [box(1, 2, -3, 4)], "reviewed": True}),
        json.dumps({"file": "f0.jpg", "boxes": [{"x": "a", "y": 0, "w": 1, "h": 1}], "reviewed": True}),
        json.dumps({"file": "f0.jpg", "boxes": "x", "reviewed": True}),
        json.dumps({"file": "f0.jpg", "boxes": [], "reviewed": "yes"}),
    ])
    def test_post_label_rejects_bad_input(self, server, payload):
        out_dir, port = server
        status, _, _ = request(port, "POST", "/api/label", payload)
        assert status == 400
        assert not (out_dir / "labels.json").exists()

    def test_image_served(self, server):
        _, port = server
        status, ctype, data = request(port, "GET", "/img/f0.jpg")
        assert status == 200
        assert ctype == "image/jpeg"
        assert data[:2] == b"\xff\xd8"

    @pytest.mark.parametrize("path", [
        "/img/../proposals.json",
        "/img/..%2Fproposals.json",
        "/img/%2E%2E%2Fproposals.json",
        "/img/secret.jpg",
        "/img/frames/f0.jpg",
        "/img/",
        "/elsewhere",
    ])
    def test_image_rejects_traversal_and_unlisted(self, server, path):
        _, port = server
        status, _, data = request(port, "GET", path)
        assert status in (400, 404)
        assert b"version" not in data and b"not listed" not in data


class TestChannelOrder:
    def _dir(self, tmp_path):
        labels = {"f0.jpg": {"reviewed": True, "updated_at": "t", "boxes": [box(10, 10, 50, 50)]}}
        write_detection_dir(tmp_path, [(100, 80)], labels)
        frames = tmp_path / "frames" / "f0.jpg"
        img = np.zeros((80, 100, 3), dtype=np.uint8)
        img[..., 0] = 200  # red in RGB
        Image.fromarray(img).save(frames, format="PNG")  # lossless so channels survive

    def test_rgb_swaps_channels_before_detect(self, tmp_path):
        # detect is production's: it takes RGB and swaps to BGR itself (#147),
        # so the default passes frames through and "rgb" pre-swaps them.
        self._dir(tmp_path)
        seen = []

        def detect(image):
            seen.append(int(image[0, 0, 0]))
            return [box(10, 10, 50, 50, score=0.9)]

        bgr = json.loads((score.score(tmp_path, detect=detect).parent / "results.json").read_text())
        rgb = json.loads((score.score(tmp_path, detect=detect, channel_order="rgb").parent / "results.json").read_text())
        assert seen == [200, 0]
        assert rgb["channel_order"] == "rgb" and bgr["channel_order"] == "bgr"

    def test_rejects_unknown_order(self, tmp_path):
        self._dir(tmp_path)
        with pytest.raises(ValueError):
            score.score(tmp_path, detect=lambda im: [], channel_order="yuv")
