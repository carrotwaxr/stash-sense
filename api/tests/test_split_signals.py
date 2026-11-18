import numpy as np

from splits import signals as sg

VIDEO_TEXT = """frame:0    pts:0       pts_time:0
lavfi.scd.mafd=0.000000
lavfi.scd.score=0.000000
lavfi.scd.time=0
lavfi.signalstats.YDIF=0.000000
lavfi.signalstats.YAVG=16.4
lavfi.signalstats.SATAVG=3.14
frame:1    pts:1       pts_time:0.166667
lavfi.scd.score=57.26
lavfi.signalstats.YDIF=12.345
lavfi.signalstats.YAVG=120.6
lavfi.signalstats.SATAVG=40.04
frame:2    pts:2       pts_time:0.333333
lavfi.signalstats.YDIF=1.25
lavfi.signalstats.YAVG=130.2
lavfi.signalstats.SATAVG=41.0
"""


def _make(rng):
    n = 600
    t = np.round(np.arange(n) / 6.0, 2)
    au_t = np.arange(240) * 0.25
    return sg.Signals(
        t=t,
        y=rng.integers(0, 256, n),
        sat=np.round(rng.uniform(0, 200, n), 1),
        ydif=np.round(rng.uniform(0, 200, n), 1),
        scd=np.round(rng.uniform(0, 200, n), 1),
        au_t=au_t,
        rms=np.round(rng.uniform(-120, 0, 240), 1),
        sil=np.array([[10.5, 12.0], [30.25, 31.5]]),
        duration=100.0,
        covered=[[0.0, 60.0], [70.0, 100.0]],
    )


def test_encode_decode_roundtrip():
    s = _make(np.random.default_rng(1))
    d = sg.decode(sg.encode(s))
    assert np.array_equal(d.t, s.t)
    assert np.array_equal(d.y, s.y)
    assert np.array_equal(d.sat, s.sat)
    assert np.array_equal(d.ydif, s.ydif)
    assert np.array_equal(d.scd, s.scd)
    assert np.array_equal(d.au_t, s.au_t)
    assert np.array_equal(d.rms, s.rms)
    assert np.allclose(d.sil, s.sil)
    assert d.duration == s.duration
    assert [list(c) for c in d.covered] == [list(c) for c in s.covered]


def test_to_dict_matches_curator_shape():
    s = _make(np.random.default_rng(2))
    d = sg.to_dict(s)
    assert set(d) == {"fr", "au", "sil", "covered"}
    assert len(d["fr"]) == 600 and len(d["au"]) == 240
    row = d["fr"][5]
    assert row == [float(s.t[5]), int(s.y[5]), float(s.sat[5]), float(s.ydif[5]), float(s.scd[5])]
    assert isinstance(row[1], int)
    assert d["au"][3] == [float(s.au_t[3]), float(s.rms[3])]
    assert d["sil"] == [[10.5, 12.0], [30.25, 31.5]]
    assert d["covered"] == [[0.0, 60.0], [70.0, 100.0]]
    back = sg.from_curator(d, 100.0)
    assert np.array_equal(back.y, s.y) and back.duration == 100.0


def test_parse_video_metadata():
    fr = sg.parse_video(VIDEO_TEXT)
    assert fr == [
        [0.0, 16, 3.1, 0.0, 0.0],
        [0.17, 121, 40.0, 12.3, 57.3],
        [0.33, 130, 41.0, 1.2, 0.0],
    ]


def test_parse_audio():
    stderr = (
        "[silencedetect @ 0x1] silence_start: 10.5\n"
        "[silencedetect @ 0x1] silence_end: 12.0 | silence_duration: 1.5\n"
        "[silencedetect @ 0x1] silence_start: 90.25\n"
    )
    stdout = (
        "frame:0    pts:0       pts_time:0\n"
        "lavfi.astats.Overall.RMS_level=-inf\n"
        "frame:1    pts:2000    pts_time:0.25\n"
        "lavfi.astats.Overall.RMS_level=-23.456\n"
    )
    au, sil = sg.parse_audio(stdout, stderr, 100.0)
    assert au == [[0.0, -120.0], [0.25, -23.5]]
    assert sil == [[10.5, 12.0], [90.25, 100.0]]
    _, sil2 = sg.parse_audio("", stderr.splitlines()[0] + "\n" + stderr.splitlines()[1], 100.0)
    assert sil2 == [[10.5, 12.0]]


def test_commands_use_nostdin():
    v = sg.video_command("/x/a.mp4")
    a = sg.audio_command("/x/a.mp4")
    for c in (v, a):
        assert c[0] == "ffmpeg" and "-nostdin" in c
    assert v[v.index("-vf") + 1] == "fps=6,scale=160:-2,scdet=t=100,signalstats,metadata=print:file=-"
    assert "silencedetect=n=-35dB:d=0.3" in a[a.index("-af") + 1]
