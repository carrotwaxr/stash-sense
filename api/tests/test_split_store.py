import json

import numpy as np

from splits import profiles as pf
from splits import signals as sg
from splits.store import SplitStore


def _signals():
    n = 60
    return sg.Signals(
        t=np.round(np.arange(n) / 6.0, 2),
        y=np.full(n, 100),
        sat=np.zeros(n),
        ydif=np.zeros(n),
        scd=np.zeros(n),
        au_t=np.arange(40) * 0.25,
        rms=np.full(40, -30.0),
        sil=np.zeros((0, 2)),
        duration=10.0,
        covered=[[0.0, 10.0]],
    )


FEAT = {"blk_len": 1.0, "dip_ratio": 0.2, "sil_len": 0.5, "scd": 10.0}


def test_signals_cache(tmp_path):
    st = SplitStore(tmp_path / "splits.db")
    assert st.get_signals("abc", 10) is None
    s = _signals()
    st.put_signals("abc", 10, sg.SIGNALS_VERSION, sg.encode(s))
    got = st.get_signals("abc", 10)
    assert np.array_equal(got.t, s.t)
    assert got.duration == s.duration
    assert st.get_signals("abc", 11) is None
    st.put_signals("old", 5, sg.SIGNALS_VERSION - 1, sg.encode(s))
    assert st.get_signals("old", 5) is None
    st.close()


def test_labels_and_profiles(tmp_path):
    st = SplitStore(tmp_path / "splits.db")
    st.add_label("ser", "k1", 3600, [600, 1200], [FEAT, None], "owner")
    st.add_label("ser", "k1", 3700, [700], [FEAT], "applied")
    st.add_label("ser", "k2", 3000, [900], None, "owner")
    st.add_label("whole", "k3", 3000, [], None, "owner")
    p = st.profiles()
    assert set(p) == {"ser"}
    assert p["ser"] == pf.build_profile(
        [
            {"key": "k1", "total": 3700, "cuts": [700], "feat": [FEAT]},
            {"key": "k2", "total": 3000, "cuts": [900], "feat": [None]},
        ]
    )
    assert p["ser"]["n_titles"] == 2
    st.close()


def test_seed_import(tmp_path):
    seed = {
        "aliases": [["^foo.*", "foo series"]],
        "labels": [
            {
                "series": "foo series",
                "key": "1",
                "total": 3000,
                "cuts": [1000],
                "feats": [FEAT],
                "source": "owner",
            },
            {
                "series": "foo series",
                "key": "2",
                "total": 3000,
                "cuts": [1500],
                "feats": None,
                "source": "applied",
            },
        ],
    }
    (tmp_path / "split_seed.json").write_text(json.dumps(seed))
    st = SplitStore(tmp_path / "splits.db")
    assert st.aliases() == []
    assert st.import_seed(tmp_path) == 2
    assert not (tmp_path / "split_seed.json").exists()
    assert (tmp_path / "split_seed.json.imported").exists()
    assert st.aliases() == [("^foo.*", "foo series")]
    assert st.profiles()["foo series"]["n_titles"] == 2
    assert st.import_seed(tmp_path) == 0
    st.close()
