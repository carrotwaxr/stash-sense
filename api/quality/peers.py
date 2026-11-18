"""Below the studio's release (#133): a file whose effective detail sits well under its studio peers'.

Peers are the same studio's other files: the same year first, then within YEAR_WINDOW years, then any year. The
first group with at least `min_peers` files decides. A file is flagged when it is under the given percentile of
its peers and also under `ratio` times their median.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

MIN_PEERS = 8
YEAR_WINDOW = 2
RATIO = 0.6


@dataclass(frozen=True)
class PeerFile:
    scene_id: str
    studio_id: str | None
    year: int | None
    eff: float  # keeper.effective_detail()[0]


def _round3(x: float) -> float:
    return float(f"{x:.3g}")


def below_studio(files: list[PeerFile], *, percentile: float = 10.0, ratio: float = RATIO,
                 min_peers: int = MIN_PEERS) -> dict[str, dict]:
    """scene_id -> evidence for files well below their studio peers."""
    by_studio: dict[str, list[PeerFile]] = defaultdict(list)
    by_studio_year: dict[tuple[str, int], list[PeerFile]] = defaultdict(list)
    for f in files:
        if not f.studio_id:
            continue
        by_studio[f.studio_id].append(f)
        if f.year is not None:
            by_studio_year[(f.studio_id, f.year)].append(f)

    out: dict[str, dict] = {}
    for f in files:
        if not f.studio_id:
            continue
        group = None
        peer_effs: list[float] = []
        candidates = []
        if f.year is not None:
            candidates.append(("studio+year", by_studio_year[(f.studio_id, f.year)]))
            window = [p for y in range(f.year - YEAR_WINDOW, f.year + YEAR_WINDOW + 1)
                      for p in by_studio_year.get((f.studio_id, y), ())]
            candidates.append(("studio+years±2", window))
        candidates.append(("studio", by_studio[f.studio_id]))
        for name, members in candidates:
            effs = [p.eff for p in members if p.scene_id != f.scene_id]
            if len(effs) >= min_peers:
                group, peer_effs = name, effs
                break
        if group is None:
            continue
        arr = np.asarray(peer_effs, dtype=float)
        median = float(np.median(arr))
        threshold = float(np.percentile(arr, percentile))
        if not (f.eff < threshold and f.eff < ratio * median):
            continue
        out[f.scene_id] = {
            "group": group,
            "peers": len(peer_effs),
            "peer_median": _round3(median),
            "threshold": _round3(threshold),
            "eff": _round3(f.eff),
            "share_better": _round3(float((arr > f.eff).mean())),
            "score": _round3(float(np.clip(1 - f.eff / median, 0, 1))) if median > 0 else 0.0,
        }
    return out
