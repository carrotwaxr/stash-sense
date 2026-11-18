"""Sprite-tile duplicate evidence (#130).

Stash's generated sprite sheet holds one frame per 1/81 of the duration. Each tile
becomes a small colour layout; a pair of scenes scores the most tiles that match
along one line ``tb = k*ta + off`` (a trim or intro shifts ``off``, a PAL speed-up
changes ``k``). A match is reliable; a miss proves nothing, because sparse sampling
lands on different instants in a trimmed copy.

Ported from stash-curator's ``mergetool/dupecheck.py`` (thresholds validated
2026-10-13; re-validate before changing). Differences: the sheet is decoded from
bytes with PIL instead of ffmpeg, and tiles that recur across many scenes (studio
logos, intro cards) can be suppressed with ``stop_tiles`` so they do not make
unrelated scenes from one studio look alike.

Pure logic apart from ``fetch_sprite``: stdlib, numpy, PIL and httpx only.
"""
import io
import re
from collections import Counter

import httpx
import numpy as np
from PIL import Image

# --- tuned constants (validated 2026-10-13; re-validate before changing) ---
GW, GH = 16, 9               # colour grid per tile
CROP = 0.12                  # fraction trimmed per side (logos, timestamps, residual bars)
BAR_LUMA = 28                # rows/cols darker than this in 90% of tiles are letterbox bars
SCALES = (1.0, 25 / 23.976, 23.976 / 25)
SPRITE_STRONG = (8, 12)      # SAME evidence: >= 8 tiles at corr 0.85, or >= 12 at 0.80
SPRITE_WEAK = (3, 5)         # some evidence

CUE = re.compile(r'(\d+):(\d+):([\d.]+) --> .*?\n.*?#xywh=(\d+),(\d+),(\d+),(\d+)')


def parse_vtt(text: str) -> list[tuple[float, int, int, int, int]]:
    """Stash thumbs VTT -> [(start_sec, x, y, w, h)] in file order."""
    out = []
    for m in CUE.finditer(text):
        hh, mm, ss, x, y, w, h = m.groups()
        out.append((int(hh) * 3600 + int(mm) * 60 + float(ss), int(x), int(y), int(w), int(h)))
    return out


def content_box(tiles) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1) inside letterbox/pillarbox bars: rows/cols dark in (nearly) every tile."""
    lum = np.stack([t.mean(axis=2) for t in tiles])
    rows = np.percentile(lum.mean(axis=2), 90, axis=0)
    cols = np.percentile(lum.mean(axis=1), 90, axis=0)
    r = np.nonzero(rows > BAR_LUMA)[0]
    c = np.nonzero(cols > BAR_LUMA)[0]
    if len(r) < 8 or len(c) < 8:
        return 0, tiles[0].shape[0], 0, tiles[0].shape[1]
    return int(r[0]), int(r[-1]) + 1, int(c[0]), int(c[-1]) + 1


def tile_feat(t) -> np.ndarray:
    """A tile's colour layout: GHxGW per-channel means, mean-removed, L2-normalized (flat -> zeros)."""
    h, w = t.shape[:2]
    t = t[int(h * CROP):h - int(h * CROP), int(w * CROP):w - int(w * CROP)]
    h, w = t.shape[:2]
    t = t[:h - h % GH, :w - w % GW]
    g = t.reshape(GH, t.shape[0] // GH, GW, t.shape[1] // GW, 3).mean(axis=(1, 3))
    g = (g - g.mean(axis=(0, 1))).ravel()
    n = np.linalg.norm(g)
    return g / n if n > 1e-3 else g * 0


def sprite_signature(image_bytes: bytes, vtt_text: str) -> dict:
    """Signature of one scene's sprite sheet.

    Returns ``{'times': float32[n], 'full': float32[n, GW*GH*3], 'c43': ...}``.
    ``c43`` (features of a centred 4:3 crop) is present only when the content is wider
    than 1.5. A sheet without usable tiles yields ``{'times': empty}``. Tile index
    everywhere (``stop_tiles``, ``skip_*``) is the row in ``full``.
    """
    img = np.asarray(Image.open(io.BytesIO(image_bytes)).convert('RGB'), dtype=np.float32)
    times, tiles = [], []
    for t, x, y, w, h in parse_vtt(vtt_text):
        tile = img[y:y + h, x:x + w]
        if tile.shape[0] >= 20 and tile.shape[1] >= 20:
            times.append(t)
            tiles.append(tile)
    if not tiles:
        return {'times': np.zeros(0, np.float32)}
    shape = Counter(t.shape for t in tiles).most_common(1)[0][0]
    keep = [i for i, t in enumerate(tiles) if t.shape == shape]
    times = [times[i] for i in keep]
    tiles = [tiles[i] for i in keep]
    y0, y1, x0, x1 = content_box(tiles)
    tiles = [t[y0:y1, x0:x1] for t in tiles]
    out = {'times': np.array(times, np.float32),
           'full': np.array([tile_feat(t) for t in tiles], np.float32)}
    h, w = tiles[0].shape[:2]
    if w / h > 1.5:
        cw = int(h * 4 / 3)
        cx = (w - cw) // 2
        out['c43'] = np.array([tile_feat(t[:, cx:cx + cw]) for t in tiles], np.float32)
    return out


def stop_tiles_for(sig: dict, others: list[dict], min_scenes: int = 4, thr: float = 0.85) -> set[int]:
    """The tile indexes of ``sig`` whose feature matches a tile in at least ``min_scenes - 1`` of ``others``.

    ``others`` are the signatures of other scenes (not ``sig`` itself). Such tiles (studio logos, intro and outro
    cards, a fixed wide shot) say nothing about whether two scenes are the same content. Flat tiles have a zero
    feature and never match.
    """
    fa = sig.get('full')
    if fa is None or len(fa) == 0:
        return set()
    seen_in = np.zeros(len(fa), np.int32)
    for other in others:
        fb = other.get('full')
        if fb is None or len(fb) == 0:
            continue
        seen_in += ((fa @ fb.T) >= thr).any(axis=1)
    return {int(i) for i in np.nonzero(seen_in >= min_scenes - 1)[0]}


def stop_tiles(sigs: dict[str, dict], min_scenes: int = 4, thr: float = 0.85) -> dict[str, set[int]]:
    """Per scene, the tile indexes ``stop_tiles_for`` finds against every other scene in ``sigs``.

    Every scene id in ``sigs`` appears in the result, with an empty set when nothing is dropped.
    """
    return {sid: stop_tiles_for(sig, [o for oid, o in sigs.items() if oid != sid], min_scenes, thr)
            for sid, sig in sigs.items()}


def _diag(ta, fa, tb, fb, thr) -> int:
    if len(fa) < 2 or len(fb) < 2:
        return 0
    ia, ib = np.nonzero(fa @ fb.T >= thr)
    if len(ia) == 0:
        return 0
    tol = max(ta[1] - ta[0], tb[1] - tb[0])
    best = 0
    for k in SCALES:
        d = tb[ib] - k * ta[ia]
        for off in np.unique(np.round(d)):
            sel = np.abs(d - off) <= tol
            na, nb = len(set(ia[sel].tolist())), len(set(ib[sel].tolist()))
            if nb >= na * 0.5:     # one b-tile soaking up many a-tiles is a static shot, not alignment
                best = max(best, na)
    return best


def _without(sig: dict, skip) -> tuple[np.ndarray, dict]:
    """(times, {'full', 'c43'?}) with the skipped tile rows removed."""
    times = sig['times']
    feats = {k: sig[k] for k in ('full', 'c43') if k in sig}
    if skip:
        keep = np.array([i not in skip for i in range(len(times))], bool)
        times = times[keep]
        feats = {k: v[keep] for k, v in feats.items()}
    return times, feats


def sprite_score(sig_a: dict, sig_b: dict, skip_a=None, skip_b=None) -> dict | None:
    """Tiles matching along one time line, best over full-frame and 4:3 crops.

    ``skip_a`` / ``skip_b`` are tile-index sets (from ``stop_tiles``) removed before matching.

    Returns ``None`` when either signature has no tiles, else the dupecheck shape
    ``{'d85': int, 'd80': int, 'tiles': int}``: ``d85`` / ``d80`` count the tiles matching
    along the best line at correlation 0.85 / 0.80, and ``tiles`` is the smaller tile count
    of the two sheets after skipping. Compare with ``SPRITE_STRONG`` (8 at d85 or 12 at d80)
    and ``SPRITE_WEAK`` (3, 5).
    """
    if 'full' not in sig_a or 'full' not in sig_b:
        return None
    ta, A = _without(sig_a, skip_a)
    tb, B = _without(sig_b, skip_b)
    res = {}
    for key, thr in (('d85', 0.85), ('d80', 0.80)):
        best = _diag(ta, A['full'], tb, B['full'], thr)
        if 'c43' in A:
            best = max(best, _diag(ta, A['c43'], tb, B['full'], thr))
        if 'c43' in B:
            best = max(best, _diag(ta, A['full'], tb, B['c43'], thr))
        res[key] = int(best)
    res['tiles'] = int(min(len(A['full']), len(B['full'])))
    return res


async def fetch_sprite(stash_url: str, api_key: str, scene_id: str) -> tuple[bytes, str] | None:
    """(sprite image bytes, VTT text) for a scene, or ``None`` when Stash has no sprite (404)."""
    headers = {'ApiKey': api_key}
    base = stash_url.rstrip('/')
    async with httpx.AsyncClient(timeout=60.0) as client:
        img = await client.get(f'{base}/scene/{scene_id}/sprite', headers=headers)
        if img.status_code == 404:
            return None
        img.raise_for_status()
        vtt = await client.get(f'{base}/scene/{scene_id}/vtt/sprite', headers=headers)
        if vtt.status_code == 404:
            return None
        vtt.raise_for_status()
    return img.content, vtt.text
