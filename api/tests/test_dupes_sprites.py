"""Sprite-tile duplicate evidence: signatures, score and stop-tile suppression (#130).

Ported from stash-curator's test_dupecheck.py; sheets are built with numpy and PIL."""
import io

import httpx
import numpy as np
import pytest
from PIL import Image

from duplicates import sprites

TW, TH = 160, 90
COLS = 9
STEP = 20.0


def _tile(rng, h=TH, w=TW):
    """A picture with structure at the colour-grid scale (random blocks, upscaled)."""
    blocks = rng.uniform(40, 255, (9, 16, 3))
    return np.repeat(np.repeat(blocks, h // 9, axis=0), w // 16, axis=1)


def _tiles(seed, n=81):
    rng = np.random.default_rng(seed)
    return [_tile(rng) for _ in range(n)]


def _sheet(tiles, step=STEP, scale=1.0, pad=None):
    """-> (png bytes, vtt text) laying `tiles` out COLS wide; `pad=(top, bottom)` adds black bars per tile."""
    th, tw = tiles[0].shape[:2]
    if pad:
        th += pad[0] + pad[1]
    rows = -(-len(tiles) // COLS)
    img = np.zeros((rows * th, COLS * tw, 3), np.uint8)
    cues = []
    for i, t in enumerate(tiles):
        x, y = (i % COLS) * tw, (i // COLS) * th
        top = pad[0] if pad else 0
        img[y + top:y + top + t.shape[0], x:x + tw] = t.astype(np.uint8)
        s = i * step * scale
        cues.append(f'{_ts(s)} --> {_ts(s + step * scale)}\nsprite.jpg#xywh={x},{y},{tw},{th}')
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, 'PNG')
    return buf.getvalue(), 'WEBVTT\n\n' + '\n\n'.join(cues) + '\n'


def _ts(sec):
    return f'{int(sec // 3600):02d}:{int(sec % 3600 // 60):02d}:{sec % 60:06.3f}'


def _sig(tiles, **kw):
    return sprites.sprite_signature(*_sheet(tiles, **kw))


def _strong(s):
    return s['d85'] >= sprites.SPRITE_STRONG[0] or s['d80'] >= sprites.SPRITE_STRONG[1]


def _weak(s):
    return s['d85'] >= sprites.SPRITE_WEAK[0] or s['d80'] >= sprites.SPRITE_WEAK[1]


def test_parse_vtt_reads_start_and_tile_box():
    vtt = ('WEBVTT\n\n00:00:00.000 --> 00:00:43.640\nabc_sprite.jpg#xywh=0,0,160,90\n\n'
           '00:01:27.280 --> 00:02:10.920\nabc_sprite.jpg#xywh=320,0,160,90\n')
    assert sprites.parse_vtt(vtt) == [(0.0, 0, 0, 160, 90), (87.28, 320, 0, 160, 90)]


def test_content_box_trims_letterbox_bars():
    rng = np.random.default_rng(0)
    tiles = []
    for _ in range(20):
        t = np.zeros((107, 160, 3), np.float32)
        t[10:97] = rng.uniform(40, 255, (87, 160, 3))
        tiles.append(t)
    assert sprites.content_box(tiles) == (10, 97, 0, 160)


def test_flat_tile_has_zero_feature():
    assert not np.any(sprites.tile_feat(np.full((90, 160, 3), 7, np.float32)))


def test_signature_shape_and_wide_content_gets_a_43_crop():
    sig = _sig(_tiles(1, 30))
    assert len(sig['times']) == 30
    assert sig['full'].shape == (30, sprites.GW * sprites.GH * 3)
    assert 'c43' in sig          # 160x90 is wider than 1.5


def test_sheet_without_tiles_scores_none():
    sig = sprites.sprite_signature(_sheet(_tiles(1, 3))[0], 'WEBVTT\n')
    assert sprites.sprite_score(sig, _sig(_tiles(1))) is None


def test_identical_sheets_score_strong():
    s = sprites.sprite_score(_sig(_tiles(1)), _sig(_tiles(1)))
    assert _strong(s) and s['d85'] >= 75 and s['tiles'] == 81


def test_sheet_offset_by_two_tiles_scores_strong():
    tiles = _tiles(2)
    s = sprites.sprite_score(_sig(tiles), _sig(tiles[2:]))
    assert _strong(s) and s['d85'] >= 75


def test_pal_retiming_scores_strong():
    tiles = _tiles(3)
    s = sprites.sprite_score(_sig(tiles), _sig(tiles, scale=23.976 / 25))
    assert _strong(s) and s['d85'] >= 75
    s = sprites.sprite_score(_sig(tiles), _sig(tiles, scale=25 / 23.976))
    assert _strong(s) and s['d85'] >= 75


def test_letterboxed_copy_matches_after_bar_trim():
    tiles = _tiles(4)
    barred = _sig(tiles, pad=(15, 15))
    s = sprites.sprite_score(_sig(tiles), barred)
    assert _strong(s) and s['d85'] >= 75


def test_unrelated_sheets_score_under_weak():
    s = sprites.sprite_score(_sig(_tiles(5)), _sig(_tiles(6)))
    assert not _weak(s)


def test_one_static_tile_matching_everything_is_ignored():
    a, b = _tiles(7), _tiles(8)
    b[40] = a[0]
    assert sprites.sprite_score(_sig(a), _sig(b))['d85'] <= 1


def _studio_scenes(n_sharing, intro=10):
    """Scenes whose first `intro` tiles are one shared studio card; the rest is unique per scene."""
    shared = _tiles(100, intro)
    return {f's{i}': _sig(shared + _tiles(200 + i, 81 - intro)) for i in range(n_sharing)}


def test_stop_tiles_suppressed():
    sigs = _studio_scenes(4)
    raw = sprites.sprite_score(sigs['s0'], sigs['s1'])
    assert _strong(raw)      # the shared card alone makes unrelated scenes look duplicated
    stops = sprites.stop_tiles(sigs)
    assert stops['s0'] == set(range(10)) and stops == {k: set(range(10)) for k in sigs}
    s = sprites.sprite_score(sigs['s0'], sigs['s1'], stops['s0'], stops['s1'])
    assert not _strong(s) and s['tiles'] == 71


def test_tile_shared_by_fewer_scenes_than_min_is_kept():
    sigs = _studio_scenes(3)
    assert sprites.stop_tiles(sigs) == {k: set() for k in sigs}
    assert sprites.stop_tiles(sigs, min_scenes=3) == {k: set(range(10)) for k in sigs}


def test_true_duplicates_survive_stop_tiles():
    tiles = _tiles(9)
    sigs = {'a': _sig(tiles), 'b': _sig(tiles), 'c': _sig(_tiles(10)), 'd': _sig(_tiles(11))}
    stops = sprites.stop_tiles(sigs)
    assert stops['c'] == set() and stops['d'] == set()
    # only two scenes carry these tiles, below min_scenes, so the pair keeps its evidence
    assert _strong(sprites.sprite_score(sigs['a'], sigs['b'], stops['a'], stops['b']))


# ------------------------------------------------------------------- fetch ---

def _patch_client(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(sprites.httpx, 'AsyncClient',
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


async def test_fetch_sprite_returns_bytes_and_vtt(monkeypatch):
    seen = []

    def handler(req):
        seen.append((req.url.path, req.headers.get('ApiKey')))
        if req.url.path.endswith('/vtt/sprite'):
            return httpx.Response(200, text='WEBVTT\n')
        return httpx.Response(200, content=b'img')

    _patch_client(monkeypatch, handler)
    assert await sprites.fetch_sprite('http://stash:6969/', 'k', '7') == (b'img', 'WEBVTT\n')
    assert seen == [('/scene/7/sprite', 'k'), ('/scene/7/vtt/sprite', 'k')]


@pytest.mark.parametrize('missing', ['/scene/7/sprite', '/scene/7/vtt/sprite'])
async def test_fetch_sprite_is_none_on_404(monkeypatch, missing):
    _patch_client(monkeypatch, lambda req: httpx.Response(404 if req.url.path == missing else 200, text='x'))
    assert await sprites.fetch_sprite('http://stash:6969', 'k', '7') is None
