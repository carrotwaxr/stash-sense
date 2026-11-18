"""The scene listing carries date and studio (#133)."""

import pytest

from stash_client_unified import StashClientUnified
from tests.fake_stash import FakeStash


@pytest.mark.asyncio
async def test_fake_listing_has_date_and_studio():
    stash = FakeStash()
    studio = stash.add_studio("Studio")
    f1 = stash.add_file("/x/a.mp4", 100)
    f2 = stash.add_file("/x/b.mp4", 100)
    with_studio = stash.add_scene(f1, date="2015-03-01", studio_id=studio)
    without = stash.add_scene(f2)

    scenes, count = await stash.get_scenes_with_files(1, 10)
    by_id = {s["id"]: s for s in scenes}

    assert count == 2
    assert by_id[with_studio]["date"] == "2015-03-01"
    assert by_id[with_studio]["studio"] == {"id": studio}
    assert by_id[without]["date"] is None
    assert by_id[without]["studio"] is None


@pytest.mark.asyncio
async def test_real_query_selects_date_and_studio(monkeypatch):
    captured = {}

    async def fake_execute(self, query, variables=None, *args, **kwargs):
        captured["query"] = query
        return {"findScenes": {"count": 0, "scenes": []}}

    monkeypatch.setattr(StashClientUnified, "_execute", fake_execute)
    client = StashClientUnified("http://localhost:9999", "test-key")

    assert await client.get_scenes_with_files(1, 10) == ([], 0)
    query = captured["query"]
    assert "date" in query.split()
    assert "studio { id }" in query
