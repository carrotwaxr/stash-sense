"""Tests for models.py - data models with zero external dependencies."""

from models import (
    StashBoxEndpoint,
    PerformerIdentity,
    DatabaseManifest,
    FaceRecord,
    PerformerMetadata,
)


class TestStashBoxEndpoint:
    def test_from_url_stashdb(self):
        result = StashBoxEndpoint.from_url("https://stashdb.org/graphql")
        assert result == StashBoxEndpoint.STASHDB

    def test_from_url_fansdb(self):
        result = StashBoxEndpoint.from_url("https://fansdb.cc/graphql")
        assert result == StashBoxEndpoint.FANSDB

    def test_from_url_pmvstash(self):
        result = StashBoxEndpoint.from_url("https://pmvstash.org/graphql")
        assert result == StashBoxEndpoint.PMVSTASH

    def test_from_url_unknown_returns_none(self):
        result = StashBoxEndpoint.from_url("https://unknown-box.example.com/graphql")
        assert result is None

    def test_from_url_partial_match(self):
        result = StashBoxEndpoint.from_url("https://stashdb.org/api/v2/graphql")
        assert result == StashBoxEndpoint.STASHDB


class TestPerformerIdentity:
    def test_universal_id_format(self):
        identity = PerformerIdentity(stashbox_endpoint="stashdb.org", stashbox_id="uuid-123-abc")
        assert identity.universal_id == "stashdb.org:uuid-123-abc"

    def test_from_universal_id_roundtrip(self):
        original = PerformerIdentity(stashbox_endpoint="fansdb.cc", stashbox_id="some-uuid-456")
        parsed = PerformerIdentity.from_universal_id(original.universal_id)
        assert parsed.stashbox_endpoint == original.stashbox_endpoint
        assert parsed.stashbox_id == original.stashbox_id

    def test_from_universal_id_with_colons_in_id(self):
        # Edge case: UUID-like IDs shouldn't have colons, but test the split behavior
        parsed = PerformerIdentity.from_universal_id("stashdb.org:abc:def:ghi")
        assert parsed.stashbox_endpoint == "stashdb.org"
        assert parsed.stashbox_id == "abc:def:ghi"

    def test_get_stashbox_url(self):
        identity = PerformerIdentity(stashbox_endpoint="stashdb.org", stashbox_id="uuid-123")
        assert identity.get_stashbox_url() == "https://stashdb.org/performers/uuid-123"

    def test_get_stashbox_url_fansdb(self):
        identity = PerformerIdentity(stashbox_endpoint="fansdb.cc", stashbox_id="abc-def")
        assert identity.get_stashbox_url() == "https://fansdb.cc/performers/abc-def"


class TestDatabaseManifest:
    def test_defaults(self):
        manifest = DatabaseManifest(
            version="2026.01.30",
            created_at="2026-01-30T00:00:00Z",
            performer_count=100,
            face_count=500,
            sources=["stashdb.org"],
        )
        assert manifest.facenet_dim == 512
        assert manifest.arcface_dim == 512
        assert manifest.detector == "yolov8"
        assert manifest.checksums == {}

    def test_custom_values(self):
        manifest = DatabaseManifest(
            version="2026.02.01",
            created_at="2026-02-01T00:00:00Z",
            performer_count=200,
            face_count=1000,
            sources=["stashdb.org", "fansdb.cc"],
            facenet_dim=256,
            detector="retinaface",
        )
        assert manifest.facenet_dim == 256
        assert manifest.detector == "retinaface"
        assert len(manifest.sources) == 2


class TestFaceRecord:
    def test_creation(self):
        record = FaceRecord(face_index=42, universal_id="stashdb.org:uuid-123", source="stashdb")
        assert record.face_index == 42
        assert record.universal_id == "stashdb.org:uuid-123"
        assert record.source == "stashdb"


class TestPerformerMetadata:
    def test_defaults(self):
        identity = PerformerIdentity(stashbox_endpoint="stashdb.org", stashbox_id="uuid-123")
        metadata = PerformerMetadata(identity=identity, name="Test Performer")
        assert metadata.disambiguation is None
        assert metadata.country is None
        assert metadata.image_count == 0
        assert metadata.other_stashbox_ids == {}
        assert metadata.local_stash_id is None
