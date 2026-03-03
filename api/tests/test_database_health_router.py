"""Tests for the database health API router."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

import database_health_router as dh_mod


@pytest.fixture
def mock_recognizer():
    """Create a mock recognizer with performers and faces."""
    recognizer = Mock()
    recognizer.performers = {f"perf_{i}": Mock() for i in range(100)}
    recognizer.faces = {f"face_{i}": Mock() for i in range(500)}
    return recognizer


@pytest.fixture
def mock_manifest():
    """Create a mock database manifest."""
    return {
        "version": "2.1.0",
        "performer_count": 107759,
        "face_count": 277097,
        "sources": ["stashdb.org", "fansdb.cc"],
        "created_at": "2026-02-12T10:00:00Z",
    }


@pytest.fixture
def client_with_recognizer(mock_recognizer, mock_manifest):
    """Create a test client with recognizer loaded."""
    original_recognizer = dh_mod._recognizer
    original_matcher = dh_mod._multi_signal_matcher
    original_manifest = dh_mod._db_manifest
    original_updater = dh_mod._db_updater

    dh_mod._recognizer = mock_recognizer
    dh_mod._multi_signal_matcher = None
    dh_mod._db_manifest = mock_manifest
    dh_mod._db_updater = None

    app = FastAPI()
    app.include_router(dh_mod.router)
    test_client = TestClient(app)

    yield test_client

    dh_mod._recognizer = original_recognizer
    dh_mod._multi_signal_matcher = original_matcher
    dh_mod._db_manifest = original_manifest
    dh_mod._db_updater = original_updater


@pytest.fixture
def client_without_recognizer(mock_manifest):
    """Create a test client without recognizer (degraded mode)."""
    original_recognizer = dh_mod._recognizer
    original_matcher = dh_mod._multi_signal_matcher
    original_manifest = dh_mod._db_manifest
    original_updater = dh_mod._db_updater

    dh_mod._recognizer = None
    dh_mod._multi_signal_matcher = None
    dh_mod._db_manifest = mock_manifest
    dh_mod._db_updater = None

    app = FastAPI()
    app.include_router(dh_mod.router)
    test_client = TestClient(app)

    yield test_client

    dh_mod._recognizer = original_recognizer
    dh_mod._multi_signal_matcher = original_matcher
    dh_mod._db_manifest = original_manifest
    dh_mod._db_updater = original_updater


# ==================== GET /health ====================


class TestHealthCheck:
    """Test GET /health."""

    def test_healthy_with_recognizer(self, client_with_recognizer):
        resp = client_with_recognizer.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["database_loaded"] is True
        assert data["performer_count"] == 100
        assert data["face_count"] == 500

    def test_degraded_without_recognizer(self, client_without_recognizer):
        resp = client_without_recognizer.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["database_loaded"] is False
        assert data["performer_count"] == 0
        assert data["face_count"] == 0


# ==================== GET /database/info ====================


class TestDatabaseInfo:
    """Test GET /database/info."""

    def test_with_recognizer(self, client_with_recognizer):
        resp = client_with_recognizer.get("/database/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "2.1.0"
        # When recognizer is loaded, uses live counts
        assert data["performer_count"] == 100
        assert data["face_count"] == 500
        assert data["sources"] == ["stashdb.org", "fansdb.cc"]
        assert data["created_at"] == "2026-02-12T10:00:00Z"

    def test_without_recognizer_uses_manifest(self, client_without_recognizer):
        resp = client_without_recognizer.get("/database/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "2.1.0"
        # Falls back to manifest counts
        assert data["performer_count"] == 107759
        assert data["face_count"] == 277097

    def test_with_tattoo_matcher(self, mock_recognizer, mock_manifest):
        """Test database info includes tattoo count when matcher is loaded."""
        original_recognizer = dh_mod._recognizer
        original_matcher = dh_mod._multi_signal_matcher
        original_manifest = dh_mod._db_manifest
        original_updater = dh_mod._db_updater

        mock_matcher = Mock()
        mock_matcher.performers_with_tattoo_embeddings = {"p1", "p2", "p3"}

        dh_mod._recognizer = mock_recognizer
        dh_mod._multi_signal_matcher = mock_matcher
        dh_mod._db_manifest = mock_manifest
        dh_mod._db_updater = None

        app = FastAPI()
        app.include_router(dh_mod.router)
        client = TestClient(app)

        try:
            resp = client.get("/database/info")
            assert resp.status_code == 200
            data = resp.json()
            assert data["tattoo_embedding_count"] == 3
        finally:
            dh_mod._recognizer = original_recognizer
            dh_mod._multi_signal_matcher = original_matcher
            dh_mod._db_manifest = original_manifest
            dh_mod._db_updater = original_updater


# ==================== GET /health/ffmpeg ====================


class TestFfmpegHealth:
    """Test GET /health/ffmpeg."""

    def test_ffmpeg_available(self, client_with_recognizer):
        with patch("database_health_router.check_ffmpeg_available", return_value=True):
            resp = client_with_recognizer.get("/health/ffmpeg")
            assert resp.status_code == 200
            data = resp.json()
            assert data["ffmpeg_available"] is True
            assert data["v2_endpoint_ready"] is True

    def test_ffmpeg_not_available(self, client_with_recognizer):
        with patch("database_health_router.check_ffmpeg_available", return_value=False):
            resp = client_with_recognizer.get("/health/ffmpeg")
            assert resp.status_code == 200
            data = resp.json()
            assert data["ffmpeg_available"] is False
            assert data["v2_endpoint_ready"] is False


# ==================== GET /health/rate-limiter ====================


class TestRateLimiterHealth:
    """Test GET /health/rate-limiter."""

    def test_returns_metrics(self, client_with_recognizer):
        mock_limiter = Mock()
        mock_limiter.get_metrics.return_value = {
            "total_requests": 42,
            "total_wait_time": 1.234,
            "avg_wait_time": 0.029,
            "queue_size": 0,
            "requests_per_second": 5.0,
        }

        mock_get_instance = AsyncMock(return_value=mock_limiter)
        # RateLimiter is imported inside the endpoint function, patch on the source module
        with patch("rate_limiter.RateLimiter.get_instance", mock_get_instance):
            resp = client_with_recognizer.get("/health/rate-limiter")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_requests"] == 42
            assert data["requests_per_second"] == 5.0
