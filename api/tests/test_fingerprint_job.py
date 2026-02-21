"""Tests for fingerprint generation job wrapper."""
import pytest
from unittest.mock import MagicMock, patch
from base_job import JobContext


class TestFingerprintJob:
    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.update_job_progress = MagicMock()
        return db

    @pytest.fixture
    def ctx(self, mock_db):
        return JobContext(job_id=1, db=mock_db, queue_manager=None)

    @pytest.mark.asyncio
    async def test_runs_generator(self, ctx):
        from jobs.fingerprint_job import FingerprintGenerationJob
        mock_progress = MagicMock()
        mock_progress.processed_scenes = 5
        mock_progress.total_scenes = 10
        mock_progress.status.value = "completed"

        async def fake_generate(**kwargs):
            yield mock_progress

        mock_generator_cls = MagicMock()
        mock_generator_instance = MagicMock()
        mock_generator_instance.generate_all = fake_generate
        mock_generator_cls.return_value = mock_generator_instance

        with patch('jobs.fingerprint_job.SceneFingerprintGenerator', mock_generator_cls), \
             patch('jobs.fingerprint_job.get_stash_client', return_value=MagicMock()), \
             patch('jobs.fingerprint_job.get_rec_db', return_value=MagicMock()), \
             patch('jobs.fingerprint_job.get_db_version', return_value='2026.01.30'):
            job = FingerprintGenerationJob()
            await job.run(ctx)

        mock_generator_cls.assert_called_once()
        call_kwargs = mock_generator_cls.call_args[1]
        assert call_kwargs['db_version'] == '2026.01.30'
        ctx._db.update_job_progress.assert_called()

    @pytest.mark.asyncio
    async def test_stops_on_request(self, ctx):
        from jobs.fingerprint_job import FingerprintGenerationJob
        ctx.request_stop()
        with patch('jobs.fingerprint_job.SceneFingerprintGenerator') as mock_gen, \
             patch('jobs.fingerprint_job.get_stash_client', return_value=MagicMock()), \
             patch('jobs.fingerprint_job.get_rec_db', return_value=MagicMock()), \
             patch('jobs.fingerprint_job.get_db_version', return_value='2026.01.30'):
            job = FingerprintGenerationJob()
            result = await job.run(ctx)
        assert result is None
        mock_gen.assert_not_called()

    @pytest.mark.asyncio
    async def test_fails_without_db_version(self, ctx):
        from jobs.fingerprint_job import FingerprintGenerationJob
        with patch('jobs.fingerprint_job.get_db_version', return_value=None):
            job = FingerprintGenerationJob()
            with pytest.raises(RuntimeError, match="No face recognition database loaded"):
                await job.run(ctx)
