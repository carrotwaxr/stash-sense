"""Tests for CPU-heavy ffmpeg knobs: nice level and thread cap."""

import asyncio
import os
import stat

import pytest

from job_models import JOB_REGISTRY, JobDefinition, JobPriority, ResourceType
from media_exec import cpu_heavy_command, run_command


class TestCpuHeavyCommand:
    def test_nice_and_threads(self):
        cmd = cpu_heavy_command(["ffmpeg", "-ss", "5", "-i", "in.mp4", "out.wav"], nice=10, threads=2)
        assert cmd == [
            "nice", "-n", "10",
            "ffmpeg", "-filter_threads", "2", "-ss", "5", "-threads", "2", "-i", "in.mp4", "out.wav",
        ]

    def test_thread_cap_before_every_input(self):
        cmd = cpu_heavy_command(["ffmpeg", "-i", "a.mp4", "-i", "b.srt", "out.mkv"], nice=0, threads=4)
        assert cmd == [
            "ffmpeg", "-filter_threads", "4",
            "-threads", "4", "-i", "a.mp4", "-threads", "4", "-i", "b.srt", "out.mkv",
        ]

    def test_zero_means_unchanged(self):
        cmd = ["ffmpeg", "-i", "in.mp4", "out.wav"]
        assert cpu_heavy_command(cmd, nice=0, threads=0) == cmd

    def test_full_path_to_ffmpeg_gets_thread_cap(self):
        cmd = cpu_heavy_command(["/usr/bin/ffmpeg", "-i", "x"], nice=0, threads=1)
        assert cmd[:3] == ["/usr/bin/ffmpeg", "-filter_threads", "1"]

    def test_non_ffmpeg_only_niced(self):
        cmd = cpu_heavy_command(["ffprobe", "-i", "x"], nice=5, threads=3)
        assert cmd == ["nice", "-n", "5", "ffprobe", "-i", "x"]

    def test_empty_command_rejected(self):
        with pytest.raises(ValueError):
            cpu_heavy_command([], nice=1, threads=1)


@pytest.fixture
def fake_ffmpeg(tmp_path):
    """An executable named ffmpeg that prints its niceness and arguments."""
    path = tmp_path / "ffmpeg"
    path.write_text('#!/bin/sh\necho "nice=$(nice) args=$*"\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


@pytest.fixture
def settings(tmp_path):
    from recommendations_db import RecommendationsDB
    from settings import init_settings
    mgr = init_settings(RecommendationsDB(str(tmp_path / "settings.db")), "gpu-high")
    yield mgr
    import settings as settings_module
    settings_module._settings_manager = None


class TestCpuHeavyJobThroughQueue:
    """A CPU-heavy job run by the queue gets its ffmpeg niced and thread-capped."""

    @pytest.fixture
    def registered(self, monkeypatch):
        for type_id, resource in (("test_cpu_media", ResourceType.CPU_HEAVY),
                                  ("test_gpu_media", ResourceType.GPU)):
            monkeypatch.setitem(JOB_REGISTRY, type_id, JobDefinition(
                type_id=type_id, display_name=type_id, description="test",
                resource=resource, default_priority=JobPriority.NORMAL,
                supports_incremental=False, schedulable=False,
            ))

    async def _run(self, tmp_path, monkeypatch, type_id, fake_ffmpeg):
        from queue_manager import QueueManager
        from recommendations_db import RecommendationsDB

        outputs = []

        class FakeMediaJob:
            async def run(self, ctx, cursor=None):
                result = await ctx.run_media_command([fake_ffmpeg, "-i", "in.mp4", "out.wav"], timeout=10)
                outputs.append(result.stdout.decode().strip())

        db = RecommendationsDB(str(tmp_path / "queue.db"))
        mgr = QueueManager(db)
        monkeypatch.setattr(mgr, "_create_job_instance", lambda t: FakeMediaJob())
        job_id = mgr.submit(type_id, triggered_by="test")
        await mgr._dispatch_once()
        await asyncio.wait_for(asyncio.gather(*mgr._running_tasks.values()), timeout=10)
        assert db.get_job(job_id)["status"] == "completed"
        return outputs[0]

    async def test_cpu_heavy_job_uses_configured_nice_and_threads(
        self, tmp_path, monkeypatch, registered, settings, fake_ffmpeg,
    ):
        settings.set("media_job_nice", 7)
        settings.set("media_job_threads", 3)
        out = await self._run(tmp_path, monkeypatch, "test_cpu_media", fake_ffmpeg)
        expected_nice = min(19, os.nice(0) + 7)
        assert out == f"nice={expected_nice} args=-filter_threads 3 -threads 3 -i in.mp4 out.wav"

    async def test_default_settings_nice_10_no_thread_cap(
        self, tmp_path, monkeypatch, registered, settings, fake_ffmpeg,
    ):
        out = await self._run(tmp_path, monkeypatch, "test_cpu_media", fake_ffmpeg)
        assert out == f"nice={min(19, os.nice(0) + 10)} args=-i in.mp4 out.wav"

    async def test_gpu_job_runs_command_unchanged(
        self, tmp_path, monkeypatch, registered, settings, fake_ffmpeg,
    ):
        settings.set("media_job_nice", 7)
        settings.set("media_job_threads", 3)
        out = await self._run(tmp_path, monkeypatch, "test_gpu_media", fake_ffmpeg)
        assert out == f"nice={os.nice(0)} args=-i in.mp4 out.wav"


class TestRunCommand:
    async def test_timeout_kills_process(self):
        with pytest.raises(TimeoutError):
            await run_command(["sleep", "5"], timeout=0.2)

    async def test_cancellation_kills_child(self, tmp_path):
        pidfile = tmp_path / "pid"
        task = asyncio.create_task(run_command(["sh", "-c", f"echo $$ > {pidfile}; exec sleep 30"]))
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text().strip():
                break
            await asyncio.sleep(0.05)
        pid = int(pidfile.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)

    async def test_returns_output(self):
        result = await run_command(["sh", "-c", "echo out; echo err >&2; exit 3"])
        assert (result.returncode, result.stdout, result.stderr) == (3, b"out\n", b"err\n")
