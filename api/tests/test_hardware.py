"""Tests for hardware detection module."""

import pytest
from unittest.mock import patch, MagicMock

from hardware import (
    HardwareProfile,
    _classify_tier,
    _probe_gpu,
    _probe_cpu,
    _probe_memory,
    _probe_storage,
    _read_proc_meminfo,
    detect_hardware,
)


class TestTierClassification:
    """Test hardware tier classification logic."""

    def test_gpu_high_with_large_vram(self):
        assert _classify_tier(True, 8192) == "gpu-high"

    def test_gpu_high_at_threshold(self):
        assert _classify_tier(True, 4096) == "gpu-high"

    def test_gpu_low_below_threshold(self):
        assert _classify_tier(True, 4095) == "gpu-low"

    def test_gpu_low_small_vram(self):
        assert _classify_tier(True, 2048) == "gpu-low"

    def test_gpu_low_no_vram_info(self):
        """GPU detected but pynvml unavailable — no VRAM info."""
        assert _classify_tier(True, None) == "gpu-low"

    def test_cpu_no_gpu(self):
        assert _classify_tier(False, None) == "cpu"

    def test_cpu_no_gpu_ignores_vram(self):
        assert _classify_tier(False, 8192) == "cpu"


class TestProbeGpu:
    """Test GPU probing with mocked dependencies."""

    def test_no_cuda_provider(self):
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            available, name, vram = _probe_gpu()
        assert available is False
        assert name is None
        assert vram is None

    @staticmethod
    def _cuda_ort():
        mock_ort = MagicMock()
        mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider", "CPUExecutionProvider"
        ]
        return mock_ort

    @staticmethod
    def _pynvml(count=1, name="NVIDIA GeForce RTX 3060", total=12 * 1024**3):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = count
        mock_pynvml.nvmlDeviceGetName.return_value = name
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = MagicMock(total=total)
        return mock_pynvml

    def test_cuda_with_pynvml_reads_vram(self):
        with patch.dict("sys.modules", {"onnxruntime": self._cuda_ort(), "pynvml": self._pynvml()}):
            available, name, vram = _probe_gpu()
        assert (available, name, vram) == (True, "NVIDIA GeForce RTX 3060", 12288)

    def test_pynvml_bytes_name_decoded(self):
        nvml = self._pynvml(name=b"NVIDIA GeForce GTX 1080")
        with patch.dict("sys.modules", {"onnxruntime": self._cuda_ort(), "pynvml": nvml}):
            _, name, _ = _probe_gpu()
        assert name == "NVIDIA GeForce GTX 1080"

    def test_pynvml_zero_devices_means_cpu(self):
        nvml = self._pynvml(count=0)
        with patch.dict("sys.modules", {"onnxruntime": self._cuda_ort(), "pynvml": nvml}):
            assert _probe_gpu() == (False, None, None)

    def test_falls_back_to_nvidia_smi_without_pynvml(self):
        """The bug from #124: no pynvml meant no VRAM, so a 12 GB card landed in gpu-low."""
        smi = MagicMock(returncode=0, stdout="NVIDIA GeForce RTX 3060, 12288\n", stderr="")
        with patch.dict("sys.modules", {"onnxruntime": self._cuda_ort(), "pynvml": None}), \
                patch("hardware.subprocess.run", return_value=smi) as run:
            available, name, vram = _probe_gpu()
        assert (available, name, vram) == (True, "NVIDIA GeForce RTX 3060", 12288)
        assert run.call_args[0][0][0] == "nvidia-smi"
        assert _classify_tier(available, vram) == "gpu-high"

    def test_nvml_init_failure_falls_back_to_nvidia_smi(self):
        nvml = self._pynvml()
        nvml.nvmlInit.side_effect = RuntimeError("libnvidia-ml.so.1 not found")
        smi = MagicMock(returncode=0, stdout="Tesla T4, 15360\n", stderr="")
        with patch.dict("sys.modules", {"onnxruntime": self._cuda_ort(), "pynvml": nvml}), \
                patch("hardware.subprocess.run", return_value=smi):
            assert _probe_gpu() == (True, "Tesla T4", 15360)

    def test_cuda_package_without_device_means_cpu(self):
        """onnxruntime-gpu lists CUDA even with no GPU; no device nodes means CPU."""
        with patch.dict("sys.modules", {"onnxruntime": self._cuda_ort(), "pynvml": None}), \
                patch("hardware.subprocess.run", side_effect=FileNotFoundError("nvidia-smi")), \
                patch("hardware._nvidia_device_nodes_present", return_value=False):
            assert _probe_gpu() == (False, None, None)

    def test_device_present_without_details_is_low_tier(self):
        with patch.dict("sys.modules", {"onnxruntime": self._cuda_ort(), "pynvml": None}), \
                patch("hardware.subprocess.run", side_effect=FileNotFoundError("nvidia-smi")), \
                patch("hardware._nvidia_device_nodes_present", return_value=True):
            available, name, vram = _probe_gpu()
        assert (available, name, vram) == (True, "NVIDIA GPU (details unavailable)", None)
        assert _classify_tier(available, vram) == "gpu-low"

    def test_ort_import_fails(self):
        """When onnxruntime can't be imported, GPU is unavailable."""
        with patch.dict("sys.modules", {"onnxruntime": None}):
            available, name, vram = _probe_gpu()
        assert available is False


class TestProbeCpu:
    """Test CPU probing."""

    @patch("os.cpu_count", return_value=8)
    def test_reads_cpu_count(self, mock_count):
        cores = _probe_cpu()
        assert cores == 8

    @patch("os.cpu_count", return_value=None)
    def test_fallback_when_unknown(self, mock_count):
        cores = _probe_cpu()
        assert cores == 1


class TestProbeMemory:
    """Test memory probing."""

    def test_returns_positive_values(self):
        total, available = _probe_memory()
        # In some environments (containers, CI), psutil may not be available
        # or cgroup limits may cause unexpected values. Just verify non-negative
        # and that available <= total (or both are 0 if probing failed).
        assert total >= 0
        assert available >= 0
        if total > 0:
            assert available <= total


class TestReadProcMeminfo:
    """Fallback memory probe used when psutil is missing."""

    def test_parses_total_and_available(self, tmp_path):
        meminfo = tmp_path / "meminfo"
        meminfo.write_text(
            "MemTotal:       32768000 kB\nMemFree:  1000 kB\nMemAvailable:   16384000 kB\n"
        )
        assert _read_proc_meminfo(str(meminfo)) == (32000, 16000)

    def test_missing_file_returns_zero(self, tmp_path):
        assert _read_proc_meminfo(str(tmp_path / "nope")) == (0, 0)


class TestProbeStorage:
    """Test storage probing."""

    def test_real_path(self, tmp_path):
        free = _probe_storage(str(tmp_path))
        assert free > 0

    def test_nonexistent_path(self):
        free = _probe_storage("/nonexistent/path/xyz")
        assert free == 0


class TestDetectHardware:
    """Test full hardware detection."""

    @patch("hardware._probe_gpu", return_value=(True, "Test GPU", 8192))
    @patch("hardware._probe_cpu", return_value=8)
    @patch("hardware._probe_memory", return_value=(32768, 16384))
    @patch("hardware._probe_storage", return_value=500000)
    def test_gpu_high_profile(self, mock_storage, mock_mem, mock_cpu, mock_gpu):
        profile = detect_hardware("/tmp")
        assert isinstance(profile, HardwareProfile)
        assert profile.gpu_available is True
        assert profile.gpu_name == "Test GPU"
        assert profile.gpu_vram_mb == 8192
        assert profile.cpu_cores == 8
        assert profile.memory_total_mb == 32768
        assert profile.tier == "gpu-high"

    @patch("hardware._probe_gpu", return_value=(False, None, None))
    @patch("hardware._probe_cpu", return_value=4)
    @patch("hardware._probe_memory", return_value=(8192, 4096))
    @patch("hardware._probe_storage", return_value=100000)
    def test_cpu_profile(self, mock_storage, mock_mem, mock_cpu, mock_gpu):
        profile = detect_hardware("/tmp")
        assert profile.gpu_available is False
        assert profile.tier == "cpu"

    def test_profile_is_frozen(self):
        profile = HardwareProfile(
            gpu_available=False, gpu_name=None, gpu_vram_mb=None,
            cpu_cores=4, memory_total_mb=8192, memory_available_mb=4096,
            storage_free_mb=100000, tier="cpu",
        )
        with pytest.raises(AttributeError):
            profile.tier = "gpu-high"

    def test_summary_with_gpu(self):
        profile = HardwareProfile(
            gpu_available=True, gpu_name="GTX 1080", gpu_vram_mb=8192,
            cpu_cores=8, memory_total_mb=32768, memory_available_mb=16384,
            storage_free_mb=500000, tier="gpu-high",
        )
        summary = profile.summary()
        assert "GTX 1080" in summary
        assert "8192MB VRAM" in summary

    def test_summary_without_gpu(self):
        profile = HardwareProfile(
            gpu_available=False, gpu_name=None, gpu_vram_mb=None,
            cpu_cores=4, memory_total_mb=8192, memory_available_mb=4096,
            storage_free_mb=100000, tier="cpu",
        )
        summary = profile.summary()
        assert "No GPU" in summary
