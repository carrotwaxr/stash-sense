"""Hardware detection and profiling.

Probes GPU, CPU, memory, and storage at startup to build an immutable
hardware profile. Used by the settings system to pick tier-appropriate
defaults (batch sizes, concurrency, frame counts).

Runs once at startup. All probes have graceful fallbacks.
"""

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HardwareProfile:
    """Immutable hardware profile built at startup."""
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_vram_mb: Optional[int]
    cpu_cores: int
    memory_total_mb: int
    memory_available_mb: int
    storage_free_mb: int
    tier: str  # "gpu-high", "gpu-low", "cpu"

    def summary(self) -> str:
        """One-line hardware summary for startup log."""
        if self.gpu_available:
            gpu_info = f"{self.gpu_name} ({self.gpu_vram_mb}MB VRAM)"
        else:
            gpu_info = "No GPU"
        return (
            f"{gpu_info}, {self.memory_total_mb}MB RAM, "
            f"{self.cpu_cores} cores, {self.storage_free_mb}MB free disk"
        )


def _classify_tier(gpu_available: bool, gpu_vram_mb: Optional[int]) -> str:
    """Classify hardware into a tier based on GPU availability and VRAM."""
    if not gpu_available:
        return "cpu"
    if gpu_vram_mb is not None and gpu_vram_mb >= 4096:
        return "gpu-high"
    return "gpu-low"


def _probe_nvml() -> Optional[tuple[int, Optional[str], Optional[int]]]:
    """Query NVML for (device_count, name, vram_mb) of device 0.

    Returns None when NVML cannot be used (pynvml missing or the driver
    library not mounted into the container).
    """
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception as e:
        logger.debug("NVML unavailable: %s", e)
        return None
    try:
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return 0, None, None
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return count, name, mem_info.total // (1024 * 1024)
    except Exception as e:
        logger.debug("NVML query failed: %s", e)
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _probe_nvidia_smi() -> Optional[tuple[str, int]]:
    """Query nvidia-smi for (name, vram_mb) of the first GPU, or None."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("nvidia-smi unavailable: %s", e)
        return None
    if result.returncode != 0:
        logger.debug("nvidia-smi failed: %s", result.stderr.strip())
        return None
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    name, _, mem = line.rpartition(",")
    try:
        return name.strip(), int(float(mem.strip()))
    except ValueError:
        logger.debug("Unparseable nvidia-smi output: %r", line)
        return None


def _nvidia_device_nodes_present() -> bool:
    """True when the NVIDIA runtime has exposed GPU device nodes."""
    return any(Path("/dev").glob("nvidia[0-9]*"))


def _probe_gpu() -> tuple[bool, Optional[str], Optional[int]]:
    """Probe GPU via ONNX Runtime providers, NVML and nvidia-smi.

    onnxruntime-gpu lists CUDAExecutionProvider whenever the package is
    installed, GPU or not, so a device has to be confirmed separately.
    VRAM decides the tier, so NVML and nvidia-smi are tried in turn.

    Returns:
        (gpu_available, gpu_name, gpu_vram_mb)
    """
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False, None, None
    except Exception as e:
        logger.debug("ONNX Runtime CUDA probe failed: %s", e)
        return False, None, None

    nvml = _probe_nvml()
    if nvml is not None:
        count, name, vram_mb = nvml
        if count == 0:
            logger.warning("CUDA runtime present but NVML reports no GPU; using CPU tier")
            return False, None, None
        return True, name, vram_mb

    smi = _probe_nvidia_smi()
    if smi is not None:
        return True, smi[0], smi[1]

    if not _nvidia_device_nodes_present():
        logger.warning(
            "CUDA runtime present but no NVIDIA device is visible; using CPU tier. "
            "Start the container with --runtime=nvidia (unRAID: Extra Parameters)."
        )
        return False, None, None

    logger.warning(
        "NVIDIA device found but VRAM could not be read (no NVML, no nvidia-smi); "
        "assuming the low GPU tier. Set NVIDIA_DRIVER_CAPABILITIES to include 'utility'."
    )
    return True, "NVIDIA GPU (details unavailable)", None


def _probe_cpu() -> int:
    """Probe CPU core count, respecting cgroup limits."""
    # Check cgroup v2 limit first (Docker containers)
    try:
        cgroup_path = Path("/sys/fs/cgroup/cpu.max")
        if cgroup_path.exists():
            content = cgroup_path.read_text().strip()
            parts = content.split()
            if parts[0] != "max":
                quota = int(parts[0])
                period = int(parts[1])
                return max(1, quota // period)
    except Exception as e:
        logger.debug("cgroup CPU probe failed: %s", e)

    return os.cpu_count() or 1


def _read_proc_meminfo(path: str = "/proc/meminfo") -> tuple[int, int]:
    """Read (total_mb, available_mb) from /proc/meminfo; (0, 0) on failure."""
    values = {}
    try:
        for line in Path(path).read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                values[key] = int(rest.split()[0]) // 1024  # kB -> MB
    except Exception as e:
        logger.debug("meminfo probe failed: %s", e)
    return values.get("MemTotal", 0), values.get("MemAvailable", 0)


def _probe_memory() -> tuple[int, int]:
    """Probe total and available memory in MB, respecting cgroup limits.

    Returns:
        (total_mb, available_mb)
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_mb = mem.total // (1024 * 1024)
        available_mb = mem.available // (1024 * 1024)
    except Exception as e:
        logger.debug("psutil memory probe failed, reading /proc/meminfo: %s", e)
        total_mb, available_mb = _read_proc_meminfo()

    # Check cgroup v2 memory limit (Docker containers)
    try:
        cgroup_path = Path("/sys/fs/cgroup/memory.max")
        if cgroup_path.exists():
            content = cgroup_path.read_text().strip()
            if content != "max":
                cgroup_limit_mb = int(content) // (1024 * 1024)
                total_mb = min(total_mb, cgroup_limit_mb) if total_mb else cgroup_limit_mb
    except Exception as e:
        logger.debug("cgroup memory probe failed: %s", e)

    return total_mb, available_mb


def _probe_storage(data_dir: str) -> int:
    """Probe free disk space at the data directory in MB."""
    try:
        usage = shutil.disk_usage(data_dir)
        return usage.free // (1024 * 1024)
    except Exception as e:
        logger.debug("Storage probe failed for %s: %s", data_dir, e)
        return 0


def detect_hardware(data_dir: str = "./data") -> HardwareProfile:
    """Run all hardware probes and build an immutable profile.

    Args:
        data_dir: Path to the data directory for storage probe.

    Returns:
        Frozen HardwareProfile dataclass.
    """
    gpu_available, gpu_name, gpu_vram_mb = _probe_gpu()
    cpu_cores = _probe_cpu()
    memory_total_mb, memory_available_mb = _probe_memory()
    storage_free_mb = _probe_storage(data_dir)
    tier = _classify_tier(gpu_available, gpu_vram_mb)

    profile = HardwareProfile(
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram_mb,
        cpu_cores=cpu_cores,
        memory_total_mb=memory_total_mb,
        memory_available_mb=memory_available_mb,
        storage_free_mb=storage_free_mb,
        tier=tier,
    )

    logger.warning(f"Hardware: {profile.summary()}")
    logger.warning(f"Resource tier: {tier}")

    return profile


# Module-level singleton, set during startup
_hardware_profile: Optional[HardwareProfile] = None


def get_hardware_profile() -> HardwareProfile:
    """Get the cached hardware profile. Must be called after detect_hardware()."""
    if _hardware_profile is None:
        raise RuntimeError("Hardware not yet detected. Call detect_hardware() during startup.")
    return _hardware_profile


def init_hardware(data_dir: str = "./data") -> HardwareProfile:
    """Detect hardware and cache the profile. Called once at startup."""
    global _hardware_profile
    _hardware_profile = detect_hardware(data_dir)
    return _hardware_profile
