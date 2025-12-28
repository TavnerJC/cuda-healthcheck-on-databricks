"""
CUDA Environment Detector for Databricks Clusters.

This module detects CUDA versions, GPU properties, and library compatibility
on Databricks GPU-enabled clusters.
"""

import json
import subprocess
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class GPUInfo:
    """GPU information structure."""

    name: str
    driver_version: str
    cuda_version: str
    compute_capability: str
    memory_total_mb: int
    gpu_index: int


@dataclass
class LibraryInfo:
    """Installed library version information."""

    name: str
    version: str
    cuda_version: Optional[str]
    is_compatible: bool
    warnings: List[str]


@dataclass
class CUDAEnvironment:
    """Complete CUDA environment information."""

    cuda_runtime_version: Optional[str]
    cuda_driver_version: Optional[str]
    nvcc_version: Optional[str]
    gpus: List[GPUInfo]
    libraries: List[LibraryInfo]
    breaking_changes: List[Dict[str, Any]]
    timestamp: str


class CUDADetector:
    """Detects CUDA environment and library compatibility on Databricks clusters."""

    def __init__(self):
        """Initialize the CUDA detector."""
        self.cuda_paths = [
            "/usr/local/cuda",
            "/usr/local/cuda-12.4",
            "/usr/local/cuda-12.6",
            "/usr/local/cuda-13.0",
        ]

    def detect_nvidia_smi(self) -> Dict[str, Any]:
        """
        Detect GPU information using nvidia-smi.

        Returns:
            Dictionary containing driver version, CUDA version, and GPU details.
        """
        try:
            # Get driver and CUDA version
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version,name,memory.total,compute_cap,index",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                return {"error": "nvidia-smi failed", "details": result.stderr}

            # Parse nvidia-smi version output for CUDA version
            version_result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10
            )

            cuda_version = None
            if version_result.returncode == 0:
                # Extract CUDA version from header (e.g., "CUDA Version: 12.4")
                match = re.search(r"CUDA Version:\s+(\d+\.\d+)", version_result.stdout)
                if match:
                    cuda_version = match.group(1)

            # Parse GPU information
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        gpus.append(
                            GPUInfo(
                                name=parts[1],
                                driver_version=parts[0],
                                cuda_version=cuda_version or "Unknown",
                                compute_capability=parts[3],
                                memory_total_mb=int(float(parts[2])),
                                gpu_index=int(parts[4]),
                            )
                        )

            return {
                "driver_version": gpus[0].driver_version if gpus else None,
                "cuda_version": cuda_version,
                "gpus": gpus,
                "success": True,
            }

        except FileNotFoundError:
            return {"error": "nvidia-smi not found", "success": False}
        except subprocess.TimeoutExpired:
            return {"error": "nvidia-smi timeout", "success": False}
        except Exception as e:
            return {"error": f"nvidia-smi error: {str(e)}", "success": False}

    def detect_cuda_runtime(self) -> Optional[str]:
        """
        Detect CUDA runtime version from /usr/local/cuda.

        Returns:
            CUDA runtime version string or None if not found.
        """
        for cuda_path in self.cuda_paths:
            version_file = Path(cuda_path) / "version.json"
            version_txt = Path(cuda_path) / "version.txt"

            # Try version.json first (newer CUDA versions)
            if version_file.exists():
                try:
                    with open(version_file, "r") as f:
                        version_data = json.load(f)
                        return version_data.get("cuda", {}).get("version")
                except Exception:
                    pass

            # Try version.txt (older CUDA versions)
            if version_txt.exists():
                try:
                    with open(version_txt, "r") as f:
                        content = f.read().strip()
                        match = re.search(r"CUDA Version\s+(\d+\.\d+\.\d+)", content)
                        if match:
                            return match.group(1)
                except Exception:
                    pass

            # Check if this CUDA path exists and extract version from path
            if Path(cuda_path).exists():
                match = re.search(r"cuda-(\d+\.\d+)", cuda_path)
                if match:
                    return match.group(1)

        return None

    def detect_nvcc_version(self) -> Optional[str]:
        """
        Detect nvcc (CUDA compiler) version.

        Returns:
            nvcc version string or None if not found.
        """
        try:
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                # Extract version from output (e.g., "release 12.4, V12.4.131")
                match = re.search(r"release (\d+\.\d+)", result.stdout)
                if match:
                    return match.group(1)

            return None

        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    def detect_pytorch(self) -> LibraryInfo:
        """
        Detect PyTorch installation and CUDA compatibility.

        Returns:
            LibraryInfo object with PyTorch details.
        """
        warnings = []

        try:
            import torch

            version = torch.__version__
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else None

            if not cuda_available:
                warnings.append("PyTorch CUDA not available - CPU-only build detected")

            # Check for version compatibility issues
            if cuda_version:
                cuda_major = int(cuda_version.split(".")[0])
                if cuda_major >= 13:
                    warnings.append("PyTorch may need rebuild for CUDA 13.x compatibility")

            return LibraryInfo(
                name="pytorch",
                version=version,
                cuda_version=cuda_version,
                is_compatible=cuda_available,
                warnings=warnings,
            )

        except ImportError:
            return LibraryInfo(
                name="pytorch",
                version="Not installed",
                cuda_version=None,
                is_compatible=False,
                warnings=["PyTorch not installed"],
            )

    def detect_tensorflow(self) -> LibraryInfo:
        """
        Detect TensorFlow installation and CUDA compatibility.

        Returns:
            LibraryInfo object with TensorFlow details.
        """
        warnings = []

        try:
            import tensorflow as tf

            version = tf.__version__

            # Check GPU availability
            gpus = tf.config.list_physical_devices("GPU")
            cuda_available = len(gpus) > 0

            if not cuda_available:
                warnings.append("TensorFlow GPU not available")

            # TensorFlow doesn't expose CUDA version directly
            # Check build info for CUDA version
            cuda_version = None
            try:
                build_info = tf.sysconfig.get_build_info()
                cuda_version = build_info.get("cuda_version")
            except Exception:
                pass

            return LibraryInfo(
                name="tensorflow",
                version=version,
                cuda_version=cuda_version,
                is_compatible=cuda_available,
                warnings=warnings,
            )

        except ImportError:
            return LibraryInfo(
                name="tensorflow",
                version="Not installed",
                cuda_version=None,
                is_compatible=False,
                warnings=["TensorFlow not installed"],
            )

    def detect_cudf(self) -> LibraryInfo:
        """
        Detect cuDF (RAPIDS) installation and CUDA compatibility.

        Returns:
            LibraryInfo object with cuDF details.
        """
        warnings = []

        try:
            import cudf

            version = cudf.__version__

            # cuDF version is tightly coupled to CUDA version
            # RAPIDS 24.12+ requires CUDA 12.x
            major_version = int(version.split(".")[0])

            if major_version >= 24:
                warnings.append("cuDF 24.x+ requires CUDA 12.x or higher")

            return LibraryInfo(
                name="cudf",
                version=version,
                cuda_version=None,  # cuDF doesn't expose this directly
                is_compatible=True,
                warnings=warnings,
            )

        except ImportError:
            return LibraryInfo(
                name="cudf",
                version="Not installed",
                cuda_version=None,
                is_compatible=False,
                warnings=["cuDF not installed"],
            )

    def detect_all_libraries(self) -> List[LibraryInfo]:
        """
        Detect all supported libraries and their CUDA compatibility.

        Returns:
            List of LibraryInfo objects for all detected libraries.
        """
        return [self.detect_pytorch(), self.detect_tensorflow(), self.detect_cudf()]

    def detect_environment(self) -> CUDAEnvironment:
        """
        Perform complete CUDA environment detection.

        Returns:
            CUDAEnvironment object with all detection results.
        """
        from datetime import datetime

        # Detect GPU and driver info
        nvidia_info = self.detect_nvidia_smi()

        # Detect CUDA versions
        cuda_runtime = self.detect_cuda_runtime()
        nvcc_version = self.detect_nvcc_version()

        # Detect libraries
        libraries = self.detect_all_libraries()

        # Identify breaking changes (to be implemented with breaking_changes module)
        breaking_changes: List[Dict[str, Any]] = []

        return CUDAEnvironment(
            cuda_runtime_version=cuda_runtime,
            cuda_driver_version=nvidia_info.get("cuda_version"),
            nvcc_version=nvcc_version,
            gpus=nvidia_info.get("gpus", []),
            libraries=libraries,
            breaking_changes=breaking_changes,
            timestamp=datetime.utcnow().isoformat(),
        )

    def to_dict(self, env: CUDAEnvironment) -> Dict[str, Any]:
        """
        Convert CUDAEnvironment to dictionary for JSON serialization.

        Args:
            env: CUDAEnvironment object

        Returns:
            Dictionary representation suitable for JSON output.
        """
        return {
            "cuda_runtime_version": env.cuda_runtime_version,
            "cuda_driver_version": env.cuda_driver_version,
            "nvcc_version": env.nvcc_version,
            "gpus": [asdict(gpu) for gpu in env.gpus],
            "libraries": [asdict(lib) for lib in env.libraries],
            "breaking_changes": env.breaking_changes,
            "timestamp": env.timestamp,
        }

    def to_json(self, env: CUDAEnvironment, indent: int = 2) -> str:
        """
        Convert CUDAEnvironment to JSON string.

        Args:
            env: CUDAEnvironment object
            indent: JSON indentation level

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict(env), indent=indent)


def detect_cuda_environment() -> Dict[str, Any]:
    """
    Convenience function to detect CUDA environment and return as dictionary.

    Returns:
        Dictionary with complete CUDA environment information.
    """
    detector = CUDADetector()
    environment = detector.detect_environment()
    return detector.to_dict(environment)


if __name__ == "__main__":
    # Quick test when run directly
    detector = CUDADetector()
    environment = detector.detect_environment()
    print(detector.to_json(environment))
