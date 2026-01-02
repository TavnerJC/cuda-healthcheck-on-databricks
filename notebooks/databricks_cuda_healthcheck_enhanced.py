#!/usr/bin/env python3
"""
Databricks CUDA Healthcheck - Integrated Detection Script

Comprehensive CUDA environment validation combining all detection layers:
- Layer 1: Environment Detection (Runtime, Driver, CUDA versions)
- Layer 2: CUDA Library Inventory (torch, cuBLAS, nvJitLink)
- Layer 3: Dependency Conflicts (mixed CUDA versions, version mismatches)
- Layer 4: DataDesigner Compatibility (feature detection, CUDA availability)

Exit Codes:
    0 - No blockers detected, environment ready
    1 - Blockers detected, fixes required

Usage:
    python databricks_cuda_healthcheck_enhanced.py

    Or in Databricks notebook:
    %run ./databricks_cuda_healthcheck_enhanced.py
"""

import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from cuda_healthcheck import CUDADetector

# Breaking Changes Detection
from cuda_healthcheck.data import BreakingChangesDatabase

# Layer 1: Environment Detection
from cuda_healthcheck.databricks import (
    detect_databricks_runtime,
    detect_gpu_auto,
    get_driver_version_for_runtime,
)

# Layer 4: DataDesigner Detection
from cuda_healthcheck.nemo import (
    detect_enabled_features,
    diagnose_cuda_availability,
    get_feature_validation_report,
)

# Recommendations Generator
# Layer 3: Compatibility Detection
# Layer 2: CUDA Package Detection
from cuda_healthcheck.utils import (
    check_cublas_nvjitlink_version_match,
    detect_mixed_cuda_versions,
    format_recommendations_for_notebook,
    get_cuda_packages_from_pip,
    parse_cuda_packages,
    validate_cuda_library_versions,
    validate_torch_branch_compatibility,
)


class IntegratedHealthcheckReport:
    """Generates formatted healthcheck report combining all detection layers."""

    def __init__(self):
        self.layers = {
            "layer1": {"name": "Environment Detection", "status": "PENDING", "data": {}},
            "layer2": {"name": "CUDA Library Inventory", "status": "PENDING", "data": {}},
            "layer3": {"name": "Dependency Conflicts", "status": "PENDING", "data": {}},
            "layer4": {"name": "DataDesigner Compatibility", "status": "PENDING", "data": {}},
        }
        self.blockers = []
        self.warnings = []

    def run_layer1_environment_detection(self):
        """Layer 1: Detect Databricks runtime, driver, and CUDA versions."""
        print("🔍 Running Layer 1: Environment Detection...")

        try:
            # Detect runtime
            runtime_info = detect_databricks_runtime()
            runtime_version = runtime_info.get("runtime_version")

            # Detect driver
            driver_info = None
            if runtime_version:
                try:
                    driver_info = get_driver_version_for_runtime(runtime_version)
                except Exception:
                    pass

            # Detect GPU
            gpu_info = detect_gpu_auto()

            # Detect CUDA environment
            detector = CUDADetector()
            env = detector.detect_environment()

            self.layers["layer1"]["data"] = {
                "runtime_version": runtime_version,
                "runtime_info": runtime_info,
                "driver_info": driver_info,
                "gpu_info": gpu_info,
                "cuda_env": env,
            }
            self.layers["layer1"]["status"] = "OK"

        except Exception as e:
            self.layers["layer1"]["status"] = "ERROR"
            self.layers["layer1"]["error"] = str(e)
            self.warnings.append(
                {
                    "layer": "Layer 1",
                    "issue": f"Environment detection error: {e}",
                }
            )

    def run_layer2_library_inventory(self):
        """Layer 2: Inventory CUDA libraries from pip."""
        print("🔍 Running Layer 2: CUDA Library Inventory...")

        try:
            # Get pip freeze output
            pip_output = get_cuda_packages_from_pip()

            # Parse packages
            packages = parse_cuda_packages(pip_output)

            self.layers["layer2"]["data"] = {
                "pip_output": pip_output,
                "packages": packages,
            }
            self.layers["layer2"]["status"] = "OK"

        except Exception as e:
            self.layers["layer2"]["status"] = "ERROR"
            self.layers["layer2"]["error"] = str(e)
            self.warnings.append(
                {
                    "layer": "Layer 2",
                    "issue": f"Library inventory error: {e}",
                }
            )

    def run_layer3_conflict_detection(self):
        """Layer 3: Detect dependency conflicts and version mismatches."""
        print("🔍 Running Layer 3: Dependency Conflict Detection...")

        try:
            packages = self.layers["layer2"]["data"].get("packages", {})
            pip_output = self.layers["layer2"]["data"].get("pip_output", "")
            runtime_version = self.layers["layer1"]["data"].get("runtime_version")

            # Check 1: cuBLAS/nvJitLink version match
            cublas_version = packages.get("cublas", {}).get("version")
            nvjitlink_version = packages.get("nvjitlink", {}).get("version")

            cublas_check = check_cublas_nvjitlink_version_match(cublas_version, nvjitlink_version)

            if cublas_check["is_mismatch"]:
                self.blockers.append(
                    {
                        "layer": "Layer 3",
                        "check": "cuBLAS/nvJitLink Version Match",
                        "severity": "BLOCKER",
                        "root_cause": "nvjitlink_mismatch",
                        "issue": cublas_check["error_message"],
                        "fix_command": cublas_check["fix_command"],
                    }
                )

            # Check 2: Mixed CUDA 11/12 packages
            mixed_check = detect_mixed_cuda_versions(pip_output)

            if mixed_check["severity"] == "BLOCKER":
                self.blockers.append(
                    {
                        "layer": "Layer 3",
                        "check": "Mixed CUDA Versions",
                        "severity": "BLOCKER",
                        "root_cause": "mixed_cuda_versions",
                        "issue": mixed_check["error_message"],
                        "fix_command": mixed_check["fix_command"],
                    }
                )

            # Check 3: PyTorch CUDA branch compatibility
            if runtime_version:
                torch_branch = packages.get("torch_cuda_branch")
                if torch_branch:
                    branch_check = validate_torch_branch_compatibility(
                        runtime_version, torch_branch
                    )

                    if not branch_check["is_compatible"]:
                        self.blockers.append(
                            {
                                "layer": "Layer 3",
                                "check": "PyTorch CUDA Branch Compatibility",
                                "severity": "BLOCKER",
                                "root_cause": "torch_branch_incompatible",
                                "issue": branch_check["issue"],
                                "fix_options": branch_check.get("fix_options", []),
                            }
                        )

            # Comprehensive validation
            validation = validate_cuda_library_versions(packages)

            self.layers["layer3"]["data"] = {
                "cublas_check": cublas_check,
                "mixed_check": mixed_check,
                "validation": validation,
            }
            self.layers["layer3"]["status"] = "OK" if len(self.blockers) == 0 else "BLOCKERS"

        except Exception as e:
            self.layers["layer3"]["status"] = "ERROR"
            self.layers["layer3"]["error"] = str(e)
            self.warnings.append(
                {
                    "layer": "Layer 3",
                    "issue": f"Conflict detection error: {e}",
                }
            )

    def run_layer4_datadesigner_compatibility(self):
        """Layer 4: Detect DataDesigner features and validate CUDA availability."""
        print("🔍 Running Layer 4: DataDesigner Compatibility...")

        try:
            # Detect enabled features
            features = detect_enabled_features()

            # Get data from previous layers
            packages = self.layers["layer2"]["data"].get("packages", {})
            runtime_version = self.layers["layer1"]["data"].get("runtime_version")
            cuda_env = self.layers["layer1"]["data"].get("cuda_env")

            # Get driver version
            driver_version = None
            if cuda_env and cuda_env.cuda_driver_version != "Not available":
                try:
                    driver_version = int(cuda_env.cuda_driver_version.split(".")[0])
                except Exception:
                    pass

            # Diagnose CUDA availability
            cuda_diag = diagnose_cuda_availability(
                features_enabled=features,
                runtime_version=runtime_version,
                torch_cuda_branch=packages.get("torch_cuda_branch"),
                driver_version=driver_version,
            )

            # Add blocker if CUDA diagnostics found issues
            if cuda_diag["severity"] == "BLOCKER":
                self.blockers.append(
                    {
                        "layer": "Layer 4",
                        "check": "CUDA Availability",
                        "severity": "BLOCKER",
                        "root_cause": cuda_diag["diagnostics"].get("root_cause"),
                        "issue": cuda_diag["diagnostics"]["issue"],
                        "fix_options": cuda_diag.get("fix_options", []),
                        "fix_command": cuda_diag.get("fix_command"),
                    }
                )

            # Validate feature requirements
            torch_version = packages.get("torch")
            torch_cuda_branch = packages.get("torch_cuda_branch")
            cuda_available = bool(
                cuda_env
                and cuda_env.cuda_runtime_version
                and cuda_env.cuda_runtime_version != "Not available"
            )

            # Get GPU memory
            gpu_memory_gb = None
            gpu_info = self.layers["layer1"]["data"].get("gpu_info", {})
            if gpu_info.get("gpus"):
                first_gpu = gpu_info["gpus"][0]
                memory_str = first_gpu.get("memory_total", "")
                if "MiB" in memory_str:
                    try:
                        memory_mb = float(memory_str.replace("MiB", "").strip())
                        gpu_memory_gb = memory_mb / 1024.0
                    except Exception:
                        pass

            feature_report = get_feature_validation_report(
                features=features,
                torch_version=torch_version,
                torch_cuda_branch=torch_cuda_branch,
                cuda_available=cuda_available,
                gpu_memory_gb=gpu_memory_gb,
            )

            # Add feature validation blockers
            for blocker in feature_report.get("blockers", []):
                self.blockers.append(
                    {
                        "layer": "Layer 4",
                        "check": f"Feature Requirement: {blocker['feature']}",
                        "severity": "BLOCKER",
                        "issue": blocker["message"],
                        "fix_options": blocker.get("fix_commands", []),
                    }
                )

            self.layers["layer4"]["data"] = {
                "features": features,
                "cuda_diag": cuda_diag,
                "feature_report": feature_report,
            }
            self.layers["layer4"]["status"] = (
                "OK" if cuda_diag["severity"] != "BLOCKER" else "BLOCKERS"
            )

        except Exception as e:
            self.layers["layer4"]["status"] = "ERROR"
            self.layers["layer4"]["error"] = str(e)
            self.warnings.append(
                {
                    "layer": "Layer 4",
                    "issue": f"DataDesigner compatibility error: {e}",
                }
            )

    def generate_report(self) -> str:
        """Generate formatted healthcheck report."""
        lines = []
        separator = "═" * 80

        # Header
        lines.append(separator)
        lines.append("DATABRICKS CUDA HEALTHCHECK REPORT")
        lines.append(separator)
        lines.append(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        lines.append("")

        # Layer 1: Environment Detection
        self._add_layer1_report(lines)

        # Layer 2: CUDA Library Inventory
        self._add_layer2_report(lines)

        # Layer 3: Dependency Conflicts
        self._add_layer3_report(lines)

        # Layer 4: DataDesigner Compatibility
        self._add_layer4_report(lines)

        # Blockers Section
        if self.blockers:
            lines.append(separator)
            lines.append("❌ BLOCKERS DETECTED: Fix required before installation")
            lines.append("")

            for i, blocker in enumerate(self.blockers, 1):
                lines.append(f"   BLOCKER {i}: {blocker.get('check', 'Unknown')}")
                lines.append(f"   → {blocker.get('issue', 'No details')}")

                if blocker.get("fix_command"):
                    lines.append(f"   → Fix: {blocker['fix_command']}")
                elif blocker.get("fix_options"):
                    lines.append("   → Fix options:")
                    for j, option in enumerate(blocker["fix_options"], 1):
                        lines.append(f"      {j}. {option}")
                lines.append("")

        # Warnings Section
        if self.warnings:
            lines.append(separator)
            lines.append("⚠️  WARNINGS:")
            lines.append("")
            for warning in self.warnings:
                lines.append(f"   {warning.get('layer', 'Unknown')}: {warning.get('issue')}")
            lines.append("")

        # Final Result
        lines.append(separator)
        if len(self.blockers) == 0:
            lines.append("✅ RESULT: Ready to install DataDesigner")
        else:
            lines.append(f"❌ RESULT: Fix {len(self.blockers)} blocker(s) before installation")
        lines.append(separator)

        return "\n".join(lines)

    def _add_layer1_report(self, lines: List[str]):
        """Add Layer 1 report section."""
        status_icon = self._get_status_icon(self.layers["layer1"]["status"])
        lines.append(f"{status_icon} Layer 1: {self.layers['layer1']['name']}")

        data = self.layers["layer1"]["data"]
        runtime_version = data.get("runtime_version")
        driver_info = data.get("driver_info")
        gpu_info = data.get("gpu_info", {})
        cuda_env = data.get("cuda_env")

        if runtime_version:
            lines.append(f"   - Runtime: {runtime_version}")
        if driver_info:
            lines.append(
                f"   - Driver: {driver_info.get('driver_min')}-{driver_info.get('driver_max')}"
            )
            lines.append(f"   - CUDA: {driver_info.get('cuda_version')}")
        if cuda_env:
            lines.append(f"   - CUDA Runtime: {cuda_env.cuda_runtime_version}")
            lines.append(f"   - CUDA Driver: {cuda_env.cuda_driver_version}")

        if gpu_info.get("gpus"):
            gpu = gpu_info["gpus"][0]
            lines.append(f"   - GPU: {gpu.get('name', 'Unknown')}")
            lines.append(f"   - GPU Memory: {gpu.get('memory_total', 'N/A')}")

        lines.append("")

    def _add_layer2_report(self, lines: List[str]):
        """Add Layer 2 report section."""
        status_icon = self._get_status_icon(self.layers["layer2"]["status"])
        lines.append(f"{status_icon} Layer 2: {self.layers['layer2']['name']}")

        packages = self.layers["layer2"]["data"].get("packages", {})

        if packages.get("torch"):
            torch_str = f"{packages['torch']}"
            if packages.get("torch_cuda_branch"):
                torch_str += f" ({packages['torch_cuda_branch']})"
            lines.append(f"   - torch: {torch_str}")

        if packages.get("cublas", {}).get("version"):
            lines.append(f"   - cublas: {packages['cublas']['version']}")

        if packages.get("nvjitlink", {}).get("version"):
            cublas_mm = packages.get("cublas", {}).get("major_minor", "")
            nvjit_mm = packages.get("nvjitlink", {}).get("major_minor", "")
            match_icon = "✅" if cublas_mm == nvjit_mm else "❌"
            lines.append(f"   - nvjitlink: {packages['nvjitlink']['version']} {match_icon}")

        if not packages.get("torch"):
            lines.append("   - torch: Not installed")

        lines.append("")

    def _add_layer3_report(self, lines: List[str]):
        """Add Layer 3 report section."""
        status_icon = self._get_status_icon(self.layers["layer3"]["status"])
        lines.append(f"{status_icon} Layer 3: {self.layers['layer3']['name']}")

        data = self.layers["layer3"]["data"]
        mixed_check = data.get("mixed_check", {})

        if mixed_check.get("severity") == "BLOCKER":
            lines.append(f"   - ❌ Mixed CUDA 11/12 detected!")
            lines.append(f"      cu11: {len(mixed_check.get('cu11_packages', []))} packages")
            lines.append(f"      cu12: {len(mixed_check.get('cu12_packages', []))} packages")
        else:
            lines.append("   - No mixed cu11/cu12 detected")

        cublas_check = data.get("cublas_check", {})
        if cublas_check.get("is_mismatch"):
            lines.append("   - ❌ cuBLAS/nvJitLink version mismatch")
        else:
            lines.append("   - cuBLAS/nvJitLink versions match")

        lines.append("")

    def _add_layer4_report(self, lines: List[str]):
        """Add Layer 4 report section."""
        status_icon = self._get_status_icon(self.layers["layer4"]["status"])
        lines.append(f"{status_icon} Layer 4: {self.layers['layer4']['name']}")

        cuda_diag = self.layers["layer4"]["data"].get("cuda_diag", {})

        if cuda_diag:
            cuda_available = cuda_diag.get("cuda_available", False)
            lines.append(f"   - torch.cuda.is_available(): {cuda_available}")

            if cuda_diag.get("gpu_device"):
                lines.append(f"   - GPU device: {cuda_diag['gpu_device']}")

            if cuda_diag["severity"] == "BLOCKER":
                lines.append(f"   - ❌ CUDA not available")
            elif cuda_diag["severity"] == "SKIPPED":
                lines.append(f"   - ⏭️  CUDA check skipped (not required)")

        features = self.layers["layer4"]["data"].get("features", {})
        enabled_count = sum(1 for f in features.values() if f.is_enabled)
        if enabled_count > 0:
            lines.append(f"   - DataDesigner features enabled: {enabled_count}")

        lines.append("")

    def _get_status_icon(self, status: str) -> str:
        """Get icon for status."""
        icons = {
            "OK": "✅",
            "BLOCKERS": "❌",
            "WARNING": "⚠️",
            "ERROR": "⚠️",
            "PENDING": "⏳",
        }
        return icons.get(status, "❓")


def main() -> int:
    """
    Main entry point for integrated healthcheck.

    Returns:
        0 if no blockers detected
        1 if blockers detected
    """
    print("\n🔍 Starting Databricks CUDA Healthcheck...")
    print("=" * 80)

    # Create report generator
    report = IntegratedHealthcheckReport()

    # Run all detection layers
    try:
        report.run_layer1_environment_detection()
        report.run_layer2_library_inventory()
        report.run_layer3_conflict_detection()
        report.run_layer4_datadesigner_compatibility()
    except Exception as e:
        print(f"\n❌ Fatal error during healthcheck: {e}")
        return 1

    # Generate and print report
    print("\n")
    report_text = report.generate_report()
    print(report_text)

    # Return appropriate exit code
    if len(report.blockers) > 0:
        print("\n💡 Tip: Address blockers above before installing DataDesigner")
        return 1
    else:
        print("\n✅ Environment validated successfully!")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
