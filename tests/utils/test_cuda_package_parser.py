"""
Unit tests for CUDA package parser.
"""

from cuda_healthcheck.utils.cuda_package_parser import (
    _extract_major_minor,
    _extract_version,
    _parse_nvidia_package,
    _parse_torch_version,
    check_cublas_nvjitlink_version_match,
    check_cuopt_nvjitlink_compatibility,
    check_pytorch_cuda_branch_compatibility,
    detect_mixed_cuda_versions,
    format_cuda_packages_report,
    parse_cuda_packages,
    validate_cuda_library_versions,
    validate_torch_branch_compatibility,
)


class TestParseTorchVersion:
    """Test PyTorch version parsing."""

    def test_torch_with_cuda_branch(self):
        """Test parsing torch with CUDA branch."""
        result = _parse_torch_version("torch==2.4.1+cu124")
        assert result == {"version": "2.4.1", "cuda_branch": "cu124"}

    def test_torch_with_cu121(self):
        """Test parsing torch with cu121."""
        result = _parse_torch_version("torch==2.3.0+cu121")
        assert result == {"version": "2.3.0", "cuda_branch": "cu121"}

    def test_torch_cpu_only(self):
        """Test parsing CPU-only torch."""
        result = _parse_torch_version("torch==2.4.1")
        assert result == {"version": "2.4.1", "cuda_branch": None}

    def test_non_torch_package(self):
        """Test non-torch package returns None."""
        result = _parse_torch_version("numpy==1.24.3")
        assert result is None


class TestExtractVersion:
    """Test version extraction."""

    def test_extract_standard_version(self):
        """Test extracting standard version."""
        assert _extract_version("nvidia-cublas-cu12==12.4.5.8") == "12.4.5.8"

    def test_extract_short_version(self):
        """Test extracting short version."""
        assert _extract_version("some-package==1.2.3") == "1.2.3"

    def test_invalid_format(self):
        """Test invalid format returns None."""
        assert _extract_version("invalid-line") is None


class TestExtractMajorMinor:
    """Test major.minor extraction."""

    def test_four_part_version(self):
        """Test extracting from 4-part version."""
        assert _extract_major_minor("12.4.5.8") == "12.4"

    def test_three_part_version(self):
        """Test extracting from 3-part version."""
        assert _extract_major_minor("12.4.127") == "12.4"

    def test_two_part_version(self):
        """Test extracting from 2-part version."""
        assert _extract_major_minor("12.4") == "12.4"

    def test_single_part_version(self):
        """Test single-part version returns None."""
        assert _extract_major_minor("12") is None

    def test_empty_version(self):
        """Test empty version returns None."""
        assert _extract_major_minor("") is None


class TestParseNvidiaPackage:
    """Test NVIDIA package parsing."""

    def test_parse_cuda_runtime(self):
        """Test parsing CUDA runtime package."""
        result = _parse_nvidia_package("nvidia-cuda-runtime-cu12==12.6.77")
        assert result == {"name": "nvidia-cuda-runtime-cu12", "version": "12.6.77"}

    def test_parse_cudnn(self):
        """Test parsing cuDNN package."""
        result = _parse_nvidia_package("nvidia-cudnn-cu12==9.1.0.70")
        assert result == {"name": "nvidia-cudnn-cu12", "version": "9.1.0.70"}

    def test_non_nvidia_package(self):
        """Test non-nvidia package returns None."""
        result = _parse_nvidia_package("torch==2.4.1")
        assert result is None


class TestParseCudaPackages:
    """Test full CUDA packages parsing."""

    def test_databricks_ml_runtime_16_4(self):
        """Test parsing Databricks ML Runtime 16.4 pip freeze."""
        pip_output = """
torch==2.4.1+cu124
nvidia-cublas-cu12==12.4.5.8
nvidia-nvjitlink-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.6.77
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.3.61
"""
        result = parse_cuda_packages(pip_output)

        assert result["torch"] == "2.4.1"
        assert result["torch_cuda_branch"] == "cu124"
        assert result["cublas"]["version"] == "12.4.5.8"
        assert result["cublas"]["major_minor"] == "12.4"
        assert result["nvjitlink"]["version"] == "12.4.127"
        assert result["nvjitlink"]["major_minor"] == "12.4"
        assert "nvidia-cuda-runtime-cu12" in result["other_nvidia"]
        assert result["other_nvidia"]["nvidia-cuda-runtime-cu12"] == "12.6.77"
        assert "nvidia-cudnn-cu12" in result["other_nvidia"]

    def test_cuopt_environment(self):
        """Test parsing environment with CuOPT."""
        pip_output = """
torch==2.4.1+cu124
nvidia-cublas-cu12==12.4.5.8
nvidia-nvjitlink-cu12==12.4.127
cuopt-server-cu12==25.12.0
"""
        result = parse_cuda_packages(pip_output)

        assert result["torch"] == "2.4.1"
        assert result["torch_cuda_branch"] == "cu124"
        assert result["nvjitlink"]["major_minor"] == "12.4"
        assert "cuopt-server-cu12" not in result["other_nvidia"]  # Not nvidia-*

    def test_cpu_only_environment(self):
        """Test parsing CPU-only environment."""
        pip_output = """
torch==2.4.1
numpy==1.24.3
pandas==2.0.3
"""
        result = parse_cuda_packages(pip_output)

        assert result["torch"] == "2.4.1"
        assert result["torch_cuda_branch"] is None
        assert result["cublas"]["version"] is None
        assert result["nvjitlink"]["version"] is None
        assert len(result["other_nvidia"]) == 0

    def test_empty_output(self):
        """Test parsing empty output."""
        result = parse_cuda_packages("")

        assert result["torch"] is None
        assert result["torch_cuda_branch"] is None
        assert result["cublas"]["version"] is None
        assert result["nvjitlink"]["version"] is None

    def test_with_comments(self):
        """Test parsing with comment lines."""
        pip_output = """
# This is a comment
torch==2.4.1+cu124
# Another comment
nvidia-nvjitlink-cu12==12.4.127
"""
        result = parse_cuda_packages(pip_output)

        assert result["torch"] == "2.4.1"
        assert result["nvjitlink"]["version"] == "12.4.127"


class TestFormatCudaPackagesReport:
    """Test report formatting."""

    def test_format_full_report(self):
        """Test formatting full report."""
        packages = {
            "torch": "2.4.1",
            "torch_cuda_branch": "cu124",
            "cublas": {"version": "12.4.5.8", "major_minor": "12.4"},
            "nvjitlink": {"version": "12.4.127", "major_minor": "12.4"},
            "other_nvidia": {
                "nvidia-cuda-runtime-cu12": "12.6.77",
                "nvidia-cudnn-cu12": "9.1.0.70",
            },
        }

        report = format_cuda_packages_report(packages)

        assert "CUDA Packages Report" in report
        assert "PyTorch: 2.4.1 (cu124)" in report
        assert "cuBLAS: 12.4.5.8 (12.4)" in report
        assert "nvJitLink: 12.4.127 (12.4)" in report
        assert "nvidia-cuda-runtime-cu12: 12.6.77" in report

    def test_format_cpu_only(self):
        """Test formatting CPU-only report."""
        packages = {
            "torch": "2.4.1",
            "torch_cuda_branch": None,
            "cublas": {"version": None, "major_minor": None},
            "nvjitlink": {"version": None, "major_minor": None},
            "other_nvidia": {},
        }

        report = format_cuda_packages_report(packages)

        assert "PyTorch: 2.4.1 (CPU-only)" in report
        assert "cuBLAS: Not installed" in report
        assert "nvJitLink: Not installed" in report


class TestCheckCuoptNvjitlinkCompatibility:
    """Test CuOPT nvJitLink compatibility checking."""

    def test_compatible_version(self):
        """Test compatible nvJitLink version."""
        packages = {
            "nvjitlink": {"version": "12.9.79", "major_minor": "12.9"},
        }

        result = check_cuopt_nvjitlink_compatibility(packages)

        assert result["is_compatible"] is True
        assert result["nvjitlink_version"] == "12.9.79"
        assert result["error_message"] is None

    def test_incompatible_version_12_4(self):
        """Test incompatible nvJitLink 12.4 (Databricks)."""
        packages = {
            "nvjitlink": {"version": "12.4.127", "major_minor": "12.4"},
        }

        result = check_cuopt_nvjitlink_compatibility(packages)

        assert result["is_compatible"] is False
        assert result["nvjitlink_version"] == "12.4.127"
        assert "incompatible with CuOPT 25.12+" in result["error_message"]
        assert "PLATFORM CONSTRAINT" in result["error_message"]

    def test_missing_nvjitlink(self):
        """Test missing nvJitLink."""
        packages = {
            "nvjitlink": {"version": None, "major_minor": None},
        }

        result = check_cuopt_nvjitlink_compatibility(packages)

        assert result["is_compatible"] is False
        assert result["error_message"] == "nvJitLink not installed"


class TestCheckPytorchCudaBranchCompatibility:
    """Test PyTorch CUDA branch compatibility checking."""

    def test_compatible_cu124_with_12_4(self):
        """Test compatible cu124 with CUDA 12.4."""
        packages = {
            "torch_cuda_branch": "cu124",
        }

        result = check_pytorch_cuda_branch_compatibility(packages, "12.4")

        assert result["is_compatible"] is True
        assert result["error_message"] is None

    def test_compatible_cu121_with_12_1(self):
        """Test compatible cu121 with CUDA 12.1."""
        packages = {
            "torch_cuda_branch": "cu121",
        }

        result = check_pytorch_cuda_branch_compatibility(packages, "12.1")

        assert result["is_compatible"] is True

    def test_incompatible_cu121_with_12_4(self):
        """Test incompatible cu121 with CUDA 12.4."""
        packages = {
            "torch_cuda_branch": "cu121",
        }

        result = check_pytorch_cuda_branch_compatibility(packages, "12.4")

        assert result["is_compatible"] is False
        assert "does not match expected CUDA 12.4" in result["error_message"]

    def test_missing_cuda_branch(self):
        """Test missing CUDA branch (CPU-only)."""
        packages = {
            "torch_cuda_branch": None,
        }

        result = check_pytorch_cuda_branch_compatibility(packages, "12.4")

        assert result["is_compatible"] is False
        assert "CPU-only" in result["error_message"]


class TestRealWorldScenarios:
    """Test real-world scenarios."""

    def test_databricks_cuopt_incompatibility(self):
        """Test detecting the Databricks CuOPT incompatibility."""
        pip_output = """
torch==2.4.1+cu124
nvidia-cublas-cu12==12.4.5.8
nvidia-nvjitlink-cu12==12.4.127
nvidia-cuda-runtime-cu12==12.6.77
cuopt-server-cu12==25.12.0
"""
        packages = parse_cuda_packages(pip_output)
        compat = check_cuopt_nvjitlink_compatibility(packages)

        assert not compat["is_compatible"]
        assert "12.4.127" in compat["nvjitlink_version"]
        assert "incompatible" in compat["error_message"].lower()

    def test_successful_environment(self):
        """Test a successful, compatible environment."""
        pip_output = """
torch==2.5.0+cu124
nvidia-cublas-cu12==12.6.3.3
nvidia-nvjitlink-cu12==12.9.79
nvidia-cuda-runtime-cu12==12.6.77
"""
        packages = parse_cuda_packages(pip_output)
        nvjitlink_compat = check_cuopt_nvjitlink_compatibility(packages)
        torch_compat = check_pytorch_cuda_branch_compatibility(packages, "12.4")

        assert nvjitlink_compat["is_compatible"]
        assert torch_compat["is_compatible"]

    def test_edge_case_cu118(self):
        """Test older CUDA branch cu118."""
        pip_output = """
torch==2.0.1+cu118
nvidia-nvjitlink-cu12==11.8.89
"""
        packages = parse_cuda_packages(pip_output)

        assert packages["torch"] == "2.0.1"
        assert packages["torch_cuda_branch"] == "cu118"
        assert packages["nvjitlink"]["version"] == "11.8.89"
        assert packages["nvjitlink"]["major_minor"] == "11.8"


class TestCheckCublasNvjitlinkVersionMatch:
    """Test cuBLAS/nvJitLink version matching."""

    def test_matching_versions_12_4(self):
        """Test matching versions (12.4)."""
        result = check_cublas_nvjitlink_version_match("12.4.5.8", "12.4.127")

        assert result["is_mismatch"] is False
        assert result["severity"] == "OK"
        assert result["cublas_major_minor"] == "12.4"
        assert result["nvjitlink_major_minor"] == "12.4"
        assert result["error_message"] is None
        assert result["fix_command"] is None

    def test_matching_versions_12_1(self):
        """Test matching versions (12.1)."""
        result = check_cublas_nvjitlink_version_match("12.1.3.1", "12.1.105")

        assert result["is_mismatch"] is False
        assert result["severity"] == "OK"
        assert result["cublas_major_minor"] == "12.1"
        assert result["nvjitlink_major_minor"] == "12.1"

    def test_mismatch_12_1_vs_12_4(self):
        """Test version mismatch (12.1 vs 12.4)."""
        result = check_cublas_nvjitlink_version_match("12.1.3.1", "12.4.127")

        assert result["is_mismatch"] is True
        assert result["severity"] == "BLOCKER"
        assert result["cublas_major_minor"] == "12.1"
        assert result["nvjitlink_major_minor"] == "12.4"
        assert "CRITICAL" in result["error_message"]
        assert "undefined symbol" in result["error_message"]
        assert "__nvJitLinkAddData_12_1" in result["error_message"]
        assert result["fix_command"] == "pip install --upgrade nvidia-nvjitlink-cu12==12.1.*"

    def test_mismatch_12_4_vs_12_1(self):
        """Test version mismatch (12.4 vs 12.1)."""
        result = check_cublas_nvjitlink_version_match("12.4.5.8", "12.1.105")

        assert result["is_mismatch"] is True
        assert result["severity"] == "BLOCKER"
        assert "__nvJitLinkAddData_12_4" in result["error_message"]
        assert result["fix_command"] == "pip install --upgrade nvidia-nvjitlink-cu12==12.4.*"

    def test_missing_nvjitlink(self):
        """Test missing nvJitLink."""
        result = check_cublas_nvjitlink_version_match("12.4.5.8", None)

        assert result["is_mismatch"] is True
        assert result["severity"] == "BLOCKER"
        assert "NOT INSTALLED" in result["error_message"]
        assert "12.4.*" in result["fix_command"]

    def test_missing_cublas(self):
        """Test missing cuBLAS."""
        result = check_cublas_nvjitlink_version_match(None, "12.4.127")

        assert result["is_mismatch"] is True
        assert result["severity"] == "BLOCKER"
        assert "NOT INSTALLED" in result["error_message"]

    def test_both_missing(self):
        """Test both libraries missing."""
        result = check_cublas_nvjitlink_version_match(None, None)

        assert result["is_mismatch"] is True
        assert result["severity"] == "BLOCKER"
        assert "Missing required libraries" in result["error_message"]

    def test_mismatch_11_8_vs_12_4(self):
        """Test major version mismatch (11.8 vs 12.4)."""
        result = check_cublas_nvjitlink_version_match("11.8.0.0", "12.4.127")

        assert result["is_mismatch"] is True
        assert result["severity"] == "BLOCKER"
        assert result["cublas_major_minor"] == "11.8"
        assert result["nvjitlink_major_minor"] == "12.4"


class TestValidateCudaLibraryVersions:
    """Test comprehensive validation."""

    def test_all_compatible(self):
        """Test fully compatible environment."""
        packages = {
            "torch": "2.4.1",
            "torch_cuda_branch": "cu124",
            "cublas": {"version": "12.4.5.8", "major_minor": "12.4"},
            "nvjitlink": {"version": "12.4.127", "major_minor": "12.4"},
            "other_nvidia": {},
        }

        result = validate_cuda_library_versions(packages)

        assert result["all_compatible"] is True
        assert len(result["blockers"]) == 0
        assert result["checks_passed"] > 0

    def test_cublas_nvjitlink_mismatch(self):
        """Test cuBLAS/nvJitLink mismatch detected."""
        packages = {
            "torch": "2.4.1",
            "torch_cuda_branch": "cu124",
            "cublas": {"version": "12.1.3.1", "major_minor": "12.1"},
            "nvjitlink": {"version": "12.4.127", "major_minor": "12.4"},
            "other_nvidia": {},
        }

        result = validate_cuda_library_versions(packages)

        assert result["all_compatible"] is False
        assert len(result["blockers"]) == 1
        assert result["blockers"][0]["severity"] == "BLOCKER"
        assert "cuBLAS/nvJitLink" in result["blockers"][0]["check"]
        assert result["blockers"][0]["fix_command"] is not None

    def test_cuopt_incompatibility_warning(self):
        """Test CuOPT incompatibility creates warning."""
        packages = {
            "torch": "2.4.1",
            "torch_cuda_branch": "cu124",
            "cublas": {"version": "12.4.5.8", "major_minor": "12.4"},
            "nvjitlink": {"version": "12.4.127", "major_minor": "12.4"},
            "other_nvidia": {},
        }

        result = validate_cuda_library_versions(packages)

        # cuBLAS/nvJitLink match, so no blockers
        assert len(result["blockers"]) == 0
        # But CuOPT incompatibility should create a warning
        assert len(result["warnings"]) == 1
        assert "CuOPT" in result["warnings"][0]["check"]

    def test_multiple_issues(self):
        """Test multiple compatibility issues."""
        packages = {
            "torch": "2.4.1",
            "torch_cuda_branch": "cu124",
            "cublas": {"version": "12.1.3.1", "major_minor": "12.1"},
            "nvjitlink": {"version": "12.4.127", "major_minor": "12.4"},
            "other_nvidia": {},
        }

        result = validate_cuda_library_versions(packages)

        assert result["all_compatible"] is False
        assert result["checks_failed"] > 0
        # Should have cuBLAS/nvJitLink mismatch as blocker
        assert len(result["blockers"]) >= 1


class TestRealWorldScenariosExtended:
    """Test extended real-world scenarios."""

    def test_databricks_ml_runtime_16_4_mismatch(self):
        """Test Databricks ML Runtime 16.4 with cuBLAS/nvJitLink mismatch."""
        pip_output = """
torch==2.4.1+cu124
nvidia-cublas-cu12==12.1.3.1
nvidia-nvjitlink-cu12==12.4.127
"""
        packages = parse_cuda_packages(pip_output)
        result = check_cublas_nvjitlink_version_match(
            packages["cublas"]["version"], packages["nvjitlink"]["version"]
        )

        assert result["is_mismatch"] is True
        assert "12.1" in result["error_message"]
        assert "12.4" in result["error_message"]

    def test_databricks_ml_runtime_16_4_correct(self):
        """Test Databricks ML Runtime 16.4 with correct versions."""
        pip_output = """
torch==2.4.1+cu124
nvidia-cublas-cu12==12.4.5.8
nvidia-nvjitlink-cu12==12.4.127
"""
        packages = parse_cuda_packages(pip_output)
        result = check_cublas_nvjitlink_version_match(
            packages["cublas"]["version"], packages["nvjitlink"]["version"]
        )

        assert result["is_mismatch"] is False
        assert result["severity"] == "OK"


class TestDetectMixedCudaVersions:
    """Test mixed CUDA 11 and CUDA 12 package detection."""

    def test_cuda_12_only(self):
        """Test environment with CUDA 12 packages only (OK)."""
        pip_output = """
torch==2.4.1+cu124
nvidia-cublas-cu12==12.4.5.8
nvidia-nvjitlink-cu12==12.4.127
cudf-cu12==24.10.1
"""
        result = detect_mixed_cuda_versions(pip_output)

        assert result["is_mixed"] is False
        assert result["has_cu11"] is False
        assert result["has_cu12"] is True
        assert result["cu11_count"] == 0
        assert result["cu12_count"] == 4
        assert result["severity"] is None
        assert result["error_message"] is None

    def test_cuda_11_only(self):
        """Test environment with CUDA 11 packages only (OK)."""
        pip_output = """
torch==2.0.1+cu118
nvidia-cublas-cu11==11.8.0.0
cupy-cuda11x==12.3.0
"""
        result = detect_mixed_cuda_versions(pip_output)

        assert result["is_mixed"] is False
        assert result["has_cu11"] is True
        assert result["has_cu12"] is False
        assert result["cu11_count"] == 3
        assert result["cu12_count"] == 0
        assert result["severity"] is None

    def test_mixed_cu11_and_cu12(self):
        """Test mixed CUDA 11 and 12 packages (BLOCKER)."""
        pip_output = """
torch==2.0.1+cu118
nvidia-cublas-cu12==12.4.5.8
cudf-cu12==24.10.1
cupy-cuda11x==12.3.0
"""
        result = detect_mixed_cuda_versions(pip_output)

        assert result["is_mixed"] is True
        assert result["has_cu11"] is True
        assert result["has_cu12"] is True
        assert result["cu11_count"] == 2
        assert result["cu12_count"] == 2
        assert result["severity"] == "BLOCKER"
        assert "CRITICAL" in result["error_message"]
        assert "LD_LIBRARY_PATH" in result["error_message"]
        assert result["fix_command"] is not None
        assert "pip uninstall" in result["fix_command"]
        assert "pip cache purge" in result["fix_command"]

    def test_mixed_torch_cu121_with_cudf_cu12(self):
        """Test PyTorch cu121 mixed with cuDF cu12."""
        pip_output = """
torch==2.1.0+cu121
torchvision==0.16.0+cu121
cudf-cu12==24.10.1
cuml-cu12==24.10.0
"""
        result = detect_mixed_cuda_versions(pip_output)

        # cu121 is CUDA 12.1, so should detect cu12
        assert result["is_mixed"] is False
        assert result["has_cu12"] is True
        assert result["cu12_count"] == 4

    def test_mixed_with_various_naming_patterns(self):
        """Test detection with various -cu11 and -cu12 naming patterns."""
        pip_output = """
some-package-cu11==1.0.0
another-cu118==2.0.0
third-cuda11==3.0.0
nvidia-lib-cu12==12.4.0
stuff-cu124==5.0.0
more-cuda12==6.0.0
"""
        result = detect_mixed_cuda_versions(pip_output)

        assert result["is_mixed"] is True
        assert result["cu11_count"] == 3
        assert result["cu12_count"] == 3
        assert result["severity"] == "BLOCKER"

    def test_empty_output(self):
        """Test with empty pip freeze output."""
        result = detect_mixed_cuda_versions("")

        assert result["is_mixed"] is False
        assert result["cu11_count"] == 0
        assert result["cu12_count"] == 0

    def test_no_cuda_packages(self):
        """Test with no CUDA packages."""
        pip_output = """
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
"""
        result = detect_mixed_cuda_versions(pip_output)

        assert result["is_mixed"] is False
        assert result["has_cu11"] is False
        assert result["has_cu12"] is False

    def test_fix_command_includes_all_packages(self):
        """Test that fix command includes all mixed packages."""
        pip_output = """
torch==2.0.1+cu118
nvidia-cublas-cu12==12.4.5.8
cudf-cu12==24.10.1
"""
        result = detect_mixed_cuda_versions(pip_output)

        assert "torch" in result["fix_command"]
        assert "nvidia-cublas-cu12" in result["fix_command"]
        assert "cudf-cu12" in result["fix_command"]

    def test_databricks_scenario_mixed(self):
        """Test real Databricks scenario with mixed packages."""
        pip_output = """
torch==2.4.1+cu124
torchvision==0.19.1+cu124
nvidia-cublas-cu12==12.4.5.8
nvidia-cudnn-cu12==9.1.0.70
cupy-cuda11x==12.3.0
"""
        result = detect_mixed_cuda_versions(pip_output)

        assert result["is_mixed"] is True
        assert result["severity"] == "BLOCKER"
        assert "cupy-cuda11x" in result["cu11_packages"]
        assert len(result["cu12_packages"]) == 4


class TestMixedCudaRealWorldScenarios:
    """Test real-world mixed CUDA scenarios."""

    def test_accidental_cupy_cu11_with_torch_cu12(self):
        """Test common mistake: installing cupy-cuda11x on CUDA 12 cluster."""
        pip_output = """
torch==2.4.1+cu124
cupy-cuda11x==12.3.0
"""
        result = detect_mixed_cuda_versions(pip_output)

        assert result["is_mixed"] is True
        assert "torch" in str(result["cu12_packages"])
        assert "cupy-cuda11x" in result["cu11_packages"]

    def test_legacy_torch_with_new_rapids(self):
        """Test legacy PyTorch cu118 with new RAPIDS cu12."""
        pip_output = """
torch==2.0.1+cu118
torchvision==0.15.2+cu118
cudf-cu12==24.10.1
cuml-cu12==24.10.0
"""
        result = detect_mixed_cuda_versions(pip_output)

        assert result["is_mixed"] is True
        assert result["cu11_count"] == 2
        assert result["cu12_count"] == 2


class TestValidateTorchBranchCompatibility:
    """Test PyTorch CUDA branch compatibility with Databricks runtimes."""

    def test_runtime_14_3_with_cu120_compatible(self):
        """Test Runtime 14.3 with cu120 (compatible)."""
        result = validate_torch_branch_compatibility(14.3, "cu120")

        assert result["is_compatible"] is True
        assert result["severity"] is None
        assert result["runtime_cuda"] == "12.0"
        assert result["runtime_driver"] == 535

    def test_runtime_14_3_with_cu121_compatible(self):
        """Test Runtime 14.3 with cu121 (compatible)."""
        result = validate_torch_branch_compatibility(14.3, "cu121")

        assert result["is_compatible"] is True
        assert result["severity"] is None

    def test_runtime_14_3_with_cu124_blocker(self):
        """Test Runtime 14.3 with cu124 (BLOCKER - critical case)."""
        result = validate_torch_branch_compatibility(14.3, "cu124")

        assert result["is_compatible"] is False
        assert result["severity"] == "BLOCKER"
        assert result["runtime_cuda"] == "12.0"
        assert result["runtime_driver"] == 535
        assert "CRITICAL" in result["issue"]
        assert "INCOMPATIBLE" in result["issue"]
        assert len(result["fix_options"]) == 2
        assert "Option 1: Downgrade PyTorch" in result["fix_options"][0]
        assert "cu120, cu121" in result["fix_options"][0]
        assert "Option 2: Upgrade to Databricks Runtime 15.1+" in result["fix_options"][1]

    def test_runtime_15_1_with_cu124_compatible(self):
        """Test Runtime 15.1 with cu124 (compatible)."""
        result = validate_torch_branch_compatibility(15.1, "cu124")

        assert result["is_compatible"] is True
        assert result["severity"] is None
        assert result["runtime_cuda"] == "12.4"
        assert result["runtime_driver"] == 550

    def test_runtime_15_2_with_cu124_compatible(self):
        """Test Runtime 15.2 with cu124 (compatible)."""
        result = validate_torch_branch_compatibility(15.2, "cu124")

        assert result["is_compatible"] is True
        assert result["severity"] is None

    def test_runtime_15_1_with_cu120_compatible(self):
        """Test Runtime 15.1 with cu120 (backward compatible)."""
        result = validate_torch_branch_compatibility(15.1, "cu120")

        assert result["is_compatible"] is True

    def test_runtime_16_4_with_cu124_compatible(self):
        """Test Runtime 16.4 with cu124 (compatible)."""
        result = validate_torch_branch_compatibility(16.4, "cu124")

        assert result["is_compatible"] is True
        assert result["runtime_cuda"] == "12.6"
        assert result["runtime_driver"] == 560

    def test_unknown_runtime_warning(self):
        """Test unknown runtime version returns warning."""
        result = validate_torch_branch_compatibility(99.9, "cu124")

        assert result["is_compatible"] is False
        assert result["severity"] == "WARNING"
        assert "Unknown Databricks runtime" in result["issue"]

    def test_branch_normalization(self):
        """Test CUDA branch normalization (cu1240 → cu124)."""
        result = validate_torch_branch_compatibility(15.1, "cu1240")

        assert result["is_compatible"] is True

    def test_fix_options_content_runtime_14_3(self):
        """Test fix options provide correct guidance for Runtime 14.3."""
        result = validate_torch_branch_compatibility(14.3, "cu124")

        # Check Option 1 (downgrade PyTorch)
        assert "cu120, cu121" in result["fix_options"][0]
        assert "download.pytorch.org/whl/cu121" in result["fix_options"][0]

        # Check Option 2 (upgrade runtime)
        assert "Runtime 15.1+" in result["fix_options"][1]
        assert "Driver 550" in result["fix_options"][1]


class TestTorchBranchRealWorldScenarios:
    """Test real-world PyTorch CUDA branch scenarios."""

    def test_databricks_14_3_cu124_issue(self):
        """Test the critical Databricks 14.3 + cu124 incompatibility."""
        result = validate_torch_branch_compatibility(14.3, "cu124")

        assert result["is_compatible"] is False
        assert result["severity"] == "BLOCKER"
        assert "immutable" in result["issue"].lower()
        assert "cannot upgrade" in result["issue"].lower()
        assert "Missing CUDA API symbols" in result["issue"]
        assert "Segmentation faults" in result["issue"]

    def test_user_upgrades_runtime_to_15_2(self):
        """Test user upgrades from 14.3 to 15.2 to use cu124."""
        # Before: Runtime 14.3 + cu124 (BLOCKER)
        result_before = validate_torch_branch_compatibility(14.3, "cu124")
        assert result_before["is_compatible"] is False

        # After: Runtime 15.2 + cu124 (OK)
        result_after = validate_torch_branch_compatibility(15.2, "cu124")
        assert result_after["is_compatible"] is True

    def test_user_downgrades_torch_to_cu121(self):
        """Test user downgrades PyTorch from cu124 to cu121 on Runtime 14.3."""
        # Before: Runtime 14.3 + cu124 (BLOCKER)
        result_before = validate_torch_branch_compatibility(14.3, "cu124")
        assert result_before["is_compatible"] is False

        # After: Runtime 14.3 + cu121 (OK)
        result_after = validate_torch_branch_compatibility(14.3, "cu121")
        assert result_after["is_compatible"] is True

    def test_all_runtimes_with_cu120(self):
        """Test cu120 is compatible with all runtimes."""
        runtimes = [14.3, 15.1, 15.2, 16.0, 16.4]

        for runtime in runtimes:
            result = validate_torch_branch_compatibility(runtime, "cu120")
            assert result["is_compatible"] is True, f"cu120 should work on Runtime {runtime}"

    def test_cu124_requires_15_1_minimum(self):
        """Test cu124 requires minimum Runtime 15.1."""
        # Runtime 14.3 + cu124 (BLOCKER)
        result_14_3 = validate_torch_branch_compatibility(14.3, "cu124")
        assert result_14_3["is_compatible"] is False

        # Runtime 15.1 + cu124 (OK)
        result_15_1 = validate_torch_branch_compatibility(15.1, "cu124")
        assert result_15_1["is_compatible"] is True
