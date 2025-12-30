# Changelog

All notable changes to the CUDA Healthcheck on Databricks project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.0] - 2025-12-30

### 🎉 Major Release: Rebrand & Enhanced Features

This release marks a significant milestone with a rebrand to better reflect the tool's purpose and platform, along with major feature enhancements.

### ✨ Added

#### **CuOPT Compatibility Detection**
- **NEW:** Automatic CuOPT compatibility checking (`detect_cuopt()` method)
- **NEW:** nvJitLink version validation (critical for CuOPT 25.12+)
- **NEW:** Breaking change detection for CuOPT + Databricks ML Runtime incompatibility
- **NEW:** Detailed warning messages with actionable guidance
- **NEW:** Integration with BreakingChangesDatabase for CuOPT issues

#### **Enhanced Environment Validation Notebook**
- **NEW:** `01_cuda_environment_validation_enhanced.py` - Production-ready notebook
- **NEW:** 8-step comprehensive validation workflow
- **NEW:** Auto-detection for Classic ML Runtime & Serverless GPU Compute
- **NEW:** CuOPT compatibility analysis cell
- **NEW:** Databricks Runtime CUDA component checking
- **NEW:** CUDA 13.0 upgrade compatibility scoring
- **NEW:** Detailed breaking changes with GitHub references
- **NEW:** Migration path guidance for all issues

#### **GPU Detection Improvements**
- **NEW:** Standardized `detect_gpu_auto()` response structure
- **NEW:** Flattened GPU info for both Classic and Serverless environments
- **NEW:** Consistent `gpus` key across all detection methods
- **NEW:** Hostname field for Classic cluster GPUs
- **FIXED:** `KeyError: 'gpus'` in Classic ML Runtime clusters

#### **Documentation**
- **NEW:** `MIGRATION_GUIDE.md` - Complete migration documentation
- **NEW:** `NOTEBOOK1_VALIDATION_SUCCESS.md` - Validation report
- **NEW:** Deprecation notices on legacy notebooks
- **NEW:** Enhanced README with clear feature comparison
- **NEW:** Updated `USE_CASE_ROUTING_OPTIMIZATION.md` with CuOPT case study

### 🔄 Changed

#### **Repository Rebrand**
- **RENAMED:** Repository from `cuda-healthcheck-1.0` to `cuda-healthcheck-on-databricks`
- **RENAMED:** Package name to `cuda-healthcheck-on-databricks` (installs as `cuda_healthcheck`)
- **UPDATED:** Version from 1.0.0 to 0.5.0 (more appropriate for beta/preview status)
- **UPDATED:** All GitHub URLs across 29 files
- **UPDATED:** Repository description to highlight Databricks-specific features
- **UPDATED:** Keywords to include: `cuopt`, `routing`, `ml-runtime`

#### **Package Metadata**
- **UPDATED:** Development Status: `4 - Beta` → `3 - Alpha` (more accurate for v0.5)
- **ADDED:** Classifiers: Data Scientists, System Administrators, AI/ML
- **ADDED:** Environment classifier: `GPU :: NVIDIA CUDA`

#### **Deprecation Strategy**
- **DEPRECATED:** `notebooks/databricks_healthcheck.py` (legacy, Classic only)
- **DEPRECATED:** `notebooks/databricks_healthcheck_serverless.py` (legacy, Serverless only)
- **STRATEGY:** Gradual deprecation with backward compatibility maintained

### 🐛 Fixed

- **FIXED:** `KeyError: 'gpus'` when using `detect_gpu_auto()` on Classic ML Runtime
- **FIXED:** Inconsistent response structure between Classic and Serverless detection
- **FIXED:** Missing `gpu_count` key in Classic cluster responses
- **FIXED:** `NameError: 'cuopt_incompatible'` in summary cell when CuOPT not installed

### 📝 Documentation

- **UPDATED:** README prioritizes enhanced notebook
- **UPDATED:** All documentation URLs to new repository name
- **UPDATED:** `EXPERIMENT_CUOPT_BENCHMARK.md` to reference enhanced notebook
- **ADDED:** Migration guide for users on legacy notebooks
- **ADDED:** Validation report showing A10G testing results

### 🎯 Breaking Changes

**None!** This release maintains full backward compatibility:
- Python package name stays `cuda_healthcheck` (no import changes needed)
- Legacy notebooks continue to work (with deprecation notices)
- GitHub auto-redirects old URLs to new repository name
- All existing code continues to function

### 📊 Validation

- **Tested on:** Databricks Classic ML Runtime 16.4
- **GPU:** NVIDIA A10G (23028 MiB, Compute 8.6)
- **Python:** 3.12.3
- **CUDA:** 12.6
- **Result:** ✅ All features validated successfully

### 🙏 Credits

- **Validation & Testing:** TavnerJC (joelc@nvidia.com)
- **Use Case:** Databricks Routing Optimization with CuOPT
- **Platform:** Databricks ML Runtime & Serverless GPU Compute

---

## [1.0.0] - 2024-12-27 (Legacy Version)

### Initial Release

#### Features
- CUDA version detection (runtime, driver, NVCC)
- GPU hardware detection
- Library compatibility checking (PyTorch, TensorFlow, cuDF/RAPIDS)
- Breaking changes database
- Databricks integration
- Delta table storage
- Compatibility scoring
- Basic notebooks for Classic and Serverless

#### Documentation
- README with installation instructions
- Environment variables guide
- Databricks deployment guide
- Use case documentation

---

## Upgrade Guide

### From 1.0.0 to 0.5.0

**What Changed:**
- Repository renamed (GitHub auto-redirects old URLs)
- Version number reflects beta/preview status
- Enhanced notebook added with CuOPT detection
- Legacy notebooks deprecated (but still functional)

**Action Required:**
1. Update your installation command:
   ```bash
   # Old
   pip install git+https://github.com/TavnerJC/cuda-healthcheck-1.0.git
   
   # New
   pip install git+https://github.com/TavnerJC/cuda-healthcheck-on-databricks.git
   ```

2. (Optional) Migrate to enhanced notebook:
   - See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for details

3. Your code continues to work:
   ```python
   # No changes needed!
   from cuda_healthcheck import CUDADetector
   ```

**Benefits of Upgrading:**
- ✅ CuOPT compatibility detection
- ✅ Auto-detection (Classic & Serverless)
- ✅ Enhanced breaking changes analysis
- ✅ Production-validated features

---

## Links

- **Repository:** https://github.com/TavnerJC/cuda-healthcheck-on-databricks
- **Issues:** https://github.com/TavnerJC/cuda-healthcheck-on-databricks/issues
- **Documentation:** https://github.com/TavnerJC/cuda-healthcheck-on-databricks/blob/main/README.md
- **Migration Guide:** https://github.com/TavnerJC/cuda-healthcheck-on-databricks/blob/main/MIGRATION_GUIDE.md

---

*Note: GitHub automatically redirects old URLs (`cuda-healthcheck-1.0`) to the new repository name.*

