# Databricks Runtime Detection

## Overview

The `runtime_detector` module provides robust detection of Databricks runtime versions using multiple fallback methods. This is critical for the CUDA Healthcheck Tool to provide accurate compatibility warnings and guidance.

## Key Features

✅ **Multi-Method Detection**: 4 fallback detection methods  
✅ **CUDA Version Mapping**: Automatic mapping of runtime → CUDA version  
✅ **Serverless Support**: Detects both Classic ML Runtime and Serverless GPU Compute  
✅ **Error Handling**: Graceful degradation when detection methods fail  
✅ **Comprehensive Testing**: 36 unit tests with 92% code coverage

---

## Quick Start

### Basic Usage

```python
from cuda_healthcheck.databricks import detect_databricks_runtime

# Detect runtime
result = detect_databricks_runtime()

print(f"Runtime: {result['runtime_version']}")
print(f"CUDA Version: {result['cuda_version']}")
print(f"Is Databricks: {result['is_databricks']}")
print(f"Is ML Runtime: {result['is_ml_runtime']}")
print(f"Is GPU Runtime: {result['is_gpu_runtime']}")
```

### Human-Readable Summary

```python
from cuda_healthcheck.databricks import get_runtime_info_summary

# Get formatted summary
summary = get_runtime_info_summary()
print(summary)
```

**Output:**
```
Databricks ML Runtime 16.4 (GPU, CUDA 12.6)
Detected via: env_var
```

### Simple Check

```python
from cuda_healthcheck.databricks import is_databricks_environment

if is_databricks_environment():
    print("Running in Databricks!")
else:
    print("Not in Databricks")
```

---

## Detection Methods

The module uses **4 fallback methods** (in order of priority):

### 1. Environment Variable (Primary)

**Method:** Check `DATABRICKS_RUNTIME_VERSION` environment variable  
**Reliability:** ⭐⭐⭐⭐⭐ (Most reliable)  
**Example Value:** `"16.4.x-gpu-ml-scala2.12"`

```python
import os
runtime = os.getenv("DATABRICKS_RUNTIME_VERSION")
# Returns: "16.4.x-gpu-ml-scala2.12"
```

### 2. Environment File (Fallback)

**Method:** Parse `/databricks/environment.yml`  
**Reliability:** ⭐⭐⭐⭐ (Reliable if file exists)  
**Format:** YAML file with runtime metadata

```yaml
# /databricks/environment.yml
databricks:
  runtime_version: "16.4.x-gpu-ml-scala2.12"
```

### 3. Workspace Indicator (Basic Detection)

**Method:** Check for `/Workspace` directory  
**Reliability:** ⭐⭐⭐ (Basic Databricks detection only)  
**Limitation:** Confirms Databricks but no version info

```python
from pathlib import Path
if Path("/Workspace").exists():
    # We're in Databricks, but don't know the version
    pass
```

### 4. IPython Config (Notebook Context)

**Method:** Check IPython config for `DATABRICKS` marker  
**Reliability:** ⭐⭐ (Works in notebooks only)  
**Limitation:** Only works when IPython is available

```python
import IPython
ipython = IPython.get_ipython()
if "DATABRICKS" in str(ipython.config):
    # Running in Databricks notebook
    pass
```

---

## Return Value Structure

The `detect_databricks_runtime()` function returns a dictionary with the following keys:

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `runtime_version` | `float` or `None` | Numeric runtime version | `16.4` |
| `runtime_version_string` | `str` or `None` | Full version string | `"16.4.x-gpu-ml-scala2.12"` |
| `is_databricks` | `bool` | Whether Databricks was detected | `True` |
| `is_ml_runtime` | `bool` | Whether this is ML runtime | `True` |
| `is_gpu_runtime` | `bool` | Whether GPU is available | `True` |
| `is_serverless` | `bool` | Whether Serverless Compute | `False` |
| `cuda_version` | `str` or `None` | Expected CUDA version | `"12.6"` |
| `detection_method` | `str` | How runtime was detected | `"env_var"` |

---

## Example Outputs

### ML Runtime 14.3

```python
{
    "runtime_version": 14.3,
    "runtime_version_string": "14.3.x-gpu-ml-scala2.12",
    "is_databricks": True,
    "is_ml_runtime": True,
    "is_gpu_runtime": True,
    "is_serverless": False,
    "cuda_version": "12.2",
    "detection_method": "env_var"
}
```

### ML Runtime 15.2

```python
{
    "runtime_version": 15.2,
    "runtime_version_string": "15.2.x-gpu-ml-scala2.12",
    "is_databricks": True,
    "is_ml_runtime": True,
    "is_gpu_runtime": True,
    "is_serverless": False,
    "cuda_version": "12.4",
    "detection_method": "env_var"
}
```

### ML Runtime 16.4

```python
{
    "runtime_version": 16.4,
    "runtime_version_string": "16.4.x-gpu-ml-scala2.12",
    "is_databricks": True,
    "is_ml_runtime": True,
    "is_gpu_runtime": True,
    "is_serverless": False,
    "cuda_version": "12.6",
    "detection_method": "env_var"
}
```

### Serverless GPU Compute v4

```python
{
    "runtime_version": None,
    "runtime_version_string": "serverless-gpu-v4",
    "is_databricks": True,
    "is_ml_runtime": False,
    "is_gpu_runtime": True,
    "is_serverless": True,
    "cuda_version": "12.6",
    "detection_method": "env_var"
}
```

### Non-Databricks Environment

```python
{
    "runtime_version": None,
    "runtime_version_string": None,
    "is_databricks": False,
    "is_ml_runtime": False,
    "is_gpu_runtime": False,
    "is_serverless": False,
    "cuda_version": None,
    "detection_method": "unknown"
}
```

---

## CUDA Version Mapping

The module automatically maps Databricks runtime versions to their expected CUDA versions:

| Runtime Version | CUDA Version | Release Date |
|----------------|--------------|--------------|
| **16.4** | **12.6** | Dec 2025 |
| **16.0** | **12.6** | Oct 2025 |
| **15.4** | **12.4** | Sep 2025 |
| **15.3** | **12.4** | Jul 2025 |
| **15.2** | **12.4** | Jun 2025 |
| **15.1** | **12.4** | May 2025 |
| **15.0** | **12.4** | Apr 2025 |
| **14.3** | **12.2** | Dec 2024 |
| **14.2** | **12.2** | Oct 2024 |
| **14.1** | **12.2** | Sep 2024 |
| **14.0** | **12.2** | Jul 2024 |
| **13.3** | **11.8** | Apr 2024 |

**Source:** [Databricks ML Runtime Release Notes](https://docs.databricks.com/release-notes/runtime/index.html)

---

## Use Cases

### 1. CuOPT Compatibility Check

```python
from cuda_healthcheck.databricks import detect_databricks_runtime

result = detect_databricks_runtime()

# CuOPT 25.12+ requires nvJitLink 12.9+
# But Databricks ML Runtime 16.4 provides nvJitLink 12.4.127
if result["runtime_version"] == 16.4:
    print("⚠️  WARNING: CuOPT 25.12+ incompatible with ML Runtime 16.4")
    print("   Reason: nvJitLink version mismatch (12.4.127 vs required 12.9+)")
    print("   Solution: Use OR-Tools or wait for ML Runtime 17.0+")
```

### 2. CUDA Version Validation

```python
from cuda_healthcheck.databricks import detect_databricks_runtime

result = detect_databricks_runtime()

if result["cuda_version"]:
    print(f"Expected CUDA version for this runtime: {result['cuda_version']}")
    
    # Compare with detected CUDA
    from cuda_healthcheck import CUDADetector
    detector = CUDADetector()
    env = detector.detect_environment()
    
    if env.cuda_runtime_version != result["cuda_version"]:
        print(f"⚠️  Mismatch detected!")
        print(f"   Expected: {result['cuda_version']}")
        print(f"   Detected: {env.cuda_runtime_version}")
```

### 3. Runtime-Specific Guidance

```python
from cuda_healthcheck.databricks import detect_databricks_runtime

result = detect_databricks_runtime()

if result["is_serverless"]:
    print("📊 Serverless GPU Compute detected")
    print("   • Limited package installation")
    print("   • No direct cluster access")
    print("   • Use pre-installed libraries when possible")
elif result["is_ml_runtime"]:
    print(f"🔬 ML Runtime {result['runtime_version']} detected")
    print(f"   • CUDA {result['cuda_version']} available")
    print("   • Full package installation supported")
else:
    print("💻 Standard Databricks runtime detected")
```

### 4. Integration with Breaking Changes Database

```python
from cuda_healthcheck.databricks import detect_databricks_runtime
from cuda_healthcheck.data import BreakingChangesDatabase

# Detect runtime
runtime_info = detect_databricks_runtime()

# Check for breaking changes
db = BreakingChangesDatabase()
changes = db.get_changes_by_cuda_version(runtime_info["cuda_version"])

print(f"Runtime: {runtime_info['runtime_version']}")
print(f"CUDA: {runtime_info['cuda_version']}")
print(f"Breaking changes: {len(changes)}")

for change in changes:
    print(f"  • {change.title} ({change.severity})")
```

---

## API Reference

### `detect_databricks_runtime() -> Dict[str, Any]`

**Primary function** for detecting Databricks runtime version.

**Returns:** Dictionary with runtime information (see structure above)

**Example:**
```python
result = detect_databricks_runtime()
print(f"Runtime: {result['runtime_version']}")
print(f"CUDA: {result['cuda_version']}")
```

---

### `get_runtime_info_summary() -> str`

Get a human-readable summary of the detected runtime.

**Returns:** Formatted string

**Example:**
```python
summary = get_runtime_info_summary()
print(summary)
# Output: "Databricks ML Runtime 16.4 (GPU, CUDA 12.6)\nDetected via: env_var"
```

---

### `is_databricks_environment() -> bool`

Simple check if running in Databricks.

**Returns:** `True` if in Databricks, `False` otherwise

**Example:**
```python
if is_databricks_environment():
    print("We're in Databricks!")
```

---

## Testing

The module includes comprehensive unit tests with **92% code coverage**.

### Run Tests

```bash
# Run all tests
pytest tests/databricks/test_runtime_detector.py -v

# Run with coverage
pytest tests/databricks/test_runtime_detector.py -v \
    --cov=cuda_healthcheck.databricks.runtime_detector \
    --cov-report=term-missing
```

### Test Coverage

```
Name                                              Stmts   Miss  Cover
---------------------------------------------------------------------
cuda_healthcheck/databricks/runtime_detector.py     134     11    92%
---------------------------------------------------------------------
TOTAL                                               134     11    92%

36 tests passed
```

---

## Error Handling

The module handles errors gracefully:

### Missing Environment Variable

```python
# When DATABRICKS_RUNTIME_VERSION is not set
result = detect_databricks_runtime()
# Falls back to other detection methods
# If all fail: {"is_databricks": False, "detection_method": "unknown"}
```

### Corrupted Environment File

```python
# When /databricks/environment.yml is invalid YAML
result = detect_databricks_runtime()
# Logs warning, returns partial result
# {"is_databricks": True, "runtime_version": None, "detection_method": "file"}
```

### IPython Not Available

```python
# When IPython import fails (e.g., not in a notebook)
result = detect_databricks_runtime()
# Silently falls back to next method
```

---

## Logging

The module uses the CUDA Healthcheck Tool's logging system:

```python
from cuda_healthcheck.databricks import detect_databricks_runtime
import logging

# Set log level
logging.getLogger("cuda_healthcheck").setLevel(logging.DEBUG)

# Now detection will log details
result = detect_databricks_runtime()
```

**Log Output:**
```
INFO: Databricks runtime detected via env_var: 16.4.x-gpu-ml-scala2.12
```

---

## Integration with CUDA Healthcheck Tool

The runtime detector is automatically used by the main healthcheck notebook:

```python
# In Databricks notebook
from cuda_healthcheck.databricks import (
    detect_databricks_runtime,
    detect_gpu_auto,
)

# Detect runtime and GPU
runtime_info = detect_databricks_runtime()
gpu_info = detect_gpu_auto()

print(f"Runtime: {runtime_info['runtime_version']}")
print(f"CUDA: {runtime_info['cuda_version']}")
print(f"GPUs: {len(gpu_info['gpus'])}")

# Automatically integrated in notebooks/01_cuda_environment_validation_enhanced.py
```

---

## Troubleshooting

### Issue: Runtime version not detected

**Symptom:** `result["is_databricks"] = False` when running in Databricks

**Solutions:**
1. Check `DATABRICKS_RUNTIME_VERSION` environment variable:
   ```python
   import os
   print(os.getenv("DATABRICKS_RUNTIME_VERSION"))
   ```
2. Verify `/databricks/environment.yml` exists:
   ```python
   from pathlib import Path
   print(Path("/databricks/environment.yml").exists())
   ```
3. Check for `/Workspace` directory:
   ```python
   print(Path("/Workspace").exists())
   ```

### Issue: CUDA version mapping returns None

**Symptom:** `result["cuda_version"] = None` for known runtime

**Solution:** This might be a new runtime version. Check the [Databricks Release Notes](https://docs.databricks.com/release-notes/runtime/index.html) and update the CUDA mapping in `runtime_detector.py`.

---

## References

- **Databricks ML Runtime Release Notes**: https://docs.databricks.com/release-notes/runtime/index.html
- **CUDA Healthcheck Tool**: https://github.com/TavnerJC/cuda-healthcheck-on-databricks
- **CuOPT nvJitLink Issue**: https://github.com/databricks-industry-solutions/routing/issues/11

---

## Changelog

### v0.5.0 (Jan 2026)
- ✅ Initial release of runtime detector module
- ✅ 4 fallback detection methods
- ✅ CUDA version mapping for runtimes 13.x through 16.x
- ✅ Serverless GPU Compute detection
- ✅ 36 unit tests with 92% coverage
- ✅ Integration with enhanced Notebook 1

---

**For more information, see the [main CUDA Healthcheck Tool documentation](../README.md).**

