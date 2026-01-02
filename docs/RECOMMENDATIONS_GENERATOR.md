# User-Friendly Recommendation Generator

## Overview

The recommendation generator converts technical error messages and blockers into clear, actionable recommendations that users can understand and follow. It provides context-aware guidance based on Databricks runtime version and offers multiple fix options when available.

---

## Functions

### `generate_recommendations()`

Converts technical blockers into Markdown-formatted user-friendly recommendations.

**Signature:**
```python
def generate_recommendations(
    blockers: List[Dict[str, Any]],
    runtime_version: Optional[float] = None
) -> str:
```

**Parameters:**
- `blockers`: List of blocker dictionaries with keys:
  - `issue` or `message`: Technical description
  - `root_cause`: Category of issue (optional)
  - `fix_command`: Technical fix command (optional)
  - `fix_commands` or `fix_options`: List of fixes (optional)
- `runtime_version`: Databricks runtime version for context-aware recommendations

**Returns:** Markdown-formatted text with user-friendly recommendations

---

### `format_recommendations_for_notebook()`

Similar to `generate_recommendations()` but optimized for Databricks notebook display.

**Signature:**
```python
def format_recommendations_for_notebook(
    blockers: List[Dict[str, Any]],
    runtime_version: Optional[float] = None,
    show_technical_details: bool = False
) -> str:
```

**Parameters:**
- `blockers`: List of blocker dictionaries
- `runtime_version`: Databricks runtime version
- `show_technical_details`: Whether to include technical error details

**Returns:** Formatted text optimized for notebook display

---

## Supported Root Causes

The generator understands 9 root cause categories and provides specific guidance for each:

| Root Cause | User-Friendly Title | Key Message |
|------------|---------------------|-------------|
| `driver_too_old` | GPU Driver Too Old | Your driver is too old for this PyTorch version |
| `torch_not_installed` | PyTorch Not Installed | PyTorch is not installed |
| `torch_no_cuda_support` | PyTorch Missing CUDA Support | PyTorch built for CPU-only |
| `cuda_libraries_missing` | CUDA Libraries Missing | CUDA libraries are missing or incompatible |
| `no_gpu_device` | No GPU Detected | No GPU was detected on your cluster |
| `driver_version_mismatch` | Driver Version Incompatible | Driver outside expected runtime range |
| `nvjitlink_mismatch` | CUDA Library Version Mismatch | cuBLAS and nvJitLink versions don't match |
| `mixed_cuda_versions` | Mixed CUDA Versions Detected | Both CUDA 11 and CUDA 12 packages installed |
| `torch_branch_incompatible` | PyTorch CUDA Branch Incompatible | PyTorch CUDA branch doesn't match runtime |

---

## Usage Examples

### Example 1: Driver Too Old on Runtime 14.3

**Input:**
```python
from cuda_healthcheck.utils import generate_recommendations

blockers = [
    {
        "issue": "Driver 535 (too old) for cu124 (requires 550+)",
        "root_cause": "driver_too_old",
        "fix_options": [
            "Downgrade PyTorch to cu120: pip install torch --index-url https://download.pytorch.org/whl/cu120",
            "Upgrade Databricks runtime to 15.2+ (supports CUDA 12.4 and Driver 550)"
        ]
    }
]

recommendations = generate_recommendations(blockers, runtime_version=14.3)
print(recommendations)
```

**Output:**
```markdown
# 🚨 Action Required: Critical Issues Detected

Your environment has issues that will prevent GPU workloads from running. Follow the recommendations below to fix them.

---

## Issue #1: GPU Driver Too Old

**What's wrong:** Your NVIDIA GPU driver is too old for the version of PyTorch you're using.

**Why it matters:** PyTorch cu124 requires Driver 550+, but you have an older driver. This prevents CUDA from working.

**Platform constraint:** ⚠️ **Runtime 14.3 has an IMMUTABLE Driver 535.** You CANNOT upgrade the driver. Your only options are: (1) downgrade PyTorch to cu120, or (2) upgrade to Runtime 15.2+

**Technical details:** Driver 535 (too old) for cu124 (requires 550+)

### 🔧 How to Fix:

1. **Downgrade PyTorch to cu120:**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu120
   ```

2. **Upgrade Databricks runtime to 15.2+ (supports CUDA 12.4 and Driver 550)**

---

## 💡 General Tips

- Always restart Python after installing packages: `dbutils.library.restartPython()`
- Use `--no-cache-dir` to ensure fresh package downloads
- Check your Databricks runtime version: Some drivers are immutable
- For persistent issues, contact Databricks support or report to our GitHub
```

---

### Example 2: PyTorch Not Installed

**Input:**
```python
blockers = [
    {
        "issue": "PyTorch is required but not installed",
        "root_cause": "torch_not_installed",
        "fix_command": "pip install torch --index-url https://download.pytorch.org/whl/cu121"
    }
]

recommendations = generate_recommendations(blockers)
print(recommendations)
```

**Output:**
```markdown
# 🚨 Action Required: Critical Issues Detected

...

## Issue #1: PyTorch Not Installed

**What's wrong:** PyTorch is not installed, but you're trying to use features that need it (like local AI model inference).

**Why it matters:** Without PyTorch, you can't run GPU-accelerated AI models locally.

**Note:** If you only need cloud-based inference (API calls), you can ignore this and disable local inference features.

**Technical details:** PyTorch is required but not installed

### 🔧 How to Fix:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---
```

---

### Example 3: Mixed CUDA 11/12 Packages

**Input:**
```python
blockers = [
    {
        "issue": "Both CUDA 11 and CUDA 12 packages detected",
        "root_cause": "mixed_cuda_versions",
        "fix_options": [
            "pip uninstall torch",
            "pip cache purge",
            "pip install torch --index-url https://download.pytorch.org/whl/cu124"
        ]
    }
]

recommendations = generate_recommendations(blockers)
print(recommendations)
```

**Output:**
```markdown
# 🚨 Action Required: Critical Issues Detected

...

## Issue #1: Mixed CUDA Versions Detected

**What's wrong:** You have both CUDA 11 and CUDA 12 packages installed at the same time. This creates library conflicts.

**Why it matters:** This causes `LD_LIBRARY_PATH` conflicts, segfaults, and symbol resolution errors. Your GPU code will be unstable.

**Technical details:** Both CUDA 11 and CUDA 12 packages detected

### 🔧 How to Fix:

1. **Uninstall command:**
   ```bash
   pip uninstall torch
   ```

2. **Run this command:**
   ```bash
   pip cache purge
   ```

3. **Install command:**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```

---
```

---

### Example 4: Notebook-Optimized Format

**Input:**
```python
from cuda_healthcheck.utils import format_recommendations_for_notebook

blockers = [
    {
        "issue": "Driver 535 too old for cu124",
        "root_cause": "driver_too_old",
        "fix_options": [
            "Downgrade to cu120",
            "Upgrade runtime to 15.2+"
        ]
    }
]

output = format_recommendations_for_notebook(blockers, runtime_version=14.3)
print(output)
```

**Output:**
```
================================================================================
🚨 ACTION REQUIRED: CRITICAL ISSUES DETECTED
================================================================================

────────────────────────────────────────────────────────────────────────────────
Issue #1: GPU Driver Too Old
────────────────────────────────────────────────────────────────────────────────

❌ Your GPU driver is too old for this PyTorch version.

⚠️  Runtime 14.3 has IMMUTABLE Driver 535 - you CANNOT upgrade it.

🔧 How to Fix:

   1. Downgrade to cu120

   2. Upgrade runtime to 15.2+

================================================================================

💡 After fixing, restart Python: dbutils.library.restartPython()
```

---

## Runtime-Specific Context

The generator provides **context-aware guidance** based on Databricks runtime:

### Runtime 14.3 (Immutable Driver 535)
```
⚠️ **Runtime 14.3 has an IMMUTABLE Driver 535.** You CANNOT upgrade the driver.
Your only options are: (1) downgrade PyTorch to cu120, or (2) upgrade to Runtime 15.2+
```

### Runtime 15.1 (Immutable Driver 550)
```
⚠️ **Runtime 15.1 has an IMMUTABLE Driver 550.** You CANNOT upgrade the driver.
```

### Runtime 15.2+ (Flexible)
```
✅ Runtime 15.2 supports CUDA 12.4 with Driver 550. You can use PyTorch cu124.
```

---

## Integration with CUDA Diagnostics

The recommendation generator integrates seamlessly with the CUDA diagnostics:

```python
from cuda_healthcheck.nemo import detect_enabled_features, diagnose_cuda_availability
from cuda_healthcheck.utils import format_recommendations_for_notebook

# Detect features
features = detect_enabled_features()

# Run diagnostics
cuda_diag = diagnose_cuda_availability(
    features,
    runtime_version=14.3,
    torch_cuda_branch="cu124",
    driver_version=535
)

# Convert to user-friendly recommendations
if cuda_diag['severity'] == 'BLOCKER':
    blockers = [
        {
            "issue": cuda_diag['diagnostics']['issue'],
            "root_cause": cuda_diag['diagnostics']['root_cause'],
            "fix_options": cuda_diag['fix_options']
        }
    ]
    
    print(format_recommendations_for_notebook(blockers, runtime_version=14.3))
```

---

## Testing

Run the test suite:
```bash
pytest tests/utils/test_recommendations.py -v
```

**Test Coverage: 20 tests**

1. ✅ No blockers
2. ✅ Driver too old on Runtime 14.3
3. ✅ PyTorch not installed
4. ✅ Mixed CUDA versions
5. ✅ nvJitLink mismatch
6. ✅ No GPU device
7. ✅ PyTorch without CUDA support
8. ✅ PyTorch branch incompatible
9. ✅ Multiple blockers
10. ✅ Unknown root cause
11. ✅ Feature-specific blocker
12. ✅ Includes general tips
13. ✅ Notebook format - no blockers
14. ✅ Notebook format - driver too old
15. ✅ Includes restart reminder
16. ✅ Technical details hidden by default
17. ✅ Technical details shown when requested
18. ✅ Multiple blockers numbered
19. ✅ Runtime 15.2+ message
20. ✅ Clean fix options

---

## API Reference

### Root Cause to Title Mapping

```python
{
    "driver_too_old": "GPU Driver Too Old",
    "torch_not_installed": "PyTorch Not Installed",
    "torch_no_cuda_support": "PyTorch Missing CUDA Support",
    "cuda_libraries_missing": "CUDA Libraries Missing",
    "no_gpu_device": "No GPU Detected",
    "driver_version_mismatch": "Driver Version Incompatible",
    "nvjitlink_mismatch": "CUDA Library Version Mismatch",
    "mixed_cuda_versions": "Mixed CUDA 11 and CUDA 12 Packages",
    "torch_branch_incompatible": "PyTorch CUDA Branch Incompatible",
}
```

### Explanation Templates

Each root cause has a specific explanation template that includes:
- **What's wrong:** Plain English description
- **Why it matters:** Impact on user's workload
- **Platform constraint:** Runtime-specific limitations (if applicable)
- **Technical details:** Original error message

---

## Benefits

✅ **Clear Communication** - Converts technical jargon to plain English  
✅ **Actionable Guidance** - Provides specific commands users can run  
✅ **Context-Aware** - Understands platform constraints like immutable drivers  
✅ **Multiple Solutions** - Offers alternatives when available  
✅ **Consistent Formatting** - Markdown for documentation, plain text for notebooks  
✅ **Educational** - Explains WHY issues occur, not just HOW to fix  

---

## See Also

- [CUDA Diagnostics](./CUDA_DIAGNOSTICS.md)
- [Databricks Runtime Detection](./DATABRICKS_RUNTIME_DETECTION.md)
- [Driver Version Mapping](./DRIVER_VERSION_MAPPING.md)
- [Enhanced Notebook](../notebooks/01_cuda_environment_validation_enhanced.py)

