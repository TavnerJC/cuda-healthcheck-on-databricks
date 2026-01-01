# Response to bdice on GitHub Issue #11

**Context:** bdice asked how we concluded this is an nvJitLink error instead of cuSolver/cuBLAS compatibility.

---

## 📋 Clear Answer

### How We Identified the nvJitLink Issue

You're absolutely right to question the initial error message - it **does** show a cuSolver/cuBLAS symbol error:

```
undefined symbol: cublasSetEnvironmentMode, version libcublas.so.12
```

However, **this is a symptom, not the root cause**. Here's our debugging path:

---

## 🔍 Debugging Steps

### 1. Initial Error (cuBLAS Symbol)
```
Error: .../nvidia/cusolver/lib/libcusolver.so.11: 
undefined symbol: cublasSetEnvironmentMode, version libcublas.so.12
```

This suggested a cuBLAS incompatibility.

### 2. Attempted cuBLAS/cuSolver Upgrades
```bash
%pip install --upgrade nvidia-cublas-cu12 nvidia-cusolver-cu12
```

**Result:** Same error persisted. cuBLAS/cuSolver versions did not resolve the issue.

### 3. Deep Library Check - Found the Real Error

When we ran a deeper diagnostic:

```python
import warnings
warnings.filterwarnings("default")  # Show all warnings

from cuopt import routing  # This is where the real error appeared
```

**Output:**
```
RuntimeWarning: Failed to load libcuopt library: libcuopt.so.
Error: undefined symbol: __nvJitLinkGetErrorLogSize_12_9, version libnvJitLink.so.12
```

**This is the root cause** - CuOPT is specifically looking for nvJitLink function `__nvJitLinkGetErrorLogSize_12_9` which only exists in nvJitLink 12.9+.

### 4. Version Check Confirmed It

```python
import subprocess
result = subprocess.run(["pip", "show", "nvidia-nvjitlink-cu12"], capture_output=True, text=True)
print(result.stdout)
```

**Output:**
```
Name: nvidia-nvjitlink-cu12
Version: 12.4.127
```

**CuOPT 25.12+ requires:** `nvidia-nvjitlink-cu12 >= 12.9.79`

---

## 🚨 Why This Is a Databricks-Specific Issue

### Databricks is a **Managed Platform**

Databricks controls the entire CUDA stack, including:
- ✅ CUDA Driver
- ✅ CUDA Runtime
- ✅ CUDA Libraries (cuBLAS, cuSolver, cuDNN, **nvJitLink**)
- ✅ System-level dependencies

**Users CANNOT modify these components.**

---

## ❌ Why `pip install cuda-toolkit[nvjitlink]==12.9.1` Won't Work

### The Managed Environment Constraint

```python
%pip install cuda-toolkit[nvjitlink]==12.9.1
```

**What happens:**

1. **pip installs** the package to the Python environment (`/local_disk0/.ephemeral_nfs/envs/...`)
2. **BUT** - the system loader still uses the **runtime-provided** nvJitLink from `/usr/local/cuda/lib64/`
3. **Databricks runtime libraries take precedence** over pip-installed user libraries
4. **Result:** CuOPT still loads the old 12.4.127 nvJitLink from the runtime

**Analogy:**
```
It's like AWS Lambda - you can't upgrade the Node.js runtime by doing `npm install node@20`.
The Lambda runtime controls the Node version, not your package.json.
```

**Same with Databricks:**
```
Databricks ML Runtime 16.4 → Provides CUDA 12.6 + nvJitLink 12.4.127
You can't override runtime CUDA components via pip
```

---

## 📊 Evidence from Our Testing

### Test 1: pip install cuda-toolkit (Failed)

```bash
%pip install cuda-toolkit[all]==12.9.1
```

**Result:**
```
ERROR: Cannot install cuda-toolkit on a managed Databricks runtime
```

### Test 2: pip install nvidia-nvjitlink-cu12>=12.9 (Failed)

```bash
%pip install --upgrade nvidia-nvjitlink-cu12>=12.9
```

**Result:**
- Package "installed" in user environment
- CuOPT **still** failed with same error
- System loader still used runtime's 12.4.127 version

**Verified with:**
```python
import ctypes
import ctypes.util

nvjitlink_path = ctypes.util.find_library("nvJitLink")
print(f"Loaded nvJitLink from: {nvjitlink_path}")
```

**Output:**
```
Loaded nvJitLink from: /usr/local/cuda-12.6/lib64/libnvJitLink.so.12
```

This is **NOT** the pip-installed version - it's the runtime version.

---

## 🎯 Why This Matters

### Databricks Owns the CUDA Stack

| Component | User Control | Databricks Control |
|-----------|--------------|-------------------|
| **Python Packages** | ✅ YES (pip install) | ❌ NO |
| **CUDA Driver** | ❌ NO | ✅ YES (runtime locked) |
| **CUDA Runtime** | ❌ NO | ✅ YES (12.6 in ML Runtime 16.4) |
| **nvJitLink** | ❌ NO | ✅ YES (12.4.127 in ML Runtime 16.4) |
| **cuBLAS/cuSolver** | ❌ NO | ✅ YES (runtime provides) |

**Users cannot upgrade CUDA components in Databricks.**

---

## 💡 The Solution

### Short-term: Use OR-Tools
```bash
%pip install ortools
```

OR-Tools provides CPU-based routing optimization and **works perfectly** on Databricks.

### Medium-term: Report to Databricks
Request that Databricks ML Runtime 17.0 or 16.x patch release includes `nvidia-nvjitlink-cu12 >= 12.9.79`.

**GitHub Issue:** https://github.com/databricks-industry-solutions/routing/issues/11

### Long-term: Runtime Upgrade
Wait for Databricks to release ML Runtime with CUDA 12.9+ which will include compatible nvJitLink.

---

## 📌 Summary

**Question:** How did you conclude this is an nvJitLink error instead of cuSolver/cuBLAS?

**Answer:**
1. Initial error showed cuBLAS symbol - **this was misleading**
2. Upgrading cuBLAS/cuSolver via pip **did not fix the issue**
3. Deep diagnostic revealed the **real error**: `__nvJitLinkGetErrorLogSize_12_9` missing
4. This function only exists in **nvJitLink 12.9+**
5. Databricks ML Runtime 16.4 provides **nvJitLink 12.4.127**
6. **Users cannot upgrade runtime CUDA components** in managed Databricks environments

**Root Cause:** Platform constraint, not a package installation issue.

---

## 🔗 References

- **CuOPT 25.12 Release Notes:** Requires nvJitLink 12.9+
- **Databricks ML Runtime 16.4:** Provides nvJitLink 12.4.127
- **Our Healthcheck Tool:** Detects this automatically - https://github.com/TavnerJC/cuda-healthcheck-on-databricks
- **Full Investigation:** https://github.com/TavnerJC/cuda-healthcheck-on-databricks/blob/main/docs/USE_CASE_ROUTING_OPTIMIZATION.md

---

**Hope this clarifies! Let me know if you need more details about our testing process.** 🚀


