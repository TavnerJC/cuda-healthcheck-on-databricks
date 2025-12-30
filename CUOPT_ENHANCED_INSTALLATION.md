# 🔧 Enhanced CuOPT Installation for Databricks ML Runtime

## Issue
Even on Classic ML Runtime with GPUs, CuOPT fails with:
```
libcuopt.so: cannot open shared object file: No such file or directory
```

## Root Cause
Missing CUDA runtime libraries that CuOPT depends on.

---

## ✅ Solution: Complete Installation with Dependencies

### **Cell 1: Install All Dependencies**

```python
# COMMAND ----------
# Install CuOPT with all required CUDA dependencies
%pip install --upgrade pip
%pip install \
  --extra-index-url=https://pypi.nvidia.com \
  nvidia-cuda-runtime-cu12==12.6.77 \
  nvidia-cudnn-cu12==9.5.1.17 \
  nvidia-cublas-cu12==12.6.3.3 \
  nvidia-cusolver-cu12==11.7.1.2 \
  nvidia-cusparse-cu12==12.5.4.2 \
  cuopt-server-cu12==26.02.00 \
  cuopt-sh-client==26.02.00 \
  pandas==2.2.3 \
  numpy==1.26.4 \
  matplotlib==3.9.2 \
  scipy==1.14.1

dbutils.library.restartPython()
```

### **Key Changes:**
1. ✅ **Pinned versions** (addresses your warning)
2. ✅ **CUDA runtime libraries** explicitly installed
3. ✅ **cuBLAS, cuSOLVER, cuSPARSE** (CuOPT dependencies)
4. ✅ **cuDNN** (for neural network operations)

---

## 🔍 Cell 2: Verify Installation

```python
# COMMAND ----------
# Verify CUDA libraries are available
import sys
import os

print("=" * 80)
print("🔍 CUDA LIBRARY VERIFICATION")
print("=" * 80)

# Check if nvidia packages are installed
try:
    import nvidia.cuda_runtime
    print("✅ nvidia-cuda-runtime: installed")
    print(f"   Location: {nvidia.cuda_runtime.__file__}")
except ImportError as e:
    print(f"❌ nvidia-cuda-runtime: NOT FOUND - {e}")

try:
    import nvidia.cublas
    print("✅ nvidia-cublas: installed")
except ImportError as e:
    print(f"❌ nvidia-cublas: NOT FOUND - {e}")

try:
    import nvidia.cusolver
    print("✅ nvidia-cusolver: installed")
except ImportError as e:
    print(f"❌ nvidia-cusolver: NOT FOUND - {e}")

# Check for libcuopt.so
print("\n" + "=" * 80)
print("🔍 SEARCHING FOR libcuopt.so")
print("=" * 80)

# Get Python site-packages location
import site
site_packages = site.getsitepackages()
print(f"Site packages: {site_packages}")

# Search for cuopt files
import subprocess
result = subprocess.run(
    ["find", site_packages[0], "-name", "*cuopt*", "-type", "f"],
    capture_output=True,
    text=True,
    timeout=30
)

if result.stdout:
    print("Found CuOPT files:")
    for line in result.stdout.strip().split('\n')[:20]:  # Show first 20
        print(f"  {line}")
else:
    print("⚠️  No CuOPT files found in site-packages")

# Try to import cuopt
print("\n" + "=" * 80)
print("🔍 TESTING CUOPT IMPORT")
print("=" * 80)

try:
    from cuopt import routing
    print("✅ CuOPT imported successfully!")
    print(f"   Module: {routing}")
except ImportError as e:
    print(f"❌ CuOPT import failed: {e}")
    print(f"\nError details:")
    print(f"  Type: {type(e).__name__}")
    print(f"  Message: {str(e)}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

print("=" * 80)
```

---

## 🔄 Alternative: Use Conda Installation

If pip still fails, try conda (more reliable for CUDA packages):

```python
# COMMAND ----------
# Alternative: Install via Conda
%conda install -c rapidsai -c conda-forge -c nvidia \
  cuopt-server=26.02.* \
  cuopt-sh-client=26.02.* \
  cuda-version=12.6 \
  -y

dbutils.library.restartPython()
```

---

## 🎯 Expected Outcomes

### **If Successful:**
```
✅ nvidia-cuda-runtime: installed
✅ nvidia-cublas: installed
✅ nvidia-cusolver: installed
✅ CuOPT imported successfully!
```

### **If Still Failing:**
The verification cell will show which specific library is missing, helping us debug further.

---

## 📊 Databricks ML Runtime Requirements

For reference, your cluster:
- **Runtime**: 16.4 ML (Spark 3.5.2, Scala 2.13)
- **Driver**: i3.xlarge (30.5 GB, 4 Cores) - CPU only
- **Worker**: g5.xlarge (64 GB, 16 Cores) - **NVIDIA A10G GPU** ✅

**Note:** CuOPT will use GPU on workers, not the driver node.

---

## 🚨 If This Still Fails

If the enhanced installation still doesn't work, it suggests:

1. **Databricks ML Runtime issue** - CuOPT may require specific runtime versions
2. **System library incompatibility** - ML Runtime might be missing base libraries
3. **CuOPT packaging issue** - The pip package might not be fully self-contained

**Fallback:** Use **OR-Tools** (proven working) from `NOTEBOOK2_ORTOOLS_WORKING.md`

