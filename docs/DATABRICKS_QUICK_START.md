# 🚀 Databricks Quick Start - Visual Guide

**Goal:** Get CUDA Healthcheck running in Databricks in under 5 minutes!

---

## 📋 Step-by-Step Visual Checklist

### ✅ Step 1: Import the Notebook

**Choose your runtime:**

| Runtime Type | Notebook to Import |
|--------------|-------------------|
| **Classic ML Runtime** (multi-worker clusters) | `databricks_healthcheck.py` |
| **Serverless GPU Compute** (single-user) | `databricks_healthcheck_serverless.py` |
| **Not sure?** | Either one works - auto-detects! |

**Import URL:**
```
https://raw.githubusercontent.com/TavnerJC/cuda-healthcheck-on-databricks/main/notebooks/databricks_healthcheck_serverless.py
```

**Steps:**
1. Go to **Workspace** → **Import**
2. Select **URL**
3. Paste URL above
4. Click **Import**

---

### ✅ Step 2: Attach to GPU Compute

**Classic Cluster:**
- Databricks Runtime 13.3 LTS ML or higher
- GPU instance (g5.4xlarge, p3.2xlarge, etc.)

**Serverless:**
- Just select "Serverless GPU Compute" in the compute dropdown

---

### ✅ Step 3: Run Cell 1 (Install Package)

**What you'll run:**
```python
%pip install git+https://github.com/TavnerJC/cuda-healthcheck-on-databricks.git
```

**What you'll see:**
```
Building wheels for collected packages: cuda-healthcheck
  Building wheel for cuda-healthcheck (pyproject.toml): started
  Building wheel for cuda-healthcheck (pyproject.toml): finished with status 'done'
Successfully built cuda-healthcheck
Installing collected packages: cuda-healthcheck
Successfully installed cuda-healthcheck-on-databricks.0
```

**Then you'll see this RED NOTE:**
```
⚠️ Note: you may need to restart the kernel using %restart_python 
or dbutils.library.restartPython() to use updated packages.
```

### 🎉 **This is GOOD NEWS!**

| What it means | What to do |
|---------------|-----------|
| ✅ Package installed successfully | Run Cell 2 next |
| ✅ Python needs restart to recognize it | Don't panic! |
| ✅ Completely normal behavior | Keep going! |

---

### ✅ Step 4: Run Cell 2 (Restart Python)

**What you'll run:**
```python
dbutils.library.restartPython()
```

**What happens:**
1. ⏸️ Notebook execution pauses (~10 seconds)
2. 🔄 Python interpreter restarts
3. 🧹 All variables cleared (expected!)
4. ✅ Package now ready to use

**Important:**
- ⚠️ **Do NOT re-run Cell 1** after restart
- ✅ Just continue to Cell 3

---

### ✅ Step 5: Run Cell 3+ (Use the Package)

**Now you can import:**
```python
from cuda_healthcheck.databricks import detect_gpu_auto
from cuda_healthcheck import CUDADetector

# This will work now!
gpu_info = detect_gpu_auto()
```

**Expected output:**
```
================================================================================
🌟 DATABRICKS ENVIRONMENT DETECTION
================================================================================
📍 Environment: Serverless GPU Compute
   • Single-user execution model
   • Direct GPU access

================================================================================
🎮 GPU DETECTION
================================================================================
✅ Detection Method: direct
✅ Found 1 GPU(s)
   GPU 0: NVIDIA A10G
      Driver: 535.161.07
      Memory: 22731 MiB
      Compute Capability: 8.6
```

---

## 🎯 Complete Flow (At a Glance)

```
┌─────────────────────────────────────────────────────────────┐
│ Cell 1: %pip install git+https://...                        │
│ Output: "Successfully installed cuda-healthcheck-on-databricks.0"     │
│ Warning: "⚠️ Note: you may need to restart..." ← NORMAL!    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Cell 2: dbutils.library.restartPython()                     │
│ Effect: Python restarts, variables cleared                  │
│ Duration: ~10 seconds                                        │
│ Note: Do NOT re-run Cell 1 after this                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Cell 3+: from cuda_healthcheck import ...                   │
│ Result: ✅ Import works! Package ready to use!              │
│ Output: GPU detection, breaking changes analysis            │
└─────────────────────────────────────────────────────────────┘
```

---

## ❌ Common Mistakes to Avoid

| ❌ Don't Do This | ✅ Do This Instead |
|------------------|-------------------|
| Skip Cell 2 (restart) | Always run the restart |
| Re-run Cell 1 after restart | Just continue to Cell 3 |
| Panic at red warning note | It's normal - means success! |
| Try to import before restart | Wait for restart to complete |

---

## 🆘 Quick Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'cuda_healthcheck'"

**Fix:**
1. ✅ Did you run Cell 1? (install)
2. ✅ Did you run Cell 2? (restart)
3. ✅ Did you wait for restart to complete?
4. ⚠️ Did you accidentally re-run Cell 1 after restart? (Don't do this!)

### Problem: "Red warning note scares me"

**Fix:** It's **good news**! It means installation succeeded. Just run Cell 2 next.

### Problem: "Variables undefined after restart"

**Fix:** This is **expected**. Restart clears all variables. Don't try to use variables from Cell 1 in Cell 3+.

---

## 📚 Next Steps

Once running successfully:
1. ✅ Explore GPU detection results
2. ✅ Review breaking changes analysis
3. ✅ Check compatibility scores
4. ✅ Save results to Delta table (optional)
5. ✅ Share notebook with team

---

## 📖 More Resources

- [Full Deployment Guide](DATABRICKS_DEPLOYMENT.md) - Detailed instructions
- [Main README](../README.md) - Complete documentation
- [Troubleshooting](DATABRICKS_DEPLOYMENT.md#-common-issues) - Common issues

---

**Total Time:** ~5 minutes ⏱️  
**Difficulty:** Easy 🟢  
**Success Rate:** 99%+ when following these steps 🎯




