# Databricks notebook source
# MAGIC %md
# MAGIC # 🧬 BioNeMo Framework Validation for Databricks
# MAGIC
# MAGIC Comprehensive validation notebook for NVIDIA BioNeMo Framework installation on Databricks.
# MAGIC This notebook extends the CUDA Healthcheck Tool with BioNeMo-specific validations.
# MAGIC
# MAGIC ## What This Notebook Does:
# MAGIC
# MAGIC 1. ✅ Validates Databricks environment (leverages existing healthcheck)
# MAGIC 2. ✅ Checks CUDA availability for BioNeMo workloads
# MAGIC 3. ✅ Validates PyTorch installation and CUDA linkage
# MAGIC 4. ✅ **Tests PyTorch Lightning GPU compatibility (NEW!)**
# MAGIC 5. ✅ **Validates BioNeMo core packages availability (NEW!)**
# MAGIC
# MAGIC ## BioNeMo Framework Overview:
# MAGIC
# MAGIC BioNeMo provides three types of packages:
# MAGIC
# MAGIC ### 1. Recipes (Native PyTorch + Megatron-FSDP + TE)
# MAGIC - `bionemo-recipes/` - Lightweight, pip-installable recipes
# MAGIC - ESM2, Geneformer, DNABERT, Llama3, ViT models
# MAGIC - Supports Megatron-FSDP, Transformer Engine, FP8
# MAGIC - Ideal for Databricks environments
# MAGIC
# MAGIC ### 2. 5D Parallelism Models (NeMo + Megatron-Core)
# MAGIC - `sub-packages/` - Explicit 5D parallelism
# MAGIC - bionemo-evo2, bionemo-geneformer, bionemo-llm
# MAGIC - Requires Docker image with NeMo pre-installed
# MAGIC
# MAGIC ### 3. Tooling (Dataloading & Processing)
# MAGIC - `sub-packages/` - Lightweight utilities
# MAGIC - bionemo-scdl (Single Cell Data Loader)
# MAGIC - bionemo-moco (Molecular Co-design)
# MAGIC - bionemo-noodles (Fast FASTA I/O)
# MAGIC
# MAGIC ## Requirements:
# MAGIC
# MAGIC - GPU-enabled Databricks cluster (A100/V100/T4)
# MAGIC - ML Runtime 14.3+ with CUDA 12.0+
# MAGIC - PyTorch 2.0+ with CUDA support
# MAGIC - PyTorch Lightning 2.0+
# MAGIC - Python 3.10+
# MAGIC
# MAGIC ## References:
# MAGIC
# MAGIC - GitHub: https://github.com/NVIDIA/bionemo-framework
# MAGIC - Documentation: https://nvidia.github.io/bionemo-framework/
# MAGIC - CUDA Healthcheck: https://github.com/TavnerJC/cuda-healthcheck-on-databricks

# COMMAND ----------
# MAGIC %md
# MAGIC ## 📦 Cell 1: Setup and Imports
# MAGIC
# MAGIC Install CUDA Healthcheck Tool and import all required dependencies.
# MAGIC This cell reuses the existing healthcheck infrastructure.

# COMMAND ----------
import sys
import json
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("=" * 80)
print("🧬 BIONEMO FRAMEWORK VALIDATION - SETUP")
print("=" * 80)

# Install CUDA Healthcheck Tool if not already installed
try:
    from cuda_healthcheck import __version__, CUDADetector
    from cuda_healthcheck.databricks import detect_databricks_runtime, detect_gpu_auto
    from cuda_healthcheck.utils import get_cuda_packages_from_pip
    print(f"\n✅ CUDA Healthcheck v{__version__} already installed")
except ImportError:
    print("\n📦 Installing CUDA Healthcheck Tool...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
        "git+https://github.com/TavnerJC/cuda-healthcheck-on-databricks.git"
    ], check=True)
    
    # Import after installation
    from cuda_healthcheck import __version__, CUDADetector
    from cuda_healthcheck.databricks import detect_databricks_runtime, detect_gpu_auto
    from cuda_healthcheck.utils import get_cuda_packages_from_pip
    print(f"✅ CUDA Healthcheck v{__version__} installed successfully")

# Import standard libraries
import importlib.util

print("\n📚 Imported dependencies:")
print(f"   • cuda_healthcheck v{__version__}")
print(f"   • Python {sys.version.split()[0]}")
print(f"   • subprocess, json, datetime")

print("\n" + "=" * 80)
print("✅ SETUP COMPLETE")
print("=" * 80)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🎮 Cell 2: CUDA Environment Validation
# MAGIC
# MAGIC Reuses existing CUDA detection from healthcheck tool.
# MAGIC Validates GPU, CUDA runtime, driver versions, and PyTorch.
# MAGIC
# MAGIC **Note:** This leverages existing functions to avoid duplication.

# COMMAND ----------
print("=" * 80)
print("🎮 CUDA ENVIRONMENT VALIDATION FOR BIONEMO")
print("=" * 80)

# Initialize results dictionary
cuda_validation_results = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "databricks_runtime": None,
    "gpu_info": None,
    "cuda_environment": None,
    "pytorch_info": None,
    "status": "PENDING",
    "blockers": [],
    "warnings": []
}

try:
    # 1. Detect Databricks Runtime
    print("\n🏃 Detecting Databricks Runtime...")
    runtime_info = detect_databricks_runtime()
    cuda_validation_results["databricks_runtime"] = runtime_info
    
    print(f"   Runtime Version: {runtime_info['runtime_version']}")
    print(f"   Is ML Runtime: {runtime_info['is_ml_runtime']}")
    print(f"   Is GPU Runtime: {runtime_info['is_gpu_runtime']}")
    print(f"   Expected CUDA: {runtime_info['cuda_version']}")
    
    if not runtime_info['is_gpu_runtime']:
        cuda_validation_results["blockers"].append({
            "check": "gpu_runtime",
            "message": "Not running on a GPU-enabled runtime",
            "severity": "BLOCKER"
        })
    
    # 2. Detect GPU Hardware
    print("\n🎮 Detecting GPU Hardware...")
    gpu_info = detect_gpu_auto()
    cuda_validation_results["gpu_info"] = gpu_info
    
    gpu_count = gpu_info.get('gpu_count', 0)
    print(f"   GPU Count: {gpu_count}")
    print(f"   Environment: {gpu_info.get('environment', 'unknown')}")
    
    if gpu_count > 0 and 'gpus' in gpu_info:
        for i, gpu in enumerate(gpu_info['gpus']):
            print(f"\n   GPU {i}: {gpu.get('name', 'Unknown')}")
            print(f"      Driver: {gpu.get('driver_version', 'N/A')}")
            print(f"      Memory: {gpu.get('memory_total', 'N/A')}")
            print(f"      Compute: {gpu.get('compute_capability', 'N/A')}")
    else:
        cuda_validation_results["blockers"].append({
            "check": "gpu_detection",
            "message": f"No GPU detected: {gpu_info.get('error', 'Unknown')}",
            "severity": "BLOCKER"
        })
    
    # 3. Detect CUDA Environment
    print("\n🔧 Detecting CUDA Environment...")
    detector = CUDADetector()
    env = detector.detect_environment()
    
    cuda_env_summary = {
        "cuda_runtime": env.cuda_runtime_version,
        "cuda_driver": env.cuda_driver_version,
        "nvcc_version": env.nvcc_version,
        "libraries": []
    }
    
    print(f"   CUDA Runtime: {env.cuda_runtime_version}")
    print(f"   CUDA Driver: {env.cuda_driver_version}")
    print(f"   NVCC Version: {env.nvcc_version}")
    
    if env.cuda_runtime_version == "Not available":
        cuda_validation_results["blockers"].append({
            "check": "cuda_runtime",
            "message": "CUDA runtime not detected",
            "severity": "BLOCKER"
        })
    
    # 4. Check PyTorch Installation
    print("\n🐍 Checking PyTorch Installation...")
    pytorch_lib = next((lib for lib in env.libraries if lib.name.lower() == "torch"), None)
    
    if pytorch_lib and pytorch_lib.version != "Not installed":
        pytorch_info = {
            "version": pytorch_lib.version,
            "cuda_version": pytorch_lib.cuda_version,
            "is_compatible": pytorch_lib.is_compatible
        }
        cuda_validation_results["pytorch_info"] = pytorch_info
        
        print(f"   PyTorch Version: {pytorch_lib.version}")
        print(f"   CUDA Version: {pytorch_lib.cuda_version}")
        print(f"   Compatible: {pytorch_lib.is_compatible}")
        
        if not pytorch_lib.is_compatible:
            cuda_validation_results["warnings"].append({
                "check": "pytorch_compatibility",
                "message": f"PyTorch {pytorch_lib.version} may have compatibility issues",
                "severity": "WARNING"
            })
    else:
        cuda_validation_results["blockers"].append({
            "check": "pytorch_installation",
            "message": "PyTorch not installed",
            "severity": "BLOCKER"
        })
    
    # Store full environment
    cuda_validation_results["cuda_environment"] = cuda_env_summary
    
    # Determine overall status
    if cuda_validation_results["blockers"]:
        cuda_validation_results["status"] = "BLOCKED"
        print("\n❌ CUDA VALIDATION FAILED")
        print(f"   Blockers: {len(cuda_validation_results['blockers'])}")
        for blocker in cuda_validation_results["blockers"]:
            print(f"      • {blocker['message']}")
    else:
        cuda_validation_results["status"] = "PASSED"
        print("\n✅ CUDA VALIDATION PASSED")
        if cuda_validation_results["warnings"]:
            print(f"   Warnings: {len(cuda_validation_results['warnings'])}")

except Exception as e:
    print(f"\n❌ Error during CUDA validation: {str(e)}")
    cuda_validation_results["status"] = "ERROR"
    cuda_validation_results["blockers"].append({
        "check": "validation_error",
        "message": str(e),
        "severity": "BLOCKER"
    })

print("\n" + "=" * 80)
print(f"CUDA VALIDATION STATUS: {cuda_validation_results['status']}")
print("=" * 80)

# Display results as DataFrame
import pandas as pd

summary_data = {
    "Check": ["Databricks Runtime", "GPU Detection", "CUDA Runtime", "PyTorch"],
    "Status": [
        "✅ PASS" if runtime_info['is_gpu_runtime'] else "❌ FAIL",
        f"✅ PASS ({gpu_count} GPU)" if gpu_count > 0 else "❌ FAIL",
        "✅ PASS" if env.cuda_runtime_version != "Not available" else "❌ FAIL",
        "✅ PASS" if pytorch_lib and pytorch_lib.version != "Not installed" else "❌ FAIL"
    ],
    "Details": [
        f"Runtime {runtime_info['runtime_version']}, CUDA {runtime_info['cuda_version']}",
        gpu_info['gpus'][0]['name'] if gpu_count > 0 and 'gpus' in gpu_info else "No GPU",
        f"Runtime {env.cuda_runtime_version}, Driver {env.cuda_driver_version}",
        f"v{pytorch_lib.version} (cu{pytorch_lib.cuda_version})" if pytorch_lib and pytorch_lib.version != "Not installed" else "Not installed"
    ]
}

df_summary = pd.DataFrame(summary_data)
display(df_summary)

# COMMAND ----------
# MAGIC %md
# MAGIC ## ⚡ Cell 3: PyTorch Lightning GPU Test (NEW!)
# MAGIC
# MAGIC Tests PyTorch Lightning compatibility with GPU acceleration.
# MAGIC This is critical for BioNeMo recipes which use Lightning for training.
# MAGIC
# MAGIC **Validates:**
# MAGIC - PyTorch Lightning installation
# MAGIC - GPU device detection via Lightning
# MAGIC - Trainer initialization with GPU strategy
# MAGIC - Simple forward pass on GPU
# MAGIC - Mixed precision (FP16) support

# COMMAND ----------
print("=" * 80)
print("⚡ PYTORCH LIGHTNING GPU COMPATIBILITY TEST")
print("=" * 80)

# Initialize results dictionary
lightning_test_results = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "pytorch_lightning_installed": False,
    "version": None,
    "gpu_devices_detected": 0,
    "trainer_initialization": False,
    "gpu_forward_pass": False,
    "mixed_precision_support": False,
    "status": "PENDING",
    "blockers": [],
    "warnings": [],
    "benchmark_results": {}
}

try:
    # 1. Check PyTorch Lightning Installation
    print("\n📦 Checking PyTorch Lightning Installation...")
    
    try:
        import pytorch_lightning as pl
        lightning_test_results["pytorch_lightning_installed"] = True
        lightning_test_results["version"] = pl.__version__
        print(f"   ✅ PyTorch Lightning v{pl.__version__} installed")
    except ImportError:
        print("   ❌ PyTorch Lightning not installed")
        print("   💡 Installing PyTorch Lightning...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "pytorch-lightning>=2.0.0"
        ], check=True)
        
        import pytorch_lightning as pl
        lightning_test_results["pytorch_lightning_installed"] = True
        lightning_test_results["version"] = pl.__version__
        print(f"   ✅ PyTorch Lightning v{pl.__version__} installed successfully")
    
    # 2. Check GPU Availability via PyTorch
    print("\n🎮 Checking GPU Availability via PyTorch...")
    import torch
    
    if not torch.cuda.is_available():
        lightning_test_results["blockers"].append({
            "check": "torch_cuda_available",
            "message": "torch.cuda.is_available() returned False",
            "severity": "BLOCKER"
        })
        print("   ❌ torch.cuda.is_available() = False")
    else:
        gpu_count = torch.cuda.device_count()
        lightning_test_results["gpu_devices_detected"] = gpu_count
        print(f"   ✅ torch.cuda.is_available() = True")
        print(f"   ✅ GPU Devices: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            print(f"      GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # 3. Test Lightning Trainer Initialization
    if lightning_test_results["gpu_devices_detected"] > 0:
        print("\n⚡ Testing Lightning Trainer Initialization...")
        
        try:
            # Create a simple Lightning module for testing
            class SimpleLightningModule(pl.LightningModule):
                def __init__(self):
                    super().__init__()
                    self.layer = torch.nn.Linear(128, 64)
                
                def forward(self, x):
                    return self.layer(x)
                
                def training_step(self, batch, batch_idx):
                    x, y = batch
                    y_hat = self(x)
                    loss = torch.nn.functional.mse_loss(y_hat, y)
                    return loss
                
                def configure_optimizers(self):
                    return torch.optim.Adam(self.parameters(), lr=1e-3)
            
            # Initialize trainer with GPU
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                max_epochs=1,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                enable_checkpointing=False
            )
            
            lightning_test_results["trainer_initialization"] = True
            print(f"   ✅ Trainer initialized successfully")
            print(f"      Accelerator: {trainer.accelerator.__class__.__name__}")
            print(f"      Devices: {trainer.num_devices}")
            
        except Exception as e:
            lightning_test_results["blockers"].append({
                "check": "trainer_initialization",
                "message": f"Trainer initialization failed: {str(e)}",
                "severity": "BLOCKER"
            })
            print(f"   ❌ Trainer initialization failed: {str(e)}")
        
        # 4. Test GPU Forward Pass
        print("\n🚀 Testing GPU Forward Pass...")
        
        try:
            model = SimpleLightningModule()
            model = model.cuda()
            
            # Create test input
            test_input = torch.randn(32, 128).cuda()
            
            # Perform forward pass
            with torch.no_grad():
                output = model(test_input)
            
            # Verify output is on GPU
            if output.is_cuda:
                lightning_test_results["gpu_forward_pass"] = True
                print(f"   ✅ GPU forward pass successful")
                print(f"      Input shape: {test_input.shape}")
                print(f"      Output shape: {output.shape}")
                print(f"      Device: {output.device}")
            else:
                lightning_test_results["warnings"].append({
                    "check": "gpu_forward_pass",
                    "message": "Output tensor not on GPU",
                    "severity": "WARNING"
                })
                print(f"   ⚠️  Output tensor not on GPU")
                
        except Exception as e:
            lightning_test_results["blockers"].append({
                "check": "gpu_forward_pass",
                "message": f"GPU forward pass failed: {str(e)}",
                "severity": "BLOCKER"
            })
            print(f"   ❌ GPU forward pass failed: {str(e)}")
        
        # 5. Test Mixed Precision (FP16) Support
        print("\n🔥 Testing Mixed Precision (FP16) Support...")
        
        try:
            # Check if GPU supports FP16
            gpu_capability = torch.cuda.get_device_capability(0)
            supports_fp16 = gpu_capability[0] >= 7  # Volta and newer
            
            if supports_fp16:
                # Test with automatic mixed precision
                with torch.cuda.amp.autocast():
                    test_input_fp16 = torch.randn(32, 128).cuda()
                    output_fp16 = model(test_input_fp16)
                
                lightning_test_results["mixed_precision_support"] = True
                print(f"   ✅ Mixed precision (FP16) supported")
                print(f"      GPU Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}")
                print(f"      Output dtype: {output_fp16.dtype}")
            else:
                lightning_test_results["warnings"].append({
                    "check": "mixed_precision",
                    "message": f"GPU compute capability {gpu_capability[0]}.{gpu_capability[1]} < 7.0, limited FP16 support",
                    "severity": "WARNING"
                })
                print(f"   ⚠️  GPU compute capability {gpu_capability[0]}.{gpu_capability[1]} < 7.0")
                print(f"      Mixed precision may have limited performance benefits")
                
        except Exception as e:
            lightning_test_results["warnings"].append({
                "check": "mixed_precision",
                "message": f"Mixed precision test failed: {str(e)}",
                "severity": "WARNING"
            })
            print(f"   ⚠️  Mixed precision test failed: {str(e)}")
        
        # 6. Benchmark GPU Performance
        print("\n📊 Benchmarking GPU Performance...")
        
        try:
            import time
            
            model = SimpleLightningModule().cuda()
            model.eval()
            
            # Warm-up
            for _ in range(10):
                with torch.no_grad():
                    _ = model(torch.randn(256, 128).cuda())
            
            # Benchmark
            torch.cuda.synchronize()
            start_time = time.time()
            
            iterations = 100
            for _ in range(iterations):
                with torch.no_grad():
                    _ = model(torch.randn(256, 128).cuda())
            
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            
            throughput = iterations / elapsed_time
            avg_latency_ms = (elapsed_time / iterations) * 1000
            
            lightning_test_results["benchmark_results"] = {
                "iterations": iterations,
                "elapsed_time_s": round(elapsed_time, 4),
                "throughput_iter_per_s": round(throughput, 2),
                "avg_latency_ms": round(avg_latency_ms, 4)
            }
            
            print(f"   ✅ Benchmark completed")
            print(f"      Iterations: {iterations}")
            print(f"      Elapsed Time: {elapsed_time:.4f}s")
            print(f"      Throughput: {throughput:.2f} iter/s")
            print(f"      Avg Latency: {avg_latency_ms:.4f}ms")
            
        except Exception as e:
            lightning_test_results["warnings"].append({
                "check": "benchmark",
                "message": f"Benchmark failed: {str(e)}",
                "severity": "WARNING"
            })
            print(f"   ⚠️  Benchmark failed: {str(e)}")
    
    # Determine overall status
    if lightning_test_results["blockers"]:
        lightning_test_results["status"] = "BLOCKED"
        print("\n❌ PYTORCH LIGHTNING TEST FAILED")
        print(f"   Blockers: {len(lightning_test_results['blockers'])}")
        for blocker in lightning_test_results["blockers"]:
            print(f"      • {blocker['message']}")
    else:
        lightning_test_results["status"] = "PASSED"
        print("\n✅ PYTORCH LIGHTNING TEST PASSED")
        if lightning_test_results["warnings"]:
            print(f"   Warnings: {len(lightning_test_results['warnings'])}")

except Exception as e:
    print(f"\n❌ Error during Lightning test: {str(e)}")
    lightning_test_results["status"] = "ERROR"
    lightning_test_results["blockers"].append({
        "check": "lightning_test_error",
        "message": str(e),
        "severity": "BLOCKER"
    })

print("\n" + "=" * 80)
print(f"PYTORCH LIGHTNING TEST STATUS: {lightning_test_results['status']}")
print("=" * 80)

# Display results as DataFrame
summary_data = {
    "Check": [
        "PyTorch Lightning Installed",
        "GPU Devices Detected",
        "Trainer Initialization",
        "GPU Forward Pass",
        "Mixed Precision (FP16)"
    ],
    "Status": [
        "✅ PASS" if lightning_test_results["pytorch_lightning_installed"] else "❌ FAIL",
        f"✅ PASS ({lightning_test_results['gpu_devices_detected']})" if lightning_test_results["gpu_devices_detected"] > 0 else "❌ FAIL",
        "✅ PASS" if lightning_test_results["trainer_initialization"] else "❌ FAIL",
        "✅ PASS" if lightning_test_results["gpu_forward_pass"] else "❌ FAIL",
        "✅ PASS" if lightning_test_results["mixed_precision_support"] else "⚠️ WARN"
    ],
    "Details": [
        f"v{lightning_test_results['version']}" if lightning_test_results["version"] else "Not installed",
        f"{lightning_test_results['gpu_devices_detected']} GPU(s) available",
        "Trainer initialized with GPU accelerator" if lightning_test_results["trainer_initialization"] else "Failed",
        "Forward pass successful on GPU" if lightning_test_results["gpu_forward_pass"] else "Failed",
        f"FP16 supported" if lightning_test_results["mixed_precision_support"] else "Limited support"
    ]
}

df_lightning = pd.DataFrame(summary_data)
display(df_lightning)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 🧬 Cell 4: BioNeMo Core Package Availability (NEW!)
# MAGIC
# MAGIC Tests availability and imports of BioNeMo Framework packages.
# MAGIC
# MAGIC **Validates:**
# MAGIC - bionemo-core (configuration, testing utilities)
# MAGIC - bionemo-scdl (single cell data loader)
# MAGIC - bionemo-moco (molecular co-design tools)
# MAGIC - bionemo-noodles (fast FASTA I/O)
# MAGIC
# MAGIC **Note:** BioNeMo recipes and 5D parallelism models require Docker images
# MAGIC and are tested separately.

# COMMAND ----------
print("=" * 80)
print("🧬 BIONEMO CORE PACKAGE AVAILABILITY TEST")
print("=" * 80)

# Initialize results dictionary
bionemo_test_results = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "packages_tested": {},
    "status": "PENDING",
    "blockers": [],
    "warnings": []
}

# Define BioNeMo packages to test
# Reference: https://github.com/NVIDIA/bionemo-framework
bionemo_packages = {
    "bionemo-core": {
        "description": "Model config and test data utilities",
        "pip_name": "bionemo-core",
        "import_name": "bionemo.core",
        "required": False,
        "support": "Active",
        "category": "5D Parallelism"
    },
    "bionemo-scdl": {
        "description": "Modular Single Cell Data Loader",
        "pip_name": "bionemo-scdl",
        "import_name": "bionemo.scdl",
        "required": False,
        "support": "Active",
        "category": "Tooling"
    },
    "bionemo-moco": {
        "description": "Molecular Co-design tools",
        "pip_name": "bionemo-moco",
        "import_name": "bionemo.moco",
        "required": False,
        "support": "Active",
        "category": "Tooling"
    },
    "bionemo-noodles": {
        "description": "Python API to fast FASTA file I/O",
        "pip_name": "bionemo-noodles",
        "import_name": "bionemo.noodles",
        "required": False,
        "support": "Maintenance",
        "category": "Tooling"
    },
    "bionemo-llm": {
        "description": "5D parallel base model (BioBert)",
        "pip_name": "bionemo-llm",
        "import_name": "bionemo.llm",
        "required": False,
        "support": "Active",
        "category": "5D Parallelism"
    },
    "bionemo-evo2": {
        "description": "5D parallel Evo2 model",
        "pip_name": "bionemo-evo2",
        "import_name": "bionemo.evo2",
        "required": False,
        "support": "Active",
        "category": "5D Parallelism"
    },
    "bionemo-geneformer": {
        "description": "5D parallel Geneformer model",
        "pip_name": "bionemo-geneformer",
        "import_name": "bionemo.geneformer",
        "required": False,
        "support": "Maintenance",
        "category": "5D Parallelism"
    }
}

print("\n📦 Testing BioNeMo Package Availability...")
print(f"   Total packages to test: {len(bionemo_packages)}")

# Test each package
for package_name, package_info in bionemo_packages.items():
    print(f"\n{'─' * 80}")
    print(f"Testing: {package_name}")
    print(f"{'─' * 80}")
    
    result = {
        "description": package_info["description"],
        "pip_name": package_info["pip_name"],
        "import_name": package_info["import_name"],
        "category": package_info["category"],
        "support": package_info["support"],
        "is_installed": False,
        "is_importable": False,
        "version": None,
        "import_error": None,
        "test_passed": False
    }
    
    try:
        # 1. Check if package is installed via pip
        print(f"   Checking installation via pip...")
        pip_result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_info["pip_name"]],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if pip_result.returncode == 0:
            result["is_installed"] = True
            # Extract version
            for line in pip_result.stdout.split('\n'):
                if line.startswith('Version:'):
                    result["version"] = line.split(':')[1].strip()
                    break
            print(f"   ✅ Package installed: v{result['version']}")
        else:
            print(f"   ❌ Package not installed")
            print(f"   💡 Install with: pip install {package_info['pip_name']}")
        
        # 2. Try to import the package
        if result["is_installed"]:
            print(f"   Testing import: {package_info['import_name']}...")
            
            try:
                # Attempt import
                spec = importlib.util.find_spec(package_info["import_name"])
                if spec is not None:
                    module = importlib.import_module(package_info["import_name"])
                    result["is_importable"] = True
                    result["test_passed"] = True
                    
                    # Try to get version from module
                    if hasattr(module, '__version__'):
                        result["version"] = module.__version__
                    
                    print(f"   ✅ Import successful")
                    
                    # Check for common submodules
                    submodules = []
                    if hasattr(module, '__all__'):
                        submodules = module.__all__[:5]  # First 5
                        print(f"   📦 Available submodules: {', '.join(submodules)}")
                else:
                    result["import_error"] = "Module spec not found"
                    print(f"   ❌ Import failed: Module spec not found")
                    
            except ImportError as e:
                result["import_error"] = str(e)
                print(f"   ❌ Import failed: {str(e)}")
            except Exception as e:
                result["import_error"] = f"Unexpected error: {str(e)}"
                print(f"   ❌ Unexpected error: {str(e)}")
    
    except subprocess.TimeoutExpired:
        result["import_error"] = "pip show command timed out"
        print(f"   ❌ pip show command timed out")
    except Exception as e:
        result["import_error"] = f"Test error: {str(e)}"
        print(f"   ❌ Test error: {str(e)}")
    
    # Store result
    bionemo_test_results["packages_tested"][package_name] = result
    
    # Track blockers and warnings
    if package_info["required"] and not result["test_passed"]:
        bionemo_test_results["blockers"].append({
            "check": package_name,
            "message": f"Required package {package_name} not available",
            "severity": "BLOCKER"
        })
    elif not result["test_passed"] and result["is_installed"]:
        bionemo_test_results["warnings"].append({
            "check": package_name,
            "message": f"Package {package_name} installed but import failed: {result['import_error']}",
            "severity": "WARNING"
        })

# Calculate statistics
total_tested = len(bionemo_packages)
installed = sum(1 for r in bionemo_test_results["packages_tested"].values() if r["is_installed"])
importable = sum(1 for r in bionemo_test_results["packages_tested"].values() if r["is_importable"])
passed = sum(1 for r in bionemo_test_results["packages_tested"].values() if r["test_passed"])

print("\n" + "=" * 80)
print("📊 BIONEMO PACKAGE TEST SUMMARY")
print("=" * 80)
print(f"   Total Packages Tested: {total_tested}")
print(f"   Installed: {installed}/{total_tested}")
print(f"   Importable: {importable}/{total_tested}")
print(f"   Tests Passed: {passed}/{total_tested}")

# Determine overall status
if bionemo_test_results["blockers"]:
    bionemo_test_results["status"] = "BLOCKED"
    print(f"\n❌ BIONEMO TEST FAILED")
    print(f"   Blockers: {len(bionemo_test_results['blockers'])}")
elif passed == 0:
    bionemo_test_results["status"] = "NO_PACKAGES"
    print(f"\n⚠️  NO BIONEMO PACKAGES INSTALLED")
    print(f"   This is expected if you haven't installed BioNeMo yet")
else:
    bionemo_test_results["status"] = "PASSED"
    print(f"\n✅ BIONEMO TEST PASSED")
    print(f"   {passed}/{total_tested} packages available")

if bionemo_test_results["warnings"]:
    print(f"\n⚠️  Warnings: {len(bionemo_test_results['warnings'])}")
    for warning in bionemo_test_results["warnings"]:
        print(f"   • {warning['message']}")

print("=" * 80)

# Display detailed results as DataFrame
summary_data = []
for package_name, result in bionemo_test_results["packages_tested"].items():
    summary_data.append({
        "Package": package_name,
        "Category": result["category"],
        "Support": result["support"],
        "Installed": "✅" if result["is_installed"] else "❌",
        "Importable": "✅" if result["is_importable"] else ("❌" if result["is_installed"] else "⏭️"),
        "Version": result["version"] if result["version"] else "N/A",
        "Status": "✅ PASS" if result["test_passed"] else ("⚠️ NOT INSTALLED" if not result["is_installed"] else "❌ IMPORT FAILED")
    })

df_bionemo = pd.DataFrame(summary_data)
display(df_bionemo)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 📋 Cell 5: Final Summary Report
# MAGIC
# MAGIC Comprehensive summary of all validation checks with recommendations.

# COMMAND ----------
print("=" * 80)
print("📋 BIONEMO FRAMEWORK VALIDATION - FINAL REPORT")
print("=" * 80)

# Aggregate all results
final_report = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "validation_sections": {
        "cuda_environment": cuda_validation_results,
        "pytorch_lightning": lightning_test_results,
        "bionemo_packages": bionemo_test_results
    },
    "overall_status": "PENDING",
    "total_blockers": 0,
    "total_warnings": 0,
    "recommendations": []
}

# Count blockers and warnings
for section in final_report["validation_sections"].values():
    final_report["total_blockers"] += len(section.get("blockers", []))
    final_report["total_warnings"] += len(section.get("warnings", []))

# Determine overall status
if final_report["total_blockers"] > 0:
    final_report["overall_status"] = "BLOCKED"
elif bionemo_test_results["status"] == "NO_PACKAGES":
    final_report["overall_status"] = "READY_FOR_INSTALL"
else:
    final_report["overall_status"] = "READY"

# Generate recommendations
print("\n🎯 VALIDATION SUMMARY:")
print("─" * 80)

# Section 1: CUDA Environment
cuda_status = "✅ PASS" if cuda_validation_results["status"] == "PASSED" else "❌ FAIL"
print(f"   1. CUDA Environment: {cuda_status}")
if cuda_validation_results["status"] == "PASSED":
    print(f"      • Runtime: {cuda_validation_results['databricks_runtime']['runtime_version']}")
    print(f"      • GPUs: {cuda_validation_results['gpu_info']['gpu_count']}")
    print(f"      • PyTorch: v{cuda_validation_results['pytorch_info']['version']}")
else:
    print(f"      • Blockers: {len(cuda_validation_results['blockers'])}")

# Section 2: PyTorch Lightning
lightning_status = "✅ PASS" if lightning_test_results["status"] == "PASSED" else "❌ FAIL"
print(f"\n   2. PyTorch Lightning: {lightning_status}")
if lightning_test_results["status"] == "PASSED":
    print(f"      • Version: v{lightning_test_results['version']}")
    print(f"      • GPU Devices: {lightning_test_results['gpu_devices_detected']}")
    print(f"      • Mixed Precision: {'✅ Supported' if lightning_test_results['mixed_precision_support'] else '⚠️ Limited'}")
    if lightning_test_results["benchmark_results"]:
        bench = lightning_test_results["benchmark_results"]
        print(f"      • Throughput: {bench['throughput_iter_per_s']} iter/s")
else:
    print(f"      • Blockers: {len(lightning_test_results['blockers'])}")

# Section 3: BioNeMo Packages
if bionemo_test_results["status"] == "NO_PACKAGES":
    bionemo_status = "⚠️ NOT INSTALLED"
elif bionemo_test_results["status"] == "PASSED":
    bionemo_status = "✅ PASS"
else:
    bionemo_status = "❌ FAIL"

print(f"\n   3. BioNeMo Packages: {bionemo_status}")
installed = sum(1 for r in bionemo_test_results["packages_tested"].values() if r["is_installed"])
importable = sum(1 for r in bionemo_test_results["packages_tested"].values() if r["is_importable"])
print(f"      • Installed: {installed}/{len(bionemo_packages)}")
print(f"      • Importable: {importable}/{len(bionemo_packages)}")

# Overall Status
print("\n" + "=" * 80)
print(f"🎯 OVERALL STATUS: {final_report['overall_status']}")
print("=" * 80)

if final_report["total_blockers"] > 0:
    print(f"\n❌ VALIDATION FAILED")
    print(f"   Total Blockers: {final_report['total_blockers']}")
    print(f"   Total Warnings: {final_report['total_warnings']}")
    
    print(f"\n🚨 BLOCKERS FOUND:")
    for section_name, section_data in final_report["validation_sections"].items():
        if section_data.get("blockers"):
            print(f"\n   {section_name.replace('_', ' ').title()}:")
            for blocker in section_data["blockers"]:
                print(f"      • {blocker['message']}")
    
    # Add recommendations
    final_report["recommendations"].append("Fix all blockers before proceeding with BioNeMo installation")
    
elif final_report["overall_status"] == "READY_FOR_INSTALL":
    print(f"\n✅ ENVIRONMENT READY FOR BIONEMO INSTALLATION")
    print(f"\n📦 Next Steps:")
    print(f"\n   1. Choose your BioNeMo installation path:")
    print(f"")
    print(f"      🎯 Option A: BioNeMo Recipes (Recommended for Databricks)")
    print(f"         - Lightweight, pip-installable")
    print(f"         - Supports ESM2, Geneformer, DNABERT, Llama3")
    print(f"         - Install: pip install git+https://github.com/NVIDIA/bionemo-framework.git#subdirectory=bionemo-recipes")
    print(f"")
    print(f"      🎯 Option B: BioNeMo Core + Tooling")
    print(f"         - Install individual packages:")
    print(f"         - pip install bionemo-core")
    print(f"         - pip install bionemo-scdl  # Single cell data loader")
    print(f"         - pip install bionemo-moco  # Molecular co-design")
    print(f"")
    print(f"      🎯 Option C: BioNeMo 5D Parallelism (Requires Docker)")
    print(f"         - Use NVIDIA BioNeMo container:")
    print(f"         - docker run --rm -it --gpus=all nvcr.io/nvidia/clara/bionemo-framework:latest")
    print(f"         - Not recommended for Databricks managed clusters")
    print(f"")
    print(f"   2. After installation, re-run this notebook to validate")
    print(f"")
    print(f"   3. Refer to BioNeMo documentation:")
    print(f"      https://nvidia.github.io/bionemo-framework/")
    
    final_report["recommendations"].extend([
        "Install BioNeMo recipes for Databricks: pip install git+https://github.com/NVIDIA/bionemo-framework.git#subdirectory=bionemo-recipes",
        "Or install individual tools: pip install bionemo-scdl bionemo-moco",
        "Re-run this notebook after installation to validate"
    ])
    
else:
    print(f"\n✅ VALIDATION PASSED - BIONEMO READY")
    print(f"\n🎉 Your environment is configured correctly for BioNeMo workloads!")
    
    # Show installed packages
    installed_packages = [name for name, result in bionemo_test_results["packages_tested"].items() if result["is_importable"]]
    if installed_packages:
        print(f"\n📦 Available BioNeMo packages:")
        for pkg in installed_packages:
            result = bionemo_test_results["packages_tested"][pkg]
            print(f"      • {pkg} v{result['version']} ({result['category']})")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Explore BioNeMo tutorials and examples")
    print(f"   2. Start with small models to validate training pipeline")
    print(f"   3. Monitor GPU utilization during training")
    print(f"   4. Refer to BioNeMo documentation for best practices")
    
    final_report["recommendations"].extend([
        "Environment is ready for BioNeMo workloads",
        "Explore BioNeMo tutorials: https://nvidia.github.io/bionemo-framework/",
        "Start with small models to validate training pipeline"
    ])

if final_report["total_warnings"] > 0:
    print(f"\n⚠️  WARNINGS ({final_report['total_warnings']}):")
    for section_name, section_data in final_report["validation_sections"].items():
        if section_data.get("warnings"):
            for warning in section_data["warnings"]:
                print(f"   • {warning['message']}")

print("\n" + "=" * 80)
print("📚 REFERENCES")
print("=" * 80)
print("   • BioNeMo Framework: https://github.com/NVIDIA/bionemo-framework")
print("   • BioNeMo Documentation: https://nvidia.github.io/bionemo-framework/")
print("   • CUDA Healthcheck Tool: https://github.com/TavnerJC/cuda-healthcheck-on-databricks")
print("   • PyTorch Lightning: https://lightning.ai/docs/pytorch/")
print("=" * 80)

# Export report to JSON
print(f"\n💾 Exporting validation report...")
report_json = json.dumps(final_report, indent=2)

# Save to DBFS (if available)
try:
    report_path = "/dbfs/tmp/bionemo_validation_report.json"
    with open(report_path, 'w') as f:
        f.write(report_json)
    print(f"   ✅ Report saved to: {report_path}")
except Exception as e:
    print(f"   ⚠️  Could not save to DBFS: {str(e)}")
    print(f"   📋 Report JSON:")
    print(report_json[:500] + "..." if len(report_json) > 500 else report_json)

print("\n✅ VALIDATION COMPLETE!")
print("=" * 80)

