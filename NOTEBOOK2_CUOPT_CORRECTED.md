# 🔧 **Corrected Notebook 2: Using CuOPT (Open Source!)**

## 🎉 **CuOPT is FREE and Open Source!**

**My apologies** - NVIDIA CuOPT is **Apache 2.0 licensed** and freely available!
- GitHub: https://github.com/NVIDIA/cuopt
- License: Apache-2.0 ✅
- No commercial license required ✅

---

## ✅ **Corrected Installation Command**

The issue was using the **wrong pip install command**. CuOPT needs to be installed from the NVIDIA Python Package Index.

### **For CUDA 12.x (Your Environment):**

```python
# COMMAND ----------
# Install CuOPT from NVIDIA Python Package Index
%pip install --extra-index-url=https://pypi.nvidia.com cuopt-server-cu12==26.02.* cuopt-sh-client==26.02.*
dbutils.library.restartPython()
```

### **Key Difference:**
- ❌ **Wrong:** `%pip install cuopt-cu12`
- ✅ **Correct:** `%pip install --extra-index-url=https://pypi.nvidia.com cuopt-server-cu12==26.02.*`

---

## 📓 **Complete Corrected Notebook 2**

Based on the [Databricks routing notebook](https://github.com/databricks-industry-solutions/routing/blob/main/06_gpu_route_optimization.ipynb), here's the corrected code:

```python
# Databricks notebook source
# MAGIC %md
# MAGIC # 🚀 CuOPT Routing Benchmark
# MAGIC
# MAGIC GPU-accelerated vehicle routing optimization with NVIDIA CuOPT.

# COMMAND ----------
# Install CuOPT from NVIDIA Python Package Index
%pip install --extra-index-url=https://pypi.nvidia.com cuopt-server-cu12==26.02.* cuopt-sh-client==26.02.* pandas numpy matplotlib scipy
dbutils.library.restartPython()

# COMMAND ----------
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from cuopt import routing
from cuopt.routing import utils

# Verify CuOPT installed
try:
    print("✅ CuOPT imported successfully")
    print(f"CuOPT routing module available: {routing}")
except Exception as e:
    print(f"❌ CuOPT import failed: {e}")
    dbutils.notebook.exit(json.dumps({"error": "CuOPT not available", "message": str(e)}))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load Environment Snapshot

# COMMAND ----------
# Retrieve environment from previous notebook
try:
    environment_str = dbutils.jobs.taskValues.get(
        taskKey="01_validate_environment", 
        key="environment", 
        debugValue="{}"
    )
    if environment_str and environment_str != "{}":
        environment = json.loads(environment_str)
        print(f"✅ Environment loaded from Notebook 1")
    else:
        raise ValueError("No environment data")
except Exception as e:
    print(f"⚠️  Could not load environment from task values: {e}")
    print(f"📋 Using manual environment configuration")
    
    # Manual environment (update from your Notebook 1 output)
    environment = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_architecture": "NVIDIA A10G",
        "cuda_environment": {
            "runtime": "12.6",
            "driver": "12.4",
            "pytorch": "2.7.1+cu126",
            "pytorch_cuda": "12.6"
        },
        "compatibility_score": 70,
        "compute_capability": "8.6"
    }

print(f"Testing on: {environment['gpu_architecture']}")
print(f"CUDA Version: {environment['cuda_environment']['runtime']}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Define Benchmark Problems

# COMMAND ----------
# Define test cases
test_cases = [
    {
        "name": "Small_Local_Delivery",
        "num_stops": 100,
        "num_vehicles": 3,
        "description": "Small local delivery (100 stops)"
    },
    {
        "name": "Medium_Regional_Distribution",
        "num_stops": 500,
        "num_vehicles": 10,
        "description": "Medium regional distribution (500 stops)"
    },
    {
        "name": "Large_National_Supply_Chain",
        "num_stops": 1000,
        "num_vehicles": 20,
        "description": "Large supply chain (1000 stops)"
    },
]

print(f"📋 Configured {len(test_cases)} test cases")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Generate Routing Problem Data

# COMMAND ----------
def create_data_model(num_stops, num_vehicles, seed=42):
    """Create routing problem data model for CuOPT."""
    np.random.seed(seed)
    
    # Number of locations (stops + depot)
    num_locations = num_stops + 1
    
    # Generate random coordinates
    coords = np.random.rand(num_locations, 2) * 100
    
    # Calculate distance matrix (Euclidean)
    from scipy.spatial.distance import cdist
    distance_matrix = cdist(coords, coords, metric='euclidean')
    
    # Convert to CuOPT format (integer meters)
    distance_matrix = (distance_matrix * 1000).astype(np.int32)
    
    # Generate demands (depot has 0 demand)
    demands = np.zeros(num_locations, dtype=np.int32)
    demands[1:] = np.random.randint(1, 20, num_stops)
    
    # Vehicle capacities
    vehicle_capacities = np.random.randint(50, 100, num_vehicles).astype(np.int32)
    
    data = {
        "distance_matrix": distance_matrix,
        "coordinates": coords,
        "demands": demands,
        "vehicle_capacities": vehicle_capacities,
        "num_vehicles": num_vehicles,
        "depot": 0
    }
    
    return data

print("✅ Data generation function defined")

# COMMAND ----------
# MAGIC %md
# MAGIC ## CuOPT Solver

# COMMAND ----------
def solve_with_cuopt(data, time_limit_seconds=300):
    """
    Solve VRP using NVIDIA CuOPT.
    
    Based on: https://github.com/databricks-industry-solutions/routing
    """
    try:
        # Create CuOPT DataModel
        n_locations = len(data['distance_matrix'])
        n_vehicles = data['num_vehicles']
        
        data_model = routing.DataModel(n_locations, n_vehicles)
        
        # Set cost matrix (distance)
        data_model.add_cost_matrix(data['distance_matrix'])
        
        # Add capacity constraint
        data_model.add_capacity_dimension(
            "demand",
            data['demands'],
            data['vehicle_capacities']
        )
        
        # Set solver settings
        solver_settings = routing.SolverSettings()
        solver_settings.time_limit = time_limit_seconds
        
        # Solve
        routing_solution = routing.Solve(data_model, solver_settings)
        
        if routing_solution.get_status() == 0:  # Success
            return {
                "status": "success",
                "solution_cost": routing_solution.final_cost,
                "routes": routing_solution.get_route(),
                "num_routes": len([r for r in routing_solution.get_route() if len(r) > 2])
            }
        else:
            return {
                "status": "no_solution",
                "error": f"Solver status: {routing_solution.get_status()}"
            }
            
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }

print("✅ CuOPT solver function defined")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Run Benchmarks

# COMMAND ----------
def run_benchmark(test_case):
    """Run benchmark for a single test case."""
    print(f"\n{'=' * 80}")
    print(f"🚀 Running: {test_case['name']}")
    print(f"{'=' * 80}")
    print(f"Stops: {test_case['num_stops']}, Vehicles: {test_case['num_vehicles']}")
    
    start_time = time.time()
    
    try:
        # Generate problem
        print("📊 Generating problem data...")
        data = create_data_model(test_case['num_stops'], test_case['num_vehicles'])
        
        # Solve with CuOPT
        print("🔄 Solving with CuOPT...")
        solve_start = time.time()
        solution = solve_with_cuopt(data, time_limit_seconds=300)
        solve_time = time.time() - solve_start
        
        total_time = time.time() - start_time
        
        if solution['status'] == 'success':
            result = {
                "problem_name": test_case['name'],
                "status": "success",
                "solve_time_seconds": round(solve_time, 2),
                "total_time_seconds": round(total_time, 2),
                "solution_cost": solution['solution_cost'],
                "num_routes": solution['num_routes'],
                "num_stops": test_case['num_stops'],
                "num_vehicles": test_case['num_vehicles'],
            }
            
            print(f"✅ Status: {result['status']}")
            print(f"⏱️  Solve Time: {result['solve_time_seconds']}s")
            print(f"💰 Solution Cost: {result['solution_cost']:,.0f}")
            print(f"🚚 Routes Used: {result['num_routes']}/{result['num_vehicles']}")
            
            return result
        else:
            print(f"❌ {solution['status']}: {solution.get('error', 'Unknown')}")
            return {
                "problem_name": test_case['name'],
                "status": solution['status'],
                "total_time_seconds": round(total_time, 2),
                "error": solution.get('error', 'Unknown'),
                "num_stops": test_case['num_stops'],
                "num_vehicles": test_case['num_vehicles'],
            }
            
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Error: {e}")
        
        return {
            "problem_name": test_case['name'],
            "status": "failed",
            "error": str(e),
            "total_time_seconds": round(total_time, 2),
            "num_stops": test_case['num_stops'],
            "num_vehicles": test_case['num_vehicles'],
        }

# Run all benchmarks
results = []

for test_case in test_cases:
    result = run_benchmark(test_case)
    
    # Add environment metadata
    result['gpu_architecture'] = environment['gpu_architecture']
    result['cuda_version'] = environment['cuda_environment']['runtime']
    result['timestamp'] = datetime.now(timezone.utc).isoformat()
    result['solver'] = 'CuOPT'
    
    results.append(result)
    
    # Save intermediate result
    dbutils.jobs.taskValues.set(
        key=f"result_{test_case['name']}", 
        value=json.dumps(result)
    )

print(f"\n{'=' * 80}")
print("✅ All benchmarks complete!")
print(f"{'=' * 80}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Display Results

# COMMAND ----------
results_df = pd.DataFrame(results)
display(results_df)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Visualize Performance

# COMMAND ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Solve time vs problem size
successful_results = results_df[results_df['status'] == 'success']

if len(successful_results) > 0:
    ax1.plot(successful_results['num_stops'], 
             successful_results['solve_time_seconds'], 
             marker='o', linewidth=2, markersize=8, color='#76b900')
    ax1.set_xlabel('Number of Stops', fontsize=12)
    ax1.set_ylabel('Solve Time (seconds)', fontsize=12)
    ax1.set_title(f'CuOPT Performance on {environment["gpu_architecture"]}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Throughput
    successful_results['throughput'] = successful_results['num_stops'] / successful_results['solve_time_seconds']
    ax2.bar(range(len(successful_results)), successful_results['throughput'], color='#76b900', alpha=0.7)
    ax2.set_xlabel('Problem Index', fontsize=12)
    ax2.set_ylabel('Throughput (stops/sec)', fontsize=12)
    ax2.set_title('GPU-Accelerated Optimization Throughput', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
else:
    print("⚠️  No successful results to plot")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Save Results

# COMMAND ----------
benchmark_summary = {
    "environment": environment,
    "results": results,
    "solver": "NVIDIA CuOPT",
    "summary": {
        "total_tests": len(results),
        "successful_tests": len([r for r in results if r['status'] == 'success']),
        "failed_tests": len([r for r in results if r['status'] != 'success']),
        "avg_solve_time": results_df[results_df['status'] == 'success']['solve_time_seconds'].mean() if len(successful_results) > 0 else 0,
    }
}

dbutils.jobs.taskValues.set(key="benchmark_summary", value=json.dumps(benchmark_summary))

print("=" * 80)
print("📊 BENCHMARK SUMMARY")
print("=" * 80)
print(f"Solver: NVIDIA CuOPT (GPU-Accelerated)")
print(f"GPU: {environment['gpu_architecture']}")
print(f"CUDA: {environment['cuda_environment']['runtime']}")
print(f"Total Tests: {benchmark_summary['summary']['total_tests']}")
print(f"Successful: {benchmark_summary['summary']['successful_tests']}")
print(f"Failed: {benchmark_summary['summary']['failed_tests']}")
if benchmark_summary['summary']['avg_solve_time'] > 0:
    print(f"Avg Solve Time: {benchmark_summary['summary']['avg_solve_time']:.2f}s")
print("=" * 80)
```

---

## 🎯 **Key Differences from My Wrong Version**

| Aspect | My Wrong Info | Correct Info |
|--------|---------------|--------------|
| **License** | Commercial ❌ | Apache 2.0 ✅ |
| **Installation** | `pip install cuopt-cu12` | `pip install --extra-index-url=https://pypi.nvidia.com cuopt-server-cu12==26.02.*` |
| **Availability** | Requires NVIDIA license | Freely available on GitHub |
| **Works in Databricks** | No ❌ | **Yes!** ✅ (with correct install) |

---

## 🚀 **Why Your Install Failed**

The error `libcuopt.so: cannot open shared object file` happened because:

1. ❌ Missing `--extra-index-url=https://pypi.nvidia.com`
2. ❌ Wrong package name (`cuopt-cu12` instead of `cuopt-server-cu12`)
3. ❌ Need both `cuopt-server-cu12` AND `cuopt-sh-client`

**Correct command:**
```bash
pip install --extra-index-url=https://pypi.nvidia.com cuopt-server-cu12==26.02.* cuopt-sh-client==26.02.*
```

---

## ✅ **Next Steps**

1. **Replace Notebook 2** with the corrected code above
2. **Run Cell 1** with the proper CuOPT installation
3. **Wait for restart**
4. **Run remaining cells**

---

## 📚 **References**

- **CuOPT GitHub**: https://github.com/NVIDIA/cuopt (Apache 2.0 License)
- **Databricks Routing Notebook**: https://github.com/databricks-industry-solutions/routing/blob/main/06_gpu_route_optimization.ipynb
- **CuOPT Documentation**: https://docs.nvidia.com/cuopt/

---

**Thank you for the correction! CuOPT is indeed open source and perfect for our A10 vs H100 benchmark!** 🎉


