# CUDA Healthcheck Tool for Databricks

Detect CUDA version incompatibilities between developer environments and Databricks clusters before they cause production failures.

## 🎯 Project Goal

Proactively identify CUDA version mismatches, library incompatibilities, and breaking changes that could cause production failures in Databricks GPU-enabled clusters.

## 🚀 Features

- **Comprehensive CUDA Detection**: Detects CUDA versions, GPU properties, and driver information
- **Library Compatibility Checking**: Analyzes PyTorch, TensorFlow, cuDF/RAPIDS compatibility
- **Breaking Changes Database**: Maintains knowledge base of CUDA version transitions
- **Databricks Integration**: Scans all GPU clusters in your workspace
- **Delta Table Storage**: Stores results in Unity Catalog for historical analysis
- **Compatibility Scoring**: Provides actionable compatibility scores and recommendations

## 📋 Prerequisites

- Python 3.10 or higher
- Databricks workspace with GPU-enabled clusters
- Databricks Personal Access Token (PAT)
- Access to Databricks Unity Catalog (for Delta table storage)

## 🔧 Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd cuda-healthcheck
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

```bash
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your-personal-access-token"
export DATABRICKS_WAREHOUSE_ID="your-warehouse-id"  # For Delta table operations
```

## 📖 Usage

### Running Local CUDA Detection

Detect CUDA environment on the current machine:

```python
from src.cuda_detector import detect_cuda_environment

results = detect_cuda_environment()
print(results)
```

Or run directly:

```bash
python -m src.cuda_detector.detector
```

### Running Complete Healthcheck

Run a complete healthcheck with breaking change analysis:

```python
from src.healthcheck import run_complete_healthcheck

results = run_complete_healthcheck()
print(results["compatibility_analysis"]["recommendation"])
```

### Scanning Databricks Clusters

Scan all GPU clusters in your workspace:

```python
from src.databricks_api import scan_clusters

results = scan_clusters(save_to_delta=True)

# View summary
print(f"Scanned {results['summary']['total_clusters']} clusters")
print(f"Found {results['summary']['total_breaking_changes']} breaking changes")
```

### Querying Breaking Changes

Query the breaking changes database:

```python
from src.data import get_breaking_changes, score_compatibility

# Get all PyTorch changes
pytorch_changes = get_breaking_changes(library="pytorch")

# Score compatibility for your environment
score = score_compatibility(
    detected_libraries=[{"name": "pytorch", "version": "2.1.0", "cuda_version": "12.1"}],
    cuda_version="13.0",
    compute_capability="8.0"
)

print(f"Compatibility Score: {score['compatibility_score']}/100")
print(f"Recommendation: {score['recommendation']}")
```

## 📁 Project Structure

```
cuda-healthcheck/
├── src/
│   ├── cuda_detector/       # Core CUDA detection logic
│   │   ├── __init__.py
│   │   └── detector.py      # GPU, driver, library detection
│   ├── databricks_api/      # Databricks integration
│   │   ├── __init__.py
│   │   └── cluster_scanner.py  # Cluster scanning and job submission
│   ├── data/                # Breaking changes database
│   │   ├── __init__.py
│   │   └── breaking_changes.py  # CUDA breaking changes catalog
│   └── healthcheck/         # Orchestration
│       ├── __init__.py
│       └── orchestrator.py  # Complete healthcheck workflow
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
├── .cursorrules            # Project guidelines
└── README.md               # This file
```

## 🔍 What Gets Detected

### CUDA Environment
- CUDA driver version (via nvidia-smi)
- CUDA runtime version (from /usr/local/cuda)
- NVCC compiler version
- GPU models and memory
- Compute capabilities

### Libraries
- **PyTorch**: Version, CUDA availability, CUDA version
- **TensorFlow**: Version, GPU availability, build info
- **cuDF/RAPIDS**: Version and CUDA compatibility

### Breaking Changes
- PyTorch CUDA 12.x → 13.x compatibility
- TensorFlow compute capability requirements
- cuDF/RAPIDS version matching
- NVIDIA container requirements (Isaac Sim, BioNeMo, Modulus)
- CuDNN API changes
- Deprecated compute capabilities

## 📊 Output Format

Results are returned as JSON-formatted dictionaries:

```json
{
  "healthcheck_id": "healthcheck-20241227-092300",
  "timestamp": "2024-12-27T09:23:00.000Z",
  "cuda_environment": {
    "cuda_runtime_version": "12.4",
    "cuda_driver_version": "12.4",
    "nvcc_version": "12.4",
    "gpus": [...]
  },
  "libraries": [...],
  "compatibility_analysis": {
    "compatibility_score": 70,
    "critical_issues": 0,
    "warning_issues": 2,
    "breaking_changes": {...},
    "recommendation": "ACCEPTABLE: Environment is mostly compatible. Review warnings."
  },
  "status": "healthy"
}
```

## 🔄 Databricks Workflow

### 1. Local Development

Use Databricks Connect to test on actual clusters:

```bash
pip install databricks-connect
databricks-connect configure
```

### 2. Cluster Scanning

The tool creates a temporary notebook on your Databricks workspace and runs it on each GPU cluster to collect CUDA information.

### 3. Delta Table Storage

Results are stored in Unity Catalog:

```sql
-- Query results
SELECT cluster_id, cuda_version, compatibility_score, timestamp
FROM main.cuda_healthcheck.healthcheck_results
ORDER BY timestamp DESC;

-- Find clusters with critical issues
SELECT cluster_id, breaking_changes
FROM main.cuda_healthcheck.healthcheck_results
WHERE array_size(filter(breaking_changes, x -> x.severity = 'CRITICAL')) > 0;
```

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

Run with coverage:

```bash
pytest --cov=src tests/
```

## 🚦 CI/CD

The project includes GitHub Actions workflows for:

- Automated testing on CUDA 12.4, 12.6, and 13.0
- Code quality checks (Black, Flake8, MyPy)
- Automated dependency updates via Renovate

## 📚 Supported CUDA Versions

- CUDA 12.4 ✅
- CUDA 12.6 ✅
- CUDA 13.0 ✅

## 🐛 Known Issues

- Databricks clusters must be in RUNNING state for scanning
- Delta table creation requires SQL warehouse permissions
- Some container-specific checks (Isaac Sim, BioNeMo) require specialized environments

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Add tests for new functionality
4. Run linters: `black src/ tests/ --line-length=100 && flake8 src/ tests/ --max-line-length=100`
5. Submit a PR

## 📄 License

[Your License Here]

## 📞 Support

For issues or questions:

- Create a GitHub issue
- Contact: [Your Contact Info]

## 🔗 References

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [Databricks GPU Clusters](https://docs.databricks.com/clusters/gpu.html)
- [PyTorch CUDA Support](https://pytorch.org/get-started/locally/)
- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [RAPIDS Documentation](https://docs.rapids.ai/)
