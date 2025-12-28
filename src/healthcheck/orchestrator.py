"""
Healthcheck Orchestrator.

Coordinates CUDA detection, breaking change analysis, and reporting.
"""

import json
from typing import Dict, Any
from datetime import datetime

from ..cuda_detector.detector import CUDADetector
from ..data.breaking_changes import BreakingChangesDatabase


def run_complete_healthcheck() -> Dict[str, Any]:
    """
    Run a complete CUDA healthcheck with breaking change analysis.

    Returns:
        Dictionary with complete healthcheck results including:
        - CUDA environment details
        - Library compatibility
        - Breaking changes
        - Compatibility score
        - Recommendations
    """
    # Step 1: Detect CUDA environment
    detector = CUDADetector()
    environment = detector.detect_environment()

    # Step 2: Analyze breaking changes
    db = BreakingChangesDatabase()

    # Extract compute capability from first GPU if available
    compute_capability = None
    if environment.gpus:
        compute_capability = environment.gpus[0].compute_capability

    # Score compatibility
    compatibility = db.score_compatibility(
        detected_libraries=[
            detector.to_dict(environment)["libraries"][i] for i in range(len(environment.libraries))
        ],
        cuda_version=environment.cuda_driver_version or "Unknown",
        compute_capability=compute_capability,
    )

    # Step 3: Combine results
    result = {
        "healthcheck_id": f"healthcheck-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        "timestamp": environment.timestamp,
        "cuda_environment": {
            "cuda_runtime_version": environment.cuda_runtime_version,
            "cuda_driver_version": environment.cuda_driver_version,
            "nvcc_version": environment.nvcc_version,
            "gpus": [
                {
                    "name": gpu.name,
                    "driver_version": gpu.driver_version,
                    "cuda_version": gpu.cuda_version,
                    "compute_capability": gpu.compute_capability,
                    "memory_total_mb": gpu.memory_total_mb,
                    "gpu_index": gpu.gpu_index,
                }
                for gpu in environment.gpus
            ],
        },
        "libraries": [
            {
                "name": lib.name,
                "version": lib.version,
                "cuda_version": lib.cuda_version,
                "is_compatible": lib.is_compatible,
                "warnings": lib.warnings,
            }
            for lib in environment.libraries
        ],
        "compatibility_analysis": compatibility,
        "status": "healthy" if compatibility["critical_issues"] == 0 else "unhealthy",
    }

    return result


if __name__ == "__main__":
    # Run healthcheck and print results
    results = run_complete_healthcheck()
    print(json.dumps(results, indent=2))
