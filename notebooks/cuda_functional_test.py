#!/usr/bin/env python3
"""
CUDA Functional Testing Script

Comprehensive CUDA functional tests for GPU validation.
Tests actual CUDA operations beyond availability checks.

Usage:
    python cuda_functional_test.py
    python cuda_functional_test.py --verbose
    python cuda_functional_test.py --json output.json

Requirements:
    - PyTorch with CUDA support
    - GPU-enabled system

Exit Codes:
    0: All tests passed
    1: Some tests failed
    2: CUDA not available
    3: PyTorch not installed
"""

import sys
import json
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple


def check_cuda_available() -> Tuple[bool, str]:
    """Check if CUDA is available via PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            return True, f"{device_count} GPU(s) detected: {device_name}"
        else:
            return False, "CUDA not available via torch.cuda.is_available()"
    except ImportError:
        return False, "PyTorch not installed"


def test_tensor_creation(device, verbose: bool = False) -> Dict[str, Any]:
    """Test 1: CUDA tensor creation."""
    import torch
    
    result = {
        "test_name": "Tensor Creation",
        "passed": False,
        "timings": [],
        "error": None
    }
    
    try:
        tensor_sizes = [(1000, 1000), (2000, 2000), (4000, 4000)]
        
        for size in tensor_sizes:
            start_time = time.time()
            tensor = torch.randn(size, device=device)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            result["timings"].append({
                "size": f"{size[0]}x{size[1]}",
                "time_ms": round(elapsed * 1000, 2)
            })
            
            if verbose:
                print(f"      Created {size[0]}×{size[1]} tensor in {elapsed*1000:.2f}ms")
            
            del tensor
        
        result["passed"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def test_matrix_multiplication(device, verbose: bool = False) -> Dict[str, Any]:
    """Test 2: Matrix multiplication performance (GFLOPS)."""
    import torch
    
    result = {
        "test_name": "Matrix Multiplication",
        "passed": False,
        "gflops": 0.0,
        "avg_time_ms": 0.0,
        "error": None
    }
    
    try:
        matrix_size = 4096
        num_iterations = 10
        
        if verbose:
            print(f"      Running {num_iterations} iterations of {matrix_size}×{matrix_size} matmul...")
        
        A = torch.randn(matrix_size, matrix_size, device=device)
        B = torch.randn(matrix_size, matrix_size, device=device)
        
        # Warm-up
        for _ in range(5):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        # Calculate GFLOPS
        flops_per_matmul = 2 * (matrix_size ** 3)
        total_flops = flops_per_matmul * num_iterations
        gflops = (total_flops / elapsed) / 1e9
        
        result["gflops"] = round(gflops, 2)
        result["avg_time_ms"] = round((elapsed / num_iterations) * 1000, 2)
        result["passed"] = True
        
        if verbose:
            print(f"      Performance: {gflops:.2f} GFLOPS")
            print(f"      Avg time per matmul: {result['avg_time_ms']}ms")
        
        del A, B, C
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def test_memory_management(device, verbose: bool = False) -> Dict[str, Any]:
    """Test 3: CUDA memory allocation and tracking."""
    import torch
    
    result = {
        "test_name": "Memory Management",
        "passed": False,
        "peak_memory_mb": 0.0,
        "memory_freed_mb": 0.0,
        "error": None
    }
    
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        initial_memory = torch.cuda.memory_allocated(0)
        if verbose:
            print(f"      Initial memory: {initial_memory / 1024**2:.2f} MB")
        
        # Allocate tensors
        tensors = []
        allocation_sizes_mb = [100, 200, 500]
        
        for size_mb in allocation_sizes_mb:
            num_elements = (size_mb * 1024 * 1024) // 4
            tensor = torch.randn(num_elements, device=device)
            tensors.append(tensor)
            
            if verbose:
                current_memory = torch.cuda.memory_allocated(0)
                print(f"      Allocated {size_mb}MB → Total: {current_memory / 1024**2:.2f} MB")
        
        peak_memory = torch.cuda.max_memory_allocated(0)
        result["peak_memory_mb"] = round(peak_memory / 1024**2, 2)
        
        # Free memory
        del tensors
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated(0)
        memory_freed = (peak_memory - final_memory) / 1024**2
        result["memory_freed_mb"] = round(memory_freed, 2)
        
        # Verify memory was freed
        if memory_freed > 700:
            result["passed"] = True
        
        if verbose:
            print(f"      Peak memory: {result['peak_memory_mb']} MB")
            print(f"      Memory freed: {result['memory_freed_mb']} MB")
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def test_cuda_streams(device, verbose: bool = False) -> Dict[str, Any]:
    """Test 4: CUDA stream operations."""
    import torch
    
    result = {
        "test_name": "CUDA Streams",
        "passed": False,
        "num_streams": 0,
        "sync_time_ms": 0.0,
        "error": None
    }
    
    try:
        num_streams = 4
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        
        result["num_streams"] = num_streams
        
        if verbose:
            print(f"      Created {num_streams} CUDA streams")
        
        # Launch operations on different streams
        results_list = []
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                A = torch.randn(2000, 2000, device=device)
                B = torch.randn(2000, 2000, device=device)
                C = torch.matmul(A, B)
                results_list.append(C)
        
        # Synchronize all streams
        start_time = time.time()
        for stream in streams:
            stream.synchronize()
        sync_time = time.time() - start_time
        
        result["sync_time_ms"] = round(sync_time * 1000, 2)
        result["passed"] = True
        
        if verbose:
            print(f"      {num_streams} concurrent operations completed")
            print(f"      Stream synchronization: {result['sync_time_ms']}ms")
        
        del results_list
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def test_mixed_precision(device, gpu_capability, verbose: bool = False) -> Dict[str, Any]:
    """Test 5: Mixed precision support (FP16, BF16, TF32)."""
    import torch
    
    result = {
        "test_name": "Mixed Precision",
        "passed": False,
        "float16": False,
        "bfloat16": False,
        "tf32": False,
        "error": None
    }
    
    try:
        # Test FP16
        if verbose:
            print("      Testing float16 (FP16)...")
        try:
            A_fp16 = torch.randn(1000, 1000, device=device, dtype=torch.float16)
            B_fp16 = torch.randn(1000, 1000, device=device, dtype=torch.float16)
            C_fp16 = torch.matmul(A_fp16, B_fp16)
            torch.cuda.synchronize()
            
            result["float16"] = True
            if verbose:
                print(f"         ✅ float16 (FP16): Supported")
            
            del A_fp16, B_fp16, C_fp16
        except Exception as e:
            if verbose:
                print(f"         ❌ float16 (FP16): Not supported - {str(e)}")
        
        # Test BF16
        if verbose:
            print("      Testing bfloat16 (BF16)...")
        if gpu_capability[0] >= 8:
            try:
                A_bf16 = torch.randn(1000, 1000, device=device, dtype=torch.bfloat16)
                B_bf16 = torch.randn(1000, 1000, device=device, dtype=torch.bfloat16)
                C_bf16 = torch.matmul(A_bf16, B_bf16)
                torch.cuda.synchronize()
                
                result["bfloat16"] = True
                if verbose:
                    print(f"         ✅ bfloat16 (BF16): Supported")
                
                del A_bf16, B_bf16, C_bf16
            except Exception as e:
                if verbose:
                    print(f"         ❌ bfloat16 (BF16): Not supported - {str(e)}")
        else:
            if verbose:
                print(f"         ⚠️  bfloat16 (BF16): Requires Ampere+ GPU")
        
        # Test TF32
        if verbose:
            print("      Testing TensorFloat-32 (TF32)...")
        if gpu_capability[0] >= 8:
            tf32_enabled = torch.backends.cuda.matmul.allow_tf32
            result["tf32"] = tf32_enabled
            
            if verbose:
                if tf32_enabled:
                    print(f"         ✅ TensorFloat-32 (TF32): Enabled")
                else:
                    print(f"         ⚠️  TensorFloat-32 (TF32): Available but disabled")
        else:
            if verbose:
                print(f"         ⚠️  TensorFloat-32 (TF32): Requires Ampere+ GPU")
        
        result["passed"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def test_cublas_gemm(device, verbose: bool = False) -> Dict[str, Any]:
    """Test 8: cuBLAS GEMM performance."""
    import torch
    
    result = {
        "test_name": "cuBLAS GEMM",
        "passed": False,
        "benchmarks": {},
        "error": None
    }
    
    try:
        matrix_sizes = [1024, 2048, 4096, 8192]
        
        if verbose:
            print(f"      Running cuBLAS GEMM benchmarks...")
        
        for size in matrix_sizes:
            A = torch.randn(size, size, device=device)
            B = torch.randn(size, size, device=device)
            
            # Warm-up
            for _ in range(3):
                C = torch.mm(A, B)
            torch.cuda.synchronize()
            
            # Benchmark
            iterations = 10
            start_time = time.time()
            for _ in range(iterations):
                C = torch.mm(A, B)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            # Calculate GFLOPS
            flops = 2 * (size ** 3) * iterations
            gflops = (flops / elapsed) / 1e9
            
            result["benchmarks"][f"{size}x{size}"] = round(gflops, 2)
            
            if verbose:
                print(f"         {size}×{size}: {gflops:.2f} GFLOPS")
            
            del A, B, C
        
        result["passed"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def test_memory_bandwidth(device, verbose: bool = False) -> Dict[str, Any]:
    """Test 12: GPU memory bandwidth."""
    import torch
    
    result = {
        "test_name": "Memory Bandwidth",
        "passed": False,
        "bandwidth_gb_s": 0.0,
        "error": None
    }
    
    try:
        size_mb = 512
        num_elements = (size_mb * 1024 * 1024) // 4
        iterations = 20
        
        if verbose:
            print(f"      Testing memory bandwidth with {size_mb}MB transfers...")
        
        A = torch.randn(num_elements, device=device)
        B = torch.empty_like(A)
        
        # Warm-up
        for _ in range(5):
            B.copy_(A)
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            B.copy_(A)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        # Calculate bandwidth
        bytes_transferred = size_mb * 1024 * 1024 * iterations
        bandwidth_gb_s = (bytes_transferred / elapsed) / 1e9
        
        result["bandwidth_gb_s"] = round(bandwidth_gb_s, 2)
        result["passed"] = True
        
        if verbose:
            print(f"      Memory bandwidth: {bandwidth_gb_s:.2f} GB/s")
        
        del A, B
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def run_all_tests(verbose: bool = False) -> Dict[str, Any]:
    """Run all CUDA functional tests."""
    import torch
    
    print("=" * 80)
    print("🔥 CUDA FUNCTIONAL TESTING")
    print("=" * 80)
    
    # Initialize results
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cuda_available": False,
        "device_info": {},
        "tests": [],
        "summary": {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0
        }
    }
    
    # Check CUDA availability
    cuda_available, cuda_message = check_cuda_available()
    results["cuda_available"] = cuda_available
    
    if not cuda_available:
        print(f"\n❌ {cuda_message}")
        print("=" * 80)
        return results
    
    print(f"\n✅ {cuda_message}")
    
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_capability = torch.cuda.get_device_capability(0)
    
    results["device_info"] = {
        "name": gpu_name,
        "compute_capability": f"{gpu_capability[0]}.{gpu_capability[1]}",
        "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    }
    
    print(f"🎮 GPU: {gpu_name}")
    print(f"   Compute Capability: {gpu_capability[0]}.{gpu_capability[1]}")
    print(f"   Total Memory: {results['device_info']['memory_total_gb']} GB")
    
    # Define tests to run
    tests_to_run = [
        ("TEST 1", test_tensor_creation, (device, verbose)),
        ("TEST 2", test_matrix_multiplication, (device, verbose)),
        ("TEST 3", test_memory_management, (device, verbose)),
        ("TEST 4", test_cuda_streams, (device, verbose)),
        ("TEST 5", test_mixed_precision, (device, gpu_capability, verbose)),
        ("TEST 8", test_cublas_gemm, (device, verbose)),
        ("TEST 12", test_memory_bandwidth, (device, verbose)),
    ]
    
    # Run tests
    for test_id, test_func, test_args in tests_to_run:
        print(f"\n{'─' * 80}")
        print(f"{test_id}: {test_func.__doc__.split(':')[1].strip()}")
        print(f"{'─' * 80}")
        
        test_result = test_func(*test_args)
        results["tests"].append(test_result)
        
        if test_result["passed"]:
            results["summary"]["passed"] += 1
            print(f"   ✅ Status: PASSED")
        else:
            results["summary"]["failed"] += 1
            print(f"   ❌ Status: FAILED")
            if test_result.get("error"):
                print(f"   Error: {test_result['error']}")
    
    results["summary"]["total_tests"] = len(tests_to_run)
    
    # Final summary
    print("\n" + "=" * 80)
    print(f"CUDA FUNCTIONAL TEST SUMMARY")
    print("=" * 80)
    print(f"   Total Tests: {results['summary']['total_tests']}")
    print(f"   ✅ Passed: {results['summary']['passed']}")
    print(f"   ❌ Failed: {results['summary']['failed']}")
    print(f"   ⏭️  Skipped: {results['summary']['skipped']}")
    
    if results["summary"]["failed"] == 0:
        print(f"\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  SOME TESTS FAILED")
    
    print("=" * 80)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CUDA Functional Testing Script")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-j", "--json", type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Check PyTorch installation
    try:
        import torch
    except ImportError:
        print("❌ PyTorch not installed")
        print("   Install: pip install torch")
        return 3
    
    # Run tests
    results = run_all_tests(verbose=args.verbose)
    
    # Save JSON if requested
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {args.json}")
    
    # Determine exit code
    if not results["cuda_available"]:
        return 2
    elif results["summary"]["failed"] > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())

