#!/usr/bin/env python3
"""
Comprehensive test runner for ContrastiveIDTask integration tests
Orchestrates all integration test scenarios with proper reporting
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def run_test_suite(test_file, markers=None, verbose=True, capture_output=True):
    """Run a specific test suite with customizable options"""
    cmd = ["python", "-m", "pytest", str(test_file)]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    if markers:
        cmd.extend(["-m", markers])
    
    cmd.extend(["--tb=short", "--durations=10"])
    
    print(f"🧪 Running: {' '.join(cmd)}")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
            duration = time.time() - start_time
            
            return {
                'success': result.returncode == 0,
                'duration': duration,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        else:
            result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
            duration = time.time() - start_time
            
            return {
                'success': result.returncode == 0,
                'duration': duration,
                'returncode': result.returncode
            }
    except Exception as e:
        duration = time.time() - start_time
        return {
            'success': False,
            'duration': duration,
            'error': str(e),
            'returncode': -1
        }


def print_test_result(test_name, result):
    """Print formatted test result"""
    status = "✅ PASSED" if result['success'] else "❌ FAILED"
    duration_str = f"{result['duration']:.1f}s"
    
    print(f"\n{status} {test_name} ({duration_str})")
    
    if not result['success']:
        if 'error' in result:
            print(f"Error: {result['error']}")
        if 'stderr' in result and result['stderr']:
            print("STDERR:")
            print(result['stderr'])
        if 'stdout' in result and result['stdout']:
            print("STDOUT:")
            print(result['stdout'][-1000:])  # Last 1000 chars to avoid spam
    
    print("-" * 60)


def check_environment():
    """Check if the test environment is ready"""
    print("🔍 Checking test environment...")
    
    checks = []
    
    # Check Python version
    if sys.version_info >= (3, 8):
        checks.append(("Python version", True, f"{sys.version_info.major}.{sys.version_info.minor}"))
    else:
        checks.append(("Python version", False, f"{sys.version_info.major}.{sys.version_info.minor} < 3.8"))
    
    # Check required packages
    required_packages = ['torch', 'numpy', 'pytest', 'pandas', 'h5py', 'yaml', 'psutil']
    
    for package in required_packages:
        try:
            __import__(package)
            checks.append((f"Package {package}", True, "Available"))
        except ImportError:
            checks.append((f"Package {package}", False, "Missing"))
    
    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None"
        checks.append(("CUDA support", cuda_available, gpu_name))
    except Exception:
        checks.append(("CUDA support", False, "Error checking"))
    
    # Check project structure
    project_root = Path(__file__).parent.parent.parent
    required_paths = [
        'src/task_factory/task/pretrain/ContrastiveIDTask.py',
        'src/configs/config_utils.py',
        'configs/id_contrastive/debug.yaml'
    ]
    
    for path in required_paths:
        full_path = project_root / path
        checks.append((f"Path {path}", full_path.exists(), str(full_path)))
    
    # Print results
    for check_name, passed, detail in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}: {detail}")
    
    # Return overall status
    all_critical_passed = all(passed for name, passed, _ in checks if name.startswith(("Python", "Package torch", "Package pytest", "Package numpy")))
    
    if all_critical_passed:
        print("✅ Environment check passed")
    else:
        print("❌ Environment check failed - some critical components missing")
    
    return all_critical_passed


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description="Run ContrastiveIDTask integration tests")
    parser.add_argument("--quick", action="store_true", help="Run only quick tests")
    parser.add_argument("--gpu", action="store_true", help="Include GPU tests")
    parser.add_argument("--performance", action="store_true", help="Include performance tests")
    parser.add_argument("--real-data", action="store_true", help="Include real data tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--capture", action="store_true", help="Capture test output")
    parser.add_argument("--report", help="Save detailed report to file")
    
    args = parser.parse_args()
    
    print("🚀 ContrastiveIDTask Integration Test Suite")
    print("=" * 70)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please install missing dependencies.")
        return 1
    
    print("\n🧪 Starting integration tests...")
    
    # Define test suites
    test_suites = []
    
    # Always run basic integration tests
    test_suites.append({
        'name': 'Basic Integration Tests',
        'file': 'test_contrastive_full_training.py',
        'markers': 'integration and not performance and not gpu'
    })
    
    # Add additional test suites based on arguments
    if args.gpu or args.all:
        import torch
        if torch.cuda.is_available():
            test_suites.append({
                'name': 'GPU Integration Tests',
                'file': 'test_contrastive_full_training.py',
                'markers': 'integration and gpu'
            })
        else:
            print("⚠️ GPU tests requested but CUDA not available")
    
    if args.performance or args.all:
        test_suites.append({
            'name': 'Performance Tests',
            'file': 'test_contrastive_full_training.py',
            'markers': 'integration and performance'
        })
    
    if args.real_data or args.all:
        test_suites.append({
            'name': 'Real Data Tests',
            'file': 'test_contrastive_real_data.py',
            'markers': 'integration'
        })
    
    if args.quick:
        # Override with only quick tests
        test_suites = [{
            'name': 'Quick Integration Tests',
            'file': 'test_contrastive_full_training.py', 
            'markers': 'integration and not slow and not performance'
        }]
    
    # Run test suites
    results = {}
    total_duration = 0
    failed_count = 0
    
    for suite in test_suites:
        print(f"\n📋 Running {suite['name']}")
        print("=" * 40)
        
        test_file = Path(__file__).parent / suite['file']
        if not test_file.exists():
            print(f"❌ Test file not found: {test_file}")
            continue
        
        result = run_test_suite(
            test_file, 
            markers=suite['markers'],
            capture_output=args.capture
        )
        
        results[suite['name']] = result
        total_duration += result['duration']
        
        if not result['success']:
            failed_count += 1
        
        print_test_result(suite['name'], result)
    
    # Print summary
    print("\n📊 Test Summary")
    print("=" * 40)
    print(f"Total suites run: {len(results)}")
    print(f"Passed: {len(results) - failed_count}")
    print(f"Failed: {failed_count}")
    print(f"Total duration: {total_duration:.1f}s")
    
    # Detailed results
    print("\n📝 Detailed Results:")
    for suite_name, result in results.items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {status} {suite_name}: {result['duration']:.1f}s")
    
    # Save detailed report if requested
    if args.report:
        report_data = {
            'timestamp': time.time(),
            'environment': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
                'cuda_available': torch.cuda.is_available() if 'torch' in sys.modules else False
            },
            'results': results,
            'summary': {
                'total_suites': len(results),
                'passed': len(results) - failed_count,
                'failed': failed_count,
                'total_duration': total_duration
            }
        }
        
        with open(args.report, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {args.report}")
    
    # Next steps recommendations
    if failed_count == 0:
        print("\n🎉 All integration tests passed! Next steps:")
        print("  1. Run with real dataset:")
        print("     python main.py --pipeline Pipeline_ID --config contrastive")
        print("  2. Try production training:")
        print("     python main.py --pipeline Pipeline_ID --config contrastive_prod")
        print("  3. Run ablation studies:")
        print("     python scripts/ablation_studies.py --preset contrastive_ablation")
    else:
        print(f"\n⚠️ {failed_count} test suite(s) failed. Please check the errors above.")
        print("  Common issues:")
        print("  - Missing dependencies (install requirements-test.txt)")
        print("  - CUDA/GPU issues (run CPU-only tests with --quick)")
        print("  - Data path issues (check config file paths)")
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    exit(main())