#!/usr/bin/env python3
"""
Comprehensive test runner for PHM-Vibench Model Factory

This script provides different testing modes and configurations
for comprehensive validation of the model factory.
"""

import argparse
import subprocess
import sys
import os
import time
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return the result."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Duration: {end_time - start_time:.2f} seconds")
    print(f"Return code: {result.returncode}")
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0


def run_smoke_tests():
    """Run quick smoke tests to validate basic functionality."""
    print("\n🚀 Running Smoke Tests")
    cmd = [
        "python", "-m", "pytest", 
        "test/test_model_factory.py::TestModelFactory::test_model_factory_import",
        "-v", "--tb=short"
    ]
    return run_command(cmd, "Smoke Tests - Basic Import")


def run_unit_tests():
    """Run all unit tests."""
    print("\n🧪 Running Unit Tests")
    cmd = [
        "python", "-m", "pytest",
        "test/test_model_factory.py",
        "test/test_utils.py",
        "-v", "--tb=short",
        "-m", "not slow and not gpu"
    ]
    return run_command(cmd, "Unit Tests - Core Functionality")


def run_integration_tests():
    """Run integration tests."""
    print("\n🔗 Running Integration Tests")
    cmd = [
        "python", "-m", "pytest",
        "test/test_integration.py",
        "-v", "--tb=short",
        "-m", "not slow"
    ]
    return run_command(cmd, "Integration Tests - End-to-End Workflows")


def run_performance_tests():
    """Run performance benchmarks."""
    print("\n⚡ Running Performance Tests")
    cmd = [
        "python", "-m", "pytest",
        "test/test_performance.py",
        "-v", "--tb=short",
        "-s"  # Don't capture output for performance results
    ]
    return run_command(cmd, "Performance Tests - Speed and Memory Benchmarks")


def run_gpu_tests():
    """Run GPU-specific tests."""
    print("\n🎮 Running GPU Tests")
    cmd = [
        "python", "-m", "pytest",
        "test/",
        "-v", "--tb=short",
        "-m", "gpu"
    ]
    return run_command(cmd, "GPU Tests - CUDA Functionality")


def run_coverage_tests():
    """Run tests with coverage reporting."""
    print("\n📊 Running Coverage Tests")
    cmd = [
        "python", "-m", "pytest",
        "test/",
        "--cov=src",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=70",
        "-v"
    ]
    return run_command(cmd, "Coverage Tests - Code Coverage Analysis")


def run_specific_model_tests(model_category):
    """Run tests for a specific model category."""
    print(f"\n🎯 Running {model_category.upper()} Model Tests")
    
    test_mapping = {
        'mlp': 'test/test_model_factory.py::TestMLPModels',
        'rnn': 'test/test_model_factory.py::TestRNNModels',
        'cnn': 'test/test_model_factory.py::TestCNNModels',
        'transformer': 'test/test_model_factory.py::TestTransformerModels',
        'no': 'test/test_model_factory.py::TestNeuralOperators',
        'isfm': 'test/test_model_factory.py::TestISFMModels'
    }
    
    if model_category not in test_mapping:
        print(f"Unknown model category: {model_category}")
        return False
    
    cmd = [
        "python", "-m", "pytest",
        test_mapping[model_category],
        "-v", "--tb=short"
    ]
    return run_command(cmd, f"{model_category.upper()} Model Tests")


def run_all_tests():
    """Run comprehensive test suite."""
    print("\n🎯 Running Complete Test Suite")
    
    results = []
    
    # 1. Smoke tests
    results.append(("Smoke Tests", run_smoke_tests()))
    
    # 2. Unit tests
    results.append(("Unit Tests", run_unit_tests()))
    
    # 3. Integration tests
    results.append(("Integration Tests", run_integration_tests()))
    
    # 4. Performance tests (optional, can be slow)
    if input("\nRun performance tests? (y/N): ").lower().startswith('y'):
        results.append(("Performance Tests", run_performance_tests()))
    
    # 5. GPU tests (if CUDA available)
    try:
        import torch
        if torch.cuda.is_available():
            if input("\nRun GPU tests? (y/N): ").lower().startswith('y'):
                results.append(("GPU Tests", run_gpu_tests()))
    except ImportError:
        pass
    
    # 6. Coverage tests
    if input("\nRun coverage analysis? (y/N): ").lower().startswith('y'):
        results.append(("Coverage Tests", run_coverage_tests()))
    
    return results


def print_summary(results):
    """Print test results summary."""
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    failed_tests = total_tests - passed_tests
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30} {status}")
    
    print("-" * 60)
    print(f"Total: {total_tests}, Passed: {passed_tests}, Failed: {failed_tests}")
    
    if failed_tests == 0:
        print("\n🎉 All tests passed! The model factory is working correctly.")
    else:
        print(f"\n⚠️  {failed_tests} test suite(s) failed. Please check the output above.")
    
    return failed_tests == 0


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="PHM-Vibench Model Factory Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --smoke          # Quick validation
  python run_tests.py --unit           # Unit tests only
  python run_tests.py --integration    # Integration tests
  python run_tests.py --performance    # Performance benchmarks
  python run_tests.py --coverage       # Coverage analysis
  python run_tests.py --model mlp      # Test specific model category
  python run_tests.py --all            # Complete test suite
        """
    )
    
    parser.add_argument('--smoke', action='store_true', 
                       help='Run smoke tests (quick validation)')
    parser.add_argument('--unit', action='store_true',
                       help='Run unit tests')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance benchmarks')
    parser.add_argument('--gpu', action='store_true',
                       help='Run GPU-specific tests')
    parser.add_argument('--coverage', action='store_true',
                       help='Run tests with coverage analysis')
    parser.add_argument('--model', choices=['mlp', 'rnn', 'cnn', 'transformer', 'no', 'isfm'],
                       help='Test specific model category')
    parser.add_argument('--all', action='store_true',
                       help='Run complete test suite')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Check if pytest is available
    try:
        import pytest
    except ImportError:
        print("❌ pytest is not installed. Please install it with: pip install pytest")
        return 1
    
    # Check if we're in the right directory
    if not Path("test").exists():
        print("❌ Test directory not found. Please run from the project root directory.")
        return 1
    
    print("PHM-Vibench Model Factory Test Runner")
    print("=" * 60)
    
    results = []
    
    if args.smoke:
        results.append(("Smoke Tests", run_smoke_tests()))
    elif args.unit:
        results.append(("Unit Tests", run_unit_tests()))
    elif args.integration:
        results.append(("Integration Tests", run_integration_tests()))
    elif args.performance:
        results.append(("Performance Tests", run_performance_tests()))
    elif args.gpu:
        results.append(("GPU Tests", run_gpu_tests()))
    elif args.coverage:
        results.append(("Coverage Tests", run_coverage_tests()))
    elif args.model:
        results.append((f"{args.model.upper()} Tests", run_specific_model_tests(args.model)))
    elif args.all:
        results = run_all_tests()
    else:
        # Default: run smoke tests
        print("No specific test type specified. Running smoke tests...")
        results.append(("Smoke Tests", run_smoke_tests()))
    
    # Print summary
    success = print_summary(results)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
