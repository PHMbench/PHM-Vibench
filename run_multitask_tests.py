#!/usr/bin/env python3
"""
Comprehensive test runner for multi-task PHM bug fixes.

This script runs all test suites created to validate the OOM fixes
and multi-task implementation improvements.

Author: PHM-Vibench Team  
Date: 2025-09-08
"""

import sys
import subprocess
import time
from pathlib import Path


def run_test_suite(test_file, description):
    """Run a single test suite and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"File: {test_file}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, test_file, '-v'
        ], capture_output=True, text=True, timeout=300)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            status = "✅ PASSED"
            print(f"{status} ({duration:.1f}s)")
            # Print any important output
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'OK' in line or 'passed' in line or any(keyword in line.lower() for keyword in ['memory', 'parameter', 'reduction']):
                    print(f"  {line}")
            return True, duration, result.stdout
        else:
            status = "❌ FAILED"
            print(f"{status} ({duration:.1f}s)")
            # Print error summary
            if result.stderr:
                print("STDERR:")
                print(result.stderr[-1000:])  # Last 1000 chars
            if result.stdout:
                # Show failed tests
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'FAILED' in line or 'ERROR' in line or 'AssertionError' in line:
                        print(f"  {line}")
            return False, duration, result.stdout
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ TIMEOUT ({duration:.1f}s)")
        return False, duration, "Test timed out"
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 ERROR ({duration:.1f}s): {e}")
        return False, duration, str(e)


def main():
    """Run all test suites and provide comprehensive summary."""
    print("PHM-Vibench Multi-Task Bug Fix Test Suite")
    print("=========================================")
    print("Validating OOM fixes and multi-task improvements\n")
    
    # Define test suites in order of importance
    test_suites = [
        ("test/test_multi_task_rul_validation.py", "RUL Label Validation (NaN Prevention)"),
        ("test/test_regression_metrics.py", "Regression Metrics Extensions"), 
        ("test/test_task_specific_metrics.py", "Task-Specific Metrics Integration"),
        ("test/test_parameter_consistency.py", "Parameter Consistency & OOM Prevention"),
        ("test/test_end_to_end_integration.py", "End-to-End Integration"),
        ("test/test_regression_backward_compatibility.py", "Backward Compatibility"),
    ]
    
    # Track results
    total_tests = len(test_suites)
    passed_tests = 0
    total_duration = 0
    results = []
    
    start_time = time.time()
    
    # Run each test suite
    for test_file, description in test_suites:
        if not Path(test_file).exists():
            print(f"\n⚠️  SKIPPED: {description}")
            print(f"   File not found: {test_file}")
            continue
            
        success, duration, output = run_test_suite(test_file, description)
        total_duration += duration
        results.append((description, success, duration, output))
        
        if success:
            passed_tests += 1
    
    # Generate comprehensive summary
    print(f"\n{'='*60}")
    print("COMPREHENSIVE TEST SUMMARY")
    print('='*60)
    
    overall_success = passed_tests == total_tests
    overall_status = "✅ ALL PASSED" if overall_success else f"❌ {passed_tests}/{total_tests} PASSED"
    
    print(f"Overall Status: {overall_status}")
    print(f"Total Duration: {total_duration:.1f}s")
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    
    print(f"\nDetailed Results:")
    print("-" * 60)
    
    for description, success, duration, output in results:
        status_icon = "✅" if success else "❌"
        print(f"{status_icon} {description:<40} ({duration:.1f}s)")
    
    # Extract key metrics from outputs
    print(f"\nKey Validation Results:")
    print("-" * 60)
    
    for description, success, duration, output in results:
        if not success:
            continue
            
        # Extract important metrics
        if "Parameter Consistency" in description:
            # Extract memory and parameter info
            lines = output.split('\n')
            for line in lines:
                if 'memory' in line.lower() and 'gb' in line.lower():
                    print(f"  Memory Optimization: {line.strip()}")
                elif 'parameter' in line.lower() and ('reduction' in line.lower() or 'total' in line.lower()):
                    print(f"  {line.strip()}")
        
        elif "Backward Compatibility" in description:
            # Extract compatibility results
            lines = output.split('\n')
            for line in lines:
                if 'reduction' in line.lower() and '%' in line:
                    print(f"  OOM Fix Effectiveness: {line.strip()}")
    
    # Check for any critical issues
    print(f"\nCritical Issue Analysis:")
    print("-" * 60)
    
    critical_issues = []
    
    for description, success, duration, output in results:
        if not success:
            critical_issues.append(f"FAILED: {description}")
        elif 'OOM' in output or 'memory' in output.lower():
            # Check for memory-related warnings
            lines = output.split('\n')
            for line in lines:
                if 'warning' in line.lower() and 'memory' in line.lower():
                    critical_issues.append(f"WARNING in {description}: {line.strip()}")
    
    if critical_issues:
        print("⚠️  Issues Found:")
        for issue in critical_issues:
            print(f"  - {issue}")
    else:
        print("✅ No critical issues detected")
    
    # Provide final recommendation
    print(f"\nFinal Recommendation:")
    print("-" * 60)
    
    if overall_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Multi-task OOM fixes are working correctly")
        print("✅ Backward compatibility maintained")
        print("✅ All bug fixes validated successfully")
        print("\n🚀 Ready for production use!")
    else:
        print("⚠️  Some tests failed - review issues above")
        print(f"✅ {passed_tests}/{total_tests} test suites passed")
        
        if passed_tests >= total_tests * 0.8:  # 80% pass rate
            print("📊 Overall quality is good despite some failures")
            print("🔧 Minor fixes may be needed for full compliance")
        else:
            print("🔴 Significant issues detected - major fixes needed")
    
    # Return appropriate exit code
    sys.exit(0 if overall_success else 1)


if __name__ == '__main__':
    main()