#!/usr/bin/env python3
"""
1-Epoch Quick Test Script for Unified Metric Learning Pipeline

This script provides a simple interface to test the unified metric learning
pipeline with 1 epoch training to verify everything works correctly before
running the full experiment.

Expected runtime: 5-10 minutes
Expected result: Pipeline validation with basic functionality check

Usage:
    python test_1epoch.py                    # Quick health check + 1-epoch test
    python test_1epoch.py --health-only      # Health check only
    python test_1epoch.py --verbose          # Detailed output
    python test_1epoch.py --config <path>    # Custom config file

Author: PHM-Vibench Team
Date: 2025-01-15
"""

import argparse
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import yaml

def print_header():
    """Print test header."""
    print("=" * 80)
    print("🚀 PHM-Vibench Unified Metric Learning - 1-Epoch Test")
    print("=" * 80)
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🎯 Purpose: Validate pipeline functionality with 1-epoch training")
    print("⏱️  Expected runtime: 5-10 minutes")
    print("📊 Testing: All 5 datasets (CWRU, XJTU, THU, Ottawa, JNU)")
    print("=" * 80)

def print_stage(stage_name: str, description: str):
    """Print stage header."""
    print(f"\n🔄 Stage: {stage_name}")
    print(f"📝 {description}")
    print("-" * 60)

def run_command(cmd: list, stage_name: str, timeout: int = 600):
    """Run a command and handle output."""
    print(f"▶️  Running: {' '.join(cmd)}")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(Path(__file__).parent.parent.parent.absolute())  # PHM-Vibench-metric root
        )

        end_time = time.time()
        duration = end_time - start_time

        if result.returncode == 0:
            print(f"✅ {stage_name} completed successfully ({duration:.1f}s)")
            return True, result.stdout, result.stderr
        else:
            print(f"❌ {stage_name} failed ({duration:.1f}s)")
            print(f"Error output: {result.stderr}")
            return False, result.stdout, result.stderr

    except subprocess.TimeoutExpired:
        print(f"⏰ {stage_name} timed out after {timeout}s")
        return False, "", f"Timeout after {timeout}s"
    except Exception as e:
        print(f"💥 {stage_name} crashed: {e}")
        return False, "", str(e)

def check_config_file(config_path: Path) -> bool:
    """Check if config file exists and is valid."""
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("💡 Please ensure the config file exists.")
        return False

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check critical settings
        if 'data' not in config:
            print("❌ Missing 'data' section in config")
            return False

        data_dir = Path(config['data']['data_dir'])
        metadata_file = config['data']['metadata_file']
        metadata_path = data_dir / metadata_file

        if not metadata_path.exists():
            print(f"❌ Metadata file not found: {metadata_path}")
            print("💡 Please update data_dir in config file to point to your actual data directory.")
            print(f"💡 Looking for: {metadata_file}")
            return False

        print(f"✅ Configuration valid: {config_path}")
        print(f"✅ Data directory: {data_dir}")
        print(f"✅ Metadata file: {metadata_path}")
        return True

    except Exception as e:
        print(f"❌ Configuration file error: {e}")
        return False

def run_health_check(config_path: str = None) -> bool:
    """Run system health check."""
    print_stage("Health Check", "Verifying system requirements and configuration")

    cmd = [
        sys.executable,
        "script/unified_metric/pipeline/quick_validate.py",
        "--mode", "health_check"
    ]

    if config_path:
        cmd.extend(["--config", config_path])

    success, stdout, stderr = run_command(cmd, "Health Check", timeout=120)

    if success:
        print("📋 Health Check Results:")
        # Parse and display key results
        lines = stdout.split('\n')
        for line in lines:
            if 'System ready' in line or 'GPU:' in line or 'Memory:' in line or 'datasets detected' in line:
                print(f"   {line}")

    return success

def run_1epoch_validation() -> bool:
    """Run 1-epoch validation test."""
    print_stage("1-Epoch Validation", "Testing pipeline with 1-epoch training")

    cmd = [
        sys.executable,
        "script/unified_metric/pipeline/quick_validate.py",
        "--mode", "full_validation",
        "--config", "script/unified_metric/configs/unified_experiments_1epoch.yaml"
    ]

    success, stdout, stderr = run_command(cmd, "1-Epoch Validation", timeout=900)  # 15 minutes

    if success:
        print("📊 1-Epoch Test Results:")
        # Parse and display key results
        lines = stdout.split('\n')
        for line in lines:
            if 'PASS' in line or 'accuracy' in line or 'completed' in line:
                print(f"   {line}")

    return success

def run_quick_pipeline_test() -> bool:
    """Run a quick pipeline test with the actual training."""
    print_stage("Pipeline Test", "Running actual 1-epoch unified pretraining")

    # Use the validation script's pipeline test which does actual 1-epoch training
    cmd = [
        sys.executable,
        "-c",
        """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from script.unified_metric.pipeline.quick_validate import UnifiedPipelineValidator
validator = UnifiedPipelineValidator('script/unified_metric/configs/unified_experiments_1epoch.yaml')
validator.test_pipeline_1_epoch()
print('✅ Pipeline test completed')
"""
    ]

    success, stdout, stderr = run_command(cmd, "Pipeline Test", timeout=600)  # 10 minutes

    return success

def print_summary(health_passed: bool, validation_passed: bool, pipeline_passed: bool):
    """Print test summary."""
    print("\n" + "=" * 80)
    print("📋 TEST SUMMARY")
    print("=" * 80)

    print(f"🔍 Health Check:      {'✅ PASS' if health_passed else '❌ FAIL'}")
    print(f"🧪 1-Epoch Validation: {'✅ PASS' if validation_passed else '❌ FAIL'}")
    print(f"🚀 Pipeline Test:     {'✅ PASS' if pipeline_passed else '❌ FAIL'}")

    overall_success = health_passed and validation_passed and pipeline_passed

    print(f"\n🎯 OVERALL RESULT: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")

    if overall_success:
        print("\n🎉 Congratulations! Your pipeline is ready for full training.")
        print("🚀 Next steps:")
        print("   1. Run full experiment: python script/unified_metric/pipeline/run_unified_experiments.py")
        print("   2. Monitor progress: tail -f results/unified_metric_learning/logs/*.log")
        print("   3. Expected full training time: ~22 hours")
    else:
        print("\n🔧 Some tests failed. Please check the output above and:")
        print("   1. Verify data directory path in config file")
        print("   2. Check GPU memory availability")
        print("   3. Ensure all dependencies are installed")
        print("   4. Check log files for detailed error messages")

    print("=" * 80)

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="1-Epoch Quick Test for Unified Metric Learning")
    parser.add_argument("--health-only", action="store_true", help="Run health check only")
    parser.add_argument("--config", default="script/unified_metric/configs/unified_experiments_1epoch.yaml",
                       help="Configuration file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print_header()

    # Check configuration file first
    config_path = Path(args.config)
    if not check_config_file(config_path):
        print("\n❌ Configuration check failed. Please fix the issues above.")
        return 1

    # Run health check
    health_passed = run_health_check(args.config)
    if not health_passed:
        print("\n❌ Health check failed. Please fix system issues before proceeding.")
        return 1

    if args.health_only:
        print("\n✅ Health check completed successfully!")
        print("🚀 System is ready for 1-epoch testing.")
        return 0

    # Run 1-epoch validation
    validation_passed = run_1epoch_validation()

    # Run quick pipeline test
    pipeline_passed = run_quick_pipeline_test()

    # Print summary
    print_summary(health_passed, validation_passed, pipeline_passed)

    # Return appropriate exit code
    return 0 if (health_passed and validation_passed and pipeline_passed) else 1

if __name__ == "__main__":
    sys.exit(main())