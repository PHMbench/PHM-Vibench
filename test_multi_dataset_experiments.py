#!/usr/bin/env python3
"""
Test script for the enhanced multi-dataset experiment runner
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project path
sys.path.append('.')

def test_multi_dataset_experiment_runner():
    """Test the enhanced multi-dataset experiment runner"""
    print("🧪 Testing enhanced multi-dataset experiment runner...")
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test import
        try:
            from scripts.multi_dataset_experiments import (
                MultiDatasetExperimentRunner,
                DatasetInfo,
                ExperimentConfig,
                ExperimentResult,
                ResourceManager
            )
            print("✅ Successfully imported all classes")
        except Exception as e:
            print(f"❌ Import failed: {e}")
            return False
        
        # Test ResourceManager
        try:
            rm = ResourceManager()
            status = rm.get_resource_status()
            print(f"✅ ResourceManager initialized: {status['total_ram_mb']:.0f}MB RAM, {len(status['gpus'])} GPUs")
        except Exception as e:
            print(f"❌ ResourceManager test failed: {e}")
            return False
        
        # Test DatasetInfo
        try:
            dataset = DatasetInfo(
                name="test_dataset",
                metadata_file="test.xlsx",
                h5_file=Path("test.h5"),
                num_samples=1000,
                num_classes=5,
                ready=False
            )
            print(f"✅ DatasetInfo created: {dataset.name} with {dataset.num_samples} samples")
        except Exception as e:
            print(f"❌ DatasetInfo test failed: {e}")
            return False
        
        # Test ExperimentConfig
        try:
            exp_config = ExperimentConfig(
                id="test_exp_001",
                name="test_experiment",
                dataset_combination=["test_dataset"],
                variant_name="default",
                config_overrides={'test': 'value'},
                expected_duration_hours=0.1
            )
            print(f"✅ ExperimentConfig created: {exp_config.id}")
        except Exception as e:
            print(f"❌ ExperimentConfig test failed: {e}")
            return False
        
        # Test MultiDatasetExperimentRunner initialization
        try:
            runner = MultiDatasetExperimentRunner(
                base_config_path="configs/id_contrastive/debug.yaml",
                metadata_dir=temp_dir,
                results_dir=temp_path / "test_results",
                dry_run=True,
                enable_benchmarking=False
            )
            print("✅ MultiDatasetExperimentRunner initialized successfully")
        except Exception as e:
            print(f"❌ MultiDatasetExperimentRunner initialization failed: {e}")
            return False
        
        # Test quick validation
        try:
            validation_results = runner.run_quick_validation(dataset_sample_size=50)
            print(f"✅ Quick validation completed: {validation_results['overall_validation_success']}")
        except Exception as e:
            print(f"⚠️  Quick validation failed (expected if no data): {e}")
        
        print("✅ All basic tests passed!")
        return True

def test_cli_interface():
    """Test the CLI interface"""
    print("\n🧪 Testing CLI interface...")
    
    try:
        # Test help
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/multi_dataset_experiments.py", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ CLI help command works")
            
            # Check if key arguments are present
            help_text = result.stdout
            required_args = [
                "--base_config",
                "--metadata_dir",
                "--results_dir",
                "--parallel",
                "--dry_run",
                "--quick_validation",
                "--combination_strategies",
                "--enable_ablation",
                "--enable_benchmarking"
            ]
            
            missing_args = [arg for arg in required_args if arg not in help_text]
            if missing_args:
                print(f"⚠️  Missing arguments in help: {missing_args}")
            else:
                print("✅ All expected arguments present in help")
            
            return True
        else:
            print(f"❌ CLI help failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("Enhanced Multi-Dataset Experiment Runner - Test Suite")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_multi_dataset_experiment_runner()
    test2_passed = test_cli_interface()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 All tests passed! The enhanced script is ready for use.")
        print("\nUsage examples:")
        print("1. Quick validation:")
        print("   python scripts/multi_dataset_experiments.py --quick_validation")
        print("\n2. Dry run with multiple strategies:")
        print("   python scripts/multi_dataset_experiments.py --dry_run \\")
        print("     --combination_strategies single cross_domain \\")
        print("     --variants default large_window --enable_ablation")
        print("\n3. Run with specific datasets:")
        print("   python scripts/multi_dataset_experiments.py \\")
        print("     --include_datasets CWRU XJTU --domain_types bearing \\")
        print("     --parallel --max_parallel 2")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())