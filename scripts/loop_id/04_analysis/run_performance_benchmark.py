#!/usr/bin/env python3
"""
Script to run comprehensive performance benchmarks for ContrastiveIDTask
Usage: python scripts/run_performance_benchmark.py [--test all|training|data|model|scalability|hardware]
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from benchmarks.contrastive_performance_benchmark import AdvancedPerformanceBenchmark


def main():
    parser = argparse.ArgumentParser(description="Run ContrastiveIDTask Performance Benchmarks")
    parser.add_argument("--test", 
                       choices=["all", "training", "data", "model", "scalability", "hardware"],
                       default="all",
                       help="Type of benchmark to run")
    parser.add_argument("--save-dir", 
                       default="./benchmark_results",
                       help="Directory to save benchmark results")
    parser.add_argument("--device", 
                       choices=["auto", "cpu", "cuda"],
                       default="auto",
                       help="Device to use for benchmarking")
    parser.add_argument("--quick", 
                       action="store_true",
                       help="Run quick benchmark with reduced test cases")
    parser.add_argument("--verbose", 
                       action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    print("="*60)
    print("ContrastiveIDTask Performance Benchmark Suite")
    print("="*60)
    print(f"Test Type: {args.test}")
    print(f"Save Directory: {args.save_dir}")
    print(f"Device: {args.device}")
    print(f"Quick Mode: {args.quick}")
    print("="*60)
    
    try:
        # Create benchmark instance
        benchmark = AdvancedPerformanceBenchmark(save_dir=args.save_dir)
        
        # Set device
        import torch
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        benchmark.device = device
        
        if args.verbose:
            import logging
            benchmark.logger.setLevel(logging.DEBUG)
        
        # Modify targets for quick mode
        if args.quick:
            print("Running in quick mode - reducing test complexity...")
            # Reduce test complexity for quick runs
            benchmark.targets['training_performance']['samples_per_second'] = 10
            benchmark.targets['scalability']['max_batch_size'] = 64
        
        # Run benchmarks
        success = False
        if args.test == "all":
            success = benchmark.run_comprehensive_benchmark()
        elif args.test == "training":
            benchmark.benchmark_training_performance()
            benchmark.generate_comprehensive_report()
            success = True
        elif args.test == "data":
            benchmark.benchmark_data_processing_performance()
            benchmark.generate_comprehensive_report()
            success = True
        elif args.test == "model":
            benchmark.benchmark_model_performance()
            benchmark.generate_comprehensive_report()
            success = True
        elif args.test == "scalability":
            benchmark.benchmark_scalability_testing()
            benchmark.generate_comprehensive_report()
            success = True
        elif args.test == "hardware":
            benchmark.benchmark_hardware_optimization()
            benchmark.generate_comprehensive_report()
            success = True
        
        if success:
            print("\n" + "="*60)
            print("BENCHMARK COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Results saved to: {args.save_dir}")
            print(f"- HTML Report: {args.save_dir}/reports/performance_report.html")
            print(f"- Summary: {args.save_dir}/reports/performance_summary.md")
            print(f"- Plots: {args.save_dir}/plots/")
            print(f"- Raw Data: {args.save_dir}/comprehensive_benchmark_results.json")
            
            # Show overall score if available
            if 'overall_performance' in benchmark.results:
                score_info = benchmark.results['overall_performance']
                print(f"\nOverall Performance Score: {score_info['score']:.1f}/100 ({score_info['grade']})")
        else:
            print("\n" + "="*60)
            print("BENCHMARK FAILED")
            print("="*60)
            sys.exit(1)
    
    except Exception as e:
        print(f"\nError running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()