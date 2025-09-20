#!/usr/bin/env python3
"""
Master Runner for All CWRU Cases

This script orchestrates the execution of all three experimental cases:
1. Case 1: Direct few-shot learning
2. Case 2: Contrastive pretraining + few-shot learning
3. Case 3: Flow + contrastive joint training + few-shot learning

After all cases complete, it automatically runs the comparison analysis
and generates the final report.

Usage:
    python run_all_cases.py [--cases case1,case2,case3] [--skip-comparison]

Author: PHM-Vibench Development Team
Date: September 2025
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
import traceback

from common_utils import setup_logger, log_system_info

class ExperimentRunner:
    """Master experiment runner for all CWRU cases"""

    def __init__(self, cases_to_run=None, skip_comparison=False):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"logs/master_{self.timestamp}.log"
        self.logger = setup_logger("MasterRunner", self.log_file)

        self.cases_to_run = cases_to_run or ['case1', 'case2', 'case3']
        self.skip_comparison = skip_comparison

        self.results = {}
        self.execution_summary = {
            'start_time': time.time(),
            'cases_completed': [],
            'cases_failed': [],
            'total_execution_time': 0
        }

    def run_case(self, case_name):
        """Run a specific case"""
        case_scripts = {
            'case1': 'case1_direct.py',
            'case2': 'case2_contrastive.py',
            'case3': 'case3_flow_contrastive.py'
        }

        if case_name not in case_scripts:
            self.logger.error(f"Unknown case: {case_name}")
            return False

        script_path = case_scripts[case_name]
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"EXECUTING {case_name.upper()}")
        self.logger.info(f"Script: {script_path}")
        self.logger.info(f"{'='*60}")

        try:
            start_time = time.time()

            # Run the case script
            result = subprocess.run([
                sys.executable, script_path
            ], capture_output=True, text=True, cwd=os.getcwd())

            execution_time = time.time() - start_time

            if result.returncode == 0:
                self.logger.info(f"✅ {case_name.upper()} completed successfully in {execution_time:.2f} seconds")
                self.logger.info(f"STDOUT:\n{result.stdout}")

                self.execution_summary['cases_completed'].append(case_name)
                self.results[case_name] = {
                    'status': 'success',
                    'execution_time': execution_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                return True

            else:
                self.logger.error(f"❌ {case_name.upper()} failed with return code {result.returncode}")
                self.logger.error(f"STDOUT:\n{result.stdout}")
                self.logger.error(f"STDERR:\n{result.stderr}")

                self.execution_summary['cases_failed'].append(case_name)
                self.results[case_name] = {
                    'status': 'failed',
                    'execution_time': execution_time,
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                return False

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ {case_name.upper()} failed with exception: {str(e)}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")

            self.execution_summary['cases_failed'].append(case_name)
            self.results[case_name] = {
                'status': 'exception',
                'execution_time': execution_time,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            return False

    def run_comparison(self):
        """Run the comparison analysis"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info("RUNNING COMPARISON ANALYSIS")
        self.logger.info(f"{'='*60}")

        try:
            start_time = time.time()

            # Run comparison script
            result = subprocess.run([
                sys.executable, 'compare_results.py'
            ], capture_output=True, text=True, cwd=os.getcwd())

            execution_time = time.time() - start_time

            if result.returncode == 0:
                self.logger.info(f"✅ Comparison analysis completed successfully in {execution_time:.2f} seconds")
                self.logger.info(f"STDOUT:\n{result.stdout}")

                self.results['comparison'] = {
                    'status': 'success',
                    'execution_time': execution_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                return True

            else:
                self.logger.error(f"❌ Comparison analysis failed with return code {result.returncode}")
                self.logger.error(f"STDOUT:\n{result.stdout}")
                self.logger.error(f"STDERR:\n{result.stderr}")

                self.results['comparison'] = {
                    'status': 'failed',
                    'execution_time': execution_time,
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                return False

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Comparison analysis failed with exception: {str(e)}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")

            self.results['comparison'] = {
                'status': 'exception',
                'execution_time': execution_time,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            return False

    def run_all(self):
        """Run all specified cases and comparison"""
        self.logger.info("="*80)
        self.logger.info("CWRU MULTI-TASK FEW-SHOT LEARNING: MASTER EXPERIMENT RUNNER")
        self.logger.info("="*80)

        # Log system information
        log_system_info(self.logger)

        self.logger.info(f"Cases to run: {self.cases_to_run}")
        self.logger.info(f"Skip comparison: {self.skip_comparison}")
        self.logger.info(f"Master log: {self.log_file}")

        # Check if scripts exist
        case_scripts = {
            'case1': 'case1_direct.py',
            'case2': 'case2_contrastive.py',
            'case3': 'case3_flow_contrastive.py'
        }

        missing_scripts = []
        for case in self.cases_to_run:
            if case in case_scripts:
                script_path = case_scripts[case]
                if not os.path.exists(script_path):
                    missing_scripts.append(script_path)

        if missing_scripts:
            self.logger.error(f"Missing scripts: {missing_scripts}")
            return False

        # Check if comparison script exists
        if not self.skip_comparison and not os.path.exists('compare_results.py'):
            self.logger.error("compare_results.py not found")
            return False

        # Run each case
        overall_success = True
        for case_name in self.cases_to_run:
            success = self.run_case(case_name)
            if not success:
                overall_success = False
                self.logger.warning(f"Case {case_name} failed, but continuing with remaining cases...")

        # Run comparison if requested and at least one case succeeded
        if not self.skip_comparison and self.execution_summary['cases_completed']:
            self.logger.info(f"\nSuccessful cases: {self.execution_summary['cases_completed']}")
            self.logger.info("Proceeding with comparison analysis...")
            comparison_success = self.run_comparison()
            if not comparison_success:
                overall_success = False

        # Final summary
        self.execution_summary['total_execution_time'] = time.time() - self.execution_summary['start_time']
        self.generate_final_report()

        return overall_success

    def generate_final_report(self):
        """Generate final execution summary report"""
        self.logger.info("\n" + "="*80)
        self.logger.info("FINAL EXECUTION SUMMARY")
        self.logger.info("="*80)

        # Overall statistics
        total_cases = len(self.cases_to_run)
        completed_cases = len(self.execution_summary['cases_completed'])
        failed_cases = len(self.execution_summary['cases_failed'])

        self.logger.info(f"📊 EXECUTION STATISTICS")
        self.logger.info(f"   Total cases requested: {total_cases}")
        self.logger.info(f"   Cases completed: {completed_cases}")
        self.logger.info(f"   Cases failed: {failed_cases}")
        self.logger.info(f"   Success rate: {completed_cases/total_cases*100:.1f}%")
        self.logger.info(f"   Total execution time: {self.execution_summary['total_execution_time']:.2f} seconds")

        # Individual case results
        self.logger.info(f"\n📋 INDIVIDUAL CASE RESULTS")
        for case_name in self.cases_to_run:
            if case_name in self.results:
                result = self.results[case_name]
                status_emoji = "✅" if result['status'] == 'success' else "❌"
                self.logger.info(f"   {case_name.upper()}: {status_emoji} {result['status']} "
                                f"({result['execution_time']:.2f}s)")

        # Comparison result
        if 'comparison' in self.results:
            comp_result = self.results['comparison']
            status_emoji = "✅" if comp_result['status'] == 'success' else "❌"
            self.logger.info(f"   COMPARISON: {status_emoji} {comp_result['status']} "
                           f"({comp_result['execution_time']:.2f}s)")

        # Success cases
        if self.execution_summary['cases_completed']:
            self.logger.info(f"\n🎉 SUCCESSFUL CASES")
            for case in self.execution_summary['cases_completed']:
                self.logger.info(f"   ✅ {case.upper()}")

        # Failed cases
        if self.execution_summary['cases_failed']:
            self.logger.info(f"\n❌ FAILED CASES")
            for case in self.execution_summary['cases_failed']:
                self.logger.info(f"   ❌ {case.upper()}")

        # Output files
        self.logger.info(f"\n📁 OUTPUT FILES")
        self.logger.info(f"   Master log: {self.log_file}")

        # Check for results
        results_files = []
        for case in self.execution_summary['cases_completed']:
            case_results = [f for f in os.listdir('results') if f.startswith(f'{case}_results_')]
            if case_results:
                results_files.extend([f"results/{f}" for f in case_results])

        if results_files:
            self.logger.info(f"   Results files: {len(results_files)} files in results/")

        # Check for figures
        if os.path.exists('figures') and os.listdir('figures'):
            figure_files = os.listdir('figures')
            self.logger.info(f"   Visualization files: {len(figure_files)} files in figures/")

        # Check for comparison report
        if os.path.exists('comparison_report.md'):
            self.logger.info(f"   Comparison report: comparison_report.md")

        # Next steps
        self.logger.info(f"\n🚀 NEXT STEPS")
        if self.execution_summary['cases_completed']:
            self.logger.info("   ✅ Review comparison_report.md for detailed analysis")
            self.logger.info("   ✅ Check figures/ for visualization plots")
            self.logger.info("   ✅ Examine individual case logs in logs/")
        else:
            self.logger.info("   ❌ No cases completed successfully")
            self.logger.info("   🔍 Check individual case logs for error details")

        # Recommendations
        if failed_cases > 0:
            self.logger.info(f"\n💡 TROUBLESHOOTING RECOMMENDATIONS")
            self.logger.info("   1. Check data paths in common_utils.py")
            self.logger.info("   2. Verify CUDA/GPU availability if using GPU")
            self.logger.info("   3. Ensure sufficient disk space for results")
            self.logger.info("   4. Review individual case logs for specific errors")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Run all CWRU experimental cases')
    parser.add_argument('--cases', type=str, default='case1,case2,case3',
                       help='Comma-separated list of cases to run (default: case1,case2,case3)')
    parser.add_argument('--skip-comparison', action='store_true',
                       help='Skip comparison analysis after running cases')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be executed without running')

    args = parser.parse_args()

    # Parse cases
    cases_to_run = [case.strip() for case in args.cases.split(',')]
    valid_cases = ['case1', 'case2', 'case3']

    # Validate cases
    invalid_cases = [case for case in cases_to_run if case not in valid_cases]
    if invalid_cases:
        print(f"Error: Invalid cases specified: {invalid_cases}")
        print(f"Valid cases: {valid_cases}")
        return 1

    if args.dry_run:
        print("DRY RUN MODE - Would execute:")
        print(f"Cases: {cases_to_run}")
        print(f"Skip comparison: {args.skip_comparison}")
        return 0

    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # Run experiments
    runner = ExperimentRunner(cases_to_run, args.skip_comparison)
    success = runner.run_all()

    # Print final summary to console
    print("\n" + "="*80)
    print("MASTER RUNNER EXECUTION COMPLETED")
    print("="*80)
    print(f"Cases requested: {cases_to_run}")
    print(f"Cases completed: {runner.execution_summary['cases_completed']}")
    print(f"Cases failed: {runner.execution_summary['cases_failed']}")
    print(f"Total time: {runner.execution_summary['total_execution_time']:.2f} seconds")
    print(f"Master log: {runner.log_file}")

    if runner.execution_summary['cases_completed']:
        print("\n✅ Check comparison_report.md for detailed results analysis")
        print("✅ Check figures/ for visualization plots")
    else:
        print("\n❌ No cases completed successfully")
        print("🔍 Check master log for troubleshooting guidance")

    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)