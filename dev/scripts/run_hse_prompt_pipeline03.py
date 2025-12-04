#!/usr/bin/env python3
"""
HSE Prompt Pipeline_03 Experiment Runner

This script provides comprehensive automation for running HSE Prompt-guided
contrastive learning experiments using the Pipeline_03 workflow.

Features:
- Cross-dataset backbone comparison experiments
- Ablation study automation
- HSEPromptPipelineIntegration adapter integration
- Automated result collection and analysis
- Stage-specific execution support
- Comprehensive logging and experiment tracking

Part of P1 Feature Enhancement for HSE Industrial Contrastive Learning.

Author: PHM-Vibench HSE Prompt Team  
Date: 2025-09-09
"""

import argparse
import os
import sys
import yaml
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import subprocess
import pandas as pd
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import PHM-Vibench components
from src.Pipeline_03_multitask_pretrain_finetune import MultiTaskPretrainFinetunePipeline
from src.utils.pipeline_config.hse_prompt_integration import HSEPromptPipelineIntegration
from src.utils.config.hse_prompt_validator import HSEPromptConfigValidator
from src.utils.evaluation.ZeroShotEvaluator import ZeroShotEvaluator
from src.configs import load_config


class HSEPromptPipeline03Runner:
    """
    Runner class for HSE Prompt experiments using Pipeline_03 workflow.
    
    Provides automation for backbone comparison, ablation studies, and
    comprehensive result analysis.
    """
    
    def __init__(self, output_dir: str = "results/hse_prompt_pipeline03"):
        """
        Initialize the runner.
        
        Args:
            output_dir: Base output directory for all experiments
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.validator = HSEPromptConfigValidator()
        self.integration_adapter = HSEPromptPipelineIntegration()
        
        # Experiment tracking
        self.experiment_log = []
        self.results_summary = {}
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup experiment logging."""
        import logging
        
        log_file = self.output_dir / f"hse_prompt_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"HSE Prompt Pipeline_03 Runner initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Log file: {log_file}")
    
    def run_unified_metric_learning(self,
                                  config_path: Optional[str] = None,
                                  num_seeds: int = 5,
                                  stage: str = 'complete') -> Dict[str, Any]:
        """
        Run the unified metric learning experiment across all 5 datasets.

        This executes the main experimental matrix (6 base experiments × 5 seeds = 30 runs)
        defined in the HSE Prompt multitask configuration.

        Args:
            config_path: Path to main configuration (defaults to hse_prompt_multitask_config.yaml)
            num_seeds: Number of random seeds to run (default: 5)
            stage: Which stage to run ('pretraining', 'finetuning', 'complete')

        Returns:
            Dictionary containing comprehensive experiment results
        """
        if config_path is None:
            config_path = "configs/pipeline_03/hse_prompt_multitask_config.yaml"

        self.logger.info(f"Starting unified metric learning experiment matrix")
        self.logger.info(f"Configuration: {config_path}")
        self.logger.info(f"Seeds: {num_seeds}, Stage: {stage}")

        # Load and validate base configuration
        try:
            base_config = load_config(config_path)
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

        # Validate configuration
        is_valid, errors, warnings = self.validator.validate_config(
            base_config, 'pretraining' if stage in ['pretraining', 'complete'] else 'finetuning'
        )
        if not is_valid:
            self.logger.error(f"Configuration validation failed: {errors}")
            raise ValueError(f"Invalid configuration: {errors}")

        if warnings:
            self.logger.warning(f"Configuration warnings: {warnings}")

        # Initialize zero-shot evaluator
        zero_shot_evaluator = ZeroShotEvaluator()

        matrix_results = {
            'experiment_type': 'unified_metric_learning',
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'seed_results': {},
            'aggregated_metrics': {},
            'zero_shot_results': {},
            'config_path': config_path
        }

        # Run experiments for each seed
        for seed in range(42, 42 + num_seeds):  # Starting from seed 42
            self.logger.info(f"Running experiment with seed: {seed}")

            try:
                # Create seed-specific configuration
                seed_config = base_config.copy()
                seed_config['environment']['seed'] = seed
                seed_config['environment']['output_dir'] = f"results/pipeline_03/unified_metric_learning/seed_{seed}"

                # Create temporary config file
                temp_config_path = self.output_dir / f"temp_unified_config_seed_{seed}.yaml"
                with open(temp_config_path, 'w') as f:
                    yaml.dump(seed_config, f, default_flow_style=False, indent=2)

                # Run Pipeline_03 experiment
                start_time = time.time()
                pipeline = MultiTaskPretrainFinetunePipeline(str(temp_config_path))

                seed_result = {
                    'seed': seed,
                    'start_time': datetime.now().isoformat()
                }

                # Stage 1: Pretraining
                if stage in ['pretraining', 'complete']:
                    self.logger.info(f"Seed {seed}: Running pretraining stage")
                    checkpoint_paths = pipeline.run_pretraining_stage()

                    seed_result.update({
                        'pretraining_checkpoints': checkpoint_paths,
                        'pretraining_time': time.time() - start_time
                    })

                    # Zero-shot evaluation after pretraining
                    if base_config.get('pipeline_03', {}).get('zero_shot_evaluation', {}).get('enabled', False):
                        self.logger.info(f"Seed {seed}: Running zero-shot evaluation")
                        zero_shot_results = zero_shot_evaluator.evaluate_checkpoints(
                            checkpoint_paths, base_config
                        )
                        seed_result['zero_shot_results'] = zero_shot_results

                # Stage 2: Finetuning
                if stage in ['finetuning', 'complete']:
                    if stage == 'finetuning':
                        # Load existing checkpoints for finetuning-only runs
                        checkpoint_paths = self._find_existing_checkpoints_for_seed(seed)

                    self.logger.info(f"Seed {seed}: Running finetuning stage")
                    finetune_start = time.time()
                    finetuning_results = pipeline.run_finetuning_stage(checkpoint_paths)

                    seed_result.update({
                        'finetuning_results': finetuning_results,
                        'finetuning_time': time.time() - finetune_start,
                        'total_time': time.time() - start_time
                    })

                seed_result.update({
                    'status': 'completed',
                    'end_time': datetime.now().isoformat()
                })

                matrix_results['seed_results'][seed] = seed_result
                matrix_results['successful_runs'] += 1

                self.logger.info(f"✓ Completed experiment for seed: {seed}")

                # Clean up temporary config
                temp_config_path.unlink()

            except Exception as e:
                self.logger.error(f"✗ Failed experiment for seed {seed}: {e}")
                matrix_results['seed_results'][seed] = {
                    'seed': seed,
                    'error': str(e),
                    'status': 'failed'
                }
                matrix_results['failed_runs'] += 1

        matrix_results['total_runs'] = matrix_results['successful_runs'] + matrix_results['failed_runs']

        # Aggregate results across seeds
        self._aggregate_matrix_results(matrix_results)

        # Save comprehensive results
        self._save_comparison_results(matrix_results, 'unified_metric_learning')

        # Generate unified metric learning analysis
        self._analyze_unified_metrics(matrix_results)

        self.logger.info(f"✓ Unified metric learning completed: {matrix_results['successful_runs']}/{matrix_results['total_runs']} successful")

        return matrix_results

    def run_backbone_comparison(self,
                               config_path: str,
                               backbones: Optional[List[str]] = None,
                               stage: str = 'complete') -> Dict[str, Any]:
        """
        Run backbone comparison experiments with HSE prompt features.
        
        Args:
            config_path: Path to base configuration file
            backbones: List of backbones to compare
            stage: Which stage to run ('pretraining', 'finetuning', 'complete')
            
        Returns:
            Dictionary containing comparison results
        """
        if backbones is None:
            backbones = ['B_08_PatchTST', 'B_04_Dlinear', 'B_06_TimesNet', 'B_09_FNO']
        
        self.logger.info(f"Starting backbone comparison with {len(backbones)} backbones")
        self.logger.info(f"Backbones: {backbones}")
        
        # Validate base configuration
        is_valid, errors, warnings = self.validator.validate_yaml_file(config_path, 'pretraining')
        if not is_valid:
            self.logger.error(f"Configuration validation failed: {errors}")
            raise ValueError(f"Invalid configuration: {errors}")
        
        if warnings:
            self.logger.warning(f"Configuration warnings: {warnings}")
        
        comparison_results = {}
        
        for backbone in backbones:
            self.logger.info(f"Running experiment with backbone: {backbone}")
            
            try:
                # Create backbone-specific configuration
                backbone_config = self.integration_adapter.create_backbone_specific_config(
                    config_path, backbone
                )
                
                # Create temporary config file
                temp_config_path = self.output_dir / f"temp_config_{backbone}.yaml"
                with open(temp_config_path, 'w') as f:
                    yaml.dump(backbone_config, f, default_flow_style=False, indent=2)
                
                # Run Pipeline_03 experiment
                start_time = time.time()
                pipeline = MultiTaskPretrainFinetunePipeline(str(temp_config_path))
                
                if stage in ['pretraining', 'complete']:
                    checkpoint_paths = pipeline.run_pretraining_stage()
                    comparison_results[backbone] = {
                        'pretraining_checkpoint': checkpoint_paths.get(backbone),
                        'pretraining_time': time.time() - start_time
                    }
                
                if stage in ['finetuning', 'complete']:
                    if stage == 'finetuning':
                        # Load existing checkpoints
                        checkpoint_paths = self._find_existing_checkpoints(backbone)
                    
                    finetune_start = time.time()
                    finetuning_results = pipeline.run_finetuning_stage(checkpoint_paths)
                    
                    if backbone not in comparison_results:
                        comparison_results[backbone] = {}
                    
                    comparison_results[backbone].update({
                        'finetuning_results': finetuning_results,
                        'finetuning_time': time.time() - finetune_start,
                        'total_time': time.time() - start_time
                    })
                
                self.logger.info(f"✓ Completed experiment for backbone: {backbone}")
                
                # Clean up temporary config
                temp_config_path.unlink()
                
            except Exception as e:
                self.logger.error(f"✗ Failed experiment for backbone {backbone}: {e}")
                comparison_results[backbone] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Save comparison results
        self._save_comparison_results(comparison_results, 'backbone_comparison')
        
        return comparison_results
    
    def run_ablation_study(self, 
                          ablation_configs: Optional[List[str]] = None,
                          stage: str = 'complete') -> Dict[str, Any]:
        """
        Run ablation study experiments.
        
        Args:
            ablation_configs: List of ablation configuration files
            stage: Which stage to run ('pretraining', 'finetuning', 'complete')
            
        Returns:
            Dictionary containing ablation study results
        """
        if ablation_configs is None:
            ablation_configs = [
                'configs/pipeline_03/ablation/hse_system_prompt_only.yaml',
                'configs/pipeline_03/ablation/hse_sample_prompt_only.yaml',
                'configs/pipeline_03/ablation/hse_no_prompt_baseline.yaml'
            ]
        
        self.logger.info(f"Starting ablation study with {len(ablation_configs)} configurations")
        
        ablation_results = {}
        
        for config_path in ablation_configs:
            config_name = Path(config_path).stem
            self.logger.info(f"Running ablation experiment: {config_name}")
            
            try:
                # Validate configuration
                config_type = 'pretraining' if stage in ['pretraining', 'complete'] else 'finetuning'
                is_valid, errors, warnings = self.validator.validate_yaml_file(config_path, config_type)
                
                if not is_valid:
                    self.logger.error(f"Configuration validation failed for {config_name}: {errors}")
                    ablation_results[config_name] = {
                        'error': f"Validation failed: {errors}",
                        'status': 'validation_failed'
                    }
                    continue
                
                if warnings:
                    self.logger.warning(f"Configuration warnings for {config_name}: {warnings}")
                
                # Run experiment
                start_time = time.time()
                pipeline = MultiTaskPretrainFinetunePipeline(config_path)
                
                experiment_result = {'config_name': config_name}
                
                if stage in ['pretraining', 'complete']:
                    checkpoint_paths = pipeline.run_pretraining_stage()
                    experiment_result.update({
                        'pretraining_checkpoints': checkpoint_paths,
                        'pretraining_time': time.time() - start_time
                    })
                
                if stage in ['finetuning', 'complete']:
                    if stage == 'finetuning':
                        # Load existing checkpoints for finetuning-only runs
                        checkpoint_paths = self._find_existing_checkpoints_for_config(config_name)
                    
                    finetune_start = time.time()
                    finetuning_results = pipeline.run_finetuning_stage(checkpoint_paths)
                    
                    experiment_result.update({
                        'finetuning_results': finetuning_results,
                        'finetuning_time': time.time() - finetune_start,
                        'total_time': time.time() - start_time
                    })
                
                ablation_results[config_name] = experiment_result
                self.logger.info(f"✓ Completed ablation experiment: {config_name}")
                
            except Exception as e:
                self.logger.error(f"✗ Failed ablation experiment {config_name}: {e}")
                ablation_results[config_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Save ablation results
        self._save_comparison_results(ablation_results, 'ablation_study')
        
        # Generate ablation analysis
        self._analyze_ablation_results(ablation_results)
        
        return ablation_results
    
    def run_custom_experiment(self, 
                            config_path: str,
                            experiment_name: str,
                            stage: str = 'complete') -> Dict[str, Any]:
        """
        Run a custom HSE prompt experiment.
        
        Args:
            config_path: Path to configuration file
            experiment_name: Name for the experiment
            stage: Which stage to run ('pretraining', 'finetuning', 'complete')
            
        Returns:
            Dictionary containing experiment results
        """
        self.logger.info(f"Starting custom experiment: {experiment_name}")
        
        # Validate configuration
        config_type = 'pretraining' if stage in ['pretraining', 'complete'] else 'finetuning'
        is_valid, errors, warnings = self.validator.validate_yaml_file(config_path, config_type)
        
        if not is_valid:
            self.logger.error(f"Configuration validation failed: {errors}")
            raise ValueError(f"Invalid configuration: {errors}")
        
        if warnings:
            self.logger.warning(f"Configuration warnings: {warnings}")
        
        try:
            start_time = time.time()
            pipeline = MultiTaskPretrainFinetunePipeline(config_path)
            
            experiment_result = {
                'experiment_name': experiment_name,
                'config_path': config_path,
                'start_time': datetime.now().isoformat()
            }
            
            if stage in ['pretraining', 'complete']:
                checkpoint_paths = pipeline.run_pretraining_stage()
                experiment_result.update({
                    'pretraining_checkpoints': checkpoint_paths,
                    'pretraining_time': time.time() - start_time
                })
            
            if stage in ['finetuning', 'complete']:
                if stage == 'finetuning':
                    # Load existing checkpoints for finetuning-only runs
                    checkpoint_paths = self._find_existing_checkpoints_for_config(experiment_name)
                
                finetune_start = time.time()
                finetuning_results = pipeline.run_finetuning_stage(checkpoint_paths)
                
                experiment_result.update({
                    'finetuning_results': finetuning_results,
                    'finetuning_time': time.time() - finetune_start,
                    'total_time': time.time() - start_time
                })
            
            experiment_result['status'] = 'completed'
            experiment_result['end_time'] = datetime.now().isoformat()
            
            # Save experiment result
            self._save_experiment_result(experiment_result, experiment_name)
            
            self.logger.info(f"✓ Completed custom experiment: {experiment_name}")
            
            return experiment_result
            
        except Exception as e:
            self.logger.error(f"✗ Failed custom experiment {experiment_name}: {e}")
            return {
                'experiment_name': experiment_name,
                'error': str(e),
                'status': 'failed'
            }
    
    def generate_comprehensive_report(self, results_dir: Optional[str] = None) -> str:
        """
        Generate comprehensive experiment report.
        
        Args:
            results_dir: Directory containing experiment results
            
        Returns:
            Path to generated report file
        """
        if results_dir is None:
            results_dir = self.output_dir
        
        self.logger.info("Generating comprehensive experiment report")
        
        # Collect all result files
        result_files = list(Path(results_dir).glob('*_results.json'))
        
        report_data = {
            'report_generated': datetime.now().isoformat(),
            'total_experiments': len(result_files),
            'experiments': []
        }
        
        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    experiment_data = json.load(f)
                    report_data['experiments'].append(experiment_data)
            except Exception as e:
                self.logger.warning(f"Could not load result file {result_file}: {e}")
        
        # Generate report file
        report_file = Path(results_dir) / f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate markdown summary
        markdown_report = self._generate_markdown_report(report_data)
        markdown_file = report_file.with_suffix('.md')
        
        with open(markdown_file, 'w') as f:
            f.write(markdown_report)
        
        self.logger.info(f"✓ Generated comprehensive report: {report_file}")
        self.logger.info(f"✓ Generated markdown summary: {markdown_file}")
        
        return str(report_file)
    
    def _find_existing_checkpoints(self, backbone: str) -> Dict[str, Optional[str]]:
        """Find existing checkpoint files for a backbone."""
        # Implementation for finding existing checkpoints
        return {backbone: None}  # Placeholder implementation
    
    def _find_existing_checkpoints_for_config(self, config_name: str) -> Dict[str, Optional[str]]:
        """Find existing checkpoint files for a configuration."""
        # Implementation for finding existing checkpoints by config
        return {}  # Placeholder implementation
    
    def _save_comparison_results(self, results: Dict[str, Any], study_type: str):
        """Save comparison results to file."""
        result_file = self.output_dir / f"{study_type}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Saved {study_type} results to: {result_file}")
    
    def _save_experiment_result(self, result: Dict[str, Any], experiment_name: str):
        """Save individual experiment result."""
        result_file = self.output_dir / f"{experiment_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        self.logger.info(f"Saved experiment result to: {result_file}")
    
    def _aggregate_matrix_results(self, matrix_results: Dict[str, Any]):
        """Aggregate results across all seeds in the experimental matrix."""
        self.logger.info("Aggregating results across seeds")

        successful_seeds = {k: v for k, v in matrix_results['seed_results'].items()
                           if v.get('status') == 'completed'}

        if not successful_seeds:
            self.logger.warning("No successful runs to aggregate")
            return

        # Aggregate timing metrics
        pretraining_times = [r.get('pretraining_time', 0) for r in successful_seeds.values()]
        finetuning_times = [r.get('finetuning_time', 0) for r in successful_seeds.values()]
        total_times = [r.get('total_time', 0) for r in successful_seeds.values()]

        matrix_results['aggregated_metrics'] = {
            'timing': {
                'avg_pretraining_time': sum(pretraining_times) / len(pretraining_times) if pretraining_times else 0,
                'avg_finetuning_time': sum(finetuning_times) / len(finetuning_times) if finetuning_times else 0,
                'avg_total_time': sum(total_times) / len(total_times) if total_times else 0,
                'std_pretraining_time': self._calculate_std(pretraining_times),
                'std_finetuning_time': self._calculate_std(finetuning_times),
                'std_total_time': self._calculate_std(total_times)
            }
        }

        # Aggregate zero-shot results if available
        zero_shot_results = [r.get('zero_shot_results', {}) for r in successful_seeds.values()
                            if 'zero_shot_results' in r]

        if zero_shot_results:
            matrix_results['aggregated_metrics']['zero_shot'] = self._aggregate_zero_shot_metrics(zero_shot_results)

        # Aggregate finetuning results if available
        finetuning_results = [r.get('finetuning_results', {}) for r in successful_seeds.values()
                             if 'finetuning_results' in r]

        if finetuning_results:
            matrix_results['aggregated_metrics']['finetuning'] = self._aggregate_finetuning_metrics(finetuning_results)

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) <= 1:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def _aggregate_zero_shot_metrics(self, zero_shot_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate zero-shot evaluation metrics across seeds."""
        aggregated = {}

        # Collect all dataset results
        for result in zero_shot_results:
            for dataset, metrics in result.items():
                if dataset not in aggregated:
                    aggregated[dataset] = {'accuracies': [], 'f1_scores': []}

                if 'accuracy' in metrics:
                    aggregated[dataset]['accuracies'].append(metrics['accuracy'])
                if 'f1_score' in metrics:
                    aggregated[dataset]['f1_scores'].append(metrics['f1_score'])

        # Calculate statistics
        for dataset, values in aggregated.items():
            if values['accuracies']:
                accuracies = values['accuracies']
                aggregated[dataset]['avg_accuracy'] = sum(accuracies) / len(accuracies)
                aggregated[dataset]['std_accuracy'] = self._calculate_std(accuracies)
                aggregated[dataset]['min_accuracy'] = min(accuracies)
                aggregated[dataset]['max_accuracy'] = max(accuracies)

            if values['f1_scores']:
                f1_scores = values['f1_scores']
                aggregated[dataset]['avg_f1'] = sum(f1_scores) / len(f1_scores)
                aggregated[dataset]['std_f1'] = self._calculate_std(f1_scores)

        return aggregated

    def _aggregate_finetuning_metrics(self, finetuning_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate finetuning metrics across seeds."""
        # Implementation for aggregating finetuning metrics
        # This would depend on the structure of finetuning_results
        return {'placeholder': 'implementation_needed'}

    def _analyze_unified_metrics(self, matrix_results: Dict[str, Any]):
        """Generate unified metric learning analysis."""
        self.logger.info("Analyzing unified metric learning results")

        analysis = {
            'experiment_summary': {
                'total_runs': matrix_results['total_runs'],
                'successful_runs': matrix_results['successful_runs'],
                'success_rate': matrix_results['successful_runs'] / matrix_results['total_runs'] if matrix_results['total_runs'] > 0 else 0
            },
            'performance_metrics': matrix_results.get('aggregated_metrics', {}),
            'success_criteria_check': self._check_success_criteria(matrix_results),
            'key_insights': self._generate_key_insights(matrix_results)
        }

        # Save analysis
        analysis_file = self.output_dir / f"unified_metric_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        self.logger.info(f"Saved unified metric learning analysis to: {analysis_file}")

    def _check_success_criteria(self, matrix_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check if experiments meet defined success criteria."""
        criteria_check = {
            'zero_shot_accuracy_threshold': 0.8,  # >80% average accuracy
            'finetuning_accuracy_threshold': 0.95,  # >95% average accuracy
            'computational_efficiency_target': 0.82,  # 82% efficiency (30 vs 150 runs)
            'results': {}
        }

        aggregated = matrix_results.get('aggregated_metrics', {})

        # Check zero-shot criteria
        if 'zero_shot' in aggregated:
            zero_shot_data = aggregated['zero_shot']
            avg_accuracies = [data.get('avg_accuracy', 0) for data in zero_shot_data.values()]
            overall_zero_shot_acc = sum(avg_accuracies) / len(avg_accuracies) if avg_accuracies else 0

            criteria_check['results']['zero_shot'] = {
                'overall_accuracy': overall_zero_shot_acc,
                'meets_threshold': overall_zero_shot_acc >= criteria_check['zero_shot_accuracy_threshold'],
                'threshold': criteria_check['zero_shot_accuracy_threshold']
            }

        # Check computational efficiency
        actual_runs = matrix_results['total_runs']
        baseline_runs = 150  # Traditional approach
        efficiency = 1 - (actual_runs / baseline_runs)

        criteria_check['results']['computational_efficiency'] = {
            'actual_runs': actual_runs,
            'baseline_runs': baseline_runs,
            'efficiency': efficiency,
            'meets_target': efficiency >= criteria_check['computational_efficiency_target'],
            'target': criteria_check['computational_efficiency_target']
        }

        return criteria_check

    def _generate_key_insights(self, matrix_results: Dict[str, Any]) -> List[str]:
        """Generate key insights from unified metric learning results."""
        insights = []

        # Success rate insight
        success_rate = matrix_results['successful_runs'] / matrix_results['total_runs'] if matrix_results['total_runs'] > 0 else 0
        if success_rate >= 0.9:
            insights.append(f"High success rate ({success_rate:.1%}) indicates robust experimental setup")
        elif success_rate < 0.7:
            insights.append(f"Low success rate ({success_rate:.1%}) suggests configuration or implementation issues")

        # Efficiency insight
        actual_runs = matrix_results['total_runs']
        if actual_runs <= 30:
            insights.append(f"Achieved computational efficiency target with only {actual_runs} runs vs traditional 150")

        # Zero-shot performance insight
        aggregated = matrix_results.get('aggregated_metrics', {})
        if 'zero_shot' in aggregated:
            zero_shot_data = aggregated['zero_shot']
            if zero_shot_data:
                avg_accuracies = [data.get('avg_accuracy', 0) for data in zero_shot_data.values()]
                if avg_accuracies:
                    overall_acc = sum(avg_accuracies) / len(avg_accuracies)
                    if overall_acc >= 0.8:
                        insights.append(f"Strong zero-shot performance ({overall_acc:.1%}) demonstrates effective unified representations")
                    else:
                        insights.append(f"Zero-shot performance ({overall_acc:.1%}) below target, may need prompt tuning")

        return insights

    def _find_existing_checkpoints_for_seed(self, seed: int) -> Dict[str, Optional[str]]:
        """Find existing checkpoint files for a specific seed."""
        checkpoint_dir = Path(f"results/pipeline_03/unified_metric_learning/seed_{seed}/checkpoints")

        if not checkpoint_dir.exists():
            self.logger.warning(f"No checkpoint directory found for seed {seed}")
            return {}

        # Find checkpoint files
        checkpoints = {}
        for checkpoint_file in checkpoint_dir.glob("*.ckpt"):
            # Extract dataset name from checkpoint filename
            dataset_name = checkpoint_file.stem.split('_')[0]  # Assuming format like 'CWRU_best.ckpt'
            checkpoints[dataset_name] = str(checkpoint_file)

        return checkpoints

    def _analyze_ablation_results(self, results: Dict[str, Any]):
        """Generate ablation study analysis."""
        self.logger.info("Analyzing ablation study results")

        # Extract key metrics for comparison
        analysis = {
            'comparison_summary': {},
            'performance_ranking': [],
            'key_insights': [],
            'prompt_effectiveness': {}
        }

        for config_name, result in results.items():
            if result.get('status') == 'failed':
                continue

            # Extract performance metrics
            analysis['comparison_summary'][config_name] = {
                'pretraining_time': result.get('pretraining_time', 0),
                'finetuning_time': result.get('finetuning_time', 0),
                'total_time': result.get('total_time', 0)
            }

            # Extract zero-shot results if available
            if 'zero_shot_results' in result:
                zero_shot_data = result['zero_shot_results']
                avg_accuracies = []
                for dataset, metrics in zero_shot_data.items():
                    if 'accuracy' in metrics:
                        avg_accuracies.append(metrics['accuracy'])

                if avg_accuracies:
                    analysis['comparison_summary'][config_name]['avg_zero_shot_accuracy'] = sum(avg_accuracies) / len(avg_accuracies)

        # Rank configurations by performance
        configs_with_accuracy = {
            config: data for config, data in analysis['comparison_summary'].items()
            if 'avg_zero_shot_accuracy' in data
        }

        if configs_with_accuracy:
            analysis['performance_ranking'] = sorted(
                configs_with_accuracy.items(),
                key=lambda x: x[1]['avg_zero_shot_accuracy'],
                reverse=True
            )

        # Generate prompt effectiveness insights
        analysis['prompt_effectiveness'] = self._analyze_prompt_effectiveness(analysis['comparison_summary'])

        # Save analysis
        analysis_file = self.output_dir / f"ablation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        self.logger.info(f"Saved ablation analysis to: {analysis_file}")

    def _analyze_prompt_effectiveness(self, comparison_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze effectiveness of different prompt configurations."""
        effectiveness = {
            'system_prompt_only': None,
            'sample_prompt_only': None,
            'no_prompt_baseline': None,
            'conclusions': []
        }

        # Extract results for each configuration type
        for config_name, metrics in comparison_summary.items():
            if 'system_prompt_only' in config_name:
                effectiveness['system_prompt_only'] = metrics.get('avg_zero_shot_accuracy')
            elif 'sample_prompt_only' in config_name:
                effectiveness['sample_prompt_only'] = metrics.get('avg_zero_shot_accuracy')
            elif 'no_prompt_baseline' in config_name:
                effectiveness['no_prompt_baseline'] = metrics.get('avg_zero_shot_accuracy')

        # Generate conclusions
        baseline_acc = effectiveness['no_prompt_baseline']
        system_acc = effectiveness['system_prompt_only']
        sample_acc = effectiveness['sample_prompt_only']

        if all(x is not None for x in [baseline_acc, system_acc, sample_acc]):
            if system_acc > baseline_acc:
                effectiveness['conclusions'].append(f"System prompts improve performance by {((system_acc - baseline_acc) * 100):.1f} percentage points")

            if sample_acc > baseline_acc:
                effectiveness['conclusions'].append(f"Sample prompts improve performance by {((sample_acc - baseline_acc) * 100):.1f} percentage points")

            if system_acc > sample_acc:
                effectiveness['conclusions'].append("System-level prompts (Dataset_id + Domain_id) more effective than sample-level prompts")
            elif sample_acc > system_acc:
                effectiveness['conclusions'].append("Sample-level prompts (Sample_rate) more effective than system-level prompts")

        return effectiveness
    
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate markdown summary report."""
        markdown = f"""# HSE Prompt Pipeline_03 Experiment Report

**Report Generated:** {report_data['report_generated']}
**Total Experiments:** {report_data['total_experiments']}

## Summary

This report contains results from HSE Prompt-guided contrastive learning experiments
using the Pipeline_03 two-stage training workflow.

## Experiments

"""
        
        for i, experiment in enumerate(report_data['experiments'], 1):
            experiment_name = experiment.get('experiment_name', f'Experiment {i}')
            status = experiment.get('status', 'unknown')
            
            markdown += f"""### {i}. {experiment_name}

- **Status:** {status}
- **Config:** {experiment.get('config_path', 'N/A')}
- **Start Time:** {experiment.get('start_time', 'N/A')}
- **End Time:** {experiment.get('end_time', 'N/A')}

"""
            
            if 'error' in experiment:
                markdown += f"**Error:** {experiment['error']}\n\n"
            else:
                if 'pretraining_time' in experiment:
                    markdown += f"**Pretraining Time:** {experiment['pretraining_time']:.2f}s\n\n"
                if 'finetuning_time' in experiment:
                    markdown += f"**Finetuning Time:** {experiment['finetuning_time']:.2f}s\n\n"
        
        markdown += """## Generated by HSE Prompt Pipeline_03 Runner

Part of PHM-Vibench HSE Industrial Contrastive Learning system.
"""
        
        return markdown


def main():
    """Main function for HSE Prompt Pipeline_03 runner."""
    parser = argparse.ArgumentParser(
        description="HSE Prompt Pipeline_03 Experiment Runner"
    )
    parser.add_argument(
        '--experiment_type',
        type=str,
        choices=['unified_metric_learning', 'backbone_comparison', 'ablation_study', 'custom'],
        required=True,
        help='Type of experiment to run'
    )
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to configuration file (required for custom experiments)'
    )
    parser.add_argument(
        '--stage',
        type=str,
        choices=['pretraining', 'finetuning', 'complete'],
        default='complete',
        help='Which stage to run (default: complete)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/hse_prompt_pipeline03',
        help='Output directory for results'
    )
    parser.add_argument(
        '--backbones',
        type=str,
        nargs='*',
        default=['B_08_PatchTST', 'B_04_Dlinear', 'B_06_TimesNet', 'B_09_FNO'],
        help='Backbones to compare (for backbone_comparison)'
    )
    parser.add_argument(
        '--ablation_configs',
        type=str,
        nargs='*',
        help='Ablation configuration files (for ablation_study)'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        help='Name for custom experiment'
    )
    parser.add_argument(
        '--generate_report',
        action='store_true',
        help='Generate comprehensive report from existing results'
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = HSEPromptPipeline03Runner(args.output_dir)
    
    try:
        if args.generate_report:
            report_path = runner.generate_comprehensive_report()
            print(f"✓ Generated comprehensive report: {report_path}")
            return
        
        # Run specified experiment type
        if args.experiment_type == 'unified_metric_learning':
            results = runner.run_unified_metric_learning(
                config_path=args.config_path,
                stage=args.stage
            )
            successful_runs = results['successful_runs']
            total_runs = results['total_runs']
            print(f"✓ Unified metric learning completed: {successful_runs}/{total_runs} successful runs")

            # Check success criteria
            if 'aggregated_metrics' in results:
                aggregated = results['aggregated_metrics']
                if 'zero_shot' in aggregated:
                    zero_shot_data = aggregated['zero_shot']
                    avg_accuracies = [data.get('avg_accuracy', 0) for data in zero_shot_data.values()]
                    if avg_accuracies:
                        overall_acc = sum(avg_accuracies) / len(avg_accuracies)
                        print(f"📊 Overall zero-shot accuracy: {overall_acc:.1%}")
                        if overall_acc >= 0.8:
                            print("✅ Success criteria met: >80% zero-shot accuracy")
                        else:
                            print("⚠️  Success criteria not met: <80% zero-shot accuracy")

        elif args.experiment_type == 'backbone_comparison':
            if args.config_path is None:
                print("❌ Config path required for backbone comparison")
                sys.exit(1)
            
            results = runner.run_backbone_comparison(
                config_path=args.config_path,
                backbones=args.backbones,
                stage=args.stage
            )
            print(f"✓ Backbone comparison completed with {len(results)} backbones")
            
        elif args.experiment_type == 'ablation_study':
            results = runner.run_ablation_study(
                ablation_configs=args.ablation_configs,
                stage=args.stage
            )
            print(f"✓ Ablation study completed with {len(results)} configurations")
            
        elif args.experiment_type == 'custom':
            if args.config_path is None or args.experiment_name is None:
                print("❌ Config path and experiment name required for custom experiments")
                sys.exit(1)
            
            result = runner.run_custom_experiment(
                config_path=args.config_path,
                experiment_name=args.experiment_name,
                stage=args.stage
            )
            
            if result.get('status') == 'completed':
                print(f"✓ Custom experiment '{args.experiment_name}' completed successfully")
            else:
                print(f"❌ Custom experiment '{args.experiment_name}' failed: {result.get('error')}")
                sys.exit(1)
        
        print(f"\n📊 Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()