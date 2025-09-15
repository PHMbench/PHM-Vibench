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
    
    def _analyze_ablation_results(self, results: Dict[str, Any]):
        """Generate ablation study analysis."""
        self.logger.info("Analyzing ablation study results")
        
        # Extract key metrics for comparison
        analysis = {
            'comparison_summary': {},
            'performance_ranking': [],
            'key_insights': []
        }
        
        for config_name, result in results.items():
            if result.get('status') == 'failed':
                continue
            
            # Extract performance metrics (placeholder implementation)
            analysis['comparison_summary'][config_name] = {
                'pretraining_time': result.get('pretraining_time', 0),
                'finetuning_time': result.get('finetuning_time', 0),
                'total_time': result.get('total_time', 0)
            }
        
        # Save analysis
        analysis_file = self.output_dir / f"ablation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        self.logger.info(f"Saved ablation analysis to: {analysis_file}")
    
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
        choices=['backbone_comparison', 'ablation_study', 'custom'],
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
        if args.experiment_type == 'backbone_comparison':
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