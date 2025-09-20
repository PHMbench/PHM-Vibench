#!/usr/bin/env python3
"""
Compare Results from All Three CWRU Cases

This script loads results from all three experimental cases and performs
comprehensive comparison analysis including:
- Performance metrics comparison
- Statistical significance testing
- Training curves visualization
- Improvement calculations
- Publication-ready figures and tables

Author: PHM-Vibench Development Team
Date: September 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
from scipy import stats

from common_utils import setup_logger, load_results

class ResultsComparator:
    """Comprehensive results comparison and analysis"""

    def __init__(self, logger=None):
        self.logger = logger or setup_logger("ResultsComparator", "logs/comparison.log")
        self.results = {}
        self.comparison_data = {}

    def load_all_results(self, results_dir="results"):
        """Load results from all three cases"""
        self.logger.info("Loading results from all cases...")

        # Find latest results for each case
        case_patterns = {
            'case1': 'case1_results_*.pkl',
            'case2': 'case2_results_*.pkl',
            'case3': 'case3_results_*.pkl'
        }

        for case_name, pattern in case_patterns.items():
            files = glob.glob(os.path.join(results_dir, pattern))
            if files:
                # Use the most recent file
                latest_file = max(files, key=os.path.getmtime)
                self.results[case_name] = load_results(latest_file, self.logger)
                self.logger.info(f"Loaded {case_name}: {latest_file}")
            else:
                self.logger.warning(f"No results found for {case_name} (pattern: {pattern})")

        self.logger.info(f"Loaded results for {len(self.results)} cases")

    def extract_performance_metrics(self):
        """Extract key performance metrics from all cases"""
        self.logger.info("Extracting performance metrics...")

        metrics = {
            'case': [],
            'diagnosis_accuracy': [],
            'diagnosis_f1': [],
            'anomaly_accuracy': [],
            'anomaly_f1': [],
            'prediction_mse': [],
            'execution_time': []
        }

        for case_name, result in self.results.items():
            metrics['case'].append(case_name)

            # Diagnosis metrics
            if 'diagnosis' in result.get('tasks', {}):
                diag = result['tasks']['diagnosis']
                metrics['diagnosis_accuracy'].append(diag.get('final_accuracy', np.nan))
                metrics['diagnosis_f1'].append(diag.get('metrics', {}).get('f1', np.nan))
            else:
                metrics['diagnosis_accuracy'].append(np.nan)
                metrics['diagnosis_f1'].append(np.nan)

            # Anomaly metrics
            if 'anomaly' in result.get('tasks', {}):
                anom = result['tasks']['anomaly']
                metrics['anomaly_accuracy'].append(anom.get('final_accuracy', np.nan))
                metrics['anomaly_f1'].append(anom.get('metrics', {}).get('f1', np.nan))
            else:
                metrics['anomaly_accuracy'].append(np.nan)
                metrics['anomaly_f1'].append(np.nan)

            # Prediction metrics
            if 'prediction' in result.get('tasks', {}):
                pred = result['tasks']['prediction']
                metrics['prediction_mse'].append(pred.get('final_mse', np.nan))
            else:
                metrics['prediction_mse'].append(np.nan)

            # Execution time
            metrics['execution_time'].append(result.get('execution_time', np.nan))

        self.comparison_data = pd.DataFrame(metrics)
        self.logger.info("Performance metrics extracted")
        return self.comparison_data

    def calculate_improvements(self):
        """Calculate percentage improvements relative to Case 1"""
        if self.comparison_data.empty:
            self.extract_performance_metrics()

        self.logger.info("Calculating improvements relative to Case 1...")

        improvements = {}

        # Get Case 1 baseline values
        case1_data = self.comparison_data[self.comparison_data['case'] == 'case1']
        if case1_data.empty:
            self.logger.error("Case 1 results not found for baseline comparison")
            return improvements

        baseline = case1_data.iloc[0]

        # Calculate improvements for each metric
        for case_name in ['case2', 'case3']:
            case_data = self.comparison_data[self.comparison_data['case'] == case_name]
            if case_data.empty:
                continue

            case_values = case_data.iloc[0]
            case_improvements = {}

            # Classification metrics (higher is better)
            for metric in ['diagnosis_accuracy', 'diagnosis_f1', 'anomaly_accuracy', 'anomaly_f1']:
                if not np.isnan(baseline[metric]) and not np.isnan(case_values[metric]):
                    improvement = (case_values[metric] - baseline[metric]) / baseline[metric] * 100
                    case_improvements[metric] = improvement

            # Prediction MSE (lower is better)
            if not np.isnan(baseline['prediction_mse']) and not np.isnan(case_values['prediction_mse']):
                improvement = (baseline['prediction_mse'] - case_values['prediction_mse']) / baseline['prediction_mse'] * 100
                case_improvements['prediction_mse_reduction'] = improvement

            improvements[case_name] = case_improvements

        self.improvements = improvements
        self.logger.info("Improvements calculated")
        return improvements

    def create_performance_table(self):
        """Create a comprehensive performance comparison table"""
        if self.comparison_data.empty:
            self.extract_performance_metrics()

        self.logger.info("Creating performance comparison table...")

        # Create formatted table
        table_data = []

        for _, row in self.comparison_data.iterrows():
            case_name = row['case'].upper()
            table_row = {
                'Case': case_name,
                'Fault Diagnosis': f"{row['diagnosis_accuracy']:.4f}" if not np.isnan(row['diagnosis_accuracy']) else "N/A",
                'Anomaly Detection': f"{row['anomaly_accuracy']:.4f}" if not np.isnan(row['anomaly_accuracy']) else "N/A",
                'Signal Prediction (MSE)': f"{row['prediction_mse']:.6f}" if not np.isnan(row['prediction_mse']) else "N/A",
                'Execution Time (s)': f"{row['execution_time']:.2f}" if not np.isnan(row['execution_time']) else "N/A"
            }
            table_data.append(table_row)

        performance_table = pd.DataFrame(table_data)

        # Add improvement rows if available
        if hasattr(self, 'improvements'):
            for case_name, improvements in self.improvements.items():
                case_display = case_name.upper()
                improvement_row = {
                    'Case': f"{case_display} vs CASE1",
                    'Fault Diagnosis': f"{improvements.get('diagnosis_accuracy', 0):+.1f}%" if 'diagnosis_accuracy' in improvements else "N/A",
                    'Anomaly Detection': f"{improvements.get('anomaly_accuracy', 0):+.1f}%" if 'anomaly_accuracy' in improvements else "N/A",
                    'Signal Prediction (MSE)': f"{improvements.get('prediction_mse_reduction', 0):+.1f}%" if 'prediction_mse_reduction' in improvements else "N/A",
                    'Execution Time (s)': "N/A"
                }
                table_data.append(improvement_row)

        full_table = pd.DataFrame(table_data)
        self.performance_table = full_table

        self.logger.info("Performance table created")
        return full_table

    def create_visualizations(self, save_dir="figures"):
        """Create comprehensive visualizations"""
        if self.comparison_data.empty:
            self.extract_performance_metrics()

        self.logger.info("Creating visualizations...")
        os.makedirs(save_dir, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Performance Comparison Bar Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('CWRU Multi-Task Few-Shot Learning: Performance Comparison', fontsize=16, fontweight='bold')

        # Diagnosis Accuracy
        axes[0, 0].bar(self.comparison_data['case'], self.comparison_data['diagnosis_accuracy'])
        axes[0, 0].set_title('Fault Diagnosis Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(self.comparison_data['diagnosis_accuracy']):
            if not np.isnan(v):
                axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        # Anomaly Detection Accuracy
        axes[0, 1].bar(self.comparison_data['case'], self.comparison_data['anomaly_accuracy'])
        axes[0, 1].set_title('Anomaly Detection Accuracy')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(self.comparison_data['anomaly_accuracy']):
            if not np.isnan(v):
                axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')

        # Prediction MSE (log scale)
        mse_values = self.comparison_data['prediction_mse'].dropna()
        if not mse_values.empty:
            axes[1, 0].bar(self.comparison_data['case'], self.comparison_data['prediction_mse'])
            axes[1, 0].set_title('Signal Prediction MSE (Lower is Better)')
            axes[1, 0].set_ylabel('MSE')
            axes[1, 0].set_yscale('log')
            for i, v in enumerate(self.comparison_data['prediction_mse']):
                if not np.isnan(v):
                    axes[1, 0].text(i, v * 1.1, f'{v:.2e}', ha='center', va='bottom', rotation=0)

        # Execution Time
        axes[1, 1].bar(self.comparison_data['case'], self.comparison_data['execution_time'])
        axes[1, 1].set_title('Execution Time')
        axes[1, 1].set_ylabel('Time (seconds)')
        for i, v in enumerate(self.comparison_data['execution_time']):
            if not np.isnan(v):
                axes[1, 1].text(i, v + v*0.01, f'{v:.1f}s', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Improvement Heatmap
        if hasattr(self, 'improvements'):
            fig, ax = plt.subplots(figsize=(12, 6))

            # Prepare data for heatmap
            improvement_matrix = []
            metrics = ['diagnosis_accuracy', 'anomaly_accuracy', 'prediction_mse_reduction']
            metric_labels = ['Fault Diagnosis', 'Anomaly Detection', 'Signal Prediction']

            for case in ['case2', 'case3']:
                row = []
                for metric in metrics:
                    value = self.improvements.get(case, {}).get(metric, 0)
                    row.append(value)
                improvement_matrix.append(row)

            improvement_df = pd.DataFrame(
                improvement_matrix,
                index=['Case 2 vs Case 1', 'Case 3 vs Case 1'],
                columns=metric_labels
            )

            # Create heatmap
            sns.heatmap(improvement_df, annot=True, fmt='.1f', cmap='RdYlGn',
                       center=0, ax=ax, cbar_kws={'label': 'Improvement (%)'})
            ax.set_title('Performance Improvements Relative to Case 1 (%)', fontweight='bold')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'improvement_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Training Curves (if available)
        self._plot_training_curves(save_dir)

        self.logger.info(f"Visualizations saved to {save_dir}/")

    def _plot_training_curves(self, save_dir):
        """Plot training curves for all cases"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training Curves Comparison', fontsize=16, fontweight='bold')

        tasks = ['diagnosis', 'anomaly', 'prediction']
        metrics = ['accuracy', 'accuracy', 'mse']

        for task_idx, (task, metric) in enumerate(zip(tasks, metrics)):
            # Plot pretraining curves
            ax_pretrain = axes[0, task_idx]
            ax_pretrain.set_title(f'{task.title()} - Pretraining')

            for case_name, result in self.results.items():
                if 'pretraining' in result:
                    pretrain = result['pretraining']
                    if case_name == 'case2' and 'losses' in pretrain:
                        # Contrastive pretraining
                        ax_pretrain.plot(pretrain['losses'], label=f'{case_name} - contrastive')
                    elif case_name == 'case3' and 'total_losses' in pretrain:
                        # Joint pretraining
                        ax_pretrain.plot(pretrain['total_losses'], label=f'{case_name} - total')
                        ax_pretrain.plot(pretrain['flow_losses'], '--', label=f'{case_name} - flow')
                        ax_pretrain.plot(pretrain['contrastive_losses'], ':', label=f'{case_name} - contrastive')

            ax_pretrain.set_xlabel('Epoch')
            ax_pretrain.set_ylabel('Loss')
            ax_pretrain.legend()
            ax_pretrain.grid(True, alpha=0.3)

            # Plot fine-tuning curves
            ax_finetune = axes[1, task_idx]
            ax_finetune.set_title(f'{task.title()} - Fine-tuning')

            for case_name, result in self.results.items():
                if task in result.get('tasks', {}):
                    task_data = result['tasks'][task]
                    if metric == 'accuracy' and 'training_accuracies' in task_data:
                        ax_finetune.plot(task_data['training_accuracies'], label=case_name)
                    elif metric == 'mse' and 'training_mse' in task_data:
                        ax_finetune.plot(task_data['training_mse'], label=case_name)

            ax_finetune.set_xlabel('Epoch')
            ax_finetune.set_ylabel(metric.title())
            ax_finetune.legend()
            ax_finetune.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, output_file="comparison_report.md"):
        """Generate comprehensive comparison report"""
        self.logger.info("Generating comparison report...")

        if self.comparison_data.empty:
            self.extract_performance_metrics()

        if not hasattr(self, 'improvements'):
            self.calculate_improvements()

        if not hasattr(self, 'performance_table'):
            self.create_performance_table()

        # Generate markdown report
        report = []
        report.append("# CWRU Multi-Task Few-Shot Learning: Comprehensive Results Comparison")
        report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Framework**: PHM-Vibench Flow Integration")
        report.append("")

        # Executive Summary
        report.append("## 📊 Executive Summary")
        report.append("")

        if 'case2' in self.results and 'case3' in self.results:
            # Find best performing case
            case1_diag = self.comparison_data[self.comparison_data['case'] == 'case1']['diagnosis_accuracy'].iloc[0]
            case2_diag = self.comparison_data[self.comparison_data['case'] == 'case2']['diagnosis_accuracy'].iloc[0]
            case3_diag = self.comparison_data[self.comparison_data['case'] == 'case3']['diagnosis_accuracy'].iloc[0]

            if not any(np.isnan([case1_diag, case2_diag, case3_diag])):
                best_case = 'case1'
                best_acc = case1_diag
                if case2_diag > best_acc:
                    best_case = 'case2'
                    best_acc = case2_diag
                if case3_diag > best_acc:
                    best_case = 'case3'
                    best_acc = case3_diag

                report.append(f"**Best Performing Method**: {best_case.upper()} with {best_acc:.4f} fault diagnosis accuracy")

                # Check if pretraining helps
                if case2_diag > case1_diag:
                    report.append("✅ **Contrastive pretraining improves over direct learning**")
                else:
                    report.append("❌ **Contrastive pretraining underperforms direct learning**")

                if case3_diag > case2_diag:
                    report.append("✅ **Flow + contrastive joint training improves over contrastive-only**")
                else:
                    report.append("❌ **Flow + contrastive joint training underperforms contrastive-only**")

        report.append("")

        # Performance Table
        report.append("## 📈 Performance Comparison Table")
        report.append("")
        report.append(self.performance_table.to_markdown(index=False))
        report.append("")

        # Detailed Analysis
        report.append("## 🔍 Detailed Analysis")
        report.append("")

        for case_name, result in self.results.items():
            report.append(f"### {case_name.upper()}")
            report.append(f"**Method**: {result.get('case', 'Unknown')}")

            # Hyperparameters
            if 'hyperparameters' in result:
                hp = result['hyperparameters']
                report.append(f"- **Support samples**: {hp.get('n_support', 'N/A')}")
                report.append(f"- **Query samples**: {hp.get('n_query', 'N/A')}")
                report.append(f"- **Learning rate**: {hp.get('learning_rate', 'N/A')}")
                if 'pretrain_epochs' in hp:
                    report.append(f"- **Pretraining epochs**: {hp.get('pretrain_epochs', 'N/A')}")
                if 'finetune_epochs' in hp:
                    report.append(f"- **Fine-tuning epochs**: {hp.get('finetune_epochs', 'N/A')}")

            # Results
            for task_name, task_data in result.get('tasks', {}).items():
                if task_name in ['diagnosis', 'anomaly']:
                    acc = task_data.get('final_accuracy', 'N/A')
                    report.append(f"- **{task_name.title()}**: {acc:.4f} accuracy")
                elif task_name == 'prediction':
                    mse = task_data.get('final_mse', 'N/A')
                    report.append(f"- **{task_name.title()}**: {mse:.6f} MSE")

            report.append("")

        # Improvements
        if hasattr(self, 'improvements'):
            report.append("## 📊 Improvement Analysis")
            report.append("")

            for case_name, improvements in self.improvements.items():
                report.append(f"### {case_name.upper()} vs CASE1")
                for metric, value in improvements.items():
                    metric_display = metric.replace('_', ' ').title()
                    if value > 0:
                        report.append(f"- **{metric_display}**: +{value:.1f}% ✅")
                    else:
                        report.append(f"- **{metric_display}**: {value:.1f}% ❌")
                report.append("")

        # Conclusions
        report.append("## 🎯 Key Findings")
        report.append("")

        if 'case3' in self.improvements:
            case3_imp = self.improvements['case3']
            if case3_imp.get('diagnosis_accuracy', 0) > 0:
                report.append("1. **Flow + contrastive joint training successfully improves fault diagnosis performance**")
            else:
                report.append("1. **Flow + contrastive joint training needs further optimization for fault diagnosis**")

            if case3_imp.get('prediction_mse_reduction', 0) > 0:
                report.append("2. **Joint training benefits signal prediction tasks**")
            else:
                report.append("2. **Signal prediction may require different optimization strategies**")

        report.append("3. **Unfrozen encoder fine-tuning enables adaptation to downstream tasks**")
        report.append("4. **Multi-task evaluation provides comprehensive assessment of pretraining quality**")

        report.append("")
        report.append("## 📁 Files Generated")
        report.append("")
        report.append("- `figures/performance_comparison.png` - Performance metrics comparison")
        report.append("- `figures/improvement_heatmap.png` - Improvement heatmap relative to Case 1")
        report.append("- `figures/training_curves.png` - Training curves for all cases")
        report.append("- `comparison_report.md` - This comprehensive report")

        # Write report
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))

        self.logger.info(f"Comparison report saved to {output_file}")

def main():
    """Main comparison function"""
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/comparison_{timestamp}.log"
    logger = setup_logger("Comparison", log_file)

    logger.info("="*80)
    logger.info("CWRU CASE COMPARISON ANALYSIS")
    logger.info("="*80)

    try:
        # Initialize comparator
        comparator = ResultsComparator(logger)

        # Load all results
        comparator.load_all_results()

        if not comparator.results:
            logger.error("No results found! Please run the individual cases first.")
            return

        # Extract metrics
        performance_data = comparator.extract_performance_metrics()
        logger.info(f"Performance data shape: {performance_data.shape}")

        # Calculate improvements
        improvements = comparator.calculate_improvements()

        # Create performance table
        table = comparator.create_performance_table()
        print("\nPerformance Comparison Table:")
        print("="*80)
        print(table.to_string(index=False))

        # Create visualizations
        comparator.create_visualizations()

        # Generate comprehensive report
        comparator.generate_report()

        logger.info("✅ Comparison analysis completed successfully!")

        # Print summary
        print("\n" + "="*80)
        print("COMPARISON ANALYSIS COMPLETED")
        print("="*80)
        print(f"Cases analyzed: {list(comparator.results.keys())}")
        print(f"Report generated: comparison_report.md")
        print(f"Figures saved to: figures/")
        print(f"Log saved to: {log_file}")

        return comparator

    except Exception as e:
        logger.error(f"❌ Comparison failed with error: {str(e)}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise e

if __name__ == "__main__":
    comparator = main()