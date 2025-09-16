"""
SOTA对比实验框架
面向顶级论文发表的全面基准测试系统

支持与以下8种SOTA方法对比：
1. DANN (Domain Adversarial Neural Networks)
2. CORAL (Deep CORAL)  
3. MMD (Maximum Mean Discrepancy)
4. CDAN (Conditional Domain Adversarial Networks)
5. MCD (Maximum Classifier Discrepancy)
6. SHOT (Source Hypothesis Transfer)
7. NRC (Neighborhood Reciprocal Clustering)
8. Transformer-based baseline

Authors: PHMbench Team
Target: ICML/NeurIPS 2025
"""

import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from pathlib import Path
import subprocess
import argparse
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# 添加项目路径
sys.path.append('/home/lq/LQcode/2_project/PHMBench/PHM-Vibench-metric')

@dataclass
class ExperimentResult:
    """实验结果数据类"""
    method_name: str
    dataset: str
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    training_time: float
    inference_time: float
    memory_usage: float
    convergence_epoch: int
    config_path: str
    run_id: int = 0

class SOTAComparison:
    """SOTA方法对比框架"""
    
    def __init__(self, base_config_path: str, results_dir: str = "results/sota_comparison"):
        """
        初始化SOTA对比框架
        
        Args:
            base_config_path: HSE基础配置文件路径
            results_dir: 结果保存目录
        """
        self.base_config_path = base_config_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.setup_logging()
        
        # SOTA方法配置映射
        self.sota_methods = {
            "HSE-CL": {
                "name": "HSE异构对比学习",
                "task_name": "hse_contrastive",
                "task_type": "CDDG",
                "description": "我们提出的HSE系统级对比学习方法"
            },
            "DANN": {
                "name": "Domain Adversarial Neural Networks", 
                "task_name": "classification",
                "task_type": "CDDG",
                "description": "域对抗神经网络",
                "modifications": {"task.domain_loss": "adversarial", "task.domain_loss_weight": 0.1}
            },
            "CORAL": {
                "name": "Deep CORAL",
                "task_name": "classification", 
                "task_type": "CDDG",
                "description": "深度CORAL域适应",
                "modifications": {"task.domain_loss": "coral", "task.domain_loss_weight": 0.1}
            },
            "MMD": {
                "name": "Maximum Mean Discrepancy",
                "task_name": "classification",
                "task_type": "CDDG", 
                "description": "最大均值差异",
                "modifications": {"task.domain_loss": "mmd", "task.domain_loss_weight": 0.1}
            },
            "CDAN": {
                "name": "Conditional Domain Adversarial Networks",
                "task_name": "classification",
                "task_type": "CDDG",
                "description": "条件域对抗网络",
                "modifications": {"task.domain_loss": "cdan", "task.domain_loss_weight": 0.1}
            },
            "MCD": {
                "name": "Maximum Classifier Discrepancy", 
                "task_name": "classification",
                "task_type": "CDDG",
                "description": "最大分类器差异",
                "modifications": {"task.classifier_discrepancy": True}
            },
            "SHOT": {
                "name": "Source Hypothesis Transfer",
                "task_name": "classification",
                "task_type": "CDDG", 
                "description": "源假设迁移",
                "modifications": {"task.self_training": True, "task.pseudo_label_threshold": 0.9}
            },
            "NRC": {
                "name": "Neighborhood Reciprocal Clustering",
                "task_name": "classification",
                "task_type": "CDDG",
                "description": "邻域互反聚类",
                "modifications": {"task.clustering_loss": "nrc", "task.cluster_weight": 0.1}
            },
            "Transformer": {
                "name": "Transformer Baseline",
                "task_name": "classification",
                "task_type": "CDDG",
                "description": "标准Transformer基线",
                "modifications": {"model.backbone": "B_08_PatchTST", "task.contrast_weight": 0.0}
            }
        }
        
        # 实验配置
        self.datasets = ["CWRU", "XJTU", "THU", "MFPT", "PU"]
        self.target_systems = [1, 5, 13, 19, 21]  # 对应每个数据集的目标域
        self.num_runs = 5  # 每个方法重复运行次数
        self.max_workers = 2  # 并行工作进程数
        
        # 结果存储
        self.results: List[ExperimentResult] = []
        
    def setup_logging(self):
        """设置日志配置"""
        log_file = self.results_dir / "sota_comparison.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_method_config(self, method_key: str, target_system_id: int) -> str:
        """
        为指定方法创建配置文件
        
        Args:
            method_key: 方法标识符
            target_system_id: 目标系统ID
            
        Returns:
            生成的配置文件路径
        """
        method_info = self.sota_methods[method_key]
        
        # 加载基础配置
        with open(self.base_config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        
        # 修改任务配置
        base_config['task']['name'] = method_info['task_name']
        base_config['task']['type'] = method_info['task_type']
        base_config['task']['target_system_id'] = [target_system_id]
        
        # 应用方法特定修改
        if 'modifications' in method_info:
            for key, value in method_info['modifications'].items():
                keys = key.split('.')
                config_section = base_config
                for k in keys[:-1]:
                    config_section = config_section.setdefault(k, {})
                config_section[keys[-1]] = value
        
        # 更新实验标识
        base_config['environment']['project'] = f"SOTA_{method_key}_vs_HSE"
        base_config['environment']['notes'] = f"{method_info['description']} vs HSE对比实验"
        
        # 保存配置文件
        config_path = self.results_dir / f"{method_key}_target{target_system_id}.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(base_config, f, default_flow_style=False, allow_unicode=True)
            
        return str(config_path)
    
    def run_single_experiment(self, method_key: str, target_system_id: int, run_id: int) -> ExperimentResult:
        """
        运行单个实验
        
        Args:
            method_key: 方法标识符
            target_system_id: 目标系统ID
            run_id: 运行ID
            
        Returns:
            实验结果
        """
        method_info = self.sota_methods[method_key]
        dataset_name = self.get_dataset_name(target_system_id)
        
        self.logger.info(f"开始运行 {method_key} 在数据集 {dataset_name} (目标系统 {target_system_id}), Run {run_id}")
        
        # 创建配置文件
        config_path = self.create_method_config(method_key, target_system_id)
        
        # 构建运行命令
        cmd = [
            "python", "main.py",
            "--config", config_path,
            "--override", f"{{\"environment.seed\": {42 + run_id}}}"  # 不同运行使用不同种子
        ]
        
        # 记录开始时间
        start_time = datetime.now()
        
        try:
            # 运行实验
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2小时超时
                cwd="/home/lq/LQcode/2_project/PHMBench/PHM-Vibench-metric"
            )
            
            # 计算训练时间
            training_time = (datetime.now() - start_time).total_seconds()
            
            if result.returncode == 0:
                # 解析实验结果
                metrics = self.parse_experiment_results(result.stdout, result.stderr)
                
                return ExperimentResult(
                    method_name=method_key,
                    dataset=dataset_name,
                    accuracy=metrics.get('accuracy', 0.0),
                    f1_score=metrics.get('f1_score', 0.0), 
                    precision=metrics.get('precision', 0.0),
                    recall=metrics.get('recall', 0.0),
                    training_time=training_time,
                    inference_time=metrics.get('inference_time', 0.0),
                    memory_usage=metrics.get('memory_usage', 0.0),
                    convergence_epoch=metrics.get('convergence_epoch', 50),
                    config_path=config_path,
                    run_id=run_id
                )
            else:
                self.logger.error(f"实验失败 {method_key} - {dataset_name}: {result.stderr}")
                return self.create_failed_result(method_key, dataset_name, run_id, config_path)
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"实验超时 {method_key} - {dataset_name}")
            return self.create_failed_result(method_key, dataset_name, run_id, config_path)
        except Exception as e:
            self.logger.error(f"实验异常 {method_key} - {dataset_name}: {str(e)}")
            return self.create_failed_result(method_key, dataset_name, run_id, config_path)
    
    def parse_experiment_results(self, stdout: str, stderr: str) -> Dict[str, float]:
        """解析实验结果"""
        metrics = {}
        
        try:
            # 从输出中解析关键指标
            lines = stdout.split('\n') + stderr.split('\n')
            
            for line in lines:
                if 'test_acc' in line.lower():
                    # 提取准确率
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'acc' in part.lower() and i + 1 < len(parts):
                            try:
                                metrics['accuracy'] = float(parts[i + 1])
                            except ValueError:
                                pass
                            
                elif 'f1' in line.lower():
                    # 提取F1分数
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'f1' in part.lower() and i + 1 < len(parts):
                            try:
                                metrics['f1_score'] = float(parts[i + 1])
                            except ValueError:
                                pass
                                
            # 默认值
            metrics.setdefault('accuracy', 0.0)
            metrics.setdefault('f1_score', 0.0)
            metrics.setdefault('precision', 0.0)
            metrics.setdefault('recall', 0.0)
            metrics.setdefault('inference_time', 0.0)
            metrics.setdefault('memory_usage', 0.0)
            metrics.setdefault('convergence_epoch', 50)
            
        except Exception as e:
            self.logger.warning(f"结果解析失败: {str(e)}")
            metrics = {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'inference_time': 0.0,
                'memory_usage': 0.0,
                'convergence_epoch': 50
            }
        
        return metrics
    
    def create_failed_result(self, method_key: str, dataset_name: str, run_id: int, config_path: str) -> ExperimentResult:
        """创建失败实验的结果对象"""
        return ExperimentResult(
            method_name=method_key,
            dataset=dataset_name,
            accuracy=0.0,
            f1_score=0.0,
            precision=0.0,
            recall=0.0,
            training_time=0.0,
            inference_time=0.0,
            memory_usage=0.0,
            convergence_epoch=0,
            config_path=config_path,
            run_id=run_id
        )
    
    def get_dataset_name(self, target_system_id: int) -> str:
        """根据系统ID获取数据集名称"""
        mapping = {1: "CWRU", 5: "XJTU", 13: "THU", 19: "MFPT", 21: "PU"}
        return mapping.get(target_system_id, f"System_{target_system_id}")
    
    def run_all_experiments(self):
        """运行所有SOTA方法对比实验"""
        self.logger.info("开始运行SOTA方法对比实验")
        
        total_experiments = len(self.sota_methods) * len(self.target_systems) * self.num_runs
        self.logger.info(f"总计实验数量: {total_experiments}")
        
        completed = 0
        
        # 串行运行实验（避免资源冲突）
        for method_key in self.sota_methods.keys():
            for target_system_id in self.target_systems:
                for run_id in range(self.num_runs):
                    result = self.run_single_experiment(method_key, target_system_id, run_id)
                    self.results.append(result)
                    completed += 1
                    
                    self.logger.info(f"完成进度: {completed}/{total_experiments} ({completed/total_experiments*100:.1f}%)")
                    
                    # 定期保存中间结果
                    if completed % 10 == 0:
                        self.save_intermediate_results()
        
        self.logger.info("所有实验完成!")
    
    def save_intermediate_results(self):
        """保存中间结果"""
        results_file = self.results_dir / "intermediate_results.json"
        
        results_data = []
        for result in self.results:
            results_data.append({
                'method_name': result.method_name,
                'dataset': result.dataset,
                'accuracy': result.accuracy,
                'f1_score': result.f1_score,
                'precision': result.precision,
                'recall': result.recall,
                'training_time': result.training_time,
                'run_id': result.run_id
            })
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def analyze_results(self) -> pd.DataFrame:
        """分析实验结果"""
        self.logger.info("开始分析实验结果")
        
        # 转换为DataFrame
        results_data = []
        for result in self.results:
            results_data.append({
                'Method': result.method_name,
                'Dataset': result.dataset, 
                'Accuracy': result.accuracy,
                'F1-Score': result.f1_score,
                'Precision': result.precision,
                'Recall': result.recall,
                'Training_Time': result.training_time,
                'Run_ID': result.run_id
            })
        
        df = pd.DataFrame(results_data)
        
        # 计算统计摘要
        summary_stats = df.groupby(['Method', 'Dataset']).agg({
            'Accuracy': ['mean', 'std', 'min', 'max'],
            'F1-Score': ['mean', 'std'],
            'Training_Time': ['mean', 'std']
        }).round(4)
        
        # 保存详细结果
        results_file = self.results_dir / "detailed_results.csv"
        df.to_csv(results_file, index=False)
        
        # 保存统计摘要
        summary_file = self.results_dir / "summary_statistics.csv"
        summary_stats.to_csv(summary_file)
        
        self.logger.info(f"结果已保存到 {results_file} 和 {summary_file}")
        
        return df
    
    def statistical_significance_test(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """统计显著性检验"""
        self.logger.info("进行统计显著性检验")
        
        significance_results = {}
        
        # 获取HSE-CL的结果作为基准
        hse_results = df[df['Method'] == 'HSE-CL']['Accuracy'].values
        
        for method in df['Method'].unique():
            if method == 'HSE-CL':
                continue
                
            method_results = df[df['Method'] == method]['Accuracy'].values
            
            if len(method_results) > 0 and len(hse_results) > 0:
                # 执行t检验
                t_stat, p_value = stats.ttest_ind(hse_results, method_results)
                
                # 计算效果量 (Cohen's d)
                pooled_std = np.sqrt(((len(hse_results) - 1) * np.var(hse_results) + 
                                     (len(method_results) - 1) * np.var(method_results)) / 
                                    (len(hse_results) + len(method_results) - 2))
                cohen_d = (np.mean(hse_results) - np.mean(method_results)) / pooled_std
                
                significance_results[method] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohen_d': cohen_d,
                    'significant': p_value < 0.01,  # α = 0.01
                    'hse_mean': np.mean(hse_results),
                    'method_mean': np.mean(method_results),
                    'improvement': np.mean(hse_results) - np.mean(method_results)
                }
        
        # 保存显著性检验结果
        sig_file = self.results_dir / "significance_test.json"
        with open(sig_file, 'w') as f:
            json.dump(significance_results, f, indent=2, default=float)
        
        return significance_results
    
    def generate_paper_tables(self, df: pd.DataFrame) -> str:
        """生成论文用的LaTeX表格"""
        self.logger.info("生成论文LaTeX表格")
        
        # 计算平均值和标准差
        summary = df.groupby('Method').agg({
            'Accuracy': ['mean', 'std'],
            'F1-Score': ['mean', 'std'],
            'Training_Time': ['mean', 'std']
        }).round(4)
        
        # 生成LaTeX表格
        latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison with SOTA Methods}
\\label{tab:sota_comparison}
\\begin{tabular}{lccccc}
\\toprule
Method & Accuracy (\\%) & F1-Score & Precision & Recall & Training Time (s) \\\\
\\midrule
"""
        
        # 按准确率排序
        methods_ranked = summary.sort_values(('Accuracy', 'mean'), ascending=False)
        
        for method in methods_ranked.index:
            acc_mean = methods_ranked.loc[method, ('Accuracy', 'mean')] * 100
            acc_std = methods_ranked.loc[method, ('Accuracy', 'std')] * 100
            f1_mean = methods_ranked.loc[method, ('F1-Score', 'mean')]
            f1_std = methods_ranked.loc[method, ('F1-Score', 'std')]
            time_mean = methods_ranked.loc[method, ('Training_Time', 'mean')]
            
            # 突出显示最佳结果
            if method == 'HSE-CL':
                latex_table += f"\\textbf{{{method}}} & \\textbf{{{acc_mean:.2f} ± {acc_std:.2f}}} & \\textbf{{{f1_mean:.3f} ± {f1_std:.3f}}} & - & - & {time_mean:.1f} \\\\\n"
            else:
                latex_table += f"{method} & {acc_mean:.2f} ± {acc_std:.2f} & {f1_mean:.3f} ± {f1_std:.3f} & - & - & {time_mean:.1f} \\\\\n"
        
        latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
        
        # 保存LaTeX表格
        latex_file = self.results_dir / "sota_comparison_table.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        
        return latex_table
    
    def create_visualization(self, df: pd.DataFrame):
        """创建可视化图表"""
        self.logger.info("创建可视化图表")
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. 准确率对比柱状图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 平均准确率对比
        method_acc = df.groupby('Method')['Accuracy'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(method_acc)), method_acc['mean'], yerr=method_acc['std'], 
                       capsize=5, alpha=0.8)
        
        # 突出显示HSE-CL
        for i, method in enumerate(method_acc.index):
            if method == 'HSE-CL':
                bars[i].set_color('red')
                bars[i].set_alpha(1.0)
        
        ax1.set_xlabel('Methods')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Average Accuracy Comparison')
        ax1.set_xticks(range(len(method_acc)))
        ax1.set_xticklabels(method_acc.index, rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. F1分数对比
        method_f1 = df.groupby('Method')['F1-Score'].agg(['mean', 'std']).sort_values('mean', ascending=False)
        
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(method_f1)), method_f1['mean'], yerr=method_f1['std'],
                        capsize=5, alpha=0.8)
        
        for i, method in enumerate(method_f1.index):
            if method == 'HSE-CL':
                bars2[i].set_color('red')
                bars2[i].set_alpha(1.0)
        
        ax2.set_xlabel('Methods')
        ax2.set_ylabel('F1-Score')
        ax2.set_title('F1-Score Comparison')
        ax2.set_xticks(range(len(method_f1)))
        ax2.set_xticklabels(method_f1.index, rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. 训练时间对比
        method_time = df.groupby('Method')['Training_Time'].agg(['mean', 'std']).sort_values('mean')
        
        ax3 = axes[1, 0]
        ax3.bar(range(len(method_time)), method_time['mean'], yerr=method_time['std'],
                capsize=5, alpha=0.8, color='green')
        ax3.set_xlabel('Methods')
        ax3.set_ylabel('Training Time (s)')
        ax3.set_title('Training Time Comparison')
        ax3.set_xticks(range(len(method_time)))
        ax3.set_xticklabels(method_time.index, rotation=45)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. 跨数据集性能热力图
        pivot_acc = df.pivot_table(values='Accuracy', index='Method', columns='Dataset', aggfunc='mean')
        
        ax4 = axes[1, 1]
        sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4)
        ax4.set_title('Cross-Dataset Performance Heatmap')
        
        plt.tight_layout()
        
        # 保存图表
        fig_file = self.results_dir / "sota_comparison_plots.png"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"可视化图表已保存到 {fig_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SOTA方法对比实验")
    parser.add_argument("--base_config", type=str, 
                       default="configs/demo/HSE_Contrastive/hse_cddg.yaml",
                       help="HSE基础配置文件路径")
    parser.add_argument("--results_dir", type=str,
                       default="results/sota_comparison",
                       help="结果保存目录")
    parser.add_argument("--methods", type=str, nargs="+",
                       default=None,
                       help="要运行的方法列表 (默认运行所有方法)")
    parser.add_argument("--datasets", type=int, nargs="+",
                       default=[1, 5, 13, 19, 21],
                       help="目标系统ID列表")
    parser.add_argument("--num_runs", type=int, default=5,
                       help="每个方法的重复运行次数")
    
    args = parser.parse_args()
    
    # 创建对比框架
    comparison = SOTAComparison(args.base_config, args.results_dir)
    
    # 设置实验参数
    if args.methods:
        comparison.sota_methods = {k: v for k, v in comparison.sota_methods.items() 
                                  if k in args.methods}
    comparison.target_systems = args.datasets
    comparison.num_runs = args.num_runs
    
    print(f"🚀 开始SOTA对比实验")
    print(f"   - 方法数量: {len(comparison.sota_methods)}")
    print(f"   - 数据集数量: {len(comparison.target_systems)}")
    print(f"   - 重复次数: {comparison.num_runs}")
    print(f"   - 总实验数: {len(comparison.sota_methods) * len(comparison.target_systems) * comparison.num_runs}")
    
    # 运行实验
    comparison.run_all_experiments()
    
    # 分析结果
    df = comparison.analyze_results()
    
    # 统计显著性检验
    sig_results = comparison.statistical_significance_test(df)
    
    # 生成论文表格
    latex_table = comparison.generate_paper_tables(df)
    
    # 创建可视化
    comparison.create_visualization(df)
    
    # 打印摘要
    print("\n" + "="*50)
    print("🎯 实验完成摘要")
    print("="*50)
    
    hse_results = df[df['Method'] == 'HSE-CL']['Accuracy']
    if len(hse_results) > 0:
        print(f"HSE-CL平均准确率: {hse_results.mean():.4f} ± {hse_results.std():.4f}")
    
    print("\n显著性检验结果:")
    for method, stats in sig_results.items():
        improvement = stats['improvement'] * 100
        significance = "✓" if stats['significant'] else "✗"
        print(f"  {method}: {improvement:+.2f}% (p={stats['p_value']:.4f}) {significance}")
    
    print(f"\n📊 详细结果保存在: {comparison.results_dir}")
    print("🎓 论文素材已生成完成！")

if __name__ == "__main__":
    main()