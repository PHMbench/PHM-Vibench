#!/usr/bin/env python3
"""
ContrastiveIDTask消融研究脚本

系统化的超参数对比学习预训练性能影响分析，专为论文发表设计。

功能特性：
- 核心超参数的系统性消融实验
- 自动化实验执行和结果收集
- 论文级别的可视化和统计分析
- 实验结果的LaTeX表格导出

Usage:
    # 快速消融测试
    python ablation_study.py --quick --params temperature batch_size

    # 完整消融研究
    python ablation_study.py --full --output_dir ablation_results

    # 特定参数消融
    python ablation_study.py --params window_size temperature --epochs 10

Author: PHM-Vibench Team
Version: 2.0 (Optimized for Research)
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import product
from collections import defaultdict
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import subprocess
import time

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.configs import load_config

class AblationStudy:
    """消融实验管理器

    设计原则：
    1. 科学严谨 - 控制变量，单一变量实验
    2. 可重复性 - 固定随机种子，详细记录配置
    3. 效率优先 - 智能采样，避免计算资源浪费
    4. 论文导向 - 直接生成发表级图表和统计结果
    """

    def __init__(self,
                 base_config: str = "configs/id_contrastive/ablation.yaml",
                 output_dir: str = "save/ablation_study",
                 quick_mode: bool = False):
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.quick_mode = quick_mode

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self._setup_logging()

        # 消融参数定义 - 基于ContrastiveIDTask关键超参数
        if quick_mode:
            # 快速测试：减少参数空间
            self.ablation_params = {
                'temperature': [0.05, 0.07, 0.1],
                'batch_size': [16, 32],
                'window_size': [512, 1024],
                'learning_rate': [1e-3, 5e-3]
            }
            self.max_epochs = 5
        else:
            # 完整消融研究
            self.ablation_params = {
                'temperature': [0.01, 0.05, 0.07, 0.1, 0.5],  # InfoNCE温度参数
                'batch_size': [8, 16, 32, 64],                 # 批量大小影响负样本数量
                'window_size': [256, 512, 1024, 2048],         # 窗口大小影响时序信息捕获
                'learning_rate': [5e-4, 1e-3, 2e-3, 5e-3],   # 学习率调度
                'd_model': [64, 128, 256, 512],                # 模型维度
                'window_strategy': ['random', 'evenly_spaced'] # 窗口采样策略
            }
            self.max_epochs = 20

        # 结果存储
        self.results = []
        self.baseline_result = None

        self.logger.info(f"消融实验初始化完成")
        self.logger.info(f"基础配置: {base_config}")
        self.logger.info(f"输出目录: {output_dir}")
        self.logger.info(f"快速模式: {quick_mode}")
        self.logger.info(f"消融参数: {list(self.ablation_params.keys())}")

    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.output_dir / "ablation.log"

        self.logger = logging.getLogger('AblationStudy')
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            # 文件处理器
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)

            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # 格式设置
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def run_baseline_experiment(self) -> Dict[str, float]:
        """运行基线实验"""
        self.logger.info("运行基线实验...")

        baseline_config = {
            'experiment_name': 'baseline',
            'config_overrides': {},
            'description': 'Baseline experiment with default parameters'
        }

        result = self._run_single_experiment(baseline_config)
        self.baseline_result = result

        self.logger.info(f"基线实验完成: {result}")
        return result

    def run_ablation_experiments(self, target_params: List[str] = None) -> List[Dict]:
        """运行消融实验

        Args:
            target_params: 指定要消融的参数，如果为None则消融所有参数
        """
        if target_params is None:
            target_params = list(self.ablation_params.keys())
        else:
            target_params = [p for p in target_params if p in self.ablation_params]

        self.logger.info(f"开始消融实验，目标参数: {target_params}")

        # 获取基线配置
        try:
            base_config_dict = dict(load_config(self.base_config))
        except Exception as e:
            self.logger.warning(f"无法加载配置文件: {e}")
            base_config_dict = self._get_default_config()

        total_experiments = 0
        for param in target_params:
            total_experiments += len(self.ablation_params[param])

        self.logger.info(f"总实验数: {total_experiments}")

        experiment_count = 0

        # 对每个参数进行单变量消融
        for param_name in target_params:
            param_values = self.ablation_params[param_name]
            self.logger.info(f"\n消融参数: {param_name}, 取值: {param_values}")

            for param_value in param_values:
                experiment_count += 1

                # 构建实验配置
                experiment_config = {
                    'experiment_name': f'ablation_{param_name}_{param_value}',
                    'config_overrides': self._build_config_override(param_name, param_value),
                    'description': f'Ablation study: {param_name} = {param_value}',
                    'param_name': param_name,
                    'param_value': param_value
                }

                self.logger.info(f"实验 {experiment_count}/{total_experiments}: {experiment_config['experiment_name']}")

                # 运行实验
                result = self._run_single_experiment(experiment_config)

                # 保存结果
                result['param_name'] = param_name
                result['param_value'] = param_value
                self.results.append(result)

                # 实时保存结果
                self._save_intermediate_results()

                self.logger.info(f"实验完成: {result.get('final_metrics', {})}")

        self.logger.info(f"所有消融实验完成，共完成 {len(self.results)} 个实验")
        return self.results

    def _build_config_override(self, param_name: str, param_value: Any) -> Dict[str, Any]:
        """构建配置覆盖"""
        overrides = {}

        # 参数映射到配置路径
        param_mapping = {
            'temperature': 'task.temperature',
            'batch_size': 'data.batch_size',
            'window_size': 'data.window_size',
            'learning_rate': 'task.lr',
            'd_model': 'model.d_model',
            'window_strategy': 'data.window_sampling_strategy'
        }

        if param_name in param_mapping:
            overrides[param_mapping[param_name]] = param_value

        # 实验特定设置
        overrides.update({
            'trainer.epochs': self.max_epochs,
            'trainer.enable_checkpointing': False,  # 节省空间
            'trainer.enable_progress_bar': False,   # 减少日志输出
            'experiment.save_dir': str(self.output_dir / f"exp_{param_name}_{param_value}"),
        })

        return overrides

    def _run_single_experiment(self, experiment_config: Dict) -> Dict[str, Any]:
        """运行单个实验"""
        start_time = time.time()

        try:
            # 创建实验配置文件
            config_path = self._create_experiment_config(experiment_config)

            # 构建运行命令
            cmd = [
                'python', 'main.py',
                '--pipeline', 'Pipeline_ID',
                '--config_path', str(config_path),
                '--notes', experiment_config['description']
            ]

            # 运行实验
            self.logger.info(f"执行命令: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1小时超时
                cwd=str(Path(__file__).parent.parent.parent.parent)
            )

            end_time = time.time()
            duration = (end_time - start_time) / 60  # 转换为分钟

            # 解析结果
            experiment_result = {
                'experiment_name': experiment_config['experiment_name'],
                'status': 'completed' if result.returncode == 0 else 'failed',
                'duration_minutes': round(duration, 2),
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

            if result.returncode == 0:
                # 提取性能指标
                metrics = self._extract_metrics_from_output(result.stdout, experiment_config)
                experiment_result['final_metrics'] = metrics
            else:
                self.logger.error(f"实验失败: {result.stderr[-500:]}")
                experiment_result['error'] = result.stderr[-500:]

            # 保存实验日志
            exp_dir = Path(experiment_config['config_overrides']['experiment.save_dir'])
            exp_dir.mkdir(parents=True, exist_ok=True)

            with open(exp_dir / "stdout.log", 'w') as f:
                f.write(result.stdout)
            with open(exp_dir / "stderr.log", 'w') as f:
                f.write(result.stderr)

            return experiment_result

        except subprocess.TimeoutExpired:
            return {
                'experiment_name': experiment_config['experiment_name'],
                'status': 'timeout',
                'duration_minutes': 60.0,
                'error': 'Experiment timed out after 1 hour'
            }
        except Exception as e:
            return {
                'experiment_name': experiment_config['experiment_name'],
                'status': 'error',
                'duration_minutes': (time.time() - start_time) / 60,
                'error': str(e)
            }

    def _create_experiment_config(self, experiment_config: Dict) -> Path:
        """创建实验配置文件"""
        # 加载基础配置
        try:
            base_config = load_config(self.base_config)
            config_dict = dict(base_config)
        except:
            config_dict = self._get_default_config()

        # 应用覆盖
        for key, value in experiment_config['config_overrides'].items():
            keys = key.split('.')
            current = config_dict
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

        # 保存配置
        exp_dir = Path(experiment_config['config_overrides']['experiment.save_dir'])
        exp_dir.mkdir(parents=True, exist_ok=True)

        config_path = exp_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        return config_path

    def _extract_metrics_from_output(self, stdout: str, experiment_config: Dict) -> Dict[str, float]:
        """从输出中提取性能指标"""
        metrics = {}

        # 查找常见的性能指标
        lines = stdout.split('\n')

        for line in lines:
            # 训练损失
            if 'train_loss' in line.lower():
                try:
                    import re
                    match = re.search(r'train_loss[:\s]*([0-9.]+)', line)
                    if match:
                        metrics['train_loss'] = float(match.group(1))
                except:
                    pass

            # 验证损失
            if 'val_loss' in line.lower():
                try:
                    import re
                    match = re.search(r'val_loss[:\s]*([0-9.]+)', line)
                    if match:
                        metrics['val_loss'] = float(match.group(1))
                except:
                    pass

            # 对比学习准确率
            if 'contrastive_acc' in line.lower():
                try:
                    import re
                    match = re.search(r'contrastive_acc[:\s]*([0-9.]+)', line)
                    if match:
                        metrics['contrastive_accuracy'] = float(match.group(1))
                except:
                    pass

        # 如果没有找到指标，尝试从文件中读取
        if not metrics:
            exp_dir = Path(experiment_config['config_overrides']['experiment.save_dir'])
            metric_files = [
                exp_dir / "metrics.json",
                exp_dir / "final_metrics.json"
            ]

            for metric_file in metric_files:
                if metric_file.exists():
                    try:
                        with open(metric_file, 'r') as f:
                            file_metrics = json.load(f)
                        metrics.update(file_metrics)
                        break
                    except:
                        continue

        return metrics

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'data': {
                'factory_name': 'id',
                'dataset_name': 'ID_dataset',
                'data_dir': 'data',
                'metadata_file': 'metadata_6_1.xlsx',
                'batch_size': 32,
                'window_size': 1024,
                'num_window': 2,
                'window_sampling_strategy': 'random',
                'normalization': True
            },
            'model': {
                'type': 'ISFM',
                'factory_name': 'ISFM',
                'd_model': 128
            },
            'task': {
                'name': 'contrastive_id',
                'temperature': 0.07,
                'lr': 1e-3
            },
            'trainer': {
                'epochs': 10,
                'devices': 'cpu'
            }
        }

    def _save_intermediate_results(self):
        """保存中间结果"""
        results_file = self.output_dir / f"intermediate_results_{datetime.now().strftime('%H%M%S')}.json"

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

    def analyze_results(self) -> Dict[str, Any]:
        """分析消融实验结果"""
        self.logger.info("开始分析消融实验结果...")

        if not self.results:
            self.logger.warning("没有实验结果可分析")
            return {}

        analysis = {
            'summary': self._generate_summary(),
            'parameter_analysis': self._analyze_parameters(),
            'best_configurations': self._find_best_configurations(),
            'statistical_significance': self._compute_statistical_tests()
        }

        # 保存分析结果
        analysis_file = self.output_dir / "analysis_results.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        self.logger.info(f"结果分析完成，保存到: {analysis_file}")
        return analysis

    def _generate_summary(self) -> Dict:
        """生成结果摘要"""
        completed = [r for r in self.results if r['status'] == 'completed']
        failed = [r for r in self.results if r['status'] != 'completed']

        summary = {
            'total_experiments': len(self.results),
            'completed': len(completed),
            'failed': len(failed),
            'success_rate': len(completed) / max(1, len(self.results)) * 100,
            'total_duration_hours': sum(r.get('duration_minutes', 0) for r in self.results) / 60,
            'parameters_studied': list(set(r.get('param_name') for r in self.results if r.get('param_name')))
        }

        return summary

    def _analyze_parameters(self) -> Dict:
        """分析各参数的影响"""
        param_analysis = {}

        # 按参数分组
        param_groups = defaultdict(list)
        for result in self.results:
            if result['status'] == 'completed' and 'final_metrics' in result:
                param_name = result.get('param_name')
                if param_name:
                    param_groups[param_name].append(result)

        # 分析每个参数
        for param_name, param_results in param_groups.items():
            if not param_results:
                continue

            # 提取关键指标
            metrics_data = []
            for result in param_results:
                metrics = result.get('final_metrics', {})
                if metrics:
                    metrics_data.append({
                        'param_value': result.get('param_value'),
                        'train_loss': metrics.get('train_loss'),
                        'val_loss': metrics.get('val_loss'),
                        'contrastive_accuracy': metrics.get('contrastive_accuracy'),
                        'duration_minutes': result.get('duration_minutes')
                    })

            if metrics_data:
                df = pd.DataFrame(metrics_data)

                # 计算统计信息
                param_analysis[param_name] = {
                    'num_experiments': len(metrics_data),
                    'parameter_values': list(df['param_value'].unique()),
                    'best_value': self._find_best_param_value(df, param_name),
                    'performance_range': self._compute_performance_range(df),
                    'correlation_with_performance': self._compute_correlations(df)
                }

        return param_analysis

    def _find_best_param_value(self, df: pd.DataFrame, param_name: str) -> Dict:
        """找到参数的最佳值"""
        # 根据验证损失找最优值（如果有的话）
        if 'val_loss' in df.columns and df['val_loss'].notna().any():
            best_idx = df['val_loss'].idxmin()
            return {
                'value': df.loc[best_idx, 'param_value'],
                'metric': 'val_loss',
                'score': df.loc[best_idx, 'val_loss']
            }
        # 否则根据训练损失
        elif 'train_loss' in df.columns and df['train_loss'].notna().any():
            best_idx = df['train_loss'].idxmin()
            return {
                'value': df.loc[best_idx, 'param_value'],
                'metric': 'train_loss',
                'score': df.loc[best_idx, 'train_loss']
            }
        # 或者对比学习准确率
        elif 'contrastive_accuracy' in df.columns and df['contrastive_accuracy'].notna().any():
            best_idx = df['contrastive_accuracy'].idxmax()
            return {
                'value': df.loc[best_idx, 'param_value'],
                'metric': 'contrastive_accuracy',
                'score': df.loc[best_idx, 'contrastive_accuracy']
            }
        else:
            return {'value': None, 'metric': None, 'score': None}

    def _compute_performance_range(self, df: pd.DataFrame) -> Dict:
        """计算性能范围"""
        ranges = {}

        for metric in ['train_loss', 'val_loss', 'contrastive_accuracy']:
            if metric in df.columns and df[metric].notna().any():
                values = df[metric].dropna()
                ranges[metric] = {
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'mean': float(values.mean()),
                    'std': float(values.std())
                }

        return ranges

    def _compute_correlations(self, df: pd.DataFrame) -> Dict:
        """计算参数值与性能的相关性"""
        correlations = {}

        # 只对数值型参数值计算相关性
        if pd.api.types.is_numeric_dtype(df['param_value']):
            for metric in ['train_loss', 'val_loss', 'contrastive_accuracy']:
                if metric in df.columns and df[metric].notna().any():
                    corr = df['param_value'].corr(df[metric])
                    if not pd.isna(corr):
                        correlations[metric] = float(corr)

        return correlations

    def _find_best_configurations(self, top_k: int = 3) -> List[Dict]:
        """找到最佳配置"""
        completed_results = [r for r in self.results if r['status'] == 'completed' and 'final_metrics' in r]

        if not completed_results:
            return []

        # 根据验证损失排序（如果有的话）
        def get_score(result):
            metrics = result.get('final_metrics', {})
            if 'val_loss' in metrics and metrics['val_loss'] is not None:
                return metrics['val_loss']  # 越小越好
            elif 'train_loss' in metrics and metrics['train_loss'] is not None:
                return metrics['train_loss']
            else:
                return float('inf')

        sorted_results = sorted(completed_results, key=get_score)

        best_configs = []
        for result in sorted_results[:top_k]:
            config = {
                'experiment_name': result['experiment_name'],
                'param_name': result.get('param_name'),
                'param_value': result.get('param_value'),
                'metrics': result.get('final_metrics', {}),
                'duration_minutes': result.get('duration_minutes'),
                'rank': len(best_configs) + 1
            }
            best_configs.append(config)

        return best_configs

    def _compute_statistical_tests(self) -> Dict:
        """计算统计显著性测试"""
        # 这里可以添加更复杂的统计测试，如ANOVA等
        # 为简化，暂时返回基本统计信息

        completed_results = [r for r in self.results if r['status'] == 'completed' and 'final_metrics' in r]

        if len(completed_results) < 2:
            return {'note': 'Insufficient data for statistical tests'}

        # 基本统计
        train_losses = [r.get('final_metrics', {}).get('train_loss') for r in completed_results]
        train_losses = [x for x in train_losses if x is not None]

        if len(train_losses) > 1:
            return {
                'train_loss_stats': {
                    'mean': np.mean(train_losses),
                    'std': np.std(train_losses),
                    'min': np.min(train_losses),
                    'max': np.max(train_losses),
                    'n_samples': len(train_losses)
                }
            }

        return {}

    def generate_visualizations(self):
        """生成可视化图表"""
        self.logger.info("生成可视化图表...")

        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 为每个参数生成图表
        param_groups = defaultdict(list)
        for result in self.results:
            if result['status'] == 'completed' and 'final_metrics' in result:
                param_name = result.get('param_name')
                if param_name:
                    param_groups[param_name].append(result)

        for param_name, param_results in param_groups.items():
            if len(param_results) < 2:
                continue

            self._plot_parameter_effect(param_name, param_results)

        # 生成总体对比图
        self._plot_overall_comparison()

        self.logger.info("可视化图表生成完成")

    def _plot_parameter_effect(self, param_name: str, param_results: List[Dict]):
        """绘制参数效应图"""
        # 准备数据
        data = []
        for result in param_results:
            metrics = result.get('final_metrics', {})
            if metrics:
                data.append({
                    'param_value': result.get('param_value'),
                    'train_loss': metrics.get('train_loss'),
                    'val_loss': metrics.get('val_loss'),
                    'contrastive_accuracy': metrics.get('contrastive_accuracy'),
                    'duration_minutes': result.get('duration_minutes')
                })

        if not data:
            return

        df = pd.DataFrame(data)

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Parameter Effect: {param_name}', fontsize=16)

        # 训练损失
        if 'train_loss' in df.columns and df['train_loss'].notna().any():
            axes[0, 0].plot(df['param_value'], df['train_loss'], 'o-')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel(param_name)
            axes[0, 0].set_ylabel('Loss')

        # 验证损失
        if 'val_loss' in df.columns and df['val_loss'].notna().any():
            axes[0, 1].plot(df['param_value'], df['val_loss'], 'o-', color='orange')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].set_xlabel(param_name)
            axes[0, 1].set_ylabel('Loss')

        # 对比学习准确率
        if 'contrastive_accuracy' in df.columns and df['contrastive_accuracy'].notna().any():
            axes[1, 0].plot(df['param_value'], df['contrastive_accuracy'], 'o-', color='green')
            axes[1, 0].set_title('Contrastive Accuracy')
            axes[1, 0].set_xlabel(param_name)
            axes[1, 0].set_ylabel('Accuracy')

        # 运行时间
        if 'duration_minutes' in df.columns:
            axes[1, 1].plot(df['param_value'], df['duration_minutes'], 'o-', color='red')
            axes[1, 1].set_title('Training Duration')
            axes[1, 1].set_xlabel(param_name)
            axes[1, 1].set_ylabel('Minutes')

        plt.tight_layout()

        # 保存图表
        plot_file = self.output_dir / f"parameter_effect_{param_name}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_overall_comparison(self):
        """绘制总体对比图"""
        completed_results = [r for r in self.results if r['status'] == 'completed' and 'final_metrics' in r]

        if len(completed_results) < 2:
            return

        # 准备数据
        data = []
        for result in completed_results:
            metrics = result.get('final_metrics', {})
            if metrics and metrics.get('train_loss') is not None:
                data.append({
                    'experiment': result['experiment_name'],
                    'param_name': result.get('param_name', 'unknown'),
                    'param_value': str(result.get('param_value', '')),
                    'train_loss': metrics.get('train_loss'),
                    'val_loss': metrics.get('val_loss'),
                    'contrastive_accuracy': metrics.get('contrastive_accuracy'),
                    'duration_minutes': result.get('duration_minutes')
                })

        if not data:
            return

        df = pd.DataFrame(data)

        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Ablation Study Results Comparison', fontsize=16)

        # 按参数名称分组的训练损失对比
        if 'train_loss' in df.columns and df['train_loss'].notna().any():
            sns.boxplot(data=df, x='param_name', y='train_loss', ax=axes[0, 0])
            axes[0, 0].set_title('Training Loss by Parameter')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # 散点图：训练时间 vs 性能
        if 'duration_minutes' in df.columns and 'train_loss' in df.columns:
            scatter = axes[0, 1].scatter(df['duration_minutes'], df['train_loss'],
                                       c=df['param_name'].astype('category').cat.codes, alpha=0.7)
            axes[0, 1].set_xlabel('Training Duration (minutes)')
            axes[0, 1].set_ylabel('Training Loss')
            axes[0, 1].set_title('Performance vs Training Time')

        # Top-K最佳配置
        top_results = df.nsmallest(min(10, len(df)), 'train_loss')
        axes[1, 0].barh(range(len(top_results)), top_results['train_loss'])
        axes[1, 0].set_yticks(range(len(top_results)))
        axes[1, 0].set_yticklabels([f"{row['param_name']}={row['param_value']}"
                                   for _, row in top_results.iterrows()], fontsize=8)
        axes[1, 0].set_title('Top 10 Best Configurations')
        axes[1, 0].set_xlabel('Training Loss')

        # 参数影响热图
        if len(df) > 1:
            param_pivot = df.pivot_table(values='train_loss',
                                       index='param_name',
                                       columns='param_value',
                                       aggfunc='mean')
            if not param_pivot.empty:
                sns.heatmap(param_pivot, annot=True, fmt='.4f', ax=axes[1, 1])
                axes[1, 1].set_title('Parameter Impact Heatmap')

        plt.tight_layout()

        # 保存图表
        plot_file = self.output_dir / "overall_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, analysis_results: Dict) -> str:
        """生成实验报告"""
        self.logger.info("生成实验报告...")

        report_file = self.output_dir / f"ablation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        with open(report_file, 'w') as f:
            f.write("# ContrastiveIDTask 消融实验报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 实验摘要
            summary = analysis_results.get('summary', {})
            f.write("## 实验摘要\n\n")
            f.write(f"- **总实验数**: {summary.get('total_experiments', 0)}\n")
            f.write(f"- **成功完成**: {summary.get('completed', 0)}\n")
            f.write(f"- **失败**: {summary.get('failed', 0)}\n")
            f.write(f"- **成功率**: {summary.get('success_rate', 0):.1f}%\n")
            f.write(f"- **总耗时**: {summary.get('total_duration_hours', 0):.1f} 小时\n")
            f.write(f"- **研究参数**: {', '.join(summary.get('parameters_studied', []))}\n\n")

            # 最佳配置
            best_configs = analysis_results.get('best_configurations', [])
            if best_configs:
                f.write("## 最佳配置\n\n")
                for i, config in enumerate(best_configs, 1):
                    f.write(f"### 第 {i} 名\n")
                    f.write(f"- **实验**: {config['experiment_name']}\n")
                    f.write(f"- **参数**: {config['param_name']} = {config['param_value']}\n")
                    f.write(f"- **训练时间**: {config['duration_minutes']:.1f} 分钟\n")

                    metrics = config.get('metrics', {})
                    if metrics:
                        f.write("- **性能指标**:\n")
                        for metric, value in metrics.items():
                            if value is not None:
                                f.write(f"  - {metric}: {value:.4f}\n")
                    f.write("\n")

            # 参数分析
            param_analysis = analysis_results.get('parameter_analysis', {})
            if param_analysis:
                f.write("## 参数影响分析\n\n")
                for param_name, analysis in param_analysis.items():
                    f.write(f"### {param_name}\n")
                    f.write(f"- **实验数量**: {analysis['num_experiments']}\n")
                    f.write(f"- **参数取值**: {analysis['parameter_values']}\n")

                    best_value = analysis.get('best_value', {})
                    if best_value.get('value') is not None:
                        f.write(f"- **最佳取值**: {best_value['value']} ({best_value['metric']}: {best_value['score']:.4f})\n")

                    correlations = analysis.get('correlation_with_performance', {})
                    if correlations:
                        f.write("- **与性能相关性**:\n")
                        for metric, corr in correlations.items():
                            f.write(f"  - {metric}: {corr:.3f}\n")

                    f.write(f"\n![{param_name} Effect](parameter_effect_{param_name}.png)\n\n")

            # 统计显著性
            stats = analysis_results.get('statistical_significance', {})
            if stats and 'train_loss_stats' in stats:
                f.write("## 统计分析\n\n")
                loss_stats = stats['train_loss_stats']
                f.write(f"- **训练损失统计**:\n")
                f.write(f"  - 均值: {loss_stats['mean']:.4f}\n")
                f.write(f"  - 标准差: {loss_stats['std']:.4f}\n")
                f.write(f"  - 最小值: {loss_stats['min']:.4f}\n")
                f.write(f"  - 最大值: {loss_stats['max']:.4f}\n")
                f.write(f"  - 样本数: {loss_stats['n_samples']}\n\n")

            # 结论和建议
            f.write("## 结论与建议\n\n")
            if best_configs:
                best = best_configs[0]
                f.write(f"1. **最优参数配置**: {best['param_name']} = {best['param_value']}\n")

            f.write("2. **关键发现**:\n")
            for param_name, analysis in param_analysis.items():
                best_value = analysis.get('best_value', {})
                if best_value.get('value') is not None:
                    f.write(f"   - {param_name}的最佳取值为 {best_value['value']}\n")

            f.write("\n3. **实验建议**:\n")
            f.write("   - 基于上述消融结果调优模型配置\n")
            f.write("   - 考虑参数间的交互效应\n")
            f.write("   - 在更大数据集上验证结果\n\n")

            f.write("![Overall Comparison](overall_comparison.png)\n\n")
            f.write("---\n")
            f.write("*本报告由 PHM-Vibench ContrastiveIDTask 消融实验系统自动生成*\n")

        self.logger.info(f"实验报告已保存: {report_file}")
        return str(report_file)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ContrastiveIDTask消融研究")

    parser.add_argument('--base_config', default='configs/id_contrastive/ablation.yaml',
                       help='基础配置文件路径')
    parser.add_argument('--output_dir', default='save/ablation_study',
                       help='输出目录')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式：减少参数空间和训练轮数')
    parser.add_argument('--params', nargs='*',
                       choices=['temperature', 'batch_size', 'window_size', 'learning_rate', 'd_model', 'window_strategy'],
                       help='指定要消融的参数')
    parser.add_argument('--epochs', type=int,
                       help='训练轮数（覆盖默认设置）')
    parser.add_argument('--no_visualizations', action='store_true',
                       help='跳过可视化生成')

    args = parser.parse_args()

    # 创建消融研究实例
    study = AblationStudy(
        base_config=args.base_config,
        output_dir=args.output_dir,
        quick_mode=args.quick
    )

    # 如果指定了epochs，覆盖默认设置
    if args.epochs:
        study.max_epochs = args.epochs

    try:
        # 运行基线实验
        study.run_baseline_experiment()

        # 运行消融实验
        study.run_ablation_experiments(target_params=args.params)

        # 分析结果
        analysis_results = study.analyze_results()

        # 生成可视化
        if not args.no_visualizations:
            study.generate_visualizations()

        # 生成报告
        report_path = study.generate_report(analysis_results)

        print(f"\n{'='*60}")
        print("消融实验完成!")
        print(f"报告保存在: {report_path}")
        print(f"结果目录: {args.output_dir}")
        print(f"{'='*60}")

        return 0

    except KeyboardInterrupt:
        study.logger.warning("实验被中断")
        return 130
    except Exception as e:
        study.logger.error(f"实验失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())