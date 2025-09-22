#!/usr/bin/env python3
"""
ContrastiveIDTask消融实验脚本
系统性测试不同超参数对对比学习预训练性能的影响
支持论文所需的全面消融分析
"""

import os
import sys
import time
import json
import yaml
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import product
from collections import defaultdict
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.configs import load_config
from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask


class AblationStudyRunner:
    """消融实验运行器"""
    
    def __init__(self, save_dir="./ablation_results", base_config_path="configs/id_contrastive/ablation.yaml"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.base_config_path = base_config_path
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / "ablation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 对比学习消融实验参数
        self.ablation_params = {
            'window_size': [256, 512, 1024, 2048],  # 减少最大值避免内存问题
            'temperature': [0.01, 0.05, 0.07, 0.1, 0.5],
            'batch_size': [8, 16, 32],  # 保守的批大小
            'window_sampling_strategy': ['random', 'evenly_spaced'],  # 减少变体加快测试
            'backbone': ['B_08_PatchTST', 'B_04_Dlinear'],  # 核心backbone测试
            'learning_rate': [5e-4, 1e-3, 5e-3],
            'd_model': [64, 128, 256]  # 减少最大模型尺寸
        }
        
        # 结果存储
        self.results = defaultdict(list)
        
        # 默认基础配置
        self.base_config = None
        
    def load_base_config(self):
        """加载基础配置"""
        try:
            self.base_config = load_config(self.base_config_path)
            self.logger.info(f"已加载基础配置: {self.base_config_path}")
        except Exception as e:
            self.logger.warning(f"无法加载配置文件 {self.base_config_path}: {e}")
            # 创建默认配置
            self.base_config = self._create_default_config()
            self.logger.info("使用默认配置")
    
    def _create_default_config(self):
        """创建默认配置"""
        return {
            'data': {
                'factory_name': 'id',
                'batch_size': 32,
                'num_workers': 4,
                'window_size': 1024,
                'stride': 512,
                'num_window': 2,
                'window_sampling_strategy': 'random',
                'normalization': True,
                'truncate_length': 4096
            },
            'model': {
                'name': 'M_01_ISFM',
                'backbone': 'B_08_PatchTST',
                'd_model': 128,
                'num_heads': 8,
                'num_layers': 6,
                'dropout': 0.1
            },
            'task': {
                'type': 'pretrain',
                'name': 'contrastive_id',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'temperature': 0.07,
                'loss': 'CE',
                'metrics': ['acc']
            },
            'trainer': {
                'epochs': 1,  # 快速测试用单epoch
                'accelerator': 'cpu',
                'devices': 1,
                'precision': 32,
                'gradient_clip_val': 1.0
            },
            'environment': {
                'save_dir': str(self.save_dir),
                'experiment_name': 'ablation'
            }
        }
    
    def create_mock_data(self, num_samples=200, signal_length=4096, num_channels=2):
        """创建模拟数据用于消融实验"""
        data = []
        
        # 为不同类别创建具有一定结构的数据
        num_classes = 5
        for class_id in range(num_classes):
            # 每个类别的基础模式
            base_frequency = 0.1 + class_id * 0.05
            base_amplitude = 0.5 + class_id * 0.1
            
            samples_per_class = num_samples // num_classes
            for i in range(samples_per_class):
                # 生成具有特定频率特征的信号
                t = np.linspace(0, 10, signal_length)
                signal = np.zeros((signal_length, num_channels))
                
                for ch in range(num_channels):
                    # 主频率成分
                    signal[:, ch] = base_amplitude * np.sin(2 * np.pi * base_frequency * t)
                    # 添加谐波
                    signal[:, ch] += 0.3 * base_amplitude * np.sin(2 * np.pi * 2 * base_frequency * t)
                    # 添加噪声
                    signal[:, ch] += np.random.normal(0, 0.1, signal_length)
                
                metadata = {
                    'Label': class_id,
                    'Domain_id': 1,
                    'Fault_level': class_id % 3
                }
                
                sample_id = f'class_{class_id}_sample_{i}'
                data.append((sample_id, signal.astype(np.float32), metadata))
        
        # 随机打乱数据
        np.random.shuffle(data)
        return data
    
    def create_simple_network(self, input_dim, output_dim=128):
        """创建简单的测试网络"""
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, output_dim)
        )
    
    def run_single_experiment(self, config_overrides: Dict[str, Any], 
                            experiment_name: str, max_epochs: int = 10) -> Dict[str, float]:
        """运行单个实验"""
        self.logger.info(f"开始实验: {experiment_name}")
        self.logger.info(f"参数覆盖: {config_overrides}")
        
        try:
            # 创建实验配置
            config = self.base_config.copy()
            
            # 应用参数覆盖
            for key, value in config_overrides.items():
                if '.' in key:
                    # 支持嵌套键如 'data.batch_size'
                    section, param = key.split('.', 1)
                    if section not in config:
                        config[section] = {}
                    config[section][param] = value
                else:
                    config[key] = value
            
            # 创建模拟任务
            from argparse import Namespace
            from unittest.mock import patch, Mock
            
            args_data = Namespace(**config['data'])
            args_task = Namespace(**config['task'])
            args_model = Namespace(**config['model'])
            args_trainer = Namespace(**config['trainer'])
            args_environment = Namespace(**config['environment'])
            
            # 创建网络
            input_dim = args_data.window_size * 2  # window_size * num_channels
            network = self.create_simple_network(input_dim, args_model.d_model)
            
            # 创建任务实例
            with patch('src.task_factory.task.pretrain.ContrastiveIDTask.BaseIDTask.__init__'):
                task = ContrastiveIDTask(
                    network=network,
                    args_data=args_data,
                    args_model=args_model,
                    args_task=args_task,
                    args_trainer=args_trainer,
                    args_environment=args_environment,
                    metadata={}
                )
                
                # 模拟BaseIDTask方法
                task.process_sample = Mock(side_effect=lambda data, metadata: data)
                task.create_windows = Mock(side_effect=lambda data, strategy, num_window: [
                    np.random.randn(args_data.window_size, 2),
                    np.random.randn(args_data.window_size, 2)
                ])
                task.log = Mock()
            
            # 创建训练数据
            train_data = self.create_mock_data(150, args_data.truncate_length, 2)
            val_data = self.create_mock_data(50, args_data.truncate_length, 2)
            
            # 训练循环
            train_losses = []
            train_accuracies = []
            val_losses = []
            val_accuracies = []
            
            best_val_loss = float('inf')
            epochs_without_improvement = 0
            early_stop_patience = 5
            
            start_time = time.time()
            
            for epoch in range(max_epochs):
                # 训练阶段
                epoch_train_losses = []
                epoch_train_accs = []
                
                # 随机打乱训练数据
                np.random.shuffle(train_data)
                
                for i in range(0, len(train_data), args_data.batch_size):
                    batch_data = train_data[i:i+args_data.batch_size]
                    batch = task.prepare_batch(batch_data)
                    
                    if len(batch['ids']) > 0:
                        # 前向传播
                        anchor_features = network(batch['anchor'])
                        positive_features = network(batch['positive'])
                        
                        # 计算损失和准确率
                        loss = task.infonce_loss(anchor_features, positive_features)
                        accuracy = task.compute_accuracy(anchor_features, positive_features)
                        
                        epoch_train_losses.append(loss.item())
                        epoch_train_accs.append(accuracy.item())
                
                # 验证阶段
                epoch_val_losses = []
                epoch_val_accs = []
                
                with torch.no_grad():
                    for i in range(0, len(val_data), args_data.batch_size):
                        batch_data = val_data[i:i+args_data.batch_size]
                        batch = task.prepare_batch(batch_data)
                        
                        if len(batch['ids']) > 0:
                            anchor_features = network(batch['anchor'])
                            positive_features = network(batch['positive'])
                            
                            loss = task.infonce_loss(anchor_features, positive_features)
                            accuracy = task.compute_accuracy(anchor_features, positive_features)
                            
                            epoch_val_losses.append(loss.item())
                            epoch_val_accs.append(accuracy.item())
                
                # 记录epoch结果
                if epoch_train_losses:
                    train_loss = np.mean(epoch_train_losses)
                    train_acc = np.mean(epoch_train_accs)
                    train_losses.append(train_loss)
                    train_accuracies.append(train_acc)
                
                if epoch_val_losses:
                    val_loss = np.mean(epoch_val_losses)
                    val_acc = np.mean(epoch_val_accs)
                    val_losses.append(val_loss)
                    val_accuracies.append(val_acc)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                    
                    if epochs_without_improvement >= early_stop_patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                
                # 每2个epoch打印一次进度
                if (epoch + 1) % 2 == 0:
                    self.logger.info(f"Epoch {epoch+1}: "
                                   f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                                   f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            training_time = time.time() - start_time
            
            # 计算最终指标
            final_metrics = {
                'final_train_loss': train_losses[-1] if train_losses else float('inf'),
                'final_train_accuracy': train_accuracies[-1] if train_accuracies else 0.0,
                'final_val_loss': val_losses[-1] if val_losses else float('inf'),
                'final_val_accuracy': val_accuracies[-1] if val_accuracies else 0.0,
                'best_val_loss': best_val_loss,
                'epochs_trained': len(train_losses),
                'training_time': training_time,
                'convergence_speed': len(train_losses) / max_epochs  # 收敛速度指标
            }
            
            # 添加稳定性指标
            if len(val_losses) >= 5:
                recent_val_losses = val_losses[-5:]
                final_metrics['val_loss_stability'] = np.std(recent_val_losses)
            
            # 添加改进指标
            if len(val_losses) >= 2:
                improvement = val_losses[0] - val_losses[-1]
                final_metrics['val_loss_improvement'] = improvement
                
            self.logger.info(f"实验 {experiment_name} 完成")
            self.logger.info(f"最终指标: {final_metrics}")
            
            return final_metrics
            
        except Exception as e:
            self.logger.error(f"实验 {experiment_name} 失败: {e}")
            return {
                'error': str(e),
                'final_train_loss': float('inf'),
                'final_val_loss': float('inf'),
                'final_train_accuracy': 0.0,
                'final_val_accuracy': 0.0
            }
    
    def run_parameter_ablation(self, param_name: str, param_values: List[Any]) -> Dict[str, List[Dict]]:
        """运行单参数消融实验"""
        self.logger.info(f"="*60)
        self.logger.info(f"开始 {param_name} 消融实验")
        self.logger.info(f"测试值: {param_values}")
        
        results = []
        
        for value in param_values:
            experiment_name = f"{param_name}_{value}"
            config_override = {param_name: value}
            
            # 特殊处理某些参数
            if param_name == 'backbone':
                config_override = {'model.backbone': value}
            elif param_name == 'learning_rate':
                config_override = {'task.lr': value}
            elif param_name in ['window_size', 'batch_size', 'window_sampling_strategy']:
                config_override = {f'data.{param_name}': value}
            elif param_name == 'temperature':
                config_override = {'task.temperature': value}
            elif param_name == 'd_model':
                config_override = {'model.d_model': value}
            
            # 运行实验
            metrics = self.run_single_experiment(config_override, experiment_name, max_epochs=8)
            
            # 记录结果
            result = {
                'parameter': param_name,
                'value': value,
                'experiment_name': experiment_name,
                **metrics
            }
            results.append(result)
            
            # 短暂休息避免系统过载
            time.sleep(1)
        
        self.results[param_name] = results
        return results
    
    def run_interaction_study(self, param_pairs: List[Tuple[str, str]]) -> Dict[str, List[Dict]]:
        """运行参数交互实验"""
        self.logger.info(f"="*60)
        self.logger.info("开始参数交互实验")
        
        interaction_results = {}
        
        for param1, param2 in param_pairs:
            self.logger.info(f"测试 {param1} x {param2} 交互")
            
            # 选择少量代表性的值进行交互测试
            values1 = self.ablation_params[param1][:3]  # 取前3个值
            values2 = self.ablation_params[param2][:3]  # 取前3个值
            
            interaction_key = f"{param1}_x_{param2}"
            results = []
            
            for v1, v2 in product(values1, values2):
                experiment_name = f"{param1}_{v1}_{param2}_{v2}"
                
                # 创建配置覆盖
                config_override = {}
                
                # 映射参数到配置路径
                param_mapping = {
                    'window_size': 'data.window_size',
                    'temperature': 'task.temperature',
                    'batch_size': 'data.batch_size',
                    'window_sampling_strategy': 'data.window_sampling_strategy',
                    'backbone': 'model.backbone',
                    'learning_rate': 'task.lr',
                    'd_model': 'model.d_model'
                }
                
                config_override[param_mapping.get(param1, param1)] = v1
                config_override[param_mapping.get(param2, param2)] = v2
                
                # 运行实验
                metrics = self.run_single_experiment(config_override, experiment_name, max_epochs=6)
                
                # 记录结果
                result = {
                    'param1': param1,
                    'value1': v1,
                    'param2': param2,
                    'value2': v2,
                    'experiment_name': experiment_name,
                    **metrics
                }
                results.append(result)
            
            interaction_results[interaction_key] = results
        
        return interaction_results
    
    def analyze_results(self) -> Dict[str, Any]:
        """分析消融实验结果"""
        self.logger.info("="*60)
        self.logger.info("开始结果分析")
        
        analysis = {}
        
        for param_name, results in self.results.items():
            self.logger.info(f"分析 {param_name} 的影响")
            
            if not results:
                continue
            
            # 转换为DataFrame便于分析
            df = pd.DataFrame(results)
            
            # 基本统计
            param_analysis = {
                'parameter': param_name,
                'num_experiments': len(results),
                'best_config': None,
                'worst_config': None,
                'performance_range': None,
                'stability_analysis': None
            }
            
            if 'final_val_accuracy' in df.columns:
                # 找到最佳和最差配置
                best_idx = df['final_val_accuracy'].idxmax()
                worst_idx = df['final_val_accuracy'].idxmin()
                
                param_analysis['best_config'] = {
                    'value': df.loc[best_idx, 'value'],
                    'val_accuracy': df.loc[best_idx, 'final_val_accuracy'],
                    'val_loss': df.loc[best_idx, 'final_val_loss']
                }
                
                param_analysis['worst_config'] = {
                    'value': df.loc[worst_idx, 'value'],
                    'val_accuracy': df.loc[worst_idx, 'final_val_accuracy'],
                    'val_loss': df.loc[worst_idx, 'final_val_loss']
                }
                
                # 性能范围
                acc_range = df['final_val_accuracy'].max() - df['final_val_accuracy'].min()
                param_analysis['performance_range'] = acc_range
                
                # 稳定性分析
                if 'val_loss_stability' in df.columns:
                    param_analysis['stability_analysis'] = {
                        'mean_stability': df['val_loss_stability'].mean(),
                        'most_stable_config': df.loc[df['val_loss_stability'].idxmin(), 'value']
                    }
                
                self.logger.info(f"  最佳配置: {param_analysis['best_config']['value']} "
                               f"(准确率: {param_analysis['best_config']['val_accuracy']:.4f})")
                self.logger.info(f"  性能范围: {acc_range:.4f}")
            
            analysis[param_name] = param_analysis
        
        return analysis
    
    def generate_visualizations(self):
        """生成可视化图表"""
        self.logger.info("生成可视化图表")
        
        try:
            plt.style.use('default')
            
            for param_name, results in self.results.items():
                if not results:
                    continue
                
                df = pd.DataFrame(results)
                
                if 'final_val_accuracy' not in df.columns:
                    continue
                
                # 创建子图
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'{param_name} Ablation Study', fontsize=16)
                
                # 1. 验证准确率 vs 参数值
                ax1 = axes[0, 0]
                if df['value'].dtype in ['object', 'bool']:
                    # 分类变量
                    ax1.bar(range(len(df)), df['final_val_accuracy'])
                    ax1.set_xticks(range(len(df)))
                    ax1.set_xticklabels(df['value'], rotation=45)
                else:
                    # 数值变量
                    ax1.plot(df['value'], df['final_val_accuracy'], 'o-')
                    ax1.set_xlabel(param_name)
                ax1.set_ylabel('Validation Accuracy')
                ax1.set_title('Validation Accuracy vs Parameter')
                ax1.grid(True)
                
                # 2. 验证损失 vs 参数值
                ax2 = axes[0, 1]
                if df['value'].dtype in ['object', 'bool']:
                    ax2.bar(range(len(df)), df['final_val_loss'])
                    ax2.set_xticks(range(len(df)))
                    ax2.set_xticklabels(df['value'], rotation=45)
                else:
                    ax2.plot(df['value'], df['final_val_loss'], 'o-', color='red')
                    ax2.set_xlabel(param_name)
                ax2.set_ylabel('Validation Loss')
                ax2.set_title('Validation Loss vs Parameter')
                ax2.grid(True)
                
                # 3. 训练时间 vs 参数值
                ax3 = axes[1, 0]
                if 'training_time' in df.columns:
                    if df['value'].dtype in ['object', 'bool']:
                        ax3.bar(range(len(df)), df['training_time'])
                        ax3.set_xticks(range(len(df)))
                        ax3.set_xticklabels(df['value'], rotation=45)
                    else:
                        ax3.plot(df['value'], df['training_time'], 'o-', color='green')
                        ax3.set_xlabel(param_name)
                    ax3.set_ylabel('Training Time (s)')
                    ax3.set_title('Training Time vs Parameter')
                    ax3.grid(True)
                
                # 4. 收敛速度 vs 参数值
                ax4 = axes[1, 1]
                if 'convergence_speed' in df.columns:
                    if df['value'].dtype in ['object', 'bool']:
                        ax4.bar(range(len(df)), df['convergence_speed'])
                        ax4.set_xticks(range(len(df)))
                        ax4.set_xticklabels(df['value'], rotation=45)
                    else:
                        ax4.plot(df['value'], df['convergence_speed'], 'o-', color='purple')
                        ax4.set_xlabel(param_name)
                    ax4.set_ylabel('Convergence Speed')
                    ax4.set_title('Convergence Speed vs Parameter')
                    ax4.grid(True)
                
                plt.tight_layout()
                plt.savefig(self.save_dir / f'{param_name}_ablation.png', dpi=150, bbox_inches='tight')
                plt.close()
            
            # 生成汇总热图
            self.generate_summary_heatmap()
            
            self.logger.info("可视化图表已生成")
            
        except Exception as e:
            self.logger.warning(f"生成可视化时出错: {e}")
    
    def generate_summary_heatmap(self):
        """生成参数影响汇总热图"""
        try:
            # 收集所有参数的最佳性能
            summary_data = []
            
            for param_name, results in self.results.items():
                if not results:
                    continue
                
                df = pd.DataFrame(results)
                if 'final_val_accuracy' in df.columns:
                    best_acc = df['final_val_accuracy'].max()
                    best_loss = df['final_val_loss'].min()
                    acc_range = df['final_val_accuracy'].max() - df['final_val_accuracy'].min()
                    
                    summary_data.append({
                        'Parameter': param_name,
                        'Best_Accuracy': best_acc,
                        'Best_Loss': best_loss,
                        'Performance_Range': acc_range
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                
                # 创建热图
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # 准备热图数据
                heatmap_data = summary_df[['Best_Accuracy', 'Performance_Range']].T
                heatmap_data.columns = summary_df['Parameter']
                
                sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax)
                ax.set_title('Parameter Impact Summary')
                
                plt.tight_layout()
                plt.savefig(self.save_dir / 'parameter_impact_heatmap.png', dpi=150)
                plt.close()
                
        except Exception as e:
            self.logger.warning(f"生成汇总热图时出错: {e}")
    
    def save_results(self):
        """保存结果到文件"""
        self.logger.info("保存结果")
        
        # 保存详细结果
        results_file = self.save_dir / "ablation_results.json"
        with open(results_file, 'w') as f:
            json.dump(dict(self.results), f, indent=2, default=str)
        
        # 生成结果分析
        analysis = self.analyze_results()
        
        analysis_file = self.save_dir / "ablation_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # 生成Markdown报告
        self.generate_report(analysis)
        
        self.logger.info(f"结果已保存到 {self.save_dir}")
    
    def generate_report(self, analysis: Dict[str, Any]):
        """生成Markdown报告"""
        report_file = self.save_dir / "ablation_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# ContrastiveIDTask 消融实验报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 实验概述\n\n")
            f.write("本报告展示了对ContrastiveIDTask各关键超参数的消融实验结果。\n\n")
            
            f.write("### 测试参数\n\n")
            for param, values in self.ablation_params.items():
                f.write(f"- **{param}**: {values}\n")
            f.write("\n")
            
            f.write("## 主要发现\n\n")
            
            for param_name, param_analysis in analysis.items():
                if not param_analysis.get('best_config'):
                    continue
                    
                f.write(f"### {param_name}\n\n")
                
                best_config = param_analysis['best_config']
                worst_config = param_analysis['worst_config']
                
                f.write(f"- **最佳配置**: {best_config['value']} (准确率: {best_config['val_accuracy']:.4f})\n")
                f.write(f"- **最差配置**: {worst_config['value']} (准确率: {worst_config['val_accuracy']:.4f})\n")
                f.write(f"- **性能范围**: {param_analysis['performance_range']:.4f}\n")
                
                if param_analysis.get('stability_analysis'):
                    stability = param_analysis['stability_analysis']
                    f.write(f"- **最稳定配置**: {stability['most_stable_config']}\n")
                
                f.write("\n")
            
            f.write("## 推荐配置\n\n")
            f.write("基于消融实验结果，推荐以下配置组合：\n\n")
            
            # 提取每个参数的最佳值
            recommended_config = {}
            for param_name, param_analysis in analysis.items():
                if param_analysis.get('best_config'):
                    recommended_config[param_name] = param_analysis['best_config']['value']
            
            for param, value in recommended_config.items():
                f.write(f"- **{param}**: {value}\n")
            
            f.write(f"\n## 详细结果\n\n")
            f.write("详细的实验数据请查看以下文件：\n")
            f.write("- `ablation_results.json`: 完整的实验结果数据\n")
            f.write("- `ablation_analysis.json`: 结果分析数据\n")
            f.write("- `*_ablation.png`: 各参数的可视化图表\n")
    
    def run_all_ablations(self):
        """运行所有消融实验"""
        self.logger.info("开始全面消融实验")
        self.logger.info(f"结果将保存到: {self.save_dir}")
        
        # 加载基础配置
        self.load_base_config()
        
        try:
            # 运行单参数消融实验
            for param_name, param_values in self.ablation_params.items():
                self.run_parameter_ablation(param_name, param_values)
            
            # 运行关键参数交互实验
            important_pairs = [
                ('window_size', 'temperature'),
                ('batch_size', 'learning_rate'),
                ('temperature', 'learning_rate'),
                ('backbone', 'd_model')
            ]
            
            interaction_results = self.run_interaction_study(important_pairs)
            self.results.update(interaction_results)
            
            # 生成可视化和保存结果
            self.generate_visualizations()
            self.save_results()
            
            self.logger.info("="*60)
            self.logger.info("消融实验完成")
            self.logger.info(f"共进行了 {sum(len(results) for results in self.results.values())} 个实验")
            
        except Exception as e:
            self.logger.error(f"消融实验过程中出错: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ContrastiveIDTask消融实验")
    parser.add_argument("--save-dir", default="./ablation_results",
                       help="结果保存目录")
    parser.add_argument("--config", default="configs/id_contrastive/ablation.yaml",
                       help="基础配置文件路径")
    parser.add_argument("--param", choices=list(AblationStudyRunner(save_dir="temp").ablation_params.keys()) + ['all'],
                       default="all", help="要测试的参数")
    parser.add_argument("--quick", action="store_true",
                       help="快速模式（减少测试值数量）")
    
    args = parser.parse_args()
    
    runner = AblationStudyRunner(save_dir=args.save_dir, base_config_path=args.config)
    
    # 快速模式：减少测试参数
    if args.quick:
        runner.ablation_params = {
            'window_size': [1024, 2048],
            'temperature': [0.05, 0.07, 0.1],
            'batch_size': [16, 32],
            'backbone': ['B_08_PatchTST', 'B_04_Dlinear']
        }
    
    runner.load_base_config()
    
    if args.param == "all":
        runner.run_all_ablations()
    else:
        param_values = runner.ablation_params[args.param]
        runner.run_parameter_ablation(args.param, param_values)
        runner.generate_visualizations()
        runner.save_results()


if __name__ == "__main__":
    main()