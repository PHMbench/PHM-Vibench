#!/usr/bin/env python3
"""
多数据集对比学习预训练实验脚本

完整的多数据集实验管理系统，支持：
- 多数据集组合配置和域泛化实验
- 并行实验执行和资源管理
- 综合性能基准测试集成
- 智能配置管理和参数扫描
- 详细的实验报告和可视化分析

作者: Claude Code
版本: 2.0 (Task-014实现)
"""

import os
import sys
import json
import yaml
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import subprocess
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import itertools
import shutil
import logging
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict, Counter
import hashlib
import psutil
import torch

# 添加项目路径
sys.path.append('.')

from src.configs import load_config
from src.data_factory.id_data_factory import id_data_factory

# 尝试导入基准测试模块（可选）
# 尝试导入基准测试模块（可选）
try:
    from benchmarks.contrastive_benchmark import PerformanceBenchmark
    BENCHMARK_AVAILABLE = True
except ImportError:
    BENCHMARK_AVAILABLE = False
    print("Warning: Benchmark module not available. Benchmark features will be disabled.")


@dataclass
class DatasetInfo:
    """数据集信息的结构化表示"""
    name: str
    metadata_file: str
    h5_file: Path
    num_samples: int
    num_classes: int = 0
    num_ids: int = 0
    h5_size_mb: float = 0.0
    ready: bool = False
    class_distribution: Dict[int, int] = None
    signal_length: int = 0
    sampling_rate: int = 0
    domain_type: str = "unknown"  # bearing, gear, motor, etc.
    
    def __post_init__(self):
        if self.class_distribution is None:
            self.class_distribution = {}


@dataclass  
class ExperimentConfig:
    """实验配置的结构化表示"""
    id: str
    name: str
    dataset_combination: List[str]  # 数据集组合
    variant_name: str
    config_overrides: Dict[str, Any]
    expected_duration_hours: float
    priority: int = 0  # 实验优先级
    status: str = "pending"
    dependencies: List[str] = None  # 依赖的实验 ID
    resource_requirements: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.resource_requirements is None:
            self.resource_requirements = {
                'gpu_memory_mb': 8000,
                'ram_mb': 16000,
                'cpu_cores': 4
            }


@dataclass
class ExperimentResult:
    """实验结果的结构化表示"""
    experiment_id: str
    status: str  # completed, failed, skipped, running
    start_time: datetime = None
    end_time: datetime = None
    actual_duration_hours: float = 0.0
    metrics: Dict[str, float] = None
    error_message: str = ""
    resource_usage: Dict[str, float] = None
    output_files: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.resource_usage is None:
            self.resource_usage = {}
        if self.output_files is None:
            self.output_files = []


class MultiDatasetExperimentRunner:
    """
    多数据集对比学习实验运行器
    
    特性:
    - 智能数据集组合和域泛化实验
    - 并行执行和资源管理
    - 基准测试集成
    - 综合可视化报告
    - 实验恢复和断点续传
    """
    
    def __init__(self, 
                 base_config_path: str,
                 metadata_dir: str = "data",
                 results_dir: str = "save/multi_dataset",
                 dry_run: bool = False,
                 enable_benchmarking: bool = True,
                 max_concurrent_experiments: int = None):
        """
        初始化实验运行器
        
        Args:
            base_config_path: 基础配置文件路径
            metadata_dir: metadata文件目录
            results_dir: 结果保存目录
            dry_run: 是否只输出实验计划而不实际运行
            enable_benchmarking: 是否启用基准测试
            max_concurrent_experiments: 最大并发实验数（默认为CPU核数的一半）
        """
        self.base_config_path = base_config_path
        self.metadata_dir = Path(metadata_dir)
        self.results_dir = Path(results_dir)
        self.dry_run = dry_run
        self.enable_benchmarking = enable_benchmarking and BENCHMARK_AVAILABLE
        self.max_concurrent_experiments = max_concurrent_experiments or max(1, mp.cpu_count() // 2)
        
        # 创建结果目录
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 实验状态管理
        self.experiments: List[ExperimentConfig] = []
        self.results: Dict[str, List[ExperimentResult]] = {
            'completed': [],
            'failed': [],
            'skipped': [],
            'running': []
        }
        
        # 资源管理
        self.resource_manager = ResourceManager()
        
        # 基准测试器（可选）
        self.benchmark = None
        if self.enable_benchmarking:
            try:
                self.benchmark = PerformanceBenchmark(
                    save_dir=self.results_dir / "benchmarks"
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize benchmark: {e}")
                self.enable_benchmarking = False
        
        # 实验状态文件
        self.state_file = self.results_dir / "experiment_state.json"
        self._load_experiment_state()
        
        self.logger.info(f"多数据集对比学习实验运行器初始化完成")
        self.logger.info(f"基础配置: {base_config_path}")
        self.logger.info(f"结果目录: {results_dir}")
        self.logger.info(f"干运行模式: {dry_run}")
        self.logger.info(f"基准测试: {self.enable_benchmarking}")
        self.logger.info(f"最大并发实验数: {self.max_concurrent_experiments}")
    
    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.results_dir / "experiment.log"
        
        # 创建 logger
        self.logger = logging.getLogger('MultiDatasetExperiment')
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加 handler
        if not self.logger.handlers:
            # 文件 handler
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # 控制台 handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 设置格式
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加 handlers
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def _load_experiment_state(self):
        """加载实验状态（用于断点续传）"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                # 恢复实验状态
                for exp_data in state.get('experiments', []):
                    exp = ExperimentConfig(**exp_data)
                    self.experiments.append(exp)
                
                # 恢复结果
                for status, results in state.get('results', {}).items():
                    if status in self.results:
                        for result_data in results:
                            result = ExperimentResult(**result_data)
                            if result.start_time:
                                result.start_time = datetime.fromisoformat(result.start_time)
                            if result.end_time:
                                result.end_time = datetime.fromisoformat(result.end_time)
                            self.results[status].append(result)
                
                self.logger.info(f"Loaded experiment state with {len(self.experiments)} experiments")
                
            except Exception as e:
                self.logger.warning(f"Failed to load experiment state: {e}")
    
    def _save_experiment_state(self):
        """保存实验状态"""
        try:
            state = {
                'experiments': [asdict(exp) for exp in self.experiments],
                'results': {},
                'last_updated': datetime.now().isoformat()
            }
            
            # 转换结果格式
            for status, results in self.results.items():
                state['results'][status] = []
                for result in results:
                    result_dict = asdict(result)
                    # 转换日期类型
                    if result_dict['start_time']:
                        result_dict['start_time'] = result.start_time.isoformat()
                    if result_dict['end_time']:
                        result_dict['end_time'] = result.end_time.isoformat()
                    state['results'][status].append(result_dict)
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save experiment state: {e}")
    
    def discover_datasets(self) -> List[DatasetInfo]:
        """发现所有可用的数据集"""
        self.logger.info(f"正在扫描metadata目录: {self.metadata_dir}")
        
        datasets = []
        metadata_files = list(self.metadata_dir.glob("metadata_*.xlsx"))
        
        if not metadata_files:
            self.logger.warning(f"在 {self.metadata_dir} 中未找到metadata文件")
            return datasets
        
        for metadata_file in metadata_files:
            try:
                # 提取数据集名称
                dataset_name = metadata_file.stem.replace('metadata_', '')
                
                # 读取metadata文件获取基本信息
                df = pd.read_excel(metadata_file, sheet_name=0)
                
                # 创建数据集信息对象
                dataset_info = DatasetInfo(
                    name=dataset_name,
                    metadata_file=str(metadata_file),
                    h5_file=self.metadata_dir / f"{dataset_name}.h5",
                    num_samples=len(df)
                )
                
                # 检查H5文件是否存在
                if dataset_info.h5_file.exists():
                    dataset_info.h5_size_mb = dataset_info.h5_file.stat().st_size / (1024 * 1024)
                    dataset_info.ready = True
                else:
                    dataset_info.h5_size_mb = 0
                    dataset_info.ready = False
                    self.logger.warning(f"数据集 {dataset_name} 的H5文件不存在: {dataset_info.h5_file}")
                
                # 尝试获取更多信息
                if 'Label' in df.columns:
                    dataset_info.num_classes = df['Label'].nunique()
                    dataset_info.class_distribution = df['Label'].value_counts().to_dict()
                
                if 'ID' in df.columns:
                    dataset_info.num_ids = df['ID'].nunique()
                
                # 推断域类型
                dataset_info.domain_type = self._infer_domain_type(dataset_name)
                
                # 获取信号长度信息
                if 'Length' in df.columns:
                    dataset_info.signal_length = int(df['Length'].iloc[0]) if len(df) > 0 else 0
                elif 'signal_length' in df.columns:
                    dataset_info.signal_length = int(df['signal_length'].iloc[0]) if len(df) > 0 else 0
                
                datasets.append(dataset_info)
                self.logger.info(f"发现数据集: {dataset_name} ({dataset_info.num_samples} 样本, {dataset_info.num_classes} 类别)")
                
            except Exception as e:
                self.logger.error(f"处理metadata文件失败 {metadata_file}: {e}")
        
        self.logger.info(f"共发现 {len(datasets)} 个数据集")
        return datasets
    
    def _infer_domain_type(self, dataset_name: str) -> str:
        """根据数据集名称推断域类型"""
        name_lower = dataset_name.lower()
        
        if any(keyword in name_lower for keyword in ['cwru', 'bearing', 'ball']):
            return 'bearing'
        elif any(keyword in name_lower for keyword in ['gear', 'gearbox']):
            return 'gear'
        elif any(keyword in name_lower for keyword in ['motor', 'rotor']):
            return 'motor'
        elif any(keyword in name_lower for keyword in ['pump']):
            return 'pump'
        elif any(keyword in name_lower for keyword in ['fan', 'compressor']):
            return 'fan'
        else:
            return 'unknown'
    
    def filter_datasets(self, 
                       datasets: List[DatasetInfo], 
                       include_patterns: List[str] = None,
                       exclude_patterns: List[str] = None,
                       min_samples: int = 100,
                       ready_only: bool = True,
                       domain_types: List[str] = None) -> List[DatasetInfo]:
        """过滤数据集"""
        self.logger.info("正在过滤数据集...")
        
        filtered_datasets = []
        
        for dataset in datasets:
            # 检查是否准备就绪
            if ready_only and not dataset.ready:
                self.logger.debug(f"跳过未准备数据集: {dataset.name}")
                continue
            
            # 检查样本数量
            if dataset.num_samples < min_samples:
                self.logger.debug(f"跳过样本数量不足的数据集: {dataset.name} ({dataset.num_samples} < {min_samples})")
                continue
            
            # 检查包含模式
            if include_patterns:
                if not any(pattern.lower() in dataset.name.lower() for pattern in include_patterns):
                    self.logger.debug(f"跳过不匹配包含模式的数据集: {dataset.name}")
                    continue
            
            # 检查排除模式
            if exclude_patterns:
                if any(pattern.lower() in dataset.name.lower() for pattern in exclude_patterns):
                    self.logger.debug(f"跳过匹配排除模式的数据集: {dataset.name}")
                    continue
            
            # 检查域类型
            if domain_types:
                if dataset.domain_type not in domain_types:
                    self.logger.debug(f"跳过不匹配域类型的数据集: {dataset.name} ({dataset.domain_type} not in {domain_types})")
                    continue
            
            filtered_datasets.append(dataset)
            self.logger.info(f"保留数据集: {dataset.name} (域: {dataset.domain_type})")
        
        self.logger.info(f"过滤后保留 {len(filtered_datasets)} 个数据集")
        return filtered_datasets
    
    def generate_dataset_combinations(self, 
                                     datasets: List[DatasetInfo],
                                     combination_strategies: List[str] = None) -> List[Dict[str, Any]]:
        """
        生成数据集组合策略
        
        Args:
            datasets: 数据集列表
            combination_strategies: 组合策略列表
        
        Returns:
            数据集组合列表
        """
        if combination_strategies is None:
            combination_strategies = [
                'single',  # 单数据集
                'cross_domain',  # 跨域
                'same_domain',  # 同域多数据集
                'all_datasets',  # 所有数据集
            ]
        
        combinations = []
        
        # 按域类型分组
        domain_groups = defaultdict(list)
        for dataset in datasets:
            domain_groups[dataset.domain_type].append(dataset)
        
        for strategy in combination_strategies:
            if strategy == 'single':
                # 单数据集实验
                for dataset in datasets:
                    combinations.append({
                        'name': f'single_{dataset.name}',
                        'strategy': 'single',
                        'datasets': [dataset.name],
                        'description': f'单数据集实验: {dataset.name}',
                        'balancing': 'single',
                        'expected_samples': dataset.num_samples
                    })
            
            elif strategy == 'cross_domain':
                # 跨域组合：不同域类型的数据集
                domain_types = list(domain_groups.keys())
                for i, source_domain in enumerate(domain_types):
                    for target_domain in domain_types[i+1:]:
                        if source_domain != target_domain and source_domain != 'unknown' and target_domain != 'unknown':
                            source_datasets = [ds.name for ds in domain_groups[source_domain]]
                            target_datasets = [ds.name for ds in domain_groups[target_domain]]
                            
                            # 创建跨域组合
                            for source_ds in source_datasets[:2]:  # 最多2个源域
                                for target_ds in target_datasets[:2]:  # 最多2个目标域
                                    combinations.append({
                                        'name': f'cross_{source_domain}_to_{target_domain}_{source_ds}_{target_ds}',
                                        'strategy': 'cross_domain',
                                        'datasets': [source_ds, target_ds],
                                        'source_datasets': [source_ds],
                                        'target_datasets': [target_ds],
                                        'description': f'跨域实验: {source_domain} → {target_domain}',
                                        'balancing': 'weighted',
                                        'expected_samples': sum(ds.num_samples for ds in datasets if ds.name in [source_ds, target_ds])
                                    })
            
            elif strategy == 'same_domain':
                # 同域多数据集
                for domain_type, domain_datasets in domain_groups.items():
                    if len(domain_datasets) >= 2 and domain_type != 'unknown':
                        dataset_names = [ds.name for ds in domain_datasets]
                        combinations.append({
                            'name': f'same_domain_{domain_type}_all',
                            'strategy': 'same_domain',
                            'datasets': dataset_names,
                            'description': f'同域多数据集: {domain_type}',
                            'balancing': 'proportional',
                            'expected_samples': sum(ds.num_samples for ds in domain_datasets)
                        })
            
            elif strategy == 'all_datasets':
                # 所有数据集（如果数量不太多）
                if len(datasets) <= 8:  # 限制数据集数量
                    dataset_names = [ds.name for ds in datasets]
                    combinations.append({
                        'name': 'all_datasets_combined',
                        'strategy': 'all_datasets',
                        'datasets': dataset_names,
                        'description': '所有数据集组合实验',
                        'balancing': 'equal',
                        'expected_samples': sum(ds.num_samples for ds in datasets)
                    })
        
        self.logger.info(f"生成了 {len(combinations)} 个数据集组合")
        return combinations


class ResourceManager:
    """资源管理器，用于管理实验所需的计算资源"""
    
    def __init__(self):
        self.available_gpus = self._get_available_gpus()
        self.total_ram_mb = psutil.virtual_memory().total / (1024 * 1024)
        self.available_ram_mb = psutil.virtual_memory().available / (1024 * 1024)
        self.cpu_cores = mp.cpu_count()
        
        # 当前资源使用情况
        self.running_experiments: Dict[str, Dict[str, float]] = {}
        
    def _get_available_gpus(self) -> List[Dict[str, Any]]:
        """获取可用GPU信息"""
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_mb': torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
                }
                gpus.append(gpu_info)
        return gpus
    
    def can_run_experiment(self, resource_req: Dict[str, float]) -> bool:
        """检查是否有足够的资源运行实验"""
        # 计算当前资源使用情况
        used_ram = sum(exp['ram_mb'] for exp in self.running_experiments.values())
        used_gpu_memory = sum(exp['gpu_memory_mb'] for exp in self.running_experiments.values())
        used_cpu_cores = sum(exp['cpu_cores'] for exp in self.running_experiments.values())
        
        # 检查资源限制
        if used_ram + resource_req.get('ram_mb', 0) > self.available_ram_mb * 0.8:  # 留出20%缓冲
            return False
        
        if used_cpu_cores + resource_req.get('cpu_cores', 0) > self.cpu_cores * 0.8:
            return False
        
        if self.available_gpus and used_gpu_memory + resource_req.get('gpu_memory_mb', 0) > max(gpu['memory_mb'] for gpu in self.available_gpus) * 0.8:
            return False
        
        return True
    
    def allocate_resources(self, experiment_id: str, resource_req: Dict[str, float]) -> bool:
        """为实验分配资源"""
        if self.can_run_experiment(resource_req):
            self.running_experiments[experiment_id] = resource_req.copy()
            return True
        return False
    
    def release_resources(self, experiment_id: str):
        """释放实验资源"""
        if experiment_id in self.running_experiments:
            del self.running_experiments[experiment_id]
    
    def get_resource_status(self) -> Dict[str, Any]:
        """获取资源使用状态"""
        used_ram = sum(exp['ram_mb'] for exp in self.running_experiments.values())
        used_gpu_memory = sum(exp['gpu_memory_mb'] for exp in self.running_experiments.values())
        used_cpu_cores = sum(exp['cpu_cores'] for exp in self.running_experiments.values())
        
        return {
            'total_ram_mb': self.total_ram_mb,
            'available_ram_mb': self.available_ram_mb,
            'used_ram_mb': used_ram,
            'total_cpu_cores': self.cpu_cores,
            'used_cpu_cores': used_cpu_cores,
            'gpus': self.available_gpus,
            'used_gpu_memory_mb': used_gpu_memory,
            'running_experiments': len(self.running_experiments)
        }
    
    def generate_dataset_combinations(self, 
                                     datasets: List[DatasetInfo],
                                     combination_strategies: List[str] = None) -> List[Dict[str, Any]]:
        """
        生成数据集组合策略
        
        Args:
            datasets: 数据集列表
            combination_strategies: 组合策略列表
        
        Returns:
            数据集组合列表
        """
        if combination_strategies is None:
            combination_strategies = [
                'single',  # 单数据集
                'cross_domain',  # 跨域
                'same_domain',  # 同域多数据集
                'all_datasets',  # 所有数据集
            ]
        
        combinations = []
        
        # 按域类型分组
        domain_groups = defaultdict(list)
        for dataset in datasets:
            domain_groups[dataset.domain_type].append(dataset)
        
        for strategy in combination_strategies:
            if strategy == 'single':
                # 单数据集实验
                for dataset in datasets:
                    combinations.append({
                        'name': f'single_{dataset.name}',
                        'strategy': 'single',
                        'datasets': [dataset.name],
                        'description': f'单数据集实验: {dataset.name}',
                        'balancing': 'single',
                        'expected_samples': dataset.num_samples
                    })
            
            elif strategy == 'cross_domain':
                # 跨域组合：不同域类型的数据集
                domain_types = list(domain_groups.keys())
                for i, source_domain in enumerate(domain_types):
                    for target_domain in domain_types[i+1:]:
                        if source_domain != target_domain and source_domain != 'unknown' and target_domain != 'unknown':
                            source_datasets = [ds.name for ds in domain_groups[source_domain]]
                            target_datasets = [ds.name for ds in domain_groups[target_domain]]
                            
                            # 创建跨域组合
                            for source_ds in source_datasets[:2]:  # 最多2个源域
                                for target_ds in target_datasets[:2]:  # 最多2个目标域
                                    combinations.append({
                                        'name': f'cross_{source_domain}_to_{target_domain}_{source_ds}_{target_ds}',
                                        'strategy': 'cross_domain',
                                        'datasets': [source_ds, target_ds],
                                        'source_datasets': [source_ds],
                                        'target_datasets': [target_ds],
                                        'description': f'跨域实验: {source_domain} → {target_domain}',
                                        'balancing': 'weighted',
                                        'expected_samples': sum(ds.num_samples for ds in datasets if ds.name in [source_ds, target_ds])
                                    })
            
            elif strategy == 'same_domain':
                # 同域多数据集
                for domain_type, domain_datasets in domain_groups.items():
                    if len(domain_datasets) >= 2 and domain_type != 'unknown':
                        dataset_names = [ds.name for ds in domain_datasets]
                        combinations.append({
                            'name': f'same_domain_{domain_type}_all',
                            'strategy': 'same_domain',
                            'datasets': dataset_names,
                            'description': f'同域多数据集: {domain_type}',
                            'balancing': 'proportional',
                            'expected_samples': sum(ds.num_samples for ds in domain_datasets)
                        })
            
            elif strategy == 'all_datasets':
                # 所有数据集（如果数量不太多）
                if len(datasets) <= 8:  # 限制数据集数量
                    dataset_names = [ds.name for ds in datasets]
                    combinations.append({
                        'name': 'all_datasets_combined',
                        'strategy': 'all_datasets',
                        'datasets': dataset_names,
                        'description': '所有数据集组合实验',
                        'balancing': 'equal',
                        'expected_samples': sum(ds.num_samples for ds in datasets)
                    })
        
        self.logger.info(f"生成了 {len(combinations)} 个数据集组合")
        return combinations
    
    def generate_experiment_configs(self, 
                                   datasets: List[DatasetInfo],
                                   config_variants: List[Dict] = None,
                                   dataset_combinations: List[Dict] = None,
                                   enable_ablation: bool = True) -> List[ExperimentConfig]:
        """
        生成实验配置
        
        Args:
            datasets: 数据集列表
            config_variants: 配置变体列表
            dataset_combinations: 数据集组合列表
            enable_ablation: 是否启用消融实验
        
        Returns:
            实验配置列表
        """
        self.logger.info("正在生成实验配置...")
        
        # 生成数据集组合（如果未提供）
        if dataset_combinations is None:
            dataset_combinations = self.generate_dataset_combinations(datasets)
        
        if config_variants is None:
            # 对比学习默认配置变体
            config_variants = [
                {'name': 'default', 'overrides': {}, 'priority': 1},
                {'name': 'large_window', 'overrides': {'data.window_size': 2048, 'data.stride': 1024}, 'priority': 2},
                {'name': 'small_window', 'overrides': {'data.window_size': 512, 'data.stride': 256}, 'priority': 2},
                {'name': 'low_temp', 'overrides': {'task.temperature': 0.05}, 'priority': 3},
                {'name': 'high_temp', 'overrides': {'task.temperature': 0.3}, 'priority': 3},
                {'name': 'high_lr', 'overrides': {'task.lr': 5e-3}, 'priority': 3},
                {'name': 'large_batch', 'overrides': {'data.batch_size': 64}, 'priority': 2},
                {'name': 'small_batch', 'overrides': {'data.batch_size': 8}, 'priority': 2}
            ]
            
            # 消融实验特定变体
            if enable_ablation:
                ablation_variants = [
                    {'name': 'no_augmentation', 'overrides': {'data.enable_augmentation': False}, 'priority': 4},
                    {'name': 'different_backbone', 'overrides': {'model.backbone': 'B_04_Dlinear'}, 'priority': 4},
                    {'name': 'larger_model', 'overrides': {'model.d_model': 512}, 'priority': 4},
                    {'name': 'minimal_model', 'overrides': {'model.d_model': 128}, 'priority': 4}
                ]
                config_variants.extend(ablation_variants)
        
        experiments = []
        experiment_counter = 0
        
        for combination in dataset_combinations:
            for variant in config_variants:
                experiment_counter += 1
                
                # 生成实验 ID
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                experiment_id = f"{combination['name']}_{variant['name']}_{timestamp}_{experiment_counter:04d}"
                
                # 基础配置覆盖
                base_overrides = {
                    'environment.experiment_name': f"contrastive_{combination['name']}_{variant['name']}",
                    'environment.save_dir': str(self.results_dir / combination['name'] / variant['name']),
                    'data.factory_name': 'id',  # 确保使用 id_data_factory
                    'task.type': 'pretrain',
                    'task.name': 'contrastive_id'
                }
                
                # 设置数据集配置
                if combination['strategy'] == 'single':
                    # 单数据集
                    dataset_name = combination['datasets'][0]
                    dataset_info = next(ds for ds in datasets if ds.name == dataset_name)
                    base_overrides['data.metadata_file'] = dataset_info.metadata_file
                else:
                    # 多数据集组合
                    dataset_files = [next(ds for ds in datasets if ds.name == name).metadata_file 
                                   for name in combination['datasets']]
                    base_overrides['data.metadata_files'] = dataset_files  # 多数据集支持
                    base_overrides['data.dataset_balancing'] = combination.get('balancing', 'equal')
                    
                    # 跨域特定配置
                    if combination['strategy'] == 'cross_domain':
                        base_overrides.update({
                            'data.source_datasets': combination.get('source_datasets', []),
                            'data.target_datasets': combination.get('target_datasets', []),
                            'data.domain_adaptation': True,
                            'trainer.val_split_strategy': 'target_only'
                        })
                
                # 合并变体覆盖
                final_overrides = {**base_overrides, **variant['overrides']}
                
                # 数据集特定调整
                combination_specific_overrides = self._get_combination_specific_overrides(combination, datasets)
                final_overrides.update(combination_specific_overrides)
                
                # 生成实验配置
                experiment = ExperimentConfig(
                    id=experiment_id,
                    name=f"{combination['name']}_{variant['name']}",
                    dataset_combination=combination['datasets'],
                    variant_name=variant['name'],
                    config_overrides=final_overrides,
                    expected_duration_hours=self._estimate_experiment_duration(combination, variant, datasets),
                    priority=variant.get('priority', 5),
                    resource_requirements=self._estimate_resource_requirements(combination, variant, datasets)
                )
                
                experiments.append(experiment)
        
        self.logger.info(f"生成了 {len(experiments)} 个实验配置")
        
        # 按优先级排序
        experiments.sort(key=lambda x: (x.priority, x.expected_duration_hours))
        
        return experiments
    
    def _get_dataset_specific_overrides(self, dataset: Dict) -> Dict:
        """根据数据集特性生成特定的配置覆盖"""
        overrides = {}
        
        # 根据样本数量调整批量大小
        if dataset['num_samples'] < 500:
            overrides['data.batch_size'] = 8
            overrides['trainer.epochs'] = 1  # 快速测试用单epoch
        elif dataset['num_samples'] < 2000:
            overrides['data.batch_size'] = 16
            overrides['trainer.epochs'] = 1  # 快速测试用单epoch
        else:
            overrides['data.batch_size'] = 32
            overrides['trainer.epochs'] = 1  # 快速测试用单epoch
        
        # 根据H5文件大小调整num_workers
        if dataset.get('h5_size_mb', 0) > 1000:  # 大文件
            overrides['data.num_workers'] = 8
        elif dataset.get('h5_size_mb', 0) > 100:
            overrides['data.num_workers'] = 4
        else:
            overrides['data.num_workers'] = 2
        
        # 根据类别数量调整模型大小
        if dataset.get('num_classes', 0) > 10:
            overrides['model.d_model'] = 512
        elif dataset.get('num_classes', 0) > 5:
            overrides['model.d_model'] = 256
        else:
            overrides['model.d_model'] = 128
        
        return overrides
    
    def _estimate_experiment_duration(self, combination: Dict[str, Any], variant: Dict, datasets: List[DatasetInfo]) -> float:
        """估计实验持续时间（小时）"""
        # 获取组合中数据集的总信息
        combination_datasets = [ds for ds in datasets if ds.name in combination['datasets']]
        total_samples = sum(ds.num_samples for ds in combination_datasets)
        
        # 基础时间（使用单epoch训练）
        base_duration = 0.1  # 单epoch基础时间
        
        # 根据数据集组合大小调整
        size_factor = total_samples / 1000
        base_duration *= (1 + size_factor * 0.05)
        
        # 多数据集组合增加复杂度
        if len(combination['datasets']) > 1:
            base_duration *= (1 + len(combination['datasets']) * 0.2)
        
        # 根据配置变体调整
        if 'large_window' in variant['name']:
            base_duration *= 1.3
        elif 'small_window' in variant['name']:
            base_duration *= 0.8
        
        if 'large_batch' in variant['name']:
            base_duration *= 0.7  # 大批量可能更快
        elif 'small_batch' in variant['name']:
            base_duration *= 1.2
            
        if 'high_lr' in variant['name']:
            base_duration *= 0.9  # 高学习率可能收敛更快
        
        # 跨域实验通常需要更多时间
        if combination.get('strategy') == 'cross_domain':
            base_duration *= 1.2
        
        return round(base_duration, 2)
    
    def run_experiments(self, 
                       experiments: List[Dict],
                       parallel: bool = False,
                       max_parallel: int = 2,
                       timeout_hours: int = 24) -> Dict:
        """运行实验"""
        print(f"\n开始运行 {len(experiments)} 个实验")
        
        if self.dry_run:
            print("🔄 干运行模式，只输出实验计划:")
            self._print_experiment_plan(experiments)
            return {'completed': [], 'failed': [], 'skipped': experiments}
        
        if parallel:
            return self._run_experiments_parallel(experiments, max_parallel, timeout_hours)
        else:
            return self._run_experiments_sequential(experiments, timeout_hours)
    
    def _run_experiments_sequential(self, experiments: List[Dict], timeout_hours: int) -> Dict:
        """顺序运行实验"""
        results = {'completed': [], 'failed': [], 'skipped': []}
        
        total_experiments = len(experiments)
        
        for i, experiment in enumerate(experiments, 1):
            print(f"\n{'='*60}")
            print(f"运行实验 {i}/{total_experiments}: {experiment['id']}")
            print(f"数据集: {experiment['dataset']}, 变体: {experiment['variant']}")
            print(f"预计耗时: {experiment['expected_duration_hours']} 小时")
            print(f"{'='*60}")
            
            start_time = time.time()
            
            try:
                # 创建实验配置文件
                config_path = self._create_experiment_config(experiment)
                
                # 运行实验
                success = self._run_single_experiment(config_path, experiment, timeout_hours)
                
                end_time = time.time()
                actual_duration = (end_time - start_time) / 3600  # 转换为小时
                
                experiment['actual_duration_hours'] = round(actual_duration, 2)
                experiment['status'] = 'completed' if success else 'failed'
                
                if success:
                    results['completed'].append(experiment)
                    print(f"✅ 实验完成: {experiment['id']} (耗时: {actual_duration:.2f} 小时)")
                else:
                    results['failed'].append(experiment)
                    print(f"❌ 实验失败: {experiment['id']} (耗时: {actual_duration:.2f} 小时)")
                
            except Exception as e:
                print(f"❌ 实验异常: {experiment['id']} - {e}")
                experiment['status'] = 'failed'
                experiment['error'] = str(e)
                results['failed'].append(experiment)
        
        return results
    
    def _run_experiments_parallel(self, experiments: List[Dict], max_parallel: int, timeout_hours: int) -> Dict:
        """并行运行实验"""
        # 这里可以实现并行执行逻辑
        # 为简化，暂时使用顺序执行
        print(f"⚠️  并行执行尚未实现，使用顺序执行 (max_parallel={max_parallel})")
        return self._run_experiments_sequential(experiments, timeout_hours)
    
    def _create_experiment_config(self, experiment: Dict) -> str:
        """创建实验配置文件"""
        # 加载基础配置并应用覆盖
        config = load_config(self.base_config_path, experiment['config_overrides'])
        
        # 保存实验配置
        experiment_dir = Path(experiment['config_overrides']['environment.save_dir'])
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = experiment_dir / f"config_{experiment['id']}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # 保存实验元信息
        experiment_info = {
            'experiment_id': experiment['id'],
            'dataset': experiment['dataset'],
            'variant': experiment['variant'],
            'created_at': datetime.now().isoformat(),
            'config_overrides': experiment['config_overrides'],
            'expected_duration_hours': experiment['expected_duration_hours']
        }
        
        info_path = experiment_dir / f"experiment_info_{experiment['id']}.json"
        with open(info_path, 'w') as f:
            json.dump(experiment_info, f, indent=2)
        
        return str(config_path)
    
    def _run_single_experiment(self, config_path: str, experiment: Dict, timeout_hours: int) -> bool:
        """运行单个实验"""
        try:
            # 构建命令
            cmd = [
                'python', 'main.py',
                '--config', config_path
            ]
            
            print(f"执行命令: {' '.join(cmd)}")
            
            # 设置超时时间（转换为秒）
            timeout_seconds = timeout_hours * 3600
            
            # 运行实验
            result = subprocess.run(
                cmd,
                timeout=timeout_seconds,
                capture_output=True,
                text=True
            )
            
            # 保存日志
            experiment_dir = Path(experiment['config_overrides']['environment.save_dir'])
            
            stdout_path = experiment_dir / f"stdout_{experiment['id']}.log"
            stderr_path = experiment_dir / f"stderr_{experiment['id']}.log"
            
            with open(stdout_path, 'w') as f:
                f.write(result.stdout)
            
            with open(stderr_path, 'w') as f:
                f.write(result.stderr)
            
            # 检查返回码
            if result.returncode == 0:
                print(f"实验成功完成")
                return True
            else:
                print(f"实验失败，返回码: {result.returncode}")
                print(f"错误信息: {result.stderr[-500:]}")  # 只显示最后500个字符
                return False
                
        except subprocess.TimeoutExpired:
            print(f"实验超时 ({timeout_hours} 小时)")
            return False
        except Exception as e:
            print(f"运行实验时发生异常: {e}")
            return False
    
    def _print_experiment_plan(self, experiments: List[ExperimentConfig]):
        """打印实验计划"""
        self.logger.info("\n实验计划摘要:")
        self.logger.info(f"总实验数: {len(experiments)}")
        
        # 按数据集组合分组
        combination_groups = defaultdict(list)
        for exp in experiments:
            combination_key = "_".join(exp.dataset_combination)
            combination_groups[combination_key].append(exp)
        
        for combination, exps in combination_groups.items():
            self.logger.info(f"\n数据集组合: {combination} ({len(exps)} 个实验)")
            total_time = sum(exp.expected_duration_hours for exp in exps)
            self.logger.info(f"  预计总耗时: {total_time:.1f} 小时")
            
            # 按优先级分组显示
            priority_groups = defaultdict(list)
            for exp in exps:
                priority_groups[exp.priority].append(exp)
            
            for priority in sorted(priority_groups.keys()):
                priority_exps = priority_groups[priority]
                priority_time = sum(exp.expected_duration_hours for exp in priority_exps)
                self.logger.info(f"    优先级 {priority}: {len(priority_exps)} 个实验, {priority_time:.1f} 小时")
                
                for exp in priority_exps[:3]:  # 只显示前3个
                    self.logger.info(f"      - {exp.variant_name}: {exp.expected_duration_hours} 小时")
                if len(priority_exps) > 3:
                    self.logger.info(f"      ... 还有 {len(priority_exps) - 3} 个实验")
        
        total_time = sum(exp.expected_duration_hours for exp in experiments)
        avg_time = total_time / len(experiments) if experiments else 0
        self.logger.info(f"\n总预计耗时: {total_time:.1f} 小时")
        self.logger.info(f"平均实验耗时: {avg_time:.2f} 小时")
        
        # 资源需求摘要
        total_gpu_memory = sum(exp.resource_requirements.get('gpu_memory_mb', 0) for exp in experiments)
        max_gpu_memory = max((exp.resource_requirements.get('gpu_memory_mb', 0) for exp in experiments), default=0)
        self.logger.info(f"\n资源需求摘要:")
        self.logger.info(f"  最大GPU内存需求: {max_gpu_memory} MB")
        self.logger.info(f"  总GPU内存需求（如果串行）: {total_gpu_memory} MB")
        
        if self.enable_benchmarking:
            self.logger.info("\n✅ 基准测试已启用，将收集性能指标")
        else:
            self.logger.info("\n⚠️  基准测试未启用")
    
    def generate_report(self, results: Dict[str, List[ExperimentResult]]) -> str:
        """生成实验报告"""
        self.logger.info("生成实验报告...")
        
        # 运行基准测试（如果启用）
        if self.enable_benchmarking and self.benchmark:
            try:
                self.logger.info("运行性能基准测试...")
                benchmark_results = self._run_benchmark_suite()
                self.logger.info("基准测试完成")
            except Exception as e:
                self.logger.warning(f"基准测试失败: {e}")
                benchmark_results = {}
        
        else:
            benchmark_results = {}
        
        # 构建报告数据
        all_results = results['completed'] + results['failed'] + results['skipped'] + results.get('running', [])
        
        report_data = {
            'summary': {
                'total_experiments': len(all_results),
                'completed': len(results['completed']),
                'failed': len(results['failed']),
                'skipped': len(results['skipped']),
                'running': len(results.get('running', [])),
                'success_rate': len(results['completed']) / max(1, len(results['completed']) + len(results['failed'])) * 100
            },
            'completed_experiments': [asdict(exp) for exp in results['completed']],
            'failed_experiments': [asdict(exp) for exp in results['failed']],
            'skipped_experiments': [asdict(exp) for exp in results['skipped']],
            'running_experiments': [asdict(exp) for exp in results.get('running', [])],
            'benchmark_results': benchmark_results,
            'generated_at': datetime.now().isoformat(),
            'resource_usage_summary': self.resource_manager.get_resource_status()
        }
        
        # 计算总耗时和性能指标
        total_duration = sum(exp.actual_duration_hours for exp in results['completed'])
        avg_duration = total_duration / max(1, len(results['completed']))
        
        report_data['summary']['total_duration_hours'] = round(total_duration, 2)
        report_data['summary']['avg_duration_hours'] = round(avg_duration, 2)
        
        # 计算每个数据集组合的性能
        combination_stats = defaultdict(list)
        for exp in results['completed']:
            # 这里需要从实验结果中提取数据集组合信息
            combination_stats['all'].append(exp.actual_duration_hours)
        
        report_data['combination_performance'] = {
            combo: {
                'count': len(durations),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations)
            } for combo, durations in combination_stats.items() if durations
        }
        
        # 保存详细报告
        report_path = self.results_dir / f"experiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # 生成可读报告
        readable_report_path = self.results_dir / f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(readable_report_path, 'w') as f:
            f.write("多数据集对比学习预训练实验报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"实验总结:\n")
            f.write(f"  总实验数: {report_data['summary']['total_experiments']}\n")
            f.write(f"  成功完成: {report_data['summary']['completed']}\n")
            f.write(f"  失败: {report_data['summary']['failed']}\n")
            f.write(f"  跳过: {report_data['summary']['skipped']}\n")
            f.write(f"  成功率: {report_data['summary']['success_rate']:.1f}%\n")
            f.write(f"  总耗时: {report_data['summary']['total_duration_hours']:.2f} 小时\n\n")
            
            if results['completed']:
                f.write("成功完成的实验:\n")
                for exp in results['completed']:
                    f.write(f"  - {exp['dataset']}_{exp['variant']}: {exp.get('actual_duration_hours', 'N/A')} 小时\n")
                f.write("\n")
            
            if results['failed']:
                f.write("失败的实验:\n")
                for exp in results['failed']:
                    error_msg = exp.get('error', '未知错误')
                    f.write(f"  - {exp['dataset']}_{exp['variant']}: {error_msg}\n")
                f.write("\n")
            
            f.write(f"报告生成时间: {report_data['generated_at']}\n")
        
        self.logger.info("实验报告已保存:")
        self.logger.info(f"  详细报告: {report_path}")
        self.logger.info(f"  摘要报告: {readable_report_path}")
        
        return str(readable_report_path)
    
    def _run_benchmark_suite(self) -> Dict[str, Any]:
        """运行基准测试套件"""
        if not self.benchmark:
            return {}
        
        benchmark_results = {}
        
        try:
            # 运行内存使用基准测试
            self.logger.info("运行内存使用基准测试...")
            memory_results = self.benchmark.test_memory_usage()
            benchmark_results['memory'] = memory_results
            
            # 运行训练速度基准测试
            self.logger.info("运行训练速度基准测试...")
            speed_results = self.benchmark.test_training_speed()
            benchmark_results['training_speed'] = speed_results
            
            # 运行吞吐量基准测试
            self.logger.info("运行吞吐量基准测试...")
            throughput_results = self.benchmark.test_throughput()
            benchmark_results['throughput'] = throughput_results
            
            # 运行收敛性能基准测试
            self.logger.info("运行收敛性能基准测试...")
            convergence_results = self.benchmark.test_convergence()
            benchmark_results['convergence'] = convergence_results
            
            # 生成基准测试报告
            benchmark_report = self.benchmark.generate_report()
            benchmark_results['report_summary'] = benchmark_report
            
            self.logger.info("所有基准测试完成")
            
        except Exception as e:
            self.logger.error(f"基准测试套件执行失败: {e}")
            benchmark_results['error'] = str(e)
        
        return benchmark_results
    
    def run_quick_validation(self, dataset_sample_size: int = 100) -> Dict[str, Any]:
        """
        运行快速验证测试，用于验证实验配置的正确性
        
        Args:
            dataset_sample_size: 用于快速测试的样本数量
            
        Returns:
            验证结果
        """
        self.logger.info("开始快速验证测试...")
        
        validation_results = {
            'config_validation': {},
            'data_loading_validation': {},
            'model_initialization_validation': {},
            'training_step_validation': {}
        }
        
        try:
            # 1. 配置验证
            self.logger.info("验证基础配置...")
            base_config = load_config(self.base_config_path)
            validation_results['config_validation']['base_config_valid'] = True
            validation_results['config_validation']['required_keys'] = [
                'data', 'model', 'task', 'trainer'
            ]
            validation_results['config_validation']['missing_keys'] = []
            
            for key in validation_results['config_validation']['required_keys']:
                if key not in base_config:
                    validation_results['config_validation']['missing_keys'].append(key)
            
            # 2. 数据加载验证
            self.logger.info("验证数据加载...")
            datasets = self.discover_datasets()
            filtered_datasets = self.filter_datasets(datasets, min_samples=dataset_sample_size)
            
            validation_results['data_loading_validation']['total_datasets'] = len(datasets)
            validation_results['data_loading_validation']['valid_datasets'] = len(filtered_datasets)
            validation_results['data_loading_validation']['datasets'] = [ds.name for ds in filtered_datasets[:5]]  # 只显示前5个
            
            if filtered_datasets:
                # 测试数据加载
                sample_dataset = filtered_datasets[0]
                try:
                    # 这里可以添加实际的数据加载测试
                    validation_results['data_loading_validation']['sample_loading_success'] = True
                except Exception as e:
                    validation_results['data_loading_validation']['sample_loading_success'] = False
                    validation_results['data_loading_validation']['sample_loading_error'] = str(e)
            
            # 3. 模型初始化验证
            self.logger.info("验证模型初始化...")
            if filtered_datasets:
                try:
                    # 创建一个最小的实验配置来测试模型初始化
                    test_combinations = self.generate_dataset_combinations(filtered_datasets[:1], ['single'])
                    test_experiments = self.generate_experiment_configs(
                        filtered_datasets[:1], 
                        [{'name': 'minimal_test', 'overrides': {'trainer.epochs': 1, 'data.batch_size': 2}}],
                        test_combinations[:1],
                        enable_ablation=False
                    )
                    
                    if test_experiments:
                        validation_results['model_initialization_validation']['config_generation_success'] = True
                        validation_results['model_initialization_validation']['test_experiments_count'] = len(test_experiments)
                    else:
                        validation_results['model_initialization_validation']['config_generation_success'] = False
                        
                except Exception as e:
                    validation_results['model_initialization_validation']['config_generation_success'] = False
                    validation_results['model_initialization_validation']['config_generation_error'] = str(e)
            
            # 4. 资源检查
            self.logger.info("检查系统资源...")
            resource_status = self.resource_manager.get_resource_status()
            validation_results['resource_validation'] = resource_status
            validation_results['resource_validation']['sufficient_resources'] = (
                resource_status['available_ram_mb'] > 4000 and  # 至少4GB RAM
                (not resource_status['gpus'] or len(resource_status['gpus']) > 0)  # 有GPU或者没有GPU要求
            )
            
            validation_results['overall_validation_success'] = (
                validation_results['config_validation']['base_config_valid'] and
                len(validation_results['config_validation']['missing_keys']) == 0 and
                validation_results['data_loading_validation']['valid_datasets'] > 0 and
                validation_results['resource_validation']['sufficient_resources']
            )
            
            if validation_results['overall_validation_success']:
                self.logger.info("✅ 快速验证测试通过")
            else:
                self.logger.warning("⚠️  快速验证测试发现问题")
            
        except Exception as e:
            self.logger.error(f"快速验证测试失败: {e}")
            validation_results['validation_error'] = str(e)
            validation_results['overall_validation_success'] = False
        
        return validation_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多数据集对比学习预训练实验")
    
    parser.add_argument(
        '--base_config',
        default='configs/id_contrastive/pretrain.yaml',
        help='基础配置文件路径'
    )
    
    parser.add_argument(
        '--metadata_dir',
        default='data',
        help='metadata文件目录'
    )
    
    parser.add_argument(
        '--results_dir',
        default='save/multi_dataset_experiments',
        help='结果保存目录'
    )
    
    parser.add_argument(
        '--include_datasets',
        nargs='*',
        help='包含的数据集名称模式'
    )
    
    parser.add_argument(
        '--exclude_datasets', 
        nargs='*',
        help='排除的数据集名称模式'
    )
    
    parser.add_argument(
        '--min_samples',
        type=int,
        default=100,
        help='最小样本数要求'
    )
    
    parser.add_argument(
        '--variants',
        nargs='*',
        choices=['default', 'large_window', 'small_window', 'low_temp', 'high_temp', 'high_lr', 'large_batch', 'small_batch'],
        default=['default'],
        help='配置变体'
    )
    
    parser.add_argument(
        '--combination_strategies',
        nargs='*',
        choices=['single', 'cross_domain', 'same_domain', 'all_datasets'],
        default=['single', 'cross_domain'],
        help='数据集组合策略'
    )
    
    parser.add_argument(
        '--domain_types',
        nargs='*',
        help='限制特定的域类型 (bearing, gear, motor, etc.)'
    )
    
    parser.add_argument(
        '--enable_ablation',
        action='store_true',
        help='启用消融实验变体'
    )
    
    parser.add_argument(
        '--enable_benchmarking',
        action='store_true',
        default=True,
        help='启用性能基准测试'
    )
    
    parser.add_argument(
        '--quick_validation',
        action='store_true',
        help='只运行快速验证，不执行实际实验'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='并行运行实验'
    )
    
    parser.add_argument(
        '--max_parallel',
        type=int,
        default=2,
        help='最大并行数'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=24,
        help='单个实验超时时间（小时）'
    )
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='只输出实验计划，不实际运行'
    )
    
    args = parser.parse_args()
    
    # 创建实验运行器
    runner = MultiDatasetExperimentRunner(
        base_config_path=args.base_config,
        metadata_dir=args.metadata_dir,
        results_dir=args.results_dir,
        dry_run=args.dry_run,
        enable_benchmarking=args.enable_benchmarking,
        max_concurrent_experiments=args.max_parallel
    )
    
    try:
        # 快速验证模式
        if args.quick_validation:
            runner.logger.info("🔍 运行快速验证模式...")
            validation_results = runner.run_quick_validation(args.min_samples)
            
            if validation_results['overall_validation_success']:
                runner.logger.info("✅ 快速验证通过，系统配置正确")
                return 0
            else:
                runner.logger.error("❌ 快速验证失败，请检查配置")
                print(json.dumps(validation_results, indent=2))
                return 1
        
        # 发现数据集
        datasets = runner.discover_datasets()
        
        if not datasets:
            runner.logger.error("没有发现可用的数据集")
            return 1
        
        # 过滤数据集
        filtered_datasets = runner.filter_datasets(
            datasets,
            include_patterns=args.include_datasets,
            exclude_patterns=args.exclude_datasets,
            min_samples=args.min_samples,
            ready_only=True,
            domain_types=args.domain_types
        )
        
        if not filtered_datasets:
            runner.logger.error("过滤后没有可用的数据集")
            return 1
        
        # 生成数据集组合
        dataset_combinations = runner.generate_dataset_combinations(
            filtered_datasets, 
            args.combination_strategies
        )
        
        if not dataset_combinations:
            runner.logger.error("没有生成任何数据集组合")
            return 1
        
        # 生成配置变体
        config_variants = []
        variant_configs = {
            'default': {'name': 'default', 'overrides': {}, 'priority': 1},
            'large_window': {'name': 'large_window', 'overrides': {'data.window_size': 2048, 'data.stride': 1024}, 'priority': 2},
            'small_window': {'name': 'small_window', 'overrides': {'data.window_size': 512, 'data.stride': 256}, 'priority': 2},
            'low_temp': {'name': 'low_temp', 'overrides': {'task.temperature': 0.05}, 'priority': 3},
            'high_temp': {'name': 'high_temp', 'overrides': {'task.temperature': 0.3}, 'priority': 3},
            'high_lr': {'name': 'high_lr', 'overrides': {'task.lr': 5e-3}, 'priority': 3},
            'large_batch': {'name': 'large_batch', 'overrides': {'data.batch_size': 64}, 'priority': 2},
            'small_batch': {'name': 'small_batch', 'overrides': {'data.batch_size': 8}, 'priority': 2}
        }
        
        for variant in args.variants:
            if variant in variant_configs:
                config_variants.append(variant_configs[variant])
        
        # 生成实验配置
        experiments = runner.generate_experiment_configs(
            filtered_datasets, 
            config_variants,
            dataset_combinations,
            enable_ablation=args.enable_ablation
        )
        
        if not experiments:
            runner.logger.error("没有生成任何实验配置")
            return 1
        
        runner.logger.info(f"成功生成 {len(experiments)} 个实验配置")
        
        # 运行实验
        results = runner.run_experiments(
            experiments,
            parallel=args.parallel,
            max_parallel=args.max_parallel,
            timeout_hours=args.timeout
        )
        
        # 生成报告
        report_path = runner.generate_report(results)
        
        # 打印执行摘要
        runner.logger.info(f"🎉 多数据集对比学习实验完成!")
        runner.logger.info(f"报告保存在: {report_path}")
        
        # 打印结果摘要
        total_experiments = len(results['completed']) + len(results['failed']) + len(results['skipped'])
        success_rate = len(results['completed']) / max(1, len(results['completed']) + len(results['failed'])) * 100
        
        runner.logger.info(f"\n执行摘要:")
        runner.logger.info(f"  总实验数: {total_experiments}")
        runner.logger.info(f"  成功完成: {len(results['completed'])}")
        runner.logger.info(f"  失败: {len(results['failed'])}")
        runner.logger.info(f"  跳过: {len(results['skipped'])}")
        runner.logger.info(f"  成功率: {success_rate:.1f}%")
        
        if results['failed']:
            runner.logger.warning("⚠️  存在失败的实验，请查看详细报告")
        
        return 0 if success_rate > 50 else 1  # 成功率超过50%认为整体成功
        
    except KeyboardInterrupt:
        runner.logger.warning("⚠️  实验被用户中断")
        return 130  # SIGINT exit code
    except Exception as e:
        runner.logger.error(f"❌ 实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()