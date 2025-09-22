#!/usr/bin/env python3
"""
多数据集对比学习实验运行器 - ContrastiveIDTask专用

科研导向的多数据集实验管理系统，支持：
- 单数据集和跨数据集域泛化实验
- 智能资源管理和并行执行
- 系统化的实验配置和结果管理
- 论文发表级别的详细报告

Usage:
    # 快速单数据集实验
    python multi_dataset_runner.py --quick --datasets CWRU

    # 跨域泛化实验
    python multi_dataset_runner.py --strategy cross_domain --source CWRU --target XJTU

    # 完整多数据集评估
    python multi_dataset_runner.py --strategy all --enable_ablation

Author: PHM-Vibench Team
Version: 3.0 (Streamlined for loop_id)
"""

import os
import sys
import json
import yaml
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import subprocess
import time
import logging
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict
import torch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.configs import load_config

@dataclass
class DatasetInfo:
    """数据集信息"""
    name: str
    metadata_file: str
    num_samples: int
    num_classes: int = 0
    num_ids: int = 0
    domain_type: str = "unknown"
    ready: bool = False

@dataclass
class ExperimentConfig:
    """实验配置"""
    id: str
    name: str
    datasets: List[str]
    strategy: str  # single, cross_domain, multi_dataset
    config_overrides: Dict[str, Any]
    priority: int = 5
    expected_duration_min: float = 10.0

@dataclass
class ExperimentResult:
    """实验结果"""
    experiment_id: str
    status: str  # completed, failed, running
    start_time: datetime = None
    end_time: datetime = None
    duration_min: float = 0.0
    metrics: Dict[str, float] = None
    error_message: str = ""
    output_dir: str = ""

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

class MultiDatasetRunner:
    """多数据集实验运行器

    核心功能：
    1. 数据集自动发现和筛选
    2. 实验配置生成（单域、跨域、多域）
    3. 智能资源管理
    4. 结果收集和报告生成
    """

    def __init__(self,
                 base_config: str = "configs/id_contrastive/debug.yaml",
                 data_dir: str = "data",
                 results_dir: str = "save/multi_dataset",
                 dry_run: bool = False):
        self.base_config = base_config
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.dry_run = dry_run

        # 创建结果目录
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        self._setup_logging()

        # 实验管理
        self.experiments: List[ExperimentConfig] = []
        self.results: List[ExperimentResult] = []

        self.logger.info(f"多数据集实验运行器初始化完成")
        self.logger.info(f"基础配置: {base_config}")
        self.logger.info(f"数据目录: {data_dir}")
        self.logger.info(f"结果目录: {results_dir}")
        self.logger.info(f"干运行模式: {dry_run}")

    def _setup_logging(self):
        """设置日志"""
        log_file = self.results_dir / "experiment.log"

        self.logger = logging.getLogger('MultiDatasetRunner')
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

    def discover_datasets(self,
                         include_patterns: List[str] = None,
                         exclude_patterns: List[str] = None,
                         min_samples: int = 100) -> List[DatasetInfo]:
        """发现可用数据集"""
        self.logger.info(f"扫描数据目录: {self.data_dir}")

        datasets = []
        metadata_files = list(self.data_dir.glob("metadata_*.xlsx"))

        if not metadata_files:
            self.logger.warning(f"未找到metadata文件")
            return datasets

        for metadata_file in metadata_files:
            try:
                # 提取数据集名称
                dataset_name = metadata_file.stem.replace('metadata_', '')

                # 读取metadata
                df = pd.read_excel(metadata_file, sheet_name=0)

                # 检查样本数量
                if len(df) < min_samples:
                    self.logger.debug(f"跳过样本不足的数据集: {dataset_name} ({len(df)} < {min_samples})")
                    continue

                # 应用过滤
                if include_patterns and not any(pattern.lower() in dataset_name.lower() for pattern in include_patterns):
                    continue

                if exclude_patterns and any(pattern.lower() in dataset_name.lower() for pattern in exclude_patterns):
                    continue

                # 检查H5文件
                h5_file = self.data_dir / f"{dataset_name}.h5"
                if not h5_file.exists():
                    self.logger.warning(f"数据集 {dataset_name} 的H5文件不存在")
                    continue

                # 创建数据集信息
                dataset_info = DatasetInfo(
                    name=dataset_name,
                    metadata_file=str(metadata_file),
                    num_samples=len(df),
                    num_classes=df.get('Label', df.get('label', pd.Series())).nunique(),
                    num_ids=df.get('ID', df.get('id', pd.Series())).nunique(),
                    domain_type=self._infer_domain_type(dataset_name),
                    ready=True
                )

                datasets.append(dataset_info)
                self.logger.info(f"发现数据集: {dataset_name} ({dataset_info.num_samples}样本, {dataset_info.domain_type})")

            except Exception as e:
                self.logger.error(f"处理metadata失败 {metadata_file}: {e}")

        self.logger.info(f"共发现 {len(datasets)} 个可用数据集")
        return datasets

    def _infer_domain_type(self, dataset_name: str) -> str:
        """推断域类型"""
        name = dataset_name.lower()

        if any(keyword in name for keyword in ['cwru', 'bearing', 'ball']):
            return 'bearing'
        elif any(keyword in name for keyword in ['gear', 'gearbox']):
            return 'gear'
        elif any(keyword in name for keyword in ['motor', 'rotor']):
            return 'motor'
        elif any(keyword in name for keyword in ['pump']):
            return 'pump'
        else:
            return 'unknown'

    def generate_experiments(self,
                           datasets: List[DatasetInfo],
                           strategy: str = "single",
                           config_variants: List[str] = None,
                           source_datasets: List[str] = None,
                           target_datasets: List[str] = None) -> List[ExperimentConfig]:
        """生成实验配置

        Args:
            datasets: 数据集列表
            strategy: 实验策略 (single, cross_domain, multi_dataset)
            config_variants: 配置变体
            source_datasets: 源域数据集（跨域实验用）
            target_datasets: 目标域数据集（跨域实验用）
        """
        self.logger.info(f"生成 {strategy} 实验配置...")

        if config_variants is None:
            config_variants = ['debug']  # 默认使用debug配置

        experiments = []

        if strategy == "single":
            # 单数据集实验
            for dataset in datasets:
                for variant in config_variants:
                    exp_id = f"single_{dataset.name}_{variant}_{datetime.now().strftime('%H%M%S')}"

                    experiment = ExperimentConfig(
                        id=exp_id,
                        name=f"单数据集_{dataset.name}_{variant}",
                        datasets=[dataset.name],
                        strategy="single",
                        config_overrides={
                            'data.metadata_file': dataset.metadata_file,
                            'experiment.name': exp_id,
                            'experiment.save_dir': str(self.results_dir / "single" / dataset.name / variant),
                        },
                        priority=1,
                        expected_duration_min=self._estimate_duration(dataset, variant)
                    )

                    # 根据变体添加特定配置
                    if variant == 'quick':
                        experiment.config_overrides.update({
                            'trainer.epochs': 1,
                            'data.batch_size': 8,
                            'data.window_size': 512
                        })
                    elif variant == 'production':
                        experiment.config_overrides.update({
                            'trainer.epochs': 100,
                            'data.batch_size': 32,
                            'data.window_size': 1024
                        })

                    experiments.append(experiment)

        elif strategy == "cross_domain":
            # 跨域泛化实验
            if source_datasets and target_datasets:
                # 指定源域和目标域
                sources = [ds for ds in datasets if ds.name in source_datasets]
                targets = [ds for ds in datasets if ds.name in target_datasets]
            else:
                # 自动生成跨域组合
                domain_groups = defaultdict(list)
                for ds in datasets:
                    domain_groups[ds.domain_type].append(ds)

                sources, targets = [], []
                domain_types = list(domain_groups.keys())
                for i, source_domain in enumerate(domain_types):
                    for target_domain in domain_types[i+1:]:
                        if source_domain != target_domain:
                            sources.extend(domain_groups[source_domain][:1])  # 每域选1个
                            targets.extend(domain_groups[target_domain][:1])

            for source in sources:
                for target in targets:
                    if source.domain_type != target.domain_type:
                        for variant in config_variants:
                            exp_id = f"cross_{source.name}_to_{target.name}_{variant}_{datetime.now().strftime('%H%M%S')}"

                            experiment = ExperimentConfig(
                                id=exp_id,
                                name=f"跨域_{source.name}_to_{target.name}_{variant}",
                                datasets=[source.name, target.name],
                                strategy="cross_domain",
                                config_overrides={
                                    'data.source_datasets': [source.metadata_file],
                                    'data.target_datasets': [target.metadata_file],
                                    'experiment.name': exp_id,
                                    'experiment.save_dir': str(self.results_dir / "cross_domain" / f"{source.name}_to_{target.name}" / variant),
                                },
                                priority=2,
                                expected_duration_min=self._estimate_duration([source, target], variant)
                            )

                            experiments.append(experiment)

        elif strategy == "multi_dataset":
            # 多数据集组合实验
            for variant in config_variants:
                exp_id = f"multi_all_{variant}_{datetime.now().strftime('%H%M%S')}"

                experiment = ExperimentConfig(
                    id=exp_id,
                    name=f"多数据集_所有_{variant}",
                    datasets=[ds.name for ds in datasets],
                    strategy="multi_dataset",
                    config_overrides={
                        'data.metadata_files': [ds.metadata_file for ds in datasets],
                        'data.dataset_balancing': 'proportional',
                        'experiment.name': exp_id,
                        'experiment.save_dir': str(self.results_dir / "multi_dataset" / variant),
                    },
                    priority=3,
                    expected_duration_min=self._estimate_duration(datasets, variant)
                )

                experiments.append(experiment)

        # 按优先级和预期时间排序
        experiments.sort(key=lambda x: (x.priority, x.expected_duration_min))

        self.logger.info(f"生成 {len(experiments)} 个实验配置")
        return experiments

    def _estimate_duration(self, datasets, variant: str) -> float:
        """估计实验持续时间（分钟）"""
        if isinstance(datasets, DatasetInfo):
            datasets = [datasets]

        total_samples = sum(ds.num_samples for ds in datasets)

        # 基础时间估计
        if variant == 'quick' or variant == 'debug':
            base_time = 5.0  # 5分钟
        elif variant == 'production':
            base_time = 120.0  # 2小时
        else:
            base_time = 30.0  # 30分钟

        # 根据数据量调整
        sample_factor = np.log10(max(total_samples, 100)) / 3
        duration = base_time * (1 + sample_factor)

        return round(duration, 1)

    def run_experiments(self,
                       experiments: List[ExperimentConfig],
                       parallel: bool = False,
                       timeout_min: int = 180) -> List[ExperimentResult]:
        """运行实验"""
        self.logger.info(f"开始运行 {len(experiments)} 个实验")

        if self.dry_run:
            self.logger.info("干运行模式 - 只显示实验计划")
            self._print_experiment_plan(experiments)
            return []

        results = []

        for i, experiment in enumerate(experiments, 1):
            self.logger.info(f"\n运行实验 {i}/{len(experiments)}: {experiment.name}")
            self.logger.info(f"数据集: {experiment.datasets}")
            self.logger.info(f"预计耗时: {experiment.expected_duration_min:.1f} 分钟")

            # 运行单个实验
            result = self._run_single_experiment(experiment, timeout_min)
            results.append(result)

            # 实时保存结果
            self._save_results(results)

            # 状态汇报
            if result.status == "completed":
                self.logger.info(f"✅ 实验完成: {result.experiment_id} ({result.duration_min:.1f}分钟)")
            else:
                self.logger.error(f"❌ 实验失败: {result.experiment_id} - {result.error_message}")

        return results

    def _run_single_experiment(self, experiment: ExperimentConfig, timeout_min: int) -> ExperimentResult:
        """运行单个实验"""
        result = ExperimentResult(
            experiment_id=experiment.id,
            status="running",
            start_time=datetime.now()
        )

        try:
            # 创建实验配置
            config_path = self._create_experiment_config(experiment)
            result.output_dir = str(Path(config_path).parent)

            # 构建命令
            cmd = [
                'python', 'main.py',
                '--pipeline', 'Pipeline_ID',
                '--config_path', config_path,
                '--notes', f"实验: {experiment.name}"
            ]

            self.logger.info(f"执行命令: {' '.join(cmd)}")

            # 运行实验
            start_time = time.time()
            result_proc = subprocess.run(
                cmd,
                timeout=timeout_min * 60,  # 转换为秒
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent.parent.parent)
            )
            end_time = time.time()

            result.end_time = datetime.now()
            result.duration_min = (end_time - start_time) / 60

            # 保存日志
            output_dir = Path(result.output_dir)
            with open(output_dir / "stdout.log", 'w') as f:
                f.write(result_proc.stdout)
            with open(output_dir / "stderr.log", 'w') as f:
                f.write(result_proc.stderr)

            # 检查结果
            if result_proc.returncode == 0:
                result.status = "completed"
                # 尝试提取性能指标
                result.metrics = self._extract_metrics(output_dir)
            else:
                result.status = "failed"
                result.error_message = f"退出码: {result_proc.returncode}"
                if result_proc.stderr:
                    result.error_message += f" - {result_proc.stderr[-300:]}"

        except subprocess.TimeoutExpired:
            result.status = "failed"
            result.error_message = f"超时 ({timeout_min} 分钟)"
            result.end_time = datetime.now()
            result.duration_min = timeout_min

        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            result.end_time = datetime.now()

        return result

    def _create_experiment_config(self, experiment: ExperimentConfig) -> str:
        """创建实验配置文件"""
        # 加载基础配置
        base_config = load_config(self.base_config)

        # 应用覆盖
        config_dict = dict(base_config)
        for key, value in experiment.config_overrides.items():
            keys = key.split('.')
            current = config_dict
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

        # 保存配置
        output_dir = Path(experiment.config_overrides.get('experiment.save_dir', self.results_dir / experiment.id))
        output_dir.mkdir(parents=True, exist_ok=True)

        config_path = output_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        # 保存实验信息
        exp_info = {
            'experiment_id': experiment.id,
            'name': experiment.name,
            'datasets': experiment.datasets,
            'strategy': experiment.strategy,
            'created_at': datetime.now().isoformat(),
            'config_overrides': experiment.config_overrides
        }

        with open(output_dir / "experiment_info.json", 'w') as f:
            json.dump(exp_info, f, indent=2)

        return str(config_path)

    def _extract_metrics(self, output_dir: Path) -> Dict[str, float]:
        """从输出目录提取性能指标"""
        metrics = {}

        # 查找可能的指标文件
        metric_files = [
            output_dir / "metrics.json",
            output_dir / "results.json",
            output_dir / "train_metrics.json"
        ]

        for metric_file in metric_files:
            if metric_file.exists():
                try:
                    with open(metric_file, 'r') as f:
                        data = json.load(f)

                    # 提取关键指标
                    if 'train_loss' in data:
                        metrics['final_train_loss'] = float(data['train_loss'])
                    if 'val_loss' in data:
                        metrics['final_val_loss'] = float(data['val_loss'])
                    if 'contrastive_accuracy' in data:
                        metrics['contrastive_accuracy'] = float(data['contrastive_accuracy'])
                    if 'best_val_loss' in data:
                        metrics['best_val_loss'] = float(data['best_val_loss'])

                    break

                except Exception as e:
                    self.logger.debug(f"提取指标失败 {metric_file}: {e}")

        return metrics

    def _save_results(self, results: List[ExperimentResult]):
        """保存实验结果"""
        results_data = []
        for result in results:
            result_dict = asdict(result)
            if result_dict['start_time']:
                result_dict['start_time'] = result.start_time.isoformat()
            if result_dict['end_time']:
                result_dict['end_time'] = result.end_time.isoformat()
            results_data.append(result_dict)

        results_file = self.results_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

    def _print_experiment_plan(self, experiments: List[ExperimentConfig]):
        """打印实验计划"""
        print(f"\n{'='*60}")
        print(f"实验计划摘要")
        print(f"{'='*60}")
        print(f"总实验数: {len(experiments)}")

        # 按策略分组
        strategy_groups = defaultdict(list)
        for exp in experiments:
            strategy_groups[exp.strategy].append(exp)

        for strategy, exps in strategy_groups.items():
            print(f"\n{strategy} 实验 ({len(exps)}个):")
            total_time = sum(exp.expected_duration_min for exp in exps)
            print(f"  预计总耗时: {total_time:.1f} 分钟 ({total_time/60:.1f} 小时)")

            for exp in exps[:3]:  # 显示前3个
                print(f"    - {exp.name}: {exp.expected_duration_min:.1f}分钟")
            if len(exps) > 3:
                print(f"    ... 还有 {len(exps)-3} 个实验")

        total_time = sum(exp.expected_duration_min for exp in experiments)
        print(f"\n预计总耗时: {total_time:.1f} 分钟 ({total_time/60:.1f} 小时)")

    def generate_report(self, results: List[ExperimentResult]) -> str:
        """生成实验报告"""
        self.logger.info("生成实验报告...")

        # 统计数据
        completed = [r for r in results if r.status == "completed"]
        failed = [r for r in results if r.status == "failed"]

        summary = {
            'total_experiments': len(results),
            'completed': len(completed),
            'failed': len(failed),
            'success_rate': len(completed) / max(1, len(results)) * 100,
            'total_duration_hours': sum(r.duration_min for r in completed) / 60,
            'avg_duration_min': sum(r.duration_min for r in completed) / max(1, len(completed))
        }

        # 性能分析
        performance_data = []
        for result in completed:
            if result.metrics:
                performance_data.append({
                    'experiment_id': result.experiment_id,
                    'datasets': result.experiment_id.split('_')[1:-2],  # 提取数据集名称
                    'duration_min': result.duration_min,
                    **result.metrics
                })

        # 保存详细报告
        report_data = {
            'summary': summary,
            'completed_experiments': [asdict(r) for r in completed],
            'failed_experiments': [asdict(r) for r in failed],
            'performance_analysis': performance_data,
            'generated_at': datetime.now().isoformat()
        }

        report_file = self.results_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        # 生成可读报告
        readable_report = self.results_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(readable_report, 'w') as f:
            f.write("# 多数据集对比学习实验报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 实验摘要\n\n")
            f.write(f"- **总实验数**: {summary['total_experiments']}\n")
            f.write(f"- **成功完成**: {summary['completed']}\n")
            f.write(f"- **失败**: {summary['failed']}\n")
            f.write(f"- **成功率**: {summary['success_rate']:.1f}%\n")
            f.write(f"- **总耗时**: {summary['total_duration_hours']:.1f} 小时\n")
            f.write(f"- **平均耗时**: {summary['avg_duration_min']:.1f} 分钟/实验\n\n")

            if completed:
                f.write("## 成功实验列表\n\n")
                for result in completed:
                    f.write(f"### {result.experiment_id}\n")
                    f.write(f"- **耗时**: {result.duration_min:.1f} 分钟\n")
                    if result.metrics:
                        f.write("- **关键指标**:\n")
                        for key, value in result.metrics.items():
                            f.write(f"  - {key}: {value:.4f}\n")
                    f.write(f"- **输出目录**: {result.output_dir}\n\n")

            if failed:
                f.write("## 失败实验列表\n\n")
                for result in failed:
                    f.write(f"### {result.experiment_id}\n")
                    f.write(f"- **错误信息**: {result.error_message}\n")
                    if result.output_dir:
                        f.write(f"- **日志目录**: {result.output_dir}\n")
                    f.write("\n")

        self.logger.info(f"报告已生成:")
        self.logger.info(f"  详细报告: {report_file}")
        self.logger.info(f"  摘要报告: {readable_report}")

        return str(readable_report)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多数据集对比学习实验运行器")

    # 基础参数
    parser.add_argument('--base_config', default='configs/id_contrastive/debug.yaml',
                       help='基础配置文件')
    parser.add_argument('--data_dir', default='data', help='数据目录')
    parser.add_argument('--results_dir', default='save/multi_dataset', help='结果目录')

    # 数据集选择
    parser.add_argument('--datasets', nargs='*', help='指定数据集名称')
    parser.add_argument('--include_patterns', nargs='*', help='包含模式')
    parser.add_argument('--exclude_patterns', nargs='*', help='排除模式')
    parser.add_argument('--min_samples', type=int, default=100, help='最小样本数')

    # 实验策略
    parser.add_argument('--strategy', choices=['single', 'cross_domain', 'multi_dataset'],
                       default='single', help='实验策略')
    parser.add_argument('--variants', nargs='*', default=['debug'],
                       choices=['debug', 'quick', 'production'],
                       help='配置变体')
    parser.add_argument('--source', nargs='*', help='源域数据集（跨域实验）')
    parser.add_argument('--target', nargs='*', help='目标域数据集（跨域实验）')

    # 运行选项
    parser.add_argument('--parallel', action='store_true', help='并行执行')
    parser.add_argument('--timeout', type=int, default=180, help='单实验超时（分钟）')
    parser.add_argument('--dry_run', action='store_true', help='只显示计划不执行')
    parser.add_argument('--quick', action='store_true', help='快速模式（自动设置quick变体）')

    args = parser.parse_args()

    # 快速模式配置
    if args.quick:
        args.variants = ['quick']
        args.base_config = 'configs/id_contrastive/debug.yaml'

    # 创建运行器
    runner = MultiDatasetRunner(
        base_config=args.base_config,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        dry_run=args.dry_run
    )

    try:
        # 发现数据集
        datasets = runner.discover_datasets(
            include_patterns=args.datasets or args.include_patterns,
            exclude_patterns=args.exclude_patterns,
            min_samples=args.min_samples
        )

        if not datasets:
            runner.logger.error("没有找到可用的数据集")
            return 1

        # 生成实验
        experiments = runner.generate_experiments(
            datasets=datasets,
            strategy=args.strategy,
            config_variants=args.variants,
            source_datasets=args.source,
            target_datasets=args.target
        )

        if not experiments:
            runner.logger.error("没有生成实验配置")
            return 1

        # 运行实验
        results = runner.run_experiments(
            experiments,
            parallel=args.parallel,
            timeout_min=args.timeout
        )

        if not args.dry_run and results:
            # 生成报告
            report_path = runner.generate_report(results)

            # 打印摘要
            completed = [r for r in results if r.status == "completed"]
            failed = [r for r in results if r.status == "failed"]

            runner.logger.info(f"\n{'='*50}")
            runner.logger.info(f"实验完成!")
            runner.logger.info(f"成功: {len(completed)}, 失败: {len(failed)}")
            runner.logger.info(f"成功率: {len(completed) / max(1, len(results)) * 100:.1f}%")
            runner.logger.info(f"报告: {report_path}")
            runner.logger.info(f"{'='*50}")

        return 0

    except KeyboardInterrupt:
        runner.logger.warning("实验被中断")
        return 130
    except Exception as e:
        runner.logger.error(f"实验失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())