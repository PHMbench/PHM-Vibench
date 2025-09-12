#!/usr/bin/env python3
"""
多数据集实验脚本
基于metadata自动批量运行对比学习预训练实验
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

# 添加项目路径
sys.path.append('.')

from src.configs import load_config


class MultiDatasetExperimentRunner:
    """多数据集实验运行器"""
    
    def __init__(self, 
                 base_config_path: str,
                 metadata_dir: str = "data",
                 results_dir: str = "save/multi_dataset",
                 dry_run: bool = False):
        """
        初始化实验运行器
        
        Args:
            base_config_path: 基础配置文件路径
            metadata_dir: metadata文件目录
            results_dir: 结果保存目录
            dry_run: 是否只输出实验计划而不实际运行
        """
        self.base_config_path = base_config_path
        self.metadata_dir = Path(metadata_dir)
        self.results_dir = Path(results_dir)
        self.dry_run = dry_run
        
        # 创建结果目录
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验记录
        self.experiments = []
        self.results = {
            'completed': [],
            'failed': [],
            'skipped': []
        }
        
        print(f"多数据集实验运行器初始化完成")
        print(f"基础配置: {base_config_path}")
        print(f"结果目录: {results_dir}")
        print(f"干运行模式: {dry_run}")
    
    def discover_datasets(self) -> List[Dict[str, Any]]:
        """发现所有可用的数据集"""
        print(f"\n正在扫描metadata目录: {self.metadata_dir}")
        
        datasets = []
        metadata_files = list(self.metadata_dir.glob("metadata_*.xlsx"))
        
        if not metadata_files:
            print(f"⚠️  在 {self.metadata_dir} 中未找到metadata文件")
            return datasets
        
        for metadata_file in metadata_files:
            try:
                # 提取数据集名称
                dataset_name = metadata_file.stem.replace('metadata_', '')
                
                # 读取metadata文件获取基本信息
                df = pd.read_excel(metadata_file, sheet_name=0)
                
                dataset_info = {
                    'name': dataset_name,
                    'metadata_file': str(metadata_file),
                    'num_samples': len(df),
                    'h5_file': self.metadata_dir / f"{dataset_name}.h5"
                }
                
                # 检查H5文件是否存在
                if dataset_info['h5_file'].exists():
                    dataset_info['h5_size_mb'] = dataset_info['h5_file'].stat().st_size / (1024 * 1024)
                    dataset_info['ready'] = True
                else:
                    dataset_info['h5_size_mb'] = 0
                    dataset_info['ready'] = False
                    print(f"⚠️  数据集 {dataset_name} 的H5文件不存在: {dataset_info['h5_file']}")
                
                # 尝试获取更多信息
                if 'Label' in df.columns:
                    dataset_info['num_classes'] = df['Label'].nunique()
                    dataset_info['class_distribution'] = df['Label'].value_counts().to_dict()
                
                if 'ID' in df.columns:
                    dataset_info['num_ids'] = df['ID'].nunique()
                
                datasets.append(dataset_info)
                print(f"✅ 发现数据集: {dataset_name} ({dataset_info['num_samples']} 样本)")
                
            except Exception as e:
                print(f"❌ 处理metadata文件失败 {metadata_file}: {e}")
        
        print(f"\n共发现 {len(datasets)} 个数据集")
        return datasets
    
    def filter_datasets(self, 
                       datasets: List[Dict], 
                       include_patterns: List[str] = None,
                       exclude_patterns: List[str] = None,
                       min_samples: int = 100,
                       ready_only: bool = True) -> List[Dict]:
        """过滤数据集"""
        print(f"\n正在过滤数据集...")
        
        filtered_datasets = []
        
        for dataset in datasets:
            # 检查是否准备就绪
            if ready_only and not dataset.get('ready', False):
                print(f"跳过未准备数据集: {dataset['name']}")
                continue
            
            # 检查样本数量
            if dataset['num_samples'] < min_samples:
                print(f"跳过样本数量不足的数据集: {dataset['name']} ({dataset['num_samples']} < {min_samples})")
                continue
            
            # 检查包含模式
            if include_patterns:
                if not any(pattern.lower() in dataset['name'].lower() for pattern in include_patterns):
                    print(f"跳过不匹配包含模式的数据集: {dataset['name']}")
                    continue
            
            # 检查排除模式
            if exclude_patterns:
                if any(pattern.lower() in dataset['name'].lower() for pattern in exclude_patterns):
                    print(f"跳过匹配排除模式的数据集: {dataset['name']}")
                    continue
            
            filtered_datasets.append(dataset)
            print(f"✅ 保留数据集: {dataset['name']}")
        
        print(f"\n过滤后保留 {len(filtered_datasets)} 个数据集")
        return filtered_datasets
    
    def generate_experiment_configs(self, 
                                   datasets: List[Dict],
                                   config_variants: List[Dict] = None) -> List[Dict]:
        """为每个数据集生成实验配置"""
        print(f"\n正在生成实验配置...")
        
        if config_variants is None:
            # 对比学习默认配置变体
            config_variants = [
                {'name': 'default', 'overrides': {}},
                {'name': 'large_window', 'overrides': {'data.window_size': 2048, 'data.stride': 1024}},
                {'name': 'small_window', 'overrides': {'data.window_size': 512, 'data.stride': 256}},
                {'name': 'low_temp', 'overrides': {'task.temperature': 0.01}},
                {'name': 'high_temp', 'overrides': {'task.temperature': 0.5}},
                {'name': 'high_lr', 'overrides': {'task.lr': 5e-3}},
                {'name': 'large_batch', 'overrides': {'data.batch_size': 64}},
                {'name': 'small_batch', 'overrides': {'data.batch_size': 8}}
            ]
        
        experiments = []
        
        for dataset in datasets:
            for variant in config_variants:
                # 基础配置覆盖
                base_overrides = {
                    'data.metadata_file': dataset['metadata_file'],
                    'environment.experiment_name': f"contrastive_{dataset['name']}_{variant['name']}",
                    'environment.save_dir': str(self.results_dir / dataset['name'] / variant['name'])
                }
                
                # 合并变体覆盖
                final_overrides = {**base_overrides, **variant['overrides']}
                
                # 数据集特定调整
                dataset_specific_overrides = self._get_dataset_specific_overrides(dataset)
                final_overrides.update(dataset_specific_overrides)
                
                experiment = {
                    'id': f"{dataset['name']}_{variant['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'dataset': dataset['name'],
                    'variant': variant['name'], 
                    'config_overrides': final_overrides,
                    'expected_duration_hours': self._estimate_experiment_duration(dataset, variant),
                    'status': 'pending'
                }
                
                experiments.append(experiment)
        
        print(f"生成了 {len(experiments)} 个实验配置")
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
    
    def _estimate_experiment_duration(self, dataset: Dict, variant: Dict) -> float:
        """估计实验持续时间（小时）"""
        base_duration = 0.5  # 基础时间
        
        # 根据数据集大小调整
        size_factor = dataset['num_samples'] / 1000
        base_duration *= (1 + size_factor * 0.1)
        
        # 根据配置变体调整
        if 'large_window' in variant['name']:
            base_duration *= 1.5
        if 'high_lr' in variant['name']:
            base_duration *= 0.8  # 可能收敛更快
        
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
    
    def _print_experiment_plan(self, experiments: List[Dict]):
        """打印实验计划"""
        print(f"\n实验计划摘要:")
        print(f"总实验数: {len(experiments)}")
        
        # 按数据集分组
        datasets_groups = {}
        for exp in experiments:
            dataset = exp['dataset']
            if dataset not in datasets_groups:
                datasets_groups[dataset] = []
            datasets_groups[dataset].append(exp)
        
        for dataset, exps in datasets_groups.items():
            print(f"\n数据集: {dataset} ({len(exps)} 个实验)")
            total_time = sum(exp['expected_duration_hours'] for exp in exps)
            print(f"  预计总耗时: {total_time:.1f} 小时")
            
            for exp in exps:
                print(f"    - {exp['variant']}: {exp['expected_duration_hours']} 小时")
        
        total_time = sum(exp['expected_duration_hours'] for exp in experiments)
        print(f"\n总预计耗时: {total_time:.1f} 小时")
    
    def generate_report(self, results: Dict) -> str:
        """生成实验报告"""
        print(f"\n生成实验报告...")
        
        report_data = {
            'summary': {
                'total_experiments': len(results['completed']) + len(results['failed']) + len(results['skipped']),
                'completed': len(results['completed']),
                'failed': len(results['failed']),
                'skipped': len(results['skipped']),
                'success_rate': len(results['completed']) / (len(results['completed']) + len(results['failed'])) * 100 if (len(results['completed']) + len(results['failed'])) > 0 else 0
            },
            'completed_experiments': results['completed'],
            'failed_experiments': results['failed'],
            'skipped_experiments': results['skipped'],
            'generated_at': datetime.now().isoformat()
        }
        
        # 计算总耗时
        total_duration = sum(exp.get('actual_duration_hours', 0) for exp in results['completed'])
        report_data['summary']['total_duration_hours'] = round(total_duration, 2)
        
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
        
        print(f"实验报告已保存:")
        print(f"  详细报告: {report_path}")
        print(f"  摘要报告: {readable_report_path}")
        
        return str(readable_report_path)


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
        choices=['default', 'large_window', 'low_temp', 'high_lr'],
        default=['default'],
        help='配置变体'
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
        dry_run=args.dry_run
    )
    
    try:
        # 发现数据集
        datasets = runner.discover_datasets()
        
        if not datasets:
            print("❌ 没有发现可用的数据集")
            return
        
        # 过滤数据集
        filtered_datasets = runner.filter_datasets(
            datasets,
            include_patterns=args.include_datasets,
            exclude_patterns=args.exclude_datasets,
            min_samples=args.min_samples,
            ready_only=True
        )
        
        if not filtered_datasets:
            print("❌ 过滤后没有可用的数据集")
            return
        
        # 生成配置变体
        config_variants = []
        variant_configs = {
            'default': {'name': 'default', 'overrides': {}},
            'large_window': {'name': 'large_window', 'overrides': {'data.window_size': 2048, 'data.stride': 1024}},
            'low_temp': {'name': 'low_temp', 'overrides': {'task.temperature': 0.05}},
            'high_lr': {'name': 'high_lr', 'overrides': {'task.lr': 5e-3}}
        }
        
        for variant in args.variants:
            if variant in variant_configs:
                config_variants.append(variant_configs[variant])
        
        # 生成实验配置
        experiments = runner.generate_experiment_configs(filtered_datasets, config_variants)
        
        if not experiments:
            print("❌ 没有生成任何实验配置")
            return
        
        # 运行实验
        results = runner.run_experiments(
            experiments,
            parallel=args.parallel,
            max_parallel=args.max_parallel,
            timeout_hours=args.timeout
        )
        
        # 生成报告
        report_path = runner.generate_report(results)
        
        print(f"\n🎉 多数据集实验完成!")
        print(f"报告保存在: {report_path}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  实验被用户中断")
    except Exception as e:
        print(f"\n❌ 实验运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()