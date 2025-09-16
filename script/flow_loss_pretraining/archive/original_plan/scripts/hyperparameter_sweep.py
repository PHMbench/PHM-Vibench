#!/usr/bin/env python3
"""
超参数搜索脚本
支持网格搜索、随机搜索和贝叶斯优化
"""

import itertools
import subprocess
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
import random
import time

class HyperparameterSweep:
    """超参数搜索类"""
    
    def __init__(self, config_template="configs/demo/Pretraining/Flow/flow_baseline_experiment.yaml"):
        self.config_template = config_template
        self.results = []
        
    def grid_search(self, param_grid, max_experiments=None, output_dir="hyperparameter_sweep"):
        """网格搜索"""
        
        print(f"🔍 开始网格搜索超参数优化")
        print(f"搜索空间: {param_grid}")
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_output_dir = f"{output_dir}_{timestamp}"
        Path(full_output_dir).mkdir(parents=True, exist_ok=True)
        
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        all_combinations = list(itertools.product(*param_values))
        
        if max_experiments and len(all_combinations) > max_experiments:
            print(f"⚠️  限制实验数量从 {len(all_combinations)} 到 {max_experiments}")
            all_combinations = random.sample(all_combinations, max_experiments)
        
        print(f"总实验数: {len(all_combinations)}")
        
        # 运行实验
        for i, combination in enumerate(all_combinations, 1):
            params = dict(zip(param_names, combination))
            
            print(f"\n🔬 实验 {i}/{len(all_combinations)}: {params}")
            
            try:
                result = self._run_single_experiment(params, f"GridSearch_{i}")
                result['experiment_id'] = i
                result['search_type'] = 'grid'
                result['params'] = params
                
                self.results.append(result)
                
                # 保存中间结果
                self._save_intermediate_results(full_output_dir)
                
            except Exception as e:
                print(f"❌ 实验 {i} 失败: {e}")
                error_result = {
                    'experiment_id': i,
                    'search_type': 'grid',
                    'params': params,
                    'status': 'failed',
                    'error': str(e)
                }
                self.results.append(error_result)
        
        # 保存最终结果
        self._save_final_results(full_output_dir)
        return self.results
    
    def random_search(self, param_distributions, n_experiments, output_dir="random_search"):
        """随机搜索"""
        
        print(f"🎲 开始随机搜索超参数优化")
        print(f"参数分布: {param_distributions}")
        print(f"实验次数: {n_experiments}")
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_output_dir = f"{output_dir}_{timestamp}"
        Path(full_output_dir).mkdir(parents=True, exist_ok=True)
        
        # 运行随机实验
        for i in range(1, n_experiments + 1):
            # 随机采样参数
            params = {}
            for param_name, distribution in param_distributions.items():
                if isinstance(distribution, list):
                    # 离散分布
                    params[param_name] = random.choice(distribution)
                elif isinstance(distribution, dict):
                    if distribution['type'] == 'uniform':
                        # 连续均匀分布
                        params[param_name] = np.random.uniform(
                            distribution['low'], distribution['high']
                        )
                    elif distribution['type'] == 'loguniform':
                        # 对数均匀分布
                        params[param_name] = np.exp(np.random.uniform(
                            np.log(distribution['low']), np.log(distribution['high'])
                        ))
                    elif distribution['type'] == 'choice':
                        # 带权重的选择
                        params[param_name] = np.random.choice(
                            distribution['choices'], p=distribution.get('weights')
                        )
            
            print(f"\n🔬 随机实验 {i}/{n_experiments}: {params}")
            
            try:
                result = self._run_single_experiment(params, f"RandomSearch_{i}")
                result['experiment_id'] = i
                result['search_type'] = 'random'
                result['params'] = params
                
                self.results.append(result)
                
                # 保存中间结果
                self._save_intermediate_results(full_output_dir)
                
            except Exception as e:
                print(f"❌ 随机实验 {i} 失败: {e}")
                error_result = {
                    'experiment_id': i,
                    'search_type': 'random',
                    'params': params,
                    'status': 'failed',
                    'error': str(e)
                }
                self.results.append(error_result)
        
        # 保存最终结果
        self._save_final_results(full_output_dir)
        return self.results
    
    def bayesian_optimization(self, param_bounds, n_experiments, output_dir="bayesian_opt"):
        """贝叶斯优化 (需要安装scikit-optimize)"""
        
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
        except ImportError:
            raise ImportError("需要安装 scikit-optimize: pip install scikit-optimize")
        
        print(f"🤖 开始贝叶斯优化")
        print(f"参数边界: {param_bounds}")
        print(f"实验次数: {n_experiments}")
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_output_dir = f"{output_dir}_{timestamp}"
        Path(full_output_dir).mkdir(parents=True, exist_ok=True)
        
        # 定义搜索空间
        dimensions = []
        param_names = []
        
        for param_name, bounds in param_bounds.items():
            param_names.append(param_name)
            
            if bounds['type'] == 'real':
                dimensions.append(Real(bounds['low'], bounds['high'], name=param_name))
            elif bounds['type'] == 'integer':
                dimensions.append(Integer(bounds['low'], bounds['high'], name=param_name))
            elif bounds['type'] == 'categorical':
                dimensions.append(Categorical(bounds['choices'], name=param_name))
        
        # 定义目标函数
        @use_named_args(dimensions)
        def objective(**params):
            """贝叶斯优化目标函数"""
            
            experiment_id = len(self.results) + 1
            print(f"\n🔬 贝叶斯实验 {experiment_id}: {params}")
            
            try:
                result = self._run_single_experiment(params, f"BayesOpt_{experiment_id}")
                result['experiment_id'] = experiment_id
                result['search_type'] = 'bayesian'
                result['params'] = params
                
                self.results.append(result)
                
                # 保存中间结果
                self._save_intermediate_results(full_output_dir)
                
                # 返回负的准确率 (因为gp_minimize是最小化)
                accuracy = result.get('accuracy', 0)
                return -accuracy
                
            except Exception as e:
                print(f"❌ 贝叶斯实验 {experiment_id} 失败: {e}")
                error_result = {
                    'experiment_id': experiment_id,
                    'search_type': 'bayesian',
                    'params': params,
                    'status': 'failed',
                    'error': str(e)
                }
                self.results.append(error_result)
                return 0  # 返回最坏性能
        
        # 执行贝叶斯优化
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_experiments,
            random_state=42,
            acq_func='EI'  # Expected Improvement
        )
        
        # 保存最终结果
        self._save_final_results(full_output_dir)
        
        return {
            'best_params': dict(zip(param_names, result.x)),
            'best_score': -result.fun,
            'all_results': self.results
        }
    
    def _run_single_experiment(self, params, experiment_name):
        """运行单个实验"""
        
        # 构建配置覆盖字符串
        override_list = []
        for param_name, param_value in params.items():
            override_list.append(f"{param_name}={param_value}")
        override_str = ",".join(override_list)
        
        # 构建命令
        cmd = [
            "python", "run_flow_experiment_batch.py", "custom",
            "--experiments", "baseline",
            "--config_override", override_str,
            "--notes", f"HyperSearch_{experiment_name}",
            "--wandb"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 运行实验
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            # 解析结果
            experiment_result = self._parse_experiment_result(experiment_name)
            experiment_result['status'] = 'success'
            experiment_result['duration'] = duration
            experiment_result['stdout'] = result.stdout[-1000:]  # 保留最后1000字符
            
            return experiment_result
        else:
            raise RuntimeError(f"实验失败，返回码: {result.returncode}, 错误: {result.stderr}")
    
    def _parse_experiment_result(self, experiment_name):
        """解析实验结果"""
        
        # 尝试从结果文件中读取指标
        results_dir = Path("results")
        
        # 查找匹配的实验目录
        experiment_dirs = list(results_dir.glob(f"*{experiment_name}*"))
        
        if not experiment_dirs:
            print(f"⚠️  未找到实验目录: {experiment_name}")
            return {'accuracy': 0, 'f1_score': 0}
        
        # 取最新的实验目录
        experiment_dir = max(experiment_dirs, key=lambda x: x.stat().st_mtime)
        
        # 尝试读取指标文件
        metrics_file = experiment_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            return {
                'accuracy': metrics.get('test_accuracy', metrics.get('accuracy', 0)),
                'f1_score': metrics.get('test_f1', metrics.get('f1_score', 0)),
                'train_time': metrics.get('training_time', 0),
                'params_count': metrics.get('model_params', 0)
            }
        
        # 如果没有找到metrics.json，尝试从Lightning日志解析
        lightning_dir = experiment_dir / "lightning_logs" / "version_0"
        metrics_csv = lightning_dir / "metrics.csv"
        
        if metrics_csv.exists():
            df = pd.read_csv(metrics_csv)
            if not df.empty:
                last_row = df.iloc[-1]
                return {
                    'accuracy': last_row.get('val_accuracy', last_row.get('test_accuracy', 0)),
                    'f1_score': last_row.get('val_f1', last_row.get('test_f1', 0)),
                    'train_time': 0,
                    'params_count': 0
                }
        
        print(f"⚠️  无法解析实验结果: {experiment_name}")
        return {'accuracy': 0, 'f1_score': 0}
    
    def _save_intermediate_results(self, output_dir):
        """保存中间结果"""
        results_file = Path(output_dir) / "intermediate_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def _save_final_results(self, output_dir):
        """保存最终结果"""
        
        # 保存JSON格式
        results_json = Path(output_dir) / "final_results.json"
        with open(results_json, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 保存CSV格式
        if self.results:
            df_data = []
            for result in self.results:
                if result.get('status') == 'success':
                    row = result.copy()
                    # 展开params字典
                    if 'params' in row:
                        params = row.pop('params')
                        for param_name, param_value in params.items():
                            row[f'param_{param_name.replace(".", "_")}'] = param_value
                    df_data.append(row)
            
            if df_data:
                df = pd.DataFrame(df_data)
                results_csv = Path(output_dir) / "final_results.csv"
                df.to_csv(results_csv, index=False)
                
                # 打印最佳结果
                if 'accuracy' in df.columns:
                    best_idx = df['accuracy'].idxmax()
                    best_result = df.loc[best_idx]
                    
                    print(f"\n🏆 最佳结果:")
                    print(f"  准确率: {best_result['accuracy']:.4f}")
                    print(f"  F1分数: {best_result.get('f1_score', 'N/A')}")
                    print(f"  参数:")
                    
                    for col in df.columns:
                        if col.startswith('param_'):
                            param_name = col.replace('param_', '').replace('_', '.')
                            print(f"    {param_name}: {best_result[col]}")
        
        print(f"\n✅ 结果已保存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Flow超参数搜索")
    parser.add_argument('--method', choices=['grid', 'random', 'bayesian'], 
                       default='grid', help='搜索方法')
    parser.add_argument('--config', type=str, 
                       default='configs/demo/Pretraining/Flow/flow_baseline_experiment.yaml',
                       help='基础配置文件')
    parser.add_argument('--max_experiments', type=int, help='最大实验数量')
    parser.add_argument('--output_dir', type=str, default='hyperparameter_sweep',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建超参数搜索器
    sweep = HyperparameterSweep(args.config)
    
    if args.method == 'grid':
        # 网格搜索参数空间
        param_grid = {
            'task.lr': [1e-4, 5e-4, 1e-3],
            'task.flow_lr': [1e-4, 5e-4, 1e-3],
            'task.contrastive_weight': [0.1, 0.3, 0.5],
            'model.hidden_dim': [256, 512],
            'task.num_steps': [50, 100, 200],
            'task.batch_size': [32, 64]
        }
        
        results = sweep.grid_search(param_grid, args.max_experiments, args.output_dir)
        
    elif args.method == 'random':
        # 随机搜索参数分布
        param_distributions = {
            'task.lr': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-2},
            'task.flow_lr': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-2},
            'task.contrastive_weight': {'type': 'uniform', 'low': 0.0, 'high': 1.0},
            'model.hidden_dim': [128, 256, 512, 1024],
            'task.num_steps': [20, 50, 100, 200, 500],
            'task.batch_size': [16, 32, 64, 128]
        }
        
        n_experiments = args.max_experiments or 50
        results = sweep.random_search(param_distributions, n_experiments, args.output_dir)
        
    elif args.method == 'bayesian':
        # 贝叶斯优化参数边界
        param_bounds = {
            'task.lr': {'type': 'real', 'low': 1e-5, 'high': 1e-2},
            'task.flow_lr': {'type': 'real', 'low': 1e-5, 'high': 1e-2},
            'task.contrastive_weight': {'type': 'real', 'low': 0.0, 'high': 1.0},
            'model.hidden_dim': {'type': 'categorical', 'choices': [128, 256, 512, 1024]},
            'task.num_steps': {'type': 'integer', 'low': 20, 'high': 500},
            'task.batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128]}
        }
        
        n_experiments = args.max_experiments or 30
        results = sweep.bayesian_optimization(param_bounds, n_experiments, args.output_dir)
    
    print(f"\n🎉 超参数搜索完成!")
    print(f"总实验数: {len(results) if isinstance(results, list) else len(results['all_results'])}")


if __name__ == "__main__":
    main()